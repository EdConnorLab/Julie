"""
spike_window.py -- one on-disk cache, many windows.

THE ARRANGEMENT
---------------
Every spike cache is built ONCE, with the widest window we ever need: 1000 ms of
pre-stimulus baseline before each trial's onset. There is no separate strict cache
to keep in sync -- the strict [onset, offset] window is a *view*, produced by
clipping at load time.

    on disk   exploded_spike_cache_pre1000ms/     <- the only source of truth
              sorted_spike_cache_pre1000ms/
              threshold_mua_spike_cache_pre1000ms/

    in code   source.load(date, round)                 -> strict [onset, offset]
              source.with_pre_stimulus(1.0).load(...)  -> [onset-1s, offset]

Clipping happens at the load boundary (``data_access.spike_source``) rather than
in each analysis, and it defaults to STRICT. That default is the whole point: a
dozen call sites do ``len(row['SpikeTimes'])`` or
``np.concatenate(group['SpikeTimes'])`` and divide by the epoch duration --
filter_good_neurons, add_trial_spike_rate_columns, run_stimulus_level_glmm,
extract_spike_counts_from_cells, demixedPCA, data_diagnostics. Handing any of
them a cache that silently carries an extra second of spikes inflates every
count and firing rate by ~40% with nothing to notice. With the clip on by
default they keep receiving exactly what they received before, and only code
that explicitly asks for a baseline ever sees one.

WHAT THE PRE-STIMULUS WINDOW ACTUALLY CONTAINS
----------------------------------------------
It is not uniform across sources, and the difference is structural, not a bug:

  * sorted_spike_cache_pre1000ms  -- full baseline on every unit. Spike trains
    come from the sorter output, which is not windowed.
  * threshold_mua_spike_cache_pre1000ms -- full baseline on every channel.
    Detection runs on the raw amplifier trace.
  * exploded_spike_cache_pre1000ms -- baseline on the MANUALLY SORTED units only.
    Unsorted channels come from compiled.pkl, which is already clipped to
    [onset, offset], so their pre-window is empty and cannot be recovered without
    re-reading spike.dat.

Strict views are identical across all three. Only reach for a pre-stimulus view
of the exploded cache if you are looking at manually-sorted units.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# What the caches on disk were built with. Everything else is derived from it.
CACHE_PRE_STIMULUS_TIME = 1.0          # seconds

# The canonical cache directory for each source.
EXPLODED_CACHE_SUBDIR = "exploded_spike_cache_pre1000ms"
SORTED_CACHE_SUBDIR = "sorted_spike_cache_pre1000ms"
MUA_CACHE_SUBDIR = "threshold_mua_spike_cache_pre1000ms"

# sorted_spike_cache_filtered is written already-strict by
# analyses.build_filtered_cache, because a dozen modules read its pickles
# straight off disk without going through a SpikeSource.
FILTERED_CACHE_SUBDIR = "sorted_spike_cache_filtered"

STRICT = 0.0

_PRE_MS_RE = re.compile(r"_pre(\d+)ms$")


def cache_pre_stimulus_time(cache_subdir: str) -> float:
    """Seconds of baseline a cache directory holds, read from its name.

    'sorted_spike_cache_pre1000ms' -> 1.0;  'sorted_spike_cache' -> 0.0.
    """
    m = _PRE_MS_RE.search(str(cache_subdir))
    return int(m.group(1)) / 1000.0 if m else 0.0


def clip_spike_window(df: pd.DataFrame, pre_stimulus_time: float = STRICT, *,
                      available: float | None = None,
                      cache_subdir: str | None = None) -> pd.DataFrame:
    """Clip SpikeTimes to ``[onset - pre_stimulus_time, offset]``.

    ``EpochStartStop`` is always the strict (onset, offset) pair, in every cache,
    so re-zeroing at plot time (spike - onset) renders retained pre-stimulus
    spikes as negative times. This only drops spikes; it never shifts them and
    never touches EpochStartStop.

    ``available`` (or ``cache_subdir``, from which it is read) is how much
    baseline the cache actually holds. Asking for more than that is an error
    rather than a silently short window -- a truncated baseline is exactly the
    kind of thing that looks like a real result.
    """
    if df is None or getattr(df, "empty", True):
        return df
    if "SpikeTimes" not in df.columns or "EpochStartStop" not in df.columns:
        return df

    pre = float(pre_stimulus_time or 0.0)
    if pre < 0:
        raise ValueError(f"pre_stimulus_time must be >= 0, got {pre}")

    if available is None and cache_subdir is not None:
        available = cache_pre_stimulus_time(cache_subdir)
    if available is not None and pre > available + 1e-9:
        raise ValueError(
            f"asked for {pre * 1000:.0f} ms of pre-stimulus baseline but "
            f"{cache_subdir or 'this cache'} only holds {available * 1000:.0f} ms. "
            f"Rebuild the cache with a wider window instead of accepting a short one.")

    # Nothing to drop when the cache is already exactly this wide.
    if available is not None and abs(pre - available) < 1e-9:
        return df

    out = df.copy()
    starts = np.array([float(a) for a, _ in out["EpochStartStop"]])
    lower = starts - pre

    def _clip(spikes, low):
        arr = np.asarray(spikes, dtype=float).ravel()
        kept = arr[arr >= low] if arr.size else arr
        return kept.tolist() if isinstance(spikes, list) else kept

    out["SpikeTimes"] = [_clip(s, lo) for s, lo in zip(out["SpikeTimes"], lower)]
    return out


def strict_view(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """The [onset, offset] window -- what every analysis gets unless it asks otherwise."""
    return clip_spike_window(df, STRICT, **kwargs)
