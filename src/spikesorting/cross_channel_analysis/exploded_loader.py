"""
exploded_loader.py — build a Session from an ``exploded_spike_cache`` pickle.

The original analysis reads ``sorted_spikes.pkl`` (manually sorted units only,
spikes as integer sample indices). To also compare **unsorted** whole-channel
spikes, we read the per-session exploded cache that already sits under
``Cortana/exploded_spike_cache/{date}_round_{round}.pkl``. Those frames hold
per-trial spike **times in seconds** for every channel — sorted units keyed like
``"Channel.C_010_Unit 1"`` and unsorted whole-channels like ``"Channel.C_006"``.

Two things to know:

* A channel that was manually sorted appears only as its sorted unit(s); its raw
  unsorted trace is dropped from the cache (see ``combine_unsorted_with_sorted``
  in ``data_access/data_loader.py``). So sorted-vs-unsorted pairs are always on
  *different* base channels — exactly the cross-channel duplicate question.
* Spikes here are times in seconds. We convert them to integer **sample ticks**
  at :data:`TICK_RATE_HZ` (``round(t * fs)``) so they slot straight into the
  existing integer-index metrics (:mod:`spiketrain_metrics`) — feeding seconds
  as floats would be silently truncated to whole seconds by those routines.
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from spikesorting.cross_channel_analysis.loader import Session, SortedUnit

# Recordings are 30 kHz. We only need a consistent time quantisation: two spikes
# 0.4 ms apart become 12 ticks apart regardless of the true rate, and the ≤33 µs
# rounding error is negligible next to the coincidence window.
TICK_RATE_HZ = 30000.0

UNSORTED_UNIT_NAME = "unsorted"

_SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_round_(\d+)\.pkl$")


def split_channel(channel_str: str) -> Tuple[str, str, bool]:
    """``"Channel.C_010_Unit 1"`` → ``("C-010", "Unit 1", True)``.

    ``"Channel.C_006"`` → ``("C-006", "unsorted", False)``. Returns
    ``(base_channel, unit_name, is_sorted)``.
    """
    s = str(channel_str).replace("Channel.", "")
    if "_Unit" in s:
        base, unit = s.split("_Unit", 1)
        return base.replace("_", "-"), "Unit" + unit, True
    return s.replace("_", "-"), UNSORTED_UNIT_NAME, False


def _session_train_ticks(spike_times_lists) -> np.ndarray:
    """Concatenate per-trial second-timestamps into sorted integer ticks."""
    parts = [np.asarray(x, dtype=float) for x in spike_times_lists if len(x)]
    if not parts:
        return np.empty(0, dtype=np.int64)
    t = np.concatenate(parts)
    ticks = np.round(t * TICK_RATE_HZ).astype(np.int64)
    ticks.sort(kind="stable")
    return ticks


def session_from_exploded_df(
    df: pd.DataFrame,
    *,
    label: Optional[str] = None,
    include_sorted: bool = True,
    include_unsorted: bool = True,
    min_spikes: int = 1,
) -> Session:
    """Build a :class:`Session` of integer-tick units from an exploded frame.

    Each unique ``Channel`` becomes one :class:`SortedUnit`; unsorted channels
    get ``unit_name="unsorted"``. ``sample_rate`` is :data:`TICK_RATE_HZ`.
    """
    chan_str = df["Channel"].astype(str)
    units: List[SortedUnit] = []
    for channel_value in chan_str.unique():
        base, unit_name, is_sorted = split_channel(channel_value)
        if is_sorted and not include_sorted:
            continue
        if (not is_sorted) and not include_unsorted:
            continue
        rows = df[chan_str == channel_value]
        ticks = _session_train_ticks(rows["SpikeTimes"])
        if ticks.size < min_spikes:
            continue
        units.append(SortedUnit(channel=base, unit_name=unit_name, spike_indices=ticks))
    # deterministic order: by contact position then unit name
    from spikesorting.cross_channel_analysis import probe_geometry as geom
    units.sort(key=lambda u: (geom.contact_index_of(u.channel)
                              if geom.contact_index_of(u.channel) is not None else 1e9,
                              u.unit_name))
    return Session(units=units, sample_rate=TICK_RATE_HZ, label=label)


def load_exploded_session(pkl_path: str, **kwargs) -> Session:
    """Load one ``{date}_round_{round}.pkl`` exploded cache file as a Session."""
    df = pd.read_pickle(pkl_path)
    label = kwargs.pop("label", os.path.splitext(os.path.basename(pkl_path))[0])
    return session_from_exploded_df(df, label=label, **kwargs)


def find_sessions(cache_dir: str) -> List[Tuple[str, str, int]]:
    """List ``(pkl_path, date, round_no)`` for every session pkl in a cache dir."""
    out = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.pkl"))):
        m = _SESSION_RE.search(os.path.basename(path))
        if m:
            out.append((path, m.group(1), int(m.group(2))))
    return out


def is_sorted_unit(unit: SortedUnit) -> bool:
    return unit.unit_name != UNSORTED_UNIT_NAME


def pair_type(session: Session, uid_a: str, uid_b: str) -> str:
    """Classify a pair: ``sorted-sorted`` | ``sorted-unsorted`` | ``unsorted-unsorted``."""
    a_sorted = is_sorted_unit(session.unit(uid_a))
    b_sorted = is_sorted_unit(session.unit(uid_b))
    if a_sorted and b_sorted:
        return "sorted-sorted"
    if a_sorted != b_sorted:
        return "sorted-unsorted"
    return "unsorted-unsorted"
