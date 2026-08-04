"""
exploded_peristim_builder.py — build a pre-stimulus variant of the exploded
(manual-sorted + unsorted) spike cache.

The canonical exploded cache (``Cortana/exploded_spike_cache/``) stores spikes
clipped to ``[stimulus onset, offset]``, so it has no pre-stimulus baseline. This
builder reuses the normal exploded pipeline (``load_and_combine_data``) but widens
the window, and writes a *separate* variant cache (e.g.
``exploded_spike_cache_pre1000ms/``), leaving the canonical cache and every
analysis that reads it untouched.

WHAT GETS A PRE-STIMULUS BASELINE:
  * manually-sorted units — YES. Their spike indices live un-windowed in
    ``sorted_spikes.pkl``, so the baseline is genuinely recovered.
  * unsorted channels     — NO. They come from ``compiled.pkl``, which is already
    clipped to [onset, offset]; their pre-stimulus spikes are not recoverable
    without the raw ``spike.dat`` (which this builder deliberately does not use,
    since some sessions lack it). Their pre-window stays empty.

Needs only ``compiled.pkl`` (repo) + ``sorted_spikes.pkl`` + ``info.rhd`` (Intan
session). No ``spike.dat``, no marker channel.

HOW TO RUN: open this file in PyCharm and click Run — edit the CONFIG block at the
bottom. (Or import ``build_exploded_peristim_cache`` from the plot driver.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_access.data_loader import load_and_combine_data, explode_spike_data
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

PROJECT_ROOT = Path(DATA_BASE_PATH)


def peristim_cache_subdir(pre_stimulus_time):
    """Cache subdir for a pre-stimulus window (strict window keeps the canonical name)."""
    if pre_stimulus_time and pre_stimulus_time > 0:
        return f"exploded_spike_cache_pre{int(round(pre_stimulus_time * 1000))}ms"
    return "exploded_spike_cache"


def build_exploded_peristim_cache(date, round_no, pre_stimulus_time, *,
                                  monkey=SUBJECT_MONKEY, cache_subdir=None, force=False):
    """Build + write the pre-stimulus exploded cache for one session.

    Returns (exploded_df, path). If the variant pkl already exists and ``force``
    is False, the build is skipped and the cached df is returned.
    """
    date = str(date)
    if cache_subdir is None:
        cache_subdir = peristim_cache_subdir(pre_stimulus_time)
    cache_dir = PROJECT_ROOT / monkey / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{date}_round_{round_no}.pkl"
    if path.exists() and not force:
        return pd.read_pickle(path), path

    combined = load_and_combine_data(date, round_no, pre_stimulus_time=pre_stimulus_time)
    if combined is None or combined.empty:
        print(f"  [skip] no trials for {date} round {round_no}")
        return None, path

    exploded = explode_spike_data(combined, date, round_no, curated_channels_only=False)
    exploded.to_pickle(path)
    print(f"  saved {path}  ({exploded['NeuronID'].nunique()} channels/units, {len(combined)} trials)")
    return exploded, path


# ===== Run directly in PyCharm — edit this block and hit Run (no CLI) =========
if __name__ == "__main__":
    DATE = "2023-09-26"
    ROUND_NO = 2
    PRE_STIMULUS_TIME = 1.0        # seconds of pre-stimulus baseline (manual-sorted units only)
    FORCE = False                  # True = rebuild even if the variant already exists

    build_exploded_peristim_cache(DATE, ROUND_NO, PRE_STIMULUS_TIME, force=FORCE)
