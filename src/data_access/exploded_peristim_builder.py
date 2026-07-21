"""
exploded_peristim_builder.py — build a pre-stimulus variant of the exploded
(manual-sorted + unsorted) spike cache.

The canonical exploded cache (``Cortana/exploded_spike_cache/``) stores spikes
clipped to ``[stimulus onset, offset]``, so it has no pre-stimulus baseline. This
builder re-derives each trial's spikes straight from the raw sources with a
widened lower bound (``onset - pre_stimulus_time``) and writes a *separate*
variant cache (e.g. ``exploded_spike_cache_pre1000ms/``), leaving the canonical
cache and every analysis that reads it untouched.

Spike sources per trial:
  * unsorted  — ``spike.dat`` (per-channel timestamps, seconds), for channels
                that were not manually sorted;
  * manual    — ``sorted_spikes.pkl`` (per-channel {unit: sample indices}).
Trial identity + metadata (MonkeyId/Name/Group) and, by default, the trial epochs
come from the session's ``compiled.pkl``. Passing ``reepoch_fnc`` instead
re-extracts the epochs from ``digitalin.dat`` with that
``false_negative_correction_duration`` (use 2 to match the SI-sorted path and
remove the overlapping-epoch artifact some sessions show in compiled.pkl); trial
metadata is then joined back on by task id.

Runs where the raw Intan session lives (``spike.dat``, ``sorted_spikes.pkl``,
``digitalin.dat``, ``notes.txt``) — i.e. the workstation, not a fresh checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels
from clat.intan.spike_file import fetch_spike_tstamps_from_file

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from data_access.data_loader import explode_spike_data, load_manually_sorted_spikes
from project_util import PROJECT_BASE_PATH, SUBJECT_MONKEY

PROJECT_ROOT = Path(PROJECT_BASE_PATH)


def peristim_cache_subdir(pre_stimulus_time):
    """Cache subdir for a pre-stimulus window (strict window keeps the canonical name)."""
    if pre_stimulus_time and pre_stimulus_time > 0:
        return f"exploded_spike_cache_pre{int(round(pre_stimulus_time * 1000))}ms"
    return "exploded_spike_cache"


def _trial_windows(round_dir, compiled, reepoch_fnc, sample_rate):
    """List of (task_id, onset_s, offset_s, (MonkeyId, MonkeyName, MonkeyGroup)) per trial.

    reepoch_fnc is None → reuse compiled.pkl's epochs (metadata+epoch from the same
    row, no join). Otherwise re-extract epochs from the marker channel with that
    correction duration and join metadata by task id.
    """
    if reepoch_fnc is None:
        out = []
        for _, r in compiled.iterrows():
            on_s, off_s = r["EpochStartStop"]
            out.append((r["TaskField"], float(on_s), float(off_s),
                        (r["MonkeyId"], r["MonkeyName"], r["MonkeyGroup"])))
        return out

    meta_by_task = {int(r["TaskField"]): (r["MonkeyId"], r["MonkeyName"], r["MonkeyGroup"])
                    for _, r in compiled.iterrows()}
    stim_epochs = epoch_using_marker_channels(
        os.path.join(round_dir, "digitalin.dat"), false_negative_correction_duration=reepoch_fnc)
    epochs_for_task = map_task_id_to_epochs_with_livenotes(
        os.path.join(round_dir, "notes.txt"), stim_epochs)
    out, matched = [], 0
    for task_id, (on_samp, off_samp) in epochs_for_task.items():
        meta = meta_by_task.get(int(task_id))
        if meta is None:
            continue
        matched += 1
        out.append((task_id, on_samp / sample_rate, off_samp / sample_rate, meta))
    print(f"  re-epoched with fnc={reepoch_fnc}: matched {matched}/{len(epochs_for_task)} "
          f"marker epochs to compiled.pkl metadata")
    return out


def _windowed_combined_data(round_dir, compiled, pre_stimulus_time, reepoch_fnc):
    """Rebuild per-trial combined (manual + unsorted) SpikeTimes dicts with a
    widened lower bound; returns a wide-format DataFrame ready for explode."""
    unsorted_by_channel, sample_rate = fetch_spike_tstamps_from_file(
        os.path.join(round_dir, "spike.dat"))

    sorted_path = os.path.join(round_dir, "sorted_spikes.pkl")
    manual = load_manually_sorted_spikes(sorted_path) if os.path.exists(sorted_path) else {}
    manual_base = {str(ch) for ch in manual}  # base channels that were manually sorted

    trials = _trial_windows(round_dir, compiled, reepoch_fnc, sample_rate)

    rows = []
    for task_id, on_s, off_s, (mid, mname, mgroup) in trials:
        lo_s = max(on_s - pre_stimulus_time, 0.0)  # clamp to recording start
        spikes = {}
        # manually sorted units first (so explode's _Unit detection strips the
        # unit suffix for BaseChannel, matching combine_unsorted_with_sorted)
        for ch, units in manual.items():
            for unit_name, idxs in units.items():
                spikes[f"{ch}_{unit_name}"] = [i / sample_rate for i in idxs
                                               if lo_s <= i / sample_rate <= off_s]
        # unsorted channels, excluding any already covered by a manual sort
        for ch, tstamps in unsorted_by_channel.items():
            if str(ch) in manual_base:
                continue
            spikes[ch] = [t for t in tstamps if lo_s <= t <= off_s]
        rows.append({
            "TaskField": task_id, "MonkeyId": mid, "MonkeyName": mname,
            "MonkeyGroup": mgroup, "SpikeTimes": spikes, "EpochStartStop": (on_s, off_s),
        })
    return pd.DataFrame(rows)


def build_exploded_peristim_cache(date, round_no, pre_stimulus_time, *,
                                  reepoch_fnc=None, monkey=SUBJECT_MONKEY,
                                  cache_subdir=None, force=False):
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

    reader = RecordingMetadataReader()
    pickle_filepath, _, round_dir = reader.get_metadata_for_spike_analysis(date, round_no, monkey)
    compiled = pd.read_pickle(pickle_filepath)
    if compiled is None or compiled.empty:
        print(f"  [skip] empty compiled.pkl for {date} round {round_no}")
        return None, path

    combined = _windowed_combined_data(str(round_dir), compiled, pre_stimulus_time, reepoch_fnc)
    if combined.empty:
        print(f"  [skip] no trials for {date} round {round_no}")
        return None, path

    exploded = explode_spike_data(combined, date, round_no, curated_channels_only=False)
    exploded.to_pickle(path)
    print(f"  saved {path}  ({exploded['NeuronID'].nunique()} channels/units, {len(combined)} trials)")
    return exploded, path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a pre-stimulus exploded spike cache")
    p.add_argument("--date", required=True, help="e.g. 2023-09-26")
    p.add_argument("--round", type=int, required=True, dest="round_no")
    p.add_argument("--pre-stimulus-time", type=float, default=1.0, dest="pre_stimulus_time",
                   help="Seconds of pre-stimulus baseline to retain (default 1.0).")
    p.add_argument("--reepoch-fnc", type=int, default=None, dest="reepoch_fnc",
                   help="Re-extract epochs from digitalin.dat with this "
                        "false_negative_correction_duration (2 matches the SI-sorted path and "
                        "fixes overlapping epochs). Default: reuse compiled.pkl epochs.")
    p.add_argument("--monkey", default=SUBJECT_MONKEY)
    p.add_argument("--force", action="store_true", help="Rebuild even if the variant exists.")
    args = p.parse_args()
    build_exploded_peristim_cache(args.date, args.round_no, args.pre_stimulus_time,
                                  reepoch_fnc=args.reepoch_fnc, monkey=args.monkey, force=args.force)
