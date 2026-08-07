"""
qc_breakdown.py -- why each neuron passed or failed the filtered-cache QC gate.

build_filtered_cache reports only a count ("0/12 neurons passed"), which is no help
when a session with healthy trial coverage loses every unit. This recomputes the same
per-neuron statistics filter_good_neurons derives and prints, for each neuron, the
value behind every criterion and which ones it failed -- then which criterion was the
BINDING one across the session (the sole reason a neuron was dropped).

It reads the same source build_filtered_cache reads and clips to the strict
[onset, offset] window first, for the same reason: filter_good_neurons divides raw
spike counts by the epoch duration, so a retained baseline would inflate every rate.

The thresholds are imported from build_filtered_cache rather than restated, and the
pass/fail verdict is checked against filter_good_neurons itself on every session, so
this cannot quietly drift away from the gate it is explaining.

HOW TO RUN (PyCharm): open and click Run -- edit the CONFIG block. Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyses.build_filtered_cache import QC
from analyses.spike_count import filter_good_neurons
from data_access.data_loader import normalize_monkey_names
from data_access.spike_window import SORTED_CACHE_SUBDIR, STRICT, clip_spike_window
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

# ===== CONFIG — edit me =======================================================
SESSIONS = ["2023-10-31_round_3", "2023-11-20_round_2"]   # [] = every session in the cache
SOURCE_SUBDIR = SORTED_CACHE_SUBDIR
SHOW_NEURONS = True          # False = per-session summary only
MAX_NEURON_ROWS = 40         # cap the per-neuron table
# ==============================================================================

CRITERIA = ("spikes", "rate", "trials", "blocks", "isi", "coverage")


def neuron_stats(df):
    """Per-neuron QC statistics -- mirrors filter_good_neurons' derivation exactly."""
    cond_col = QC.get("condition_col", "MonkeyName")
    n_blocks = QC["n_time_blocks"]
    refractory_s = QC["refractory_ms"] / 1000.0
    n_conditions = QC.get("n_conditions")
    have_cond = cond_col in df.columns

    rows = []
    for neuron_id, group in df.groupby("NeuronID"):
        group = group.sort_values(by="EpochStartStop",
                                  key=lambda col: col.apply(lambda x: x[0]))
        counts = group["SpikeTimes"].apply(len)
        total_spikes = int(counts.sum())
        total_time = group["EpochStartStop"].apply(lambda x: x[1] - x[0]).sum()
        total_trials = len(group)

        # temporal stability: blocks of trials with any spikes
        block_size = max(1, total_trials // n_blocks)
        active_blocks = 0
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else total_trials
            if counts.iloc[start:end].sum() > 0:
                active_blocks += 1

        all_spikes = np.concatenate(group["SpikeTimes"].values) if total_spikes else np.array([])
        if all_spikes.size >= 2:
            isis = np.diff(np.sort(all_spikes))
            isi_violation = float(np.mean(isis < refractory_s))
        else:
            isi_violation = 1.0

        if have_cond:
            reps = group.groupby(cond_col).size()
            min_reps = int(reps.min()) if (
                n_conditions is None or reps.shape[0] >= n_conditions) else 0
        else:
            min_reps = np.nan

        rows.append({
            "NeuronID": neuron_id,
            "TotalSpikes": total_spikes,
            "MeanFiringRateHz": total_spikes / total_time if total_time > 0 else 0.0,
            "TotalTrials": total_trials,
            "ActiveBlocks": active_blocks,
            "ISIViolationRate": isi_violation,
            "MinRepsPerCondition": min_reps,
        })
    return pd.DataFrame(rows)


def failures(row):
    """Which criteria this neuron fails, in the gate's own terms."""
    bad = []
    if row["TotalSpikes"] < QC["min_total_spikes"]:
        bad.append("spikes")
    if row["MeanFiringRateHz"] < QC["min_firing_rate_hz"]:
        bad.append("rate")
    if row["TotalTrials"] < QC["min_trial_count"]:
        bad.append("trials")
    if row["ActiveBlocks"] < QC["min_active_blocks"]:
        bad.append("blocks")
    if row["ISIViolationRate"] > QC["max_isi_violation_rate"]:
        bad.append("isi")
    min_reps = QC.get("min_reps_per_condition")
    if min_reps is not None and not pd.isna(row["MinRepsPerCondition"]) \
            and row["MinRepsPerCondition"] < min_reps:
        bad.append("coverage")
    return bad


def report_session(path):
    df = pd.read_pickle(path)
    df = clip_spike_window(normalize_monkey_names(df), STRICT, cache_subdir=SOURCE_SUBDIR)
    stats = neuron_stats(df)
    if stats.empty:
        print(f"\n=== {path.stem} ===  no neurons")
        return
    stats["Fails"] = stats.apply(failures, axis=1)
    passed = stats[stats["Fails"].str.len() == 0]

    print(f"\n=== {path.stem} ===  {len(stats)} neurons, {len(passed)} passed"
          f"   (thresholds: spikes>={QC['min_total_spikes']}, rate>={QC['min_firing_rate_hz']}Hz, "
          f"blocks>={QC['min_active_blocks']}/{QC['n_time_blocks']}, "
          f"isi<={QC['max_isi_violation_rate']}, reps>={QC.get('min_reps_per_condition')})")

    if SHOW_NEURONS:
        print(f"  {'NeuronID':<44}{'spikes':>8}{'rate':>7}{'blk':>5}{'isi':>8}{'reps':>6}  fails")
        for r in stats.head(MAX_NEURON_ROWS).itertuples():
            name = r.NeuronID if len(r.NeuronID) <= 43 else "..." + r.NeuronID[-40:]
            print(f"  {name:<44}{r.TotalSpikes:>8}{r.MeanFiringRateHz:>7.2f}"
                  f"{r.ActiveBlocks:>5}{r.ISIViolationRate:>8.3f}{r.MinRepsPerCondition:>6}"
                  f"  {','.join(r.Fails) if r.Fails else 'PASS'}")

    # which criterion is actually doing the work
    counts = {c: int(stats["Fails"].apply(lambda f, c=c: c in f).sum()) for c in CRITERIA}
    sole = {c: int(stats["Fails"].apply(lambda f, c=c: f == [c]).sum()) for c in CRITERIA}
    print("  failed by criterion :", ", ".join(f"{c}={counts[c]}" for c in CRITERIA if counts[c]))
    sole_txt = ", ".join(f"{c}={sole[c]}" for c in CRITERIA if sole[c])
    print("  SOLE reason (binding):", sole_txt or "(every failure had multiple causes)")

    # never let this drift from the gate it explains
    truth = set(filter_good_neurons(df, **QC))
    if truth != set(passed["NeuronID"]):
        print(f"  [WARNING] disagrees with filter_good_neurons "
              f"({len(truth)} vs {len(passed)}) -- this script needs updating to match it")


def main():
    src = Path(DATA_BASE_PATH) / SUBJECT_MONKEY / SOURCE_SUBDIR
    paths = ([src / f"{s}.pkl" for s in SESSIONS] if SESSIONS
             else sorted(src.glob("*.pkl")))
    print(f"source: {src}")
    for path in paths:
        if not path.exists():
            print(f"\n=== {path.stem} ===  NOT FOUND at {path}")
            continue
        report_session(path)


if __name__ == "__main__":
    main()
