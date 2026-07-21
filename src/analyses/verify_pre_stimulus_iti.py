"""
verify_pre_stimulus_iti.py — Inter-trial-interval (ITI) sanity check for pre-stimulus rasters.

Before regenerating spike caches with a pre-stimulus window, we need to know the
window will not reach back into the *previous* trial (its stimulus response or the
reward pulse that fires shortly after the previous trial's offset).

For a trial with onset ``t``, a pre-stimulus window of ``pre`` seconds spans
``[t - pre, t]``. That window stays clean as long as ``pre`` is smaller than the
inter-trial interval (ITI = this trial's onset - previous trial's offset). This
script reads the already-built caches (which store ``EpochStartStop`` per trial),
computes the ITI distribution, and reports what fraction of trials a given
``pre`` would contaminate.

ITI is defined exactly as in ``analyses.spontaneous_activity.get_inter_trial_intervals``
(gap = current onset - previous offset), computed here directly to keep this a
dependency-light diagnostic.

Usage (from repo root):
    PYTHONPATH=src python3 -m analyses.verify_pre_stimulus_iti
    PYTHONPATH=src python3 -m analyses.verify_pre_stimulus_iti --pre 0.2 \
        --cache-dir Cortana/sorted_spike_cache
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = two levels up from this file (…/src/analyses/this.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def session_epochs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Unique (onset, offset) epoch tuples for a session, in chronological order."""
    return sorted(set(df["EpochStartStop"]))


def inter_trial_gaps(epochs: list[tuple[float, float]]) -> list[float]:
    """Gap before each trial: current onset - previous offset (mirrors
    analyses.spontaneous_activity.get_inter_trial_intervals)."""
    return [epochs[i][0] - epochs[i - 1][1] for i in range(1, len(epochs))]


def analyze_cache_dir(cache_dir: Path, pre: float) -> None:
    files = sorted(glob.glob(os.path.join(str(cache_dir), "*.pkl")))
    if not files:
        print(f"  (no .pkl files found in {cache_dir})")
        return

    all_gaps: list[float] = []
    all_durations: list[float] = []
    per_session: list[tuple[str, int, float]] = []  # (name, n_trials, min_gap)
    skipped = 0

    for f in files:
        try:
            df = pd.read_pickle(f)
        except Exception as exc:  # unreadable / not a spike cache
            skipped += 1
            print(f"  skip {os.path.basename(f)}: {type(exc).__name__}: {exc}")
            continue
        if df is None or getattr(df, "empty", True) or "EpochStartStop" not in df.columns:
            skipped += 1
            continue
        epochs = session_epochs(df)
        gaps = inter_trial_gaps(epochs)
        all_gaps.extend(gaps)
        all_durations.extend(b - a for a, b in epochs)
        if gaps:
            per_session.append((os.path.basename(f), len(epochs), min(gaps)))

    if not all_gaps:
        print("  (no inter-trial gaps computed)")
        return

    g = np.asarray(all_gaps)
    d = np.asarray(all_durations)
    contaminated = int((g < pre).sum())
    overlaps = int((g < 0).sum())

    print(f"  files: {len(files)}   read: {len(files) - skipped}   skipped: {skipped}")
    print(f"  trials with epochs: {len(d)}   inter-trial gaps: {len(g)}")
    print(f"  stimulus duration (s): min {d.min():.3f}  median {np.median(d):.3f}  max {d.max():.3f}")
    print(
        f"  ITI gap (s): min {g.min():.4f}  p1 {np.percentile(g, 1):.4f}  "
        f"p5 {np.percentile(g, 5):.4f}  median {np.median(g):.4f}  max {g.max():.3f}"
    )
    for thr in (0.1, 0.2, 0.3, 0.5):
        m = g < thr
        print(f"    gaps < {thr:.1f}s: {int(m.sum()):5d}  ({m.mean() * 100:.2f}%)")
    print(f"    negative gaps (overlapping epochs): {overlaps}")

    verdict = "OK" if contaminated / len(g) < 0.01 else "REVIEW"
    print(
        f"  => pre={pre:.3f}s would touch the previous trial on "
        f"{contaminated}/{len(g)} gaps ({contaminated / len(g) * 100:.2f}%)  [{verdict}]"
    )
    if per_session:
        per_session.sort(key=lambda x: x[2])
        print("  smallest min-ITI sessions:")
        for name, n_trials, mn in per_session[:6]:
            print(f"    {name}: {n_trials} trials, min ITI {mn:.4f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Inter-trial-interval check for pre-stimulus rasters")
    p.add_argument("--pre", type=float, default=0.2, help="Pre-stimulus window in seconds (default 0.2)")
    p.add_argument(
        "--cache-dir",
        action="append",
        default=None,
        help="Cache directory of spike pkls (repeatable). "
        "Defaults to Cortana/sorted_spike_cache and Cortana/exploded_spike_cache under the repo root.",
    )
    args = p.parse_args()

    if args.cache_dir:
        cache_dirs = [Path(c) for c in args.cache_dir]
    else:
        cache_dirs = [
            REPO_ROOT / "Cortana" / "sorted_spike_cache",
            REPO_ROOT / "Cortana" / "exploded_spike_cache",
        ]

    for cache_dir in cache_dirs:
        print(f"\n===== {cache_dir} =====")
        analyze_cache_dir(cache_dir, args.pre)


if __name__ == "__main__":
    main()
