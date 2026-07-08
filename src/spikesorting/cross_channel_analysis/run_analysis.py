"""
run_analysis.py — one command from ``sorted_spikes.pkl`` to figures + report.

Given a session directory (or an explicit pickle path) this:
    1. loads the sorted units,
    2. computes pairwise duplicate metrics for every cross-channel unit pair,
    3. writes an overview figure (coincidence matrix + distance-vs-coincidence),
    4. writes one diagnostic figure per candidate duplicate pair,
    5. writes a ranked ``candidate_duplicates.csv`` and a text summary.

Waveform/footprint panels are added when voltages are available — either found
next to the pickle as ``voltages.npz`` (as written by the synthetic generator)
or loaded from the raw recording with ``--with-voltages`` (heavy; needs the
Intan files + windowsort).

Examples
--------
    # analyse a real session (spike-trains only)
    python -m spikesorting.cross_channel_analysis.run_analysis /path/to/231030_round2

    # include the raw-recording footprints
    python -m spikesorting.cross_channel_analysis.run_analysis /path/to/session --with-voltages

    # kick the tyres on a fabricated session with a known duplicate
    python -m spikesorting.cross_channel_analysis.run_analysis --demo
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Optional

import numpy as np

from . import spiketrain_metrics as stm
from . import plots
from .loader import load_sorted_spikes, Session


def _maybe_load_voltages(session_dir: str, use_raw: bool) -> Optional[Dict[str, np.ndarray]]:
    """Load voltages from a sidecar ``voltages.npz`` or the raw recording."""
    npz = os.path.join(session_dir, "voltages.npz")
    if os.path.exists(npz):
        data = np.load(npz)
        return {k: data[k] for k in data.files}
    if use_raw:
        from .waveforms import load_voltages
        print("Loading raw voltages via windowsort (this can take a while)...")
        return load_voltages(session_dir)
    return None


def write_csv(pairs, path: str):
    fields = ["uid_a", "uid_b", "channel_a", "channel_b", "n_a", "n_b",
              "coincidence", "ratio_over_chance", "distance_um", "contacts_apart",
              "refractory_a", "refractory_b", "refractory_merged", "merged_refractory_ok"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for p in pairs:
            w.writerow([
                p.uid_a, p.uid_b, p.channel_a, p.channel_b, p.n_a, p.n_b,
                f"{p.coincidence:.4f}", f"{p.ratio_over_chance:.2f}",
                "" if p.distance_um is None else f"{p.distance_um:.1f}",
                "" if p.contacts_apart is None else p.contacts_apart,
                f"{p.refractory_a:.4f}", f"{p.refractory_b:.4f}",
                f"{p.refractory_merged:.4f}", p.merged_refractory_ok,
            ])


def analyze_session(
    session: Session,
    out_dir: str,
    *,
    voltages_by_channel: Optional[Dict[str, np.ndarray]] = None,
    coincidence_window_ms: float = 0.4,
    coincidence_threshold: float = 0.3,
    ratio_threshold: float = 5.0,
):
    """Run the full analysis for one loaded session and write outputs."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Session '{session.label}': {len(session.units)} units on "
          f"{len(session.channels())} channels @ {session.sample_rate:g} Hz")

    pairs = stm.all_pairs(session, window_ms=coincidence_window_ms)
    write_csv(pairs, os.path.join(out_dir, "all_pairs.csv"))

    candidates = stm.candidate_duplicates(
        pairs, coincidence_threshold=coincidence_threshold, ratio_threshold=ratio_threshold)
    write_csv(candidates, os.path.join(out_dir, "candidate_duplicates.csv"))

    # overview figure
    import matplotlib.pyplot as plt
    fig = plots.plot_coincidence_matrix(session, window_ms=coincidence_window_ms)
    fig.savefig(os.path.join(out_dir, "overview_coincidence_matrix.png"), dpi=150)
    plt.close(fig)

    fig = plots.plot_distance_vs_coincidence(pairs)
    fig.savefig(os.path.join(out_dir, "overview_distance_vs_coincidence.png"), dpi=150)
    plt.close(fig)

    # per-candidate reports
    for rank, pair in enumerate(candidates, 1):
        fig = plots.plot_pair_report(session, pair, voltages_by_channel=voltages_by_channel)
        safe = f"{pair.uid_a}__{pair.uid_b}".replace(" ", "").replace("/", "-")
        fig.savefig(os.path.join(out_dir, f"pair_{rank:02d}_{safe}.png"), dpi=150)
        plt.close(fig)

    _write_summary(session, pairs, candidates, os.path.join(out_dir, "summary.txt"))
    print(f"Found {len(candidates)} candidate duplicate pair(s). "
          f"Outputs written to {out_dir}")
    return pairs, candidates


def _write_summary(session, pairs, candidates, path):
    lines = [
        f"Cross-channel duplicate-unit analysis — session '{session.label}'",
        f"{len(session.units)} units on {len(session.channels())} channels, "
        f"sample rate {session.sample_rate:g} Hz",
        "",
        f"{len(candidates)} candidate duplicate pair(s) "
        f"(coincidence >= threshold and well above chance):",
        "",
    ]
    if not candidates:
        lines.append("  (none)")
    for rank, p in enumerate(candidates, 1):
        dist = "n/a" if p.distance_um is None else f"{p.distance_um:.0f}um/{p.contacts_apart}ct"
        flag = "" if p.merged_refractory_ok else "  [!] merged train breaks refractoriness"
        lines.append(
            f"  {rank:2d}. {p.uid_a:20s} <-> {p.uid_b:20s}  "
            f"coinc={p.coincidence:.2f}  x{p.ratio_over_chance:.0f}  dist={dist}{flag}")
    lines += ["", "Top pairs overall (by coincidence):"]
    for p in pairs[:10]:
        lines.append(f"     {p.uid_a:20s} <-> {p.uid_b:20s}  coinc={p.coincidence:.2f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_demo(out_dir: str):
    from .make_synthetic_session import make_synthetic_session
    sorted_spikes, voltages, sr = make_synthetic_session(with_voltages=True)
    # build a Session directly from the in-memory dict
    from .loader import SortedUnit
    units = []
    for ch, by_unit in sorted_spikes.items():
        for name, idx in by_unit.items():
            units.append(SortedUnit(ch, name, np.sort(np.asarray(idx, dtype=np.int64))))
    session = Session(units=units, sample_rate=sr, label="synthetic-demo")
    return analyze_session(session, out_dir, voltages_by_channel=voltages)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", nargs="?", help="session dir or path to sorted_spikes.pkl")
    p.add_argument("--out", help="output directory (default: <session>/cross_channel_analysis)")
    p.add_argument("--sample-rate", type=float, default=None,
                   help="override sample rate (else read from info.rhd)")
    p.add_argument("--with-voltages", action="store_true",
                   help="load raw recording for footprint panels (heavy)")
    p.add_argument("--window-ms", type=float, default=0.4, help="coincidence half-window")
    p.add_argument("--coincidence-threshold", type=float, default=0.3)
    p.add_argument("--ratio-threshold", type=float, default=5.0)
    p.add_argument("--demo", action="store_true", help="run on a fabricated session")
    args = p.parse_args(argv)

    if args.demo:
        out = args.out or os.path.join(os.getcwd(), "cross_channel_demo_out")
        _run_demo(out)
        return

    if not args.session:
        p.error("provide a session directory / sorted_spikes.pkl, or use --demo")

    if os.path.isdir(args.session):
        pkl = os.path.join(args.session, "sorted_spikes.pkl")
        session_dir = args.session
    else:
        pkl = args.session
        session_dir = os.path.dirname(os.path.abspath(pkl))

    session = load_sorted_spikes(pkl, sample_rate=args.sample_rate)
    out = args.out or os.path.join(session_dir, "cross_channel_analysis")
    voltages = _maybe_load_voltages(session_dir, args.with_voltages)
    analyze_session(
        session, out,
        voltages_by_channel=voltages,
        coincidence_window_ms=args.window_ms,
        coincidence_threshold=args.coincidence_threshold,
        ratio_threshold=args.ratio_threshold,
    )


if __name__ == "__main__":
    main()
