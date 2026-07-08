"""
exploded_analysis.py — cross-channel coincidence over ALL units in a session.

Unlike the sorted-only pipeline in ``run_analysis.py`` (which also computes
waveform footprints from the raw recording), this works purely from spike trains
in the exploded cache, so it can include **unsorted** whole-channels. It compares
every cross-channel pair — sorted-sorted, sorted-unsorted, unsorted-unsorted —
by **coincidence fraction** (the one piece of evidence available without the raw
voltages), alongside channel distance and the merged-train refractory check.

Per session it writes:
    <out>/<date>_round<round>/
        all_pairs.csv              every cross-channel pair, ranked
        candidate_duplicates.csv   high-coincidence, well-above-chance pairs
        coincidence_matrix.png     units × units heatmap
        distance_vs_coincidence.png  coloured by pair type

Batch mode additionally writes a top-level ``all_candidates.csv`` aggregating
candidate pairs across every session (with ``session`` and ``pair_type`` columns).
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spikesorting.cross_channel_analysis import spiketrain_metrics as stm
from spikesorting.cross_channel_analysis import plots
from spikesorting.cross_channel_analysis.loader import Session
from spikesorting.cross_channel_analysis.exploded_loader import (
    load_exploded_session, find_sessions, pair_type,
)

_PAIR_TYPE_COLORS = {
    "sorted-sorted": "#4E79A7",
    "sorted-unsorted": "#F28E2B",
    "unsorted-unsorted": "#59A14F",
}

_CSV_FIELDS = [
    "pair_type", "uid_a", "uid_b", "channel_a", "channel_b", "n_a", "n_b",
    "coincidence", "ratio_over_chance", "distance_um", "contacts_apart",
    "refractory_a", "refractory_b", "refractory_merged", "merged_refractory_ok",
]


def _row(session: Session, p: stm.PairMetrics, extra: Optional[dict] = None) -> dict:
    row = {
        "pair_type": pair_type(session, p.uid_a, p.uid_b),
        "uid_a": p.uid_a, "uid_b": p.uid_b,
        "channel_a": p.channel_a, "channel_b": p.channel_b,
        "n_a": p.n_a, "n_b": p.n_b,
        "coincidence": round(p.coincidence, 4),
        "ratio_over_chance": round(p.ratio_over_chance, 2),
        "distance_um": "" if p.distance_um is None else round(p.distance_um, 1),
        "contacts_apart": "" if p.contacts_apart is None else p.contacts_apart,
        "refractory_a": round(p.refractory_a, 4),
        "refractory_b": round(p.refractory_b, 4),
        "refractory_merged": round(p.refractory_merged, 4),
        "merged_refractory_ok": p.merged_refractory_ok,
    }
    if extra:
        row.update(extra)
    return row


def _write_rows(rows: List[dict], path: str, fields: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_distance_by_type(session, pairs, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ptype, color in _PAIR_TYPE_COLORS.items():
        pts = [p for p in pairs if pair_type(session, p.uid_a, p.uid_b) == ptype
               and p.distance_um is not None]
        if pts:
            ax.scatter([p.distance_um for p in pts], [p.coincidence for p in pts],
                       c=color, s=28, alpha=0.75, label=ptype, edgecolors="none")
    ax.axhline(0.3, color="crimson", ls="--", lw=1, label="candidate threshold")
    ax.set_xlabel("channel distance (µm)")
    ax.set_ylabel("coincidence fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{session.label}: channel distance vs coincidence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def analyze_exploded_session(
    pkl_path: str,
    out_dir: str,
    *,
    date: Optional[str] = None,
    round_no: Optional[int] = None,
    window_ms: float = 0.4,
    coincidence_threshold: float = 0.3,
    ratio_threshold: float = 5.0,
    min_spikes: int = 50,
    include_sorted: bool = True,
    include_unsorted: bool = True,
    make_figures: bool = True,
) -> List[dict]:
    """Analyse one exploded-cache session. Returns its candidate rows (dicts).

    ``min_spikes`` drops near-silent channels: a unit with only a handful of
    spikes yields a meaningless coincidence of 1.0 against any dense partner
    (and looks like a "duplicate" of everything, even distant channels), so
    those units are excluded before any pairing.
    """
    session = load_exploded_session(
        pkl_path, include_sorted=include_sorted, include_unsorted=include_unsorted,
        min_spikes=min_spikes)
    os.makedirs(out_dir, exist_ok=True)

    n_sorted = sum(1 for u in session.units if u.unit_name != "unsorted")
    n_unsorted = len(session.units) - n_sorted
    print(f"[{session.label}] {len(session.units)} units "
          f"({n_sorted} sorted, {n_unsorted} unsorted) on "
          f"{len(session.channels())} channels")
    if len(session.units) < 2:
        print(f"[{session.label}] fewer than 2 units; skipping")
        return []

    pairs = stm.all_pairs(session, window_ms=window_ms)
    all_rows = [_row(session, p) for p in pairs]
    _write_rows(all_rows, os.path.join(out_dir, "all_pairs.csv"), _CSV_FIELDS)

    candidates = stm.candidate_duplicates(
        pairs, coincidence_threshold=coincidence_threshold, ratio_threshold=ratio_threshold)
    cand_rows = [_row(session, p) for p in candidates]
    _write_rows(cand_rows, os.path.join(out_dir, "candidate_duplicates.csv"), _CSV_FIELDS)

    if make_figures:
        fig = plots.plot_coincidence_matrix(session, window_ms=window_ms)
        fig.savefig(os.path.join(out_dir, "coincidence_matrix.png"), dpi=150)
        plt.close(fig)
        _plot_distance_by_type(session, pairs,
                               os.path.join(out_dir, "distance_vs_coincidence.png"))

    print(f"[{session.label}] {len(candidates)} candidate duplicate pair(s)")
    # tag candidate rows with the session for the master table
    sess_label = session.label
    for r in cand_rows:
        r["session"] = sess_label
    return cand_rows


def analyze_exploded_batch(
    cache_dir: str,
    out_root: str,
    *,
    window_ms: float = 0.4,
    coincidence_threshold: float = 0.3,
    ratio_threshold: float = 5.0,
    min_spikes: int = 50,
    include_sorted: bool = True,
    include_unsorted: bool = True,
    make_figures: bool = True,
) -> str:
    """Analyse every session pkl in ``cache_dir``; write a master candidates CSV.

    Returns the path to ``all_candidates.csv``.
    """
    os.makedirs(out_root, exist_ok=True)
    sessions = find_sessions(cache_dir)
    if not sessions:
        raise FileNotFoundError(f"No '*_round_*.pkl' sessions found in {cache_dir}")
    print(f"Found {len(sessions)} session(s) in {cache_dir}")

    master: List[dict] = []
    for pkl_path, date, round_no in sessions:
        sub = os.path.join(out_root, f"{date}_round{round_no}")
        try:
            cand_rows = analyze_exploded_session(
                pkl_path, sub, date=date, round_no=round_no,
                window_ms=window_ms, coincidence_threshold=coincidence_threshold,
                ratio_threshold=ratio_threshold, min_spikes=min_spikes,
                include_sorted=include_sorted,
                include_unsorted=include_unsorted, make_figures=make_figures)
        except Exception as e:
            print(f"[{date} r{round_no}] FAILED: {e}")
            continue
        master.extend(cand_rows)

    master.sort(key=lambda r: r["coincidence"], reverse=True)
    master_path = os.path.join(out_root, "all_candidates.csv")
    _write_rows(master, master_path, ["session"] + _CSV_FIELDS)
    print(f"\nWrote {len(master)} candidate pair(s) across {len(sessions)} sessions "
          f"→ {master_path}")
    return master_path
