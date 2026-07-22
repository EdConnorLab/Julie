"""
run_benchmark.py — drive the response-window detector bake-off.

Two-phase workflow
------------------
1. ``--mode select``  Pick ~N example cells (stratified across region/unit-type/
   session), write ``candidate_cells.csv`` + a blank ``ground_truth_template.xlsx``,
   and render an UNMARKED raster+PSTH per cell into ``rasters_for_annotation/`` so
   you can inspect and fill in the windows you actually see.

2. ``--mode score``   Run every detector on each cell, render a comparison figure
   per cell (raster + PSTH + one window-lane per method + a TRUTH lane) into
   ``comparison/``, and — if the filled template is present — write
   ``scores_per_cell.csv``, ``scoreboard.csv`` and ``scoreboard.png`` picking the
   method that best matches your annotations.

``--mode score`` also runs without a filled template (it just skips scoring), so
you can eyeball the overlays first.

Needs the lab machine's cell lists + spike caches. Verify the code itself with
``python -m analyses.response_window_benchmark.synthetic_smoke_test`` (no caches).

Run it
------
    cd src
    python -m analyses.response_window_benchmark.run_benchmark --mode select --n 20
    #   ... inspect rasters_for_annotation/, fill ground_truth_template.xlsx ...
    python -m analyses.response_window_benchmark.run_benchmark --mode score
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd

# --- make repo src/ importable when run as a plain script (PyCharm ▶) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analyses.zombies_raster_review.run_zombies_rasters import _select_unit_rows  # noqa: E402
from analyses.zombies_raster_review.zombies_raster import plot_zombies_raster     # noqa: E402
from analyses.response_window_benchmark import cell_selection as cs               # noqa: E402
from analyses.response_window_benchmark import ground_truth as gt                 # noqa: E402
from analyses.response_window_benchmark import scoring                            # noqa: E402
from analyses.response_window_benchmark.detectors import default_detectors        # noqa: E402
from analyses.response_window_benchmark.plotting import plot_cell_with_windows    # noqa: E402

OUT_DIR = os.path.join(_HERE, "output")


def _safe(key: str) -> str:
    return (key.replace(" ", "").replace("/", "-").replace("|", "__")
            .replace(".", "").replace(":", "").replace("\\", "-"))


def _sessions(cands: pd.DataFrame):
    """Group candidate rows by (source_key, date, round) so each session loads once."""
    by = defaultdict(list)
    for _, r in cands.iterrows():
        by[(r["_source_key"], str(r["Date"]), int(r["Round No."]))].append(r)
    return by


def _make_source(source_key: str):
    from analyses.jun2026_grant_investigation.raster_review_by_source import SOURCES
    return SOURCES[source_key].make_source()


def _unit_rows(session_df, row):
    req = SimpleNamespace(match_column=row["_match_column"], match_value=row["_match_value"])
    return _select_unit_rows(session_df, req)


# --------------------------------------------------------------------------- #
# Phase 1 — select + render for annotation
# --------------------------------------------------------------------------- #
def run_select(out_dir: str = OUT_DIR, *, n: int = 20, seed: int = 0,
               xlim: float = 2.4, source_keys: Optional[List[str]] = None) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    cands = cs.build_candidates(n, source_keys=source_keys, seed=seed)
    if cands.empty:
        print("[select] no candidate cells found (are the cell lists available?)")
        return cands
    cs.save_candidates(cands, os.path.join(out_dir, "candidate_cells.csv"))
    gt.write_template(cands, os.path.join(out_dir, "ground_truth_template.xlsx"))

    raster_dir = os.path.join(out_dir, "rasters_for_annotation")
    for (sk, date, rnd), rows in _sessions(cands).items():
        try:
            session_df = _make_source(sk).load(date, rnd)
        except Exception as e:
            print(f"[select {sk} {date} r{rnd}] load failed: {e}")
            continue
        if session_df is None or getattr(session_df, "empty", True):
            print(f"[select {sk} {date} r{rnd}] no data")
            continue
        for row in rows:
            unit = _unit_rows(session_df, row)
            if unit.empty:
                print(f"[select] no rows for {row['cell_key']}")
                continue
            plot_zombies_raster(
                unit, neuron_label=f"{row['Source']} · {row['NeuronID']}",
                window_s=None, xlim=xlim,
                save_path=os.path.join(raster_dir, f"{_safe(row['cell_key'])}.png"))
    print(f"[select] done. Inspect {raster_dir}/ and fill "
          f"{os.path.join(out_dir, 'ground_truth_template.xlsx')}")
    return cands


# --------------------------------------------------------------------------- #
# Phase 2 — run detectors, plot comparison, score
# --------------------------------------------------------------------------- #
def run_score(out_dir: str = OUT_DIR, *, truth_path: Optional[str] = None,
              bin_s: float = 0.05, xlim: float = 2.4, iou_thresh: float = 0.3) -> None:
    cand_csv = os.path.join(out_dir, "candidate_cells.csv")
    if not os.path.exists(cand_csv):
        raise FileNotFoundError(f"{cand_csv} not found — run --mode select first")
    cands = cs.load_candidates(cand_csv)

    truth_path = truth_path or os.path.join(out_dir, "ground_truth_template.xlsx")
    truth = gt.load_truth(truth_path) if os.path.exists(truth_path) else {}
    if not truth:
        print(f"[score] no annotations at {truth_path} — plotting overlays only, no scoring")

    detectors = default_detectors(bin_s=bin_s)
    comp_dir = os.path.join(out_dir, "comparison")
    results_by_cell: Dict[str, Dict[str, list]] = {}

    for (sk, date, rnd), rows in _sessions(cands).items():
        try:
            session_df = _make_source(sk).load(date, rnd)
        except Exception as e:
            print(f"[score {sk} {date} r{rnd}] load failed: {e}")
            continue
        if session_df is None or getattr(session_df, "empty", True):
            continue
        for row in rows:
            key = row["cell_key"]
            unit = _unit_rows(session_df, row)
            if unit.empty:
                print(f"[score] no rows for {key}")
                continue
            from analyses.response_window_benchmark.psth import extract_trials
            td = extract_trials(unit, t_stop=xlim)
            results = {d.name: d.detect(td) for d in detectors}
            results_by_cell[key] = {name: r.windows for name, r in results.items()}
            plot_cell_with_windows(
                unit, detectors, results,
                truth_windows=truth.get(key),
                title=f"{row['Source']} · {row['NeuronID']}  ({row.get('Region','?')}/{row.get('UnitType','?')})",
                bin_s=bin_s, xlim=xlim,
                save_path=os.path.join(comp_dir, f"{_safe(key)}.png"))

    if truth:
        per_cell, scoreboard = scoring.score_all(
            results_by_cell, truth, [d.name for d in detectors], iou_thresh=iou_thresh)
        per_cell.to_csv(os.path.join(out_dir, "scores_per_cell.csv"), index=False)
        scoreboard.to_csv(os.path.join(out_dir, "scoreboard.csv"), index=False)
        scoring.plot_scoreboard(scoreboard, os.path.join(out_dir, "scoreboard.png"))
        print("\n=== SCOREBOARD (best first) ===")
        print(scoreboard.to_string(index=False))
    print(f"[score] comparison figures -> {comp_dir}/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["select", "score"], default="select")
    p.add_argument("--n", type=int, default=20, help="select: how many cells")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--truth", default=None, help="score: path to filled template")
    p.add_argument("--bin-ms", type=float, default=50.0)
    p.add_argument("--xlim", type=float, default=2.4)
    p.add_argument("--iou", type=float, default=0.3)
    args = p.parse_args(argv)

    if args.mode == "select":
        run_select(args.out, n=args.n, seed=args.seed, xlim=args.xlim)
    else:
        run_score(args.out, truth_path=args.truth, bin_s=args.bin_ms / 1000.0,
                  xlim=args.xlim, iou_thresh=args.iou)


if __name__ == "__main__":
    # Command-line args -> argparse (`--mode select ...`); a bare ▶ Run in
    # PyCharm (no args) -> the RUN CONFIG block below.
    if len(sys.argv) > 1:
        _cli()
    else:
        # ====================================================================
        #  RUN CONFIG — edit these, then press ▶ Run in PyCharm.
        #    "select"  pick cells + render unmarked rasters + write the template.
        #    "score"   run all detectors, render comparisons, score vs template.
        # ====================================================================
        MODE = "select"          # "select" | "score"
        N_CELLS = 20             # select: number of example cells
        SEED = 0
        XLIM_S = 2.4
        BIN_MS = 50.0
        IOU_THRESH = 0.3         # score: IoU for counting a predicted window a hit
        # ====================================================================
        if MODE == "select":
            run_select(OUT_DIR, n=N_CELLS, seed=SEED, xlim=XLIM_S)
        else:
            run_score(OUT_DIR, bin_s=BIN_MS / 1000.0, xlim=XLIM_S, iou_thresh=IOU_THRESH)
