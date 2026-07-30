"""
run_benchmark.py — drive the response-window detector bake-off.

Detectors measure baseline from the real pre-stimulus window, so cells must come
from a pre-stim cache (spikes retained from ``onset - pre`` onward).

Modes
-----
* ``--mode answerkey``  Score against Ed's hand-picked window cells. Reads the
  answer-key xlsx (cells + true windows), matches each cell to the pre-stim
  exploded cache (``exploded_spike_cache_pre1000ms``) by (Date, Round, Channel),
  runs every detector, renders a comparison figure per cell, and writes the
  scoreboard. Online-thresholded cells (no pre-stim) are skipped + reported.
  This is the main path today.

* ``--mode select`` / ``--mode score``  The generic two-phase workflow for
  SI-sorted cells enumerated straight from a pre-stim cache (select -> you
  annotate a template -> score). Kept for when you want to test cells that
  aren't in the answer key.

Needs the lab machine's pre-stim caches. Verify the code with
``python -m analyses.response_window_benchmark.synthetic_smoke_test`` (no caches).

    cd src
    python -m analyses.response_window_benchmark.run_benchmark --mode answerkey \
        --answer-key /path/to/Ed_handpicked_window_cells_ANOVA_passed_Zombies.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- make repo src/ importable when run as a plain script (PyCharm ▶) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analyses.zombies_raster_review.run_zombies_rasters import _select_unit_rows  # noqa: E402
from analyses.response_window_benchmark import cell_selection as cs               # noqa: E402
from analyses.response_window_benchmark import ground_truth as gt                 # noqa: E402
from analyses.response_window_benchmark import answer_key as ak                   # noqa: E402
from analyses.response_window_benchmark import scoring                            # noqa: E402
from analyses.response_window_benchmark.psth import extract_trials               # noqa: E402
from analyses.response_window_benchmark.detectors import (                        # noqa: E402
    default_detectors, build_detectors,
)
from analyses.response_window_benchmark.plotting import (                         # noqa: E402
    plot_cell_raster, plot_cell_with_windows,
)

OUT_DIR = os.path.join(_HERE, "output")
DEFAULT_CACHE_SUBDIR = "sorted_spike_cache_pre1000ms"
DEFAULT_PRE_STIM = 1.0


def _safe(key: str) -> str:
    return (key.replace(" ", "").replace("/", "-").replace("|", "__")
            .replace(".", "").replace(":", "").replace("\\", "-"))


def _sessions(cands: pd.DataFrame):
    """Group candidate rows by (source_key, cache_subdir, date, round)."""
    by = defaultdict(list)
    for _, r in cands.iterrows():
        by[(r["_source_key"], str(r.get("_cache_subdir", "")),
            str(r["Date"]), int(r["Round No."]))].append(r)
    return by


def _make_source(source_key: str, cache_subdir: str):
    if source_key == "si_prestim":
        from data_access.spike_source import SISortedSpikeSource
        return SISortedSpikeSource(cache_subdir=cache_subdir)
    if source_key == "mixed_prestim":
        from data_access.spike_source import MixedManualSpikeSource
        return MixedManualSpikeSource(cache_subdir=cache_subdir)
    if source_key == "mua_prestim":
        from data_access.spike_source import ThresholdMUASpikeSource
        try:
            return ThresholdMUASpikeSource(cache_subdir=cache_subdir)  # if supported
        except TypeError:
            return ThresholdMUASpikeSource()
    from analyses.jun2026_grant_investigation.raster_review_by_source import SOURCES
    return SOURCES[source_key].make_source()


def _unit_rows(session_df, row):
    mc, mv = row["_match_column"], row["_match_value"]
    if mc == "NeuronID_regionless":
        # match on the NeuronID with its {Region}_ prefix stripped, so the sheet's
        # region (even "Unknown") never has to equal the cache's region.
        nid = session_df["NeuronID"].astype(str)
        regionless = nid.str.split("_", n=1).str[1]
        return session_df[regionless == mv]
    req = SimpleNamespace(match_column=mc, match_value=mv)
    return _select_unit_rows(session_df, req)


def _pre_of(rows) -> float:
    try:
        return float(rows[0].get("_pre_stim", 0.0) or 0.0)
    except Exception:
        return 0.0


def _resolved_neuron_id(rows_df, fallback: str) -> str:
    if "NeuronID" in rows_df.columns and len(rows_df):
        vals = rows_df["NeuronID"].dropna().unique()
        if len(vals):
            return str(vals[0])
    return fallback


def _n_prestim_spikes(td) -> int:
    return int(sum(int((t < 0).sum()) for t in td.trials)) if td.n_trials else 0


# --------------------------------------------------------------------------- #
# Core loop shared by score / answerkey
# --------------------------------------------------------------------------- #
def _apply_remap(cands: pd.DataFrame, remap: Dict[str, str]) -> pd.DataFrame:
    """Translate each cell's match value through a cross-cache unit remap.

    The answer key names units as they appear in the cache the windows were drawn
    from; a different cache can give the SAME neuron a different unit number (see
    cache_diagnostics). Cells with no confident counterpart are dropped rather
    than silently scored against the wrong neuron.
    """
    out, dropped = [], []
    for _, r in cands.iterrows():
        mv = str(r["_match_value"])
        if mv in remap:
            r = r.copy()
            if remap[mv] != mv:
                r["_remapped_from"] = mv
            r["_match_value"] = remap[mv]
            out.append(r)
        else:
            dropped.append(mv)
    if dropped:
        print(f"[bench] {len(dropped)} cell(s) dropped — no confident counterpart "
              f"in the target cache:")
        for d in dropped:
            print(f"    [no-remap] {d}")
    df = pd.DataFrame(out).reset_index(drop=True) if out else pd.DataFrame(columns=cands.columns)
    n_changed = int(df["_remapped_from"].notna().sum()) if "_remapped_from" in df.columns else 0
    print(f"[bench] remap: {len(df)}/{len(cands)} cell(s) kept, {n_changed} renamed")
    return df


def _benchmark(cands: pd.DataFrame, truth: Dict[str, list], out_dir: str, *,
               bin_s: float, xlim: Optional[float], iou_thresh: float,
               plot_subdir: str = "comparison",
               remap: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, list]]:
    # bin_s=None -> per-cell Shimazaki-Shinomoto width (see detectors.auto_bin_s)
    if remap:
        cands = _apply_remap(cands, remap)
        if cands.empty:
            print("[bench] nothing left to score after remapping")
            return {}
    comp_dir = os.path.join(out_dir, plot_subdir)
    results_by_cell: Dict[str, Dict[str, list]] = {}
    meta_by_cell: Dict[str, dict] = {}
    method_names = [d.name for d in default_detectors()]

    for (sk, csub, date, rnd), rows in _sessions(cands).items():
        try:
            session_df = _make_source(sk, csub).load(date, rnd)
        except Exception as e:
            print(f"[bench {sk} {date} r{rnd}] load failed: {e}")
            continue
        if session_df is None or getattr(session_df, "empty", True):
            print(f"[bench {sk} {date} r{rnd}] session not in cache — skipped")
            continue
        pre = _pre_of(rows)
        for row in rows:
            key = row["cell_key"]
            unit = _unit_rows(session_df, row)
            if unit.empty:
                print(f"[bench] cell not found in session: {key}")
                continue
            td = extract_trials(unit, t_stop=xlim, pre_stimulus_time=pre)
            if td.n_trials == 0:
                print(f"[bench] no trials for {key}")
                continue
            if pre > 0 and _n_prestim_spikes(td) == 0:
                print(f"[bench][warn] {key}: pre-stim window empty — vs-baseline "
                      f"detectors will be unreliable/skipped")
            nid = _resolved_neuron_id(unit, row.get("NeuronID", key))
            detectors = build_detectors(td, bin_s)
            cell_bin = detectors[0].bin_s
            meta_by_cell[key] = {"NeuronID": nid, "UnitType": row.get("UnitType", "?"),
                                 "Region": row.get("Region", "?"),
                                 "bin_ms": round(cell_bin * 1000, 1),
                                 "n_trials": td.n_trials,
                                 "t_stop_s": round(td.t_stop, 2)}
            results = {}
            for d in detectors:
                try:
                    results[d.name] = d.detect(td)
                except Exception as e:
                    print(f"[bench] {d.name} failed on {key}: {e}")
            results_by_cell[key] = {name: r.windows for name, r in results.items()}
            title = f"{nid}   ({row.get('UnitType','?')})   bin={cell_bin*1000:.0f}ms"
            if key in truth:
                title += "   truth=" + ",".join(
                    f"{int(a*1000)}-{int(b*1000)}ms" for a, b in truth[key])
            plot_cell_with_windows(
                td, detectors, results, truth_windows=truth.get(key),
                title=title, bin_s=cell_bin, xlim=xlim,
                save_path=os.path.join(comp_dir, f"{_safe(key)}.png"))

    if truth:
        scored = {k: v for k, v in truth.items() if k in results_by_cell}
        per_cell, scoreboard = scoring.score_all(
            results_by_cell, scored, method_names, iou_thresh=iou_thresh)
        diag = scoring.diagnostic_table(results_by_cell, scored, method_names,
                                        meta_by_cell=meta_by_cell, iou_thresh=iou_thresh)
        per_cell.to_csv(os.path.join(out_dir, "scores_per_cell.csv"), index=False)
        scoreboard.to_csv(os.path.join(out_dir, "scoreboard.csv"), index=False)
        diag.to_csv(os.path.join(out_dir, "diagnostic_table.csv"), index=False)
        scoring.plot_scoreboard(scoreboard, os.path.join(out_dir, "scoreboard.png"))
        print("\n=== SCOREBOARD (best first) ===")
        print(scoreboard.to_string(index=False))
        print(f"\n[bench] per-cell diagnostic table -> {os.path.join(out_dir, 'diagnostic_table.csv')}")
    print(f"[bench] comparison figures -> {comp_dir}/")
    return results_by_cell


# --------------------------------------------------------------------------- #
# Mode: answerkey
# --------------------------------------------------------------------------- #
def run_answer_key(xlsx_path: str, out_dir: str = OUT_DIR, *,
                   cache_subdir: str = ak.DEFAULT_CACHE_SUBDIR, pre_stim: float = 1.0,
                   bin_s: Optional[float] = None, xlim: Optional[float] = None,
                   iou_thresh: float = 0.3, skip_unsorted: bool = True,
                   remap_csv: Optional[str] = None,
                   remap_min_coincidence: float = 0.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cands, truth, skipped = ak.load_answer_key(
        xlsx_path, cache_subdir=cache_subdir, pre_stim=pre_stim, skip_unsorted=skip_unsorted)
    if cands.empty:
        print("[answerkey] no usable cells")
        return
    cs.save_candidates(cands, os.path.join(out_dir, "answerkey_cells.csv"))
    pd.DataFrame(skipped, columns=["cell_key", "reason"]).to_csv(
        os.path.join(out_dir, "answerkey_skipped.csv"), index=False)
    remap = None
    if remap_csv:
        if os.path.exists(remap_csv):
            from analyses.response_window_benchmark.cache_diagnostics import load_unit_remap
            remap = load_unit_remap(remap_csv, min_coincidence=remap_min_coincidence)
        else:
            print(f"[answerkey] remap CSV not found: {remap_csv} — "
                  f"run cache_diagnostics first, or set REMAP_CSV = None")
    _benchmark(cands, truth, out_dir, bin_s=bin_s, xlim=xlim, iou_thresh=iou_thresh,
               remap=remap)


# --------------------------------------------------------------------------- #
# Mode: inspect — why didn't a requested cell match?
# --------------------------------------------------------------------------- #
def run_inspect(xlsx_path: str, out_dir: str = OUT_DIR, *,
                cache_subdir: str = ak.DEFAULT_CACHE_SUBDIR, pre_stim: float = 1.0) -> pd.DataFrame:
    """For every answer-key cell, report whether its session loaded, whether the
    unit was found, and — when not — which units that session actually contains.

    Answers "[bench] cell not found in session" without guessing: usually the SI
    sort produced different unit numbers than the answer key expects.
    """
    cands, truth, _ = ak.load_answer_key(xlsx_path, cache_subdir=cache_subdir, pre_stim=pre_stim)
    rows = []
    for (sk, csub, date, rnd), reqs in _sessions(cands).items():
        try:
            session_df = _make_source(sk, csub).load(date, rnd)
            err = ""
        except Exception as e:
            session_df, err = None, str(e)
        available = []
        if session_df is not None and not getattr(session_df, "empty", True):
            available = sorted(session_df["NeuronID"].astype(str).unique())
        for row in reqs:
            want = row["_match_value"]                       # regionless id
            found = not _unit_rows(session_df, row).empty if available else False
            # same base channel, different unit number?
            base = str(want).split("_Unit")[0]
            near = [a for a in available if base in a]
            rows.append({
                "cell_key": row["cell_key"], "Date": date, "Round No.": rnd,
                "requested": want,
                "session_loaded": bool(available), "unit_found": found,
                "n_units_in_session": len(available),
                "same_channel_units": "; ".join(near) or "(none)",
                "error": err,
            })
    rep = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "inspect_matching.csv")
    rep.to_csv(path, index=False)
    miss = rep[~rep["unit_found"]]
    print(f"\n[inspect] {len(rep) - len(miss)}/{len(rep)} cells matched; "
          f"{len(miss)} did not")
    for _, r in miss.iterrows():
        why = ("session not in cache" if not r["session_loaded"]
               else f"unit absent; same-channel units present: {r['same_channel_units']}")
        print(f"    {r['requested']}  —  {why}")
    print(f"[inspect] full report -> {path}")
    return rep


# --------------------------------------------------------------------------- #
# Mode: select (SI-sorted from a pre-stim cache) + render for annotation
# --------------------------------------------------------------------------- #
def run_select(out_dir: str = OUT_DIR, *, n: int = 20, seed: int = 0,
               xlim: Optional[float] = None, cache_subdir: str = DEFAULT_CACHE_SUBDIR,
               pre_stim: float = DEFAULT_PRE_STIM,
               sessions: Optional[List[Tuple[str, int]]] = None) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    cands = cs.build_si_prestim_candidates(cache_subdir=cache_subdir, pre_stim=pre_stim,
                                           n=n, seed=seed, sessions=sessions)
    if cands.empty:
        print("[select] no candidate cells found (is the pre-stim cache present?)")
        return cands
    cs.save_candidates(cands, os.path.join(out_dir, "candidate_cells.csv"))
    gt.write_template(cands, os.path.join(out_dir, "ground_truth_template.xlsx"))

    raster_dir = os.path.join(out_dir, "rasters_for_annotation")
    for (sk, csub, date, rnd), rows in _sessions(cands).items():
        try:
            session_df = _make_source(sk, csub).load(date, rnd)
        except Exception as e:
            print(f"[select {sk} {date} r{rnd}] load failed: {e}")
            continue
        if session_df is None or getattr(session_df, "empty", True):
            continue
        pre = _pre_of(rows)
        for row in rows:
            unit = _unit_rows(session_df, row)
            if unit.empty:
                continue
            td = extract_trials(unit, t_stop=xlim, pre_stimulus_time=pre)
            plot_cell_raster(td, title=f"{row['Source']} · {row['NeuronID']}",
                             save_path=os.path.join(raster_dir, f"{_safe(row['cell_key'])}.png"))
    print(f"[select] done. Inspect {raster_dir}/ and fill "
          f"{os.path.join(out_dir, 'ground_truth_template.xlsx')}")
    return cands


def run_score(out_dir: str = OUT_DIR, *, truth_path: Optional[str] = None,
              bin_s: Optional[float] = None, xlim: Optional[float] = None,
              iou_thresh: float = 0.3) -> None:
    cand_csv = os.path.join(out_dir, "candidate_cells.csv")
    if not os.path.exists(cand_csv):
        raise FileNotFoundError(f"{cand_csv} not found — run --mode select first")
    cands = cs.load_candidates(cand_csv)
    truth_path = truth_path or os.path.join(out_dir, "ground_truth_template.xlsx")
    truth = gt.load_truth(truth_path) if os.path.exists(truth_path) else {}
    if not truth:
        print(f"[score] no annotations at {truth_path} — plotting overlays only")
    _benchmark(cands, truth, out_dir, bin_s=bin_s, xlim=xlim, iou_thresh=iou_thresh)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["answerkey", "inspect", "select", "score"],
                   default="answerkey")
    p.add_argument("--answer-key", default=None, help="answerkey: path to the xlsx")
    p.add_argument("--n", type=int, default=20, help="select: how many cells")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--cache-subdir", default=None,
                   help="pre-stim cache subdir (default depends on mode)")
    p.add_argument("--pre-stim", type=float, default=DEFAULT_PRE_STIM,
                   help="seconds of pre-stimulus baseline (must match the cache)")
    p.add_argument("--truth", default=None, help="score: path to filled template")
    p.add_argument("--bin-ms", type=float, default=None,
                   help="PSTH bin in ms; omit for a per-cell optimal width "
                        "(Shimazaki-Shinomoto), which is the default")
    p.add_argument("--xlim", type=float, default=None,
                   help="analysis/plot end (s); default = each cell's epoch end")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--remap", default=None,
                   help="cache_unit_remap.csv from cache_diagnostics; translates "
                        "answer-key unit names into the target cache's names")
    p.add_argument("--remap-min-coincidence", type=float, default=0.5)
    args = p.parse_args(argv)

    if args.mode == "inspect":
        if not args.answer_key:
            p.error("--answer-key is required for --mode inspect")
        run_inspect(args.answer_key, args.out,
                    cache_subdir=args.cache_subdir or ak.DEFAULT_CACHE_SUBDIR,
                    pre_stim=args.pre_stim)
    elif args.mode == "answerkey":
        if not args.answer_key:
            p.error("--answer-key is required for --mode answerkey")
        run_answer_key(args.answer_key, args.out,
                       cache_subdir=args.cache_subdir or ak.DEFAULT_CACHE_SUBDIR,
                       pre_stim=args.pre_stim, bin_s=(args.bin_ms / 1000.0 if args.bin_ms else None),
                       xlim=args.xlim, iou_thresh=args.iou,
                       remap_csv=args.remap,
                       remap_min_coincidence=args.remap_min_coincidence)
    elif args.mode == "select":
        run_select(args.out, n=args.n, seed=args.seed, xlim=args.xlim,
                   cache_subdir=args.cache_subdir or DEFAULT_CACHE_SUBDIR, pre_stim=args.pre_stim)
    else:
        run_score(args.out, truth_path=args.truth, bin_s=(args.bin_ms / 1000.0 if args.bin_ms else None),
                  xlim=args.xlim, iou_thresh=args.iou)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        # ====================================================================
        #  RUN CONFIG — edit these, then press ▶ Run in PyCharm.
        # ====================================================================
        MODE = "answerkey"                           # "answerkey" | "select" | "score"
        # -- answerkey --
        ANSWER_KEY_XLSX = ""                          # path to Ed's handpicked xlsx
        ANSWERKEY_CACHE = ak.DEFAULT_CACHE_SUBDIR     # exploded_spike_cache_pre1000ms
        # -- select (SI-sorted) --
        N_CELLS, SEED = 20, 0
        SELECT_CACHE = DEFAULT_CACHE_SUBDIR           # sorted_spike_cache_pre1000ms
        SESSIONS = None                               # e.g. [("2023-09-26", 2)]; None = all
        # -- common --
        PRE_STIM = DEFAULT_PRE_STIM                   # seconds; must match the cache
        XLIM = None                                   # None = each cell's epoch end
        BIN_MS = None   # None = per-cell optimal bin width
        IOU_THRESH = 0.3
        # Cross-cache unit remap from cache_diagnostics.py. The answer key names
        # units as they appear in the cache the windows were drawn from; the
        # pre-stim cache can call the SAME neuron something else. Set to None to
        # score by raw name (only correct if the caches agree).
        REMAP_CSV = os.path.join(OUT_DIR, "cache_unit_remap.csv")
        REMAP_MIN_COINCIDENCE = 0.5
        # ====================================================================
        if MODE == "answerkey":
            if not ANSWER_KEY_XLSX:
                raise SystemExit("set ANSWER_KEY_XLSX (path to the handpicked xlsx)")
            run_answer_key(ANSWER_KEY_XLSX, OUT_DIR, cache_subdir=ANSWERKEY_CACHE,
                           pre_stim=PRE_STIM, bin_s=(BIN_MS / 1000.0 if BIN_MS else None), xlim=XLIM,
                           iou_thresh=IOU_THRESH, remap_csv=REMAP_CSV,
                           remap_min_coincidence=REMAP_MIN_COINCIDENCE)
        elif MODE == "select":
            run_select(OUT_DIR, n=N_CELLS, seed=SEED, xlim=XLIM,
                       cache_subdir=SELECT_CACHE, pre_stim=PRE_STIM, sessions=SESSIONS)
        else:
            run_score(OUT_DIR, bin_s=(BIN_MS / 1000.0 if BIN_MS else None), xlim=XLIM, iou_thresh=IOU_THRESH)
