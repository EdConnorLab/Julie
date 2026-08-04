"""
cell_selection.py — pick the example cells to benchmark on.

Primary path (:func:`build_si_prestim_candidates`): enumerate SI-sorted units
directly from a pre-stimulus cache (e.g. ``sorted_spike_cache_pre1000ms``, the
pkled sessions built by ``analyze_sorted_spikes.py --pre-stimulus-time``) and
draw a stratified sample across region/session. These carry the real
pre-stimulus baseline the detectors need.

Legacy path (:func:`build_candidates_from_lists`): the five grant/cache cell
lists in ``jun2026_grant_investigation.raster_review_by_source`` — kept for when
the pre-stim caches exist for the MUA/mixed sources too.

The old ``ListWindow_ms`` per cell is reference ONLY (mostly produced by the
current detector), never ground truth.

Requires the lab machine's caches; not runnable in a bare checkout (the smoke
test uses synthetic data).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

from .keys import make_cell_key, infer_region, infer_unit_type

# Runner-only columns (carried in candidate_cells.csv, not in the blank template)
RUNNER_COLS = ["_source_key", "_match_column", "_match_value", "_cache_subdir", "_pre_stim"]


# --------------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------------- #
def _stratified_sample(pool: pd.DataFrame, n: int, seed: int,
                       strata_key: Callable[[pd.Series], object]) -> pd.DataFrame:
    """Round-robin across strata, preferring not-yet-used sessions, until ``n``."""
    if pool.empty:
        return pool
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    strata = defaultdict(list)
    for _, r in pool.iterrows():
        strata[strata_key(r)].append(r)

    picked, seen_sessions, keys = [], defaultdict(int), list(strata)
    idx = {k: 0 for k in keys}
    while len(picked) < min(n, len(pool)):
        advanced = False
        for k in keys:
            if len(picked) >= n:
                break
            bucket, j = strata[k], idx[k]
            while j < len(bucket):
                cand = bucket[j]
                sess = (cand["Date"], cand["Round No."])
                if seen_sessions[sess] == 0 or j == len(bucket) - 1:
                    picked.append(cand)
                    seen_sessions[sess] += 1
                    idx[k] = j + 1
                    advanced = True
                    break
                j += 1
            else:
                idx[k] = j
        if not advanced:
            break
    return pd.DataFrame(picked).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Primary: SI-sorted pre-stim cache
# --------------------------------------------------------------------------- #
def _cache_dir(cache_subdir: str) -> Path:
    from project_util import DATA_BASE_PATH, SUBJECT_MONKEY
    return Path(DATA_BASE_PATH) / SUBJECT_MONKEY / cache_subdir


def scan_prestim_sessions(cache_subdir: str) -> List[Tuple[str, int]]:
    """(date, round) for every ``{date}_round_{n}.pkl`` in the cache dir."""
    d = _cache_dir(cache_subdir)
    sessions = []
    for p in sorted(d.glob("*_round_*.pkl")):
        m = re.match(r"(.+)_round_(\d+)\.pkl$", p.name)
        if m:
            sessions.append((m.group(1), int(m.group(2))))
    if not sessions:
        print(f"[cell-selection] no session pkls found in {d}")
    return sessions


def build_si_prestim_candidates(
    *,
    cache_subdir: str = "sorted_spike_cache_pre1000ms",
    pre_stim: float = 1.0,
    n: int = 20,
    seed: int = 0,
    sessions: Optional[List[Tuple[str, int]]] = None,
) -> pd.DataFrame:
    """Stratified sample of ``n`` SI-sorted units from the pre-stim cache.

    ``sessions`` restricts to specific (date, round) pairs; if None, every
    session pkl in the cache is scanned. Returns META + RUNNER columns.
    """
    from data_access.spike_source import SISortedSpikeSource
    # Sources clip to the strict window unless asked otherwise, so read the window
    # this cache holds back out of its name and ask for all of it.
    from data_access.spike_window import cache_pre_stimulus_time
    src = SISortedSpikeSource(cache_subdir=cache_subdir,
                              pre_stimulus_time=cache_pre_stimulus_time(cache_subdir))
    if sessions is None:
        sessions = scan_prestim_sessions(cache_subdir)

    rows = []
    for date, rnd in sessions:
        try:
            df = src.load(date, rnd)
        except Exception as e:
            print(f"[cell-selection] {date} r{rnd}: load failed ({e})")
            continue
        if df is None or getattr(df, "empty", True) or "NeuronID" not in df.columns:
            continue
        for nid in df["NeuronID"].dropna().unique():
            mv = str(nid)
            rows.append({
                "cell_key": make_cell_key("si_prestim", date, rnd, mv),
                "Source": "si_prestim",
                "NeuronID": mv,
                "Date": str(date),
                "Round No.": int(rnd),
                "Region": infer_region(mv),
                "UnitType": infer_unit_type(mv),
                "ListWindow_ms": "",
                "_source_key": "si_prestim",
                "_match_column": "NeuronID",
                "_match_value": mv,
                "_cache_subdir": cache_subdir,
                "_pre_stim": pre_stim,
            })
    pool = pd.DataFrame(rows)
    if pool.empty:
        print("[cell-selection] no SI-sorted units found in the pre-stim cache")
        return pool
    pool = pool.drop_duplicates("cell_key").reset_index(drop=True)
    out = _stratified_sample(pool, n, seed, lambda r: r["Region"])
    print(f"[cell-selection] {len(out)}/{len(pool)} SI-sorted cells "
          f"({out['Region'].nunique()} regions, "
          f"{out.groupby(['Date','Round No.']).ngroups} sessions)")
    return out


# --------------------------------------------------------------------------- #
# Legacy: the five grant/cache lists (for when their pre-stim caches exist)
# --------------------------------------------------------------------------- #
def _all_candidate_rows(source_keys: Optional[List[str]] = None) -> pd.DataFrame:
    from analyses.jun2026_grant_investigation.raster_review_by_source import SOURCES
    from analyses.zombies_raster_review.unit_lists import dedup_by_unit

    source_keys = source_keys or list(SOURCES)
    rows = []
    for sk in source_keys:
        spec = SOURCES[sk]
        try:
            reqs = dedup_by_unit(spec.load_requests())
        except Exception as e:
            print(f"[cell-selection] {sk}: could not load list ({e})")
            continue
        for r in reqs:
            mv = str(r.match_value)
            rows.append({
                "cell_key": make_cell_key(sk, r.date, r.round_no, mv),
                "Source": spec.display, "NeuronID": mv,
                "Date": str(r.date), "Round No.": int(r.round_no),
                "Region": infer_region(mv, r.label), "UnitType": infer_unit_type(mv, sk),
                "ListWindow_ms": (f"{r.window_ms[0]:.0f}-{r.window_ms[1]:.0f}" if r.window_ms else ""),
                "_source_key": sk, "_match_column": r.match_column, "_match_value": mv,
                "_cache_subdir": "", "_pre_stim": 0.0,
            })
    df = pd.DataFrame(rows)
    return df.drop_duplicates("cell_key").reset_index(drop=True) if not df.empty else df


def build_candidates_from_lists(n: int = 20, *, source_keys: Optional[List[str]] = None,
                                seed: int = 0) -> pd.DataFrame:
    pool = _all_candidate_rows(source_keys)
    if pool.empty:
        return pool
    out = _stratified_sample(pool, n, seed, lambda r: (r["Region"], r["UnitType"]))
    print(f"[cell-selection] {len(out)} cells "
          f"({out['Region'].nunique()} regions, {out['UnitType'].nunique()} unit types, "
          f"{out.groupby(['Date','Round No.']).ngroups} sessions)")
    return out


# --------------------------------------------------------------------------- #
def save_candidates(df: pd.DataFrame, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[cell-selection] candidates -> {path}")
    return path


def load_candidates(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
