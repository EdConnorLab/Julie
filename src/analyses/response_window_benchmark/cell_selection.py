"""
cell_selection.py — pick the example cells to benchmark on.

The candidate pool is the union of the five cell lists wired up in
``analyses.jun2026_grant_investigation.raster_review_by_source`` (grant_xlsx +
the four SI/MUA cache lists), which already span brain regions and sessions.
:func:`build_candidates` draws a stratified sample across (region x unit-type),
spreading sessions, so ~20 cells cover the variety you need for a fair bake-off.

The pre-existing ``ListWindow_ms`` for each cell is carried through for reference
ONLY — most were produced by the current detector, so they are not ground truth.

Requires the lab machine's cell-list files + spike caches; not runnable in a bare
checkout (the smoke test uses synthetic data instead).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

import pandas as pd

from .keys import make_cell_key, infer_region, infer_unit_type

# Runner-only columns (carried in candidate_cells.csv, not in the blank template)
RUNNER_COLS = ["_source_key", "_match_column", "_match_value"]


def _all_candidate_rows(source_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per unique unit across the requested SOURCES lists."""
    # Imported lazily: raster_review_by_source pulls in the whole plotting stack.
    from analyses.jun2026_grant_investigation.raster_review_by_source import SOURCES
    from analyses.zombies_raster_review.unit_lists import dedup_by_unit

    source_keys = source_keys or list(SOURCES)
    rows = []
    for sk in source_keys:
        spec = SOURCES[sk]
        try:
            reqs = dedup_by_unit(spec.load_requests())
        except Exception as e:  # missing list file
            print(f"[cell-selection] {sk}: could not load list ({e})")
            continue
        for r in reqs:
            mv = str(r.match_value)
            rows.append({
                "cell_key": make_cell_key(sk, r.date, r.round_no, mv),
                "Source": spec.display,
                "NeuronID": mv,
                "Date": str(r.date),
                "Round No.": int(r.round_no),
                "Region": infer_region(mv, r.label),
                "UnitType": infer_unit_type(mv, sk),
                "ListWindow_ms": (f"{r.window_ms[0]:.0f}-{r.window_ms[1]:.0f}"
                                  if r.window_ms else ""),
                "_source_key": sk,
                "_match_column": r.match_column,
                "_match_value": mv,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("cell_key").reset_index(drop=True)
    return df


def build_candidates(
    n: int = 20,
    *,
    source_keys: Optional[List[str]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Stratified sample of ``n`` cells across (Region x UnitType), spreading
    sessions. Deterministic given ``seed``. Returns META + RUNNER columns."""
    pool = _all_candidate_rows(source_keys)
    if pool.empty:
        return pool
    # shuffle deterministically, then round-robin across strata so no single
    # region/unit-type/session dominates the sample
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    strata = defaultdict(list)
    for _, r in pool.iterrows():
        strata[(r["Region"], r["UnitType"])].append(r)

    picked, seen_sessions = [], defaultdict(int)
    keys = list(strata)
    idx = {k: 0 for k in keys}
    while len(picked) < min(n, len(pool)):
        advanced = False
        for k in keys:
            if len(picked) >= n:
                break
            bucket = strata[k]
            # within a stratum, prefer a not-yet-used session first
            j = idx[k]
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
    out = pd.DataFrame(picked).reset_index(drop=True)
    print(f"[cell-selection] {len(out)} cells "
          f"({out['Region'].nunique()} regions, {out['UnitType'].nunique()} unit types, "
          f"{out.groupby(['Date','Round No.']).ngroups} sessions)")
    return out


def save_candidates(df: pd.DataFrame, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[cell-selection] candidates -> {path}")
    return path


def load_candidates(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
