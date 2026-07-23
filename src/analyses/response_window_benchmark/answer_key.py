"""
answer_key.py — ingest Ed's hand-picked window cells as the scoring ground truth.

The answer-key xlsx (e.g. ``Ed_handpicked_window_cells_ANOVA_passed_Zombies.xlsx``)
has one row per (cell, window):

    Date | Round No. | Time Window | Cell | P Value

* ``Time Window`` is ``(start_ms, end_ms)`` — the window Ed marked by eye.
* ``Cell`` uses the OLD naming: ``Channel.C_027_Unit 1`` (a manually-sorted unit)
  or ``Channel.C_004`` (an online-thresholded whole channel).

Matching to the pre-stim caches
-------------------------------
Region is not stored here and is not needed: within one (Date, Round No.) a given
``Channel.C_XXX`` is unique, so we load that session from the pre-stim exploded
cache and match the ``Channel`` column to ``Cell`` (the real, region-carrying
NeuronID is then read back from the matched rows for display).

Only manually-sorted units (``_Unit`` in the name) have pre-stimulus spikes in
``exploded_spike_cache_pre1000ms``; online-thresholded channels were recorded
only within the stimulus epoch, so their pre-stim window is empty. Those are
skipped by default (``skip_unsorted=True``) and reported.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .keys import make_cell_key

Window = Tuple[float, float]

DEFAULT_CACHE_SUBDIR = "exploded_spike_cache_pre1000ms"
DEFAULT_SOURCE_KIND = "mixed_prestim"


def _parse_window_ms(s) -> Window:
    """``"(0.0, 300.0)"`` (ms) -> ``(0.0, 0.3)`` (s). Accepts tuples too."""
    if isinstance(s, (tuple, list)):
        a, b = float(s[0]), float(s[1])
    else:
        a, b = (float(x) for x in str(s).strip().strip("()").split(","))
    return (a / 1000.0, b / 1000.0)


def is_manual_unit(cell: str) -> bool:
    """True for a manually-sorted unit (has ``_Unit``) — the ones with pre-stim."""
    return "_Unit" in str(cell)


def load_answer_key(
    xlsx_path: str,
    *,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    pre_stim: float = 1.0,
    source_kind: str = DEFAULT_SOURCE_KIND,
    skip_unsorted: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[Window]], List[Tuple[str, str]]]:
    """Parse the answer key into (candidates_df, truth_by_cell, skipped).

    * ``candidates_df`` — one row per usable (session, cell) with the META +
      RUNNER columns the benchmark runner consumes.
    * ``truth_by_cell`` — ``{cell_key: [(start_s, end_s), ...]}`` (a cell may have
      several windows within a session).
    * ``skipped`` — ``[(cell_key, reason), ...]`` for cells left out (e.g. online-
      thresholded with no pre-stimulus data).
    """
    df = pd.read_excel(xlsx_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Round No."] = df["Round No."].astype(int)
    df["Cell"] = df["Cell"].astype(str).str.strip()

    rows: List[dict] = []
    truth: Dict[str, List[Window]] = {}
    skipped: List[Tuple[str, str]] = []

    for (date, rnd, cell), g in df.groupby(["Date", "Round No.", "Cell"]):
        windows = [_parse_window_ms(w) for w in g["Time Window"]]
        unit_type = "manual_SU" if is_manual_unit(cell) else "unsorted(online)"
        cell_key = make_cell_key(source_kind, date, rnd, cell)

        if skip_unsorted and not is_manual_unit(cell):
            skipped.append((cell_key, "online-thresholded: no pre-stimulus data"))
            continue

        p = float(g["P Value"].min()) if "P Value" in g.columns else float("nan")
        rows.append({
            "cell_key": cell_key,
            "Source": source_kind,
            "NeuronID": cell,                       # provisional; resolved from cache at run time
            "Date": date,
            "Round No.": int(rnd),
            "Region": "?",
            "UnitType": unit_type,
            "AnswerWindow_ms": "; ".join(f"{int(a*1000)}-{int(b*1000)}" for a, b in windows),
            "AnswerP": p,
            "_source_key": source_kind,
            "_match_column": "Channel",
            "_match_value": cell,
            "_cache_subdir": cache_subdir,
            "_pre_stim": pre_stim,
        })
        truth[cell_key] = windows

    candidates = pd.DataFrame(rows)
    print(f"[answer-key] {len(candidates)} usable cell(s), {len(skipped)} skipped "
          f"(no pre-stim). Sessions: "
          f"{candidates.groupby(['Date','Round No.']).ngroups if not candidates.empty else 0}")
    for ck, why in skipped:
        print(f"    [skip] {ck}  — {why}")
    return candidates, truth, skipped
