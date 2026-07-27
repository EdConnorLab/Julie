"""
answer_key.py — ingest hand-picked window cells as the scoring ground truth.

Two spreadsheet layouts are auto-detected:

FORMAT A — "cell" layout (e.g. Ed_handpicked_window_cells_ANOVA_passed_Zombies.xlsx)
    Date | Round No. | Time Window | Cell | [P Value]
  * ``Time Window`` = ``(start_ms, end_ms)``.
  * ``Cell`` uses OLD naming: ``Channel.C_027_Unit 1`` (a manually-sorted unit) or
    ``Channel.C_004`` (an online-thresholded whole channel).
  * Loaded from the pre-stim EXPLODED cache (mixed manual); online-thresholded
    channels have no pre-stimulus data and are skipped.

FORMAT B — "NeuronID" layout (e.g. window_test.xlsx)
    NeuronID | Source | WindowStart | WindowStop
  * ``NeuronID`` = full new-style id ``AMG_2023-09-26_2_Channel.C_003_Unit 1``.
  * ``WindowStart``/``WindowStop`` in ms; a NeuronID may appear on several rows
    (multiple windows).
  * ``Source`` picks the cache: "SI sorted" -> sorted_spike_cache_pre1000ms,
    "mixed"/"manual" -> exploded_spike_cache_pre1000ms,
    "mua" -> threshold_mua_spike_cache_pre1000ms.

Matching to the pre-stim caches
-------------------------------
Region is NOT used as an identifier: within one (Date, Round No.) a given
``Channel.C_XXX[_Unit N]`` is unique. Format A matches the exploded cache's
``Channel`` column to ``Cell``; Format B matches the SI cache's ``NeuronID`` with
the region prefix stripped (so ``Unknown_...`` in the sheet still resolves to the
cache's ``AMG_...``). The real, region-carrying NeuronID is read back from the
matched rows for display.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .keys import make_cell_key

Window = Tuple[float, float]

DEFAULT_CACHE_SUBDIR = "exploded_spike_cache_pre1000ms"   # Format A default
DEFAULT_SOURCE_KIND = "mixed_prestim"

# Format-B Source label -> (source_kind, pre-stim cache subdir)
_SOURCE_MAP = {
    "si sorted": ("si_prestim", "sorted_spike_cache_pre1000ms"),
    "si": ("si_prestim", "sorted_spike_cache_pre1000ms"),
    "sorted": ("si_prestim", "sorted_spike_cache_pre1000ms"),
    "mixed": ("mixed_prestim", "exploded_spike_cache_pre1000ms"),
    "manual": ("mixed_prestim", "exploded_spike_cache_pre1000ms"),
    "mua": ("mua_prestim", "threshold_mua_spike_cache_pre1000ms"),
    "threshold mua": ("mua_prestim", "threshold_mua_spike_cache_pre1000ms"),
    "threshold_mua": ("mua_prestim", "threshold_mua_spike_cache_pre1000ms"),
}
_DEFAULT_B_SOURCE = ("si_prestim", "sorted_spike_cache_pre1000ms")


def _parse_window_ms(s) -> Window:
    """``"(0.0, 300.0)"`` (ms) -> ``(0.0, 0.3)`` (s). Accepts tuples too."""
    if isinstance(s, (tuple, list)):
        a, b = float(s[0]), float(s[1])
    else:
        a, b = (float(x) for x in str(s).strip().strip("()").split(","))
    return (a / 1000.0, b / 1000.0)


def is_manual_unit(cell: str) -> bool:
    """True for a sorted unit (has ``_Unit``) — the ones with a pre-stim window."""
    return "_Unit" in str(cell)


def regionless(neuron_id: str) -> str:
    """Strip the leading ``{Region}_`` so region never has to match.

    ``AMG_2023-09-26_2_Channel.C_003_Unit 1`` -> ``2023-09-26_2_Channel.C_003_Unit 1``.
    """
    parts = str(neuron_id).split("_", 1)
    return parts[1] if len(parts) == 2 else str(neuron_id)


def parse_neuron_id(neuron_id: str) -> Tuple[str, str, int, str]:
    """``AMG_2023-09-26_2_Channel.C_003_Unit 1`` ->
    ``("AMG", "2023-09-26", 2, "Channel.C_003_Unit 1")``."""
    parts = str(neuron_id).split("_", 3)
    if len(parts) < 4:
        raise ValueError(f"unexpected NeuronID format: {neuron_id!r}")
    return parts[0], parts[1], int(parts[2]), parts[3]


def _fmt_ms(windows: List[Window]) -> str:
    return "; ".join(f"{int(a * 1000)}-{int(b * 1000)}" for a, b in windows)


# --------------------------------------------------------------------------- #
def load_answer_key(
    xlsx_path: str,
    *,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    pre_stim: float = 1.0,
    source_kind: str = DEFAULT_SOURCE_KIND,
    skip_unsorted: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[Window]], List[Tuple[str, str]]]:
    """Parse an answer key (either layout) into (candidates_df, truth_by_cell, skipped)."""
    df = pd.read_excel(xlsx_path)
    cols = set(df.columns)
    if {"NeuronID", "WindowStart", "WindowStop"} <= cols:
        return _load_neuronid_format(df, pre_stim=pre_stim)
    if {"Cell", "Time Window"} <= cols:
        return _load_cell_format(df, cache_subdir=cache_subdir, pre_stim=pre_stim,
                                 source_kind=source_kind, skip_unsorted=skip_unsorted)
    raise ValueError(
        f"unrecognized answer-key columns {sorted(cols)}; expected either "
        f"(NeuronID, WindowStart, WindowStop) or (Cell, Time Window, Date, Round No.)")


def _load_cell_format(df, *, cache_subdir, pre_stim, source_kind, skip_unsorted):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Round No."] = df["Round No."].astype(int)
    df["Cell"] = df["Cell"].astype(str).str.strip()

    rows, truth, skipped = [], {}, []
    for (date, rnd, cell), g in df.groupby(["Date", "Round No.", "Cell"]):
        windows = [_parse_window_ms(w) for w in g["Time Window"]]
        cell_key = make_cell_key(source_kind, date, rnd, cell)
        if skip_unsorted and not is_manual_unit(cell):
            skipped.append((cell_key, "online-thresholded: no pre-stimulus data"))
            continue
        p = float(g["P Value"].min()) if "P Value" in g.columns else float("nan")
        rows.append({
            "cell_key": cell_key, "Source": source_kind, "NeuronID": cell,
            "Date": date, "Round No.": int(rnd), "Region": "?",
            "UnitType": "manual_SU", "AnswerWindow_ms": _fmt_ms(windows), "AnswerP": p,
            "_source_key": source_kind, "_match_column": "Channel", "_match_value": cell,
            "_cache_subdir": cache_subdir, "_pre_stim": pre_stim,
        })
        truth[cell_key] = windows
    return _finish(rows, truth, skipped)


def _load_neuronid_format(df, *, pre_stim):
    df = df.copy()
    df["NeuronID"] = df["NeuronID"].astype(str).str.strip()
    has_source = "Source" in df.columns

    rows, truth, skipped = [], {}, []
    for nid, g in df.groupby("NeuronID"):
        windows = [(_ws / 1000.0, _we / 1000.0)
                   for _ws, _we in zip(g["WindowStart"], g["WindowStop"])]
        src_label = (str(g["Source"].iloc[0]).strip().lower() if has_source else "si sorted")
        source_kind, cache_subdir = _SOURCE_MAP.get(src_label, _DEFAULT_B_SOURCE)
        region, date, rnd, channel = parse_neuron_id(nid)
        cell_key = make_cell_key(source_kind, date, rnd, channel)
        rows.append({
            "cell_key": cell_key, "Source": source_kind, "NeuronID": nid,
            "Date": date, "Round No.": int(rnd), "Region": region,
            "UnitType": "SI_SU", "AnswerWindow_ms": _fmt_ms(windows), "AnswerP": float("nan"),
            "_source_key": source_kind, "_match_column": "NeuronID_regionless",
            "_match_value": regionless(nid), "_cache_subdir": cache_subdir, "_pre_stim": pre_stim,
        })
        truth[cell_key] = windows
    return _finish(rows, truth, skipped)


def _finish(rows, truth, skipped):
    candidates = pd.DataFrame(rows)
    nsess = candidates.groupby(["Date", "Round No."]).ngroups if not candidates.empty else 0
    print(f"[answer-key] {len(candidates)} usable cell(s), {len(skipped)} skipped "
          f"(no pre-stim). Sessions: {nsess}")
    for ck, why in skipped:
        print(f"    [skip] {ck}  — {why}")
    return candidates, truth, skipped
