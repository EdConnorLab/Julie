"""
ground_truth.py — the hand-annotation template and its loader.

The windows already attached to the grant/cache cell lists were produced by the
CURRENT detector and are NOT treated as truth. Instead the user inspects each
cell's raster and writes the window(s) they actually see into an Excel template
this module generates, and the loader reads them back for scoring.

Template columns
----------------
``cell_key`` (identity — do not edit), plus human-readable ``Source``,
``NeuronID``, ``Date``, ``Round No.``, ``Region``, ``UnitType`` and a
``ListWindow_ms`` reference column (the OLD detector's window, shown only so you
can see what it did — it is not the answer). Then, to fill in:

    TrueWindow1_start_ms, TrueWindow1_end_ms,
    TrueWindow2_start_ms, TrueWindow2_end_ms, ... (up to ``max_windows``)
    ResponseType   free text (e.g. transient / sustained / multi-peak / suppression)
    NoResponse     put any non-empty value (e.g. "y") if the cell has NO response
    Notes

Leave the TrueWindow cells blank for windows you don't use. A row with
``NoResponse`` set (and no windows) is scored as a genuine no-response cell.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

Window = Tuple[float, float]

META_COLS = ["cell_key", "Source", "NeuronID", "Date", "Round No.",
             "Region", "UnitType", "ListWindow_ms"]


def template_columns(max_windows: int = 3) -> List[str]:
    win_cols = []
    for i in range(1, max_windows + 1):
        win_cols += [f"TrueWindow{i}_start_ms", f"TrueWindow{i}_end_ms"]
    return META_COLS + win_cols + ["ResponseType", "NoResponse", "Notes"]


def write_template(cells: pd.DataFrame, path: str, *, max_windows: int = 3) -> str:
    """Write a blank annotation template. ``cells`` must contain ``META_COLS``
    (cell_selection.build_candidates produces exactly these)."""
    import os
    cols = template_columns(max_windows)
    out = pd.DataFrame(columns=cols)
    for c in META_COLS:
        out[c] = cells[c].values if c in cells.columns else ""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_excel(path, index=False)
    print(f"[ground-truth] template ({len(out)} cells) -> {path}")
    return path


def load_truth(path: str, *, max_windows: int = 8) -> Dict[str, List[Window]]:
    """Read a filled template into ``{cell_key: [(start_s, end_s), ...]}``.

    A row flagged ``NoResponse`` (and with no windows) maps to ``[]`` and is
    scored as a real no-response cell. Rows with neither windows nor a
    NoResponse flag are treated as UNannotated and skipped (not scored).
    """
    df = pd.read_excel(path)
    truth: Dict[str, List[Window]] = {}
    for _, r in df.iterrows():
        key = str(r["cell_key"])
        windows: List[Window] = []
        for i in range(1, max_windows + 1):
            sc, ec = f"TrueWindow{i}_start_ms", f"TrueWindow{i}_end_ms"
            if sc in df.columns and ec in df.columns:
                s, e = r.get(sc), r.get(ec)
                if pd.notna(s) and pd.notna(e):
                    windows.append((float(s) / 1000.0, float(e) / 1000.0))
        no_resp = "NoResponse" in df.columns and pd.notna(r.get("NoResponse")) \
            and str(r.get("NoResponse")).strip() != ""
        if windows:
            truth[key] = windows
        elif no_resp:
            truth[key] = []
        # else: unannotated -> skip
    print(f"[ground-truth] loaded {len(truth)} annotated cell(s) from {path}")
    return truth
