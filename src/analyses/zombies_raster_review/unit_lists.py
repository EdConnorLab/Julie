"""
unit_lists.py — turn the two curated unit lists into a common request format.

Two spreadsheets drive this side investigation, one per spike source:

* **MixedManualSpikeSource** — an Excel sheet (columns ``Date``, ``Round No.``,
  ``Time Window``, ``Cell``, ``P Value``). ``Cell`` is the ``str(Channel)`` of a
  unit, e.g. ``"Channel.C_027_Unit 1"`` (manually sorted) or ``"Channel.C_020"``
  (an unsorted whole-channel). ``Time Window`` is a ``"(start_ms, end_ms)"``
  string.

* **SISortedSpikeSource** — a CSV (columns ``NeuronID``, ``WindowStart_ms``,
  ``WindowEnd_ms``, ``F-statistic``, ``p-value``, ``Date``, ``Round No.``).
  ``NeuronID`` is the full id, e.g.
  ``"AMG_2023-09-26_2_Channel.C_018_Unit 1"``.

Both collapse to a list of :class:`RasterRequest`, which
``run_zombies_rasters.py`` groups by session and resolves against the loaded
spike-source DataFrames. The ANOVA-significant window travels with each request
so the raster can highlight *why* the unit was selected.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RasterRequest:
    """One unit to plot, with the response window that made it significant."""
    source_kind: str                       # "mixed" | "si"
    date: str                              # "YYYY-MM-DD"
    round_no: int
    # how to find the unit's rows in the loaded session DataFrame:
    match_column: str                      # "Channel" (mixed) or "NeuronID" (si)
    match_value: str                       # e.g. "Channel.C_027_Unit 1" / full NeuronID
    label: str                             # human-readable label for the figure/file
    window_ms: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None

    @property
    def window_s(self) -> Optional[Tuple[float, float]]:
        if self.window_ms is None:
            return None
        return (self.window_ms[0] / 1000.0, self.window_ms[1] / 1000.0)


def _parse_window(text) -> Optional[Tuple[float, float]]:
    """Parse a ``"(0.0, 300.0)"`` window string into ``(0.0, 300.0)``."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    try:
        lo, hi = ast.literal_eval(str(text))
        return (float(lo), float(hi))
    except (ValueError, SyntaxError, TypeError):
        return None


def load_mixed_manual_requests(xlsx_path: str) -> List[RasterRequest]:
    """Parse the MixedManualSpikeSource Excel list into requests."""
    df = pd.read_excel(xlsx_path)
    requests: List[RasterRequest] = []
    for _, row in df.iterrows():
        date = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        round_no = int(row["Round No."])
        cell = str(row["Cell"]).strip()
        window = _parse_window(row.get("Time Window"))
        pval = row.get("P Value")
        requests.append(RasterRequest(
            source_kind="mixed",
            date=date,
            round_no=round_no,
            match_column="Channel",
            match_value=cell,
            label=f"{date}_round{round_no}_{cell}",
            window_ms=window,
            p_value=float(pval) if pd.notna(pval) else None,
        ))
    return requests


def load_si_sorted_requests(csv_path: str) -> List[RasterRequest]:
    """Parse the SISortedSpikeSource CSV list into requests.

    A neuron may appear on several rows (multiple significant windows); each row
    becomes its own request so every window gets a raster. Adjust downstream if
    you prefer one raster per neuron.
    """
    df = pd.read_csv(csv_path)
    requests: List[RasterRequest] = []
    for _, row in df.iterrows():
        neuron_id = str(row["NeuronID"]).strip()
        date = str(row["Date"]).strip()
        round_no = int(row["Round No."])
        window = (float(row["WindowStart_ms"]), float(row["WindowEnd_ms"]))
        pval = row.get("p-value")
        requests.append(RasterRequest(
            source_kind="si",
            date=date,
            round_no=round_no,
            match_column="NeuronID",
            match_value=neuron_id,
            label=neuron_id,
            window_ms=window,
            p_value=float(pval) if pd.notna(pval) else None,
        ))
    return requests


def dedup_by_unit(requests: List[RasterRequest]) -> List[RasterRequest]:
    """Keep one request per (source, date, round, unit) — the first window seen.

    Useful when you want one raster per unit rather than one per window.
    """
    seen = set()
    out = []
    for r in requests:
        key = (r.source_kind, r.date, r.round_no, r.match_value)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
