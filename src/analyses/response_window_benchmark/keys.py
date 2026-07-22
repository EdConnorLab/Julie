"""
keys.py — canonical cell identity + metadata inference, shared across selection,
ground-truth I/O, and the benchmark runner so the same cell resolves the same way
everywhere.
"""
from __future__ import annotations

import re
from typing import Optional


def make_cell_key(source_key: str, date: str, round_no, match_value: str) -> str:
    """Stable, unique key for one unit in one session under one source list."""
    return f"{source_key}|{date}|{int(round_no)}|{match_value}"


def infer_region(match_value: str, label: str = "") -> str:
    """Brain-region tag from a NeuronID/label prefix, e.g. ``AMG_2023...`` -> AMG.

    Falls back to '?' when the identifier carries no region prefix (e.g. a bare
    ``Channel.C_003`` from the mixed-manual grant list).
    """
    for s in (match_value or "", label or ""):
        m = re.match(r"^([A-Za-z]{2,4})_\d{4}-\d{2}-\d{2}", str(s))
        if m:
            return m.group(1).upper()
    return "?"


def infer_unit_type(match_value: str, source_key: str = "") -> str:
    """SU (sorted unit), MUA (threshold multiunit), or unsorted (whole channel)."""
    mv = str(match_value)
    if "_Unit" in mv:
        return "SU"
    if source_key.startswith("cache_mua"):
        return "MUA"
    return "unsorted"
