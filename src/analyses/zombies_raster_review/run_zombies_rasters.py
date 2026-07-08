"""
run_zombies_rasters.py — batch Zombies rasters from the curated unit lists.

Reads a unit list (Excel for MixedManualSpikeSource, CSV for SISortedSpikeSource),
groups requests by session so each session is loaded once, resolves each unit's
per-trial rows, and renders the improved Zombies raster (see ``zombies_raster.py``)
with its ANOVA-significant window highlighted.

Examples
--------
    cd src

    # manually-sorted + unsorted units (MixedManualSpikeSource)
    python -m analyses.zombies_raster_review.run_zombies_rasters \
        --source mixed \
        --list analyses/zombies_raster_review/unit_lists/zombies_mixed_manual_anova_passed.xlsx

    # SpikeInterface-sorted units (SISortedSpikeSource)
    python -m analyses.zombies_raster_review.run_zombies_rasters \
        --source si \
        --list analyses/zombies_raster_review/unit_lists/zombies_si_sorted_anova_passed.csv

    # no data on hand — render a fabricated unit to preview the format
    python -m analyses.zombies_raster_review.run_zombies_rasters --demo

Defaults for ``--list`` point at the copies bundled in ``unit_lists/``.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from .unit_lists import (
    RasterRequest, load_mixed_manual_requests, load_si_sorted_requests,
)
from .zombies_raster import plot_zombies_raster

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LISTS = {
    "mixed": os.path.join(_HERE, "unit_lists", "zombies_mixed_manual_anova_passed.xlsx"),
    "si": os.path.join(_HERE, "unit_lists", "zombies_si_sorted_anova_passed.csv"),
}


def _make_source(source_kind: str):
    """Build the SpikeSource lazily (imports pull in DB/IO-heavy deps)."""
    from data_access.spike_source import MixedManualSpikeSource, SISortedSpikeSource
    return MixedManualSpikeSource() if source_kind == "mixed" else SISortedSpikeSource()


def _select_unit_rows(session_df: pd.DataFrame, req: RasterRequest) -> pd.DataFrame:
    """Rows of ``session_df`` belonging to the requested unit.

    Mixed matches on ``str(Channel) == match_value``; SI matches on ``NeuronID``.
    Falls back to a ``NeuronID`` suffix match so minor formatting drift (enum
    repr vs string) still resolves.
    """
    if req.match_column == "Channel":
        chan = session_df["Channel"].astype(str)
        rows = session_df[chan == req.match_value]
        if rows.empty and "NeuronID" in session_df.columns:
            rows = session_df[session_df["NeuronID"].astype(str).str.endswith(req.match_value)]
        return rows
    rows = session_df[session_df["NeuronID"].astype(str) == req.match_value]
    return rows


def run_from_requests(
    requests: List[RasterRequest],
    out_dir: str,
    *,
    xlim: float = 2.0,
    psth_bin_ms: float = 50.0,
) -> Dict[str, int]:
    """Load each needed session once, then render every requested unit."""
    os.makedirs(out_dir, exist_ok=True)
    # group by (source, date, round) so each session loads a single time
    by_session: Dict[tuple, List[RasterRequest]] = defaultdict(list)
    for r in requests:
        by_session[(r.source_kind, r.date, r.round_no)].append(r)

    stats = {"plotted": 0, "skipped_no_rows": 0, "skipped_no_trials": 0, "sessions_failed": 0}
    for (source_kind, date, round_no), reqs in sorted(by_session.items()):
        source = _make_source(source_kind)
        try:
            session_df = source.load(date, round_no)
        except Exception as e:  # missing files / DB unavailable
            print(f"[session {date} r{round_no} ({source_kind})] load failed: {e}")
            stats["sessions_failed"] += 1
            continue
        if session_df is None or session_df.empty:
            print(f"[session {date} r{round_no} ({source_kind})] no data")
            stats["sessions_failed"] += 1
            continue

        for req in reqs:
            rows = _select_unit_rows(session_df, req)
            if rows.empty:
                print(f"[{req.label}] no matching rows in session")
                stats["skipped_no_rows"] += 1
                continue
            safe = req.label.replace(" ", "").replace("/", "-").replace(".", "")
            save_path = os.path.join(out_dir, f"{source_kind}_{safe}.png")
            fig = plot_zombies_raster(
                rows, neuron_label=req.label, window_s=req.window_s,
                p_value=req.p_value, xlim=xlim, psth_bin_ms=psth_bin_ms,
                save_path=save_path,
            )
            if fig is None:
                stats["skipped_no_trials"] += 1
            else:
                stats["plotted"] += 1

    print(f"\nDone: {stats}")
    return stats


def run_from_list(source_kind: str, list_path: str, out_dir: str, **kw) -> Dict[str, int]:
    if source_kind == "mixed":
        requests = load_mixed_manual_requests(list_path)
    elif source_kind == "si":
        requests = load_si_sorted_requests(list_path)
    else:
        raise ValueError(f"unknown source '{source_kind}' (use 'mixed' or 'si')")
    print(f"Loaded {len(requests)} unit request(s) from {os.path.basename(list_path)}")
    return run_from_requests(requests, out_dir, **kw)


def _run_demo(out_dir: str):
    from .make_synthetic_zombies_session import make_synthetic_zombies_unit
    os.makedirs(out_dir, exist_ok=True)
    df = make_synthetic_zombies_unit()
    plot_zombies_raster(
        df,
        neuron_label="DEMO — AMG_2023-09-26_2_Channel.C_018_Unit 1",
        window_s=(0.1, 0.45), p_value=0.004,
        save_path=os.path.join(out_dir, "demo_zombies_raster.png"),
    )
    print(f"Demo raster written to {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mixed", "si"], help="which spike source")
    p.add_argument("--list", dest="list_path", help="unit list (Excel for mixed, CSV for si)")
    p.add_argument("--out", help="output directory for figures")
    p.add_argument("--xlim", type=float, default=2.0, help="x-axis limit in seconds")
    p.add_argument("--psth-bin-ms", type=float, default=50.0)
    p.add_argument("--demo", action="store_true", help="render a fabricated unit")
    args = p.parse_args(argv)

    if args.demo:
        _run_demo(args.out or os.path.join(os.getcwd(), "zombies_raster_demo_out"))
        return

    if not args.source:
        p.error("--source is required (or use --demo)")
    list_path = args.list_path or DEFAULT_LISTS[args.source]
    out_dir = args.out or os.path.join(os.getcwd(), f"zombies_rasters_{args.source}")
    run_from_list(args.source, list_path, out_dir, xlim=args.xlim, psth_bin_ms=args.psth_bin_ms)


if __name__ == "__main__":
    main()
