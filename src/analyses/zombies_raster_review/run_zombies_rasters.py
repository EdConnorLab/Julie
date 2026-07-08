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

from analyses.zombies_raster_review.unit_lists import (
    RasterRequest, load_mixed_manual_requests, load_si_sorted_requests,
)
from analyses.zombies_raster_review.zombies_raster import (
    plot_zombies_raster, plot_overlay_raster,
)

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


# --------------------------------------------------------------------------- #
# Overlay mode — compare the two sorts of one channel
# --------------------------------------------------------------------------- #
def _channel_token(channel: str) -> str:
    """Normalise a channel string to a ``C_018``-style token for matching."""
    tok = str(channel).replace("Channel.", "").replace("-", "_").strip()
    return tok.split("_Unit")[0]


def _units_on_channel(session_df, base_channel: str) -> Dict[str, "pd.DataFrame"]:
    """Split a session DataFrame into per-unit frames on one base channel.

    Keyed by a short unit label (the tail of NeuronID, or the Channel string).
    """
    token = _channel_token(base_channel)
    if "BaseChannel" in session_df.columns:
        mask = session_df["BaseChannel"].astype(str).apply(lambda c: _channel_token(c) == token)
    else:
        mask = session_df["Channel"].astype(str).apply(lambda c: _channel_token(c) == token)
    sub = session_df[mask]
    out: Dict[str, "pd.DataFrame"] = {}
    id_col = "NeuronID" if "NeuronID" in sub.columns else "Channel"
    for uid, udf in sub.groupby(sub[id_col].astype(str)):
        label = str(uid).split("Channel.")[-1]  # trim location/date prefix
        out[label] = udf
    return out


def run_overlay_for_channel(
    date: str, round_no: int, base_channel: str, out_dir: str,
    *, window_ms=None, xlim: float = 2.0, psth_bin_ms: float = 50.0,
) -> Optional[str]:
    """Overlay the manual (mixed) and SI sorts of one channel on a single raster."""
    os.makedirs(out_dir, exist_ok=True)
    unit_dfs: Dict[str, "pd.DataFrame"] = {}
    for source_kind in ("mixed", "si"):
        source = _make_source(source_kind)
        try:
            sdf = source.load(date, round_no)
        except Exception as e:
            print(f"[overlay] {source_kind} load failed for {date} r{round_no}: {e}")
            continue
        if sdf is None or sdf.empty:
            continue
        for label, udf in _units_on_channel(sdf, base_channel).items():
            unit_dfs[f"{source_kind}: {label}"] = udf

    if not unit_dfs:
        print(f"[overlay] no units found on {base_channel} for {date} r{round_no}")
        return None

    window_s = (window_ms[0] / 1000.0, window_ms[1] / 1000.0) if window_ms else None
    tok = _channel_token(base_channel)
    save_path = os.path.join(out_dir, f"overlay_{date}_round{round_no}_{tok}.png")
    plot_overlay_raster(
        unit_dfs, title=f"Sort comparison — {tok} · {date} round {round_no}",
        window_s=window_s, xlim=xlim, psth_bin_ms=psth_bin_ms, save_path=save_path,
    )
    return save_path


# --------------------------------------------------------------------------- #
# Demos (no DB / recordings)
# --------------------------------------------------------------------------- #
def _run_demo(out_dir: str):
    from analyses.zombies_raster_review.make_synthetic_zombies_session import (
        make_synthetic_zombies_unit, make_synthetic_overlay_pair,
    )
    os.makedirs(out_dir, exist_ok=True)
    # single-unit raster (with a few subject 81G trials that must be excluded)
    df = make_synthetic_zombies_unit(include_subject=True)
    plot_zombies_raster(
        df,
        neuron_label="DEMO — AMG_2023-09-26_2_Channel.C_018_Unit 1",
        window_s=(0.1, 0.45), p_value=0.004,
        save_path=os.path.join(out_dir, "demo_single_unit.png"),
    )
    # overlay of two sorts of the same channel
    pair = make_synthetic_overlay_pair()
    plot_overlay_raster(
        pair, title="DEMO overlay — manual vs SI sort of C-018",
        window_s=(0.1, 0.45),
        save_path=os.path.join(out_dir, "demo_overlay.png"),
    )
    print(f"Demo rasters written to {out_dir}")


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
    # ========================================================================
    #  RUN CONFIG — edit these, then just press ▶ Run in PyCharm.
    #  (No terminal or command-line arguments needed.)
    #
    #  MODE options:
    #    "demo"     fabricated data — no DB / recordings needed. Writes a
    #               single-unit raster and an overlay raster so you can see
    #               both formats immediately.
    #    "mixed"    one raster per unit in the mixed-manual Excel list
    #               (MixedManualSpikeSource: manually sorted + unsorted).
    #    "si"       one raster per unit in the SI-sorted CSV list
    #               (SISortedSpikeSource).
    #    "overlay"  overlay the two sorts (manual vs SI) of ONE channel in one
    #               session, matched trial-for-trial — for cross-checking sorts.
    # ========================================================================
    MODE = "demo"

    # Where figures are written (created if missing). Defaults to an `output/`
    # folder next to this file so they show up right in the PyCharm project tree.
    OUT_DIR = os.path.join(_HERE, "output")

    # -- MODE == "mixed" / "si" --
    LIST_PATH = None            # None → use the bundled list in unit_lists/

    # -- MODE == "overlay" --
    OVERLAY_DATE = "2023-09-26"
    OVERLAY_ROUND = 2
    OVERLAY_CHANNEL = "C_018"   # base channel; both sorts of it are overlaid
    OVERLAY_WINDOW_MS = None    # e.g. (100, 450) to shade a response window

    # -- plot tuning (applies to all modes) --
    XLIM_S = 2.0
    PSTH_BIN_MS = 50.0
    # ========================================================================

    if MODE == "demo":
        _run_demo(os.path.join(OUT_DIR, "demo"))
    elif MODE in ("mixed", "si"):
        list_path = LIST_PATH or DEFAULT_LISTS[MODE]
        run_from_list(MODE, list_path, os.path.join(OUT_DIR, MODE),
                      xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS)
    elif MODE == "overlay":
        run_overlay_for_channel(
            OVERLAY_DATE, OVERLAY_ROUND, OVERLAY_CHANNEL,
            os.path.join(OUT_DIR, "overlay"),
            window_ms=OVERLAY_WINDOW_MS, xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS,
        )
    else:
        raise ValueError(f"unknown MODE '{MODE}'")
