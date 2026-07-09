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

    # overlay: coincidence-match the two sorts per session and overlay each neuron
    python -m analyses.zombies_raster_review.run_zombies_rasters --overlay

    # no data on hand — render a fabricated unit to preview the format
    python -m analyses.zombies_raster_review.run_zombies_rasters --demo

Defaults for ``--list`` point at the copies bundled in ``unit_lists/``.
Overlay matching is by spike-time coincidence (SI and the manual sort can name
the same neuron on different channels); see ``coincidence_match.py``.
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
from analyses.zombies_raster_review.coincidence_match import (
    session_units, match_units_across_sources, group_matches, MatchGroup,
    DEFAULT_COINCIDENCE_THRESHOLD, DEFAULT_RATIO_THRESHOLD, DEFAULT_WINDOW_MS,
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
    xlim: float = 2.4,
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
# Overlay mode — match the two sorts by spike-time coincidence, then overlay
# --------------------------------------------------------------------------- #
# SpikeInterface names a neuron after its strongest channel, but the manual sort
# may have put that same neuron on a *different* channel — so the two sorts can't
# be paired by channel name. Instead we pair units whose spike trains are
# coincident (same neuron fires on both records within a fraction of a ms), the
# logic borrowed from ``spikesorting.cross_channel_analysis``. See
# ``coincidence_match.py``.


def _channel_token(channel: str) -> str:
    """Normalise a channel string to a ``C_018``-style token (utility)."""
    tok = str(channel).replace("Channel.", "").replace("-", "_").strip()
    return tok.split("_Unit")[0]


def _units_on_channel(session_df, base_channel: str) -> Dict[str, "pd.DataFrame"]:
    """Split a session DataFrame into per-unit frames on one base channel.

    Keyed by a short unit label (the tail of NeuronID, or the Channel string).
    A convenience for inspecting one channel; overlay matching no longer relies
    on it (see the coincidence-based flow below).
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


def _short_mixed_label(cell_id: str) -> str:
    """``Channel.C_011_Unit 1`` → ``manual: C_011_Unit 1`` for the overlay lane."""
    return "manual: " + str(cell_id).split("Channel.")[-1]


def _short_si_label(neuron_id: str) -> str:
    """``AMG_..._Channel.C_020_Unit 1`` → ``SI: C_020_Unit 1``."""
    return "SI: " + str(neuron_id).split("Channel.")[-1]


def _cell_token(uid: str) -> str:
    """``AMG_..._Channel.C_020_Unit 1`` / ``Channel.C_011_Unit 1`` → ``C_020_Unit 1``."""
    return str(uid).split("Channel.")[-1]


def _as_window_list(value) -> List[tuple]:
    """Normalise an anchor-window value to a list of ``(lo_s, hi_s)`` windows.

    Accepts ``None`` (→ ``[]``), a single ``(lo, hi)`` tuple, or a list of them —
    a cell can carry several significant windows (it appears on several list rows).
    """
    if value is None:
        return []
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(isinstance(x, (int, float)) for x in value)):
        return [(float(value[0]), float(value[1]))]
    return [(float(w[0]), float(w[1])) for w in value if w is not None]


def _group_windows(group: MatchGroup, mixed_windows: dict, si_windows: dict) -> List[tuple]:
    """Every ANOVA window to shade for a group, tagged by source list and cell.

    Returns ``(lo_s, hi_s, source, cell_label)`` for each window of each *listed*
    cell in the group — so a mixed-list cell and an SI-list cell (and cells with
    multiple windows) all get marked, each showing which list it came from.
    """
    specs: List[tuple] = []
    for cid in group.mixed_ids:
        for lo, hi in _as_window_list(mixed_windows.get(cid)):
            specs.append((lo, hi, "mixed", _cell_token(cid)))
    for sid in group.si_ids:
        for lo, hi in _as_window_list(si_windows.get(sid)):
            specs.append((lo, hi, "SI", _cell_token(sid)))
    return specs


def render_overlays_for_units(
    units_mixed: Dict[str, "pd.DataFrame"],
    units_si: Dict[str, "pd.DataFrame"],
    out_dir: str,
    *,
    label: str,
    mixed_anchor_windows: Optional[dict] = None,
    si_anchor_windows: Optional[dict] = None,
    listed_only: bool = False,
    window_ms: float = DEFAULT_WINDOW_MS,
    coincidence_threshold: float = DEFAULT_COINCIDENCE_THRESHOLD,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
) -> List[str]:
    """Match two sorts' units by coincidence and render one overlay per group.

    DB-free core (takes already-loaded per-unit frames). ``*_anchor_windows`` map
    an ANOVA-passed unit id → its significant window (s); when either is given,
    only groups containing at least one anchor are drawn, and the anchor's window
    is shaded. Returns the saved figure paths.

    ``listed_only`` restricts matching to the *listed* cells on both sides (the
    keys of the anchor-window dicts), so every unit shown is from your xlsx/csv.
    With it off (default), a listed cell is matched against *all* units in the
    other sort, so its cross-sort twin is found even if that twin isn't listed.
    """
    os.makedirs(out_dir, exist_ok=True)
    mixed_anchor_windows = mixed_anchor_windows or {}
    si_anchor_windows = si_anchor_windows or {}
    anchored_only = bool(mixed_anchor_windows or si_anchor_windows)

    if listed_only:
        # keep only cells that appear in the lists, on both sides
        units_mixed = {k: v for k, v in units_mixed.items() if k in mixed_anchor_windows}
        units_si = {k: v for k, v in units_si.items() if k in si_anchor_windows}

    matches = match_units_across_sources(
        units_mixed, units_si, window_ms=window_ms,
        coincidence_threshold=coincidence_threshold, ratio_threshold=ratio_threshold,
    )
    groups = group_matches(matches)

    written: List[str] = []
    for i, g in enumerate(groups, start=1):
        is_anchored = (any(cid in mixed_anchor_windows for cid in g.mixed_ids) or
                       any(sid in si_anchor_windows for sid in g.si_ids))
        if anchored_only and not is_anchored:
            continue

        unit_dfs: Dict[str, "pd.DataFrame"] = {}
        for cid in g.mixed_ids:
            if cid in units_mixed:
                unit_dfs[_short_mixed_label(cid)] = units_mixed[cid]
        for sid in g.si_ids:
            if sid in units_si:
                unit_dfs[_short_si_label(sid)] = units_si[sid]
        if len(unit_dfs) < 2:
            continue  # need at least one unit from each sort to overlay

        windows = _group_windows(g, mixed_anchor_windows, si_anchor_windows)
        # per-pair coincidence, in the overlay's lane-label terms, for the probe map
        pair_coincidences = [(_short_mixed_label(mid), _short_si_label(sid), c)
                             for mid, sid, c in g.pairs]
        # units are named in the legend and the probe map, so keep the title
        # short — some groups have >10 units and listing them all is unreadable.
        # "best" coincidence here; the probe map shows every pair's value.
        n_mixed, n_si = len(g.mixed_ids), len(g.si_ids)
        title = (f"Sort match #{i} — {label}   ·   "
                 f"{n_mixed} manual + {n_si} SI units   ·   "
                 f"best coincidence {g.best_coincidence:.2f}")
        save_path = os.path.join(out_dir, f"overlay_{label}_group{i:02d}.png")
        fig = plot_overlay_raster(
            unit_dfs, title=title, windows=windows, pair_coincidences=pair_coincidences,
            xlim=xlim, psth_bin_ms=psth_bin_ms, save_path=save_path,
        )
        if fig is not None:
            written.append(save_path)

    print(f"[overlay {label}] {len(matches)} cross-sort match(es) → "
          f"{len(groups)} group(s) → {len(written)} overlay(s) written")
    return written


def run_overlay_by_coincidence(
    date: str, round_no: int, out_dir: str,
    *,
    mixed_anchor_windows: Optional[dict] = None,
    si_anchor_windows: Optional[dict] = None,
    listed_only: bool = False,
    window_ms: float = DEFAULT_WINDOW_MS,
    coincidence_threshold: float = DEFAULT_COINCIDENCE_THRESHOLD,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
) -> List[str]:
    """Load both sorts for one session and overlay coincidence-matched units."""
    os.makedirs(out_dir, exist_ok=True)
    loaded = {}
    for source_kind in ("mixed", "si"):
        source = _make_source(source_kind)
        try:
            loaded[source_kind] = source.load(date, round_no)
        except Exception as e:
            print(f"[overlay {date} r{round_no}] {source_kind} load failed: {e}")
            loaded[source_kind] = None

    missing = [k for k, v in loaded.items() if v is None or getattr(v, "empty", True)]
    if missing:
        print(f"[overlay {date} r{round_no}] need both sorts; missing/empty: {missing}")
        return []

    units_mixed = session_units(loaded["mixed"], "Channel")
    units_si = session_units(loaded["si"], "NeuronID")
    return render_overlays_for_units(
        units_mixed, units_si, out_dir, label=f"{date}_round{round_no}",
        mixed_anchor_windows=mixed_anchor_windows, si_anchor_windows=si_anchor_windows,
        listed_only=listed_only,
        window_ms=window_ms, coincidence_threshold=coincidence_threshold,
        ratio_threshold=ratio_threshold, xlim=xlim, psth_bin_ms=psth_bin_ms,
    )


def run_overlay_from_lists(
    mixed_list_path: str, si_list_path: str, out_dir: str,
    *,
    only_date: Optional[str] = None,
    only_round: Optional[int] = None,
    listed_only: bool = False,
    window_ms: float = DEFAULT_WINDOW_MS,
    coincidence_threshold: float = DEFAULT_COINCIDENCE_THRESHOLD,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
) -> List[str]:
    """Overlay coincidence-matched sorts for every ``(date, round)`` in the lists.

    The ANOVA-passed cells in the two lists are the *anchors*: each session's
    overlays are restricted to coincidence groups that contain at least one
    anchor, and the anchor's significant window is shaded. Pass ``only_date`` /
    ``only_round`` to restrict to a single session.

    ``listed_only=True`` shows *only* cells that are in the xlsx/csv lists (both
    sides), for investigating the two lists on their own. Left ``False`` (the
    default) it also brings in each listed cell's cross-sort twin even when that
    twin isn't listed — the fuller "all cells" view.
    """
    mixed_reqs = load_mixed_manual_requests(mixed_list_path)
    si_reqs = load_si_sorted_requests(si_list_path)

    # (date, round) -> {cell_id: [window_s, ...]}. A cell can appear on several
    # list rows with different significant windows, so accumulate them all.
    mixed_windows: Dict[tuple, dict] = defaultdict(dict)
    si_windows: Dict[tuple, dict] = defaultdict(dict)
    for r in mixed_reqs:
        wl = mixed_windows[(r.date, r.round_no)].setdefault(r.match_value, [])
        if r.window_s is not None:
            wl.append(r.window_s)
    for r in si_reqs:
        wl = si_windows[(r.date, r.round_no)].setdefault(r.match_value, [])
        if r.window_s is not None:
            wl.append(r.window_s)

    sessions = sorted(set(mixed_windows) | set(si_windows))
    if only_date is not None:
        sessions = [s for s in sessions
                    if s[0] == only_date and (only_round is None or s[1] == only_round)]
    print(f"Overlay by coincidence over {len(sessions)} session(s)")

    written: List[str] = []
    for date, round_no in sessions:
        written += run_overlay_by_coincidence(
            date, round_no, out_dir,
            mixed_anchor_windows=mixed_windows.get((date, round_no)),
            si_anchor_windows=si_windows.get((date, round_no)),
            listed_only=listed_only,
            window_ms=window_ms, coincidence_threshold=coincidence_threshold,
            ratio_threshold=ratio_threshold, xlim=xlim, psth_bin_ms=psth_bin_ms,
        )
    print(f"\nDone: {len(written)} overlay(s) across {len(sessions)} session(s)")
    return written


# --------------------------------------------------------------------------- #
# Demos (no DB / recordings)
# --------------------------------------------------------------------------- #
def _run_demo(out_dir: str):
    from analyses.zombies_raster_review.make_synthetic_zombies_session import (
        make_synthetic_zombies_unit, make_synthetic_overlay_pair,
        make_synthetic_cross_source_session,
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
    # coincidence-matched overlay: the SAME neuron is on C_011 (mixed) and C_020
    # (SI) — different channels — so it is matched by spike-time coincidence, not
    # by channel name. C_020 is the anchor (as if it were the ANOVA-passed cell).
    mixed_df, si_df = make_synthetic_cross_source_session()
    units_mixed = session_units(mixed_df, "Channel")
    units_si = session_units(si_df, "NeuronID")
    render_overlays_for_units(
        units_mixed, units_si, out_dir, label="DEMO_coincidence",
        si_anchor_windows={"AMG_2023-09-26_2_Channel.C_020_Unit 1": (0.1, 0.45)},
    )
    print(f"Demo rasters written to {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mixed", "si"], help="which spike source")
    p.add_argument("--list", dest="list_path", help="unit list (Excel for mixed, CSV for si)")
    p.add_argument("--out", help="output directory for figures")
    p.add_argument("--xlim", type=float, default=2.4, help="x-axis limit in seconds")
    p.add_argument("--psth-bin-ms", type=float, default=50.0)
    p.add_argument("--demo", action="store_true", help="render a fabricated unit")
    p.add_argument("--overlay", action="store_true",
                   help="coincidence-match the two sorts and overlay (uses both lists)")
    p.add_argument("--listed-only", action="store_true",
                   help="overlay only cells in the xlsx/csv lists (with --overlay)")
    args = p.parse_args(argv)

    if args.demo:
        _run_demo(args.out or os.path.join(os.getcwd(), "zombies_raster_demo_out"))
        return

    if args.overlay:
        out_dir = args.out or os.path.join(os.getcwd(), "zombies_rasters_overlay")
        run_overlay_from_lists(DEFAULT_LISTS["mixed"], DEFAULT_LISTS["si"], out_dir,
                               listed_only=args.listed_only,
                               xlim=args.xlim, psth_bin_ms=args.psth_bin_ms)
        return

    if not args.source:
        p.error("--source is required (or use --demo / --overlay)")
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
    #    "overlay"  for each (date, round) in the lists, match the two sorts by
    #               spike-time COINCIDENCE (not channel name — SI and manual can
    #               put the same neuron on different channels) and overlay each
    #               matched group, matched trial-for-trial. See coincidence_match.
    # ========================================================================
    MODE = "demo"

    # Where figures are written (created if missing). Defaults to an `output/`
    # folder next to this file so they show up right in the PyCharm project tree.
    OUT_DIR = os.path.join(_HERE, "output")

    # -- MODE == "mixed" / "si" --
    LIST_PATH = None            # None → use the bundled list in unit_lists/

    # -- MODE == "overlay" --
    #  By default sweeps every (date, round) in the two lists. Set
    #  OVERLAY_DATE (and optionally OVERLAY_ROUND) to restrict to one session.
    OVERLAY_DATE = None         # e.g. "2023-09-26" to do just one date
    OVERLAY_ROUND = None        # e.g. 2 (only used when OVERLAY_DATE is set)
    #  LISTED_ONLY: True  → show ONLY cells that are in the xlsx/csv lists
    #                       (for investigating the two lists on their own).
    #               False → also bring in each listed cell's cross-sort twin even
    #                       when that twin isn't listed (the fuller "all cells"
    #                       view; useful once you extend beyond the lists).
    LISTED_ONLY = False
    #  Coincidence tuning (cross-sort). A pair is matched when its coincidence is
    #  >= COINCIDENCE_THRESHOLD and >= RATIO_THRESHOLD × chance.
    COINCIDENCE_THRESHOLD = DEFAULT_COINCIDENCE_THRESHOLD
    RATIO_THRESHOLD = DEFAULT_RATIO_THRESHOLD
    COINCIDENCE_WINDOW_MS = DEFAULT_WINDOW_MS

    # -- plot tuning (applies to all modes) --
    XLIM_S = 2.4
    PSTH_BIN_MS = 50.0
    # ========================================================================

    if MODE == "demo":
        _run_demo(os.path.join(OUT_DIR, "demo"))
    elif MODE in ("mixed", "si"):
        list_path = LIST_PATH or DEFAULT_LISTS[MODE]
        run_from_list(MODE, list_path, os.path.join(OUT_DIR, MODE),
                      xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS)
    elif MODE == "overlay":
        run_overlay_from_lists(
            DEFAULT_LISTS["mixed"], DEFAULT_LISTS["si"],
            os.path.join(OUT_DIR, "overlay"),
            only_date=OVERLAY_DATE, only_round=OVERLAY_ROUND,
            listed_only=LISTED_ONLY,
            window_ms=COINCIDENCE_WINDOW_MS,
            coincidence_threshold=COINCIDENCE_THRESHOLD,
            ratio_threshold=RATIO_THRESHOLD,
            xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS,
        )
    else:
        raise ValueError(f"unknown MODE '{MODE}'")
