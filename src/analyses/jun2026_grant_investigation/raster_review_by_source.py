"""
raster_review_by_source.py — raster + PSTH + waveform-footprint plots for every
DATA_SOURCE of ``replicate_analysis``, plus cross-source overlays.

``replicate_analysis.py`` was run five ways via its ``DATA_SOURCE`` switch:

    grant_xlsx | cache_kw | cache_anova | cache_mua_kw | cache_mua_anova

Each value selects a different *cell list* and the spike source behind it. This
module bridges those five lists into the plotting engine of
``analyses.zombies_raster_review`` so you can see the units, not just the numbers.

Two products
------------
* **Singles** (``run_singles``) — for every unit in a source's list, one figure:
  raster + PSTH (left) and probe map + cross-channel waveform footprint (right),
  rendered by :func:`zombies_raster.plot_overlay_raster` used single-lane. One
  figure per **unit** (windows deduped; the unit's significant window is shaded).
* **Overlays** (``run_overlay``) — for a pair of sources, coincidence-match the
  two sorts of each neuron per session and overlay them on one figure (raster +
  PSTH + probe + footprint). Drives the three requested pairs:
  ``grant_xlsx`` vs ``cache_kw`` / ``cache_anova`` / ``cache_mua_anova``.

How each DATA_SOURCE maps to plottable data
-------------------------------------------
Each list only names the *cells*; the actual per-trial spike TIMES needed for
rasters come from that source's spike cache (the lists carry spike counts, not
times, so they cannot be plotted directly).

* **grant_xlsx** — cells from Ed's grant xlsx (``common.HIS_XLSX``), which is in
  the mixed-manual format (``Date``, ``Round No.``, ``Time Window``,
  ``Cell`` = ``str(Channel)``); spike times looked up per cell in the
  **exploded_spike_cache** via ``MixedManualSpikeSource``. First ``GRANT_NCELLS``
  rows, to match what ``replicate_analysis`` (``NCELLS=74``) regresses.
* **cache_kw / cache_anova** — cells from the
  ``si_sorted_..._significant_windows_p{KW,ANOVA}_passed.pkl`` files; spike times
  from **sorted_spike_cache_filtered** via
  ``SISortedSpikeSource(cache_subdir="sorted_spike_cache_filtered", pre_filtered)``
  (``spike_count_connector._si_sorted_source``).
* **cache_mua_kw / cache_mua_anova** — cells from the
  ``threshold_mua_..._significant_windows_p{KW,ANOVA}_passed.pkl`` files; spike
  times from **threshold_mua_spike_cache** via
  ``ThresholdMUASpikeSource(mad, 4.0, ref 1.0)``.

The significance-pkl paths and the Date/Round normalisation come from
``spike_count_connector`` (``LISTS`` / ``_windows_for``) — the same window lists
``replicate_analysis`` consumes — so the plotted cells are the analysed cells.
(``replicate_analysis`` additionally drops (neuron, window) combos missing any of
the 9 stimulus monkeys — a regression-input requirement — so a few plotted cells
may not have entered a given fit.)

Waveform footprints need the raw recording (``amplifier.dat`` / info.rhd +
``windowsort``); where that is absent the footprint panel simply reads
"waveforms unavailable" and the raster/PSTH/probe still render.

Run it
------
Edit the RUN CONFIG block at the bottom and press Run in PyCharm, or::

    cd src
    python -m analyses.jun2026_grant_investigation.raster_review_by_source \
        --mode singles --source cache_kw
    python -m analyses.jun2026_grant_investigation.raster_review_by_source \
        --mode overlay --pair grant_xlsx cache_mua_anova
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# --- make the repo's src/ importable when run as a plain script (PyCharm ▶) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))            # .../Julie/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_access.spike_source import MixedManualSpikeSource                    # noqa: E402
from analyses.zombies_raster_review.zombies_raster import plot_overlay_raster  # noqa: E402
from analyses.zombies_raster_review.run_zombies_rasters import (               # noqa: E402
    _select_unit_rows, _cell_token, _lane_label, _unit_location,
    render_overlays_for_units,
)
from analyses.zombies_raster_review.unit_lists import (                        # noqa: E402
    RasterRequest, load_mixed_manual_requests, dedup_by_unit,
)
from analyses.zombies_raster_review.coincidence_match import (                 # noqa: E402
    session_units, DEFAULT_COINCIDENCE_THRESHOLD, DEFAULT_RATIO_THRESHOLD,
    DEFAULT_WINDOW_MS,
)
from analyses.zombies_raster_review.waveform_footprint import extract_footprints  # noqa: E402
from analyses.jun2026_grant_investigation import common                        # noqa: E402
from analyses.jun2026_grant_investigation import spike_count_connector as scc  # noqa: E402

# First N rows of the grant xlsx = the cells replicate_analysis regresses
# (its NCELLS=74, matching Ed's script). Kept local to avoid importing the
# script module (which runs a __main__ guard).
GRANT_NCELLS = 74


# --------------------------------------------------------------------------- #
# Per-DATA_SOURCE request loaders
# --------------------------------------------------------------------------- #
def _grant_requests() -> List[RasterRequest]:
    """grant_xlsx cells as mixed-manual requests (first GRANT_NCELLS rows)."""
    reqs = load_mixed_manual_requests(common.HIS_XLSX)
    return reqs[:GRANT_NCELLS] if GRANT_NCELLS and GRANT_NCELLS > 0 else reqs


def _cache_requests(list_name: str, source_key: str) -> List[RasterRequest]:
    """Cache-list cells as SI-style (NeuronID-matched) requests.

    ``list_name`` is a ``spike_count_connector.LISTS`` key ("KW", "ANOVA",
    "MUA_KW", "MUA_ANOVA"). ``_windows_for`` returns one row per (neuron, window)
    with normalised ``Date`` ("YYYY-MM-DD") and integer ``Round No.``.
    """
    w = scc._windows_for(list_name)
    reqs: List[RasterRequest] = []
    for _, row in w.iterrows():
        neuron_id = str(row["NeuronID"]).strip()
        window = (float(row["WindowStart_ms"]), float(row["WindowEnd_ms"]))
        reqs.append(RasterRequest(
            source_kind=source_key,
            date=str(row["Date"]),
            round_no=int(row["Round No."]),
            match_column="NeuronID",
            match_value=neuron_id,
            label=neuron_id,
            window_ms=window,
            p_value=None,
        ))
    return reqs


# --------------------------------------------------------------------------- #
# Source registry — one entry per DATA_SOURCE
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceSpec:
    key: str                                   # DATA_SOURCE value (internal id)
    display: str                               # friendly name for titles/filenames/
    #                                            dirs/lane prefix (grant → "original",
    #                                            cache_* → "updated_*")
    id_col: str                                # "Channel" (mixed) or "NeuronID"
    make_source: Callable[[], object]          # fresh spike-source instance
    load_requests: Callable[[], List[RasterRequest]]

    @property
    def prefix(self) -> str:                   # lane-label prefix, e.g. "original: "
        return f"{self.display}: "


SOURCES: Dict[str, SourceSpec] = {
    "grant_xlsx": SourceSpec(
        "grant_xlsx", "original", "Channel",
        lambda: MixedManualSpikeSource(),
        _grant_requests,
    ),
    "cache_kw": SourceSpec(
        "cache_kw", "updated_kw", "NeuronID",
        scc._si_sorted_source,                  # spikes from sorted_spike_cache_filtered
        lambda: _cache_requests("KW", "cache_kw"),
    ),
    "cache_anova": SourceSpec(
        "cache_anova", "updated_anova", "NeuronID",
        scc._si_sorted_source,                  # spikes from sorted_spike_cache_filtered
        lambda: _cache_requests("ANOVA", "cache_anova"),
    ),
    "cache_mua_kw": SourceSpec(
        "cache_mua_kw", "updated_mua_kw", "NeuronID",
        scc._mua_source,
        lambda: _cache_requests("MUA_KW", "cache_mua_kw"),
    ),
    "cache_mua_anova": SourceSpec(
        "cache_mua_anova", "updated_mua_anova", "NeuronID",
        scc._mua_source,
        lambda: _cache_requests("MUA_ANOVA", "cache_mua_anova"),
    ),
}

# The three overlay pairs the user asked for (side A = grant, side B = a cache list).
OVERLAY_PAIRS: List[Tuple[str, str]] = [
    ("grant_xlsx", "cache_kw"),
    ("grant_xlsx", "cache_anova"),
    ("grant_xlsx", "cache_mua_anova"),
]


def _safe(label: str) -> str:
    """Filesystem-safe token from a lane label."""
    return (label.replace(" ", "").replace("/", "-")
            .replace(".", "").replace(":", "").replace("\\", "-"))


# --------------------------------------------------------------------------- #
# Singles — one raster+PSTH+probe+footprint figure per unit
# --------------------------------------------------------------------------- #
def run_singles(
    source_key: str,
    out_dir: str,
    *,
    show_waveforms: bool = True,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
) -> Dict[str, int]:
    """Render one figure per unit in ``source_key``'s list.

    Each figure is ``plot_overlay_raster`` used single-lane, so it carries the
    raster + PSTH + probe map + waveform footprint. Windows are deduped to one
    figure per unit; the unit's significant window is shaded. Sessions load once.
    """
    spec = SOURCES[source_key]
    requests = dedup_by_unit(spec.load_requests())
    os.makedirs(out_dir, exist_ok=True)
    print(f"[singles {source_key}] {len(requests)} unit(s) → {out_dir}")

    by_session: Dict[tuple, List[RasterRequest]] = defaultdict(list)
    for r in requests:
        by_session[(r.date, r.round_no)].append(r)

    stats = {"plotted": 0, "skipped_no_rows": 0, "skipped_no_trials": 0, "sessions_failed": 0}
    for (date, round_no), reqs in sorted(by_session.items()):
        source = spec.make_source()
        try:
            session_df = source.load(date, round_no)
        except Exception as e:  # missing cache / recording / DB
            print(f"[{source_key} {date} r{round_no}] load failed: {e}")
            stats["sessions_failed"] += 1
            continue
        if session_df is None or getattr(session_df, "empty", True):
            print(f"[{source_key} {date} r{round_no}] no data")
            stats["sessions_failed"] += 1
            continue

        # resolve each request's rows and its lane label
        resolved: Dict[str, Tuple[RasterRequest, "object"]] = {}
        for req in reqs:
            rows = _select_unit_rows(session_df, req)
            if rows.empty:
                print(f"[{req.label}] no matching rows in session")
                stats["skipped_no_rows"] += 1
                continue
            resolved[_lane_label(spec.prefix, req.match_value)] = (req, rows)

        # footprints for the whole session's units, cut once from the recording
        footprints: dict = {}
        if show_waveforms and resolved:
            footprints = extract_footprints(
                {lab: rr[1] for lab, rr in resolved.items()}, date, round_no)

        for label, (req, rows) in resolved.items():
            if "_Unit" in req.match_value:
                status, tag = "sorted unit", "SU"
            elif source_key.startswith("cache_mua"):
                status, tag = "multiunit (MUA)", "MUA"    # threshold-MUA channel
            else:
                status, tag = "unsorted", "unsorted"      # whole-channel, never sorted
            loc = _unit_location(rows)
            title = f"{spec.display}  ·  {req.label}  ·  {status}"
            if loc:
                title += f"   ·   {loc}"
            if req.p_value is not None:
                title += f"   ·   p={req.p_value:.3g}"
            # include the session (date_round) so same-named cells from different
            # sessions can't overwrite each other
            sess = f"{req.date}_{req.round_no}"
            save_path = os.path.join(
                out_dir,
                f"{spec.display}_{sess}_{_safe(_cell_token(req.match_value))}_{tag}.png")
            fig = plot_overlay_raster(
                {label: rows},
                title=title,
                window_s=req.window_s,
                footprints=({label: footprints.get(label)} if show_waveforms else None),
                locations={label: loc},
                show_probe=True,
                xlim=xlim,
                psth_bin_ms=psth_bin_ms,
                save_path=save_path,
            )
            stats["plotted" if fig is not None else "skipped_no_trials"] += 1

    print(f"[singles {source_key}] done: {stats}")
    return stats


# --------------------------------------------------------------------------- #
# Overlay — coincidence-match two sources' sorts per session and overlay
# --------------------------------------------------------------------------- #
def run_overlay(
    source_a: str,
    source_b: str,
    out_dir: str,
    *,
    listed_only: bool = False,
    show_waveforms: bool = True,
    only_date: Optional[str] = None,
    only_round: Optional[int] = None,
    coincidence_threshold: float = 0.4,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    footprint_similarity_threshold: float = 0.6,
    window_ms: float = DEFAULT_WINDOW_MS,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
) -> List[str]:
    """Overlay ``source_a`` vs ``source_b``, coincidence-matched per session.

    For every ``(date, round)`` present in either list, load both sources, match
    units across sorts by spike-time coincidence (``render_overlays_for_units``),
    and overlay each matched neuron group. ``source_a``'s cells are the amber
    (side-A) anchors, ``source_b``'s the violet (side-B) anchors; each anchor's
    significant window is shaded. ``listed_only`` restricts to cells in both
    lists; off (default) it also pulls in each listed cell's cross-sort twin.

    A pair is only overlaid if it is a confident *same neuron* match: spike-time
    coincidence ≥ ``coincidence_threshold`` (default 0.4, up from the module's
    permissive 0.2) AND — when ``show_waveforms`` is on — footprint cosine
    similarity ≥ ``footprint_similarity_threshold`` (default 0.6; set 0 to skip).
    Both are printed per session so you can see how many pairs survive and retune.
    """
    a, b = SOURCES[source_a], SOURCES[source_b]
    os.makedirs(out_dir, exist_ok=True)

    # (date, round) -> {unit_id: [window_s, ...]}, keyed as session_units keys them
    a_windows: Dict[tuple, dict] = defaultdict(dict)
    b_windows: Dict[tuple, dict] = defaultdict(dict)
    for r in a.load_requests():
        wl = a_windows[(r.date, r.round_no)].setdefault(r.match_value, [])
        if r.window_s is not None:
            wl.append(r.window_s)
    for r in b.load_requests():
        wl = b_windows[(r.date, r.round_no)].setdefault(r.match_value, [])
        if r.window_s is not None:
            wl.append(r.window_s)

    sessions = sorted(set(a_windows) | set(b_windows))
    if only_date is not None:
        sessions = [s for s in sessions
                    if s[0] == only_date and (only_round is None or s[1] == only_round)]
    print(f"[overlay {source_a} vs {source_b}] {len(sessions)} session(s) → {out_dir}")

    written: List[str] = []
    for date, round_no in sessions:
        loaded = {}
        for kind, spec in (("a", a), ("b", b)):
            try:
                loaded[kind] = spec.make_source().load(date, round_no)
            except Exception as e:
                print(f"[overlay {source_a} vs {source_b} {date} r{round_no}] "
                      f"{spec.key} load failed: {e}")
                loaded[kind] = None
        da, db = loaded["a"], loaded["b"]
        if da is None or getattr(da, "empty", True) or db is None or getattr(db, "empty", True):
            missing = [spec.key for kind, spec in (("a", a), ("b", b))
                       if loaded[kind] is None or getattr(loaded[kind], "empty", True)]
            print(f"[overlay {source_a} vs {source_b} {date} r{round_no}] "
                  f"need both sources; missing/empty: {missing}")
            continue

        units_a = session_units(da, a.id_col)
        units_b = session_units(db, b.id_col)

        footprints_by_id = None
        if show_waveforms:
            # both sorts cut from the same voltages → directly comparable
            footprints_by_id = extract_footprints({**units_a, **units_b}, date, round_no)

        written += render_overlays_for_units(
            units_a, units_b, out_dir, label=f"{date}_round{round_no}",
            mixed_anchor_windows=a_windows.get((date, round_no)),
            si_anchor_windows=b_windows.get((date, round_no)),
            listed_only=listed_only, footprints_by_id=footprints_by_id,
            coincidence_threshold=coincidence_threshold, ratio_threshold=ratio_threshold,
            footprint_similarity_threshold=footprint_similarity_threshold,
            window_ms=window_ms, xlim=xlim, psth_bin_ms=psth_bin_ms,
            a_prefix=a.prefix, b_prefix=b.prefix,
        )

    print(f"[overlay {source_a} vs {source_b}] done: {len(written)} figure(s)")
    return written


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #
def _singles_dir(out_dir: str, key: str) -> str:
    return os.path.join(out_dir, "singles", SOURCES[key].display)


def _overlay_dir(out_dir: str, a: str, b: str) -> str:
    return os.path.join(out_dir, "overlay", f"{SOURCES[a].display}_vs_{SOURCES[b].display}")


def run_all_singles(out_dir: str, **kw) -> None:
    """Singles for all five DATA_SOURCE lists into ``out_dir/singles/<display>/``."""
    for key in SOURCES:
        run_singles(key, _singles_dir(out_dir, key), **kw)


def run_all_overlays(out_dir: str, **kw) -> None:
    """Overlays for all three requested pairs into ``out_dir/overlay/<a>_vs_<b>/``."""
    for a, b in OVERLAY_PAIRS:
        run_overlay(a, b, _overlay_dir(out_dir, a, b), **kw)


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["singles", "overlay", "all"], default="singles")
    p.add_argument("--source", choices=list(SOURCES), help="singles: which DATA_SOURCE")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"), help="overlay: two DATA_SOURCE keys")
    p.add_argument("--out", default=os.path.join(_HERE, "output"))
    p.add_argument("--no-waveforms", action="store_true", help="skip the footprint panel")
    p.add_argument("--listed-only", action="store_true", help="overlay: cells in both lists only")
    p.add_argument("--coincidence", type=float, default=0.4,
                   help="overlay: spike-time coincidence floor for a cross-sort match")
    p.add_argument("--footprint-sim", type=float, default=0.6,
                   help="overlay: waveform-footprint cosine floor (0 = off)")
    p.add_argument("--xlim", type=float, default=2.4)
    p.add_argument("--psth-bin-ms", type=float, default=50.0)
    args = p.parse_args(argv)

    show_wf = not args.no_waveforms
    overlay_kw = dict(coincidence_threshold=args.coincidence,
                      footprint_similarity_threshold=args.footprint_sim)
    if args.mode == "all":
        run_all_singles(args.out, show_waveforms=show_wf, xlim=args.xlim, psth_bin_ms=args.psth_bin_ms)
        run_all_overlays(args.out, show_waveforms=show_wf, listed_only=args.listed_only,
                         xlim=args.xlim, psth_bin_ms=args.psth_bin_ms, **overlay_kw)
    elif args.mode == "singles":
        src = args.source or "grant_xlsx"
        run_singles(src, _singles_dir(args.out, src),
                    show_waveforms=show_wf, xlim=args.xlim, psth_bin_ms=args.psth_bin_ms)
    else:  # overlay
        a, b = tuple(args.pair) if args.pair else OVERLAY_PAIRS[0]
        run_overlay(a, b, _overlay_dir(args.out, a, b),
                    listed_only=args.listed_only, show_waveforms=show_wf,
                    xlim=args.xlim, psth_bin_ms=args.psth_bin_ms, **overlay_kw)


if __name__ == "__main__":
    # ========================================================================
    #  RUN CONFIG — edit these, then just press ▶ Run in PyCharm.
    #
    #  MODE options:
    #    "singles"  one raster+PSTH+probe+footprint figure per unit in ONE list.
    #    "overlay"  overlay ONE pair of sources, coincidence-matched per session.
    #    "all"      every source's singles + all three overlay pairs.
    # ========================================================================
    MODE = "overlay"                       # "singles" | "overlay" | "all"

    OUT_DIR = os.path.join(_HERE, "output")

    # -- MODE == "singles" --  (one of SOURCES)
    SINGLES_SOURCE = "grant_xlsx"            # grant_xlsx | cache_kw | cache_anova
    #                                       | cache_mua_kw | cache_mua_anova

    # -- MODE == "overlay" --  (side A = grant, side B = a cache list)
    OVERLAY_PAIR = ("grant_xlsx", "cache_anova")   # or (.., "cache_anova"),
    #                                             (.., "cache_mua_anova")
    LISTED_ONLY = False                    # True → only cells in BOTH lists
    #  Same-neuron matching gates (overlay only). A cross-sort pair is drawn only
    #  if its spike trains are coincident AND (when waveforms are on) its footprints
    #  are similar enough. Raise to be stricter, lower if too many real pairs drop.
    #  Watch the console: it prints how many pairs each gate keeps.
    COINCIDENCE_THRESHOLD = 0.4            # spike-time coincidence floor (was 0.2)
    FOOTPRINT_SIM_THRESHOLD = 0.6          # waveform-footprint cosine floor (0 = off)

    # -- applies to all modes --
    SHOW_WAVEFORMS = True                  # False → drop the footprint panel (faster);
    #                                        also disables the footprint gate
    ONLY_DATE = None                       # e.g. "2023-09-26" to restrict (overlay)
    ONLY_ROUND = None                      # e.g. 2 (only used with ONLY_DATE)
    XLIM_S = 2.4
    PSTH_BIN_MS = 50.0
    # ========================================================================

    _overlay_kw = dict(
        coincidence_threshold=COINCIDENCE_THRESHOLD,
        footprint_similarity_threshold=FOOTPRINT_SIM_THRESHOLD,
    )
    if MODE == "singles":
        run_singles(SINGLES_SOURCE, _singles_dir(OUT_DIR, SINGLES_SOURCE),
                    show_waveforms=SHOW_WAVEFORMS, xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS)
    elif MODE == "overlay":
        _a, _b = OVERLAY_PAIR
        run_overlay(_a, _b, _overlay_dir(OUT_DIR, _a, _b),
                    listed_only=LISTED_ONLY, show_waveforms=SHOW_WAVEFORMS,
                    only_date=ONLY_DATE, only_round=ONLY_ROUND,
                    xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS, **_overlay_kw)
    elif MODE == "all":
        run_all_singles(OUT_DIR, show_waveforms=SHOW_WAVEFORMS,
                        xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS)
        run_all_overlays(OUT_DIR, show_waveforms=SHOW_WAVEFORMS, listed_only=LISTED_ONLY,
                         xlim=XLIM_S, psth_bin_ms=PSTH_BIN_MS, **_overlay_kw)
    else:
        raise ValueError(f"unknown MODE {MODE!r}")
