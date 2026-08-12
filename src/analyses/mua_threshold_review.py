"""
mua_threshold_review.py — draw the answer-key cells at chosen MUA parameters, one folder
per parameter set, so a threshold can be JUDGED BY EYE and not only by its score.

``mua_threshold_tuning`` ranks parameters by how well the offline MUA reproduces the
curated unsorted cells. A number is not a raster, though: an f1 of 0.66 does not say
whether the extra spikes are a plausible denser multiunit or a wall of noise, and the
tuner's own report says when its objective cannot separate adjacent multipliers. This
renders the picture the number stands for.

Each figure is the SAME two-lane overlay ``raster_review_by_source.py --mode pairs``
draws -- ``zombies_raster.plot_overlay_raster``, with raster, PSTH, probe map, footprint
and peak-channel waveform:

    lane A   original unsorted, from the exploded cache (the answer key)
    lane B   offline MUA at one candidate parameter set, detected here from amplifier.dat

and the title carries that cell's agreement scores, so the picture and the numbers that
chose it are on the same page.

WHY IT DETECTS RATHER THAN READING A CACHE
------------------------------------------
The pairs figure reads lane B from ``threshold_mua_spike_cache_*``, which means seeing a
candidate requires rebuilding that cache for every parameter set you want to look at, and
editing ``spike_count_connector._mua_source()`` to point at it. Detection here is done in
memory from the same ``amplifier.dat``, with the same ``detect_mad_spikes`` and the same
windowing the builder uses, so several candidates can be compared side by side before
anything is written. Nothing here writes a spike cache.

ONE CLOCK, DELIBERATELY THE AMPLIFIER'S
---------------------------------------
Both lanes are emitted on the AMPLIFIER's clock: the answer key's spikes and its
``EpochStartStop`` are shifted onto the marker-derived epochs (see
``mua_threshold_tuning`` for why the two clocks differ on a stitched session). The raster
would work on either clock -- it re-zeroes each trial by its own ``EpochStartStop`` -- but
the FOOTPRINT panel would not: it cuts waveform snippets from the continuous recording at
each lane's raw spike times, so on a stitched session compiled-clock times land on the
wrong samples and the waveform comes out as noise. On an unstitched session the two
clocks are identical and this changes nothing.

HOW TO RUN
----------
Needs the recordings and the exploded cache, so run it on the rig. PyCharm: edit the
CONFIG block at the bottom and hit Run. Or::

    cd src
    python -m analyses.mua_threshold_review --multipliers 4.0,5.5,7.0
    python -m analyses.mua_threshold_review --multipliers 5.5 --max-sessions 2
    python -m analyses.mua_threshold_review --multipliers 5.5,7.0 --out ~/mua_review

Output, one directory per parameter set::

    <out>/mad4.0_ref1.0/<date>_r<round>_<channel>.png
    <out>/mad5.5_ref1.0/<date>_r<round>_<channel>.png
    <out>/mad7.0_ref1.0/<date>_r<round>_<channel>.png
    <out>/scores.csv

Same filename in each, so flipping between folders holds the cell still and changes only
the threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from analyses.mua_threshold_tuning import (
    EXPLODED_CACHE_SUBDIR, GRANT_NCELLS, MATCH_WINDOW_MS, AnswerCell, _align_trials,
    _as_cache_list, _explain_no_rows, _match_rows, continuous_epochs, filter_sessions,
    load_answer_key_csv, load_grant_answer_key, load_session_frames, score_candidate,
)
from data_access.mua_threshold_calibration import (
    _parse_sessions, iter_filtered_channels, session_round_dir,
)
from data_access.threshold_detection import detect_mad_spikes, estimate_noise
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

LANE_A_PREFIX = "original: "        # matches raster_review_by_source's lane naming
XLIM = 2.4
PSTH_BIN_MS = 50.0


def lane_b_prefix(noise_method, multiplier) -> str:
    """Lane label for the candidate, e.g. ``'MUA mad x5.5: '``.

    The literal ``MUA`` matters: ``run_zombies_rasters._lane_label`` tags a whole-channel
    cell ``(MU)`` only when the prefix says MUA, and ``(unsorted)`` otherwise. Lane B is
    genuinely multiunit, so mislabelling it would read as a second copy of lane A.
    """
    return f"MUA {noise_method} x{multiplier:g}: "


def param_dirname(noise_method, multiplier, refractory_ms) -> str:
    """``mad5.5_ref1.0`` -- byte-for-byte the parameter token a MUA cache filename carries.

    Same formatting as ``ThresholdMUASpikeCacheManager``'s label
    (``{date}_round_{n}_{method}{multiplier}_ref{refractory}.pkl``), so the folder you
    liked names the cache you have to build.
    """
    return f"{noise_method}{float(multiplier)}_ref{float(refractory_ms)}"


def output_dir(monkey=SUBJECT_MONKEY) -> Path:
    return Path(DATA_BASE_PATH) / monkey / "mua_calibration" / "review"


def _lane_frames(key_rows, trials, mua_times):
    """``(lane A, lane B)`` per-trial frames for one cell, both on the amplifier clock.

    Lane A is the answer key's own rows with ``EpochStartStop`` and ``SpikeTimes`` moved
    onto the marker-derived epochs; lane B is the same rows with the candidate's spikes
    in those windows. Sharing the rows keeps the trial metadata (stimulus monkey, group,
    task id) identical between lanes, which is what lets the overlay match them
    trial-for-trial instead of falling back to row order.
    """
    rows = key_rows.reset_index(drop=True)
    keep = [t.row_index for t in trials]
    lane_a = rows.iloc[keep].copy().reset_index(drop=True)
    epochs = [(t.mua_onset_s, t.mua_onset_s + t.duration_s) for t in trials]
    lane_a["EpochStartStop"] = epochs
    lane_a["SpikeTimes"] = [list(np.asarray(t.key_times, dtype=float) + t.mua_onset_s)
                            for t in trials]

    lane_b = lane_a.copy()
    lane_b["SpikeTimes"] = [
        list(mua_times[np.searchsorted(mua_times, t.mua_onset_s):
                       np.searchsorted(mua_times, t.mua_onset_s + t.duration_s)])
        for t in trials
    ]
    return lane_a, lane_b


def _render_session(date, round_no, cells, *, params, out_root, monkey=SUBJECT_MONKEY,
                    cache_subdirs, show_waveforms=True, xlim=XLIM,
                    psth_bin_ms=PSTH_BIN_MS, match_window_ms=MATCH_WINDOW_MS):
    """Render every (cell x parameter set) figure for one session. Returns score rows."""
    from analyses.zombies_raster_review.run_zombies_rasters import (
        _cell_token, _lane_label, _unit_location,
    )
    from analyses.zombies_raster_review.waveform_footprint import extract_footprints
    from analyses.zombies_raster_review.zombies_raster import plot_overlay_raster

    date = str(date)
    channels = [c.channel for c in cells]
    frames = load_session_frames(date, round_no, cache_subdirs, monkey=monkey)
    if not frames:
        print(f"[review] {date} r{round_no}: session in none of {cache_subdirs} — skipped")
        return []

    round_dir = session_round_dir(date, round_no, monkey)
    epochs_by_task, rows_out = None, []
    # {label: lane frame} for the whole session, so footprints are cut in ONE pass --
    # loading the recording for windowsort is by far the most expensive step here.
    lanes_by_cell, all_lanes = {}, {}

    for channel, voltage, sample_rate in iter_filtered_channels(round_dir,
                                                                only_channels=channels):
        if epochs_by_task is None:
            try:
                epochs_by_task = continuous_epochs(round_dir, sample_rate)
            except Exception as exc:
                print(f"[review] {date} r{round_no}: marker epochs unavailable "
                      f"({type(exc).__name__}: {exc}); using compiled.pkl epochs — correct "
                      f"only if this session was NOT stitched.")
                epochs_by_task = {}
        epochs = epochs_by_task or None

        key_rows, from_cache = None, None
        for sub, df in frames.items():
            hit = _match_rows(df, channel, date, round_no)
            if not hit.empty:
                key_rows, from_cache = hit, sub
                break
        if key_rows is None:
            _explain_no_rows(frames[next(iter(frames))], channel, date, round_no,
                             caches=list(frames))
            continue

        trials, tstats = _align_trials(key_rows, epochs)
        if not trials:
            print(f"[review] {date} r{round_no} {channel}: no trials survived alignment "
                  f"— skipped")
            continue

        token = _cell_token(channel)
        lab_a = _lane_label(LANE_A_PREFIX, channel)
        for method, mult, refr in params:
            sigma = float(estimate_noise(voltage, method))
            refr_samples = max(1, int(refr / 1000.0 * sample_rate))
            mua_times = detect_mad_spikes(voltage, -mult * sigma, refr_samples) / sample_rate
            scores = score_candidate(trials, mua_times, match_window_ms=match_window_ms)
            lane_a, lane_b = _lane_frames(key_rows, trials, mua_times)
            lab_b = _lane_label(lane_b_prefix(method, mult), channel)
            lanes_by_cell[(channel, method, mult, refr)] = (lab_a, lane_a, lab_b, lane_b, scores)
            all_lanes[lab_a] = lane_a
            all_lanes[f"{lab_b} @{param_dirname(method, mult, refr)}"] = lane_b
            rows_out.append(dict(
                Date=date, **{"Round No.": int(round_no)}, Channel=channel,
                cell=f"{date}_r{round_no}_{channel}", noise_method=method,
                threshold_multiplier=mult, refractory_ms=refr, sigma_uv=sigma,
                threshold_uv=-mult * sigma, n_trials=tstats["n_trials"],
                exploded_cache=from_cache, **scores))
        print(f"  {date} r{round_no} {token}: {tstats['n_trials']} trials, "
              f"{sum(t.key_times.size for t in trials)} answer-key spikes"
              + (f" [{from_cache}]" if len(frames) > 1 else ""))

    if not lanes_by_cell:
        return rows_out

    footprints = {}
    if show_waveforms:
        # cuts snippets from the continuous recording at each lane's spike times; both
        # lanes are on the amplifier clock, so this is valid on stitched sessions too
        footprints = extract_footprints(all_lanes, date, round_no)

    for (channel, method, mult, refr), (lab_a, lane_a, lab_b, lane_b, scores) in \
            lanes_by_cell.items():
        pdir = Path(out_root) / param_dirname(method, mult, refr)
        pdir.mkdir(parents=True, exist_ok=True)
        lanes = {lab_a: lane_a, lab_b: lane_b}
        locations = {lab_a: _unit_location(lane_a), lab_b: _unit_location(lane_b)}
        loc = next((v for v in locations.values() if v), None)
        title = (f"original unsorted  vs  MUA {method} x{mult:g} (ref {refr:g} ms)"
                 f"  ·  {_cell_token(channel)}  ·  {date}_r{round_no}"
                 + (f"   ·   {loc}" if loc else "") + "\n"
                 f"f1 {scores['f1']:.2f}  ·  recall {scores['recall']:.2f}  ·  "
                 f"precision {scores['precision']:.2f}  ·  "
                 f"{scores['rate_ratio']:.2f}x the answer key's spikes")
        fp = {lab_a: footprints.get(lab_a),
              lab_b: footprints.get(f"{lab_b} @{param_dirname(method, mult, refr)}")}
        safe = f"{date}_r{round_no}_{_cell_token(channel)}".replace(" ", "").replace(".", "")
        plot_overlay_raster(
            lanes, title=title, footprints=fp if show_waveforms else None,
            locations=locations, show_probe=True, xlim=xlim, psth_bin_ms=psth_bin_ms,
            save_path=str(pdir / f"{safe}.png"))
    return rows_out


def review(cells: Optional[Sequence[AnswerCell]] = None, *,
           multipliers: Sequence[float] = (4.0, 5.5, 7.0), noise_method: str = "mad",
           refractory_ms: float = 1.0, monkey=SUBJECT_MONKEY, out_dir=None,
           exploded_cache_subdir=EXPLODED_CACHE_SUBDIR, all_grant_rows: bool = False,
           sessions=None, max_sessions: Optional[int] = None, show_waveforms: bool = True,
           xlim: float = XLIM, psth_bin_ms: float = PSTH_BIN_MS):
    """Render every answer-key cell at every requested parameter set. Returns the scores."""
    cells = cells if cells is not None else load_grant_answer_key(
        n_rows=None if all_grant_rows else GRANT_NCELLS)
    cells = filter_sessions(cells, sessions=sessions, max_sessions=max_sessions)
    if not cells:
        print("[review] answer key is empty — nothing to draw.")
        return pd.DataFrame()

    params = [(noise_method, float(m), float(refractory_ms)) for m in multipliers]
    out_root = Path(out_dir) if out_dir else output_dir(monkey)
    cache_subdirs = _as_cache_list(exploded_cache_subdir)
    print(f"[review] {len(cells)} cell(s), {len(params)} parameter set(s) → {out_root}")
    for p in params:
        print(f"         {param_dirname(*p)}")

    by_session = {}
    for c in cells:
        by_session.setdefault(c.session, []).append(c)

    rows, failed = [], []
    for i, ((date, round_no), session_cells) in enumerate(sorted(by_session.items()), 1):
        print(f"[{i}/{len(by_session)}] {date} round {round_no} ({len(session_cells)} cell(s))")
        try:
            rows.extend(_render_session(
                date, round_no, session_cells, params=params, out_root=out_root,
                monkey=monkey, cache_subdirs=cache_subdirs, show_waveforms=show_waveforms,
                xlim=xlim, psth_bin_ms=psth_bin_ms))
        except Exception as exc:
            failed.append((f"{date} r{round_no}", f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED — {type(exc).__name__}: {exc}")

    scores = pd.DataFrame(rows)
    if not scores.empty:
        out_root.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_root / "scores.csv", index=False)
        n_figs = len(scores)
        print(f"\n[review] {n_figs} figure(s) over {scores['cell'].nunique()} cell(s) "
              f"→ {out_root}")
        print(f"[review] per-cell scores → {out_root / 'scores.csv'}")
        summary = scores.groupby(["noise_method", "threshold_multiplier", "refractory_ms"]).agg(
            n_cells=("cell", "nunique"), f1=("f1", "mean"), recall=("recall", "mean"),
            precision=("precision", "mean"), rate_ratio=("rate_ratio", "median")).reset_index()
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for tag, err in failed:
        print(f"[review] FAILED {tag}: {err}")
    return scores


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multipliers", default="4.0,5.5,7.0",
                   help="comma-separated multipliers to render, one folder each")
    p.add_argument("--noise-method", default="mad")
    p.add_argument("--refractory-ms", type=float, default=1.0)
    p.add_argument("--answer-key", default=None,
                   help="CSV/xlsx of your own cells; default is the grant list")
    p.add_argument("--all-grant-rows", action="store_true")
    p.add_argument("--exploded-cache", default=EXPLODED_CACHE_SUBDIR,
                   help="cache subdir(s), comma separated, tried in order")
    p.add_argument("--sessions", default=None, help="'2023-09-26:2,2023-10-03:4'")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-waveforms", action="store_false", dest="show_waveforms",
                   help="skip the footprint panel (much faster: no recording reload)")
    p.add_argument("--xlim", type=float, default=XLIM)
    a = p.parse_args(argv)

    return review(load_answer_key_csv(a.answer_key) if a.answer_key else None,
                  multipliers=[float(m) for m in a.multipliers.split(",")],
                  noise_method=a.noise_method, refractory_ms=a.refractory_ms,
                  all_grant_rows=a.all_grant_rows, out_dir=a.out,
                  exploded_cache_subdir=[s.strip() for s in a.exploded_cache.split(",") if s.strip()],
                  sessions=_parse_sessions(a.sessions) if a.sessions else None,
                  max_sessions=a.max_sessions, show_waveforms=a.show_waveforms, xlim=a.xlim)


# ===== Run directly in PyCharm — edit this block and hit Run (no CLI) =========
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _cli()
        raise SystemExit(0)

    # One folder per multiplier, same filename in each, so flipping between folders
    # holds the cell still and changes only the threshold. Include the current default
    # (4.0) as the reference point even when you are only weighing two candidates.
    MULTIPLIERS = (4.0, 5.5, 7.0)
    NOISE_METHOD = "mad"
    REFRACTORY_MS = 1.0

    ANSWER_KEY_CSV = None       # your own cells; None = the grant list's unsorted cells
    ALL_GRANT_ROWS = False
    EXPLODED_CACHE = [EXPLODED_CACHE_SUBDIR, "exploded_spike_cache_gitrecovered"]

    SESSIONS = None             # e.g. [("2023-09-26", 2)]
    MAX_SESSIONS = None         # e.g. 2 for a quick look before rendering all of them
    SHOW_WAVEFORMS = True       # False is much faster (skips reloading the recording)
    OUT_DIR = None              # None = <data>/<monkey>/mua_calibration/review

    review(load_answer_key_csv(ANSWER_KEY_CSV) if ANSWER_KEY_CSV else None,
           multipliers=MULTIPLIERS, noise_method=NOISE_METHOD, refractory_ms=REFRACTORY_MS,
           all_grant_rows=ALL_GRANT_ROWS, exploded_cache_subdir=EXPLODED_CACHE,
           sessions=SESSIONS, max_sessions=MAX_SESSIONS, show_waveforms=SHOW_WAVEFORMS,
           out_dir=OUT_DIR)
