"""
mua_threshold_tuning.py — choose the offline MUA detection parameters by reproducing a
CURATED answer key of unsorted cells, scored with a train/test split.

WHY AN ANSWER KEY AND NOT spike.dat
-----------------------------------
The obvious calibration target is ``spike.dat``, the online detection itself. It is the
wrong target wholesale, for three reasons that are properties of how the data were
collected, not of the code:

  * the online thresholds were dialled in BY HAND across 32 channels and not tracked, so
    an unknown subset of channels is thresholded at a value nobody stands behind;
  * interrupted sessions have a spike.dat per sub-folder, so a whole-recording target has
    to be stitched before it means anything;
  * the goal here is MULTIUNIT activity, and the online thresholds were set while
    watching for units on a scope.

What IS trustworthy is the small set of unsorted cells that were curated out of that
detection and carried into the grant analysis: those channels were looked at, and they
are what any retuned cache still has to reproduce. So this module targets THOSE cells,
and reads their spike trains from the exploded cache rather than from ``spike.dat`` --
no raw online file is opened, and the stitching problem never arises for the spikes.

    answer key   the grant list's UNSORTED cells (``Channel.C_020``, no ``_Unit``),
                 spike times from ``exploded_spike_cache_pre1000ms`` via
                 MixedManualSpikeSource -- the SAME lane A that
                 ``raster_review_by_source.py --mode pairs`` draws.
    candidate    offline MUA on the same channel at some (noise method, multiplier,
                 refractory), i.e. lane B of that same figure.
    score        how well lane B reproduces lane A, per cell.

So the number this maximises is the number the pairs figure shows you.

A CELL CAN OUTLIVE THE CACHE THAT HELD IT
-----------------------------------------
Every channel that has since been MANUALLY SORTED has lost its unsorted row:
``combine_unsorted_with_sorted`` drops a channel's whole-channel row as soon as
``sorted_spikes.pkl`` names that channel, so the cell exists in the current exploded
cache only as ``Channel.C_020_Unit 1`` and friends. Its answer key is not gone, but it
lives in an exploded cache built BEFORE that sort.

``--exploded-cache`` therefore takes a LIST, tried in order per cell::

    --exploded-cache exploded_spike_cache_pre1000ms,exploded_spike_cache_gitrecovered

Each cell resolves from the newest cache that still has it, and every score row records
which one -- so a mixed-vintage answer key is stated in the output and auditable in the
CSV rather than silently assumed.

THE ONE CLOCK PROBLEM, HANDLED
------------------------------
The answer key's times come from ``compiled.pkl``; the candidate's come from
``amplifier.dat``. On a STITCHED session those are different clocks -- compiled.pkl
carries per-sub-folder RESET times while the amplifier is continuous (the reason
``mua_peristim_builder`` exists at all). Comparing them raw would score a stitched
session as pure noise and drag the whole fit.

Every trial is therefore aligned to its own onset before scoring: the answer key by its
``EpochStartStop``, the candidate by the marker-derived epoch for the same task id
(``mua_peristim_builder._continuous_epochs``). Both become "seconds since stimulus
onset", so the clocks cancel, and spikes can only match WITHIN a trial. The per-session
report prints the median onset difference, so a session whose two clocks disagree is
visible rather than silently down-weighted.

WHAT IS SCORED, AND WHY NOT PLAIN COINCIDENCE
---------------------------------------------
The repo's ``coincidence`` is ``n_matched / min(n_a, n_b)``, which is the right gate for
"are these the same neuron" but the wrong objective to maximise here: a shallower
threshold detects strictly more spikes, so it can only ever raise the fraction of the
answer key that is covered. Maximising it alone drives the multiplier to the bottom of
the grid, detecting everything, matching everything, and reproducing nothing.

So each cell is scored both ways and combined:

    recall     answer-key spikes with a candidate spike within ±MATCH_WINDOW_MS
    precision  candidate spikes with an answer-key spike in that window
    f1         harmonic mean -- the default objective, peaked rather than monotone
    coincidence  the repo's definition, reported for continuity with the pairs figure
    rate_ratio   candidate spikes / answer-key spikes (raster density, 1.0 = same)

Precision below 1 is expected and wanted: the answer key is one hand-set threshold on a
multiunit channel, so a good MUA detector finds spikes it missed. F1 balances that
against the runaway noise detection a too-shallow threshold produces.

TRAIN / TEST
------------
Cells are split BY SESSION, never within one: channels in a session share a recording,
a noise floor and any stitching quirk, so scoring a held-out channel beside its
session-mates would leak. With few sessions the split is leave-one-session-out.

Be clear about what the split can and cannot tell you. One scalar chosen from a 1-D grid
on ~30 cells is a low-variance decision -- cross-validation here is a STABILITY check,
not a defence against a flexible model. What it answers is: does every fold pick the
same multiplier, and does the winner still win on cells it never saw? If the folds
disagree, the objective is flat or a couple of cells are driving it, and that is the
signal to widen the answer key (``--all-grant-rows``, or ``--answer-key`` with your own
CSV) rather than to trust the winner.

``TIE_BREAK='shallow'`` (default) takes the SHALLOWEST multiplier within one standard
error of the best. Ties are common because the objective is flat near its peak, and the
shallower threshold is the one that keeps more multiunit activity -- the point of the
detector.

HOW TO RUN
----------
Needs the raw Intan sessions (amplifier.dat + digitalin.dat + notes.txt + info.rhd) and
the exploded cache, so run it on the rig. PyCharm: open this file, edit the CONFIG block
at the bottom, hit Run. Or::

    cd src
    python -m analyses.mua_threshold_tuning
    python -m analyses.mua_threshold_tuning --all-grant-rows --plots
    python -m analyses.mua_threshold_tuning --answer-key my_cells.csv --multipliers 3,4,5,6,7

Read-only: CSVs and figures go to ``<data>/<monkey>/mua_calibration/tuning/``. It writes
no spike cache -- see NEXT STEPS at the bottom of this file for what to do with the
multiplier it names.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_access.mua_threshold_calibration import (
    _parse_sessions, iter_filtered_channels, session_round_dir,
)
from data_access.spike_window import EXPLODED_CACHE_SUBDIR
# The MUA builder's own marker-epoch reader. Private there, imported here rather than
# copied or promoted: it is the ONE definition of "the epochs on the amplifier's clock",
# and the tuner must window candidates exactly the way the cache does or it would tune
# against something the cache will not reproduce.
from data_access.mua_peristim_builder import _continuous_epochs as continuous_epochs
from data_access.spike_source import MixedManualSpikeSource
from data_access.threshold_detection import detect_mad_spikes, estimate_noise
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

# Match window half-width. 0.4 ms is coincidence_match.DEFAULT_WINDOW_MS -- the same
# window the pairs figure uses to call two trains the same neuron.
MATCH_WINDOW_MS = 0.4

DEFAULT_MULTIPLIERS = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0)

# The grant list's first N rows are the cells replicate_analysis regresses (its
# NCELLS=74). None = every row in the sheet, which is the cheapest way to grow the
# answer key: the later rows are the same kind of curated cell.
GRANT_NCELLS = 74

# A cell needs this many answer-key spikes before its score means anything.
MIN_ANSWER_KEY_SPIKES = 30

# Trials whose two clocks disagree on duration by more than this are dropped: the task
# id matched but the epochs describe different events.
MAX_DURATION_MISMATCH_S = 0.05


# --------------------------------------------------------------------------- #
# Answer key
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnswerCell:
    """One curated unsorted channel to reproduce."""
    date: str                 # "YYYY-MM-DD"
    round_no: int
    channel: str              # "Channel.C_020"

    @property
    def session(self) -> Tuple[str, int]:
        return (self.date, self.round_no)

    @property
    def label(self) -> str:
        return f"{self.date}_r{self.round_no}_{self.channel}"


def _dedup(cells: Sequence[AnswerCell]) -> List[AnswerCell]:
    """One entry per (date, round, channel), input order preserved."""
    seen, out = set(), []
    for c in cells:
        if (c.date, c.round_no, c.channel) in seen:
            continue
        seen.add((c.date, c.round_no, c.channel))
        out.append(c)
    return out


def load_grant_answer_key(xlsx_path=None, n_rows: Optional[int] = GRANT_NCELLS) -> List[AnswerCell]:
    """The grant list's UNSORTED cells -- exactly the set ``--mode pairs`` plots.

    ``n_rows`` mirrors ``spike_count_connector.GRANT_NCELLS``; pass ``None`` to take
    every row of the sheet instead of the first 74, which is the cheapest way to enlarge
    the answer key.
    """
    from analyses.jun2026_grant_investigation import common
    from analyses.jun2026_grant_investigation.grant_mua_matching import is_mua_name
    from analyses.zombies_raster_review.unit_lists import load_mixed_manual_requests

    reqs = load_mixed_manual_requests(xlsx_path or common.HIS_XLSX)
    total = len(reqs)
    if n_rows and n_rows > 0:
        reqs = reqs[:n_rows]
    # is_mua_name: no '_Unit' in the name -> an unsorted whole-channel MUA cell
    cells = [AnswerCell(r.date, int(r.round_no), r.match_value)
             for r in reqs if is_mua_name(r.match_value)]
    cells = _dedup(cells)
    print(f"[answer-key] grant list: {total} row(s)"
          + (f", first {n_rows} used" if n_rows else ", all rows used")
          + f" -> {len(cells)} unique unsorted cell(s) "
            f"in {len({c.session for c in cells})} session(s)")
    return cells


def filter_sessions(cells: Sequence[AnswerCell], *, sessions=None,
                    max_sessions: Optional[int] = None) -> List[AnswerCell]:
    """Narrow an answer key to some sessions -- for a quick first run.

    The full pass reads and filters one channel of every session's amplifier.dat, so it
    is worth confirming on two sessions that the answer key resolves and the clocks line
    up before waiting for all of them. Two sessions cannot SELECT a multiplier (the folds
    would be two), but they will show you the shape of the curve.

    ``sessions`` is an explicit ``[(date, round), ...]``; ``max_sessions`` takes the first
    N in date order.
    """
    out = list(cells)
    if sessions:
        wanted = {(str(d), int(r)) for d, r in sessions}
        out = [c for c in out if c.session in wanted]
    if max_sessions:
        keep = sorted({c.session for c in out})[:max_sessions]
        out = [c for c in out if c.session in set(keep)]
    if len(out) != len(cells):
        print(f"[answer-key] narrowed to {len(out)} cell(s) in "
              f"{len({c.session for c in out})} session(s)")
    return out


def load_answer_key_csv(path) -> List[AnswerCell]:
    """Your own answer key: a CSV/xlsx with Date, Round No., and Cell (or Channel).

    Use this to add example cells beyond the grant list -- any unsorted channel whose
    online threshold you are willing to stand behind. Sorted units (``_Unit`` in the
    name) are rejected rather than silently scored: this tunes a channel-level MUA
    detector, which cannot reproduce one sorted unit out of a channel.
    """
    path = str(path)
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    col = next((c for c in ("Cell", "Channel", "channel", "cell") if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path}: need a 'Cell' or 'Channel' column, got {list(df.columns)}")
    cells = []
    for _, row in df.iterrows():
        name = str(row[col]).strip()
        if "_Unit" in name:
            raise ValueError(
                f"{path}: '{name}' is a sorted unit. The answer key must be unsorted "
                f"whole-channel cells -- a channel-level MUA detector cannot reproduce "
                f"one sorted unit out of a channel.")
        cells.append(AnswerCell(pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                                int(row["Round No."]), name))
    cells = _dedup(cells)
    print(f"[answer-key] {path}: {len(cells)} unique unsorted cell(s) "
          f"in {len({c.session for c in cells})} session(s)")
    return cells


# --------------------------------------------------------------------------- #
# Trial alignment
# --------------------------------------------------------------------------- #
def _match_rows(session_df, channel, date, round_no):
    """One channel's rows in one loaded session frame. Empty frame if absent, no printing.

    Resolution is delegated to ``run_zombies_rasters._select_unit_rows`` -- the SAME
    matcher ``raster_review_by_source --mode pairs`` uses for lane A -- so the cell this
    tunes against is the cell that figure draws. It tries ``str(Channel)`` first and falls
    back to a NeuronID suffix, which is what absorbs the enum-repr drift between cache
    vintages; an exact string comparison here would miss cells the figure resolves.
    """
    from analyses.zombies_raster_review.run_zombies_rasters import _select_unit_rows
    from analyses.zombies_raster_review.unit_lists import RasterRequest

    return _select_unit_rows(session_df, RasterRequest(
        source_kind="mixed", date=str(date), round_no=int(round_no),
        match_column="Channel", match_value=channel, label=channel))


def _explain_no_rows(session_df, channel, date, round_no, caches=()):
    """Say WHY a channel did not resolve. The distinction is the useful part.

    A channel that has been MANUALLY SORTED since the answer key was made is gone from
    the unsorted rows by construction -- ``combine_unsorted_with_sorted`` drops the
    whole-channel row once ``sorted_spikes.pkl`` names that channel -- so no threshold
    can reproduce it and no amount of re-running will help. That is a different problem
    from a channel spelled differently, which is a different problem again from a channel
    that simply is not in the session.
    """
    present = sorted(session_df["Channel"].astype(str).unique())
    token = re.sub(r"\D", "", channel)                   # '020' from 'Channel.C_020'
    near = [c for c in present if token and token in c]
    unsorted_near = [c for c in near if "_Unit" not in c]
    where = f" in {', '.join(caches)}" if caches else ""
    print(f"[tune] {date} r{round_no} {channel}: no unsorted rows{where} — skipped")
    if near and not unsorted_near:
        print(f"       present only as {near} — this channel has been MANUALLY SORTED since "
              f"the answer key was made, so its unsorted whole-channel row no longer exists.")
        print("       Add a cache built before that sort to --exploded-cache (they are tried "
              "in order), or drop the cell.")
    elif near:
        print(f"       present as {near} — the answer key spells it differently; use one of "
              f"those spellings in --answer-key.")
    else:
        print(f"       {len(present)} channel(s) in this session, e.g. {present[:8]}")


def select_answer_key_rows(session_df, channel, date, round_no):
    """One channel's rows, or ``None`` after printing why not. Single-frame convenience."""
    rows = _match_rows(session_df, channel, date, round_no)
    if not rows.empty:
        return rows
    _explain_no_rows(session_df, channel, date, round_no)
    return None


def _as_cache_list(exploded_cache_subdir) -> List[str]:
    """Accept one cache subdir or several; several are tried in order, per cell."""
    if isinstance(exploded_cache_subdir, str):
        return [exploded_cache_subdir]
    return [str(s) for s in exploded_cache_subdir]


def load_session_frames(date, round_no, cache_subdirs, monkey=SUBJECT_MONKEY):
    """``{cache subdir: session frame}`` for the caches that actually hold this session.

    Several caches are allowed because the answer key can outlive a cache: a channel that
    has since been manually sorted keeps its unsorted row only in an exploded cache built
    BEFORE that sort. Listing the current cache first and an older one after it resolves
    each cell from the newest cache that still has it, and every score row records which
    cache it came from, so a mixed-vintage answer key stays auditable rather than implicit.
    """
    frames = {}
    for sub in cache_subdirs:
        try:
            df = MixedManualSpikeSource(cache_subdir=sub).load(date, round_no)
        except FileNotFoundError:
            continue                     # this vintage does not have this session
        if df is not None and not getattr(df, "empty", True):
            frames[sub] = df
    return frames


@dataclass
class Trial:
    """One trial, with both clocks resolved to 'seconds since stimulus onset'."""
    task_id: int
    key_times: np.ndarray      # answer-key spikes, aligned (s from onset)
    mua_onset_s: float         # trial onset on the amplifier clock
    duration_s: float
    # position of this trial in the answer-key rows it came from, so a caller that wants
    # the trial's METADATA (stimulus monkey, group) can get it back -- mua_threshold_review
    # rebuilds both raster lanes from these, and a raster grouped by the wrong monkey
    # would be worse than no raster.
    row_index: int = -1


def _align_trials(rows: pd.DataFrame, epochs_by_task: Optional[Dict[int, Tuple[float, float]]],
                  spike_col: str = "SpikeTimes"):
    """``(trials, stats)`` -- the answer key's trials with the amplifier-clock onset attached.

    ``epochs_by_task`` is the marker-derived epoch per task id; pass ``None`` when the
    markers are unavailable, and the compiled epochs are used for both sides (correct on
    an unstitched session, wrong on a stitched one -- the caller warns).
    """
    trials: List[Trial] = []
    offsets, dropped_no_epoch, dropped_duration = [], 0, 0
    for position, (_, row) in enumerate(rows.iterrows()):
        on_c, off_c = (float(x) for x in row["EpochStartStop"])
        duration = off_c - on_c
        if epochs_by_task is None:
            on_m, off_m = on_c, off_c
        else:
            epoch = epochs_by_task.get(int(row["TaskField"]))
            if epoch is None:
                dropped_no_epoch += 1
                continue
            on_m, off_m = float(epoch[0]), float(epoch[1])
            if abs((off_m - on_m) - duration) > MAX_DURATION_MISMATCH_S:
                dropped_duration += 1
                continue
        offsets.append(on_m - on_c)
        spikes = np.asarray(list(row[spike_col]), dtype=float)
        trials.append(Trial(task_id=int(row["TaskField"]),
                            key_times=np.sort(spikes - on_c) if spikes.size else spikes,
                            mua_onset_s=on_m, duration_s=duration, row_index=position))
    stats = {
        "n_trials": len(trials),
        "dropped_no_epoch": dropped_no_epoch,
        "dropped_duration_mismatch": dropped_duration,
        "median_clock_offset_s": float(np.median(offsets)) if offsets else float("nan"),
        "max_clock_offset_s": float(np.max(np.abs(offsets))) if offsets else float("nan"),
    }
    return trials, stats


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _matched(a: np.ndarray, b: np.ndarray, tol_s: float) -> int:
    """Spikes of ``a`` with at least one ``b`` within ``±tol_s``. Both sorted."""
    if a.size == 0 or b.size == 0:
        return 0
    pos = np.searchsorted(b, a)
    left = b[np.clip(pos - 1, 0, b.size - 1)]
    right = b[np.clip(pos, 0, b.size - 1)]
    return int(np.count_nonzero(np.minimum(np.abs(a - left), np.abs(a - right)) <= tol_s))


def score_candidate(trials: Sequence[Trial], mua_times: np.ndarray, *,
                    match_window_ms: float = MATCH_WINDOW_MS) -> dict:
    """Agreement between one candidate MUA train and the answer key, over all trials.

    Counts are accumulated per trial and summed, so a spike can only ever match one in
    the SAME trial -- with both sides aligned to their own onset, that is what keeps a
    stitched session's clock difference from creating or destroying matches.
    """
    tol = match_window_ms / 1000.0
    n_key = n_mua = m_key = m_mua = 0
    for tr in trials:
        lo = np.searchsorted(mua_times, tr.mua_onset_s)
        hi = np.searchsorted(mua_times, tr.mua_onset_s + tr.duration_s)
        cand = mua_times[lo:hi] - tr.mua_onset_s
        n_key += tr.key_times.size
        n_mua += cand.size
        m_key += _matched(tr.key_times, cand, tol)
        m_mua += _matched(cand, tr.key_times, tol)

    recall = m_key / n_key if n_key else float("nan")
    precision = m_mua / n_mua if n_mua else float("nan")
    f1 = (2 * recall * precision / (recall + precision)
          if n_key and n_mua and (recall + precision) > 0 else 0.0)
    # the repo's coincidence: matched fraction of the SMALLER train
    coincidence = ((m_key / n_key if n_key <= n_mua else m_mua / n_mua)
                   if n_key and n_mua else 0.0)
    return {
        "n_answer_key": n_key, "n_mua": n_mua,
        "recall": recall, "precision": precision, "f1": f1,
        "coincidence": coincidence,
        "rate_ratio": n_mua / n_key if n_key else float("nan"),
    }


def score_session(date, round_no, cells: Sequence[AnswerCell], *,
                  multipliers=DEFAULT_MULTIPLIERS, noise_methods=("mad",),
                  refractory_ms=(1.0,), monkey=SUBJECT_MONKEY,
                  match_window_ms=MATCH_WINDOW_MS,
                  exploded_cache_subdir=EXPLODED_CACHE_SUBDIR,
                  verbose=True) -> pd.DataFrame:
    """Score every candidate against every answer-key cell of one session.

    One row per (cell, noise method, multiplier, refractory). Each channel's trace is
    read and filtered once; only the answer key's channels are touched.
    """
    date = str(date)
    channels = [c.channel for c in cells]
    # strict [onset, offset] window -- the unsorted rows have no pre-stimulus baseline
    cache_subdirs = _as_cache_list(exploded_cache_subdir)
    frames = load_session_frames(date, round_no, cache_subdirs, monkey=monkey)
    if not frames:
        print(f"[tune] {date} r{round_no}: session in none of {cache_subdirs} — skipped")
        return pd.DataFrame()

    round_dir = session_round_dir(date, round_no, monkey)

    rows_out: List[dict] = []
    epochs_by_task = None
    seen_channels = set()
    for channel, voltage, sample_rate in iter_filtered_channels(round_dir,
                                                                only_channels=channels):
        seen_channels.add(channel)
        if epochs_by_task is None:                    # needs the sample rate; do it once
            try:
                epochs_by_task = continuous_epochs(round_dir, sample_rate)
            except Exception as exc:
                print(f"[tune] {date} r{round_no}: marker epochs unavailable "
                      f"({type(exc).__name__}: {exc}). Falling back to compiled.pkl epochs "
                      f"for both sides — correct only if this session was NOT stitched.")
                epochs_by_task = {}
        epochs = epochs_by_task or None

        # newest cache that still carries this channel as an unsorted row
        key_rows, from_cache = None, None
        for sub, df in frames.items():
            hit = _match_rows(df, channel, date, round_no)
            if not hit.empty:
                key_rows, from_cache = hit, sub
                break
        if key_rows is None:
            first = next(iter(frames))
            _explain_no_rows(frames[first], channel, date, round_no, caches=list(frames))
            continue
        trials, tstats = _align_trials(key_rows, epochs)
        if not trials:
            print(f"[tune] {date} r{round_no} {channel}: no trials survived alignment "
                  f"({tstats}) — skipped")
            continue

        n_key_total = int(sum(t.key_times.size for t in trials))
        sigma = {m: float(estimate_noise(voltage, m)) for m in noise_methods}
        if verbose:
            print(f"  {date} r{round_no} {channel}: {tstats['n_trials']} trials, "
                  f"{n_key_total} answer-key spikes"
                  + (f" [{from_cache}]" if len(frames) > 1 else "")
                  + f", clock offset {tstats['median_clock_offset_s'] * 1000:+.1f} ms"
                  + (f", dropped {tstats['dropped_no_epoch']}/{tstats['dropped_duration_mismatch']} "
                     f"trials (no epoch / duration mismatch)"
                     if tstats["dropped_no_epoch"] or tstats["dropped_duration_mismatch"] else ""))

        for method in noise_methods:
            for refr in refractory_ms:
                refr_samples = max(1, int(refr / 1000.0 * sample_rate))
                for mult in multipliers:
                    idx = detect_mad_spikes(voltage, -mult * sigma[method], refr_samples)
                    scores = score_candidate(trials, idx / sample_rate,
                                             match_window_ms=match_window_ms)
                    rows_out.append(dict(
                        Date=date, **{"Round No.": int(round_no)}, Channel=channel,
                        cell=f"{date}_r{round_no}_{channel}",
                        noise_method=method, threshold_multiplier=mult, refractory_ms=refr,
                        sigma_uv=sigma[method], threshold_uv=-mult * sigma[method],
                        n_trials=tstats["n_trials"],
                        median_clock_offset_ms=tstats["median_clock_offset_s"] * 1000.0,
                        exploded_cache=from_cache,       # which vintage this cell came from
                        usable=n_key_total >= MIN_ANSWER_KEY_SPIKES,
                        **scores))

    # An answer-key channel the amplifier does not carry is a real problem (wrong channel
    # name, or a channel that was never saved), not something to drop quietly.
    missing = [c for c in channels if c not in seen_channels]
    if missing:
        print(f"[tune] {date} r{round_no}: {len(missing)} answer-key channel(s) absent from "
              f"amplifier.dat and NOT scored: {missing}")
    return pd.DataFrame(rows_out)


def score_answer_key(cells: Sequence[AnswerCell], **kwargs) -> pd.DataFrame:
    """Score every session of the answer key. One bad session never stops the run."""
    by_session: Dict[Tuple[str, int], List[AnswerCell]] = {}
    for c in cells:
        by_session.setdefault(c.session, []).append(c)

    # Say where the answer key is being read from. It is the first thing to check when
    # cells come back unresolved, and it is not obvious: it comes from
    # MixedManualSpikeSource's default, which is the pre-stimulus variant, under
    # JULIE_DATA_PATH -- three defaults deep, none of them in this file.
    root = Path(DATA_BASE_PATH) / kwargs.get("monkey", SUBJECT_MONKEY)
    subdirs = _as_cache_list(kwargs.get("exploded_cache_subdir", EXPLODED_CACHE_SUBDIR))
    print(f"[tune] answer-key spikes from {root}{os.sep}<cache>{os.sep}<date>_round_<n>.pkl")
    for i, sub in enumerate(subdirs):
        exists = (root / sub).is_dir()
        print(f"       cache {i + 1}: {sub}" + ("" if exists else "   [DIRECTORY NOT FOUND]"))

    frames, failed = [], []
    for i, ((date, round_no), session_cells) in enumerate(sorted(by_session.items()), 1):
        print(f"[{i}/{len(by_session)}] {date} round {round_no} "
              f"({len(session_cells)} answer-key cell(s))")
        try:
            frames.append(score_session(date, round_no, session_cells, **kwargs))
        except Exception as exc:
            failed.append((f"{date} r{round_no}", f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED — {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n[tune] {len(failed)} session(s) failed:")
        for tag, err in failed:
            print(f"    {tag}: {err}")
    frames = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    scored = out["cell"].nunique() if not out.empty else 0
    if scored < len(cells):
        print(f"[tune] {scored}/{len(cells)} answer-key cell(s) scored "
              f"({len(cells) - scored} lost to the reasons printed above)")
    # Which vintage each cell came from -- a mixed-vintage answer key is fine, but it
    # should be stated rather than buried in the per-cell CSV.
    if not out.empty and out["exploded_cache"].nunique() > 1:
        by_cache = out.groupby("exploded_cache")["cell"].nunique().to_dict()
        print(f"[tune] answer key drawn from {len(by_cache)} cache vintages: {by_cache}")
    return out


# --------------------------------------------------------------------------- #
# Selection and cross-validation
# --------------------------------------------------------------------------- #
PARAM_KEYS = ["noise_method", "threshold_multiplier", "refractory_ms"]


def _usable(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[scores["usable"]] if "usable" in scores.columns else scores


def param_summary(scores: pd.DataFrame, objective: str = "f1") -> pd.DataFrame:
    """Mean/SEM of every metric per parameter set, across cells."""
    df = _usable(scores)
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(PARAM_KEYS, dropna=False).agg(
        n_cells=("cell", "nunique"),
        objective_mean=(objective, "mean"),
        objective_sem=(objective, lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        f1_mean=("f1", "mean"),
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        coincidence_mean=("coincidence", "mean"),
        rate_ratio_median=("rate_ratio", "median"),
    ).reset_index()
    return agg.sort_values(PARAM_KEYS, ignore_index=True)


def select_params(scores: pd.DataFrame, *, objective: str = "f1",
                  tie_break: str = "shallow") -> Optional[dict]:
    """The winning parameter set. ``None`` if nothing is scorable.

    ``tie_break='shallow'`` applies the one-standard-error rule and then takes the
    SHALLOWEST multiplier that is statistically indistinguishable from the best -- the
    objective is flat near its peak, and of two equally good thresholds the shallower one
    keeps more of the multiunit activity this detector exists to capture.
    ``tie_break='none'`` takes the raw argmax.
    """
    summary = param_summary(scores, objective=objective)
    if summary.empty:
        return None
    best = summary.loc[summary["objective_mean"].idxmax()]
    if tie_break == "shallow":
        floor = best["objective_mean"] - best["objective_sem"]
        within = summary[summary["objective_mean"] >= floor]
        # shallowest multiplier first; among equals prefer the best objective
        chosen = within.sort_values(["threshold_multiplier", "objective_mean"],
                                    ascending=[True, False]).iloc[0]
    elif tie_break == "none":
        chosen = best
    else:
        raise ValueError(f"unknown tie_break {tie_break!r}")
    out = chosen.to_dict()
    out["argmax_multiplier"] = float(best["threshold_multiplier"])
    out["argmax_objective"] = float(best["objective_mean"])
    return out


def _session_folds(sessions: Sequence[Tuple[str, int]], n_splits: Optional[int]):
    """Grouped folds over SESSIONS, deterministic, no shuffling.

    Leave-one-session-out when ``n_splits`` is None and there are few sessions; that is
    the most honest split available with this many groups. Larger session counts fall
    back to k contiguous groups.
    """
    sessions = sorted(set(sessions))
    if n_splits is None:
        n_splits = len(sessions) if len(sessions) <= 12 else 5
    n_splits = max(2, min(n_splits, len(sessions)))
    return [[s for j, s in enumerate(sessions) if j % n_splits == i] for i in range(n_splits)]


def cross_validate(scores: pd.DataFrame, *, objective: str = "f1",
                   tie_break: str = "shallow", n_splits: Optional[int] = None) -> pd.DataFrame:
    """Grouped-by-session CV: pick on train cells, score on held-out cells.

    Each row is one fold: the parameters train chose, what they scored on test, and what
    the best possible test score was (the oracle). A small gap between the two, and the
    same multiplier chosen every fold, is what "not overfit" looks like here.
    """
    df = _usable(scores)
    if df.empty:
        return pd.DataFrame()
    sessions = list({(d, r) for d, r in zip(df["Date"], df["Round No."])})
    folds = _session_folds(sessions, n_splits)

    rows = []
    for i, test_sessions in enumerate(folds, 1):
        if not test_sessions:
            continue
        in_test = pd.Series(list(zip(df["Date"], df["Round No."])), index=df.index).isin(test_sessions)
        train, test = df[~in_test], df[in_test]
        if train.empty or test.empty:
            continue
        picked = select_params(train, objective=objective, tie_break=tie_break)
        if picked is None:
            continue
        mask = np.ones(len(test), dtype=bool)
        for k in PARAM_KEYS:
            mask &= (test[k] == picked[k]).to_numpy()
        test_at_pick = test[mask]
        test_summary = param_summary(test, objective=objective)
        oracle = test_summary.loc[test_summary["objective_mean"].idxmax()] if not test_summary.empty else None
        rows.append({
            "fold": i,
            "n_train_cells": int(train["cell"].nunique()),
            "n_test_cells": int(test["cell"].nunique()),
            "test_sessions": ", ".join(f"{d} r{r}" for d, r in sorted(test_sessions)),
            "picked_noise_method": picked["noise_method"],
            "picked_multiplier": float(picked["threshold_multiplier"]),
            "picked_refractory_ms": float(picked["refractory_ms"]),
            "train_objective": float(picked["objective_mean"]),
            "test_objective": float(test_at_pick[objective].mean()) if not test_at_pick.empty else float("nan"),
            "test_oracle_objective": float(oracle["objective_mean"]) if oracle is not None else float("nan"),
            "test_oracle_multiplier": float(oracle["threshold_multiplier"]) if oracle is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def print_report(scores: pd.DataFrame, cv: pd.DataFrame, final: Optional[dict], *,
                 objective: str = "f1"):
    """The whole answer, in the order it should be read."""
    df = _usable(scores)
    n_cells = df["cell"].nunique() if not df.empty else 0
    n_sessions = len({(d, r) for d, r in zip(df["Date"], df["Round No."])}) if not df.empty else 0
    dropped = (scores["cell"].nunique() - n_cells) if not scores.empty else 0

    print("\n" + "=" * 88)
    print(f" MUA THRESHOLD TUNING — {n_cells} answer-key cell(s) in {n_sessions} session(s), "
          f"objective = {objective}")
    if dropped:
        print(f" ({dropped} cell(s) dropped: fewer than {MIN_ANSWER_KEY_SPIKES} answer-key spikes)")
    print("=" * 88)

    summary = param_summary(scores, objective=objective)
    if not summary.empty:
        print("\n  across all cells:")
        cols = PARAM_KEYS + ["n_cells", "objective_mean", "objective_sem", "recall_mean",
                             "precision_mean", "coincidence_mean", "rate_ratio_median"]
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if not cv.empty:
        print("\n  leave-session-out folds (picked on train, scored on held-out cells):")
        print(cv.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        picks = cv["picked_multiplier"].unique()
        gap = float((cv["test_oracle_objective"] - cv["test_objective"]).mean())
        if len(picks) == 1:
            print(f"\n  every fold picked x{picks[0]:g} — the choice does not depend on which "
                  f"cells it saw.")
        else:
            print(f"\n  folds disagreed: picked {sorted(picks)}. The objective is flat or a few "
                  f"cells drive it;\n  widen the answer key (--all-grant-rows, or --answer-key "
                  f"with more curated cells) before trusting one value.")
        # A gap to the oracle is normally the overfitting signal. It is NOT when every
        # fold's oracle sits DEEPER than the pick: that is the shallow tie-break doing
        # exactly what it was asked to, and calling it overfitting would send you looking
        # for a problem that is a setting.
        oracles = cv["test_oracle_multiplier"].dropna()
        picks = cv["picked_multiplier"]
        biased_shallow = len(oracles) == len(picks) and bool((oracles >= picks).all()) \
            and bool((oracles > picks).any())
        if gap < 0.02:
            print(f"  mean gap to the held-out oracle: {gap:.3f} (small — generalises)")
        elif biased_shallow:
            print(f"  mean gap to the held-out oracle: {gap:.3f} — but every fold's oracle "
                  f"({sorted(oracles.unique())}) is DEEPER than the pick "
                  f"({sorted(picks.unique())}).")
            print("  That gap is the 'shallow' tie-break, not overfitting: it is trading "
                  "agreement for multiunit inclusiveness\n  on purpose. --tie-break none "
                  "picks the raw argmax instead. Draw both and decide by eye "
                  "(analyses.mua_threshold_review).")
        else:
            print(f"  mean gap to the held-out oracle: {gap:.3f} "
                  f"(large — treat the winner as provisional)")

    # Is the answer key big enough? The honest answer is not a cell count -- it is
    # whether the folds agree and the peak is separated from its neighbours.
    if not summary.empty and n_cells:
        top = summary.nlargest(2, "objective_mean")
        separated = (len(top) > 1 and
                     (top.iloc[0]["objective_mean"] - top.iloc[1]["objective_mean"])
                     > top.iloc[0]["objective_sem"])
        folds_agree = (not cv.empty) and cv["picked_multiplier"].nunique() == 1
        if folds_agree and separated:
            print(f"\n  answer key ({n_cells} cells / {n_sessions} sessions) is sufficient: "
                  f"the folds agree AND the peak clears its neighbour by more than one SEM.")
        elif folds_agree:
            print(f"\n  answer key ({n_cells} cells / {n_sessions} sessions): folds agree, but the "
                  f"peak is within one SEM of its neighbour —\n  the objective cannot separate "
                  f"adjacent multipliers. Fine if you only need it to this resolution; add cells "
                  f"if you need more.")
        else:
            print(f"\n  answer key ({n_cells} cells / {n_sessions} sessions) is too thin to settle "
                  f"this: add cells (--all-grant-rows first, then --answer-key with\n  your own "
                  f"curated unsorted channels) and rerun. More SESSIONS help more than more "
                  f"channels — folds are grouped by session.")

    if final:
        print(f"\n  CHOSEN: noise_method={final['noise_method']!r}  "
              f"threshold_multiplier={final['threshold_multiplier']:g}  "
              f"refractory_ms={final['refractory_ms']:g}")
        print(f"    {objective} {final['objective_mean']:.3f} ± {final['objective_sem']:.3f} "
              f"over {int(final['n_cells'])} cells   "
              f"(recall {final['recall_mean']:.2f}, precision {final['precision_mean']:.2f}, "
              f"coincidence {final['coincidence_mean']:.2f}, "
              f"median rate {final['rate_ratio_median']:.2f}x the answer key)")
        if abs(final["threshold_multiplier"] - final["argmax_multiplier"]) > 1e-9:
            print(f"    (raw argmax was x{final['argmax_multiplier']:g} at "
                  f"{final['argmax_objective']:.3f}; the shallower value is within one "
                  f"standard error and keeps more multiunit activity)")
    print("=" * 88 + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_tuning(scores: pd.DataFrame, final: Optional[dict], out_path, *,
                objective: str = "f1"):
    """Objective vs multiplier (per cell and averaged) plus the recall/precision tradeoff.

    The left panel is the evidence for the choice: a peak that most cells share is a real
    optimum, a flat cloud is not. The right panel says what the choice costs -- where
    recall and precision cross, and where the raster stops matching the answer key's
    density.
    """
    import matplotlib.pyplot as plt

    df = _usable(scores)
    if df.empty:
        return None
    summary = param_summary(df, objective=objective)
    methods = sorted(df["noise_method"].unique())
    refrs = sorted(df["refractory_ms"].unique())

    fig, (ax_o, ax_t) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- per-cell curves + mean, for the chosen family ----------------------
    method = final["noise_method"] if final else methods[0]
    refr = final["refractory_ms"] if final else refrs[0]
    fam = df[(df["noise_method"] == method) & (df["refractory_ms"] == refr)]
    for cell, g in fam.groupby("cell"):
        g = g.sort_values("threshold_multiplier")
        ax_o.plot(g["threshold_multiplier"], g[objective], color="0.75", lw=0.8, zorder=1)
    fam_summary = summary[(summary["noise_method"] == method) & (summary["refractory_ms"] == refr)]
    fam_summary = fam_summary.sort_values("threshold_multiplier")
    ax_o.errorbar(fam_summary["threshold_multiplier"], fam_summary["objective_mean"],
                  yerr=fam_summary["objective_sem"], color="tab:blue", lw=2, marker="o",
                  capsize=3, zorder=3, label=f"mean ± SEM ({method}, ref{refr:g})")
    # other families, mean only, so a different noise method stays comparable
    for m in methods:
        for r in refrs:
            if (m, r) == (method, refr):
                continue
            other = summary[(summary["noise_method"] == m) & (summary["refractory_ms"] == r)]
            if other.empty:
                continue
            other = other.sort_values("threshold_multiplier")
            ax_o.plot(other["threshold_multiplier"], other["objective_mean"], lw=1.2, ls="--",
                      marker=".", label=f"mean ({m}, ref{r:g})")
    if final:
        ax_o.axvline(final["threshold_multiplier"], color="crimson", lw=1.4, ls="-",
                     label=f"chosen x{final['threshold_multiplier']:g}")
        if abs(final["threshold_multiplier"] - final["argmax_multiplier"]) > 1e-9:
            ax_o.axvline(final["argmax_multiplier"], color="crimson", lw=1.0, ls=":",
                         label=f"argmax x{final['argmax_multiplier']:g}")
    ax_o.axvline(4.0, color="0.4", lw=1.0, ls="-.", label="current default x4.0")
    ax_o.set_xlabel("threshold multiplier")
    ax_o.set_ylabel(f"{objective} vs answer key")
    ax_o.set_title(f"agreement with the answer key\n(thin grey = one cell, "
                   f"n={fam['cell'].nunique()})", fontsize=10)
    ax_o.legend(fontsize=7, framealpha=0.85)

    # --- the tradeoff -------------------------------------------------------
    fs = fam_summary
    ax_t.plot(fs["threshold_multiplier"], fs["recall_mean"], "o-", label="recall (answer key found)")
    ax_t.plot(fs["threshold_multiplier"], fs["precision_mean"], "s-", label="precision (MUA in answer key)")
    ax_t.plot(fs["threshold_multiplier"], fs["coincidence_mean"], "^-", color="0.5",
              label="coincidence (repo definition)")
    ax_t.plot(fs["threshold_multiplier"], fs["rate_ratio_median"], "d--", color="tab:orange",
              label="median rate ratio (MUA / answer key)")
    ax_t.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if final:
        ax_t.axvline(final["threshold_multiplier"], color="crimson", lw=1.4)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("threshold multiplier")
    ax_t.set_ylabel("fraction / ratio")
    ax_t.set_title("what the choice costs", fontsize=10)
    ax_t.legend(fontsize=7, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def output_dir(monkey=SUBJECT_MONKEY) -> Path:
    return Path(DATA_BASE_PATH) / monkey / "mua_calibration" / "tuning"


def tune(cells: Optional[Sequence[AnswerCell]] = None, *,
         multipliers=DEFAULT_MULTIPLIERS, noise_methods=("mad",), refractory_ms=(1.0,),
         objective: str = "f1", tie_break: str = "shallow", n_splits: Optional[int] = None,
         monkey=SUBJECT_MONKEY, match_window_ms=MATCH_WINDOW_MS,
         all_grant_rows: bool = False, out_dir=None, label: str = "grant_unsorted",
         plots: bool = True, scores: Optional[pd.DataFrame] = None,
         sessions=None, max_sessions: Optional[int] = None,
         exploded_cache_subdir=EXPLODED_CACHE_SUBDIR):
    """Score, cross-validate, report, write. Returns ``(scores, cv, final)``.

    Pass ``scores`` to re-analyse a previous run's per-cell CSV without re-reading any
    recording -- changing the objective or the tie-break costs nothing that way.
    """
    out_dir = Path(out_dir) if out_dir else output_dir(monkey)
    if scores is None:
        cells = cells if cells is not None else load_grant_answer_key(
            n_rows=None if all_grant_rows else GRANT_NCELLS)
        cells = filter_sessions(cells, sessions=sessions, max_sessions=max_sessions)
        if not cells:
            print("[tune] answer key is empty — nothing to tune against.")
            return pd.DataFrame(), pd.DataFrame(), None
        scores = score_answer_key(cells, multipliers=multipliers, noise_methods=noise_methods,
                                  refractory_ms=refractory_ms, monkey=monkey,
                                  match_window_ms=match_window_ms,
                                  exploded_cache_subdir=exploded_cache_subdir)
    if scores.empty:
        print("[tune] no cell could be scored — check the exploded cache and the recordings.")
        return scores, pd.DataFrame(), None

    cv = cross_validate(scores, objective=objective, tie_break=tie_break, n_splits=n_splits)
    final = select_params(scores, objective=objective, tie_break=tie_break)
    print_report(scores, cv, final, objective=objective)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / f"{label}_per_cell_scores.csv", index=False)
    param_summary(scores, objective=objective).to_csv(out_dir / f"{label}_param_summary.csv", index=False)
    if not cv.empty:
        cv.to_csv(out_dir / f"{label}_cv_folds.csv", index=False)
    print(f"  wrote tables to {out_dir}")
    if plots:
        p = plot_tuning(scores, final, out_dir / f"{label}_tuning.png", objective=objective)
        if p:
            print(f"  wrote {p}")
    return scores, cv, final


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answer-key", default=None,
                   help="CSV/xlsx of your own cells (Date, Round No., Cell); "
                        "default is the grant list's unsorted cells")
    p.add_argument("--all-grant-rows", action="store_true",
                   help=f"use every row of the grant sheet, not just the first {GRANT_NCELLS}")
    p.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS))
    p.add_argument("--noise-methods", default="mad", help="'mad', 'rms', or 'mad,rms'")
    p.add_argument("--refractory-ms", default="1.0", help="comma-separated refractory periods")
    p.add_argument("--objective", default="f1", choices=["f1", "coincidence", "recall", "precision"])
    p.add_argument("--tie-break", default="shallow", choices=["shallow", "none"])
    p.add_argument("--splits", type=int, default=None,
                   help="CV folds over sessions (default: leave-one-session-out)")
    p.add_argument("--match-window-ms", type=float, default=MATCH_WINDOW_MS)
    p.add_argument("--exploded-cache", default=EXPLODED_CACHE_SUBDIR,
                   help=f"exploded cache subdir(s) the answer key is read from, comma "
                        f"separated and tried in order (default {EXPLODED_CACHE_SUBDIR}). "
                        f"Append an older cache for cells whose channel has since been "
                        f"manually sorted, e.g. "
                        f"'{EXPLODED_CACHE_SUBDIR},exploded_spike_cache_gitrecovered'")
    p.add_argument("--sessions", default=None,
                   help="restrict to these sessions, e.g. '2023-09-26:2,2023-10-03:4'")
    p.add_argument("--max-sessions", type=int, default=None,
                   help="restrict to the first N sessions — for a quick first run")
    p.add_argument("--rescore", default=None,
                   help="re-analyse a previous *_per_cell_scores.csv (no recording read)")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--no-plots", action="store_false", dest="plots")
    a = p.parse_args(argv)

    scores = pd.read_csv(a.rescore) if a.rescore else None
    cells = load_answer_key_csv(a.answer_key) if (a.answer_key and not a.rescore) else None
    label = a.label or ("rescored" if a.rescore else
                        ("custom" if a.answer_key else
                         ("grant_all_rows" if a.all_grant_rows else "grant_unsorted")))
    return tune(cells,
                multipliers=tuple(float(m) for m in a.multipliers.split(",")),
                noise_methods=tuple(m.strip() for m in a.noise_methods.split(",")),
                refractory_ms=tuple(float(r) for r in a.refractory_ms.split(",")),
                objective=a.objective, tie_break=a.tie_break, n_splits=a.splits,
                match_window_ms=a.match_window_ms, all_grant_rows=a.all_grant_rows,
                out_dir=a.out, label=label, plots=a.plots, scores=scores,
                sessions=_parse_sessions(a.sessions) if a.sessions else None,
                max_sessions=a.max_sessions,
                exploded_cache_subdir=[s.strip() for s in a.exploded_cache.split(",") if s.strip()])


# ===== Run directly in PyCharm — edit this block and hit Run (no CLI) =========
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:                     # called with flags -> behave as a CLI
        _cli()
        raise SystemExit(0)

    # --- the answer key -------------------------------------------------------
    ANSWER_KEY_CSV = None       # path to your own cells (Date, Round No., Cell); None = grant list
    ALL_GRANT_ROWS = False      # True = every row of the grant sheet, not just the first 74.
    #                             The cheapest way to enlarge the answer key — try this first
    #                             if the CV folds disagree.

    # For a quick first run: narrow to a couple of sessions to confirm the answer key
    # resolves and the clocks line up, before waiting for the whole set. Two sessions
    # cannot SELECT a multiplier, but they show you the shape of the curve.
    SESSIONS = None             # e.g. [("2023-09-26", 2), ("2023-10-03", 4)]; None = all
    MAX_SESSIONS = None         # e.g. 2 = the first two sessions in date order

    # Where the answer key's spikes come from:
    #   <JULIE_DATA_PATH>/<monkey>/<cache>/<date>_round_<n>.pkl
    # A list is tried in order, per cell, and each cell records which cache it came from.
    # Append an older cache for cells whose channel has since been MANUALLY SORTED: the
    # merge drops a channel's unsorted whole-channel row once sorted_spikes.pkl names it,
    # so those cells only have an answer key in a cache built before that sort.
    EXPLODED_CACHE = [EXPLODED_CACHE_SUBDIR, "exploded_spike_cache_gitrecovered"]

    # --- the grid to try ------------------------------------------------------
    MULTIPLIERS = DEFAULT_MULTIPLIERS
    NOISE_METHODS = ("mad",)               # ("mad", "rms") to compare noise estimators
    REFRACTORY_MS = (1.0,)                 # e.g. (1.0, 1.5, 2.0)

    # --- how the winner is picked --------------------------------------------
    OBJECTIVE = "f1"            # "f1" (balanced) | "coincidence" | "recall" | "precision".
    #                             Only f1 is peaked: recall and coincidence rise as the
    #                             threshold gets shallower, so maximising either alone
    #                             just picks the bottom of the grid.
    TIE_BREAK = "shallow"       # shallowest multiplier within 1 SEM of the best
    N_SPLITS = None             # None = leave-one-session-out
    PLOTS = True

    # Re-analysing a previous run? Point this at its *_per_cell_scores.csv and no
    # recording is read — changing OBJECTIVE or TIE_BREAK then costs nothing.
    RESCORE_CSV = None

    _scores = pd.read_csv(RESCORE_CSV) if RESCORE_CSV else None
    _cells = load_answer_key_csv(ANSWER_KEY_CSV) if (ANSWER_KEY_CSV and not RESCORE_CSV) else None
    tune(_cells, multipliers=MULTIPLIERS, noise_methods=NOISE_METHODS,
         refractory_ms=REFRACTORY_MS, objective=OBJECTIVE, tie_break=TIE_BREAK,
         n_splits=N_SPLITS, all_grant_rows=ALL_GRANT_ROWS, plots=PLOTS, scores=_scores,
         sessions=SESSIONS, max_sessions=MAX_SESSIONS, exploded_cache_subdir=EXPLODED_CACHE)

    # NEXT STEPS once you have a multiplier M (nothing here writes a spike cache):
    #
    #   1. rebuild the MUA cache with it. Each parameter set gets its own pkl
    #      ('{date}_round_{n}_mad{M}_ref{r}.pkl'), so the existing mad4.0 cache survives:
    #
    #        from data_access.rebuild_peristim_caches import rebuild
    #        rebuild("mua", threshold_multiplier=M)
    #
    #   2. point the answer-key figure at it and LOOK, which is what the score is a proxy
    #      for. In spike_count_connector.py, _mua_source() hardcodes the parameters:
    #
    #        return ThresholdMUASpikeSource(noise_method='mad', threshold_multiplier=M,
    #                                       refractory_ms=1.0)
    #
    #      then run analyses.jun2026_grant_investigation.raster_review_by_source with
    #      MODE="pairs". Lane A is the answer key, lane B the retuned MUA, same channel,
    #      same axes. Put the 4.0 back afterwards unless you mean to move everything that
    #      reads that connector -- the MUA_KW / MUA_ANOVA lists included -- onto M.
    #
    #   3. only then rerun the analyses that read the cache
    #      (run_mua_preprocessing.py -> the MUA_KW / MUA_ANOVA lists).
