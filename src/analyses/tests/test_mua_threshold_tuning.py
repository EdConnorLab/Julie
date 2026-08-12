"""
Tests for the MUA threshold tuner and calibration — no recordings, no DB.

Run either::

    pytest src/analyses/tests/test_mua_threshold_tuning.py
    python  src/analyses/tests/test_mua_threshold_tuning.py

TWO KINDS OF TEST HERE, AND THE FIRST KIND IS THE POINT
-------------------------------------------------------
``mua_threshold_calibration`` deliberately carries its OWN copies of three things that
already exist in the pipeline — the amplifier reshape rule, the peak-to-peak amplitude,
and the trough-to-peak width. They are copies so that adding a calibration tool did not
require editing ``threshold_detection``, ``waveforms`` or ``zombies_raster``, which the
caches, rasters and regressions all read.

The cost of a copy is drift. ``test_amplifier_memmap_matches_the_pipeline_reader`` and
``test_trough_to_peak_matches_the_raster_panel`` are the guard: they run the copy and the
original on the same input and require identical answers. If one of these fails, the
original changed and the copy in ``mua_threshold_calibration`` has to be brought along —
that is the failure telling you where to look, not a bug in the test.

The rest cover the tuner's own logic: trial alignment across two clocks (the stitched
session case), the agreement scoring, and the grouped train/test split.
"""
import os
import sys

import numpy as np
import pandas as pd

# allow running as a plain script (add repo 'src' to path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analyses import mua_threshold_tuning as tuning
from analyses.mua_threshold_tuning import AnswerCell, Trial
from data_access import mua_threshold_calibration as calib
from data_access.threshold_detection import (
    detect_mad_spikes, estimate_noise, read_amplifier_data_robust,
)

SR = 30_000.0


# --------------------------------------------------------------------------- #
# The copies must agree with their originals
# --------------------------------------------------------------------------- #
def test_amplifier_memmap_matches_the_pipeline_reader(tmp_path=None):
    """``calib._open_amplifier_memmap`` must reshape exactly like the cache's reader.

    Including the awkward part: a file with a trailing partial sample, which is the bug
    ``read_amplifier_data_robust`` exists to work around.
    """
    import tempfile
    from clat.intan.channels import Channel

    out_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    names = [c.value for c in list(Channel)[:3]]        # real channel names, whatever they are
    amp_channels = [{"native_channel_name": n} for n in names]
    nch, n_samples = len(names), 100
    data = np.arange(nch * n_samples, dtype=np.int16).reshape(n_samples, nch)

    path = os.path.join(out_dir, "amplifier.dat")
    with open(path, "wb") as f:
        f.write(data.tobytes())
        f.write(np.int16([777]).tobytes())              # trailing partial sample

    mm, channels = calib._open_amplifier_memmap(path, amp_channels)
    assert mm.shape == (n_samples, nch)                 # partial sample dropped
    assert [str(c) for c in channels] == [str(Channel(n)) for n in names]

    original = read_amplifier_data_robust(path, amp_channels)
    for i, ch in enumerate(channels):
        copied = np.asarray(mm[:, i], dtype=np.float64) * calib.MICROVOLTS_PER_BIT
        assert np.array_equal(copied, original[ch]), f"{ch}: copy diverged from the reader"


def test_trough_to_peak_matches_the_raster_panel():
    """``calib.trough_to_peak_ms`` must equal ``zombies_raster._trough_to_peak_ms``.

    Same waveform in, same number out — including every case the gates reject, since a
    disagreement there would show up as a width on one figure and a dash on another.
    """
    from analyses.zombies_raster_review.zombies_raster import _trough_to_peak_ms

    t = np.arange(-15, 30)
    biphasic = -np.exp(-0.5 * (t / 3.0) ** 2) + 0.35 * np.exp(-0.5 * ((t - 9) / 5.0) ** 2)
    cases = [
        biphasic,                                   # the canonical shape
        -biphasic,                                  # positive-going -> rejected
        -np.exp(-0.5 * (t / 3.0) ** 2),             # monophasic -> rejected
        np.array([-5.0, 0.0, 1.0]),                 # trough at the edge -> rejected
        np.array([0.0, -1.0]),                      # too short -> rejected
    ]
    for i, wave in enumerate(cases):
        assert calib.trough_to_peak_ms(wave, SR) == _trough_to_peak_ms(wave, SR), f"case {i}"
    assert calib.trough_to_peak_ms(biphasic, SR) is not None      # the canonical one resolves
    assert calib.trough_to_peak_ms(-biphasic, SR) is None


def test_peak_to_peak():
    assert calib.peak_to_peak([1.0, -2.0, 3.0]) == 5.0
    assert calib.peak_to_peak([]) == 0.0


# --------------------------------------------------------------------------- #
# Trial alignment — the stitched-session case
# --------------------------------------------------------------------------- #
def _answer_rows(onsets, offset_s, duration=1.5, spikes_per_trial=5):
    """Answer-key rows on the COMPILED clock (= amplifier clock - ``offset_s``)."""
    rows = []
    for i, on in enumerate(onsets):
        times = on + np.linspace(0.1, duration - 0.1, spikes_per_trial)
        rows.append({"TaskField": i,
                     "EpochStartStop": (on - offset_s, on + duration - offset_s),
                     "SpikeTimes": list(times - offset_s)})
    return pd.DataFrame(rows)


def test_alignment_cancels_a_stitched_clock_offset():
    """A session whose two clocks differ by 41.7 s must align, not score as noise."""
    onsets = [3.0, 9.0, 15.0]
    epochs = {i: (on, on + 1.5) for i, on in enumerate(onsets)}
    for offset in (0.0, 41.7):
        trials, stats = tuning._align_trials(_answer_rows(onsets, offset), epochs)
        assert stats["n_trials"] == 3
        assert abs(stats["median_clock_offset_s"] - offset) < 1e-9
        # both sides now read 'seconds since onset', so the offset is gone
        for tr, on in zip(trials, onsets):
            assert tr.mua_onset_s == on
            assert np.allclose(tr.key_times, np.linspace(0.1, 1.4, 5))


def test_alignment_drops_trials_it_cannot_place():
    """A task id with no marker epoch, or an epoch of a different length, is dropped."""
    onsets = [3.0, 9.0, 15.0]
    rows = _answer_rows(onsets, 0.0)

    _, stats = tuning._align_trials(rows, {0: (3.0, 4.5)})
    assert stats["n_trials"] == 1 and stats["dropped_no_epoch"] == 2

    wrong_length = {i: (on, on + 9.9) for i, on in enumerate(onsets)}
    _, stats = tuning._align_trials(rows, wrong_length)
    assert stats["n_trials"] == 0 and stats["dropped_duration_mismatch"] == 3

    # no markers at all -> compiled epochs for both sides (right only if unstitched)
    _, stats = tuning._align_trials(rows, None)
    assert stats["n_trials"] == 3 and stats["median_clock_offset_s"] == 0.0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _session_df(channels, n_trials=3):
    """A minimal exploded-cache session frame: one row per (trial, channel)."""
    rows = []
    for i in range(n_trials):
        on = 3.0 + 6.0 * i
        for ch in channels:
            rows.append({"TaskField": i, "Channel": ch,
                         "NeuronID": f"AMG_2023-09-26_2_{ch}",
                         "EpochStartStop": (on, on + 1.5),
                         "SpikeTimes": [on + 0.2, on + 0.4]})
    return pd.DataFrame(rows)


def test_answer_key_rows_resolve_like_the_pairs_figure():
    """Cells must resolve through the same matcher lane A of --mode pairs uses.

    An exact ``str(Channel) ==`` comparison misses the NeuronID-suffix fallback, which is
    what absorbs enum-repr drift between cache vintages -- and missing it looks exactly
    like an empty cache.
    """
    df = _session_df(["Channel.C_020", "Channel.C_026"])
    rows = tuning.select_answer_key_rows(df, "Channel.C_020", "2023-09-26", 2)
    assert rows is not None and len(rows) == 3

    # Channel column spelled differently, NeuronID still carries the canonical name:
    # the fallback must find it where an exact comparison would not.
    drifted = df.copy()
    drifted["Channel"] = drifted["Channel"].str.replace("Channel.C_", "C-", regex=False)
    assert (drifted["Channel"].astype(str) == "Channel.C_020").sum() == 0
    rows = tuning.select_answer_key_rows(drifted, "Channel.C_020", "2023-09-26", 2)
    assert rows is not None and len(rows) == 3


def test_manually_sorted_channel_is_reported_as_such(capsys=None):
    """A channel that has since been sorted is gone from the unsorted rows by
    construction. The message has to say that, not 'not in the exploded cache'."""
    import contextlib
    import io

    df = _session_df(["Channel.C_020_Unit 1", "Channel.C_020_Unit 2", "Channel.C_026"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = tuning.select_answer_key_rows(df, "Channel.C_020", "2023-09-26", 2)
    out = buf.getvalue()
    assert rows is None
    assert "MANUALLY SORTED" in out, out
    assert "Channel.C_020_Unit 1" in out, out

    # a channel that is simply absent gets the other message, listing what is there
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tuning.select_answer_key_rows(df, "Channel.C_007", "2023-09-26", 2) is None
    out = buf.getvalue()
    assert "MANUALLY SORTED" not in out and "channel(s) in this session" in out, out


def test_cache_list_is_tried_in_order_per_cell():
    """A cell resolves from the newest cache that still has it as an unsorted row.

    This is the manually-sorted case end to end: C_020 survives only in the old cache,
    C_026 is in both, and each must come from the right one.
    """
    new = _session_df(["Channel.C_020_Unit 1", "Channel.C_026"])     # C_020 since sorted
    old = _session_df(["Channel.C_020", "Channel.C_026"])            # pre-sort vintage
    frames = {"current": new, "gitrecovered": old}

    resolved = {}
    for channel in ("Channel.C_020", "Channel.C_026"):
        for sub, df in frames.items():
            hit = tuning._match_rows(df, channel, "2023-09-26", 2)
            if not hit.empty:
                resolved[channel] = sub
                break
    assert resolved == {"Channel.C_020": "gitrecovered", "Channel.C_026": "current"}

    assert tuning._as_cache_list("a") == ["a"]
    assert tuning._as_cache_list(["a", "b"]) == ["a", "b"]


def test_score_candidate():
    trial = Trial(task_id=0, key_times=np.array([0.1, 0.2, 0.3]),
                  mua_onset_s=10.0, duration_s=1.0)

    exact = tuning.score_candidate([trial], np.array([10.1, 10.2, 10.3]))
    assert exact["recall"] == 1.0 and exact["precision"] == 1.0
    assert exact["f1"] == 1.0 and exact["rate_ratio"] == 1.0

    # one extra MUA spike: full recall, precision drops -- the multiunit case
    extra = tuning.score_candidate([trial], np.array([10.1, 10.2, 10.3, 10.9]))
    assert extra["recall"] == 1.0 and abs(extra["precision"] - 0.75) < 1e-9
    assert abs(extra["coincidence"] - 1.0) < 1e-9      # matched fraction of the SMALLER train
    assert extra["f1"] < exact["f1"]                    # ... which f1 notices and coincidence does not

    assert tuning.score_candidate([trial], np.array([]))["f1"] == 0.0
    # a spike in a different trial must never match across the boundary
    assert tuning.score_candidate([trial], np.array([50.1, 50.2]))["n_mua"] == 0


def test_f1_peaks_while_recall_and_coincidence_do_not():
    """The reason the objective is f1: on a real trace recall only ever rises as the
    threshold gets shallower, so maximising it walks to the bottom of the grid."""
    rng = np.random.default_rng(3)
    n = int(SR * 20)
    v = rng.normal(0.0, 10.0, n)
    t = np.arange(-15, 30)
    tmpl = -np.exp(-0.5 * (t / 3.0) ** 2) + 0.35 * np.exp(-0.5 * ((t - 9) / 5.0) ** 2)
    tmpl /= -tmpl.min()
    pos = np.sort(rng.choice(np.arange(200, n - 200, int(0.004 * SR)), 2000, replace=False))
    amps = 15.0 + rng.exponential(50.0, 2000)
    for p, a in zip(pos, amps):
        v[p - 15:p - 15 + tmpl.size] += tmpl * a

    sigma = estimate_noise(v, "mad")
    true_mult = 6.0
    key = pos[amps >= true_mult * sigma] / SR              # the 'online' detection
    trials = [Trial(task_id=0, key_times=key, mua_onset_s=0.0, duration_s=n / SR)]

    mults = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    scored = [tuning.score_candidate(trials, detect_mad_spikes(v, -m * sigma, 30) / SR)
              for m in mults]
    f1 = [s["f1"] for s in scored]
    recall = [s["recall"] for s in scored]

    assert all(b <= a + 1e-9 for a, b in zip(recall, recall[1:])), "recall must be monotone"
    peak = mults[int(np.argmax(f1))]
    assert abs(peak - true_mult) <= 1.0, f"f1 peaked at {peak}, true {true_mult}"
    assert peak not in (mults[0], mults[-1]), "f1 must peak inside the grid, not at an end"


# --------------------------------------------------------------------------- #
# Selection and the grouped split
# --------------------------------------------------------------------------- #
def _fake_scores():
    """Per-cell scores over 4 sessions x 2 cells, f1 peaking at x6 with a flat shoulder."""
    curve = {3.0: 0.40, 4.0: 0.62, 5.0: 0.80, 6.0: 0.88, 7.0: 0.879, 8.0: 0.70}
    rng = np.random.default_rng(0)
    rows = []
    for s in range(4):
        for c in range(2):
            for mult, f1 in curve.items():
                rows.append({"Date": f"2023-09-2{s}", "Round No.": 1,
                             "Channel": f"Channel.C_00{c}", "cell": f"s{s}_c{c}",
                             "noise_method": "mad", "threshold_multiplier": mult,
                             "refractory_ms": 1.0, "usable": True,
                             "f1": f1 + rng.normal(0, 0.004),
                             "recall": 1.2 - 0.1 * mult, "precision": 0.1 * mult,
                             "coincidence": 0.9, "rate_ratio": 8.0 / mult})
    return pd.DataFrame(rows)


def test_tie_break_prefers_the_shallower_threshold():
    """x6 and x7 are within a standard error; the shallower one keeps more multiunit."""
    scores = _fake_scores()
    shallow = tuning.select_params(scores, tie_break="shallow")
    raw = tuning.select_params(scores, tie_break="none")
    assert shallow["threshold_multiplier"] <= raw["threshold_multiplier"]
    assert shallow["threshold_multiplier"] == 6.0
    assert shallow["argmax_multiplier"] in (6.0, 7.0)


def test_cross_validation_is_grouped_by_session():
    scores = _fake_scores()
    cv = tuning.cross_validate(scores)
    assert len(cv) == 4                                    # 4 sessions -> leave-one-session-out
    for _, fold in cv.iterrows():
        assert fold["n_test_cells"] == 2                   # both cells of one session
        assert len(fold["test_sessions"].split(", ")) == 1  # no session split across train/test
    assert cv["picked_multiplier"].nunique() == 1          # stable choice


def test_oracle_gap_is_attributed_to_the_tie_break_not_to_overfitting():
    """When every fold's oracle is DEEPER than the pick, the gap is the shallow
    tie-break's doing. Reporting that as overfitting sends you hunting a setting."""
    import contextlib
    import io

    scores = _fake_scores()
    cv = tuning.cross_validate(scores)
    # the shoulder at x7 makes the held-out oracle land deeper than the x6 pick
    cv["test_oracle_multiplier"] = cv["picked_multiplier"] + 1.0
    cv["test_objective"] = cv["test_oracle_objective"] - 0.05
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tuning.print_report(scores, cv, tuning.select_params(scores))
    out = buf.getvalue()
    assert "is the 'shallow' tie-break, not overfitting" in out, out
    assert "--tie-break none" in out

    # an oracle SHALLOWER than the pick is not that pattern, and must not claim it is
    cv["test_oracle_multiplier"] = cv["picked_multiplier"] - 1.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tuning.print_report(scores, cv, tuning.select_params(scores))
    assert "not overfitting" not in buf.getvalue()


def test_unusable_cells_are_excluded_from_the_fit():
    scores = _fake_scores()
    scores.loc[scores["cell"] == "s0_c0", "usable"] = False
    assert tuning.param_summary(scores)["n_cells"].max() == 7


def test_session_folds():
    assert len(tuning._session_folds([("a", 1), ("b", 2), ("c", 3)], None)) == 3   # LOSO
    many = tuning._session_folds([(f"s{i}", 1) for i in range(20)], None)
    assert len(many) == 5 and sum(len(f) for f in many) == 20                      # k-fold
    assert len(tuning._session_folds([("a", 1), ("b", 2)], 5)) == 2                # clamped


def test_review_lanes_land_on_the_amplifier_clock():
    """Both raster lanes must carry AMPLIFIER-clock times and epochs.

    The raster alone would work on either clock (it re-zeroes by each row's
    EpochStartStop), but the footprint panel cuts snippets from the continuous recording
    at raw spike times -- so compiled-clock times on a stitched session would silently
    produce a waveform made of noise.
    """
    from analyses import mua_threshold_review as review

    offset = 37.4                                  # compiled clock = amplifier - offset
    onsets = [3.0, 10.0, 17.0]
    rows = pd.DataFrame([
        {"TaskField": i, "Channel": "Channel.C_005", "MonkeyName": f"M{i}",
         "MonkeyGroup": "Zombies",
         "EpochStartStop": (on - offset, on + 1.5 - offset),
         "SpikeTimes": [on - offset + 0.2, on - offset + 0.9]}
        for i, on in enumerate(onsets)])
    epochs = {i: (on, on + 1.5) for i, on in enumerate(onsets)}
    trials, stats = tuning._align_trials(rows, epochs)
    assert abs(stats["median_clock_offset_s"] - offset) < 1e-9

    mua = np.array([on + d for on in onsets for d in (0.2, 0.5, 0.9)])
    lane_a, lane_b = review._lane_frames(rows, trials, mua)
    for lane in (lane_a, lane_b):
        for (start, stop), on in zip(lane["EpochStartStop"], onsets):
            assert (start, stop) == (on, on + 1.5)
        for spikes, on in zip(lane["SpikeTimes"], onsets):
            assert all(on <= s < on + 1.5 for s in spikes), spikes
    # trial identity is shared, so the overlay matches lanes by task id, not row order
    assert list(lane_a["TaskField"]) == list(lane_b["TaskField"])
    assert list(lane_a["MonkeyName"]) == list(lane_b["MonkeyName"])
    # lane B is tagged multiunit, not "(unsorted)" -- it is not a second copy of lane A
    from analyses.zombies_raster_review.run_zombies_rasters import _lane_label
    assert _lane_label(review.lane_b_prefix("mad", 5.5), "Channel.C_005").endswith("(MU)")
    # the folder name is the MUA cache filename's parameter token, verbatim
    assert review.param_dirname("mad", 5.5, 1.0) == "mad5.5_ref1.0"


def test_filter_sessions():
    cells = [AnswerCell("2023-09-26", 2, "Channel.C_005"),
             AnswerCell("2023-09-26", 2, "Channel.C_011"),
             AnswerCell("2023-10-03", 4, "Channel.C_026")]
    assert len(tuning.filter_sessions(cells)) == 3
    assert len(tuning.filter_sessions(cells, max_sessions=1)) == 2
    assert tuning.filter_sessions(cells, sessions=[("2023-10-03", 4)]) == cells[2:]


def test_answer_key_rejects_sorted_units(tmp_path=None):
    """A sorted unit cannot be reproduced by a channel-level detector, so it is refused."""
    import tempfile
    out_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    path = os.path.join(out_dir, "cells.csv")
    pd.DataFrame([{"Date": "2023-09-26", "Round No.": 2, "Cell": "Channel.C_020_Unit 1"}]
                 ).to_csv(path, index=False)
    try:
        tuning.load_answer_key_csv(path)
    except ValueError as e:
        assert "sorted unit" in str(e)
    else:
        raise AssertionError("a sorted unit should have been refused")


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn(tempfile.mkdtemp()) if "tmp_path" in fn.__code__.co_varnames else fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {failures} failure(s)")
    raise SystemExit(1 if failures else 0)
