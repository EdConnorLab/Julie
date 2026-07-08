"""
Smoke tests for the Zombies raster review — no DB / recordings required.

Run either::

    pytest src/analyses/zombies_raster_review/tests/test_smoke.py
    python  src/analyses/zombies_raster_review/tests/test_smoke.py
"""
import os
import sys
import tempfile

# allow running as a plain script (add repo 'src' to path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analyses.zombies_raster_review.unit_lists import (
    load_mixed_manual_requests, load_si_sorted_requests,
)
from analyses.zombies_raster_review.make_synthetic_zombies_session import (
    make_synthetic_zombies_unit, make_synthetic_overlay_pair,
    make_synthetic_cross_source_session,
)
from analyses.zombies_raster_review.zombies_raster import (
    zombies_trials_by_monkey, plot_zombies_raster, plot_overlay_raster,
    _window_label_ms, _unit_channel_token, SUBJECT_MONKEY_ID,
)
from analyses.zombies_raster_review.coincidence_match import (
    session_units, unit_spike_train, match_units_across_sources, group_matches,
)
from analyses.zombies_raster_review.run_zombies_rasters import (
    _select_unit_rows, _units_on_channel, _channel_token,
    render_overlays_for_units,
)

_LISTS = os.path.join(os.path.dirname(__file__), "..", "unit_lists")


def test_parse_mixed_list():
    reqs = load_mixed_manual_requests(
        os.path.join(_LISTS, "zombies_mixed_manual_anova_passed.xlsx"))
    assert len(reqs) > 0
    r = reqs[0]
    assert r.source_kind == "mixed"
    assert r.match_column == "Channel"
    assert r.match_value.startswith("Channel.")
    # every date normalised to YYYY-MM-DD
    assert all(len(x.date) == 10 and x.date[4] == "-" for x in reqs)
    # windows parse to (lo, hi) ms
    assert any(x.window_ms is not None for x in reqs)


def test_parse_si_list():
    reqs = load_si_sorted_requests(
        os.path.join(_LISTS, "zombies_si_sorted_anova_passed.csv"))
    assert len(reqs) > 0
    r = reqs[0]
    assert r.source_kind == "si"
    assert r.match_column == "NeuronID"
    assert r.window_ms is not None and r.window_ms[1] > r.window_ms[0]


def test_trials_grouped_and_ranked():
    df = make_synthetic_zombies_unit()
    groups = zombies_trials_by_monkey(df)
    assert len(groups) >= 8
    # ranked monkeys come out in ascending rank order (1,2,3,...)
    ranks = [g.rank for g in groups if g.rank is not None]
    assert ranks == sorted(ranks)
    assert all(len(g.trials) > 0 for g in groups)


def test_select_unit_rows_mixed_and_si():
    df = make_synthetic_zombies_unit()
    # SI-style match on NeuronID
    from analyses.zombies_raster_review.unit_lists import RasterRequest
    si_req = RasterRequest("si", "2023-09-26", 2, "NeuronID",
                           "AMG_2023-09-26_2_Channel.C_018_Unit 1", "lbl")
    assert not _select_unit_rows(df, si_req).empty
    # Mixed-style match on Channel string
    mixed_req = RasterRequest("mixed", "2023-09-26", 2, "Channel",
                              "Channel.C_018_Unit 1", "lbl")
    assert not _select_unit_rows(df, mixed_req).empty


def test_subject_monkey_excluded():
    # subject (81G) is present in the data but must never appear in the raster
    df = make_synthetic_zombies_unit(include_subject=True)
    assert (df["MonkeyName"] == SUBJECT_MONKEY_ID).any(), "fixture should include 81G"
    groups = zombies_trials_by_monkey(df)
    assert all(g.monkey != SUBJECT_MONKEY_ID for g in groups), "81G leaked into raster"
    # all remaining monkeys are ranked (81G was the only unranked one)
    assert all(g.rank is not None for g in groups)


def test_plot_writes_png():
    df = make_synthetic_zombies_unit()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.png")
        fig = plot_zombies_raster(df, neuron_label="synthetic unit",
                                  window_s=(0.1, 0.45), p_value=0.004,
                                  save_path=path)
        assert fig is not None
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_channel_token_and_units_on_channel():
    assert _channel_token("Channel.C_018_Unit 1") == "C_018"
    assert _channel_token("C-018") == "C_018"
    df = make_synthetic_zombies_unit()
    units = _units_on_channel(df, "C_018")
    assert len(units) >= 1


def test_overlay_writes_png():
    pair = make_synthetic_overlay_pair()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "overlay.png")
        fig = plot_overlay_raster(pair, title="manual vs SI",
                                  window_s=(0.1, 0.45), save_path=path)
        assert fig is not None
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_window_label_ms():
    # (s) -> "ms" label used on top of the shaded window
    assert _window_label_ms((0.3, 0.4)) == "300–400 ms"
    assert _window_label_ms((0.1, 0.45)) == "100–450 ms"


def test_cross_source_coincidence_matches_across_channels():
    # same neuron is on C_011 (mixed) and C_020 (SI); channel names differ, so
    # only a coincidence match can pair them. Decoys must not match across sorts.
    mixed_df, si_df = make_synthetic_cross_source_session()
    units_mixed = session_units(mixed_df, "Channel")
    units_si = session_units(si_df, "NeuronID")
    assert "Channel.C_011_Unit 1" in units_mixed
    assert any("C_020_Unit 1" in k for k in units_si)

    matches = match_units_across_sources(units_mixed, units_si)
    assert matches, "expected at least the true cross-channel match"
    top = matches[0]
    assert "C_011" in top.mixed_id and "C_020" in top.si_id
    assert top.coincidence > 0.5 and top.ratio_over_chance > 5.0
    # the independent decoys (C_005 mixed, C_007 SI) must NOT be matched together
    assert not any(("C_005" in m.mixed_id and "C_007" in m.si_id) for m in matches)


def test_group_matches_bundles_connected_units():
    mixed_df, si_df = make_synthetic_cross_source_session()
    matches = match_units_across_sources(
        session_units(mixed_df, "Channel"), session_units(si_df, "NeuronID"))
    groups = group_matches(matches)
    assert groups
    g = groups[0]
    assert any("C_011" in c for c in g.mixed_ids)
    assert any("C_020" in s for s in g.si_ids)


def test_render_overlays_for_units_writes_png():
    mixed_df, si_df = make_synthetic_cross_source_session()
    units_mixed = session_units(mixed_df, "Channel")
    units_si = session_units(si_df, "NeuronID")
    with tempfile.TemporaryDirectory() as tmp:
        written = render_overlays_for_units(
            units_mixed, units_si, tmp, label="test",
            si_anchor_windows={"AMG_2023-09-26_2_Channel.C_020_Unit 1": (0.1, 0.45)},
        )
        assert len(written) >= 1
        for p in written:
            assert os.path.exists(p) and os.path.getsize(p) > 0


def test_listed_only_toggle():
    # listed_only restricts the overlay to cells that are in the lists (both
    # sides); the default also brings in a listed cell's unlisted cross-sort twin
    mixed_df, si_df = make_synthetic_cross_source_session()
    um = session_units(mixed_df, "Channel")
    us = session_units(si_df, "NeuronID")
    si_anchor = {"AMG_2023-09-26_2_Channel.C_020_Unit 1": (0.1, 0.45)}
    with tempfile.TemporaryDirectory() as tmp:
        # default: C_020 (SI, listed) drags in its unlisted mixed twin C_011
        w_all = render_overlays_for_units(um, us, tmp, label="all",
                                          si_anchor_windows=si_anchor)
        assert len(w_all) == 1
        # listed_only: C_011 isn't listed, so there's no cross-sort pair to draw
        w_listed = render_overlays_for_units(um, us, tmp, label="listed",
                                             si_anchor_windows=si_anchor,
                                             listed_only=True)
        assert len(w_listed) == 0
        # listed_only with BOTH sides listed: the pair overlays
        w_both = render_overlays_for_units(
            um, us, tmp, label="both", listed_only=True,
            mixed_anchor_windows={"Channel.C_011_Unit 1": (0.1, 0.45)},
            si_anchor_windows=si_anchor)
        assert len(w_both) == 1


def test_unit_channel_token():
    df = make_synthetic_zombies_unit(channel="Channel.C_011_Unit 1",
                                     neuron_id="AMG_2023-09-26_2_Channel.C_020_Unit 1")
    # Channel column wins; "_Unit" suffix and "Channel." prefix are stripped
    assert _unit_channel_token(df) == "C_011"


def test_overlay_probe_toggle_both_render():
    # the probe-map panel (default on) and the plain layout both produce a figure
    pair = make_synthetic_overlay_pair()
    with tempfile.TemporaryDirectory() as tmp:
        for flag, name in ((True, "with_probe.png"), (False, "no_probe.png")):
            path = os.path.join(tmp, name)
            fig = plot_overlay_raster(pair, title="probe toggle", window_s=(0.1, 0.45),
                                      show_probe=flag, save_path=path)
            assert fig is not None
            assert os.path.exists(path) and os.path.getsize(path) > 0


if __name__ == "__main__":
    test_parse_mixed_list()
    test_parse_si_list()
    test_trials_grouped_and_ranked()
    test_select_unit_rows_mixed_and_si()
    test_subject_monkey_excluded()
    test_plot_writes_png()
    test_channel_token_and_units_on_channel()
    test_overlay_writes_png()
    test_window_label_ms()
    test_cross_source_coincidence_matches_across_channels()
    test_group_matches_bundles_connected_units()
    test_render_overlays_for_units_writes_png()
    test_listed_only_toggle()
    test_unit_channel_token()
    test_overlay_probe_toggle_both_render()
    print("All zombies_raster_review smoke tests passed.")
