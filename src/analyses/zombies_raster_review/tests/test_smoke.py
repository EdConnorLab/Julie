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
    make_synthetic_zombies_unit,
)
from analyses.zombies_raster_review.zombies_raster import (
    zombies_trials_by_monkey, plot_zombies_raster,
)
from analyses.zombies_raster_review.run_zombies_rasters import _select_unit_rows

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


def test_plot_writes_png():
    df = make_synthetic_zombies_unit()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.png")
        fig = plot_zombies_raster(df, neuron_label="synthetic unit",
                                  window_s=(0.1, 0.45), p_value=0.004,
                                  save_path=path)
        assert fig is not None
        assert os.path.exists(path) and os.path.getsize(path) > 0


if __name__ == "__main__":
    test_parse_mixed_list()
    test_parse_si_list()
    test_trials_grouped_and_ranked()
    test_select_unit_rows_mixed_and_si()
    test_plot_writes_png()
    print("All zombies_raster_review smoke tests passed.")
