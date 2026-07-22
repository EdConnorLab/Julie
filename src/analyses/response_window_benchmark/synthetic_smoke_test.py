"""
synthetic_smoke_test.py — verify the harness end-to-end WITHOUT the lab caches.

Builds a synthetic unit with a KNOWN response window (via the existing
``make_synthetic_zombies_session`` helper), runs every detector, and checks that:

* each detector returns a DetectorResult without error,
* the principled vs-baseline detectors recover a window overlapping the truth,
* a no-response unit yields (mostly) empty detections,
* the figure renders and the template/scoring round-trips.

Run:  ``python -m analyses.response_window_benchmark.synthetic_smoke_test``
Also importable as pytest (functions named ``test_*``).
"""
from __future__ import annotations

import os

import numpy as np

from analyses.zombies_raster_review.make_synthetic_zombies_session import (
    make_synthetic_zombies_unit,
)
from analyses.response_window_benchmark.psth import extract_trials, optimal_bin_width
from analyses.response_window_benchmark.detectors import default_detectors
from analyses.response_window_benchmark.plotting import plot_cell_with_windows
from analyses.response_window_benchmark import scoring

TRUE_WINDOW = (0.1, 0.45)
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "smoke")


def _overlaps(windows, truth):
    return any(min(w[1], truth[1]) - max(w[0], truth[0]) > 0 for w in windows)


def test_detectors_recover_known_window():
    df = make_synthetic_zombies_unit(window_s=TRUE_WINDOW, peak_resp_hz=45.0,
                                     baseline_hz=4.0, seed=1)
    td = extract_trials(df, t_stop=2.4)
    assert td.n_trials > 0
    results = {d.name: d.detect(td) for d in default_detectors()}

    for name, res in results.items():
        assert res is not None, f"{name} returned None"
        for lo, hi in res.windows:
            assert 0.0 <= lo < hi <= 2.4 + 1e-6, f"{name} window out of range: {(lo, hi)}"

    # the principled vs-baseline methods must find the injected response
    for name in ("baseline_z", "poisson", "cluster_perm"):
        assert _overlaps(results[name].windows, TRUE_WINDOW), \
            f"{name} missed the injected window {TRUE_WINDOW}: got {results[name].windows}"
    return results


def test_no_response_unit_is_mostly_quiet():
    df = make_synthetic_zombies_unit(window_s=None, peak_resp_hz=0.0,
                                     baseline_hz=6.0, seed=2)
    td = extract_trials(df, t_stop=2.4)
    res = {d.name: d.detect(td) for d in default_detectors()}
    # cluster-permutation controls false positives -> expect no window on flat data
    assert len(res["cluster_perm"].windows) == 0, \
        f"cluster_perm false-positived on flat data: {res['cluster_perm'].windows}"


def test_optimal_bin_width_positive():
    df = make_synthetic_zombies_unit(window_s=TRUE_WINDOW, seed=3)
    td = extract_trials(df, t_stop=2.4)
    best, cands, costs = optimal_bin_width(td)
    assert best > 0 and np.isfinite(costs).any()


def test_plot_and_score_roundtrip():
    df = make_synthetic_zombies_unit(window_s=TRUE_WINDOW, seed=4)
    td = extract_trials(df, t_stop=2.4)
    detectors = default_detectors()
    results = {d.name: d.detect(td) for d in detectors}
    fig = plot_cell_with_windows(
        df, detectors, results, truth_windows=[TRUE_WINDOW],
        title="synthetic smoke test",
        save_path=os.path.join(_OUT, "synthetic_comparison.png"))
    assert fig is not None or os.path.exists(os.path.join(_OUT, "synthetic_comparison.png"))

    # scoring round-trip
    preds = {"cellA": {name: r.windows for name, r in results.items()}}
    truth = {"cellA": [TRUE_WINDOW]}
    per_cell, scoreboard = scoring.score_all(preds, truth, [d.name for d in detectors])
    assert not scoreboard.empty and "F1" in scoreboard.columns


def main():
    print("Running synthetic smoke test...")
    results = test_detectors_recover_known_window()
    print("\nDetected windows on synthetic cell (truth = 100-450 ms):")
    for name, res in results.items():
        wins = ", ".join(f"{lo*1000:.0f}-{hi*1000:.0f}ms" for lo, hi in res.windows) or "(none)"
        print(f"  {name:14} {wins}")
    test_no_response_unit_is_mostly_quiet()
    test_optimal_bin_width_positive()
    test_plot_and_score_roundtrip()
    print(f"\nAll smoke checks passed. Figure -> {_OUT}/synthetic_comparison.png")


if __name__ == "__main__":
    main()
