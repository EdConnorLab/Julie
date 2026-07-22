"""
synthetic_smoke_test.py — verify the harness end-to-end WITHOUT the lab caches.

Builds a synthetic unit that has BOTH a pre-stimulus baseline (spikes before
onset) and a KNOWN response window, runs every detector, and checks that:

* each detector returns a DetectorResult without error,
* the vs-baseline detectors (which now use the real pre-stim window) recover a
  window overlapping the truth,
* a no-response unit yields no cluster-permutation window,
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
PRE = 1.0
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "smoke")


def _make_prestim_unit(window_s=TRUE_WINDOW, *, baseline_hz=5.0, peak_hz=45.0, seed=1):
    """A synthetic unit with a real pre-stimulus baseline.

    ``make_synthetic_zombies_unit`` fills only [0, epoch] (onset = EpochStartStop[0]
    = 0); we prepend Poisson baseline spikes in [-PRE, 0) per trial so
    extract_trials(pre_stimulus_time=PRE) sees a genuine pre-stim window.
    """
    df = make_synthetic_zombies_unit(window_s=window_s, peak_resp_hz=peak_hz,
                                     baseline_hz=baseline_hz, seed=seed).copy()
    rng = np.random.default_rng(seed + 7)
    spikes = []
    for sp in df["SpikeTimes"]:
        sp = np.asarray(sp, dtype=float)
        n_base = rng.poisson(baseline_hz * PRE)
        pre_sp = rng.uniform(-PRE, 0.0, size=n_base)
        spikes.append(np.sort(np.concatenate([pre_sp, sp])))
    df["SpikeTimes"] = spikes
    return df


def _overlaps(windows, truth):
    return any(min(w[1], truth[1]) - max(w[0], truth[0]) > 0 for w in windows)


def test_detectors_recover_known_window():
    td = extract_trials(_make_prestim_unit(seed=1), t_stop=2.4, pre_stimulus_time=PRE)
    assert td.n_trials > 0 and td.has_baseline
    results = {d.name: d.detect(td) for d in default_detectors()}

    for name, res in results.items():
        assert res is not None, f"{name} returned None"
        for lo, hi in res.windows:
            assert 0.0 <= lo < hi <= 2.4 + 1e-6, f"{name} window out of range: {(lo, hi)}"

    for name in ("baseline_z", "poisson", "cluster_perm"):
        assert _overlaps(results[name].windows, TRUE_WINDOW), \
            f"{name} missed the injected window {TRUE_WINDOW}: got {results[name].windows}"
    return results


def test_no_response_unit_is_mostly_quiet():
    td = extract_trials(_make_prestim_unit(window_s=None, peak_hz=0.0, baseline_hz=6.0, seed=2),
                        t_stop=2.4, pre_stimulus_time=PRE)
    res = {d.name: d.detect(td) for d in default_detectors()}
    assert len(res["cluster_perm"].windows) == 0, \
        f"cluster_perm false-positived on flat data: {res['cluster_perm'].windows}"


def test_baseline_detector_needs_prestim():
    """Without a pre-stim period, the vs-baseline detectors must refuse clearly."""
    from analyses.response_window_benchmark.detectors import BaselineZScoreDetector
    td = extract_trials(make_synthetic_zombies_unit(window_s=TRUE_WINDOW, seed=3),
                        t_stop=2.4, pre_stimulus_time=0.0)
    try:
        BaselineZScoreDetector().detect(td)
        assert False, "expected a ValueError when no pre-stim baseline is available"
    except ValueError:
        pass


def test_optimal_bin_width_positive():
    td = extract_trials(_make_prestim_unit(seed=3), t_stop=2.4, pre_stimulus_time=PRE)
    best, cands, costs = optimal_bin_width(td)
    assert best > 0 and np.isfinite(costs).any()


def test_plot_and_score_roundtrip():
    df = _make_prestim_unit(seed=4)
    td = extract_trials(df, t_stop=2.4, pre_stimulus_time=PRE)
    detectors = default_detectors()
    results = {d.name: d.detect(td) for d in detectors}
    fig = plot_cell_with_windows(
        df, detectors, results, truth_windows=[TRUE_WINDOW],
        title="synthetic smoke test (pre-stim 1000 ms)", pre_stimulus_time=PRE,
        save_path=os.path.join(_OUT, "synthetic_comparison.png"))
    assert fig is not None or os.path.exists(os.path.join(_OUT, "synthetic_comparison.png"))

    preds = {"cellA": {name: r.windows for name, r in results.items()}}
    truth = {"cellA": [TRUE_WINDOW]}
    per_cell, scoreboard = scoring.score_all(preds, truth, [d.name for d in detectors])
    assert not scoreboard.empty and "F1" in scoreboard.columns


def main():
    print("Running synthetic smoke test (with pre-stimulus baseline)...")
    results = test_detectors_recover_known_window()
    print("\nDetected windows on synthetic cell (truth = 100-450 ms, baseline = pre-stim):")
    for name, res in results.items():
        wins = ", ".join(f"{lo*1000:.0f}-{hi*1000:.0f}ms" for lo, hi in res.windows) or "(none)"
        print(f"  {name:14} {wins}")
    test_no_response_unit_is_mostly_quiet()
    test_baseline_detector_needs_prestim()
    test_optimal_bin_width_positive()
    test_plot_and_score_roundtrip()
    print(f"\nAll smoke checks passed. Figure -> {_OUT}/synthetic_comparison.png")


if __name__ == "__main__":
    main()
