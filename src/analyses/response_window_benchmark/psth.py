"""
psth.py — shared spike-shaping utilities for the detector bake-off.

Turns a single unit's per-trial DataFrame (the shape a ``SpikeSource.load``
returns) into the arrays the detectors operate on:

* :func:`extract_trials`      per-trial spike times aligned to stimulus onset,
                              KEEPING the pre-stimulus period (negative times)
* :func:`binned_matrix`       (n_trials x n_bins) counts over the POST-stim window
* :func:`prestim_counts`      (n_trials x n_bins) counts over the PRE-stim window
* :func:`prestim_baseline`    baseline firing from the real pre-stimulus period
* :func:`trial_averaged_rate` mean rate (Hz) +/- SEM over any [lo, hi]
* :func:`optimal_bin_width`   Shimazaki & Shinomoto (2007) MISE-optimal PSTH bin

Baseline
--------
``EpochStartStop[0]`` is stimulus onset; the SI-sorted pre-stimulus caches
(e.g. ``sorted_spike_cache_pre1000ms``) store spikes from ``onset - pre`` onward,
so once aligned (``s - onset``) the pre-stimulus spikes are negative, in
``[-pre, 0)``. The baseline is measured DIRECTLY from that window — there is no
within-trial baseline estimation. Detectors that need a baseline require
``pre_stimulus_time > 0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analyses.zombies_raster_review.zombies_raster import GROUP


@dataclass
class TrialData:
    """One unit's trials, aligned to stimulus onset (onset = 0).

    ``trials`` may contain NEGATIVE spike times: those are the pre-stimulus
    baseline, present when the unit was loaded from a pre-stim cache. The
    response-detection window is ``[0, t_stop]``; the baseline window is
    ``[-pre_stimulus_time, 0)``.
    """
    trials: List[np.ndarray]
    labels: np.ndarray
    t_stop: float
    pre_stimulus_time: float = 0.0
    neuron_id: str = ""

    @property
    def t_start(self) -> float:
        return -self.pre_stimulus_time

    @property
    def has_baseline(self) -> bool:
        return self.pre_stimulus_time > 0.0

    @property
    def n_trials(self) -> int:
        return len(self.trials)


def extract_trials(
    neuron_df: pd.DataFrame,
    *,
    spike_col: str = "SpikeTimes",
    group: str = GROUP,
    t_stop: Optional[float] = None,
    pre_stimulus_time: float = 0.0,
) -> TrialData:
    """Per-trial spikes for one unit, aligned to onset, keeping ``pre_stimulus_time``
    seconds of baseline before onset (as negative times).

    Mirrors ``raster_plotting._align_spikes_to_epoch``: keep spikes in
    ``[start - pre_stimulus_time, stop]`` and re-zero to ``start`` (onset).
    """
    df = neuron_df[neuron_df["MonkeyGroup"] == group]
    trials: List[np.ndarray] = []
    labels: List[str] = []
    durations: List[float] = []
    for _, r in df.iterrows():
        start, stop = r["EpochStartStop"]
        s = np.asarray(list(r[spike_col]), dtype=float)
        s = s[(s >= start - pre_stimulus_time) & (s <= stop)]
        trials.append(s - start)
        labels.append(r["MonkeyName"])
        durations.append(stop - start)

    if t_stop is None:
        t_stop = float(min(durations)) if durations else 0.0

    nid = str(df["NeuronID"].iloc[0]) if ("NeuronID" in df.columns and len(df)) else ""
    return TrialData(trials=trials, labels=np.asarray(labels, dtype=object),
                     t_stop=float(t_stop), pre_stimulus_time=float(pre_stimulus_time),
                     neuron_id=nid)


def bin_edges(lo: float, hi: float, bin_s: float) -> np.ndarray:
    """Left-closed bin edges covering [lo, hi] with width ``bin_s``."""
    n = max(int(round((hi - lo) / bin_s)), 1)
    return lo + bin_s * np.arange(n + 1)


def _counts(td: TrialData, bin_s: float, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = bin_edges(lo, hi, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if td.n_trials == 0:
        return np.zeros((0, centers.size)), centers
    counts = np.vstack([np.histogram(t, bins=edges)[0] for t in td.trials]).astype(float)
    return counts, centers


def binned_matrix(td: TrialData, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(n_trials x n_bins) spike-count matrix over the POST-stim window [0, t_stop]."""
    return _counts(td, bin_s, 0.0, td.t_stop)


def prestim_counts(td: TrialData, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(n_trials x n_bins) spike-count matrix over the PRE-stim window [-pre, 0)."""
    return _counts(td, bin_s, -td.pre_stimulus_time, 0.0)


def analysis_edges(td: TrialData, bin_s: float) -> np.ndarray:
    """Bin edges over the post-stim response window [0, t_stop]."""
    return bin_edges(0.0, td.t_stop, bin_s)


def trial_averaged_rate(
    td: TrialData, bin_s: float, *, lo: float = 0.0, hi: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean firing rate (Hz) and SEM across trials over [lo, hi], plus centres."""
    hi = td.t_stop if hi is None else hi
    counts, centers = _counts(td, bin_s, lo, hi)
    if counts.shape[0] == 0:
        z = np.zeros(centers.size)
        return centers, z, z
    rate = counts / bin_s
    mean = rate.mean(axis=0)
    sem = (rate.std(axis=0) / np.sqrt(rate.shape[0])) if rate.shape[0] > 1 else np.zeros_like(mean)
    return centers, mean, sem


@dataclass
class Baseline:
    """Baseline firing measured from the real pre-stimulus window.

    ``mean_count``/``std_count`` are the mean and SD of the summed (over trials)
    spike count per bin during baseline — directly comparable to a post-stim
    summed-PSTH bin of the same width. ``per_trial_rate`` is each trial's own
    baseline rate (Hz), for within-trial baseline subtraction.
    """
    rate_hz: float
    mean_count: float
    std_count: float
    per_trial_rate: np.ndarray = field(default_factory=lambda: np.asarray([]))


def prestim_baseline(td: TrialData, bin_s: float) -> Baseline:
    """Baseline firing statistics from the pre-stimulus window [-pre, 0).

    Raises if the unit carries no pre-stimulus period.
    """
    if not td.has_baseline:
        raise ValueError(
            "no pre-stimulus period on this unit — load it from a pre-stim cache "
            "(e.g. SISortedSpikeSource(cache_subdir='sorted_spike_cache_pre1000ms')) "
            "and pass pre_stimulus_time to extract_trials")
    pre, centers = prestim_counts(td, bin_s)
    if pre.shape[0] == 0 or pre.shape[1] == 0:
        return Baseline(0.0, 0.0, 0.0, np.zeros(td.n_trials))
    summed = pre.sum(axis=0)                                   # per bin, over trials
    per_trial_rate = pre.sum(axis=1) / td.pre_stimulus_time    # Hz per trial
    mean_count = float(summed.mean())
    std_count = float(summed.std())
    rate_hz = float(per_trial_rate.mean())
    return Baseline(rate_hz=rate_hz, mean_count=mean_count, std_count=std_count,
                    per_trial_rate=per_trial_rate)


def optimal_bin_width(
    td: TrialData, *, candidates: Optional[Sequence[float]] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Shimazaki & Shinomoto (2007) MISE-optimal PSTH bin width over [0, t_stop].

    Minimises ``C(Δ) = (2*k̄ - v) / (n*Δ)^2`` over candidate widths, where ``k̄``
    and ``v`` are the mean and biased variance of the summed per-bin counts and
    ``n`` is the trial count. Returns ``(best_bin_s, widths, costs)``.

    Reference: Shimazaki H, Shinomoto S (2007). Neural Computation 19:1503-1527.
    """
    n = td.n_trials
    span = td.t_stop
    if n == 0 or span <= 0:
        return 0.05, np.asarray([0.05]), np.asarray([np.inf])
    if candidates is None:
        candidates = np.geomspace(0.005, max(0.02, span / 4.0), 30)
    candidates = np.asarray(candidates, dtype=float)

    costs = np.empty_like(candidates)
    for i, delta in enumerate(candidates):
        edges = bin_edges(0.0, span, delta)
        summed = np.zeros(edges.size - 1)
        for t in td.trials:
            summed += np.histogram(t, bins=edges)[0]
        costs[i] = (2.0 * summed.mean() - summed.var()) / (n * delta) ** 2
    return float(candidates[int(np.argmin(costs))]), candidates, costs
