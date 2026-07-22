"""
psth.py — shared spike-shaping utilities for the detector bake-off.

Everything the detectors need to turn a single unit's per-trial DataFrame (the
shape a ``SpikeSource.load`` returns) into the arrays they operate on:

* :func:`extract_trials`      per-trial spike times aligned to stimulus onset
* :func:`binned_matrix`       (n_trials x n_bins) spike-count matrix
* :func:`trial_averaged_rate` mean firing rate (Hz) +/- SEM across trials
* :func:`estimate_baseline`   a robust within-trial baseline (these epochs start
                              AT stimulus onset, so there is no pre-stimulus
                              period — see the note below)
* :func:`optimal_bin_width`   Shimazaki & Shinomoto (2007) MISE-optimal PSTH bin

Baseline note
-------------
The recording epochs begin at stimulus onset (``EpochStartStop[0]`` == onset,
and :func:`analyses.zombies_raster_review.zombies_raster._align` re-zeros to it),
so there is **no pre-stimulus window** to normalise against. ``estimate_baseline``
therefore supports two strategies:

* ``"pre"``    — treat an early slice ``[0, pre_s]`` as baseline. Only valid when
                 the response is known to start after ``pre_s``; risky for fast
                 responses. Off by default.
* ``"robust"`` — estimate the baseline from the low part of the PSTH itself
                 (a low quantile of per-bin rates, then mean/SD of the bins at or
                 below it). Works without any pre-stimulus period and is the
                 default. This is what the stationarity-based detectors also lean
                 on implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the raster engine's onset alignment + grouping so the benchmark sees
# exactly the same trials the rasters plot.
from analyses.zombies_raster_review.zombies_raster import _align, GROUP


@dataclass
class TrialData:
    """One unit's trials, aligned to stimulus onset.

    Attributes
    ----------
    trials : list of np.ndarray
        Per-trial spike times in seconds from stimulus onset (>= 0).
    labels : np.ndarray
        Per-trial stimulus condition (``MonkeyName``); parallel to ``trials``.
    t_start, t_stop : float
        Analysis window in seconds from onset. ``t_start`` is 0; ``t_stop`` is
        the shortest trial duration so every trial covers the full window.
    neuron_id : str
    """
    trials: List[np.ndarray]
    labels: np.ndarray
    t_start: float
    t_stop: float
    neuron_id: str = ""

    @property
    def n_trials(self) -> int:
        return len(self.trials)


def extract_trials(
    neuron_df: pd.DataFrame,
    *,
    spike_col: str = "SpikeTimes",
    group: str = GROUP,
    t_stop: Optional[float] = None,
) -> TrialData:
    """Per-trial spikes for one unit, aligned to stimulus onset.

    Parameters
    ----------
    neuron_df : DataFrame
        Rows for a SINGLE unit (already filtered to one NeuronID/Channel), with
        columns ``MonkeyGroup``, ``MonkeyName``, ``EpochStartStop`` and a spike
        column. Same input shape the rasters take.
    group : str
        Stimulus group to keep (default "Zombies").
    t_stop : float, optional
        Analysis-window end (s). Default: the shortest trial duration, so every
        trial spans [0, t_stop].
    """
    df = neuron_df[neuron_df["MonkeyGroup"] == group]
    trials: List[np.ndarray] = []
    labels: List[str] = []
    durations: List[float] = []
    for _, r in df.iterrows():
        start, stop = r["EpochStartStop"]
        trials.append(_align(r[spike_col], start, stop))
        labels.append(r["MonkeyName"])
        durations.append(stop - start)

    if t_stop is None:
        t_stop = float(min(durations)) if durations else 0.0

    nid = str(df["NeuronID"].iloc[0]) if ("NeuronID" in df.columns and len(df)) else ""
    return TrialData(trials=trials, labels=np.asarray(labels, dtype=object),
                     t_start=0.0, t_stop=float(t_stop), neuron_id=nid)


def bin_edges(t_start: float, t_stop: float, bin_s: float) -> np.ndarray:
    """Left-closed bin edges covering [t_start, t_stop] with width ``bin_s``."""
    n = int(round((t_stop - t_start) / bin_s))
    n = max(n, 1)
    return t_start + bin_s * np.arange(n + 1)


def binned_matrix(td: TrialData, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(n_trials x n_bins) spike-count matrix and the bin centres (s).

    Only spikes within [t_start, t_stop] are counted (extract_trials already
    trims to the epoch, and t_stop <= every trial's duration).
    """
    edges = bin_edges(td.t_start, td.t_stop, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if td.n_trials == 0:
        return np.zeros((0, centers.size)), centers
    counts = np.vstack([np.histogram(t, bins=edges)[0] for t in td.trials]).astype(float)
    return counts, centers


def trial_averaged_rate(
    td: TrialData, bin_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean firing rate (Hz) and SEM across trials, plus bin centres (s)."""
    counts, centers = binned_matrix(td, bin_s)
    if counts.shape[0] == 0:
        z = np.zeros(centers.size)
        return centers, z, z
    rate = counts / bin_s
    mean = rate.mean(axis=0)
    sem = (rate.std(axis=0) / np.sqrt(rate.shape[0])) if rate.shape[0] > 1 else np.zeros_like(mean)
    return centers, mean, sem


@dataclass
class Baseline:
    """Baseline firing statistics used by the vs-baseline detectors."""
    rate_hz: float          # baseline firing rate (Hz)
    mean_count: float       # expected spikes per bin at baseline (rate * bin_s * n_trials for summed PSTH)
    std_count: float        # SD of per-bin summed counts at baseline
    method: str


def estimate_baseline(
    td: TrialData,
    bin_s: float,
    *,
    method: str = "robust",
    pre_s: float = 0.15,
    low_quantile: float = 0.5,
) -> Baseline:
    """Estimate a baseline firing level for one unit.

    method="robust" (default): the epochs have no pre-stimulus period, so take
    the baseline from the low part of the trial-averaged PSTH — the mean/SD of
    the summed per-bin counts among bins at or below ``low_quantile`` of the
    per-bin rate. This is robust to a response occupying part of the trial.

    method="pre": use the early slice [0, pre_s] as baseline (only valid if the
    response starts after ``pre_s``).
    """
    counts, centers = binned_matrix(td, bin_s)
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        return Baseline(0.0, 0.0, 0.0, method)

    summed = counts.sum(axis=0)                     # summed PSTH (per bin, over trials)
    per_bin_rate = summed / (bin_s * td.n_trials)   # Hz per bin

    if method == "pre":
        mask = centers < pre_s
        if not mask.any():
            mask = centers < centers[min(1, centers.size - 1)]  # at least first bin
        base_bins = summed[mask]
    else:  # robust
        thr = np.quantile(per_bin_rate, low_quantile)
        mask = per_bin_rate <= thr
        if not mask.any():
            mask = np.ones_like(summed, dtype=bool)
        base_bins = summed[mask]

    mean_count = float(base_bins.mean()) if base_bins.size else 0.0
    std_count = float(base_bins.std()) if base_bins.size else 0.0
    rate_hz = mean_count / (bin_s * td.n_trials) if td.n_trials else 0.0
    return Baseline(rate_hz=rate_hz, mean_count=mean_count, std_count=std_count, method=method)


def optimal_bin_width(
    td: TrialData, *, candidates: Optional[Sequence[float]] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Shimazaki & Shinomoto (2007) MISE-optimal PSTH bin width.

    Minimises the cost ``C(Δ) = (2*k̄ - v) / (n*Δ)^2`` over candidate widths Δ,
    where ``k̄`` and ``v`` are the mean and (biased) variance of the summed
    per-bin spike counts and ``n`` is the trial count. Assumes an inhomogeneous
    Poisson process; no knowledge of the true rate needed.

    Returns ``(best_bin_s, candidate_widths, costs)``.

    Reference: Shimazaki H, Shinomoto S (2007). "A method for selecting the bin
    size of a time histogram." Neural Computation 19(6):1503-1527.
    """
    n = td.n_trials
    span = td.t_stop - td.t_start
    if n == 0 or span <= 0:
        return 0.05, np.asarray([0.05]), np.asarray([np.inf])

    if candidates is None:
        # 5 ms .. span/4, geometrically spaced
        hi = max(0.02, span / 4.0)
        candidates = np.geomspace(0.005, hi, 30)
    candidates = np.asarray(candidates, dtype=float)

    costs = np.empty_like(candidates)
    for i, delta in enumerate(candidates):
        edges = bin_edges(td.t_start, td.t_stop, delta)
        if edges.size < 2:
            costs[i] = np.inf
            continue
        summed = np.zeros(edges.size - 1)
        for t in td.trials:
            summed += np.histogram(t, bins=edges)[0]
        kbar = summed.mean()
        v = summed.var()                       # biased variance (population)
        costs[i] = (2.0 * kbar - v) / (n * delta) ** 2

    best = candidates[int(np.argmin(costs))]
    return float(best), candidates, costs
