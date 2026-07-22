"""
detectors.py — the candidate response-window detectors, one interface.

Every detector implements :class:`WindowDetector` and returns a
:class:`DetectorResult` holding the ``(start_s, end_s)`` windows it found (plus
per-method diagnostics for plotting). All operate on a :class:`~psth.TrialData`
(per-trial spikes aligned to stimulus onset), so they are directly comparable on
the same cell.

Methods
-------
CurrentThreshold   the existing detector: z-score the summed PSTH over the WHOLE
                   trial, keep bins with z > thr (its own mean is the baseline).
                   Included verbatim as the baseline to beat.
UpwardCusum        one-sided CUSUM change detection on the z-scored PSTH.
BaselineZScore     z-score against a robust within-trial baseline (not the whole
                   trial), N-sigma threshold, k-consecutive-bin rule.
PoissonBaseline    Hanes / Poisson-surprise: per-bin p that firing exceeds the
                   baseline Poisson rate; keep runs of significant bins.
ClusterPermutation Maris-Oostenveld: per-bin one-sample stat vs baseline across
                   trials, temporal clustering, sign-flip permutation null with
                   family-wise error control over time.
Zeta               ZETA-inspired (Montijn 2021): binning-free deviation of the
                   cumulative spike distribution from stationarity, bootstrap
                   null; windows read off the instantaneous-rate excursion.

References
----------
* Hanes, Thompson & Schall (1995) Exp Brain Res 103:85-96 (Poisson spike-train
  analysis of response timing); Legendy & Salcman (1985) J Neurophysiol
  53:926-939 (Poisson surprise).
* Maris & Oostenveld (2007) J Neurosci Methods 164:177-190 (cluster-based
  permutation, FWE control over time).
* Montijn et al. (2021) eLife 10:e71969 (ZETA, binning-free responsiveness).
* Shimazaki & Shinomoto (2007) Neural Comput 19:1503-1527 (optimal PSTH bin).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .psth import (
    TrialData, bin_edges, binned_matrix, estimate_baseline, optimal_bin_width,
)

Window = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Shared window helpers
# --------------------------------------------------------------------------- #
def runs_from_mask(mask: np.ndarray, edges: np.ndarray) -> List[Window]:
    """Contiguous True runs in ``mask`` -> (start_s, end_s) using ``edges``.

    A run of bins [i..j] spans ``(edges[i], edges[j+1])``.
    """
    windows: List[Window] = []
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            windows.append((float(edges[i]), float(edges[j + 1])))
            i = j + 1
        else:
            i += 1
    return windows


def merge_close(windows: List[Window], max_gap_s: float) -> List[Window]:
    """Merge windows separated by a gap <= ``max_gap_s``."""
    if not windows:
        return []
    windows = sorted(windows)
    out = [list(windows[0])]
    for lo, hi in windows[1:]:
        if lo - out[-1][1] <= max_gap_s:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def filter_min_duration(windows: List[Window], min_dur_s: float) -> List[Window]:
    return [(lo, hi) for lo, hi in windows if (hi - lo) >= min_dur_s - 1e-9]


# --------------------------------------------------------------------------- #
# Verbatim copies of the current detector's pure helpers (kept local so this
# module doesn't drag in spike_count -> tqdm -> DB import chain). Behaviour
# matches response_window_finder.threshold_window_detection exactly.
# --------------------------------------------------------------------------- #
def _zscore(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    return np.zeros(len(data)) if np.std(data) == 0 else (data - np.mean(data)) / np.std(data)


def _fill_gap_if_one_data_point_away(change_points, norm_data, threshold=0.5):
    if not change_points:
        return []
    filled, i = [], 0
    while i < len(change_points) - 1:
        filled.append(change_points[i])
        if change_points[i + 1] == change_points[i] + 2:
            if (norm_data[change_points[i] + 1] >= threshold * norm_data[change_points[i + 1]]
                    or norm_data[change_points[i] + 1] >= threshold * norm_data[change_points[i]]):
                filled.append(change_points[i] + 1)
        i += 1
    filled.append(change_points[-1])
    return filled


def _threshold_and_fill_gap(z_scored_data, threshold=0.6):
    change_points = np.flatnonzero(np.asarray(z_scored_data) > threshold).tolist()
    change_points = sorted(set(change_points))
    return _fill_gap_if_one_data_point_away(change_points, z_scored_data)


def _extract_consecutive_ranges(numbers):
    result = []
    if len(numbers) > 0:
        numbers = sorted(numbers)
        start = end = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] == end + 1:
                end = numbers[i]
            else:
                if start != end:
                    result.append((start, end))
                start = end = numbers[i]
        if start != end:
            result.append((start, end))
    return result


def _remove_consecutive_tuples(tuples):
    return [(s, e) for s, e in tuples if e != s + 1]


# --------------------------------------------------------------------------- #
# Result + base class
# --------------------------------------------------------------------------- #
@dataclass
class DetectorResult:
    windows: List[Window]
    curve: Optional[np.ndarray] = None       # per-bin statistic for plotting
    centers: Optional[np.ndarray] = None     # bin centres (s) for ``curve``
    threshold: Optional[float] = None        # threshold line for ``curve``
    extra: Dict = field(default_factory=dict)


class WindowDetector:
    """Detector interface. Subclasses set ``name``/``color`` and implement
    :meth:`detect`."""
    name: str = "detector"
    color: str = "#333333"

    def detect(self, td: TrialData) -> DetectorResult:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}'>"


# --------------------------------------------------------------------------- #
# 1. Current threshold detector (the baseline to beat)
# --------------------------------------------------------------------------- #
class CurrentThresholdDetector(WindowDetector):
    """Reproduces ``response_window_finder.threshold_window_detection``.

    z-scores the summed PSTH across the WHOLE trial and keeps bins with
    ``z > threshold`` (its own whole-trial mean is the baseline — the behaviour
    we are trying to improve on), with the same single-gap fill and drop of
    2-bin windows.
    """
    name = "current"
    color = "#9E9E9E"

    def __init__(self, bin_s: float = 0.05, threshold: float = 0.5):
        self.bin_s = bin_s
        self.threshold = threshold

    def detect(self, td: TrialData) -> DetectorResult:
        counts, centers = binned_matrix(td, self.bin_s)
        edges = bin_edges(td.t_start, td.t_stop, self.bin_s)
        if counts.shape[0] == 0:
            return DetectorResult([], curve=None, centers=centers)
        summed = counts.sum(axis=0)
        z = _zscore(summed)
        change_points = _threshold_and_fill_gap(z, self.threshold)
        ranges = _remove_consecutive_tuples(_extract_consecutive_ranges(change_points))
        windows = [(float(edges[i]), float(edges[j + 1])) for i, j in ranges]
        return DetectorResult(windows, curve=z, centers=centers, threshold=self.threshold)


# --------------------------------------------------------------------------- #
# 2. Upward CUSUM
# --------------------------------------------------------------------------- #
class UpwardCusumDetector(WindowDetector):
    """One-sided CUSUM on the z-scored summed PSTH.

    Accumulates positive deviations ``S_t = max(0, S_{t-1} + x_t - k)`` and marks
    bins where ``S_t > h``. Unlike the legacy two-sided CUSUM, only UPWARD
    excursions count, so a firing *decrease* is never reported as a response.
    """
    name = "cusum"
    color = "#8E6FCE"

    def __init__(self, bin_s: float = 0.05, k: float = 0.5, h: float = 1.0,
                 min_dur_s: float = 0.05):
        self.bin_s = bin_s
        self.k = k
        self.h = h
        self.min_dur_s = min_dur_s

    def detect(self, td: TrialData) -> DetectorResult:
        counts, centers = binned_matrix(td, self.bin_s)
        edges = bin_edges(td.t_start, td.t_stop, self.bin_s)
        if counts.shape[0] == 0:
            return DetectorResult([], centers=centers)
        z = _zscore(counts.sum(axis=0))
        s = np.zeros(z.size)
        for t in range(1, z.size):
            s[t] = max(0.0, s[t - 1] + z[t] - self.k)
        mask = s > self.h
        windows = filter_min_duration(runs_from_mask(mask, edges), self.min_dur_s)
        return DetectorResult(windows, curve=s, centers=centers, threshold=self.h)


# --------------------------------------------------------------------------- #
# 3. Baseline-relative z-score
# --------------------------------------------------------------------------- #
class BaselineZScoreDetector(WindowDetector):
    """z-score against a ROBUST within-trial baseline, not the whole trial.

    Fixes the current detector's core flaw (a strong response inflates its own
    baseline): baseline mean/SD come from the low part of the PSTH (see
    :func:`psth.estimate_baseline`). A bin is flagged when its summed count is
    ``n_sigma`` SDs above baseline (SD floored at the Poisson expectation
    ``sqrt(mean)`` so a flat baseline can't make the z blow up). Runs must be at
    least ``min_consec`` bins; gaps up to ``max_gap`` bins are bridged.
    """
    name = "baseline_z"
    color = "#2F80ED"

    def __init__(self, bin_s: float = 0.05, n_sigma: float = 3.0,
                 min_consec: int = 2, max_gap_bins: int = 1,
                 baseline_method: str = "robust"):
        self.bin_s = bin_s
        self.n_sigma = n_sigma
        self.min_consec = min_consec
        self.max_gap_bins = max_gap_bins
        self.baseline_method = baseline_method

    def detect(self, td: TrialData) -> DetectorResult:
        counts, centers = binned_matrix(td, self.bin_s)
        edges = bin_edges(td.t_start, td.t_stop, self.bin_s)
        if counts.shape[0] == 0:
            return DetectorResult([], centers=centers)
        summed = counts.sum(axis=0)
        base = estimate_baseline(td, self.bin_s, method=self.baseline_method)
        sd = max(base.std_count, np.sqrt(max(base.mean_count, 1e-9)), 1e-9)
        z = (summed - base.mean_count) / sd
        mask = z > self.n_sigma
        windows = runs_from_mask(mask, edges)
        windows = merge_close(windows, self.max_gap_bins * self.bin_s)
        windows = filter_min_duration(windows, self.min_consec * self.bin_s)
        return DetectorResult(windows, curve=z, centers=centers,
                              threshold=self.n_sigma,
                              extra={"baseline_hz": base.rate_hz})


# --------------------------------------------------------------------------- #
# 4. Poisson-surprise vs baseline (Hanes-style)
# --------------------------------------------------------------------------- #
class PoissonBaselineDetector(WindowDetector):
    """Per-bin Poisson surprise that firing EXCEEDS the baseline rate.

    Baseline rate ``lambda`` (Hz) is estimated robustly; the expected summed
    count in a bin is ``mu = lambda * bin_s * n_trials``. The surprise of an
    observed summed count ``k`` is ``S = -log10 P(X >= k | Poisson(mu))``. Bins
    with ``S > surprise_thr`` (default 2 -> p < 0.01) are kept as runs. This is
    the Hanes/Legendy family: a real per-bin false-positive rate instead of an
    arbitrary z.
    """
    name = "poisson"
    color = "#27AE60"

    def __init__(self, bin_s: float = 0.05, surprise_thr: float = 2.0,
                 min_consec: int = 2, max_gap_bins: int = 1,
                 baseline_method: str = "robust"):
        self.bin_s = bin_s
        self.surprise_thr = surprise_thr
        self.min_consec = min_consec
        self.max_gap_bins = max_gap_bins
        self.baseline_method = baseline_method

    def detect(self, td: TrialData) -> DetectorResult:
        from scipy import stats
        counts, centers = binned_matrix(td, self.bin_s)
        edges = bin_edges(td.t_start, td.t_stop, self.bin_s)
        if counts.shape[0] == 0:
            return DetectorResult([], centers=centers)
        summed = counts.sum(axis=0)
        base = estimate_baseline(td, self.bin_s, method=self.baseline_method)
        mu = max(base.rate_hz * self.bin_s * td.n_trials, 1e-9)
        # P(X >= k) = sf(k-1); surprise = -log10(p)
        p = stats.poisson.sf(summed - 1, mu)
        surprise = -np.log10(np.clip(p, 1e-300, 1.0))
        mask = surprise > self.surprise_thr
        windows = runs_from_mask(mask, edges)
        windows = merge_close(windows, self.max_gap_bins * self.bin_s)
        windows = filter_min_duration(windows, self.min_consec * self.bin_s)
        return DetectorResult(windows, curve=surprise, centers=centers,
                              threshold=self.surprise_thr,
                              extra={"baseline_hz": base.rate_hz, "mu": mu})


# --------------------------------------------------------------------------- #
# 5. Cluster-based permutation (Maris & Oostenveld)
# --------------------------------------------------------------------------- #
class ClusterPermutationDetector(WindowDetector):
    """Temporal cluster-based permutation test, FWE-controlled over time.

    Per bin, a one-sample statistic tests whether the per-trial rate exceeds the
    baseline: ``t_i = mean_trials(d_i) / (sd_i / sqrt(n))`` where
    ``d_{trial,i} = rate_{trial,i} - baseline_rate``. Bins with ``t_i >
    t_thresh`` seed candidate clusters (contiguous runs); each cluster's mass is
    the sum of its ``t_i``. The null is built by sign-flipping each trial's whole
    difference vector ``n_perm`` times and recording the max cluster mass; a
    cluster survives if its mass exceeds the ``1-alpha`` quantile of that null.
    This chooses the window AND corrects for multiple comparisons across time in
    one step, so no arbitrary per-bin threshold leaks into the result.

    Set ``mode="category"`` to instead test differences across ``MonkeyName``
    (per-bin one-way F, permute labels) — stimulus-selective windows.
    """
    name = "cluster_perm"
    color = "#EB5757"

    def __init__(self, bin_s: float = 0.05, t_thresh: float = 2.0,
                 n_perm: int = 500, alpha: float = 0.05, seed: int = 0,
                 mode: str = "baseline", baseline_method: str = "robust"):
        self.bin_s = bin_s
        self.t_thresh = t_thresh
        self.n_perm = n_perm
        self.alpha = alpha
        self.seed = seed
        self.mode = mode
        self.baseline_method = baseline_method

    # -- per-bin statistics -------------------------------------------------- #
    @staticmethod
    def _one_sample_t(d: np.ndarray) -> np.ndarray:
        """One-sample t of each column of ``d`` (n_trials x n_bins) vs 0."""
        n = d.shape[0]
        mean = d.mean(axis=0)
        sd = d.std(axis=0, ddof=1) if n > 1 else np.ones(d.shape[1])
        sd = np.where(sd < 1e-12, 1e-12, sd)
        return mean / (sd / np.sqrt(n))

    @staticmethod
    def _oneway_f(rate: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per-bin one-way ANOVA F across groups (rate: n_trials x n_bins)."""
        groups = [rate[labels == g] for g in np.unique(labels)]
        groups = [g for g in groups if g.shape[0] > 0]
        n_tot = sum(g.shape[0] for g in groups)
        k = len(groups)
        if k < 2 or n_tot <= k:
            return np.zeros(rate.shape[1])
        grand = np.concatenate(groups, axis=0).mean(axis=0)
        ss_b = sum(g.shape[0] * (g.mean(axis=0) - grand) ** 2 for g in groups)
        ss_w = sum(((g - g.mean(axis=0)) ** 2).sum(axis=0) for g in groups)
        df_b, df_w = k - 1, n_tot - k
        ms_w = np.where(ss_w <= 0, 1e-12, ss_w / df_w)
        return (ss_b / df_b) / ms_w

    @staticmethod
    def _clusters(stat: np.ndarray, thr: float) -> List[Tuple[int, int, float]]:
        """(start_bin, end_bin_inclusive, mass) for supra-threshold runs."""
        mask = stat > thr
        out = []
        i, n = 0, mask.size
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                out.append((i, j, float(stat[i:j + 1].sum())))
                i = j + 1
            else:
                i += 1
        return out

    def detect(self, td: TrialData) -> DetectorResult:
        counts, centers = binned_matrix(td, self.bin_s)
        edges = bin_edges(td.t_start, td.t_stop, self.bin_s)
        if counts.shape[0] < 2:
            return DetectorResult([], centers=centers)
        rate = counts / self.bin_s
        rng = np.random.default_rng(self.seed)

        if self.mode == "category":
            labels = td.labels
            obs = self._oneway_f(rate, labels)
            clusters = self._clusters(obs, self.t_thresh)
            null_max = np.zeros(self.n_perm)
            for p in range(self.n_perm):
                perm = rng.permutation(labels)
                cl = self._clusters(self._oneway_f(rate, perm), self.t_thresh)
                null_max[p] = max((m for _, _, m in cl), default=0.0)
        else:  # baseline
            base = estimate_baseline(td, self.bin_s, method=self.baseline_method)
            d = rate - base.rate_hz                      # per-trial deviation (Hz)
            obs = self._one_sample_t(d)
            clusters = self._clusters(obs, self.t_thresh)
            null_max = np.zeros(self.n_perm)
            for p in range(self.n_perm):
                signs = rng.choice([-1.0, 1.0], size=d.shape[0])[:, None]
                cl = self._clusters(self._one_sample_t(d * signs), self.t_thresh)
                null_max[p] = max((m for _, _, m in cl), default=0.0)

        crit = np.quantile(null_max, 1.0 - self.alpha) if self.n_perm else np.inf
        windows, pvals = [], []
        for i, j, mass in clusters:
            pval = (1 + int((null_max >= mass).sum())) / (self.n_perm + 1)
            if pval < self.alpha:
                windows.append((float(edges[i]), float(edges[j + 1])))
                pvals.append(pval)
        return DetectorResult(windows, curve=obs, centers=centers,
                              threshold=self.t_thresh,
                              extra={"cluster_pvals": pvals, "null_crit": float(crit),
                                     "mode": self.mode})


# --------------------------------------------------------------------------- #
# 6. ZETA-inspired, binning-free
# --------------------------------------------------------------------------- #
class ZetaDetector(WindowDetector):
    """ZETA-inspired binning-free responsiveness + window extraction.

    Pool all spikes (across trials) into [t_start, t_stop] and build the
    cumulative spike-time distribution. Under a stationary rate the cumulative is
    linear in time; the mean-subtracted deviation ``d(t)`` (a Brownian-bridge-
    like curve) has extreme value ZETA = max|d|. Significance comes from a
    bootstrap null that breaks stimulus locking by circularly jittering each
    trial's spikes ``n_boot`` times. If responsive, windows are read off the
    instantaneous rate (fine Gaussian-smoothed pooled density): contiguous
    stretches where the rate exceeds its mean by ``rate_frac`` of the peak
    excursion, retained only if they overlap the ZETA peak's lobe.

    Faithful to ZETA's binning-free core (Montijn et al. 2021, eLife 10:e71969);
    the window read-out is a practical stand-in for their multi-scale-derivative
    onset/offset routine and is labelled as such.
    """
    name = "zeta"
    color = "#F2994A"

    def __init__(self, n_boot: int = 250, alpha: float = 0.05, seed: int = 0,
                 smooth_s: float = 0.02, rate_frac: float = 0.5):
        self.n_boot = n_boot
        self.alpha = alpha
        self.seed = seed
        self.smooth_s = smooth_s
        self.rate_frac = rate_frac

    # -- cumulative deviation bridge ---------------------------------------- #
    @staticmethod
    def _deviation(spikes: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
        """Mean-subtracted deviation of the cumulative spike fraction from
        linear, sampled at the (sorted) spike times. Returns (t, d)."""
        s = np.sort(spikes[(spikes >= t0) & (spikes <= t1)])
        if s.size < 2:
            return np.asarray([t0, t1]), np.zeros(2)
        frac = np.arange(1, s.size + 1) / s.size          # empirical CDF
        lin = (s - t0) / (t1 - t0)                          # stationary CDF
        d = frac - lin
        return s, d - d.mean()

    def detect(self, td: TrialData) -> DetectorResult:
        t0, t1 = td.t_start, td.t_stop
        pooled = np.concatenate(td.trials) if td.n_trials else np.asarray([])
        pooled = pooled[(pooled >= t0) & (pooled <= t1)]
        centers = 0.5 * (bin_edges(t0, t1, self.smooth_s)[:-1]
                         + bin_edges(t0, t1, self.smooth_s)[1:])
        if pooled.size < 3 or td.n_trials < 2:
            return DetectorResult([], centers=centers)

        t_dev, d = self._deviation(pooled, t0, t1)
        zeta = float(np.max(np.abs(d)))
        t_peak = float(t_dev[int(np.argmax(np.abs(d)))])

        # bootstrap null: circularly jitter each trial's spikes to break locking
        rng = np.random.default_rng(self.seed)
        span = t1 - t0
        null_zeta = np.empty(self.n_boot)
        for b in range(self.n_boot):
            jit = []
            for tr in td.trials:
                tr = tr[(tr >= t0) & (tr <= t1)]
                if tr.size:
                    shifted = t0 + np.mod((tr - t0) + rng.uniform(0, span), span)
                    jit.append(shifted)
            allsp = np.concatenate(jit) if jit else np.asarray([])
            if allsp.size < 2:
                null_zeta[b] = 0.0
                continue
            _, db = self._deviation(allsp, t0, t1)
            null_zeta[b] = np.max(np.abs(db))
        pval = (1 + int((null_zeta >= zeta).sum())) / (self.n_boot + 1)

        # instantaneous rate (pooled density, Gaussian-smoothed) for the read-out
        edges = bin_edges(t0, t1, self.smooth_s)
        dens = np.histogram(pooled, bins=edges)[0] / (self.smooth_s * td.n_trials)
        rate = _gaussian_smooth(dens, sigma_bins=1.5)

        windows: List[Window] = []
        if pval < self.alpha and rate.size:
            level = rate.mean() + self.rate_frac * (rate.max() - rate.mean())
            mask = rate > level
            cand = runs_from_mask(mask, edges)
            # keep lobes overlapping the ZETA peak, else the strongest lobe
            over = [w for w in cand if w[0] <= t_peak <= w[1]]
            windows = over if over else ([max(cand, key=lambda w: w[1] - w[0])] if cand else [])

        return DetectorResult(windows, curve=rate, centers=centers,
                              threshold=None,
                              extra={"zeta": zeta, "p": pval, "t_peak": t_peak})


def _gaussian_smooth(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Reflect-padded 1-D Gaussian smoothing (no scipy dependency needed)."""
    if sigma_bins <= 0 or x.size == 0:
        return x
    radius = int(max(1, round(3 * sigma_bins)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_bins) ** 2)
    k /= k.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, k, mode="valid")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def default_detectors(bin_s: float = 0.05) -> List[WindowDetector]:
    """The full bake-off suite, in a stable order (also fixes plot colours)."""
    return [
        CurrentThresholdDetector(bin_s=bin_s, threshold=0.5),
        UpwardCusumDetector(bin_s=bin_s),
        BaselineZScoreDetector(bin_s=bin_s, n_sigma=3.0),
        PoissonBaselineDetector(bin_s=bin_s, surprise_thr=2.0),
        ClusterPermutationDetector(bin_s=bin_s, mode="baseline"),
        ZetaDetector(),
    ]
