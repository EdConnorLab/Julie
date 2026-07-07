"""
spiketrain_metrics.py — quantify whether two units are the same neuron.

The physics: one action potential from a neuron near two adjacent contacts
crosses threshold on *both* channels within a fraction of a millisecond. So if
two per-channel units are really one neuron, their spike trains are
near-synchronous. That shows up three ways, all computed here:

1. **Coincidence fraction** — the fraction of one train's spikes that have a
   partner in the other train within a few tenths of a millisecond. This is the
   single most decisive number: genuine duplicates sit near 1.0, independent
   neurons near 0 (chance level ~ rate * 2 * window).

2. **Cross-correlogram (CCG)** — histogram of spike-time differences. A
   duplicate produces a tall, needle-thin peak centred on zero lag. Two truly
   distinct-but-connected neurons produce a broader, offset bump instead.

3. **Refractory check on the merged train** — a single neuron cannot fire twice
   within its refractory period (~1.5 ms). If A and B are the same neuron, their
   *pooled* auto-correlogram keeps a clean hole at zero; if they are two
   different neurons, merging fills that hole in.

All routines take spike **sample indices** and a ``sample_rate`` so lags come
out in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Coincidence
# --------------------------------------------------------------------------- #
def match_count(a: np.ndarray, b: np.ndarray, delta_samples: float) -> int:
    """Number of spikes in ``a`` that have at least one spike in ``b`` within
    ``±delta_samples``. Both arrays must be sorted. O(n log n)."""
    if a.size == 0 or b.size == 0:
        return 0
    # nearest neighbour in b for each a via searchsorted
    pos = np.searchsorted(b, a)
    left = np.clip(pos - 1, 0, b.size - 1)
    right = np.clip(pos, 0, b.size - 1)
    dist = np.minimum(np.abs(a - b[left]), np.abs(a - b[right]))
    return int(np.count_nonzero(dist <= delta_samples))


@dataclass
class CoincidenceResult:
    n_a: int
    n_b: int
    n_matched: int          # spikes of the smaller train with a partner
    coincidence: float      # n_matched / min(n_a, n_b)
    expected_chance: float   # coincidence expected if the trains were independent
    ratio_over_chance: float  # coincidence / expected_chance


def coincidence(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sample_rate: float,
    window_ms: float = 0.4,
    recording_n_samples: Optional[int] = None,
) -> CoincidenceResult:
    """Coincidence fraction between two spike trains.

    ``window_ms`` is the half-width of the coincidence window (0.4 ms ≈ 12
    samples at 30 kHz — a couple of samples of jitter plus propagation delay).

    ``expected_chance`` is what the coincidence would be if the two trains were
    independent Poisson processes of the same rates, so ``ratio_over_chance``
    tells you how far above chance the synchrony is.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    delta = window_ms * 1e-3 * sample_rate
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return CoincidenceResult(n_a, n_b, 0, 0.0, 0.0, 0.0)

    # count matches on the smaller train (bounded by min(n_a, n_b))
    if n_a <= n_b:
        n_matched = match_count(a, b, delta)
    else:
        n_matched = match_count(b, a, delta)
    coinc = n_matched / min(n_a, n_b)

    # chance level: prob a given spike of the smaller train falls within ±delta
    # of any of the larger train's spikes, under a uniform-rate assumption.
    span = recording_n_samples or (int(max(a[-1], b[-1])) + 1)
    larger_n = max(n_a, n_b)
    expected = min(1.0, larger_n * (2.0 * delta) / span)
    ratio = coinc / expected if expected > 0 else float("inf")
    return CoincidenceResult(n_a, n_b, n_matched, coinc, expected, ratio)


# --------------------------------------------------------------------------- #
# Correlograms
# --------------------------------------------------------------------------- #
@dataclass
class Correlogram:
    counts: np.ndarray       # spike-pair counts per bin
    bin_centers_ms: np.ndarray
    bin_size_ms: float
    window_ms: float


def cross_correlogram(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sample_rate: float,
    window_ms: float = 25.0,
    bin_size_ms: float = 0.5,
) -> Correlogram:
    """Cross-correlogram of ``b`` relative to ``a`` over ``±window_ms``.

    Bin ``t`` counts pairs where ``t_b - t_a ≈ t``. A duplicate shows a spike at
    the zero-lag bin far above the flanks.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    window = window_ms * 1e-3 * sample_rate
    bin_size = bin_size_ms * 1e-3 * sample_rate
    n_bins = int(np.ceil(window / bin_size))
    edges = np.arange(-n_bins, n_bins + 1) * bin_size  # symmetric around 0
    counts = np.zeros(edges.size - 1, dtype=np.int64)

    if a.size and b.size:
        lo = np.searchsorted(b, a - window, side="left")
        hi = np.searchsorted(b, a + window, side="right")
        for i in range(a.size):
            if hi[i] > lo[i]:
                diffs = b[lo[i]:hi[i]] - a[i]
                counts += np.histogram(diffs, bins=edges)[0]

    centers = (edges[:-1] + edges[1:]) / 2.0 / sample_rate * 1e3
    return Correlogram(counts, centers, bin_size_ms, window_ms)


def auto_correlogram(
    a: np.ndarray,
    *,
    sample_rate: float,
    window_ms: float = 25.0,
    bin_size_ms: float = 0.5,
) -> Correlogram:
    """Auto-correlogram (self-pairs at zero lag removed)."""
    acg = cross_correlogram(a, a, sample_rate=sample_rate,
                            window_ms=window_ms, bin_size_ms=bin_size_ms)
    # remove the n self-matches that land in the zero-lag bin
    zero_bin = np.argmin(np.abs(acg.bin_centers_ms))
    acg.counts[zero_bin] = max(0, acg.counts[zero_bin] - np.asarray(a).size)
    return acg


def merge_trains(a: np.ndarray, b: np.ndarray, *, sample_rate: float,
                 dedup_window_ms: float = 0.4) -> np.ndarray:
    """Pool two trains, collapsing near-coincident spikes into one.

    Used for the refractory sanity check: if A and B are the same neuron, the
    merged train should still respect the refractory period.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    merged = np.sort(np.concatenate([a, b]))
    if merged.size == 0:
        return merged
    delta = dedup_window_ms * 1e-3 * sample_rate
    keep = np.ones(merged.size, dtype=bool)
    keep[1:] = np.diff(merged) > delta
    return merged[keep]


def refractory_violation_rate(a: np.ndarray, *, sample_rate: float,
                              refractory_ms: float = 1.5) -> float:
    """Fraction of inter-spike intervals shorter than the refractory period.

    Near 0 for a clean single unit. If pooling two units pushes this well above
    each unit's own value, the two are probably *distinct* neurons."""
    a = np.asarray(a, dtype=np.int64)
    if a.size < 2:
        return 0.0
    refr = refractory_ms * 1e-3 * sample_rate
    isi = np.diff(a)
    return float(np.count_nonzero(isi < refr) / isi.size)


# --------------------------------------------------------------------------- #
# Pairwise sweep across a whole session
# --------------------------------------------------------------------------- #
@dataclass
class PairMetrics:
    uid_a: str
    uid_b: str
    channel_a: str
    channel_b: str
    n_a: int
    n_b: int
    coincidence: float
    ratio_over_chance: float
    distance_um: Optional[float]
    contacts_apart: Optional[int]
    same_channel: bool
    refractory_a: float
    refractory_b: float
    refractory_merged: float

    @property
    def merged_refractory_ok(self) -> bool:
        """Merged train is still (nearly) refractory — consistent with one neuron."""
        return self.refractory_merged <= max(self.refractory_a, self.refractory_b) + 0.005


def all_pairs(
    session,
    *,
    window_ms: float = 0.4,
    refractory_ms: float = 1.5,
    include_same_channel: bool = False,
) -> List[PairMetrics]:
    """Compute pairwise duplicate metrics for every pair of units in a session.

    ``session`` is a :class:`loader.Session`. Returns a list sorted by
    coincidence (most duplicate-like first).
    """
    from . import probe_geometry as geom

    sr = session.sample_rate
    units = session.units
    out: List[PairMetrics] = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            ua, ub = units[i], units[j]
            same_ch = ua.channel == ub.channel
            if same_ch and not include_same_channel:
                continue
            cr = coincidence(ua.spike_indices, ub.spike_indices,
                             sample_rate=sr, window_ms=window_ms)
            merged = merge_trains(ua.spike_indices, ub.spike_indices, sample_rate=sr,
                                  dedup_window_ms=window_ms)
            out.append(PairMetrics(
                uid_a=ua.uid, uid_b=ub.uid,
                channel_a=ua.channel, channel_b=ub.channel,
                n_a=ua.n_spikes, n_b=ub.n_spikes,
                coincidence=cr.coincidence,
                ratio_over_chance=cr.ratio_over_chance,
                distance_um=geom.channel_distance_um(ua.channel, ub.channel),
                contacts_apart=geom.contacts_apart(ua.channel, ub.channel),
                same_channel=same_ch,
                refractory_a=refractory_violation_rate(ua.spike_indices, sample_rate=sr,
                                                       refractory_ms=refractory_ms),
                refractory_b=refractory_violation_rate(ub.spike_indices, sample_rate=sr,
                                                       refractory_ms=refractory_ms),
                refractory_merged=refractory_violation_rate(merged, sample_rate=sr,
                                                            refractory_ms=refractory_ms),
            ))
    out.sort(key=lambda p: p.coincidence, reverse=True)
    return out


def coincidence_matrix(session, *, window_ms: float = 0.4
                       ) -> Tuple[np.ndarray, List[str]]:
    """Full symmetric coincidence matrix (units × units) and the uid order."""
    sr = session.sample_rate
    units = session.units
    n = len(units)
    mat = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            c = coincidence(units[i].spike_indices, units[j].spike_indices,
                            sample_rate=sr, window_ms=window_ms).coincidence
            mat[i, j] = mat[j, i] = c
    return mat, [u.uid for u in units]


def candidate_duplicates(
    pairs: List[PairMetrics],
    *,
    coincidence_threshold: float = 0.3,
    ratio_threshold: float = 5.0,
) -> List[PairMetrics]:
    """Filter to pairs that look like duplicates: high coincidence AND well
    above chance."""
    return [p for p in pairs
            if p.coincidence >= coincidence_threshold
            and p.ratio_over_chance >= ratio_threshold]
