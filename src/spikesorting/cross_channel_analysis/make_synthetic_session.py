"""
make_synthetic_session.py — build a fake session with a *known* duplicate.

Real ``sorted_spikes.pkl`` files live next to the raw Intan recordings, which
aren't in this repo. This generator fabricates a small session whose ground
truth we control, so the analysis and plots can be exercised end-to-end and the
metrics sanity-checked:

    * neuron N1 fires on contact-adjacent channels C-021 and C-010 → its spikes
      appear as "Unit 1" on **both** channels (jittered by a sample or two).
      This is the planted duplicate.
    * neuron N2 fires independently, appearing only on C-026 as "Unit 1".
    * channel C-021 also has a second, independent "Unit 2".

It can also synthesise matching voltage traces (a template scaled per channel by
distance from the source) so the footprint plots have something to chew on.

Everything is deterministic given ``seed`` — no wall-clock randomness — so the
smoke test is reproducible.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

from . import probe_geometry as geom

SAMPLE_RATE = 30000.0


def _poisson_train(rng, rate_hz, duration_s, sample_rate, refractory_ms=2.0):
    """Homogeneous-ish spike train with a hard refractory period."""
    n = int(rate_hz * duration_s * 1.3) + 10
    isis = rng.exponential(1.0 / rate_hz, size=n)
    refr = refractory_ms * 1e-3
    isis = np.clip(isis, refr, None)
    times = np.cumsum(isis)
    times = times[times < duration_s]
    return np.round(times * sample_rate).astype(np.int64)


def _spike_template(radius=25):
    """A simple biphasic extracellular spike (negative trough, positive rebound)."""
    t = np.arange(-radius, radius)
    trough = -np.exp(-((t + 2) ** 2) / (2 * 3.0 ** 2))
    rebound = 0.4 * np.exp(-((t - 6) ** 2) / (2 * 5.0 ** 2))
    return (trough + rebound)


def make_synthetic_session(
    *,
    duration_s: float = 120.0,
    seed: int = 0,
    with_voltages: bool = False,
) -> Tuple[dict, Optional[Dict[str, np.ndarray]], float]:
    """Return ``(sorted_spikes_dict, voltages_by_channel_or_None, sample_rate)``.

    ``sorted_spikes_dict`` has the same shape as a real ``sorted_spikes.pkl``
    (channel-name keys instead of Channel enums — the loader accepts both).
    """
    rng = np.random.default_rng(seed)
    sr = SAMPLE_RATE

    # --- ground-truth neurons ---------------------------------------------- #
    n1 = _poisson_train(rng, rate_hz=8.0, duration_s=duration_s, sample_rate=sr)
    n2 = _poisson_train(rng, rate_hz=5.0, duration_s=duration_s, sample_rate=sr)
    extra = _poisson_train(rng, rate_hz=3.0, duration_s=duration_s, sample_rate=sr)

    # N1 detected on two adjacent channels, with tiny per-channel jitter and a
    # few missed spikes on the weaker channel — the planted duplicate.
    jitter = rng.integers(-2, 3, size=n1.size)
    n1_on_c021 = np.sort(n1 + jitter)
    keep = rng.random(n1.size) < 0.85  # C-010 misses ~15% of N1's spikes
    jitter2 = rng.integers(-2, 3, size=n1.size)
    n1_on_c010 = np.sort((n1 + jitter2)[keep])

    sorted_spikes = {
        "C-021": {"Unit 1": n1_on_c021, "Unit 2": np.sort(extra)},
        "C-010": {"Unit 1": n1_on_c010},
        "C-026": {"Unit 1": np.sort(n2)},
    }

    voltages = None
    if with_voltages:
        voltages = _synthesize_voltages(sorted_spikes, n1, n2, extra, duration_s, sr, rng)

    return sorted_spikes, voltages, sr


def _synthesize_voltages(sorted_spikes, n1, n2, extra, duration_s, sr, rng):
    """Build voltage traces on a handful of channels around the probe.

    N1 is placed at contact of C-021 and bleeds onto neighbours (amplitude
    decays with contact distance); N2 and the extra unit sit elsewhere.
    """
    template = _spike_template()
    radius = template.size // 2
    n_samples = int(duration_s * sr)

    # a window of channels spanning the interesting contacts
    channels = ["C-025", "C-006", "C-021", "C-010", "C-026", "C-005"]
    voltages = {c: rng.normal(0, 0.05, size=n_samples) for c in channels}

    def deposit(source_channel, times, amp):
        src = geom.contact_index_of(source_channel)
        for ch in channels:
            ci = geom.contact_index_of(ch)
            decay = np.exp(-abs(ci - src) / 1.5)  # spatial spread over contacts
            scale = amp * decay
            if scale < 1e-3:
                continue
            for t in times:
                s, e = t - radius, t + radius
                if 0 <= s and e < n_samples:
                    voltages[ch][s:e] += scale * template

    deposit("C-021", n1, amp=1.0)      # the duplicated neuron
    deposit("C-026", n2, amp=0.9)      # independent neuron
    deposit("C-021", extra, amp=0.6)   # C-021 Unit 2, smaller
    return voltages


def write_synthetic_session(out_dir: str, *, seed: int = 0,
                            with_voltages: bool = False) -> str:
    """Write ``sorted_spikes.pkl`` (and optional ``voltages.npz``) to ``out_dir``.

    Returns the path to the written pickle.
    """
    os.makedirs(out_dir, exist_ok=True)
    sorted_spikes, voltages, _ = make_synthetic_session(seed=seed, with_voltages=with_voltages)
    pkl_path = os.path.join(out_dir, "sorted_spikes.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(sorted_spikes, f)
    if voltages is not None:
        np.savez_compressed(os.path.join(out_dir, "voltages.npz"), **voltages)
    return pkl_path
