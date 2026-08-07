"""
waveforms.py — cross-channel spike waveforms (the spatial-footprint evidence).

Spike-train synchrony (see ``spiketrain_metrics.py``) is the strongest single
signal, but the clinching evidence that two units are one neuron is that they
share the same **spatial footprint**: the average voltage across *all* contacts,
triggered on a unit's spike times, peaks at one contact and decays smoothly with
distance. If unit A (on channel X) and unit B (on channel Y) are the same
neuron, both footprints peak at the same contact and have near-identical shape.

This needs the raw/filtered voltage traces — a dict ``{channel_name: 1-D
np.ndarray}`` exactly like ``InputDataManager.voltages_by_channel`` in
``spikesorting/datahandler.py``. It is therefore optional: the spike-train
analysis runs without it. Use :func:`load_voltages` to build that dict from a
session directory (this triggers windowsort's preprocessing), or pass your own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# windowsort centres its spike view on ±25 samples (spikes.py). Match it.
DEFAULT_RADIUS = 25


def load_voltages(session_dir: str) -> Dict[str, np.ndarray]:
    """Load per-channel voltage traces for a session via windowsort.

    Returns ``{channel_name: 1-D array}``. Heavy: reads (and, first time,
    high-pass filters and caches) the whole recording. Requires ``windowsort``
    and ``clat`` to be importable.
    """
    from windowsort.datahandler import InputDataManager
    mgr = InputDataManager(session_dir)
    return {
        (ch.value if hasattr(ch, "value") else str(ch)): np.asarray(v)
        for ch, v in mgr.voltages_by_channel.items()
    }


def extract_waveforms(
    voltages: np.ndarray,
    spike_indices: np.ndarray,
    *,
    radius: int = DEFAULT_RADIUS,
    max_spikes: Optional[int] = 500,
) -> np.ndarray:
    """Cut ``[-radius, +radius]`` snippets around each spike on one channel.

    Returns an ``(n_spikes, 2*radius)`` array; spikes too close to an edge are
    dropped. ``max_spikes`` subsamples (evenly) for speed.
    """
    voltages = np.asarray(voltages)
    idx = np.asarray(spike_indices, dtype=np.int64)
    if max_spikes is not None and idx.size > max_spikes:
        idx = idx[np.linspace(0, idx.size - 1, max_spikes).astype(np.int64)]
    valid = (idx - radius >= 0) & (idx + radius < voltages.size)
    idx = idx[valid]
    if idx.size == 0:
        return np.empty((0, 2 * radius))
    offsets = np.arange(-radius, radius)
    return voltages[idx[:, None] + offsets[None, :]]


def mean_waveform(voltages, spike_indices, *, radius=DEFAULT_RADIUS,
                  max_spikes=500) -> np.ndarray:
    """Mean spike waveform on one channel (1-D, length ``2*radius``)."""
    w = extract_waveforms(voltages, spike_indices, radius=radius, max_spikes=max_spikes)
    if w.shape[0] == 0:
        return np.zeros(2 * radius)
    return w.mean(axis=0)


def peak_to_peak(wave) -> float:
    """Peak-to-peak amplitude of one waveform, in the units of ``wave`` (µV)."""
    wave = np.asarray(wave, dtype=float)
    return float(wave.max() - wave.min()) if wave.size else 0.0


def trough_to_peak_ms(wave, sample_rate: float) -> Optional[float]:
    """Trough-to-peak width (ms), but ONLY for a biphasic, negative-going spike
    with a clear positive rebound after the trough; otherwise ``None``.

    The trough-to-peak time is only meaningful for that canonical shape, so it is
    gated: the trough must be an interior minimum AND the dominant excursion
    (negative-going), and the post-trough maximum must be an interior peak the
    waveform actually turns back down from AND rises above baseline. A monophasic
    or positive-going waveform (or one still rising at the window edge) returns
    ``None`` rather than an edge-pinned, meaningless number.

    Lives here, next to the waveform cutting, so the raster info panel and the MUA
    threshold calibration report the same number for the same waveform.
    """
    wave = np.asarray(wave, dtype=float)
    n = wave.size
    if n < 3:
        return None
    trough = int(np.argmin(wave))
    if not (0 < trough < n - 1):                  # a real, interior trough
        return None
    if abs(wave[trough]) < wave.max():            # the negative deflection must dominate
        return None
    seg = wave[trough:]
    pk_rel = int(np.argmax(seg))
    if pk_rel == 0 or pk_rel == seg.size - 1:     # no interior rebound peak captured
        return None
    if wave[trough + pk_rel] <= 0:                # rebound must rise above baseline
        return None
    return pk_rel / sample_rate * 1000.0


@dataclass
class Footprint:
    """Spatial footprint of a unit: its mean waveform on every channel."""
    uid: str
    channels: List[str]                 # ordered by physical contact index
    waveforms: np.ndarray               # (n_channels, 2*radius)
    peak_channel: str
    peak_amplitude: float               # peak-to-peak on the peak channel
    sample_rate: Optional[float] = None  # Hz, so a real time axis can be labelled


def compute_footprint(
    uid: str,
    spike_indices: np.ndarray,
    voltages_by_channel: Dict[str, np.ndarray],
    *,
    radius: int = DEFAULT_RADIUS,
    max_spikes: int = 500,
    sample_rate: Optional[float] = None,
) -> Footprint:
    """Average this unit's spikes across *all* channels to get its footprint.

    Channels are ordered by physical contact index so the footprint reads like a
    depth profile.
    """
    from . import probe_geometry as geom

    channels = sorted(
        voltages_by_channel.keys(),
        key=lambda c: (geom.contact_index_of(c) if geom.contact_index_of(c) is not None else 1e9),
    )
    waveforms = np.stack([
        mean_waveform(voltages_by_channel[c], spike_indices, radius=radius, max_spikes=max_spikes)
        for c in channels
    ])
    p2p = waveforms.max(axis=1) - waveforms.min(axis=1)
    peak_i = int(np.argmax(p2p))
    return Footprint(uid=uid, channels=channels, waveforms=waveforms,
                     peak_channel=channels[peak_i], peak_amplitude=float(p2p[peak_i]),
                     sample_rate=sample_rate)


def footprint_similarity(fa: Footprint, fb: Footprint) -> float:
    """Cosine similarity between two footprints (flattened across channels).

    ~1.0 → same spatial signature → same neuron. Requires the two footprints to
    share the same channel ordering (they will if built from the same
    ``voltages_by_channel``).
    """
    a = fa.waveforms.ravel()
    b = fb.waveforms.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
