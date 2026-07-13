"""
threshold_detection.py — threshold-based spike detection from raw amplifier.dat files.

Threshold = -multiplier * sigma_noise
sigma_noise = median(|signal|) / 0.6745    (robust median estimator)
"""

import numpy as np
from scipy.signal import butter, filtfilt


def estimate_noise_std(voltage):
    """Robust noise estimator (Quian Quiroga et al. 2004)."""
    return np.median(np.abs(voltage)) / 0.6745


def detect_spikes(voltage, sample_rate, *, threshold_multiplier=4,
                  refractory_ms=1.0):
    """
    Detect spikes via negative threshold crossing.

    Parameters
    ----------
    voltage : np.ndarray
        Highpass-filtered voltage trace (microvolts or raw int16).
    sample_rate : float
        Sampling rate in Hz.
    threshold_multiplier : float
        Number of sigma_noise for threshold (default 4 → -4*sigma).
    refractory_ms : float
        Minimum inter-spike interval in ms (dead-time after each detection).

    Returns
    -------
    spike_indices : np.ndarray[int]
        Sample indices of detected spikes.
    threshold : float
        The computed threshold value (negative).
    sigma_noise : float
        Estimated noise standard deviation.
    """
    sigma_noise = estimate_noise_std(voltage)
    threshold = -threshold_multiplier * sigma_noise

    below = np.where(voltage < threshold)[0]
    if len(below) == 0:
        return np.array([], dtype=int), threshold, sigma_noise

    refractory_samples = int(refractory_ms / 1000.0 * sample_rate)
    spike_indices = [below[0]]
    for idx in below[1:]:
        if idx - spike_indices[-1] >= refractory_samples:
            spike_indices.append(idx)

    return np.array(spike_indices, dtype=int), threshold, sigma_noise


def highpass_filter(data, sample_rate, cutoff=300, order=5):
    """Butterworth highpass filter (same as existing datahandler)."""
    nyquist = 0.5 * sample_rate
    b, a = butter(order, cutoff / nyquist, btype='high', analog=False)
    return filtfilt(b, a, data.astype(np.float64))


def detect_spikes_for_recording(voltages_by_channel, sample_rate, *,
                                threshold_multiplier=4,
                                apply_filter=True):
    """
    Run threshold detection on every channel in a recording.

    Parameters
    ----------
    voltages_by_channel : dict[Channel, np.ndarray]
        Raw (or preprocessed) voltage arrays keyed by Channel enum.
    sample_rate : float
    threshold_multiplier : float
    apply_filter : bool
        If True, apply 300 Hz highpass before detection.

    Returns
    -------
    spike_times_by_channel : dict[Channel, list[float]]
        Spike timestamps (seconds) per channel.
    detection_info : dict[Channel, dict]
        Per-channel threshold and sigma_noise for diagnostics.
    """
    spike_times_by_channel = {}
    detection_info = {}

    for channel, voltage in voltages_by_channel.items():
        v = voltage * 0.195 if voltage.dtype == np.int16 else voltage.copy()

        if apply_filter:
            v = highpass_filter(v, sample_rate)

        indices, thresh, sigma = detect_spikes(
            v, sample_rate, threshold_multiplier=threshold_multiplier,
        )

        spike_times_by_channel[channel] = (indices / sample_rate).tolist()
        detection_info[channel] = {"threshold": thresh, "sigma_noise": sigma,
                                   "n_spikes": len(indices)}

    return spike_times_by_channel, detection_info


# ---------------------------------------------------------------------------
# MAD/RMS multi-unit-activity (MUA) detector.
# Added for ThresholdMUASpikeSource. Ported from a labmate's offline pipeline
# (`_count_mad_negative_spikes`). Separate from the Quian-Quiroga detect_spikes
# above so nothing existing changes.
# ---------------------------------------------------------------------------
def estimate_noise(voltage, method='mad'):
    """Per-channel noise scale. 'mad' -> median(|v|)/0.6745 (robust sigma, same as
    estimate_noise_std); 'rms' -> sqrt(mean(v**2))."""
    if method == 'mad':
        return np.median(np.abs(voltage)) / 0.6745
    if method == 'rms':
        return float(np.sqrt(np.mean(np.square(voltage))))
    raise ValueError(f"unknown noise method {method!r}")


def detect_mad_spikes(voltage, threshold, refractory_samples):
    """Negative-going threshold detector: above->below crossings of `threshold` (<0),
    each snapped to the local trough, with a refractory period enforced between
    troughs. Faithful port of the offline `_count_mad_negative_spikes`, but returns
    the kept trough SAMPLE INDICES (np.ndarray[int]) instead of a count."""
    voltage = np.asarray(voltage)
    if voltage.size < 2:
        return np.array([], dtype=int)
    below = voltage < threshold
    crossings = np.where(np.diff(below.astype(np.int8)) == 1)[0] + 1
    if len(crossings) == 0:
        return np.array([], dtype=int)
    n = len(voltage)
    troughs = []
    for c in crossings:
        window_end = min(c + refractory_samples, n)
        troughs.append(c + int(np.argmin(voltage[c:window_end])))
    kept = [troughs[0]]
    for s in troughs[1:]:
        if s - kept[-1] >= refractory_samples:
            kept.append(s)
    return np.array(kept, dtype=int)


def detect_mad_spikes_for_recording(voltages_by_channel, sample_rate, *,
                                    noise_method='mad', threshold_multiplier=4.0,
                                    refractory_ms=1.0, apply_filter=True):
    """Run the MAD/RMS negative-crossing MUA detector on every channel.

    threshold = -threshold_multiplier * estimate_noise(v, noise_method).
    Returns (spike_times_by_channel: dict[Channel, list[float] seconds], info).
    Mirrors detect_spikes_for_recording's signature so a cache manager can swap
    detectors; it does NOT modify that function."""
    refractory_samples = max(1, int(refractory_ms / 1000.0 * sample_rate))
    spike_times_by_channel = {}
    detection_info = {}
    for channel, voltage in voltages_by_channel.items():
        v = voltage * 0.195 if voltage.dtype == np.int16 else voltage.astype(np.float64, copy=True)
        if apply_filter:
            v = highpass_filter(v, sample_rate)
        sigma = estimate_noise(v, noise_method)
        threshold = -threshold_multiplier * sigma
        idx = detect_mad_spikes(v, threshold, refractory_samples)
        spike_times_by_channel[channel] = (idx / sample_rate).tolist()
        detection_info[channel] = {"threshold": threshold, "noise": sigma,
                                   "noise_method": noise_method, "n_spikes": len(idx)}
    return spike_times_by_channel, detection_info