"""
spike_utils.py — Low-level spike computation functions used across the codebase.
"""


def calculate_spike_rate(spikes, time_range):
    """Spike rate (Hz) within a time range."""
    if not spikes or time_range[0] >= time_range[1]:
        return 0.0
    start, end = time_range
    count = sum(1 for s in spikes if start <= s < end)
    duration = end - start
    return count / duration if duration > 0 else 0.0


def get_spike_count(spikes, time_range):
    """Spike count within a time range."""
    if not spikes or time_range[0] >= time_range[1]:
        return 0
    start, end = time_range
    return sum(1 for s in spikes if start <= s < end)