
"""
single_unit_plots.py
"""

import os
import pickle

import pandas as pd
from clat.intan.rhd import load_intan_rhd_format

from analyses.raster_plotting import plot_raster_by_group


def main():
    date = "2023-10-05"
    round_name = "231005_round2"
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = os.path.join(cortana_path, date, round_name)

    sorted_data = read_sorted_data(round_path)

    for unit in sorted_data["SpikeTimes"].iloc[0]:
        plot_raster_for_unit(sorted_data, unit, experiment_name=round_name)


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_manually_sorted_spikes(path):
    """Load a dict of manually sorted spike indices from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


# ── Data loading ─────────────────────────────────────────────────────────────

def read_sorted_data(round_path,
                     manually_sorted_spikes_filename="sorted_spikes.pkl",
                     compiled_trials_filename="compiled.pkl"):
    """Load compiled trials and attach per-unit spike timestamps."""
    raw = pd.read_pickle(os.path.join(round_path, compiled_trials_filename)).reset_index(drop=True)
    sorted_spikes = load_manually_sorted_spikes(os.path.join(round_path, manually_sorted_spikes_filename))

    rhd_path = os.path.join(round_path, "info.rhd")
    sample_rate = load_intan_rhd_format.read_data(rhd_path)["frequency_parameters"]["amplifier_sample_rate"]

    return calculate_spike_timestamps(raw, sorted_spikes, sample_rate)


def calculate_spike_timestamps(df, spike_indices_by_unit_by_channel, sample_rate):
    """
    Add a 'SpikeTimes' column: dict mapping unit names to spike-time lists
    filtered to each trial's epoch.
    """
    def _for_row(epoch_start_stop):
        epoch_start, epoch_stop = epoch_start_stop
        result = {}
        for channel, units in reversed(spike_indices_by_unit_by_channel.items()):
            for unit_name, spike_indices in units.items():
                key = f"{channel}_{unit_name}"
                result[key] = [
                    idx / sample_rate
                    for idx in spike_indices
                    if epoch_start <= idx / sample_rate < epoch_stop
                ]
        return result

    out = df.copy(deep=True)
    out["SpikeTimes"] = out["EpochStartStop"].apply(_for_row)
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_raster_for_unit(data, unit, experiment_name=None, save=False):
    """Plot a ranked raster for a single sorted unit."""
    unit_data = data.copy()
    spike_col = f"SpikeTimes_{unit}"
    unit_data[spike_col] = data["SpikeTimes"].apply(lambda x: x.get(unit, []))

    save_path = None
    if experiment_name and save:
        save_dir = os.path.join("/png_raster_plots/", experiment_name)
        save_path = os.path.join(save_dir, f"{experiment_name}_{unit}.png")

    plot_raster_by_group(
        unit_data, spike_col,
        title=f"Raster Plots (by Rank): {unit}",
        save_path=save_path,
    )
