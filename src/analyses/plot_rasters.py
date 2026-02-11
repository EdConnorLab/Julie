"""
plot_rasters.py — Batch raster plotting using SpikeSource abstraction.
"""

import os
from pathlib import Path

import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.raster_plotting import plot_raster_by_group
from analyses.intan_data_processor.single_channel_analysis import plot_raster_for_channel
from data_access.spike_source import SpikeSource, MixedManualSpikeSource, SISortedSpikeSource


# ── Save helpers ─────────────────────────────────────────────────────────────

RASTER_SAVE_DIR = "/home/connorlab/Documents/GitHub/Julie/Cortana/mixed_manual_raster_plots/"


def _neuron_save_path(neuron_id):
    return os.path.join(RASTER_SAVE_DIR, f"{neuron_id}.png")


# ── Per-round entry point ────────────────────────────────────────────────────

def plot_rasters_for_round(date, round_no, channels=None, save=False):
    """Plot rasters for a round. If channels is None, uses curated channels from metadata."""
    metadata = RecordingMetadataReader()
    pkl_name = metadata.get_pickle_filename_for_specific_round(date, round_no)
    file_path = (Path(__file__).parent / ".." / ".." / "Cortana" / "compiled" / pkl_name).resolve()
    raw_data = pd.read_pickle(file_path)

    if channels is None:
        channels = metadata.get_curated_channels(date, round_no)

    for channel in channels:
        print(f"Working on channel {channel}")
        plot_raster_for_channel(raw_data, channel, date, round_no, save=save)

# ── Source-based batch plotting ──────────────────────────────────────────────

def plot_rasters_for_source(source: SpikeSource, date: str, round_no: int,
                            *, save=False, min_trials=7, min_spikes=None):
    """Plot rasters for every neuron from a SpikeSource for one session."""
    df = source.load(date, round_no)
    if df is None:
        print(f"No data from {source.name} for {date} round {round_no}")
        return

    for neuron_id in df["NeuronID"].dropna().unique():
        neuron_df = df[df["NeuronID"] == neuron_id]
        num_trials = len(neuron_df)
        total_spikes = sum(len(s) for s in neuron_df["SpikeTimes"])

        if num_trials < min_trials:
            print(f"Skipping {neuron_id}: only {num_trials} trials (min: {min_trials})")
            continue
        if min_spikes is not None and total_spikes < min_spikes:
            print(f"Skipping {neuron_id}: only {total_spikes} spikes (min: {min_spikes})")
            continue

        print(f"Plotting {neuron_id} — {num_trials} trials, {total_spikes} spikes")
        save_path = _neuron_save_path(neuron_id) if save else None
        plot_raster_by_group(
            neuron_df, "SpikeTimes",
            title=f"Raster Plot (by Rank): {neuron_id}",
            save_path=save_path,
        )


# ── Convenience wrappers ─────────────────────────────────────────────────────

def plot_mixed_manual_rasters(date, round_no, save=False, min_trials=7, min_spikes=400):
    source = MixedManualSpikeSource()
    plot_rasters_for_source(source, date, round_no, save=save, min_trials=min_trials, min_spikes=min_spikes)


def plot_si_sorted_rasters(date, round_no, save=False, min_trials=7):
    source = SISortedSpikeSource()
    plot_rasters_for_source(source, date, round_no, save=save, min_trials=min_trials)


if __name__ == "__main__":
    # plot_mixed_manual_rasters("2023-09-26", 2, save=True)
    plot_rasters_for_round("2023-09-26", 3, save=True)