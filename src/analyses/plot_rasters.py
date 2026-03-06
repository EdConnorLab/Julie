"""
plot_rasters.py — Batch raster plotting using SpikeSource abstraction.
"""

import os
from pathlib import Path

import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.plotting_util import extract_target_channel_data, raster_save_path
from analyses.raster_plotting import plot_raster_by_group, plot_multiunit_raster_sidebyside, \
    plot_multiunit_raster_overlaid
from data_access.spike_source import SpikeSource, MixedManualSpikeSource, SISortedSpikeSource, ThresholdSpikeSource
from project_util import SUBJECT_MONKEY, PROJECT_BASE_PATH

RASTER_SAVE_DIR = "raster_plots"
RASTER_BASE_PATH = Path(PROJECT_BASE_PATH) / SUBJECT_MONKEY / RASTER_SAVE_DIR


def _neuron_save_path(neuron_id):
    return os.path.join(RASTER_SAVE_DIR, f"{neuron_id}.png")


# Multi-unit comparison rasters
def plot_multiunit_rasters(source: SpikeSource, date: str, round_no: int,
                           base_channel: str, *,
                           mode="overlaid",
                           save=False,
                           title=None):
    """
    Plot rasters for all units on the same base channel, either overlaid or side-by-side.

    Parameters
    ----------
    source : SpikeSource
    date, round_no : session identifiers
    base_channel : str
        Base channel name (e.g. "C-003"). All units on this channel are plotted.
    mode : {"overlaid", "sidebyside"}
        "overlaid"   — all units on the same axes, different colors (default).
        "sidebyside" — one subplot column per unit.
    save : bool
    title : str, optional
    """
    df = source.load(date, round_no)
    print(df)
    if df is None:
        print(f"No data from {source.name} for {date} round {round_no}")
        return
    df["BaseChannel"] = df["BaseChannel"].astype(str)
    units_on_channel = df[df["BaseChannel"].str.contains(base_channel)]
    if units_on_channel.empty:
        print(f"No units found on base channel {base_channel}")
        return

    unit_ids = units_on_channel["NeuronID"].dropna().unique().tolist()
    print(f"Plotting {unit_ids}")
    unit_dfs = {}
    for uid in unit_ids:
        label = uid.rsplit("_", 1)[-1] if "_" in uid else uid
        unit_dfs[label] = df[df["NeuronID"] == uid]

    if not title:
        title = f"Multi-unit raster ({mode}): {base_channel} — {date} round {round_no}"

    save_path = None
    if save:
        save_path = os.path.join(RASTER_SAVE_DIR, f"{date}_round{round_no}_{base_channel}_{mode}.png")

    if mode == "sidebyside":
        plot_multiunit_raster_sidebyside(unit_dfs, title=title, save_path=save_path)
    else:
        plot_multiunit_raster_overlaid(unit_dfs, title=title, save_path=save_path)

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

def plot_raster_for_channel(raw_data, channel, date, round_no, save=False):
    """Plot a ranked raster for a single channel."""
    channel_data = extract_target_channel_data(channel, raw_data)
    spike_col = f"SpikeTimes_{channel.value}"

    save_path = raster_save_path(date, round_no, channel) if save else None

    plot_raster_by_group(
        channel_data, spike_col,
        title=f"Raster Plots (by Rank): {date} Round {round_no}: {channel.value}",
        save_path=save_path,
    )

def plot_rasters_for_round(date, round_no, channels=None, save=False):
    """Plot rasters for a round. If channels is None, uses curated channels from metadata."""
    metadata = RecordingMetadataReader()
    pkl_name = metadata.get_pickle_filename_for_specific_round(date, round_no)
    file_path = (Path(__file__).parent / ".." / ".." / "Cortana" / "compiled" / pkl_name).resolve()
    raw_data = pd.read_pickle(file_path)

    if channels is None:
        print(f"Fetching curated channels from metadata...")
        channels = metadata.get_curated_channels(date, round_no)

    for channel in channels:
        print(f"Working on channel {channel}")
        plot_raster_for_channel(raw_data, channel, date, round_no, save=save)

### Source-based batch plotting

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


# ---- SpikeSource Specific Rasters -----

def plot_mixed_manual_rasters(date, round_no, save=False, min_trials=7, min_spikes=400):
    source = MixedManualSpikeSource()
    plot_rasters_for_source(source, date, round_no, save=save, min_trials=min_trials, min_spikes=min_spikes)


def plot_si_sorted_rasters(date, round_no, save=False, min_trials=7):
    source = SISortedSpikeSource()
    plot_rasters_for_source(source, date, round_no, save=save, min_trials=min_trials)


if __name__ == "__main__":
    # plot_mixed_manual_rasters("2023-09-26", 2, save=True)
    # plot_rasters_for_round("2023-09-26", 3, save=True)
    # source = MixedManualSpikeSource()
    source= ThresholdSpikeSource()
    date = "2023-09-26"
    round_no = 3
    # plot_rasters_for_source(source, date, round_no)
    plot_rasters_for_round(date, round_no, save=False)
    # Overlaid (default) — all units on same axes, different colors
    # plot_multiunit_rasters(source, date, round_no, "C_005", mode="sidebyside")


    ### PLOT RASTERS FOR UNIT ------------------------------------------------------------------------
    # date = "2023-10-05"
    # round_name = "231005_round2"
    # cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    # round_path = os.path.join(cortana_path, date, round_name)
    #
    # sorted_data = read_sorted_data(round_path)
    #
    # for unit in sorted_data["SpikeTimes"].iloc[0]:
    #     plot_raster_for_unit(sorted_data, unit, experiment_name=round_name)


    ### PLOT RASTERS FOR CHANNEL-----------------------------------------------------------------------
    # date = "2023-10-04"
    # round_no = 2
    # metadata = RecordingMetadataReader()
    # pkl_name = metadata.get_pickle_filename_for_specific_round(date, round_no)
    # channels = metadata.get_valid_channels(date, round_no)
    #
    # file_path = (Path(__file__).parent / ".." / ".." / ".." / "Cortana" / "compiled" / pkl_name).resolve()
    # raw_data = pd.read_pickle(file_path)
    #
    # for channel in channels:
    #     print(f"Working on channel {channel}")
    #     plot_raster_for_channel(raw_data, channel, date, round_no, save=True)
