"""
single_channel_plots.py — Channel-level raster plots and spike-rate histograms.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from clat.intan.channels import Channel
from matplotlib import pyplot as plt

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.raster_plotting import plot_raster_by_group
from analyses.spike_utils import calculate_spike_rate


def main():
    date = "2023-10-04"
    round_no = 2
    metadata = RecordingMetadataReader()
    pkl_name = metadata.get_pickle_filename_for_specific_round(date, round_no)
    channels = metadata.get_valid_channels(date, round_no)

    file_path = (Path(__file__).parent / ".." / ".." / ".." / "Cortana" / "compiled" / pkl_name).resolve()
    raw_data = pd.read_pickle(file_path)

    for channel in channels:
        print(f"Working on channel {channel}")
        plot_raster_for_channel(raw_data, channel, date, round_no, save=True)


# ── Data extraction ──────────────────────────────────────────────────────────

def extract_target_channel_data(channel: Channel, data):
    """Add a SpikeTimes_{channel} column extracted from the SpikeTimes dict."""
    out = data.copy()
    out[f"SpikeTimes_{channel.value}"] = data["SpikeTimes"].apply(
        lambda x: x[next(filter(lambda k: k.value == channel.value, x.keys()), None)]
    )
    return out


# ── Raster plotting ──────────────────────────────────────────────────────────

def _raster_save_path(date, round_no, channel):
    date_fmt = date.replace("-", "")[2:]
    folder = f"{date_fmt}_round{round_no}_new"
    save_dir = "/mixed_manual_raster_plots/"
    return os.path.join(save_dir, folder, f"{date}_round{round_no}_{channel}.png")


def plot_raster_for_channel(raw_data, channel, date, round_no, save=False):
    """Plot a ranked raster for a single channel."""
    channel_data = extract_target_channel_data(channel, raw_data)
    spike_col = f"SpikeTimes_{channel.value}"

    save_path = _raster_save_path(date, round_no, channel) if save else None

    plot_raster_by_group(
        channel_data, spike_col,
        title=f"Raster Plots (by Rank): {date} Round {round_no}: {channel.value}",
        save_path=save_path,
    )


# ── Spike rate binning ───────────────────────────────────────────────────────

def calculate_binned_spike_rate(spikes, epoch, num_bins):
    if spikes is None or epoch is None:
        return [0.0] * num_bins
    start, end = epoch
    bin_dur = (end - start) / num_bins
    return np.array([
        calculate_spike_rate(spikes, (start + i * bin_dur, start + (i + 1) * bin_dur))
        for i in range(num_bins)
    ])


def calculate_spikerates_per_bin(channel_data, channel, num_bins):
    channel_data["BinnedSpikeRates"] = channel_data.apply(
        lambda row: calculate_binned_spike_rate(
            row[f"SpikeTimes_{channel.value}"], row["EpochStartStop"], num_bins
        ),
        axis=1,
    )
    return channel_data


# ── Histogram plotting ───────────────────────────────────────────────────────

def plot_channel_histograms(data, channel, num_bins=10):
    channel_data = extract_target_channel_data(channel, data)
    channel_data = calculate_spikerates_per_bin(channel_data, channel, num_bins)

    plot_histograms_for_individual_monkeys(channel_data, channel)
    plot_average_among_groups(channel_data, channel)
    plt.show()


def plot_histograms_for_individual_monkeys(channel_data, channel):
    groups = channel_data["MonkeyGroup"].dropna().unique().tolist()
    n_groups = len(groups)
    N = len(channel_data)
    num_bins = len(channel_data["BinnedSpikeRates"].iloc[0])
    x_prop = np.linspace(0, 1, num_bins)
    ymax = 50

    fig = plt.figure(figsize=(15, 10))
    legend_info = None

    for row_idx, group_name in enumerate(groups):
        group_data = channel_data[channel_data["MonkeyGroup"] == group_name]
        unique_monkeys = group_data["MonkeyName"].dropna().unique().tolist()

        for col_idx, monkey_name in enumerate(unique_monkeys):
            monkey_data = group_data[group_data["MonkeyName"] == monkey_name]
            ax = fig.add_subplot(n_groups, len(unique_monkeys),
                                 row_idx * len(unique_monkeys) + col_idx + 1)

            err_bar = _plot_single_monkey_histogram(ax, monkey_data, monkey_name, x_prop, ymax)
            if legend_info is None:
                legend_info = ([err_bar], ["Mean"])

        subplot_h = 1 / n_groups
        fig.text(0.08, 1 - (row_idx * subplot_h + subplot_h / 2),
                 group_name, ha="center", va="center", rotation="vertical")

    fig.text(0.5, 0.04, "Proportion of Total Time", ha="center")
    fig.text(0.04, 0.5, "Spike Rate", ha="center", va="center", rotation="vertical")
    fig.text(0.99, 0.95, f"N: {N}", ha="right", va="bottom")
    fig.suptitle(f"Individual Monkey Spike Rates: Channel: {channel.value}")
    if legend_info:
        fig.legend(*legend_info, loc="upper right")
    plt.subplots_adjust(hspace=0.5, wspace=1.0)
    fig.canvas.mpl_connect("button_press_event", _on_click_histo)
    return fig


def _plot_single_monkey_histogram(ax, monkey_data, monkey_name, x_prop, ymax):
    rates = np.vstack(monkey_data["BinnedSpikeRates"])
    n_traces = rates.shape[0]
    mean = np.mean(rates, axis=0)
    std = np.std(rates, axis=0)

    for row in rates:
        ax.plot(x_prop, row, color="black", alpha=0.3, linewidth=0.75)
    err = ax.errorbar(x_prop, mean, yerr=std, color="orange", alpha=0.75, label="Mean", elinewidth=0.8)
    ax.set_ylim(0, ymax)
    ax.set_title(monkey_name)
    ax.text(0.5, 0.85, f"n={n_traces}", transform=ax.transAxes, ha="center", va="bottom")
    return err


def _on_click_histo(event):
    ax = event.inaxes
    if ax is None:
        return
    fig, new_ax = plt.subplots()
    for line in ax.lines[:-1]:
        new_ax.plot(line.get_xdata(), line.get_ydata(), color="gray", alpha=0.3)
    mean_line = ax.lines[-1]
    new_ax.errorbar(mean_line.get_xdata(), mean_line.get_ydata(), color="orange", alpha=0.75, label="Mean")
    new_ax.set_title(ax.get_title())
    new_ax.set_ylim(ax.get_ylim())
    plt.show()


def plot_average_among_groups(channel_data, channel):
    grouped = channel_data.groupby("MonkeyGroup")["BinnedSpikeRates"].apply(np.vstack).reset_index()
    num_bins = len(channel_data["BinnedSpikeRates"].iloc[0])
    x_prop = np.linspace(0, 1, num_bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in grouped.iterrows():
        rates = row["BinnedSpikeRates"]
        mean = np.mean(rates, axis=0)
        sem = np.std(rates, axis=0) / np.sqrt(rates.shape[0])
        ax.errorbar(x_prop, mean, yerr=sem, label=f"Group: {row['MonkeyGroup']}", alpha=0.75)

    ax.set_xlabel("Proportion of Total Time")
    ax.set_ylabel("Average Spike Rate")
    ax.set_title(f"Average Spike Rates Among Groups: Channel: {channel.value}")
    ax.legend()
    return fig


if __name__ == "__main__":
    main()
