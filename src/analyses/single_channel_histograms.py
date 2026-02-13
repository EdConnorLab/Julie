"""
single_channel_histograms.py — Channel-level spike-rate histograms.
"""

import numpy as np
from matplotlib import pyplot as plt

from analyses.plotting_util import extract_target_channel_data
from analyses.spike_utils import calculate_spike_rate


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


### Histogram plotting

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
    pass