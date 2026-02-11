"""
spike_rate_plot.py — Spike rate over time (PSTH-style) plots.

x-axis: time (seconds relative to epoch start)
y-axis: firing rate (Hz)

Works with any SpikeSource.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from analyses.enums.monkey_names import get_monkeys_by_rank


# ── Core binning ─────────────────────────────────────────────────────────────

def _bin_spikes_to_rate(spike_times, epoch_start, epoch_stop, bin_size):
    """
    Bin spike times into firing-rate bins (Hz) for a single trial.

    Returns
    -------
    bin_centers : np.ndarray  (seconds, relative to epoch start)
    rates       : np.ndarray  (Hz per bin)
    """
    duration = epoch_stop - epoch_start
    n_bins = max(1, int(np.ceil(duration / bin_size)))
    edges = np.linspace(0, duration, n_bins + 1)

    aligned = np.array([s - epoch_start for s in spike_times
                        if epoch_start <= s < epoch_stop])

    counts, _ = np.histogram(aligned, bins=edges)
    rates = counts / bin_size
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, rates


def compute_trial_rates(data, spike_col, bin_size=0.05):
    """
    Compute per-trial firing rates in time bins.

    Parameters
    ----------
    data : pd.DataFrame
        Must have columns: spike_col, 'EpochStartStop'.
    spike_col : str
    bin_size : float
        Bin width in seconds (default 50 ms).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (bin_centers, rates) per trial.
    """
    results = []
    for _, row in data.iterrows():
        start, stop = row["EpochStartStop"]
        spikes = row[spike_col]
        centers, rates = _bin_spikes_to_rate(spikes, start, stop, bin_size)
        results.append((centers, rates))
    return results


# ── Single-neuron spike rate plot ────────────────────────────────────────────

def plot_spike_rate(data, spike_col, *,
                    bin_size=0.05,
                    order_by_rank=True,
                    title=None,
                    save_path=None):
    """
    Spike rate over time, one line per monkey (mean +/- SEM across trials),
    grouped by MonkeyGroup.

    Parameters
    ----------
    data : pd.DataFrame
        Exploded spike data (one row per trial). Must have:
        'MonkeyGroup', 'MonkeyName', 'EpochStartStop', and spike_col.
    spike_col : str
        Column with per-trial spike time lists.
    bin_size : float
        Bin width in seconds (default 50 ms).
    order_by_rank : bool
    title : str, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    groups = data["MonkeyGroup"].dropna().unique().tolist()
    n_groups = len(groups)

    max_monkeys = 0
    for g in groups:
        n = data[data["MonkeyGroup"] == g]["MonkeyName"].nunique()
        max_monkeys = max(max_monkeys, n)

    fig, axes = plt.subplots(max_monkeys, n_groups,
                             figsize=(5 * n_groups, 2.5 * max_monkeys),
                             squeeze=False, sharex=True)

    for col_idx, group_name in enumerate(groups):
        group_data = data[data["MonkeyGroup"] == group_name]
        unique_monkeys = group_data["MonkeyName"].dropna().unique().tolist()

        if order_by_rank:
            ranked = get_monkeys_by_rank(group_name)
            monkey_list = [m for m in ranked if m in unique_monkeys]
        else:
            monkey_list = unique_monkeys

        for row_idx, monkey_name in enumerate(monkey_list):
            ax = axes[row_idx, col_idx]
            monkey_data = group_data[group_data["MonkeyName"] == monkey_name]
            trial_rates = compute_trial_rates(monkey_data, spike_col, bin_size)

            if not trial_rates:
                continue

            # Align all trials to the shortest common length
            min_len = min(len(r) for _, r in trial_rates)
            rate_matrix = np.array([r[:min_len] for _, r in trial_rates])
            centers = trial_rates[0][0][:min_len]

            mean_rate = np.mean(rate_matrix, axis=0)
            sem = np.std(rate_matrix, axis=0) / np.sqrt(rate_matrix.shape[0])

            ax.plot(centers, mean_rate, color="steelblue", linewidth=1.2)
            ax.fill_between(centers, mean_rate - sem, mean_rate + sem,
                            color="steelblue", alpha=0.25)
            ax.set_ylabel(monkey_name, fontsize=9, rotation=0, labelpad=40, va="center")
            ax.text(0.98, 0.92, f"n={len(trial_rates)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)

        # Blank out unused rows
        for row_idx in range(len(monkey_list), max_monkeys):
            axes[row_idx, col_idx].set_visible(False)

        axes[0, col_idx].set_title(group_name, fontsize=11)

    # Shared labels
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
    fig.text(0.02, 0.5, "Firing Rate (Hz)", ha="center", va="center",
             rotation="vertical", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])

    _save_or_show(fig, save_path)
    return fig


# ── Group-average spike rate ────────────────────────────────────────────────

def plot_spike_rate_by_group(data, spike_col, *,
                             bin_size=0.05,
                             title=None,
                             save_path=None):
    """
    One line per MonkeyGroup (mean +/- SEM across all trials in that group).
    """
    groups = data["MonkeyGroup"].dropna().unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    for group_name in groups:
        group_data = data[data["MonkeyGroup"] == group_name]
        trial_rates = compute_trial_rates(group_data, spike_col, bin_size)
        if not trial_rates:
            continue

        min_len = min(len(r) for _, r in trial_rates)
        rate_matrix = np.array([r[:min_len] for _, r in trial_rates])
        centers = trial_rates[0][0][:min_len]

        mean_rate = np.mean(rate_matrix, axis=0)
        sem = np.std(rate_matrix, axis=0) / np.sqrt(rate_matrix.shape[0])

        ax.plot(centers, mean_rate, linewidth=1.5, label=group_name)
        ax.fill_between(centers, mean_rate - sem, mean_rate + sem, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.legend()
    if title:
        ax.set_title(title, fontsize=13)

    _save_or_show(fig, save_path)
    return fig


# ── Multi-unit spike rate (overlaid) ────────────────────────────────────────

_UNIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_multiunit_spike_rate(unit_dataframes, *,
                              bin_size=0.05,
                              title=None,
                              save_path=None):
    """
    Overlay spike-rate traces for multiple units (mean across all trials per unit).

    Parameters
    ----------
    unit_dataframes : dict[str, pd.DataFrame]
        label -> exploded DataFrame with 'SpikeTimes' and 'EpochStartStop'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, udf) in enumerate(unit_dataframes.items()):
        color = _UNIT_COLORS[i % len(_UNIT_COLORS)]
        trial_rates = compute_trial_rates(udf, "SpikeTimes", bin_size)
        if not trial_rates:
            continue

        min_len = min(len(r) for _, r in trial_rates)
        rate_matrix = np.array([r[:min_len] for _, r in trial_rates])
        centers = trial_rates[0][0][:min_len]

        mean_rate = np.mean(rate_matrix, axis=0)
        sem = np.std(rate_matrix, axis=0) / np.sqrt(rate_matrix.shape[0])

        ax.plot(centers, mean_rate, color=color, linewidth=1.5, label=label)
        ax.fill_between(centers, mean_rate - sem, mean_rate + sem,
                        color=color, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=13)

    _save_or_show(fig, save_path)
    return fig


# ── Batch plotting via SpikeSource ──────────────────────────────────────────

def plot_spike_rates_for_source(source, date, round_no, *,
                                bin_size=0.05,
                                save=False,
                                save_dir=None,
                                min_trials=7):
    """
    Plot spike rate over time for every neuron from a SpikeSource.
    """
    df = source.load(date, round_no)
    if df is None:
        print(f"No data from {source.name} for {date} round {round_no}")
        return

    for neuron_id in df["NeuronID"].dropna().unique():
        neuron_df = df[df["NeuronID"] == neuron_id]
        if len(neuron_df) < min_trials:
            continue

        save_path = None
        if save and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{neuron_id}_rate.png")

        plot_spike_rate(
            neuron_df, "SpikeTimes",
            bin_size=bin_size,
            title=f"Spike Rate: {neuron_id}",
            save_path=save_path,
        )


# ── Helper ──────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
