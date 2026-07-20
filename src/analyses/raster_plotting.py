"""
raster_plotting.py — Reusable raster plot grouped by MonkeyGroup, ordered by rank.
"""
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.enums.monkey_names import get_monkeys_by_rank

_UNIT_COLORS_FOR_RASTER = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]
def plot_raster_by_group(data, spike_col, *,
                         order_by_rank=True,
                         xlim=2.2,
                         pre_stimulus_time=0.0,
                         title=None,
                         save_path=None):
    """
    Plot raster plots of spike times grouped by MonkeyGroup, one subplot per monkey.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: 'MonkeyGroup', 'MonkeyName', 'EpochStartStop',
        and the column named by `spike_col`.
    spike_col : str
        Column name containing per-trial spike time lists.
    order_by_rank : bool
        If True, order monkeys within each group by dominance rank.
    xlim : float
        X-axis limit (seconds after epoch start).
    pre_stimulus_time : float
        Seconds of pre-stimulus baseline to show left of onset (default 0.0).
        Requires a cache built with a matching pre-stimulus window; the axis
        then spans [-pre_stimulus_time, xlim] with an onset marker at t=0.
    title : str, optional
        Figure super-title.
    save_path : str, optional
        If provided, save the figure to this path (parent dirs created automatically).

    Returns
    -------
    matplotlib.figure.Figure
    """
    groups = data["MonkeyGroup"].dropna().unique().tolist()
    n_groups = len(groups)
    N = len(data)

    # Determine grid dimensions
    max_rows = 0
    for group in groups:
        monkeys = data[data["MonkeyGroup"] == group]["MonkeyName"].dropna().unique()
        max_rows = max(max_rows, len(monkeys))

    fig = plt.figure(figsize = (4*n_groups + 2, 1.2*max_rows + 2))

    for col_idx, group_name in enumerate(groups):
        group_data = data[data["MonkeyGroup"] == group_name]
        unique_monkeys = group_data["MonkeyName"].dropna().unique().tolist()

        if order_by_rank:
            ranked = get_monkeys_by_rank(group_name)
            monkey_list = [m for m in ranked if m in unique_monkeys]
        else:
            monkey_list = unique_monkeys

        for row_idx, monkey_name in enumerate(monkey_list):
            monkey_data = group_data[group_data["MonkeyName"] == monkey_name]
            subplot_idx = row_idx * n_groups + col_idx + 1
            ax = fig.add_subplot(max_rows, n_groups, subplot_idx)

            aligned_spikes_list = _align_spikes_to_epoch(monkey_data, spike_col,
                                                         pre_stimulus_time=pre_stimulus_time)

            ax.eventplot(aligned_spikes_list, color="black", linewidths=1)
            _apply_prestim_time_axis(ax, xlim, pre_stimulus_time)
            ax.set_yticks([len(aligned_spikes_list)])
            ax.text(1.05, 0.5, monkey_name,
                    transform=ax.transAxes, ha="left", va="center", fontsize=14)

        fig.text(
            0.6 / n_groups + col_idx / n_groups, 0.92,
            group_name, ha="center", va="center",
        )

    fig.text(0.5, 0.05, "Time (s)", ha="center", va="center")
    fig.text(0.99, 0.95, f"N: {N}", ha="right", va="bottom")
    if title:
        fig.suptitle(title, fontsize=16)

    plt.subplots_adjust(hspace=1.0, wspace=1.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


# Multi-unit rasters (two modes)

def plot_multiunit_raster_overlaid(unit_dataframes, *,
                                   order_by_rank=True,
                                   xlim=2.2,
                                   pre_stimulus_time=0.0,
                                   title=None,
                                   save_path=None):
    """
    Overlay multiple units on the same axes with different colors.

    Parameters
    ----------
    unit_dataframes : dict[str, pd.DataFrame]
        Mapping of unit_label -> DataFrame.  Each DataFrame must have columns:
        'MonkeyGroup', 'MonkeyName', 'EpochStartStop', 'SpikeTimes'.
    order_by_rank : bool
    xlim : float
    title : str, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    unit_labels = list(unit_dataframes.keys())
    colors = {label: _UNIT_COLORS_FOR_RASTER[i % len(_UNIT_COLORS_FOR_RASTER)]
              for i, label in enumerate(unit_labels)}

    # Use the first unit's data to determine layout
    ref_data = next(iter(unit_dataframes.values()))
    groups = ref_data["MonkeyGroup"].dropna().unique().tolist()
    n_groups = len(groups)

    max_rows = 0
    for group in groups:
        monkeys = ref_data[ref_data["MonkeyGroup"] == group]["MonkeyName"].dropna().unique()
        max_rows = max(max_rows, len(monkeys))

    fig = plt.figure(figsize=(4 * n_groups + 2, 1.2 * max_rows + 2))

    for col_idx, group_name in enumerate(groups):
        ref_group = ref_data[ref_data["MonkeyGroup"] == group_name]
        unique_monkeys = ref_group["MonkeyName"].dropna().unique().tolist()

        if order_by_rank:
            ranked = get_monkeys_by_rank(group_name)
            monkey_list = [m for m in ranked if m in unique_monkeys]
        else:
            monkey_list = unique_monkeys

        for row_idx, monkey_name in enumerate(monkey_list):
            subplot_idx = row_idx * n_groups + col_idx + 1
            ax = fig.add_subplot(max_rows, n_groups, subplot_idx)

            for label, udf in unit_dataframes.items():
                monkey_data = udf[
                    (udf["MonkeyGroup"] == group_name) &
                    (udf["MonkeyName"] == monkey_name)
                ]
                if monkey_data.empty:
                    continue
                aligned = _align_spikes_to_epoch(monkey_data, "SpikeTimes",
                                                 pre_stimulus_time=pre_stimulus_time)
                ax.eventplot(aligned, color=colors[label], linewidths=0.8)

            _apply_prestim_time_axis(ax, xlim, pre_stimulus_time)
            ax.text(1.05, 0.5, monkey_name,
                    transform=ax.transAxes, ha="left", va="center", fontsize=14)

        fig.text(
            0.6 / n_groups + col_idx / n_groups, 0.92,
            group_name, ha="center", va="center",
        )

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[l], lw=2, label=l) for l in unit_labels]
    fig.legend(handles=handles, loc="upper right", fontsize=9)

    fig.text(0.5, 0.05, "Time (s)", ha="center", va="center")
    if title:
        fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=1.0, wspace=1.0)

    _save_or_show(fig, save_path)
    return fig


def plot_multiunit_raster_sidebyside(unit_dataframes, *,
                                     order_by_rank=True,
                                     xlim=2.2,
                                     pre_stimulus_time=0.0,
                                     title=None,
                                     save_path=None):
    """
    Side-by-side subplots — one column per unit, rows are monkeys (all groups pooled
    in rank order within each group).

    Parameters
    ----------
    unit_dataframes : dict[str, pd.DataFrame]
        Same format as plot_multiunit_raster_overlaid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    unit_labels = list(unit_dataframes.keys())
    n_units = len(unit_labels)
    colors = {label: _UNIT_COLORS_FOR_RASTER[i % len(_UNIT_COLORS_FOR_RASTER)]
              for i, label in enumerate(unit_labels)}

    # Build a consistent monkey ordering across all units
    ref_data = next(iter(unit_dataframes.values()))
    groups = ref_data["MonkeyGroup"].dropna().unique().tolist()

    monkey_order = []
    for group_name in groups:
        gd = ref_data[ref_data["MonkeyGroup"] == group_name]
        unique = gd["MonkeyName"].dropna().unique().tolist()
        if order_by_rank:
            ranked = get_monkeys_by_rank(group_name)
            unique = [m for m in ranked if m in unique]
        monkey_order.extend([(group_name, m) for m in unique])

    n_rows = len(monkey_order)
    fig, axes = plt.subplots(n_rows, n_units,
                             figsize=(4 * n_units + 1, 1.2 * n_rows + 2),
                             squeeze=False)

    for u_idx, label in enumerate(unit_labels):
        udf = unit_dataframes[label]
        axes[0, u_idx].set_title(label, fontsize=11)

        for r_idx, (group_name, monkey_name) in enumerate(monkey_order):
            ax = axes[r_idx, u_idx]
            monkey_data = udf[
                (udf["MonkeyGroup"] == group_name) &
                (udf["MonkeyName"] == monkey_name)
            ]
            if not monkey_data.empty:
                aligned = _align_spikes_to_epoch(monkey_data, "SpikeTimes",
                                                 pre_stimulus_time=pre_stimulus_time)
                ax.eventplot(aligned, color=colors[label], linewidths=0.8)

            _apply_prestim_time_axis(ax, xlim, pre_stimulus_time)
            if u_idx == 0:
                ax.set_ylabel(monkey_name, fontsize=9, rotation=0, labelpad=40, va="center")
            ax.set_yticks([])

    fig.text(0.5, 0.02, "Time (s)", ha="center", va="center")
    if title:
        fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    _save_or_show(fig, save_path)
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def _align_spikes_to_epoch(monkey_data, spike_col, pre_stimulus_time=0.0):
    """Align spike times to epoch start (stimulus onset) for each trial.

    Spikes from ``start - pre_stimulus_time`` through ``stop`` are kept and
    re-zeroed to the onset (``start``), so pre-stimulus spikes become negative
    times. ``pre_stimulus_time=0.0`` reproduces the original strict
    ``[start, stop]`` behavior exactly. Note that pre-stimulus spikes only exist
    in caches built with a matching pre-stimulus window; against a strict cache
    this simply finds nothing before onset.
    """
    aligned = []
    for _, row in monkey_data.iterrows():
        spikes = row[spike_col]
        start, stop = row["EpochStartStop"]
        trial_spikes = [s - start for s in spikes if start - pre_stimulus_time <= s <= stop]
        aligned.append(trial_spikes)
    return aligned


def _apply_prestim_time_axis(ax, xlim, pre_stimulus_time=0.0):
    """Set the raster time axis, extending left to reveal a pre-stimulus window.

    When ``pre_stimulus_time == 0`` this is identical to the original
    ``ax.set_xlim(0, xlim)``. When positive, the axis starts at
    ``-pre_stimulus_time`` and a dashed onset marker is drawn at t=0.
    """
    left = -pre_stimulus_time if pre_stimulus_time and pre_stimulus_time > 0 else 0
    ax.set_xlim(left, xlim)
    if pre_stimulus_time and pre_stimulus_time > 0:
        ax.axvline(0, color="k", lw=0.8, ls="--", zorder=0)  # stimulus onset