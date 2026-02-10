"""
raster_plotting.py — Reusable raster plot grouped by MonkeyGroup, ordered by rank.
"""

import os
from matplotlib import pyplot as plt
from analyses.enums.monkey_names import get_monkeys_by_rank


def plot_raster_by_group(data, spike_col, *,
                         order_by_rank=True,
                         xlim=2.2,
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

            aligned_spikes_list = _align_spikes_to_epoch(monkey_data, spike_col)

            ax.eventplot(aligned_spikes_list, color="black", linewidths=1)
            ax.set_xlim(0, xlim)
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


def _align_spikes_to_epoch(monkey_data, spike_col):
    """Align spike times to epoch start for each trial."""
    aligned = []
    for _, row in monkey_data.iterrows():
        spikes = row[spike_col]
        start, stop = row["EpochStartStop"]
        trial_spikes = [s - start for s in spikes if start <= s <= stop]
        aligned.append(trial_spikes)
    return aligned