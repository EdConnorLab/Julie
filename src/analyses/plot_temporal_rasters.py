"""
plot_psth.py — Population PSTH heatmaps for temporal coding analysis.

Neurons on y-axis, time bins on x-axis, color = firing rate.
Flexible scope and sorting options.
"""

from pathlib import Path
import glob
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from analyses.enums.monkey_names import get_monkeys_by_rank
from project_util import SUBJECT_MONKEY, DATA_BASE_PATH

SAVE_DIR = Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "psth_heatmaps"


def plot_psth_heatmap(df: pd.DataFrame, *,
                      scope="group",
                      sort_by="peak_latency",
                      bin_ms=25,
                      xlim=2.2,
                      order_by_rank=True,
                      save=False):
    """
    Plot population PSTH heatmaps.

    Parameters
    ----------
    df : pd.DataFrame
        All trials across sessions. Must contain columns:
        'NeuronID', 'MonkeyGroup', 'MonkeyName', 'EpochStartStop', 'SpikeTimes'.
    scope : {"stimulus", "group", "grand"}
        "stimulus"  — one heatmap per stimulus monkey
        "group"     — one heatmap per MonkeyGroup (averaged across stimuli)
        "grand"     — one heatmap across all stimuli (grand average)
    sort_by : {"peak_latency", "firing_rate", "region"}
        "peak_latency"  — sort by time of peak firing (reveals sequential activation)
        "firing_rate"   — sort by overall firing rate (highest at top)
        "region"        — group by brain region (AMG vs ER), then by firing rate within
    bin_ms : int
        Bin width in milliseconds.
    xlim : float
        X-axis limit in seconds.
    order_by_rank : bool
        Rank-order stimulus monkeys within each group (for scope="stimulus").
    save : bool
        Save figures to SAVE_DIR.
    """

    bin_s = bin_ms / 1000.0
    n_bins = int(xlim / bin_s)
    bin_edges = np.linspace(0, xlim, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    neuron_ids = sorted(df["NeuronID"].dropna().unique().tolist())
    n_neurons = len(neuron_ids)
    print(f"{n_neurons} neurons, scope={scope}, sort_by={sort_by}")

    # ── Build PSTH matrices ──────────────────────────────────────────────
    # psth_dict[(group, monkey)] = array of shape (n_neurons, n_bins)
    # Each entry is the trial-averaged firing rate for that neuron × condition.
    groups = df["MonkeyGroup"].dropna().unique().tolist()

    psth_dict = {}
    for group_name in groups:
        group_df = df[df["MonkeyGroup"] == group_name]
        unique_monkeys = group_df["MonkeyName"].dropna().unique().tolist()
        if order_by_rank:
            ranked = get_monkeys_by_rank(group_name)
            unique_monkeys = [m for m in ranked if m in unique_monkeys]

        for monkey_name in unique_monkeys:
            mat = np.full((n_neurons, n_bins), np.nan)
            for n_idx, nid in enumerate(neuron_ids):
                trial_df = df[
                    (df["NeuronID"] == nid) &
                    (df["MonkeyGroup"] == group_name) &
                    (df["MonkeyName"] == monkey_name)
                ]
                if trial_df.empty:
                    continue
                counts = _bin_spikes(trial_df, bin_edges)  # (n_trials, n_bins)
                # mean firing rate in Hz
                mat[n_idx, :] = counts.mean(axis=0) / bin_s
            psth_dict[(group_name, monkey_name)] = mat

    # ── Assemble heatmap(s) based on scope ───────────────────────────────
    if scope == "stimulus":
        heatmaps = {}
        for (group_name, monkey_name), mat in psth_dict.items():
            heatmaps[f"{group_name} — {monkey_name}"] = mat

    elif scope == "group":
        heatmaps = {}
        for group_name in groups:
            mats = [v for (g, _), v in psth_dict.items() if g == group_name]
            stacked = np.stack(mats, axis=0)  # (n_stimuli, n_neurons, n_bins)
            heatmaps[group_name] = np.nanmean(stacked, axis=0)

    elif scope == "grand":
        all_mats = list(psth_dict.values())
        stacked = np.stack(all_mats, axis=0)
        heatmaps["All stimuli (grand average)"] = np.nanmean(stacked, axis=0)

    else:
        raise ValueError(f"Unknown scope: {scope!r}")

    # ── Sort neurons ─────────────────────────────────────────────────────
    # Compute sort order once from the grand-average PSTH so ordering is
    # consistent across all heatmaps in this call.
    all_mats = list(psth_dict.values())
    grand_mat = np.nanmean(np.stack(all_mats, axis=0), axis=0)  # (n_neurons, n_bins)

    sort_order = _sort_neurons(neuron_ids, grand_mat, sort_by)
    sorted_ids = [neuron_ids[i] for i in sort_order]

    # ── Plot ─────────────────────────────────────────────────────────────
    for title_str, mat in heatmaps.items():
        sorted_mat = mat[sort_order, :]

        fig, ax = plt.subplots(figsize=(10, max(4, n_neurons * 0.06 + 1)))
        im = ax.imshow(
            sorted_mat,
            aspect="auto",
            origin="upper",
            extent=[0, xlim, n_neurons, 0],
            cmap="inferno",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Firing rate (Hz)", fontsize=9)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Neuron", fontsize=10)

        # Y-tick labels: show a subset to avoid clutter
        tick_step = max(1, n_neurons // 30)
        ytick_pos = list(range(0, n_neurons, tick_step))
        ytick_labels = [_short_label(sorted_ids[i]) for i in ytick_pos]
        ax.set_yticks([p + 0.5 for p in ytick_pos])
        ax.set_yticklabels(ytick_labels, fontsize=5)

        # If sorted by region, draw a separator line
        if sort_by == "region":
            _draw_region_separator(ax, sorted_ids, n_neurons)

        ax.set_title(
            f"{title_str} — sorted by {sort_by} — {bin_ms}ms bins",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_title = title_str.replace(" ", "_").replace("—", "-")
            fname = SAVE_DIR / f"psth_{safe_title}_{sort_by}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved → {fname}")
            plt.close(fig)
        else:
            plt.show()


# ── Sorting helpers ──────────────────────────────────────────────────────────

def _sort_neurons(neuron_ids, grand_mat, sort_by):
    """Return index array that sorts neurons according to sort_by."""
    n_neurons = len(neuron_ids)

    if sort_by == "peak_latency":
        # Sort by the time bin of peak firing rate
        peak_bins = np.nanargmax(grand_mat, axis=1)
        # Tie-break by peak amplitude (descending)
        peak_vals = np.nanmax(grand_mat, axis=1)
        order = np.lexsort((-peak_vals, peak_bins))

    elif sort_by == "firing_rate":
        mean_rates = np.nanmean(grand_mat, axis=1)
        order = np.argsort(mean_rates)[::-1]  # highest first

    elif sort_by == "region":
        # Infer region from NeuronID prefix (e.g. "AMG_..." or "ER_...")
        regions = [_infer_region(nid) for nid in neuron_ids]
        mean_rates = np.nanmean(grand_mat, axis=1)
        # Sort by region name, then firing rate descending within region
        order = sorted(
            range(n_neurons),
            key=lambda i: (regions[i], -mean_rates[i]),
        )
        order = np.array(order)

    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    return order


def _infer_region(neuron_id):
    """Extract brain region from NeuronID prefix."""
    prefix = neuron_id.split("_")[0].upper()
    if prefix in ("AMG", "AMYGDALA"):
        return "AMG"
    elif prefix in ("ER", "EC", "ENTORHINAL"):
        return "ER"
    return prefix


def _draw_region_separator(ax, sorted_ids, n_neurons):
    """Draw horizontal line(s) between brain region blocks."""
    regions = [_infer_region(nid) for nid in sorted_ids]
    current = regions[0]
    for i in range(1, n_neurons):
        if regions[i] != current:
            ax.axhline(y=i, color="white", linewidth=1.5, linestyle="--")
            current = regions[i]


# ── Spike binning ────────────────────────────────────────────────────────────

def _bin_spikes(trial_df, bin_edges):
    """
    Bin aligned spike times for all trials.

    Returns
    -------
    counts : np.ndarray, shape (n_trials, n_bins)
    """
    all_counts = []
    for _, row in trial_df.iterrows():
        start, stop = row["EpochStartStop"]
        aligned = [s - start for s in row["SpikeTimes"] if start <= s <= stop]
        counts, _ = np.histogram(aligned, bins=bin_edges)
        all_counts.append(counts)
    return np.array(all_counts)


def _short_label(neuron_id):
    parts = neuron_id.split("_", 1)
    return parts[1] if len(parts) > 1 else neuron_id


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_trials(pkl_dir):
    """Load all pkl files and concatenate into a single DataFrame."""
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")

    dfs = []
    for f in pkl_files:
        with open(f, 'rb') as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
        else:
            print(f"  Skipping {os.path.basename(f)} — not a DataFrame")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(pkl_files)} files → {len(combined)} total trials")
    return combined


# ── CLI / quick-run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    PKL_DIR = str(Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "sorted_spike_cache_filtered")

    df = load_all_trials(PKL_DIR)
    df = df[df["MonkeyName"] != "NewMonkey"]
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Switch these as needed ───────────────────────────────────────────
    plot_psth_heatmap(
        df,
        scope="group",          # "stimulus" | "group" | "grand"
        sort_by="peak_latency", # "peak_latency" | "firing_rate" | "region"
        bin_ms=25,
        save=False,
    )