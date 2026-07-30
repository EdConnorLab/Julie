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
import matplotlib
matplotlib.use("Qt5Agg")
from analyses.enums.monkey_names import get_monkeys_by_rank
from project_util import SUBJECT_MONKEY, DATA_BASE_PATH

SAVE_DIR = Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "psth_heatmaps"


def plot_psth_heatmap(df: pd.DataFrame, *,
                      scope="group",
                      sort_by="peak_latency",
                      normalize="zscore",
                      neurons_per_fig=20,
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
    scope : {"stimulus", "stimulus_grid", "group", "grand"}
        "stimulus"      — one heatmap per stimulus monkey (separate figures)
        "stimulus_grid" — one figure per MonkeyGroup; columns = stimulus monkeys,
                          shared neuron order & color scale for direct comparison
        "group"         — one heatmap per MonkeyGroup (averaged across stimuli)
        "grand"         — one heatmap across all stimuli (grand average)
    sort_by : {"peak_latency", "firing_rate", "region"}
        "peak_latency"  — sort by time of peak firing (reveals sequential activation)
        "firing_rate"   — sort by overall firing rate (highest at top)
        "region"        — group by brain region (AMG vs ER), then by firing rate within
    normalize : {"zscore", "minmax", "none"}
        "zscore"  — z-score each neuron's PSTH (mean=0, std=1). Best for revealing
                    temporal profile shapes regardless of firing rate.
        "minmax"  — scale each neuron to [0, 1]. Preserves relative peak structure.
        "none"    — raw firing rate (Hz). High-FR neurons dominate the colormap.
    neurons_per_fig : int
        Max neurons per figure (~20). Neurons are split by brain region first
        (AMG vs ER), then chunked within each region.
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

    neuron_ids = sorted(df["NeuronID"].dropna().unique().tolist())
    neuron_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)
    print(f"{n_neurons} neurons, scope={scope}, sort_by={sort_by}")

    # ── Build PSTH matrices (vectorized) ─────────────────────────────────
    # Pre-align all spikes once, store as a numpy array per row
    aligned_spikes = []
    for _, row in df.iterrows():
        start, stop = row["EpochStartStop"]
        spikes = np.array(row["SpikeTimes"], dtype=np.float64)
        mask = (spikes >= start) & (spikes <= stop)
        aligned_spikes.append(spikes[mask] - start)
    df = df.copy()
    df["_aligned"] = aligned_spikes

    # Group once — avoids repeated boolean indexing on the full_dominance_first df
    grouped = df.groupby(["MonkeyGroup", "MonkeyName", "NeuronID"])

    # psth_dict[(group, monkey)] = array of shape (n_neurons, n_bins)
    groups = df["MonkeyGroup"].dropna().unique().tolist()

    # Accumulate per (group, monkey, neuron): sum of bin counts and trial count
    psth_sums = {}  # (group, monkey) -> (n_neurons, n_bins) cumulative counts
    psth_counts = {}  # (group, monkey) -> (n_neurons,) trial counts

    for (grp, mnk, nid), sub in grouped:
        if nid not in neuron_idx:
            continue
        n_idx = neuron_idx[nid]
        key = (grp, mnk)

        if key not in psth_sums:
            psth_sums[key] = np.zeros((n_neurons, n_bins), dtype=np.float64)
            psth_counts[key] = np.zeros(n_neurons, dtype=np.int64)

        # Concatenate all aligned spikes for this condition, histogram once per trial
        trial_sum = np.zeros(n_bins, dtype=np.float64)
        n_trials = 0
        for spk in sub["_aligned"]:
            counts, _ = np.histogram(spk, bins=bin_edges)
            trial_sum += counts
            n_trials += 1

        psth_sums[key][n_idx] += trial_sum
        psth_counts[key][n_idx] += n_trials

    # Convert to mean firing rate (Hz)
    psth_dict = {}
    for key in psth_sums:
        grp, mnk = key
        mat = np.full((n_neurons, n_bins), np.nan)
        valid = psth_counts[key] > 0
        mat[valid] = psth_sums[key][valid] / psth_counts[key][valid, None] / bin_s
        psth_dict[key] = mat

    # Clean up temp column
    df.drop(columns=["_aligned"], inplace=True)

    # ── Assemble heatmap(s) based on scope ───────────────────────────────
    if scope == "stimulus_grid":
        # Handled separately below — skip the normal heatmaps dict
        pass

    elif scope == "stimulus":
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

    # ── Sort neurons within each region, then chunk ────────────────────
    # Compute sort order from grand-average PSTH
    all_mats = list(psth_dict.values())
    grand_mat = np.nanmean(np.stack(all_mats, axis=0), axis=0)  # (n_neurons, n_bins)

    sort_order = _sort_neurons(neuron_ids, grand_mat, sort_by)
    sorted_ids = [neuron_ids[i] for i in sort_order]

    # Split sorted neurons by region
    region_groups = {}  # region_name -> [(sorted_idx_in_full, neuron_id), ...]
    for pos, full_idx in enumerate(sort_order):
        nid = neuron_ids[full_idx]
        region = _infer_region(nid)
        region_groups.setdefault(region, []).append((full_idx, nid))

    # Build chunks: split each region into pages of neurons_per_fig
    chunks = []  # list of (chunk_label, [full_indices], [neuron_ids])
    for region in sorted(region_groups.keys()):
        members = region_groups[region]
        for i in range(0, len(members), neurons_per_fig):
            batch = members[i:i + neurons_per_fig]
            page = i // neurons_per_fig + 1
            total_pages = (len(members) + neurons_per_fig - 1) // neurons_per_fig
            label = f"{region} ({page}/{total_pages})"
            chunks.append((label, [idx for idx, _ in batch],
                           [nid for _, nid in batch]))

    print(f"Split into {len(chunks)} figure chunks across "
          f"{len(region_groups)} regions")

    # ══════════════════════════════════════════════════════════════════════
    # STIMULUS GRID — one multi-column figure per group per chunk
    # ══════════════════════════════════════════════════════════════════════
    if scope == "stimulus_grid":
        for group_name in groups:
            # Ordered stimulus monkeys for this group
            stim_keys = [(g, m) for (g, m) in psth_dict if g == group_name]
            if not stim_keys:
                continue
            if order_by_rank:
                ranked = get_monkeys_by_rank(group_name)
                stim_keys = sorted(stim_keys,
                                   key=lambda k: ranked.index(k[1])
                                   if k[1] in ranked else 999)
            monkey_names = [m for _, m in stim_keys]
            n_cols = len(monkey_names)

            # Sort neurons using this GROUP's average (not grand average)
            group_mats = [psth_dict[k] for k in stim_keys]
            group_avg = np.nanmean(np.stack(group_mats, axis=0), axis=0)

            # Identify neurons that actually have data in this group
            has_data = ~np.all(np.isnan(group_avg), axis=1)
            group_sort_order = _sort_neurons(neuron_ids, group_avg, sort_by)
            # Push all-NaN neurons to the bottom
            group_sort_order = np.array(
                [i for i in group_sort_order if has_data[i]] +
                [i for i in group_sort_order if not has_data[i]]
            )
            group_sorted_ids = [neuron_ids[i] for i in group_sort_order]

            # Build chunks from this group-specific sort order
            grp_region_groups = {}
            for full_idx in group_sort_order:
                if not has_data[full_idx]:
                    continue  # skip neurons with no data in this group
                nid = neuron_ids[full_idx]
                region = _infer_region(nid)
                grp_region_groups.setdefault(region, []).append((full_idx, nid))

            grp_chunks = []
            for region in sorted(grp_region_groups.keys()):
                members = grp_region_groups[region]
                for i in range(0, len(members), neurons_per_fig):
                    batch = members[i:i + neurons_per_fig]
                    page = i // neurons_per_fig + 1
                    total_pages = (len(members) + neurons_per_fig - 1) // neurons_per_fig
                    label = f"{region} ({page}/{total_pages})"
                    grp_chunks.append((label, [idx for idx, _ in batch],
                                       [nid for _, nid in batch]))

            for chunk_label, chunk_indices, chunk_ids in grp_chunks:
                n_chunk = len(chunk_indices)

                # Gather per-stimulus matrices for this chunk
                col_mats = []
                for key in stim_keys:
                    col_mats.append(psth_dict[key][chunk_indices, :])

                # ── Normalize per neuron across ALL stimuli (shared scale) ─
                if normalize == "zscore":
                    # Compute mean/std from the concatenation of all stimuli
                    concat = np.concatenate(col_mats, axis=1)  # (n_chunk, n_bins*n_cols)
                    row_mean = np.nanmean(concat, axis=1, keepdims=True)
                    row_std = np.nanstd(concat, axis=1, keepdims=True)
                    row_std[row_std == 0] = 1
                    col_mats = [(m - row_mean) / row_std for m in col_mats]
                    cbar_label = "Firing rate (z-scored)"
                elif normalize == "minmax":
                    concat = np.concatenate(col_mats, axis=1)
                    row_min = np.nanmin(concat, axis=1, keepdims=True)
                    row_max = np.nanmax(concat, axis=1, keepdims=True)
                    row_range = row_max - row_min
                    row_range[row_range == 0] = 1
                    col_mats = [(m - row_min) / row_range for m in col_mats]
                    cbar_label = "Firing rate (min-max scaled)"
                else:
                    cbar_label = "Firing rate (Hz)"

                # Shared vmin/vmax across all columns
                all_vals = np.concatenate([m.ravel() for m in col_mats])
                vmin = np.nanpercentile(all_vals, 1)
                vmax = np.nanpercentile(all_vals, 99)

                fig, axes = plt.subplots(
                    1, n_cols,
                    figsize=(2.2 * n_cols + 3, max(3, n_chunk * 0.25 + 1)),
                    sharey=True,
                    gridspec_kw={"right": 0.92},
                )
                if n_cols == 1:
                    axes = [axes]

                for col_idx, (monkey_name, col_mat) in enumerate(
                        zip(monkey_names, col_mats)):
                    ax = axes[col_idx]
                    im = ax.imshow(
                        col_mat,
                        aspect="auto",
                        origin="upper",
                        extent=[0, xlim, n_chunk, 0],
                        cmap="inferno",
                        interpolation="nearest",
                        vmin=vmin, vmax=vmax,
                    )
                    ax.set_title(monkey_name, fontsize=9, pad=4)
                    ax.tick_params(axis="x", labelsize=7)
                    if col_idx == 0:
                        ytick_labels = [_short_label(nid) for nid in chunk_ids]
                        ax.set_yticks([i + 0.5 for i in range(n_chunk)])
                        ax.set_yticklabels(ytick_labels, fontsize=7)
                    if col_idx == n_cols // 2:
                        ax.set_xlabel("Time (s)", fontsize=9)

                # Place colorbar in dedicated space to the right
                cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.65])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(cbar_label, fontsize=9)

                norm_label = f" — {normalize}" if normalize != "none" else ""
                fig.suptitle(
                    f"{group_name} — {chunk_label}\n"
                    f"sorted by {sort_by}{norm_label} — {bin_ms}ms bins",
                    fontsize=11,
                )
                plt.subplots_adjust(wspace=0.05, top=0.88, bottom=0.08)

                if save:
                    SAVE_DIR.mkdir(parents=True, exist_ok=True)
                    safe_grp = group_name.replace(" ", "_")
                    safe_chunk = chunk_label.replace(" ", "_").replace("/", "of")
                    fname = SAVE_DIR / f"psth_grid_{safe_grp}_{safe_chunk}_{sort_by}_{normalize}.png"
                    fig.savefig(fname, dpi=150, bbox_inches="tight")
                    print(f"Saved → {fname}")
                    plt.close(fig)
                else:
                    plt.show()

        return  # stimulus_grid is fully handled above

    # ══════════════════════════════════════════════════════════════════════
    # STANDARD SCOPES (stimulus / group / grand)
    # ══════════════════════════════════════════════════════════════════════
    for title_str, mat in heatmaps.items():
        for chunk_label, chunk_indices, chunk_ids in chunks:
            chunk_mat = mat[chunk_indices, :]
            n_chunk = len(chunk_indices)

            # ── Normalize per neuron (row) ───────────────────────────
            if normalize == "zscore":
                row_mean = np.nanmean(chunk_mat, axis=1, keepdims=True)
                row_std = np.nanstd(chunk_mat, axis=1, keepdims=True)
                row_std[row_std == 0] = 1
                chunk_mat = (chunk_mat - row_mean) / row_std
                cbar_label = "Firing rate (z-scored)"
            elif normalize == "minmax":
                row_min = np.nanmin(chunk_mat, axis=1, keepdims=True)
                row_max = np.nanmax(chunk_mat, axis=1, keepdims=True)
                row_range = row_max - row_min
                row_range[row_range == 0] = 1
                chunk_mat = (chunk_mat - row_min) / row_range
                cbar_label = "Firing rate (min-max scaled)"
            else:
                cbar_label = "Firing rate (Hz)"

            fig, ax = plt.subplots(figsize=(10, max(3, n_chunk * 0.25 + 1)))
            im = ax.imshow(
                chunk_mat,
                aspect="auto",
                origin="upper",
                extent=[0, xlim, n_chunk, 0],
                cmap="inferno",
                interpolation="nearest",
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(cbar_label, fontsize=9)

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Neuron", fontsize=10)

            # Y-tick labels — every neuron is labeled at ~20 per fig
            ytick_labels = [_short_label(nid) for nid in chunk_ids]
            ax.set_yticks([i + 0.5 for i in range(n_chunk)])
            ax.set_yticklabels(ytick_labels, fontsize=7)

            norm_label = f" — {normalize}" if normalize != "none" else ""
            ax.set_title(
                f"{title_str} — {chunk_label}\n"
                f"sorted by {sort_by}{norm_label} — {bin_ms}ms bins",
                fontsize=11,
            )
            plt.tight_layout()

            if save:
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                safe_title = title_str.replace(" ", "_").replace("—", "-")
                safe_chunk = chunk_label.replace(" ", "_").replace("/", "of")
                fname = SAVE_DIR / f"psth_{safe_title}_{safe_chunk}_{sort_by}_{normalize}.png"
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


# ── Spike binning (now handled inline via groupby + np.histogram) ────────────


def _short_label(neuron_id):
    parts = neuron_id.split("_", 5)
    return parts[0] + parts[4] + parts[5] if len(parts) > 1 else neuron_id


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


if __name__ == "__main__":
    PKL_DIR = str(Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "sorted_spike_cache_filtered")

    df = load_all_trials(PKL_DIR)
    df = df[df["MonkeyName"] != "NewMonkey"]
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Switch these as needed ───────────────────────────────────────────
    plot_psth_heatmap(
        df,
        scope="group",          # "stimulus" | "group" | "grand" | "stimulus_grid"
        sort_by="peak_latency", # "peak_latency" | "firing_rate" | "region"
        normalize="zscore",     # "zscore" | "minmax" | "none"
        neurons_per_fig=500,
        bin_ms=25,
        save=False,
    )
