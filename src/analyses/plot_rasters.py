import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import cycle
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.enums.monkey_names import get_monkeys_by_rank
from analyses.intan_data_processor.single_channel_analysis import read_pickle, plot_raster_for_monkeys_by_rank, \
    save_raster_plots, save_raster_plots_by_neuron
import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.sorters as ss


def plot_rasters_for_specific_round(date, round_no):
    metadata_reader = RecordingMetadataReader()
    experiment_data_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_no)
    channels = metadata_reader.get_valid_channels(date, round_no)
    script_dir = Path(__file__).parent
    file_path = (script_dir / '..' / '..' / '..' / 'Cortana' / 'compiled' / experiment_data_filename).resolve()
    raw_data = read_pickle(file_path)
    for channel in channels:
        print("Working on channel %s" % channel)
        plot_raster_for_monkeys_by_rank(raw_data, channel, date, round_no, save=True)


def plot_rasters_for_specific_round_and_channel(date, round_no, channels):
    metadata_reader = RecordingMetadataReader()
    experiment_data_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_no)
    script_dir = Path(__file__).parent
    file_path = (script_dir / '..' / '..' / '..' / 'Cortana' / 'compiled' / experiment_data_filename).resolve()
    raw_data = read_pickle(file_path)
    for channel in channels:
        print("Working on channel %s" % channel)
        plot_raster_for_monkeys_by_rank(raw_data, channel, date, round_no, save=True)



def plot_rasters_with_exploded_spike_cache_data(cache_dir='/home/connorlab/Documents/GitHub/Julie/Cortana/exploded_spike_cache/', save=False, min_trials=7, min_spikes=400):
    cache_dir = Path(cache_dir)

    for file in sorted(cache_dir.glob("*.pkl")):
        # Parse date and round from filename
        try:
            stem = file.stem  # e.g., "2023-09-29_round_3"
            date, round_part = stem.split('_round_')
            round_no = int(round_part)
        except Exception as e:
            print(f"Could not parse date/round from filename '{file.name}': {e}")
            continue

        # Load DataFrame
        with open(file, 'rb') as f:
            df = pickle.load(f)

        for neuron_id in df['NeuronID'].dropna().unique():
            neuron_df = df[df['NeuronID'] == neuron_id]

            # filtering out neurons with too few trials or too few spikes
            num_trials = len(neuron_df)
            total_spikes = sum(len(spikes) for spikes in neuron_df['SpikeTimes'])
            if num_trials < min_trials:
                print(f"⏭️ Skipping Neuron {neuron_id}: only {num_trials} trials (min required: {min_trials})")
                continue
            if total_spikes < min_spikes:
                print(f"⏭️ Skipping Neuron {neuron_id}: only {total_spikes} spikes (min required: {min_spikes})")
                continue

            print(f"✅ Plotting Neuron {neuron_id} — {num_trials} trials, {total_spikes} spikes")

            unique_monkey_groups = neuron_df['MonkeyGroup'].dropna().unique().tolist()
            max_rows = max([
                len(neuron_df[neuron_df['MonkeyGroup'] == group]['MonkeyName'].dropna().unique())
                for group in unique_monkey_groups
            ])

            fig = plt.figure(figsize=(15 * len(unique_monkey_groups), 45 * max_rows))

            for col_idx, group_name in enumerate(unique_monkey_groups):
                group_data = neuron_df[neuron_df['MonkeyGroup'] == group_name]
                unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
                monkeys_list = [
                    m for m in get_monkeys_by_rank(group_name) if m in unique_monkeys
                ]

                for row_idx, monkey_name in enumerate(monkeys_list):
                    monkey_data = group_data[group_data['MonkeyName'] == monkey_name]
                    subplot_idx = row_idx * len(unique_monkey_groups) + col_idx + 1
                    ax = fig.add_subplot(max_rows, len(unique_monkey_groups), subplot_idx)

                    filtered_spike_times_list = []
                    for _, row in monkey_data.iterrows():
                        spikes = row['SpikeTimes']
                        start, stop = row['EpochStartStop']
                        aligned_spikes = [s - start for s in spikes if start <= s <= stop]
                        filtered_spike_times_list.append(aligned_spikes)

                    ax.eventplot(filtered_spike_times_list, color='black', linewidths=0.5)
                    ax.set_xlim(0, 2.2)
                    ax.text(1.05, 0.5, monkey_name, transform=ax.transAxes,
                            ha='left', va='center', fontsize=14)

                fig.text(0.6 / len(unique_monkey_groups) + col_idx / len(unique_monkey_groups), 0.92,
                         group_name, ha='center', va='center')

            fig.text(0.5, 0.05, 'Time (s)', ha='center')
            fig.text(0.99, 0.95, f'N: {len(neuron_df)}', ha='right')
            fig.suptitle(f'Raster Plot (by Rank): {neuron_id}', fontsize=16)
            plt.subplots_adjust(hspace=1.0, wspace=1.0)
            plt.show()

            if save:
                save_raster_plots_by_neuron(fig, neuron_id)


def plot_rasters_with_sorted_spike_cache_data(cache_dir='/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache/', save=False, min_trials=7):
    cache_dir = Path(cache_dir)

    for file in sorted(cache_dir.glob("*.pkl")):
        # Parse date and round from filename
        try:
            stem = file.stem  # e.g., "2023-09-29_round_3"
            date, round_part = stem.split('_round_')
            round_no = int(round_part)
        except Exception as e:
            print(f"Could not parse date/round from filename '{file.name}': {e}")
            continue

        # Load DataFrame
        with open(file, 'rb') as f:
            df = pickle.load(f)

        for neuron_id in df['NeuronID'].dropna().unique():
            neuron_df = df[df['NeuronID'] == neuron_id]

            # filtering out neurons with too few trials or too few spikes
            num_trials = len(neuron_df)
            total_spikes = sum(len(spikes) for spikes in neuron_df['SpikeTimes'])
            if num_trials < min_trials:
                print(f"⏭️ Skipping Neuron {neuron_id}: only {num_trials} trials (min required: {min_trials})")
                continue
            print(f"✅ Plotting Neuron {neuron_id} — {num_trials} trials, {total_spikes} spikes")

            unique_monkey_groups = neuron_df['MonkeyGroup'].dropna().unique().tolist()
            max_rows = max([
                len(neuron_df[neuron_df['MonkeyGroup'] == group]['MonkeyName'].dropna().unique())
                for group in unique_monkey_groups
            ])

            fig = plt.figure(figsize=(15 * len(unique_monkey_groups), 45 * max_rows))

            for col_idx, group_name in enumerate(unique_monkey_groups):
                group_data = neuron_df[neuron_df['MonkeyGroup'] == group_name]
                unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
                monkeys_list = [
                    m for m in get_monkeys_by_rank(group_name) if m in unique_monkeys
                ]

                for row_idx, monkey_name in enumerate(monkeys_list):
                    monkey_data = group_data[group_data['MonkeyName'] == monkey_name]
                    subplot_idx = row_idx * len(unique_monkey_groups) + col_idx + 1
                    ax = fig.add_subplot(max_rows, len(unique_monkey_groups), subplot_idx)

                    filtered_spike_times_list = []
                    for _, row in monkey_data.iterrows():
                        spikes = row['SpikeTimes']
                        start, stop = row['EpochStartStop']
                        aligned_spikes = [s - start for s in spikes if start <= s <= stop]
                        filtered_spike_times_list.append(aligned_spikes)

                    ax.eventplot(filtered_spike_times_list, color='black', linewidths=0.5)
                    ax.set_xlim(0, 2.2)
                    ax.text(1.05, 0.5, monkey_name, transform=ax.transAxes,
                            ha='left', va='center', fontsize=14)

                fig.text(0.6 / len(unique_monkey_groups) + col_idx / len(unique_monkey_groups), 0.92,
                         group_name, ha='center', va='center')

            fig.text(0.5, 0.05, 'Time (s)', ha='center')
            fig.text(0.99, 0.95, f'N: {len(neuron_df)}', ha='right')
            fig.suptitle(f'Raster Plot (by Rank): {neuron_id}', fontsize=16)
            plt.subplots_adjust(hspace=1.0, wspace=1.0)
            plt.show()

            if save:
                save_raster_plots_by_neuron(fig, neuron_id)


def _ensure_base_channel(df):
    if 'BaseChannel' in df.columns:
        return df
    if 'Channel' in df.columns:
        df = df.copy()
        df['BaseChannel'] = df['Channel'].astype(str).apply(lambda s: s.split('_Unit')[0] if '_Unit' in s else s)
        return df
    raise ValueError("Need 'BaseChannel' or 'Channel' to group units by channel.")


def _build_trial_table(df_group_monkey_unit, all_task_ids):
    # align spikes to epoch start, ensure a list for each trial in all_task_ids
    trial_map = {}
    for _, row in df_group_monkey_unit.iterrows():
        spikes = row['SpikeTimes']
        start, stop = row['EpochStartStop']
        aligned = [s - start for s in spikes if start <= s <= stop]
        trial_map[row['TaskField']] = aligned
    return [trial_map.get(tid, []) for tid in all_task_ids]


def plot_sorted_units_by_base_channel(
    cache_dir,
    channel_filter=None,          # e.g., "Channel.C_007" or ["Channel.C_007", "Channel.C_012"]
    min_trials=7,
    save_dir=None,                # e.g., "/path/to/rasters" or None
    max_time=2.2,
    monkeys_by_rank_fn=None       # optional: your get_monkeys_by_rank; else alphabetical
):
    """
    For one .pkl file: makes one figure per BaseChannel.
    Each subplot = (group, monkey). Within each subplot, all UNITS (same BaseChannel) are overlaid in different colors.
    Trials are aligned by TaskField so rows match across units.
    """
    cache_dir = Path(cache_dir)

    for file in sorted(cache_dir.glob("*.pkl")):
        # Parse date and round from filename
        try:
            stem = file.stem  # e.g., "2023-09-29_round_3"
            date, round_part = stem.split('_round_')
            round_no = int(round_part)
        except Exception as e:
            print(f"Could not parse date/round from filename '{file.name}': {e}")
            continue

        # Load DataFrame
        with open(file, 'rb') as f:
            df = pickle.load(f)

        df = _ensure_base_channel(df)

        needed = {'NeuronID', 'SpikeTimes', 'EpochStartStop', 'BaseChannel'}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        # which channels?
        all_bases = df['BaseChannel'].dropna().unique().tolist()
        if channel_filter is None:
            base_channels = all_bases
        else:
            base_channels = [channel_filter] if isinstance(channel_filter, str) else list(channel_filter)

        # units per base
        units_by_base = defaultdict(list)
        for base in base_channels:
            units = (
                df[df['BaseChannel'] == base]
                .dropna(subset=['Channel'])
                .sort_values('Channel')['Channel']
                .unique()
                .tolist()
            )
            units_by_base[base] = units

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        saved = []
        for base in base_channels:
            units = units_by_base.get(base, [])
            if not units:
                continue

            # keep units with enough trials
            good_units = []
            for u in units:
                n_trials = len(df[(df['BaseChannel'] == base) & (df['Channel'] == u)])
                if n_trials >= min_trials:
                    good_units.append(u)
            if len(good_units) < 2:
                print(f"{base} does NOT have more than 2 units to plot ")
                continue

            sub = df[(df['BaseChannel'] == base) & (df['Channel'].isin(good_units))]
            has_meta = all(c in sub.columns for c in ['MonkeyGroup', 'MonkeyName', 'TaskField'])

            if has_meta:
                groups = sorted(sub['MonkeyGroup'].dropna().unique().tolist())
                monkeys_by_group = {}
                max_rows = 0
                for g in groups:
                    mm = sub[sub['MonkeyGroup'] == g]['MonkeyName'].dropna().unique().tolist()
                    if monkeys_by_rank_fn is not None:
                        # keep dataset monkeys in your ranking order
                        ranked = [m for m in monkeys_by_rank_fn(g) if m in mm]
                        mm = ranked
                    else:
                        mm = sorted(mm)
                    monkeys_by_group[g] = mm
                    max_rows = max(max_rows, len(mm))
            else:
                groups, monkeys_by_group, max_rows = [], {}, 0

            if groups and max_rows > 0:
                fig = plt.figure(figsize=(15 * max(1, len(groups)), 45 * max(1, max_rows)))
                color_cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5']))
                unit_colors = {u: next(color_cyc) for u in good_units}

                for cidx, g in enumerate(groups):
                    for ridx, monkey in enumerate(monkeys_by_group[g]):
                        ax = fig.add_subplot(max_rows, len(groups), ridx * len(groups) + cidx + 1)

                        # master trial order for this monkey
                        all_tids = sorted(sub[(sub['MonkeyGroup'] == g) & (sub['MonkeyName'] == monkey)]['TaskField'].dropna().unique().tolist())
                        if not all_tids:
                            ax.set_axis_off()
                            continue

                        offsets = np.arange(len(all_tids))
                        for u in good_units:
                            df_um = sub[(sub['Channel'] == u) & (sub['MonkeyGroup'] == g) & (sub['MonkeyName'] == monkey)]
                            positions = _build_trial_table(df_um, all_tids)
                            if positions:
                                ax.eventplot(positions, lineoffsets=offsets, colors=unit_colors[u], linewidths=0.6)

                        ax.set_xlim(0, max_time)
                        ax.set_ylim(-0.5, len(all_tids) - 0.5)
                        ax.set_yticks([])
                        ax.text(1.02, 0.5, monkey, transform=ax.transAxes, ha='left', va='center', fontsize=12)
                        if ridx == 0:
                            ax.set_title(g, fontsize=14)

                fig.suptitle(f"Raster overlay by units — {file.name} — {base}", fontsize=16)
                fig.text(0.5, 0.05, "Time (s)", ha='center')


            else:
                # fallback: no group/monkey metadata; single axes overlay
                fig, ax = plt.subplots(figsize=(14, 8))
                color_cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5']))
                unit_colors = {u: next(color_cyc) for u in good_units}
                all_tids = sorted(sub['TaskField'].dropna().unique().tolist())
                offsets = np.arange(len(all_tids))
                for u in good_units:
                    df_u = sub[sub['Channel'] == u]
                    positions = _build_trial_table(df_u, all_tids)
                    ax.eventplot(positions, lineoffsets=offsets, colors=unit_colors[u], linewidths=0.6)
                ax.set_xlim(0, max_time)
                ax.set_ylim(-0.5, len(all_tids) - 0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{file.name} — {base} (overlay by unit)")

            # legend
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=c, lw=2, label=str(u)) for u,c in unit_colors.items()]
            if handles:
                fig.legend(handles=handles, loc='upper right', title="Units (Channel column)")

            if save_dir:
                out = Path(save_dir) / f"{file.stem}__{base.replace('/','_')}.png"
                fig.savefig(out, dpi=200, bbox_inches="tight")
                print("Saved:", out)

            plt.show()

def _robust_base_mask_exploded(df_exploded, base_str):
    """
    True for rows belonging to the given base channel (string), tolerant to enum/string mismatches.
    Matches if:
      - BaseChannel == base_str
      - Channel == base_str
      - Channel.split('_Unit')[0] == base_str
    """
    mask = np.zeros(len(df_exploded), dtype=bool)

    if 'BaseChannel' in df_exploded.columns:
        mask |= (df_exploded['BaseChannel'].astype(str) == base_str)

    if 'Channel' in df_exploded.columns:
        ch = df_exploded['Channel'].astype(str)
        mask |= (ch == base_str) | (ch.str.split('_Unit').str[0] == base_str)

    return mask


def plot_dual_rasters_by_base_channel(
    sorted_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache",
    exploded_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/exploded_spike_cache",
    max_time=2.2,
    monkey_groups=("Zombies", "Best Frans")  # exactly 2 groups to display
):
    """
    For each .pkl in `sorted_cache_dir`, find the matching exploded file (same filename).
    For each base channel that has sorted units (derived from df_sorted['Channel']),
    make ONE figure showing the two requested groups.
    For each group and monkey (one row per monkey):
      - Left subplot: combined raster from EXPLODED (all spikes in black)
      - Right subplot: overlay raster from SORTED (each unit a different color)
    Monkey rows are ordered strictly by get_monkeys_by_rank(group) (extras, if any, appended alphabetically).
    """
    sorted_cache_dir = Path(sorted_cache_dir)
    exploded_cache_dir = Path(exploded_cache_dir)

    # normalize to exactly two groups
    if len(monkey_groups) > 2:
        monkey_groups = tuple(monkey_groups[:2])

    for file in sorted(sorted_cache_dir.glob("*.pkl")):
        # load sorted + exploded
        try:
            with open(file, "rb") as f:
                df_sorted = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (sorted) due to error: {e}")
            continue

        exp_path = exploded_cache_dir / file.name
        if not exp_path.exists():
            print(f"❌ No matching exploded file for {file.name} at {exp_path}")
            continue
        try:
            with open(exp_path, "rb") as f:
                df_exploded = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (exploded) due to error: {e}")
            continue

        print(f'For this experimental round: {file.name}' )
        exploded_base_channels = df_exploded['BaseChannel'].unique()
        sorted_channels = df_sorted['BaseChannel'].unique()
        print(f'we have {exploded_base_channels} as base channels in exploded')
        print(f'we have {sorted_channels} as sorted channels')

        # ---- STEP 2: bases from df_sorted['Channel'] (only sorted units) ----
        if 'Channel' not in df_sorted.columns:
            print(f"❌ {file.stem}: no 'Channel' in sorted; skipping file.")
            continue

        df_sorted['_ChannelStr'] = df_sorted['Channel'].astype(str)
        df_sorted['_BaseChannelStr'] = df_sorted['BaseChannel'].astype(str)
        base_list = sorted(df_sorted['_BaseChannelStr'].unique().tolist())

        # ---- STEP 3: loop bases ----
        for base in base_list:
            base_str = str(base)  # normalize (enums safe)

            # STEP 4: combined (exploded) — per group → per monkey row
            units = sorted(df_sorted.loc[df_sorted['BaseChannel'] == base_str, '_ChannelStr'].unique().tolist())

            # build ranked monkey lists from EXPLODED using robust base mask
            monkeys_by_group = {}
            max_rows = 0
            base_mask_uns = _robust_base_mask_exploded(df_exploded, base_str)
            for g in monkey_groups:
                df_uns_bg = df_exploded[base_mask_uns & (df_exploded['MonkeyGroup'] == g)]
                present = df_uns_bg['MonkeyName'].dropna().astype(str).unique().tolist()
                ranked = [m for m in get_monkeys_by_rank(g) if m in present]
                extras = sorted([m for m in present if m not in ranked])
                row_list = ranked + extras
                monkeys_by_group[g] = row_list
                max_rows = max(max_rows, len(row_list))

            if max_rows == 0:
                print(f"{file.stem} — {base_str}: no monkeys for {monkey_groups}; skipping base.")
                continue

            # figure grid: rows = monkeys, cols = groups*2 (left combined, right overlay)
            n_groups = len(monkey_groups)
            n_cols = n_groups * 2
            n_rows = max_rows
            fig = plt.figure(figsize=(14 * n_groups, 3.0 + 2.4 * n_rows))

            # color map for units (overlay)
            color_cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7']))
            unit_colors = {u: next(color_cyc) for u in units}

            for gi, g in enumerate(monkey_groups):
                rows_for_group = monkeys_by_group[g]
                for ridx in range(n_rows):
                    idx_uns = ridx * n_cols + (gi * 2) + 1
                    idx_srt = ridx * n_cols + (gi * 2) + 2
                    # ... inside your big for-loop over ridx (right after you create ax_uns / ax_srt)

                    # --- left: combined (exploded) ---
                    ax_uns = fig.add_subplot(n_rows, n_cols, idx_uns)
                    if ridx == 0:
                        ax_uns.set_title(f"{g} — combined", fontsize=12)

                    if ridx < len(rows_for_group):
                        monkey = rows_for_group[ridx]
                        df_gm_uns = df_exploded[base_mask_uns &
                                                (df_exploded['MonkeyGroup'] == g) &
                                                (df_exploded['MonkeyName'] == monkey)]
                        all_tids_uns = sorted(df_gm_uns['TaskField'].dropna().unique().tolist())
                        if all_tids_uns:
                            positions_uns = _build_trial_table(df_gm_uns, all_tids_uns)
                            ax_uns.eventplot(positions_uns, linewidths=0.5, colors="black")  # black by default
                            ax_uns.set_ylim(-0.5, len(all_tids_uns) - 0.5)
                        ax_uns.text(1.01, 0.5, monkey, transform=ax_uns.transAxes,
                                    ha='left', va='center', fontsize=10)
                    else:
                        ax_uns.set_axis_off()
                        continue

                    ax_uns.set_xlim(0, max_time)
                    # hide x tick labels except on bottom row
                    if ridx != n_rows - 1:
                        ax_uns.tick_params(axis='x', which='both', labelbottom=False)
                    else:
                        # keep ticks; no per-axes xlabel (we’ll add one shared label)
                        ax_uns.set_xlabel("")

                    # --- right: overlay (sorted) ---
                    ax_srt = fig.add_subplot(n_rows, n_cols, idx_srt)
                    if ridx == 0:
                        ax_srt.set_title(f"{g} — overlay", fontsize=12)

                    df_gm_srt = df_sorted[(df_sorted['_BaseChannelStr'] == base_str) &
                                                (df_sorted['MonkeyGroup'] == g) &
                                                (df_sorted['MonkeyName'] == monkey)]
                    all_tids = sorted(df_gm_srt['TaskField'].dropna().unique().tolist())
                    if all_tids:
                        for u in units:
                            df_um = df_gm_srt[df_gm_srt['_ChannelStr'] == u]
                            if not df_um.empty:
                                positions = _build_trial_table(df_um, all_tids)
                                ax_srt.eventplot(positions, colors=unit_colors[u], linewidths=0.6)
                        ax_srt.set_ylim(-0.2, len(all_tids) - 0.5)

                    ax_srt.set_xlim(0, max_time)
                    if ridx != n_rows - 1:
                        ax_srt.tick_params(axis='x', which='both', labelbottom=False)
                    else:
                        ax_srt.set_xlabel("")

            # legend
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=c, lw=2, label=str(u)) for u,c in unit_colors.items()]
            if handles:
                fig.legend(handles=handles, loc='upper right', title="Units")

            fig.suptitle(f"Dual Raster — {file.name} — {base_str}", fontsize=14)
            plt.subplots_adjust(
                left=0.06,  # a bit of space on left
                right=0.88,  # more space on right so monkey labels fit
                top=0.92,
                bottom=0.08,
                hspace=0.4,  # vertical space between rows
                wspace=0.35  # horizontal space between columns
            )
            fig.supxlabel("Time (s)", fontsize=11)

            plt.show()


def plot_dual_rasters_one_group_by_base_channel(
    sorted_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache",
    exploded_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/exploded_spike_cache",
    max_time=2.2,
    monkey_group="Zombies"   # <- plot ONE group per figure
):
    """
    For each .pkl pair (sorted+exploded with same filename):
      - Get base channels from df_sorted (string view of BaseChannel).
      - For each base channel, build ONE figure for the given monkey_group.
        Left col: combined raster from EXPLODED (black).
        Right col: overlay raster from SORTED (one color per unit).
      - Rows = monkeys in `monkey_group`, ordered by get_monkeys_by_rank(group).
    Relies on: _ensure_base_channel, _build_trial_table, get_monkeys_by_rank, _robust_base_mask_exploded.
    """
    sorted_cache_dir = Path(sorted_cache_dir)
    exploded_cache_dir = Path(exploded_cache_dir)

    for file in sorted(sorted_cache_dir.glob("*.pkl")):
        # load sorted + exploded
        try:
            with open(file, "rb") as f:
                df_sorted = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (sorted) due to error: {e}")
            continue

        exp_path = exploded_cache_dir / file.name
        if not exp_path.exists():
            print(f"❌ No matching exploded file for {file.name} at {exp_path}")
            continue
        try:
            with open(exp_path, "rb") as f:
                df_exploded = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (exploded) due to error: {e}")
            continue

        # string views for safe masking
        if 'Channel' in df_sorted.columns:
            df_sorted['_ChannelStr'] = df_sorted['Channel'].astype(str)
        df_sorted['_BaseChannelStr'] = df_sorted['BaseChannel'].astype(str)

        # base channels to iterate (from sorted; guarantees sorted cells exist for base)
        base_list = sorted(df_sorted['_BaseChannelStr'].dropna().unique().tolist())

        for base_str in base_list:
            # all sorted units (channels) that belong to this base
            units = sorted(
                df_sorted.loc[df_sorted['_BaseChannelStr'] == base_str, '_ChannelStr']
                .dropna().astype(str).unique().tolist()
            )
            if not units:
                continue  # nothing to overlay

            # build ranked monkey list for THIS group from exploded (robust base mask)
            base_mask_uns = _robust_base_mask_exploded(df_exploded, base_str)
            df_uns_bg = df_exploded[base_mask_uns & (df_exploded['MonkeyGroup'].astype(str) == monkey_group)]
            present = df_uns_bg['MonkeyName'].dropna().astype(str).unique().tolist()

            ranked = [m for m in get_monkeys_by_rank(monkey_group) if m in present]
            extras = sorted([m for m in present if m not in ranked])
            monkeys = ranked + extras
            n_rows = len(monkeys)
            if n_rows == 0:
                print(f"{file.stem} — {base_str}: no monkeys for group '{monkey_group}'; skipping.")
                continue

            # figure: 2 columns (combined | overlay), rows = monkeys
            n_cols = 2
            # tighter, shorter subplots to avoid overlap
            fig = plt.figure(figsize=(10, max(1.0, 1.1 * n_rows + 1.2)))

            # color cycle for units (overlay)
            color_cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7']))
            unit_colors = {u: next(color_cyc) for u in units}

            for ridx, monkey in enumerate(monkeys):
                # indices for the pair of subplots for this monkey
                idx_uns = ridx * n_cols + 1
                idx_srt = ridx * n_cols + 2

                # === LEFT: combined (EXPLODED) ===
                ax_uns = fig.add_subplot(n_rows, n_cols, idx_uns)
                if ridx == 0:
                    ax_uns.set_title(f"Raw Unsorted (simply thresholded)", fontsize=12)

                df_gm_uns = df_exploded[base_mask_uns &
                                        (df_exploded['MonkeyGroup'].astype(str) == monkey_group) &
                                        (df_exploded['MonkeyName'].astype(str) == monkey)]
                all_tids_uns = sorted(df_gm_uns['TaskField'].dropna().unique().tolist())
                if all_tids_uns:
                    positions_uns = _build_trial_table(df_gm_uns, all_tids_uns)
                    ax_uns.eventplot(positions_uns, linewidths=0.5, colors="black")
                    ax_uns.set_ylim(-0.5, len(all_tids_uns) - 0.5)

                ax_uns.set_xlim(0, max_time)
                # only bottom row shows xticks
                if ridx != n_rows - 1:
                    ax_uns.tick_params(axis='x', which='both', labelbottom=False)
                else:
                    ax_uns.set_xlabel("")
                ax_uns.set_yticks([])
                # put monkey name just outside right edge of left subplot
                ax_uns.text(1.005, 0.5, monkey, transform=ax_uns.transAxes,
                            ha='left', va='center', fontsize=10)

                # === RIGHT: overlay (SORTED) ===
                ax_srt = fig.add_subplot(n_rows, n_cols, idx_srt)
                if ridx == 0:
                    ax_srt.set_title(f"Sorted (using spikeinterface)", fontsize=12)

                df_gm_srt = df_sorted[(df_sorted['_BaseChannelStr'] == base_str) &
                                      (df_sorted['MonkeyGroup'].astype(str) == monkey_group) &
                                      (df_sorted['MonkeyName'].astype(str) == monkey)]
                all_tids = sorted(df_gm_srt['TaskField'].dropna().unique().tolist())
                if all_tids:
                    for u in units:
                        df_um = df_gm_srt[df_gm_srt['_ChannelStr'] == u]
                        if df_um.empty:
                            continue
                        positions = _build_trial_table(df_um, all_tids)
                        ax_srt.eventplot(positions, colors=unit_colors[u], linewidths=0.6)
                    ax_srt.set_ylim(-0.5, len(all_tids) - 0.5)

                ax_srt.set_xlim(0, max_time)
                if ridx != n_rows - 1:
                    ax_srt.tick_params(axis='x', which='both', labelbottom=False)
                else:
                    ax_srt.set_xlabel("")
                ax_srt.set_yticks([])

            # legend once (top-right)
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=c, lw=2, label=str(u)) for u,c in unit_colors.items()]
            if handles:
                fig.legend(handles=handles, loc='upper right', title="Units")

            # layout & labels: tighter subplots and a single shared x label
            plt.subplots_adjust(
                left=0.06, right=0.9, top=0.9, bottom=0.1,
                hspace=0.35, wspace=0.3
            )
            fig.supxlabel("Time (s)", fontsize=11)
            fig.suptitle(f"Dual Raster {monkey_group} {file.name} {base_str}", fontsize=13)

            plt.show()


import pickle
from itertools import cycle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.widgets as sw

def _plot_unit_templates_panel(ax, we, unit_ids, peak_sign='neg'):
    """
    Draw a compact overlay of mean templates (best channel per unit) into `ax`.
    """
    # templates: (n_units, n_channels, n_samples)
    templates = we.get_all_templates()
    chan_ids = np.asarray(we.channel_ids)
    t_ms = np.arange(templates.shape[2]) * (1.0 / we.sampling_frequency) * 1000.0

    # map unit_ids -> index in templates
    # we.sorting.unit_ids can be non-0..N; build index
    unit_index = {u: i for i, u in enumerate(we.sorting.unit_ids)}

    # best channel index per unit
    if peak_sign in ('neg', 'negative'):
        best_idx = np.argmin(templates.min(axis=2), axis=1)
    elif peak_sign in ('pos', 'positive'):
        best_idx = np.argmax(templates.max(axis=2), axis=1)
    else:
        # both: use absolute extremum
        worst = np.argmax(np.abs(np.stack([templates.max(axis=2), templates.min(axis=2)], axis=0)), axis=0)
        best_idx = np.take_along_axis(
            np.argmax(np.abs(templates), axis=2), worst, axis=1
        )  # safe fallback

    # color cycle
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5']))
    for u in unit_ids:
        if u not in unit_index:  # skip if not in WE (mapping mismatch)
            continue
        ui = unit_index[u]
        ch_i = best_idx[ui]
        y = templates[ui, ch_i, :]
        ax.plot(t_ms, y, lw=1.6, alpha=0.95, label=str(u), color=next(colors))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("µV")
    ax.set_title("Waveforms (best ch)")
    # compact legend
    leg = ax.legend(fontsize=7, loc='upper right', frameon=False, ncols=1)
    if leg:
        for lh in leg.legend_handles:
            lh.set_linewidth(2.0)

def plot_dual_rasters_with_waveforms_one_group_by_base_channel(
    sorted_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache",
    exploded_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/exploded_spike_cache",
    max_time=2.2,
    monkey_group="Zombies",
    we=None,                       # <-- optional: SpikeInterface WaveformExtractor
    peak_sign="neg"                # <-- waveform polarity for "best channel"
):
    sorted_cache_dir = Path(sorted_cache_dir)
    exploded_cache_dir = Path(exploded_cache_dir)

    for file in sorted(sorted_cache_dir.glob("*.pkl")):
        try:
            with open(file, "rb") as f:
                df_sorted = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (sorted) due to error: {e}")
            continue

        exp_path = exploded_cache_dir / file.name
        if not exp_path.exists():
            print(f"❌ No matching exploded file for {file.name} at {exp_path}")
            continue
        try:
            with open(exp_path, "rb") as f:
                df_exploded = _ensure_base_channel(pickle.load(f))
        except Exception as e:
            print(f"❌ Skipping {file.name} (exploded) due to error: {e}")
            continue

        if 'Channel' in df_sorted.columns:
            df_sorted['_ChannelStr'] = df_sorted['Channel'].astype(str)
        df_sorted['_BaseChannelStr'] = df_sorted['BaseChannel'].astype(str)

        base_list = sorted(df_sorted['_BaseChannelStr'].dropna().unique().tolist())

        for base_str in base_list:
            units = sorted(
                df_sorted.loc[df_sorted['_BaseChannelStr'] == base_str, '_ChannelStr']
                .dropna().astype(str).unique().tolist()
            )
            if not units:
                continue

            base_mask_uns = _robust_base_mask_exploded(df_exploded, base_str)
            df_uns_bg = df_exploded[base_mask_uns & (df_exploded['MonkeyGroup'].astype(str) == monkey_group)]
            present = df_uns_bg['MonkeyName'].dropna().astype(str).unique().tolist()

            ranked = [m for m in get_monkeys_by_rank(monkey_group) if m in present]
            extras = sorted([m for m in present if m not in ranked])
            monkeys = ranked + extras
            n_rows = len(monkeys)
            if n_rows == 0:
                print(f"{file.stem} — {base_str}: no monkeys for group '{monkey_group}'; skipping.")
                continue

            # ----- layout: 3 columns if we is given, else 2 -----
            have_waveforms = we is not None
            n_cols = 3 if have_waveforms else 2

            # figure size: a bit wider to host the waveform column
            fig_w = 13 if have_waveforms else 10
            fig = plt.figure(figsize=(fig_w, max(1.0, 1.1 * n_rows + 1.2)))

            color_cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7']))
            unit_colors = {u: next(color_cyc) for u in units}

            for ridx, monkey in enumerate(monkeys):
                # column indices per row
                idx_uns = ridx * n_cols + 1
                idx_srt = ridx * n_cols + 2
                idx_wav = ridx * n_cols + 3 if have_waveforms else None

                # === LEFT: combined (EXPLODED) ===
                ax_uns = fig.add_subplot(n_rows, n_cols, idx_uns)
                if ridx == 0:
                    ax_uns.set_title("Raw Unsorted", fontsize=12)
                df_gm_uns = df_exploded[base_mask_uns &
                                        (df_exploded['MonkeyGroup'].astype(str) == monkey_group) &
                                        (df_exploded['MonkeyName'].astype(str) == monkey)]
                all_tids_uns = sorted(df_gm_uns['TaskField'].dropna().unique().tolist())
                if all_tids_uns:
                    positions_uns = _build_trial_table(df_gm_uns, all_tids_uns)
                    ax_uns.eventplot(positions_uns, linewidths=0.5, colors="black")
                    ax_uns.set_ylim(-0.5, len(all_tids_uns) - 0.5)
                ax_uns.set_xlim(0, max_time)
                if ridx != n_rows - 1:
                    ax_uns.tick_params(axis='x', which='both', labelbottom=False)
                ax_uns.set_yticks([])
                ax_uns.text(1.005, 0.5, monkey, transform=ax_uns.transAxes,
                            ha='left', va='center', fontsize=10)

                # === MIDDLE: overlay (SORTED) ===
                ax_srt = fig.add_subplot(n_rows, n_cols, idx_srt)
                if ridx == 0:
                    ax_srt.set_title("Sorted (overlay)", fontsize=12)
                df_gm_srt = df_sorted[(df_sorted['_BaseChannelStr'] == base_str) &
                                      (df_sorted['MonkeyGroup'].astype(str) == monkey_group) &
                                      (df_sorted['MonkeyName'].astype(str) == monkey)]
                all_tids = sorted(df_gm_srt['TaskField'].dropna().unique().tolist())
                if all_tids:
                    for u in units:
                        df_um = df_gm_srt[df_gm_srt['_ChannelStr'] == u]
                        if df_um.empty:
                            continue
                        positions = _build_trial_table(df_um, all_tids)
                        ax_srt.eventplot(positions, colors=unit_colors[u], linewidths=0.6)
                    ax_srt.set_ylim(-0.5, len(all_tids) - 0.5)
                ax_srt.set_xlim(0, max_time)
                if ridx != n_rows - 1:
                    ax_srt.tick_params(axis='x', which='both', labelbottom=False)
                ax_srt.set_yticks([])

                # === RIGHT: waveforms (best ch per unit) ===
                if have_waveforms:
                    ax_wav = fig.add_subplot(n_rows, n_cols, idx_wav)
                    # Show each unit’s best-channel template, all overlaid
                    # Note: we use WE unit_ids; they may be ints. We overlay whatever ids exist.
                    # If you want the exact subset "units", just pass unit IDs that match WE.
                    _plot_unit_templates_panel(ax_wav, we, list(we.sorting.unit_ids), peak_sign=peak_sign)
                    if ridx != n_rows - 1:
                        ax_wav.tick_params(axis='x', which='both', labelbottom=False)

            # one legend (units) for the raster overlay
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=c, lw=2, label=str(u)) for u,c in unit_colors.items()]
            if handles:
                fig.legend(handles=handles, loc='upper right', title="Units (raster)")

            plt.subplots_adjust(left=0.05, right=0.92, top=0.9, bottom=0.1, hspace=0.35, wspace=0.35)
            fig.supxlabel("Time (s)", fontsize=11)
            fig.suptitle(f"{monkey_group} — {file.name} — Base {base_str}", fontsize=13)
            plt.show()


def process_noise():
    noises = []
    julie = []
    kelsey = []
    joseph = []
    for noise in noises:
        kelsey_noise = noise.get_kelsey()
        allen_noise = noise.get_allen()
        if kelsey_noise.topics == 'Chai':
            joseph.get_boba(with_whom=None)
        elif kelsey_noise.topics == 'Run':
            julie.run_away()
            joseph.run(with_whom='Kelsey')
            allen.run(with_whom='Kelsey')
        elif kelsey_noise.topics == 'Boba':
            joseph = julie.get_joseph()
            joseph.get_boba(with_whom=kelsey)
        elif kelsey_noise.topics == 'Cry':
            allen = julie.get_allen()
            allen.have_cry_session(with_whom=[kelsey, julie])
            joseph.get_boba(with_whom=None)
        elif kelsey_noise.topics == 'No Data':
            joseph.get_boba(with_whom=None)
            noises.append('Cry')
        else:
            joseph.get_boba(with_whom=None)

if __name__ == '__main__':
    # cells_to_be_plotted = pd.read_excel(
    #     "/home/connorlab/Documents/GitHub/Julie/src/analyses/response_window_finder/CUSUM_window_cells_ANOVA_passed.xlsx")
    # subset_df = cells_to_be_plotted[['Date', 'Round No.', 'Cell']]
    # all_cells = subset_df.drop_duplicates(subset=['Date', 'Round No.', 'Cell'])
    #
    # reversed_df = all_cells.groupby(['Date', 'Round No.']).agg(list)
    # for index, row in reversed_df.iterrows():
    #     row['Cell'] = [convert_to_enum(item) for item in row['Cell']]
    #     date = index[0]
    #     round_no = index[1]
    #     plot_rasters_for_specific_round_and_channel(date, round_no, row['Cell'])
    #     # plot_rasters_for_specific_round_and_channel("2023-10-04", 3, [Channel.C_002])
    # generate_rasters_from_exploded_spike_cache()
    # plot_rasters_with_exploded_spike_cache_data(save=True)
    # plot_rasters_with_sorted_spike_cache_data()
    # plot_sorted_units_by_base_channel("/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache")
    plot_dual_rasters_one_group_by_base_channel()
    # sorting_TDC = ss.read_sorter_folder('tridesclous_output')
    # we = si.extract_waveforms(
    #     recording_preprocessed, sorting_TDC,
    #     folder="waveforms_cache2",
    #     ms_before=1.5, ms_after=2.0,
    #     max_spikes_per_unit=50, return_scaled=True, load_if_exists=True
    # )
    # plot_dual_rasters_with_waveforms_one_group_by_base_channel(we=we)
