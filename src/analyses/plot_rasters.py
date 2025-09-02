import os
import pickle
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.enums.monkey_names import get_monkeys_by_rank
from analyses.intan_data_processor.single_channel_analysis import read_pickle, plot_raster_for_monkeys_by_rank, \
    save_raster_plots, save_raster_plots_by_neuron

from channel_enum_resolvers import convert_to_enum


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
            # plt.show()

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
    plot_rasters_with_sorted_spike_cache_data()