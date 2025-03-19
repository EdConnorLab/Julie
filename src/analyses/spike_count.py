import pandas as pd

import channel_enum_resolvers
import spike_rate_computation
from channel_enum_resolvers import drop_duplicate_channels, is_channel_in_dict, get_value_from_dict_with_channel
from single_channel_analysis import read_pickle, get_spike_count
from data_readers.recording_metadata_reader import RecordingMetadataReader
from spike_rate_computation import read_sorted_data

"""
Structure of spike_count.py

get_spike_count_for_each_trial
    count_spikes_from_sorted_data
    count_spikes_from_raw_unsorted_data
    
get_spike_counts_for_given_time_window

count_spikes_for_specific_cell_time_windowed

add_metadata_to_spike_counts

"""

def get_spike_count_for_each_trial(date, round_number):
    """
    Return: number of spikes for each trial for a given experimental round
    """
    reader = RecordingMetadataReader()
    pickle_filepath, valid_channels, round_path = reader.get_metadata_for_spike_analysis(date, round_number)

    raw_trial_data = read_pickle(pickle_filepath)
    raw_data_spike_counts = count_spikes_from_raw_unsorted_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():

        sorted_data = read_sorted_data(round_path)
        sorted_data_spike_counts = count_spikes_from_sorted_data(sorted_data)

        raw_data_spike_counts = drop_duplicate_channels(raw_data_spike_counts, sorted_data)
        spike_counts = pd.concat([sorted_data_spike_counts, raw_data_spike_counts])
    else:
        spike_counts = raw_data_spike_counts

    return spike_counts


def count_spikes_from_sorted_data(sorted_data):
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    spike_count_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())
    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_counts = {}
        for channel in unique_channels:
            spike_counts = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_counts.append(get_spike_count(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")
            monkey_specific_spike_counts[channel] = spike_counts
        spike_count_by_unit[monkey] = pd.Series(monkey_specific_spike_counts)
    return spike_count_by_unit


def count_spikes_from_raw_unsorted_data(raw_unsorted_data, valid_channels):
    unique_monkeys = raw_unsorted_data['MonkeyName'].dropna().unique().tolist()
    spike_count_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_unsorted_data[raw_unsorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_counts = {}
        for channel in valid_channels:
            spike_counts = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_counts.append(get_spike_count(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")
            monkey_specific_spike_counts[channel] = spike_counts
        spike_count_per_channel[monkey] = pd.Series(monkey_specific_spike_counts)
    return spike_count_per_channel

def get_spike_counts_for_time_chunks(monkeys, raw_data, channels, chunk_size):
    """
    chunk size in seconds
    """
    monkey_spike_counts = pd.DataFrame()
    for monkey in monkeys:
        monkey_data = raw_data[raw_data['MonkeyName'] == monkey]
        spike_counts_by_channel = {}
        for index, row in monkey_data.iterrows():
            for channel in channels:
                spike_count_for_each_channel = []
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    start_time, end_time = row['EpochStartStop']
                    time_chunks = [start_time + i * chunk_size for i in
                                   range(int((end_time - start_time) / chunk_size) + 1)]
                    for i in range(len(time_chunks) - 1):
                        time_range = (time_chunks[i], time_chunks[i + 1])
                        spike_count = get_spike_count(data, time_range)
                        spike_count_for_each_channel.append(spike_count)
                else:
                    print(f"No data for {channel} in row {index}")
                spike_counts_by_channel[channel] = spike_count_for_each_channel
            monkey_spike_counts[monkey] = pd.Series(spike_counts_by_channel)
    return monkey_spike_counts

def get_spike_counts_for_given_time_window(monkeys, raw_data, channels, time_window):
    monkey_spike_counts = pd.DataFrame()
    for monkey in monkeys:
        monkey_data = raw_data[raw_data['MonkeyName'] == monkey]
        spike_counts_by_channel = {}
        for channel in channels:
            spike_count_for_each_channel = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    start_time, _ = row['EpochStartStop']
                    window_start_micro, window_end_micro = time_window
                    window_start_sec = window_start_micro * 0.001
                    window_end_sec = window_end_micro * 0.001
                    spike_count_for_each_channel.append(get_spike_count(data, (start_time + window_start_sec,
                                                               start_time + window_end_sec)))
                else:
                    print(f"No data for {channel} in row {index}")
            spike_counts_by_channel[channel] = spike_count_for_each_channel
        monkey_spike_counts[monkey] = pd.Series(spike_counts_by_channel)
    return monkey_spike_counts


def count_spikes_for_specific_cell_time_windowed(raw_data, cell, time_window):
    unique_monkeys = raw_data['MonkeyName'].dropna().unique().tolist()
    spike_count_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_data[raw_data['MonkeyName'] == monkey]
        monkey_spike_counts = {}
        spike_counts = []
        for index, row in monkey_data.iterrows():
            if is_channel_in_dict(cell, row['SpikeTimes']):
                data = get_value_from_dict_with_channel(cell, row['SpikeTimes'])
                if time_window is not None:
                    window_start_milli, window_end_milli = time_window
                    window_start_sec = window_start_milli * 0.001
                    window_end_sec = window_end_milli * 0.001
                    start_time, _ = row['EpochStartStop']
                    spike_counts.append(
                        get_spike_count(data, (start_time + window_start_sec, start_time + window_end_sec)))
                else:
                    spike_counts.append(get_spike_count(data, row['EpochStartStop']))
            else:
                print(f"No data for {cell} in row {index}")
        monkey_spike_counts[cell] = spike_counts
        spike_count_per_channel[monkey] = pd.Series(monkey_spike_counts)
    return spike_count_per_channel


def get_spike_count_for_single_neuron_with_time_window(cell_metadata):
    """
    Spike count for a channel with time window (handles both sorted and unsorted channels)

    Parameters:
        cell_metadata (pandas.DataFrame) contains the following columns:
            - 'Date': need to convert to YYYY-MM-DD format
            - 'Round No.': int (i.e. 2)
            - 'Cell': string (i.e. Channel.C_013 or Channel.C_010_Unit 1)
            - 'Time Window': in ms (i.e. (250, 750))

    Returns:
    all_spike_count (pandas.DataFrame)

    """
    reader = RecordingMetadataReader()
    rows_with_unique_rounds = cell_metadata.drop_duplicates(subset=['Date', 'Round No.'])
    experimental_rounds = rows_with_unique_rounds[['Date', 'Round No.']]

    results = []
    for _, row in experimental_rounds.iterrows():
        pickle_filepath, _, round_dir_path = reader.get_metadata_for_spike_analysis(row['Date'], row['Round No.'])
        raw_trial_data = read_pickle(pickle_filepath)
        sorted_file = round_dir_path / 'sorted_spikes.pkl'

        if sorted_file.exists():
            sorted_data = read_sorted_data(round_dir_path)

        cells = cell_metadata[((cell_metadata['Date'] == row['Date']) & (cell_metadata['Round No.'] == row['Round No.']))]
        for _, cell in cells.iterrows():
            if isinstance(cell['Time Window'], str):
                time_window = tuple(float(num) for num in cell['Time Window'].strip('()').split(','))
            else:
                time_window = cell['Time Window']
            if 'Unit' not in cell['Cell']:  # unsorted cells
                cell['Cell'] = channel_enum_resolvers.convert_to_enum(cell['Cell'])
                unsorted_cells_spike_count = count_spikes_for_specific_cell_time_windowed(raw_trial_data, cell['Cell'],
                                                                                          time_window)
                unsorted_cells_spike_count_dict = unsorted_cells_spike_count.to_dict(orient='records')[0]
                unsorted_cells_spike_count_dict['Cell'] = cell['Cell']
                unsorted_cells_spike_count_dict['Date'] = row['Date']
                unsorted_cells_spike_count_dict['Round No.'] = row['Round No.']
                unsorted_cells_spike_count_dict['Time Window'] = time_window
                results.append(unsorted_cells_spike_count_dict)

            else:  # sorted cells
                sorted_cells_spike_count = count_spikes_for_specific_cell_time_windowed(
                    sorted_data, cell['Cell'], time_window)
                sorted_cells_spike_count_dict = sorted_cells_spike_count.to_dict(orient='records')[0]
                sorted_cells_spike_count_dict['Cell'] = cell['Cell']
                sorted_cells_spike_count_dict['Date'] = row['Date']
                sorted_cells_spike_count_dict['Round No.'] = row['Round No.']
                sorted_cells_spike_count_dict['Time Window'] = time_window
                results.append(sorted_cells_spike_count_dict)

    all_spike_count = pd.DataFrame(results)
    all_spike_count.set_index('Cell', inplace=True)

    return all_spike_count


def add_metadata_to_spike_counts(spike_count_df, date, round_number, time_window):
    """
    Adds metadata to spike counts DataFrame.
    """
    spike_count_df['Date'] = date
    spike_count_df['Round No.'] = round_number
    spike_count_df['Time Window'] = [time_window]  * len(spike_count_df)
    return spike_count_df
