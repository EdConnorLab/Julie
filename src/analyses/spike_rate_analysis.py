import os
from pathlib import Path
import pandas as pd
import networkx as nx

from clat.intan.channels import Channel
from clat.intan.rhd import load_intan_rhd_format

from analyses.single_channel_analysis import read_pickle, calculate_spike_rate
from analyses.single_unit_analysis import calculate_spike_timestamps
from recording_metadata_reader import RecordingMetadataReader


def get_spike_rates_for_each_trial(date, round_number):
    """
    Computes spike rates for each trial for a given experimental round

    """
    metadata_reader = RecordingMetadataReader()
    pickle_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_number) + ".pk1"
    compiled_dir = (Path(__file__).parent.parent.parent / 'compiled').resolve()
    pickle_filepath = os.path.join(compiled_dir, pickle_filename)
    print(f'Reading pickle file as raw data: {pickle_filepath}')
    raw_trial_data = read_pickle(pickle_filepath)
    intan_dir = metadata_reader.get_intan_folder_name_for_specific_round(date, round_number)
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = Path(os.path.join(cortana_path, date, intan_dir))
    valid_channels = set(metadata_reader.get_valid_channels(date, round_number))

    raw_data_spike_rates = compute_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        print(f"Reading sorted data: {round_path}")
        sorted_data = read_sorted_data(round_path)
        channels_with_units = (sorted_data['SpikeTimes'][0].keys())
        sorted_channels = [int(s.split('_')[1][1:]) for s in channels_with_units]
        sorted_enum_channels = list(set([Channel(f'C-{channel:03}') for channel in sorted_channels]))
        # remove channel only if it exists as index
        for channel in sorted_enum_channels:
            if channel in raw_data_spike_rates.index:
                raw_data_spike_rates = raw_data_spike_rates.drop(channel)
        sorted_data_spike_rates = compute_spike_rates_from_sorted_data(sorted_data)
        spike_rates = pd.concat([sorted_data_spike_rates, raw_data_spike_rates])
    else:
        spike_rates = raw_data_spike_rates

    # print('-------------------- spike rates for each trial --------------------')
    # print(spike_rates)
    return spike_rates

def get_average_spike_rates_for_each_monkey(date, round_number):
    """
    Computes average spike rates for each monkey photo -- spike rates are averaged over 10 trials

    Parameters:
        date (str): the date of the experimental round (e.g. '2023-10-03')
        round_number (int): the experimental round number

    Returns:
        average spike rates across 10 trials for each monkey
    """
    metadata_reader = RecordingMetadataReader()
    pickle_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_number) + ".pk1"
    compiled_dir = (Path(__file__).parent.parent.parent / 'compiled').resolve()
    pickle_filepath = os.path.join(compiled_dir, pickle_filename)
    print(f'Reading pickle file as raw data: {pickle_filepath}')
    raw_trial_data = read_pickle(pickle_filepath)

    intan_dir = metadata_reader.get_intan_folder_name_for_specific_round(date, round_number)
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = Path(os.path.join(cortana_path, date, intan_dir))
    valid_channels = set(metadata_reader.get_valid_channels(date, round_number))

    average_spike_rates_from_raw_data = compute_average_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels)

    # Check if the experimental round is sorted
    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        print(f"Reading sorted_file data: {round_path}")
        sorted_data = read_sorted_data(round_path)
        channels_with_units = (sorted_data['SpikeTimes'][0].keys())
        sorted_channels = [int(s.split('_')[1][1:]) for s in channels_with_units]
        sorted_enum_channels = list(set([Channel(f'C-{channel:03}') for channel in sorted_channels]))
        # remove channel only if it exists as index
        for channel in sorted_enum_channels:
            if channel in average_spike_rates_from_raw_data.index:
                average_spike_rates_from_raw_data = average_spike_rates_from_raw_data.drop(channel)
        average_spike_rates_from_sorted_data = compute_average_spike_rates_from_sorted_data(sorted_data)
        average_spike_rates = pd.concat([average_spike_rates_from_sorted_data, average_spike_rates_from_raw_data])
    else:
        average_spike_rates = average_spike_rates_from_raw_data

    # print('-------------------- average spike rates for each monkey ---------------------')
    # print(average_spike_rates)

    return average_spike_rates


def read_sorted_data(round_path, sorted_spikes_filename='sorted_spikes.pkl', compiled_trials_filename='compiled.pk1'):
    compiled_trials_filepath = os.path.join(round_path, compiled_trials_filename)
    raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)
    rhd_file_path = os.path.join(round_path, 'info.rhd')
    sorted_spikes_filepath = os.path.join(round_path, sorted_spikes_filename)
    sorted_spikes = read_pickle(sorted_spikes_filepath)
    sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
    sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)
    return sorted_data


def compute_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    spike_rate_per_channel = pd.DataFrame()
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_spike_rates = {}
        for channel in valid_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")
            monkey_spike_rates[channel] = spike_rates
        spike_rate_per_channel[monkey] = pd.Series(monkey_spike_rates)
    return spike_rate_per_channel


def compute_average_spike_rates_from_raw_trial_data(raw_trial_data, valid_channels):
    # average spike rates for each monkey
    unique_monkeys = raw_trial_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    for monkey in unique_monkeys:
        monkey_data = raw_trial_data[raw_trial_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        for channel in valid_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                    print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit


def compute_spike_rates_from_sorted_data(sorted_data):
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())
    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}
        for channel in unique_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")
            monkey_specific_spike_rate[channel] = spike_rates
        spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)
    return spike_rate_by_unit


def compute_average_spike_rates_from_sorted_data(sorted_data):
    # average spike rates for each monkey
    unique_monkeys = sorted_data['MonkeyName'].dropna().unique().tolist()
    avg_spike_rate_by_unit = pd.DataFrame(index=[])
    unique_channels = set()
    unique_channels.update(sorted_data['SpikeTimes'][0].keys())

    for monkey in unique_monkeys:
        monkey_data = sorted_data[sorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_rate = {}

        for channel in unique_channels:
            spike_rates = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_rates.append(calculate_spike_rate(data, row['EpochStartStop']))
                else:
                   print(f"No data for {channel} in row {index}")

            avg_spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0
            monkey_specific_spike_rate[channel] = avg_spike_rate

        # Add monkey-specific spike rates to the DataFrame
        avg_spike_rate_by_unit[monkey] = pd.Series(monkey_specific_spike_rate)

    print(avg_spike_rate_by_unit)
    return avg_spike_rate_by_unit


def get_value_from_dict_with_channel(channel, dictionary):
    if isinstance(channel, str):
        return dictionary[channel]
    else:
        for key, value in dictionary.items():
            if key.value == channel.value:
                return value


def is_channel_in_dict(channel, diction):
    if isinstance(channel, str):
        if channel in list(diction.keys()):
            return True
    else:
        for key in diction:
            if channel.value == key.value:
                return True


def set_node_attributes_with_default(graph, values_dict, attribute_name, default_value=0):
    for node in graph.nodes():
        value = values_dict.get(node, default_value)
        nx.set_node_attributes(graph, {node: value}, attribute_name)


def compute_population_spike_rates_for_ER():
    # ER Population Average Spike Rate
    ER_population = pd.DataFrame()
    reader = RecordingMetadataReader()
    ER = reader.get_metadata_for_brain_region('ER')

    for index, row in ER.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round = row['Round No.']
        avg_spike_rates_for_specific_round = get_average_spike_rates_for_each_monkey(date, round)
        ER_population = pd.concat([ER_population, avg_spike_rates_for_specific_round])
    population_spike_rate = ER_population.mean()
    print(population_spike_rate)
    return population_spike_rate

def compute_population_spike_rates_for_AMG():
    AMG_population = pd.DataFrame()
    reader = RecordingMetadataReader()
    AMG = reader.get_metadata_for_brain_region('AMG')

    for index, row in AMG.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round = row['Round No.']
        avg_spike_rates_for_specific_round = get_average_spike_rates_for_each_monkey(date, round)
        AMG_population = pd.concat([AMG_population, avg_spike_rates_for_specific_round])
    population_spike_rate = AMG_population.mean()
    print(population_spike_rate)
    return population_spike_rate


def compute_overall_average_spike_rates_for_each_round(date, round_number):
    overall_average_spike_rates = get_average_spike_rates_for_each_monkey(date, round_number).mean()
    return overall_average_spike_rates



if __name__ == '__main__':

    df = get_spike_rates_for_each_trial("2023-10-04", 4)
    #dat = compute_overall_average_spike_rates_for_each_round("2023-09-29", 2)
    # avg_spike_rates = compute_average_spike_rates_of_each_unit_for_specific_round("2023-09-29", 2)
    # ones with errors
    # avg_spike_rates = compute_average_spike_rates("2023-09-29", 1)
    # avg_spike_rates = compute_average_spike_rates("2023-11-08", 1)
