import pandas as pd

from analyses.channel_enum_resolvers import is_channel_in_dict, get_value_from_dict_with_channel, convert_to_enum
from analyses.intan_data_processor.single_channel_analysis import get_spike_count
from analyses.spike_count import load_and_combine_data


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


def get_spike_count_for_single_neuron_with_time_window(neuron_specific_time_windows):
    """
    Spike count for a channel with time window (handles both sorted and unsorted channels)

    Parameters:
        neuron_specific_time_windows (pandas.DataFrame) contains the following columns:
            - 'Date': need to convert to YYYY-MM-DD format
            - 'Round No.': int (i.e. 2)
            - 'Cell': string (i.e. Channel.C_013 or Channel.C_010_Unit 1)
            - 'Time Window': in ms (i.e. (250, 750))

    Returns:
    all_spike_count (pandas.DataFrame)

    """
    neuron_specific_time_windows[['Date', 'Round No.']] = neuron_specific_time_windows['NeuronID'].apply(
        lambda x: pd.Series(x.split('_', 2)[:2])
    )
    neuron_specific_time_windows['Round No.'] = neuron_specific_time_windows['Round No.'].astype(int)
    rows_with_unique_rounds = neuron_specific_time_windows.drop_duplicates(subset=['Date', 'Round No.'])
    experimental_rounds = rows_with_unique_rounds[['Date', 'Round No.']]

    results = []
    for _, row in experimental_rounds.iterrows():
        combined_data = load_and_combine_data(row['Date'], row['Round No.'])
        cells = neuron_specific_time_windows[
            ((neuron_specific_time_windows['Date'] == row['Date']) & (neuron_specific_time_windows['Round No.'] == row['Round No.']))]
        for _, cell in cells.iterrows():
            if isinstance(cell['Time Window'], str):
                time_window = tuple(float(num) for num in cell['Time Window'].strip('()').split(','))
            else:
                time_window = cell['Time Window']
                cell['Cell'] = convert_to_enum(cell['Cell'])
                unsorted_cells_spike_count = count_spikes_for_specific_cell_time_windowed(raw_trial_data, cell['Cell'],
                                                                                          time_window)
                unsorted_cells_spike_count_dict = unsorted_cells_spike_count.to_dict(orient='records')[0]
                unsorted_cells_spike_count_dict['Cell'] = cell['Cell']
                unsorted_cells_spike_count_dict['Date'] = row['Date']
                unsorted_cells_spike_count_dict['Round No.'] = row['Round No.']
                unsorted_cells_spike_count_dict['Time Window'] = time_window
                results.append(unsorted_cells_spike_count_dict)

    all_spike_count = pd.DataFrame(results)
    all_spike_count.set_index('Cell', inplace=True)

    return all_spike_count
