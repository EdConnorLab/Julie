import pandas as pd
from analyses.enums.monkey_names import Zombies
from analyses.intan_data_processor.single_channel_analysis import read_pickle, get_spike_count

from analyses.channel_enum_resolvers import is_channel_in_dict, get_value_from_dict_with_channel, convert_to_enum
from analyses.data_loader import load_raw_data, combine_unsorted_with_sorted
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.intan_data_processor.single_unit_analysis import read_sorted_data

"""
Data Preparation Module for Spike Data Analysis
-----------------------------------------------

This module contains functions for loading, exploding, binning, 
and aggregating spike data for downstream analyses (e.g., permutation ANOVA).

Function structure:

1. Data Loading and Combination
   - load_and_combine_data(): Load raw unsorted and sorted spike data, and combine.

2. Data Explosion (Wide → Long Format)
   - explode_spike_data(): Explode spike times into long-format DataFrame with metadata.

3. Spike Time Binning
   - bin_spike_times(): Bin spike times into spike counts per time bin.

4. Pipeline Helpers
   - prepare_exploded_spike_data(): Prepare exploded spike data (pre-binning).
   - prepare_binned_spike_data(): Prepare binned spike data with spike counts.

5. Data Aggregation
   - aggregate_trial_level(): Aggregate spike counts across time bins per trial.
   - aggregate_bin_level(): Aggregate spike counts at bin level across trials.

Usage notes:
- Use `prepare_exploded_spike_data()` when you need spike times for flexible analyses.
- Use `prepare_binned_spike_data()` when you need binned spike counts for statistical tests.
- Aggregation functions are optional helpers for trial-level or time-resolved analysis.

"""


def load_and_combine_data(date, round_no):
    """Load and combine raw unsorted and sorted spike data."""
    raw_unsorted_data, _, sorted_data = load_raw_data(date, round_no)
    combined_data = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    return combined_data


# --- Data explosion (wide → long format) ---
def explode_spike_data(combined_data, date, round_no):
    """Explode spike times into long-format DataFrame with metadata."""
    rows = []
    for _, row in combined_data.iterrows():
        for channel_enum, spike_list in row['SpikeTimes'].items():
            rows.append({
                'TaskField': row['TaskField'],
                'MonkeyId': row['MonkeyId'],
                'MonkeyGroup': row['MonkeyGroup'],
                'MonkeyName': row['MonkeyName'],
                'Channel': channel_enum,
                'SpikeTimes': spike_list,
                'EpochStartStop': row['EpochStartStop']
            })
    exploded_df = pd.DataFrame(rows)

    sample_channel = str(exploded_df['Channel'].iloc[0])
    has_unit = "_Unit" in sample_channel

    def normalize_channel(ch):
        ch_str = str(ch) if not hasattr(ch, 'value') else str(ch)
        return ch_str.split("_Unit")[0] if has_unit else ch_str

    exploded_df['BaseChannel'] = exploded_df['Channel'].apply(normalize_channel)
    exploded_df['Date'] = date
    exploded_df['Round No.'] = round_no
    exploded_df['NeuronID'] = (
            exploded_df['Date'].astype(str) + "_" +
            exploded_df['Round No.'].astype(str) + "_" +
            exploded_df['Channel'].astype(str)
    )

    return exploded_df


def prepare_exploded_spike_data(date, round_no):
    """Prepare exploded spike data (pre-binning)."""
    combined_data = load_and_combine_data(date, round_no)
    exploded_df = explode_spike_data(combined_data, date, round_no)
    return exploded_df


def bin_spike_times(exploded_df, bin_size):
    """Bin spike times into spike counts per time bin."""
    binned_rows = []

    for _, row in exploded_df.iterrows():
        start_time, end_time = row['EpochStartStop']
        spike_times = row['SpikeTimes']
        time_bins = [start_time + i * bin_size for i in range(int((end_time - start_time) / bin_size) + 1)]

        for bin_index in range(len(time_bins) - 1):
            count = sum(time_bins[bin_index] <= t < time_bins[bin_index + 1] for t in spike_times)
            binned_rows.append({
                'MonkeyGroup': row['MonkeyGroup'],
                'MonkeyName': row['MonkeyName'],
                'TaskField': row['TaskField'],
                'Channel': row['Channel'],
                'BaseChannel': row['BaseChannel'],
                'EpochStartStop': row['EpochStartStop'],
                'TimeBinIndex': bin_index,
                'SpikeCount': count,
                'Date': row['Date'],
                'Round No.': row['Round No.'],
                'NeuronID': row['NeuronID']
            })

    return pd.DataFrame(binned_rows)


def prepare_binned_spike_data(date, round_no, bin_size):
    """Prepare binned spike data with spike counts."""
    exploded_df = prepare_exploded_spike_data(date, round_no)
    binned_df = bin_spike_times(exploded_df, bin_size)
    return binned_df


# --- Aggregate ---
def aggregate_trial_level(df):
    """Collapse spike counts across time bins per trial (trial-level total spike count)."""
    return df.groupby(['NeuronID', 'TaskField', 'MonkeyName', 'MonkeyGroup'], as_index=False)['SpikeCount'].sum()


def aggregate_timebin_level(df):
    """Aggregate spike counts at bin level across trials (time-resolved spike count per neuron and monkey group)."""
    return df.groupby(['NeuronID', 'MonkeyGroup', 'TimeBinIndex'], as_index=False)['SpikeCount'].sum()


# --- everything below needs refactoring

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
            if 'Unit' not in cell['Cell']:  # unsorted cells
                cell['Cell'] = convert_to_enum(cell['Cell'])
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

