import pandas as pd

from analyses.data_loader import load_raw_data, combine_unsorted_with_sorted
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader

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
def explode_spike_data(combined_data, date, round_no, only_valid_channels=False):
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
    if only_valid_channels:
        reader = RecordingMetadataReader()
        valid_channels = reader.get_valid_channels(date, round_no)
        valid_channels_list = [str(ch) for ch in valid_channels]
        exploded_df = exploded_df[exploded_df['BaseChannel'].isin(valid_channels_list)]

    return exploded_df


def prepare_exploded_spike_data(date, round_no, only_valid_channels=False):
    """Prepare exploded spike data (pre-binning)."""
    combined_data = load_and_combine_data(date, round_no)
    exploded_df = explode_spike_data(combined_data, date, round_no, only_valid_channels)
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


def prepare_binned_spike_data(date, round_no, bin_size, only_valid_channels=False):
    """Prepare binned spike data with spike counts."""
    exploded_df = prepare_exploded_spike_data(date, round_no, only_valid_channels)
    binned_df = bin_spike_times(exploded_df, bin_size)
    return binned_df


# --- Aggregate ---
def aggregate_trial_level(df):
    """Collapse spike counts across time bins per trial (trial-level total spike count)."""
    return df.groupby(['NeuronID', 'TaskField', 'MonkeyName', 'MonkeyGroup'], as_index=False)['SpikeCount'].sum()


def aggregate_timebin_level(df):
    """Aggregate spike counts at bin level across trials (time-resolved spike count per neuron and monkey group)."""
    return df.groupby(['NeuronID', 'MonkeyGroup', 'TimeBinIndex'], as_index=False)['SpikeCount'].sum()


def extract_spike_counts_from_windows(window_df):
    """
    Extract spike counts for each neuron and time window directly from raw spike times.

    Parameters:
    - window_df: DataFrame with ['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']

    Returns:
    - DataFrame with columns:
        ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'TaskField',
         'WindowStart_ms', 'WindowEnd_ms', 'SpikeCount']
    """
    from tqdm import tqdm
    from collections import defaultdict

    spike_count_rows = []
    cache = {}  # (date, round_no) → exploded_df

    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Extracting spike counts"):
        neuron_id = row['NeuronID']
        start_ms = row['WindowStart_ms']
        end_ms = row['WindowEnd_ms']
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        # Parse date and round_no from NeuronID
        parts = neuron_id.split('_', 3)
        date_str, round_no = str(parts[0]), int(parts[1])
        cache_key = (date_str, round_no)
        # Load + cache exploded data
        if cache_key not in cache:
            exploded_df = prepare_exploded_spike_data(date_str, round_no)
            cache[cache_key] = exploded_df
        else:
            exploded_df = cache[cache_key]

        neuron_df = exploded_df[exploded_df['NeuronID'] == neuron_id]
        for _, trial_row in neuron_df.iterrows():
            spike_times = trial_row['SpikeTimes']
            epoch_start, _ = trial_row['EpochStartStop']
            window_start_abs = epoch_start + start_sec
            window_end_abs = epoch_start + end_sec
            count = sum(window_start_abs <= t < window_end_abs for t in spike_times)

            spike_count_rows.append({
                'NeuronID': neuron_id,
                'MonkeyName': trial_row['MonkeyName'],
                'MonkeyGroup': trial_row['MonkeyGroup'],
                'TaskField': trial_row['TaskField'],
                'WindowStart_ms': row['WindowStart_ms'],
                'WindowEnd_ms': row['WindowEnd_ms'],
                'SpikeCount': count
            })

    return pd.DataFrame(spike_count_rows)



