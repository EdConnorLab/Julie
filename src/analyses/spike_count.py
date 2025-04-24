import pandas as pd

from analyses.cache_utils import ExplodedSpikeCacheManager
from tqdm import tqdm
from collections import defaultdict
"""
Data Preparation Module for Spike Data Analysis
-----------------------------------------------

This module contains functions for exploding, binning, 
and aggregating spike data for downstream analyses (e.g., permutation ANOVA).

"""

def prepare_exploded_spike_data(date, round_no, only_valid_channels=False):
    """Prepare exploded spike data (pre-binning)."""
    cache = ExplodedSpikeCacheManager()
    return cache.load_or_compute(date, round_no, only_valid_channels=only_valid_channels)


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
    Extract spike counts for each neuron and time window from cached raw spike times.
    """
    spike_count_rows = []
    cache = {}  # {(date, round_no): exploded_df}

    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Extracting spike counts"):
        neuron_id = row['NeuronID']
        start_ms, end_ms = row['WindowStart_ms'], row['WindowEnd_ms']
        start_sec, end_sec = start_ms / 1000, end_ms / 1000

        # Extract date and round from NeuronID (e.g., "2023-09-26_3_Channel.C_003_Unit 1")
        parts = neuron_id.split('_', 3)
        date_str, round_no = parts[0], int(parts[1])
        cache_key = (date_str, round_no)

        # Use cache manager
        if cache_key not in cache:
            exploded_df = prepare_exploded_spike_data(date_str, round_no)
            cache[cache_key] = exploded_df
        else:
            exploded_df = cache[cache_key]

        # Match spike rows
        neuron_trials = exploded_df[exploded_df['NeuronID'] == neuron_id]
        for _, trial_row in neuron_trials.iterrows():
            spike_times = trial_row['SpikeTimes']
            epoch_start, _ = trial_row['EpochStartStop']
            window_start = epoch_start + start_sec
            window_end = epoch_start + end_sec
            count = sum(window_start <= t < window_end for t in spike_times)

            spike_count_rows.append({
                'NeuronID': neuron_id,
                'MonkeyName': trial_row['MonkeyName'],
                'MonkeyGroup': trial_row['MonkeyGroup'],
                'TaskField': trial_row['TaskField'],
                'WindowStart_ms': start_ms,
                'WindowEnd_ms': end_ms,
                'SpikeCount': count
            })

    return pd.DataFrame(spike_count_rows)

