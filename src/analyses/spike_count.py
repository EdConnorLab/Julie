import pandas as pd

from analyses.cache_utils import ExplodedSpikeCacheManager, SortedSpikeCacheManager
from tqdm import tqdm
from collections import defaultdict

from analyses.spike_source import SpikeSource, SISortedSpikeSource, MixedManualSpikeSource

"""
Data Preparation Module for Spike Data Analysis
-----------------------------------------------

This module contains functions for exploding, binning, 
and aggregating spike data for downstream analyses (e.g., permutation ANOVA).

"""



def filter_good_neurons(exploded_df,
                        min_total_spikes=500,
                        min_firing_rate_hz=1.0,
                        min_trial_ratio=0.7,
                        min_trial_count=300):
    """
    Filters neurons based on total spikes, firing rate, spike coverage across trials, and trial participation count.

    Parameters:
    - exploded_df: DataFrame with columns ['NeuronID', 'SpikeTimes', 'EpochStartStop']
    - min_total_spikes: minimum total spike count
    - min_firing_rate_hz: minimum average firing rate in Hz
    - min_trial_ratio: minimum ratio of trials where spikes are present
    - min_trial_count: minimum number of trials the neuron participated in

    Returns:
    - List of NeuronIDs that passed all filters
    """
    if exploded_df is None or getattr(exploded_df, "empty", True):
        return []

    if "NeuronID" not in exploded_df.columns:
        return []

    neuron_stats = []

    for neuron_id, group in exploded_df.groupby("NeuronID"):
        spike_counts = group["SpikeTimes"].apply(len)
        total_spikes = spike_counts.sum()
        trial_durations = group["EpochStartStop"].apply(lambda x: x[1] - x[0])
        total_time = trial_durations.sum()
        mean_firing_rate = total_spikes / total_time if total_time > 0 else 0
        spike_trials = (spike_counts > 0).sum()
        total_trials = len(group)

        neuron_stats.append({
            "NeuronID": neuron_id,
            "TotalSpikes": total_spikes,
            "MeanFiringRateHz": mean_firing_rate,
            "SpikeTrialRatio": spike_trials / total_trials,
            "TotalTrials": total_trials
        })

    stats_df = pd.DataFrame(neuron_stats)

    # Apply filters
    good_neurons = stats_df[
        (stats_df["TotalSpikes"] >= min_total_spikes) &
        (stats_df["MeanFiringRateHz"] >= min_firing_rate_hz) &
        (stats_df["SpikeTrialRatio"] >= min_trial_ratio) &
        (stats_df["TotalTrials"] >= min_trial_count)
        ]["NeuronID"].tolist()

    return good_neurons



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

def prepare_binned_spike_data(
    date,
    round_no,
    bin_size,
    *,
    source: SpikeSource | None = None,
) -> pd.DataFrame:
    source = source or MixedManualSpikeSource()  # sensible default

    """Prepare binned spike data with spike counts."""
    exploded_df = source.load(date, round_no)
    if exploded_df is None or getattr(exploded_df, "empty", True):
        return pd.DataFrame()

    good_neurons = filter_good_neurons(exploded_df)
    if not good_neurons:
        return pd.DataFrame()

    filtered_df = exploded_df[exploded_df["NeuronID"].isin(good_neurons)]
    if filtered_df.empty:
        return pd.DataFrame()

    return bin_spike_times(filtered_df, bin_size)


# --- Aggregate ---
def aggregate_trial_level(df):
    """Collapse spike counts across time bins per trial (trial-level total spike count)."""
    return df.groupby(['NeuronID', 'TaskField', 'MonkeyName', 'MonkeyGroup'], as_index=False)['SpikeCount'].sum()


def aggregate_timebin_level(df):
    """Aggregate spike counts at bin level across trials (time-resolved spike count per neuron and monkey group)."""
    return df.groupby(['NeuronID', 'MonkeyGroup', 'TimeBinIndex'], as_index=False)['SpikeCount'].sum()


def extract_spike_counts_from_windows(window_df: pd.DataFrame, source: SpikeSource) -> pd.DataFrame:
    """
    Extract spike counts for each neuron and time window from cached raw spike times.
    """
    spike_count_rows = []
    cache = {}  # {(date, round_no): exploded_df}

    for _, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Extracting spike counts"):
        neuron_id = row['NeuronID']
        start_ms, end_ms = row['WindowStart_ms'], row['WindowEnd_ms']
        start_sec, end_sec = start_ms / 1000, end_ms / 1000

        # Extract date and round from NeuronID (e.g., "AMG_2023-09-26_3_Channel.C_003_Unit 1")
        date_str = row["Date"]
        round_no = int(row["Round No."])
        cache_key = (date_str, round_no)

        # Use cache manager
        if cache_key not in cache:
            exploded_df = source.load(date_str, round_no)
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

## TODO: needs to be tested
def extract_spike_counts_from_cells(neurons_df, use_sorted=False):
    """
    Given a list of neurons with NeuronID,
    extract spike counts per trial from cached exploded spike data.

    Parameters
    ----------
    neurons_df : pd.DataFrame
        Must contain column: 'NeuronID'

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'TaskField', 'SpikeCount']
    """
    all_rows = []
    cache = {}

    neurons_df_sorted = neurons_df.sort_values(by=['NeuronID'])

    for _, row in tqdm(neurons_df_sorted.iterrows(), total=len(neurons_df_sorted), desc="Extracting spike counts"):
        neuron_id = row['NeuronID']
        parts = neuron_id.split('_', 4)
        date_str = parts[1]
        round_no = int(parts[2])
        cache_key = (date_str, round_no)

        # if cache_key not in cache:
            # cache[cache_key] = prepare_exploded_spike_data(date_str, round_no, use_sorted=use_sorted)
        exploded_df = cache[cache_key]

        matching_trials = exploded_df[exploded_df['NeuronID'] == neuron_id]
        matching_trials = matching_trials.copy()
        matching_trials['SpikeCount'] = matching_trials['SpikeTimes'].apply(len)

        all_rows.append(matching_trials[
            ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'TaskField', 'SpikeCount']
        ])

    return pd.concat(all_rows, ignore_index=True)