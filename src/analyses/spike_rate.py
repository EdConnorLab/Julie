import pandas as pd
from tqdm import tqdm

from analyses.spike_count import prepare_exploded_spike_data


def add_trial_spike_rate_columns(exploded_df):
    """
    Adds SpikeCount, EpochDuration, and SpikeRate columns to exploded spike data.
    """
    df = exploded_df.copy()
    df['SpikeCount'] = df['SpikeTimes'].apply(len)
    df['EpochDuration'] = df['EpochStartStop'].apply(lambda x: x[1] - x[0])
    df['SpikeRate'] = df['SpikeCount'] / df['EpochDuration']
    return df

def get_mean_spike_rate_per_neuron_monkey(trial_level_spike_rate_df):
    # Take mean spike rate per NeuronID × MonkeyName
    mean_rate_df = (
        trial_level_spike_rate_df
        .groupby(['NeuronID', 'MonkeyName', 'MonkeyGroup'])['SpikeRate']
        .mean()
        .reset_index()
        .rename(columns={'SpikeRate': 'MeanSpikeRate'})
    )

    return mean_rate_df

def compute_mean_spike_rate_table(exploded_df):
    trial_level_spike_rate_df= add_trial_spike_rate_columns(exploded_df)
    df = get_mean_spike_rate_per_neuron_monkey(trial_level_spike_rate_df)
    return df

def compute_mean_spike_rate_for_cells(neurons_df):
    """
    Compute mean spike rate for a list of significant neurons across trials.

    Parameters
    ----------
    neurons_df : pd.DataFrame
        Must contain 'NeuronID' column.

    Returns
    -------
    pd.DataFrame
        Columns: ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'MeanSpikeRate']
    """
    all_rows = []
    cache = {}

    for _, row in tqdm(neurons_df.iterrows(), total=len(neurons_df), desc="Computing mean spike rate"):
        neuron_id = row['NeuronID']

        # Extract date and round_no
        parts = neuron_id.split('_', 4)
        date_str = parts[1]
        round_no = int(parts[2])
        cache_key = (date_str, round_no)

        if cache_key not in cache:
            cache[cache_key] = prepare_exploded_spike_data(date_str, round_no)
        exploded_df = cache[cache_key]

        matching_trials = exploded_df[exploded_df['NeuronID'] == neuron_id]
        if matching_trials.empty:
            continue
        matching_trials = add_trial_spike_rate_columns(matching_trials)
        all_rows.append(matching_trials[['NeuronID', 'MonkeyName', 'MonkeyGroup', 'SpikeRate']])

    # Concatenate all trial-level spike rates
    trial_level_spike_rate_df = pd.concat(all_rows, ignore_index=True)
    # Now average by NeuronID × MonkeyName
    mean_rate_df = get_mean_spike_rate_per_neuron_monkey(trial_level_spike_rate_df)
    return mean_rate_df

if __name__ == "__main__":
    neurons_df = pd.read_pickle("/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl")
    mean_rate = compute_mean_spike_rate_for_cells(neurons_df)
    print(mean_rate)