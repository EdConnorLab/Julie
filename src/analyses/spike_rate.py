

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
        .groupby(['NeuronID', 'MonkeyName'])['SpikeRate']
        .mean()
        .reset_index()
        .rename(columns={'SpikeRate': 'MeanSpikeRate'})
    )

    return mean_rate_df

def compute_mean_spike_rate_table(exploded_df):
    trial_level_spike_rate_df= add_trial_spike_rate_columns(exploded_df)
    df = get_mean_spike_rate_per_neuron_monkey(trial_level_spike_rate_df)
    return df