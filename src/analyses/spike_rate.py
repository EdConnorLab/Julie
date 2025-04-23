

def compute_trial_level_spike_rate(exploded_df):
    """
    Adds SpikeCount, EpochDuration, and SpikeRate columns to exploded spike data.
    """
    df = exploded_df.copy()
    df['SpikeCount'] = df['SpikeTimes'].apply(len)
    df['EpochDuration'] = df['EpochStartStop'].apply(lambda x: x[1] - x[0])
    df['SpikeRate'] = df['SpikeCount'] / df['EpochDuration']
    return df

def compute_mean_spike_rate(trial_level_spike_rate_df):
    # Take mean spike rate per NeuronID × MonkeyName
    mean_rate_df = (
        trial_level_spike_rate_df
        .groupby(['NeuronID', 'MonkeyName'])['SpikeRate']
        .mean()
        .reset_index()
        .rename(columns={'SpikeRate': 'MeanSpikeRate'})
    )

    return mean_rate_df