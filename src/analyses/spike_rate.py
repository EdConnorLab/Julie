import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.spike_count import extract_spike_counts_from_windows
from data_access.spike_source import SpikeSource


def _add_trial_spike_rate_columns(exploded_df: pd.DataFrame) -> pd.DataFrame:
    df = exploded_df.copy()
    df["SpikeCount"] = df["SpikeTimes"].apply(len)
    df["EpochDuration"] = df["EpochStartStop"].apply(lambda x: float(x[1] - x[0]))
    df["SpikeRate"] = df["SpikeCount"] / df["EpochDuration"].replace(0, np.nan)
    return df

def compute_mean_spike_rate_for_windows_from_source(
    windows_df: pd.DataFrame,
    *,
    source: SpikeSource,
) -> pd.DataFrame:
    """
    Mean rate per NeuronID × MonkeyName × (WindowStart_ms, WindowEnd_ms).
    Uses your canonical extract_spike_counts_from_windows (already source-based).
    """
    if windows_df is None or windows_df.empty:
        return pd.DataFrame(columns=["NeuronID", "MonkeyName", "MonkeyGroup", "WindowStart_ms", "WindowEnd_ms", "MeanSpikeRate"])

    counts_df = extract_spike_counts_from_windows(windows_df, source=source)
    if counts_df is None or counts_df.empty:
        return pd.DataFrame(columns=["NeuronID", "MonkeyName", "MonkeyGroup", "WindowStart_ms", "WindowEnd_ms", "MeanSpikeRate"])

    counts_df = counts_df.copy()
    counts_df["WindowDur_s"] = (counts_df["WindowEnd_ms"] - counts_df["WindowStart_ms"]) / 1000.0
    counts_df["SpikeRate"] = counts_df["SpikeCount"] / counts_df["WindowDur_s"].replace(0, np.nan)

    mean_rate = (
        counts_df
        .groupby(["NeuronID", "MonkeyName", "MonkeyGroup", "WindowStart_ms", "WindowEnd_ms"], as_index=False)
        .agg(MeanSpikeRate=("SpikeRate", "mean"))
    )
    return mean_rate


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


def compute_mean_spike_rate_for_cells_from_source(
    neurons_df: pd.DataFrame,
    *,
    source: SpikeSource,
) -> pd.DataFrame:
    """
    Returns mean spike rate per NeuronID × MonkeyName (and MonkeyGroup).
    Expects the source to return trial-level exploded rows.
    """
    if neurons_df is None or neurons_df.empty:
        return pd.DataFrame(columns=["NeuronID", "MonkeyName", "MonkeyGroup", "MeanSpikeRate"])

    cache: Dict[Tuple[str, int], Optional[pd.DataFrame]] = {}
    rows = []

    # If Date/Round columns exist (newer outputs), prefer them.
    has_session_cols = ("Date" in neurons_df.columns) and ("Round No." in neurons_df.columns)

    for _, r in neurons_df.iterrows():
        neuron_id = r["NeuronID"]

        if has_session_cols:
            date_str = str(r["Date"])
            round_no = int(r["Round No."])
        else:
            # fallback: parse from NeuronID (older style)
            parts = str(neuron_id).split("_", 4)
            date_str = parts[1]
            round_no = int(parts[2])

        key = (date_str, round_no)
        if key not in cache:
            cache[key] = source.load(date_str, round_no)

        exploded_df = cache[key]
        if exploded_df is None or getattr(exploded_df, "empty", True):
            continue

        trials = exploded_df[exploded_df["NeuronID"] == neuron_id]
        if trials.empty:
            continue

        trials = _add_trial_spike_rate_columns(trials)
        rows.append(trials[["NeuronID", "MonkeyName", "MonkeyGroup", "SpikeRate"]])

    if not rows:
        return pd.DataFrame(columns=["NeuronID", "MonkeyName", "MonkeyGroup", "MeanSpikeRate"])

    trial_level = pd.concat(rows, ignore_index=True)
    mean_rate = (
        trial_level
        .groupby(["NeuronID", "MonkeyName", "MonkeyGroup"], as_index=False)
        .agg(MeanSpikeRate=("SpikeRate", "mean"))
    )
    return mean_rate

def compute_population_spike_rates_for_anatomical_region(
        source: SpikeSource,
        anatomical_region: str = "AMG",
) -> pd.DataFrame:
    if anatomical_region not in ("ER", "AMG"):
        raise ValueError(f"anatomical_region must be 'ER' or 'AMG', got '{anatomical_region}'")
    reader = RecordingMetadataReader()
    region = reader.get_metadata_for_brain_region(anatomical_region)

    all_trials = []
    for _, row in region.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']

        exploded_df = source.load(date, round_no)
        if exploded_df is None or exploded_df.empty:
            continue

        trials = _add_trial_spike_rate_columns(exploded_df)
        all_trials.append(trials)

    if not all_trials:
        return pd.Series(dtype=float)

    combined = pd.concat(all_trials, ignore_index=True)
    population_spike_rate = combined.groupby('MonkeyName')['SpikeRate'].mean()
    return population_spike_rate

if __name__ == "__main__":
    pass
    # neurons_df = pd.read_pickle('/old/old_analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl')
    # neuron_mean_rate = compute_mean_spike_rate_for_cells(neurons_df)
    # print(neuron_mean_rate.columns)
    # print(neuron_mean_rate.head())
    # windows_df = pd.read_pickle("/old/old_analysis_cache/Zombies_significant_windows_pANOVAorGLM_passed.pkl")
    # window_mean_rate = compute_mean_spike_rate_for_windows(windows_df)
    # print(window_mean_rate.columns)
    # print(window_mean_rate.head())