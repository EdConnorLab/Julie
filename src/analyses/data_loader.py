import warnings

import pandas as pd
from pandas import read_pickle

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.intan_data_processor.single_unit_analysis import read_sorted_data


def load_raw_data(date, round_number):
    reader = RecordingMetadataReader()
    pickle_filepath, curated_channels, round_path = reader.get_metadata_for_spike_analysis(date, round_number)
    raw_trial_data = read_pickle(pickle_filepath)

    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        sorted_data = read_sorted_data(round_path)
    else:
        sorted_data = None

    return raw_trial_data, curated_channels, sorted_data


def combine_unsorted_with_sorted(raw_unsorted_data, sorted_data):
    if sorted_data is None or sorted_data.empty:
        return raw_unsorted_data

    # Step 1: Collect base channel names from sorted_data
    sorted_base_channels = set()
    for spike_dict in sorted_data['SpikeTimes']:
        for ch in spike_dict.keys():
            base_channel = ch.rsplit('_Unit', 1)[0]  # Remove unit info to get base channel
            sorted_base_channels.add(base_channel)

    # Step 2: Prepare unsorted spikes, excluding channels already present in sorted_data
    unsorted_spike_dicts = {}
    for _, row in raw_unsorted_data.iterrows():
        key = (row['TaskField'], row['MonkeyId'], row['MonkeyName'], row['MonkeyGroup'], row['EpochStartStop'])
        spike_times = unsorted_spike_dicts.setdefault(key, {})

        for ch, spikes in row['SpikeTimes'].items():
            ch_str = str(ch)  # Convert enum to string
            if ch_str not in sorted_base_channels:
                spike_times[ch] = spikes

    # Step 3: Prepare sorted spikes, using the same keys
    combined_spike_dicts = {}
    for _, row in sorted_data.iterrows():
        key = (row['TaskField'], row['MonkeyId'], row['MonkeyName'], row['MonkeyGroup'], row['EpochStartStop'])
        combined_spike_dicts[key] = row['SpikeTimes'].copy()

    # Step 4: Merge unsorted spikes into sorted spikes
    for key, unsorted_spikes in unsorted_spike_dicts.items():
        if key in combined_spike_dicts:
            combined_spike_dicts[key].update(unsorted_spikes)
        else:
            combined_spike_dicts[key] = unsorted_spikes

    # Step 5: Reconstruct final DataFrame
    combined_rows = []
    for (taskfield, monkeyid, monkeyname, monkeygroup, epoch), spike_times in combined_spike_dicts.items():
        combined_rows.append({
            'TaskField': taskfield,
            'MonkeyId': monkeyid,
            'MonkeyName': monkeyname,
            'MonkeyGroup': monkeygroup,
            'SpikeTimes': spike_times,
            'EpochStartStop': epoch
        })

    combined_df = pd.DataFrame(combined_rows)
    return combined_df


def load_and_combine_data(date, round_no):
    """Load and combine raw unsorted and sorted spike data."""
    raw_unsorted_data, _, sorted_data = load_raw_data(date, round_no)
    combined_data = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    return combined_data


# --- Data explosion (wide → long format) ---
def explode_spike_data(combined_data, date, round_no, curated_channels_only=False):
    """Explode spike times into long-format DataFrame with metadata."""
    reader = RecordingMetadataReader()
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

    metadata = reader.get_metadata_for_preliminary_analysis()

    exploded_df['BaseChannel'] = exploded_df['Channel'].apply(normalize_channel)
    exploded_df['Date'] = date
    exploded_df['Round No.'] = round_no
    # add locations!
    metadata['Date'] = metadata['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    exploded_df = exploded_df.merge(
        metadata[['Date', 'Round No.', 'Location']],
        on=['Date', 'Round No.'],
        how='left'
    )
    exploded_df['Location'] = exploded_df['Location'].fillna('Unknown')

    exploded_df['NeuronID'] = (
            exploded_df['Location'].astype(str) + "_" +
            exploded_df['Date'].astype(str) + "_" +
            exploded_df['Round No.'].astype(str) + "_" +
            exploded_df['Channel'].astype(str)
    )
    if curated_channels_only:
        warnings.warn(
            "curated_channels_only in explode_spike_data is deprecated; "
            "filtering is now handled at load time",
            DeprecationWarning,
        )

    return exploded_df


if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = load_raw_data(date, round_no)
    print(raw_unsorted_data)
    print(sorted_data)
    combined = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    print(combined)
