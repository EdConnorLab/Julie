import pandas as pd
from pandas import read_pickle

from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from single_unit_analysis import read_sorted_data


def load_raw_data(date, round_number):
    reader = RecordingMetadataReader()
    pickle_filepath, valid_channels, round_path = reader.get_metadata_for_spike_analysis(date, round_number)
    raw_trial_data = read_pickle(pickle_filepath)

    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        sorted_data = read_sorted_data(round_path)
    else:
        sorted_data = None

    return raw_trial_data, valid_channels, sorted_data

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


if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = load_raw_data(date, round_no)
    print(raw_unsorted_data)
    print(sorted_data)
    combined = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    print(combined)