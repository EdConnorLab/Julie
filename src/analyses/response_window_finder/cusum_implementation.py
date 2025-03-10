import numpy as np
import pandas as pd

from channel_enum_resolvers import convert_to_enum
from cusum import compute_total_sum_of_spikes, min_max_scale, cusum, extract_consecutive_ranges, \
    find_corresponding_values_for_index_ranges, z_score, find_all_corresponding_values_falling_within_index_ranges
from initial_4feature_lin_reg import get_metadata_for_preliminary_analysis
from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from spike_rate_computation import get_raw_data_and_channels_from_files, get_raw_spike_tstamp_data

if __name__ == '__main__':
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    time_chunk_size = 0.05  # in sec
    rounded_time = np.round(np.arange(time_chunk_size, 3.00, time_chunk_size), 2)
    results = []
    raw_metadata = get_metadata_for_preliminary_analysis()
    metadata = raw_metadata.copy()
    for _, row in metadata.iterrows():
        date = str(row['Date'])
        round_no = row['Round No.']
        date_only = row['Date'].strftime('%Y-%m-%d')
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, valid_channels, time_chunk_size)

        spike_counts['response_windows'] = None
        for index, r in spike_counts.iterrows():
            data = r['total_sum']
            normalized_data = z_score(data)
            # Parameters
            k = 1.1  # sensitivity parameter
            h = 1.1 # threshold
            cusum_pos, cusum_neg, change_points = cusum(normalized_data, 0, k, h)
            windows = extract_consecutive_ranges(change_points)
            time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
            if len(time_windows) > 0:
                print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                print(time_windows)
                results.append({
                    'Date': date_only,
                    'Round No.': round_no,
                    'Cell': str(index),
                    'Time Window': time_windows
                })
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_excel('response_windows.xlsx')
