import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from channel_enum_resolvers import convert_to_enum
from monkey_names import Zombies
from recording_metadata_reader import RecordingMetadataReader
from spike_count import get_spike_counts_for_time_chunks, get_spike_count_for_single_neuron_with_time_window
from spike_rate_computation import get_raw_data_and_channels_from_files


def cusum(data, mu, k, h):
    """
    Perform CUSUM change detection.
    """
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    change_points = []
    for t in range(1, len(data)):
        cusum_pos[t] = max(0, cusum_pos[t - 1] + data[t] - mu - k)
        cusum_neg[t] = max(0, cusum_neg[t - 1] + mu - data[t] - k)
        if cusum_pos[t] > h or cusum_neg[t] > h:
            change_points.append(t)
    return cusum_pos, cusum_neg, change_points


def extract_consecutive_ranges(numbers):
    """
    Extracts ranges of consecutive numbers from a list of numbers.
    """
    result = []
    if len(numbers) > 0:
        numbers = sorted(numbers)
        start = numbers[0]
        end = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] == end + 1:
                end = numbers[i]
            else:
                if start != end:  # Only append if start and end are not the same
                    result.append((start-1, end))
                start = end = numbers[i]
        if start != end:  # Check again for the last range
            result.append((start-1, end))
    return result


def find_corresponding_values_for_index_ranges(index_ranges, values):
    """
    Extracts elements from a list based on the start and end indices provided in window_index_ranges.
    """
    extracted_values = []
    for start, end in index_ranges:
        if start <= len(values) and end < len(values):
            extracted_values.append((values[start], values[end]))
    return extracted_values

def find_all_corresponding_values_falling_within_index_ranges(index_ranges, values):
    all_corresponding_values = []
    for start, end in index_ranges:
        if start <= len(values) and end < len(values):
            all_corresponding_values.append(values[start:end+1])
    return all_corresponding_values

def fill_missing_ones(numbers):
    if not numbers:
        return []

    filled = []
    i = 0
    while i < len(numbers) - 1:
        filled.append(numbers[i])
        if numbers[i + 1] == numbers[i] + 2:
            filled.append(numbers[i] + 1)
        i += 1
    filled.append(numbers[len(numbers) - 1])
    return filled


def compute_total_sum_of_spikes(raw_data, monkeys, channels, chunk_size):

    spike_counts = get_spike_counts_for_time_chunks(monkeys, raw_data, channels, chunk_size)
    spike_counts['total_sum'] = spike_counts.apply(lambda row: [sum(elements) for elements in zip(*row)], axis=1)

    return spike_counts


def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def z_score(data):
    if np.std(data) == 0:
        return np.zeros(len(data))
    else:
        return (data - np.mean(data)) / np.std(data)

def plot_spike_count_with_response_windows(time, normalized_data, data, cusum_pos, cusum_neg, change_points, threshold):
    # Plotting
    y_values_at_change_points = [data[i] for i in change_points]
    t_values_at_change_points = [time[i] for i in change_points]

    plt.figure(figsize=(12, 6))
    if normalized_data is not None:
        plt.plot(time[:len(normalized_data)], normalized_data, label='Normalized Data')
    if data is not None:
        plt.plot(time[:len(data)], data, label='Data')
    plt.plot(time[:len(data)], cusum_pos, label='CUSUM+', linestyle='--')
    plt.plot(time[:len(data)], cusum_neg, label='CUSUM-', linestyle='--')
    plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    # Using your function to get consecutive ranges
    consecutive_ranges = extract_consecutive_ranges(change_points)
    overall_max = np.maximum.reduce([data, cusum_pos, cusum_neg])
    # max_of_normalized_data = np.maximum.reduce([normalized_data, cusum_pos, cusum_neg])
    # Shading windows
    for start, end in consecutive_ranges:
        plt.fill_betweenx([0, max(overall_max)], time[start], time[end], color='red', alpha=0.4)

    plt.title(f'{date_only} Round {round_no} {index} --- windows based on CUSUM algorithm')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]

    time_chunk_size = 0.05  # in sec
    rounded_time = np.round(np.arange(time_chunk_size, 3.50, time_chunk_size), 2)

    response_window_test = pd.read_excel('response_window_algorithm_validation_test.xlsx')
    for _, row in response_window_test.iterrows():
        date = str(row['Date'])
        round_no = row['Round No.']
        date_only = row['Date'].strftime('%Y-%m-%d')
        channels_to_read = convert_to_enum(row['Cell'])
        raw_unsorted_data, _, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, [channels_to_read], time_chunk_size)

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
            # Plotting
            y_values_at_change_points = [data[i] for i in change_points]
            t_values_at_change_points = [rounded_time[i] for i in change_points]

            plt.figure(figsize=(12, 6))
            # Using your function to get consecutive ranges
            consecutive_ranges = extract_consecutive_ranges(change_points)
            overall_max_with_data = np.maximum.reduce([data, cusum_pos, cusum_neg])
            overall_max_with_norm_data = np.maximum.reduce([normalized_data, cusum_pos, cusum_neg])
            # if normalized_data is not None:
            #     plt.plot(rounded_time[:len(normalized_data)], normalized_data, label='Normalized Data')
            #     for start, end in consecutive_ranges:
            #         plt.fill_betweenx([0, max(overall_max_with_norm_data)], rounded_time[start], rounded_time[end], color='red',
            #                           alpha=0.4)

            if data is not None:
                plt.plot(rounded_time[:len(data)], data, label='Data')
                for start, end in consecutive_ranges:
                    plt.fill_betweenx([0, max(overall_max_with_data)], rounded_time[start], rounded_time[end], color='red',
                                      alpha=0.4)
                plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)

            plt.plot(rounded_time[:len(data)], cusum_pos, label='CUSUM+', linestyle='--')
            plt.plot(rounded_time[:len(data)], cusum_neg, label='CUSUM-', linestyle='--')
            plt.axhline(y=h, color='green', linestyle='--', label='Threshold')



            plt.title(f'{date_only} Round {round_no} {index} --- windows based on CUSUM algorithm')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

    '''
    results = []
    for _, row in all_rounds.iterrows():
        date = str(row['Date'])
        round_no = row['Round No.']
        date_only = row['Date'].strftime('%Y-%m-%d')
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
        spike_counts = compute_total_sum_of_spikes(raw_unsorted_data, zombies, valid_channels, time_chunk_size)

        spike_counts['response_windows'] = None
        for index, r in spike_counts.iterrows():
            # Monkey-specific windows
            # for monkey in zombies:
            #     monkey_specific = r[monkey]
            #     monkey_std_dev = np.std(monkey_specific)
            #     if monkey_std_dev > 0 and max(monkey_specific) > 2:
            #         monkey_mean = np.mean(monkey_specific)  # baseline mean
            #         normalized_data = (monkey_specific - monkey_mean) / monkey_std_dev
            #         max_data = max(monkey_specific)
            #         cusum_pos, cusum_neg, change_points = cusum(normalized_data, max_data, k = 0.9, h = 0.2)
            #         monkey_windows = extract_consecutive_ranges(change_points)
            #         monkey_time_windows = extract_values_from_ranges(monkey_windows, rounded_time)
            #         if len(monkey_time_windows) > 0:
            #             print(f"{monkey} specific {date_only} round no. {round_no} {index}")
            #             print(monkey_time_windows)

            total_sum_data = r['total_sum']
            std_dev = np.std(total_sum_data)
            if std_dev > 0:
                mean = np.mean(total_sum_data) # baseline mean
                # normalized_data = (total_sum_data - mean) / std_dev
                # max_data = max(total_sum_data)
                normalized_data = min_max_scale(total_sum_data)
                max_data = max(normalized_data)
                # Parameters
                k = 0.6  # sensitivity parameter
                h = 0.2  # threshold

                cusum_pos, cusum_neg, change_points = cusum(normalized_data, max_data, k, h)
                windows = extract_consecutive_ranges(change_points)
                time_windows = find_corresponding_values_for_index_ranges(windows, rounded_time)
                spikes_in_window = find_corresponding_values_for_index_ranges(windows, total_sum_data)

                if len(time_windows) > 0:
                    print(f"---------------- {date_only} round no. {round_no} {index}----------------")
                    # print(windows)
                    print(time_windows)

                # Plotting
                # plot_spike_count_with_response_windows(rounded_time, normalized_data, total_sum_data, cusum_pos, cusum_neg, change_points, h)

                if len(time_windows) > 0:
                    results.append({
                        'Date': date_only,
                        'Round No.': round_no,
                        'Cell': str(index),
                        'Time Window': time_windows
                    })

    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values(by=['Date', 'Round No.', 'Cell'])
    # results_sorted.to_excel('cusum_window_before_explode.xlsx')

    print(results_sorted.head())
    results_expanded = results_sorted.explode('Time Window')
    # results_expanded.to_excel('CUSUM_window_cells.xlsx')
    # print('cusum results saved!')
    print(results_expanded.shape)
    
    '''
        # Date Created: 2025-01-29
        # ANOVA for windows found from cusum algorithm
    '''
    # results_expanded['Time Window'] = results_expanded['Time Window'].apply(
    #     lambda s: tuple(int(float(num) * 1000) for num in s.strip('()').split(',')))

    print(
        "--------------------------------------------- cusum windows ----------------------------------------------------------")
    print(results_expanded)

    cusum_spike_count = get_spike_count_for_single_neuron_with_time_window(results_expanded)
    print(cusum_spike_count)

    zombies_columns = [col for col in zombies if col in cusum_spike_count.columns]
    additional_columns = ['Date', 'Round No.', 'Time Window']
    zombies_cusum_spike_count = cusum_spike_count[zombies_columns + additional_columns]
    cusum_anova_results, cusum_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(
        zombies_cusum_spike_count)

    print('------------------------------------ cusum window results -----------------------------------')
    # print(cusum_anova_results)
    print(cusum_sig_results)
    print(cusum_sig_results.shape)
    # cusum_anova_results.to_excel('CUSUM_ANOVA_results.xlsx')
    # cusum_sig_results.to_excel('CUSUM_window_cells_ANOVA_passed.xlsx')
    '''
