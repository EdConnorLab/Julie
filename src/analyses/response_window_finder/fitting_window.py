import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from anova_on_spike_counts import perform_anova_on_dataframe_rows_for_time_windowed
from cusum import extract_consecutive_ranges, find_corresponding_values_for_index_ranges
from monkey_names import Zombies
from spike_count import count_spikes_per_bin, get_spike_count_for_single_neuron_with_time_window
from spike_rate_computation import get_raw_data_and_channels_from_files

def unique_elements(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist] # Flatten the list of lists into a single list
    unique = list(set(flat_list)) # Remove duplicates

    return sorted(unique)

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


def expand_window_from_max_threshold(spike_count, threshold = 0.5):
    print(spike_count)
    max_value = max(spike_count)
    all_windows = []
    window_indices = []
    if max_value > 2:
        indices_of_max = [ind for ind, value in enumerate(spike_count) if value == max_value]
        print(f"max values are in {indices_of_max}")
        for i in indices_of_max:
            window_indices = [i]
            after = i + 1
            before = i - 1
            while after <= len(spike_count) - 1:
                if spike_count[after] > max_value * threshold:
                    window_indices.append(after)
                    after += 1
                else:
                    break
            while before >= 0:
                if spike_count[before] > max_value * threshold:
                    window_indices.append(before)
                    before -= 1
                else:
                    break
            if len(window_indices) > 1:
                window_indices.sort()
                spike_values_in_window = [spike_count[i] for i in window_indices]
                print(f"spike values in window {spike_values_in_window}")
                all_windows.append(window_indices)
    else:
        print("No spikes detected")
    # remove duplicates
    unique_windows = unique_elements(all_windows)
    print(f"unique windows {unique_windows}")
    final_windows = extract_consecutive_ranges(unique_windows)
    print(f"final windows {final_windows}")
    spike_values_in_window = [spike_count[i] for i in unique_windows]
    print(f"all spike values: {spike_values_in_window}")
    return final_windows, spike_values_in_window


def expand_window_from_dynamic_threshold(spike_count, threshold = 0.5):
    max_value = max(spike_count)
    start_threshold = 0.7
    high_value = max_value * start_threshold
    all_indices = []
    all_windows = []
    print(spike_count)
    print(f'max_value is: {max_value}')
    print(f'high value is: {high_value}')
    if max_value > 2:
        starting_indices = [ind for ind, value in enumerate(spike_count) if value >= high_value]
        print(f"starting indices are {starting_indices}")
        for i in starting_indices:
            initial_window_start = i
            all_indices.append(i)
            while i < len(spike_count) - 1:
                if spike_count[i + 1] > spike_count[i] * threshold:
                    all_indices.append(i + 1)
                    i += 1
                else:
                    break
            i = initial_window_start
            while i > 0:
                if spike_count[i - 1] > spike_count[i] * threshold:
                    all_indices.append(i - 1)
                    i -= 1
                else:
                    break
            if len(all_indices) > 1:
                all_indices.sort()
                all_windows.append(all_indices)
    else:
        pass
        # print(f"no spikes detected")

    # remove duplicates
    #print(f"spike count {spike_count}")
    win_indices = unique_elements(all_windows)
    print(f"window indices {win_indices}")
    # combine two windows if they are only two indices apart
    #final_win_indices= fill_missing_ones(win_indices)
    final_win_indices = win_indices
    final_windows = extract_consecutive_ranges(final_win_indices)
    print(f"final windows {final_windows}")
    spike_values_in_window = [spike_count[i] for i in final_win_indices]
    print(f"all spike values: {spike_values_in_window}")
    return final_windows, spike_values_in_window, final_win_indices

if __name__ == '__main__':

    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[6]
    del zombies[-1]
    excel = pd.ExcelFile("/home/connorlab/Documents/GitHub/Julie/window_finder_results/Cortana_Windows.xlsx")
    all_rounds = excel.parse('AllRounds')
    results = []
    time_chunk_size = 0.05  # in sec
    time = np.arange(time_chunk_size, 4.00, time_chunk_size)
    rounded_time = np.round(time, 2)
    # date = "2023-09-26"
    # round_no = 1
    for index, row in all_rounds.iterrows():
        date = str(row['Date'])
        round_no = row['Round']
        date_only = row['Date'].strftime('%Y-%m-%d')
        print(f"we will search for windows in {date_only} round {round_no}")
        raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)

        time_chunk_size = 0.05  # in sec

        spike_counts = count_spikes_per_bin(zombies, raw_unsorted_data, valid_channels, time_chunk_size)
        spike_counts['total_sum'] = spike_counts.apply(lambda row: [sum(elements) for elements in zip(*row)], axis=1)
        total_sum = spike_counts['total_sum']
        # print(total_sum)
        for channel, spike_total in total_sum.items():
            windows1, spike_values, win_indices = expand_window_from_dynamic_threshold(spike_total, threshold=0.5)
            time_windows = find_corresponding_values_for_index_ranges(windows1, rounded_time)

            if len(windows1) > 0:
                print('')
                print(f"Final Results: {date_only} Round No. {round_no} ")
                print(f"---------------For {channel}: {windows1}")
                y_values_at_change_points = [spike_total[i] for i in win_indices]
                t_values_at_change_points = [rounded_time[i] for i in win_indices]
                # print(f"---------------For {ind}: {spike_values}")
                # print(f"---------------For {ind}: {time_windows}")
                # smoothed_data = gaussian_filter1d(spike_total, sigma = 0.8)
                # # Plot
                plt.figure(figsize=(12, 6))
                plt.plot(time[:len(spike_total)], spike_total, label='Total Spike Count')
                plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)
                # plt.plot(time[:len(smoothed_data)], smoothed_data, label='smoothed data')
                # Using your function to get consecutive ranges
                consecutive_ranges = extract_consecutive_ranges(win_indices)
                print(f"consecutive ranges: {consecutive_ranges}")
                # Shading consecutive ranges
                for start, end in consecutive_ranges:
                    plt.fill_betweenx([0, max(spike_total)], rounded_time[start], rounded_time[end], color='red', alpha=0.4)


                plt.title(f'{date_only} Round {round_no} {channel} Fitting Window Algorithm')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.show()

                results.append({
                    'Date': date_only,
                    'Round No.': round_no,
                    'Cell': str(channel),
                    'Time Window': time_windows
                })

    # Save
    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values(by = ['Date', 'Round No.', 'Cell'])
    results_sorted.to_excel('fit_window_before_explode.xlsx')

    print(results_sorted.shape)
    results_expanded = results_sorted.explode('Time Window')
    results_expanded.to_excel('fit_window_after_explode.xlsx')

    '''
        Date Created: 2025-01-29
        ANOVA for windows found from fitting window on max count algorithm
    '''
    results_expanded['Time Window'] = results_expanded['Time Window'].apply(
        lambda s: tuple(int(float(num) * 1000) for num in s.strip('()').split(',')))

    print(
        "----------------------------------------------- fit windows ----------------------------------------------------------")
    print(results_expanded)
    fit_spike_count = get_spike_count_for_single_neuron_with_time_window(results_expanded)

    print(fit_spike_count)
    zombies_columns = [col for col in zombies if col in fit_spike_count.columns]
    additional_columns = ['Date', 'Round No.', 'Time Window']
    zombies_fit_spike_count = fit_spike_count[zombies_columns + additional_columns]

    fit_anova_results, fit_sig_results = perform_anova_on_dataframe_rows_for_time_windowed(zombies_fit_spike_count)
    print('------------------------------------ fit window results -----------------------------------')
    # print(fit_anova_results)
    print(fit_sig_results)
    fit_anova_results.to_excel('fit_anova_results.xlsx')
    fit_sig_results.to_excel('fit_sig_results.xlsx')

