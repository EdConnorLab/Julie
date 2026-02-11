from itertools import zip_longest
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analyses.spike_count import prepare_binned_spike_data, aggregate_timebin_level, extract_spike_counts_from_windows
from analyses.spike_source import SISortedSpikeSource, MixedManualSpikeSource, SpikeSource


def threshold_and_fill_gap(z_scored_data, threshold=0.6):
    # Find change points where the z-score exceeds the threshold
    change_points = np.flatnonzero(np.asarray(z_scored_data) > threshold).tolist()
    change_points = sorted(list(set(change_points)))
    filled = fill_gap_if_one_data_point_away(change_points, z_scored_data)
    return filled


def fill_gap_if_one_data_point_away(change_points, norm_data, threshold=0.5):
    if not change_points:
        return []
    filled = []
    i = 0
    while i < len(change_points) - 1:
        filled.append(change_points[i])
        if change_points[i + 1] == change_points[i] + 2:
            if norm_data[change_points[i] + 1] >= (threshold * norm_data[change_points[i + 1]]) or norm_data[
                change_points[i] + 1] >= (threshold * norm_data[change_points[i]]):
                filled.append(change_points[i] + 1)
        i += 1
    filled.append(change_points[len(change_points) - 1])
    return filled


def list_addition(lists):
    return [sum(x) for x in zip_longest(*lists, fillvalue=0)]

def compute_timebinned_spikecount_per_neuron(
    date: str,
    round_no: int,
    bin_size: float,
    monkey_group: str,
    *,
    source: Optional[SpikeSource] = None,
) -> pd.DataFrame:
    """
    Returns a dataframe:
      NeuronID | TotalSpikeCountList  (list of spike counts per time bin)
    """

    # Backward compatible behavior:
    # - if caller passes source, it wins
    # - otherwise infer from use_sorted

    binned_spike_data = prepare_binned_spike_data(
        date,
        round_no,
        bin_size,
        source=source
    )

    if binned_spike_data is None or binned_spike_data.empty:
        print(f"[Warning] No valid neurons after filtering on {date}, round {round_no}")
        return pd.DataFrame(columns=["NeuronID", "TotalSpikeCountList"])

    bin_level_spike_data = aggregate_timebin_level(binned_spike_data)

    group_data = bin_level_spike_data[bin_level_spike_data["MonkeyGroup"] == monkey_group]
    if group_data.empty:
        return pd.DataFrame(columns=["NeuronID", "TotalSpikeCountList"])

    group_data = group_data.sort_values(["NeuronID", "TimeBinIndex"])
    spikecount_df = (
        group_data.groupby("NeuronID")["SpikeCount"]
        .apply(list)
        .reset_index(name="TotalSpikeCountList")
    )
    return spikecount_df


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
                    result.append((start, end))
                start = end = numbers[i]
        if start != end:  # Check again for the last range
            result.append((start, end))

    # print(result)
    return result


def remove_consecutive_tuples(tuples):
    # List to hold tuples that do not contain consecutive numbers
    filtered_tuples = []

    # Iterate over each tuple in the input list
    for start, end in tuples:
        # Check if the numbers are consecutive
        if end != start + 1:
            filtered_tuples.append((start, end))

    return filtered_tuples


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
            all_corresponding_values.append(values[start:end + 1])
    return all_corresponding_values


def z_score(data):
    if np.std(data) == 0:
        return np.zeros(len(data))
    else:
        return (data - np.mean(data)) / np.std(data)


def detect_response_windows_for_session(
    date: str,
    round_no: int,
    *,
    source: SpikeSource,
    bin_size: float = 0.05,
    monkey_group: str = "Zombies",
    threshold: float = 0.5,
    plot: bool = False,
) -> pd.DataFrame:
    results = []

    rounded_time = np.round(np.arange(bin_size, 3.50, bin_size), 2)

    timebin_spikecount_list = compute_timebinned_spikecount_per_neuron(
        date,
        round_no,
        bin_size,
        monkey_group,
        source=source,                  # ✅ propagate
    )

    for _, r in timebin_spikecount_list.iterrows():
        data = r["TotalSpikeCountList"]
        neuron = r["NeuronID"]
        normalized_data = z_score(np.asarray(data))

        change_points = threshold_and_fill_gap(normalized_data, threshold)
        windows = extract_consecutive_ranges(change_points)
        filtered_windows = remove_consecutive_tuples(windows)
        time_windows = find_corresponding_values_for_index_ranges(filtered_windows, rounded_time)

        for start_time, end_time in time_windows:
            results.append({
                "NeuronID": neuron,
                "WindowStart_ms": int(start_time * 1000),
                "WindowEnd_ms": int(end_time * 1000),
                "Date": date,
                "Round No.": int(round_no)
            })

        if plot:
            y_values_at_change_points = [data[i] for i in change_points]
            t_values_at_change_points = [rounded_time[i] for i in change_points]

            plt.figure(figsize=(12, 6))
            overall_max_for_simple_thresholding = np.maximum.reduce([normalized_data, data])

            if normalized_data is not None:
                plt.plot(rounded_time[:len(normalized_data)], normalized_data, label='Normalized Data')
                for start, end in filtered_windows:
                    plt.fill_betweenx([0, max(overall_max_for_simple_thresholding)], rounded_time[start],
                                      rounded_time[end], color='red', alpha=0.4)

            if data is not None:
                plt.plot(rounded_time[:len(data)], data, label='Data')
                for start, end in filtered_windows:
                    plt.fill_betweenx([0, max(overall_max_for_simple_thresholding)], rounded_time[start],
                                      rounded_time[end], color='red', alpha=0.4)
                plt.scatter(t_values_at_change_points, y_values_at_change_points, color='red', zorder=5)

            plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
            plt.title(f'{neuron} --- window: {time_windows}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame(columns=["NeuronID", "WindowStart_ms", "WindowEnd_ms", "Date", "Round No."])
    return results_df.sort_values(["Date", "Round No.", "NeuronID"]).reset_index(drop=True)



if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1
    bin_size = 0.05
    monkey_group = 'Zombies'

    results_df = detect_response_windows_for_session(
        date, round_no, bin_size=bin_size, monkey_group=monkey_group, threshold=0.5, plot=True
    )

    print(results_df)
