import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind

from analyses.spike_count import prepare_binned_spike_data, aggregate_trial_level
from glm_permutation_tests import run_permutation_anova


# ================================
# Core statistical tests
# ================================

def anova_test(groups):
    """Standard one-way ANOVA."""
    return f_oneway(*groups)


def kruskal_test(groups):
    """Non-parametric Kruskal-Wallis test."""
    return kruskal(*groups)


def u_test(groups):
    """
    Perform Mann-Whitney U test (non-parametric test for two independent samples).
    (non-parametric alternative to t-test)
    Assumes two groups only.
    """
    if len(groups) != 2:
        raise ValueError(f"Mann-Whitney U test requires exactly 2 groups, but got {len(groups)}.")

    group1, group2 = groups
    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
    return stat, p_val


def t_test(groups):
    """
    Perform independent two-sample t-test.
    Assumes two groups only.
    """
    if len(groups) != 2:
        raise ValueError(f"T-test requires exactly 2 groups, but got {len(groups)}.")

    group1, group2 = groups
    stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test, safer for unequal variances
    return stat, p_val


def permutation_anova_test(groups, num_permutations=1000):
    """Permutation-based one-way ANOVA."""
    data = np.concatenate(groups)
    original_group_sizes = [len(group) for group in groups]
    observed_f_stat, _ = f_oneway(*groups)

    permutation_f_stats = []
    for _ in range(num_permutations):
        np.random.shuffle(data)
        new_groups = np.split(data, np.cumsum(original_group_sizes)[:-1])
        f_stat, _ = f_oneway(*new_groups)
        permutation_f_stats.append(f_stat)

    p_value = np.mean([f_stat >= observed_f_stat for f_stat in permutation_f_stats])
    return observed_f_stat, p_value, permutation_f_stats

# ================================
# Row-wise DataFrame tests
# ================================

def perform_statistical_test_on_dataframe_rows(df, test_func, alpha=0.05, print_results=True, **kwargs):
    """
    General-purpose function to apply a statistical test to each row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row contains lists or arrays of values for each group.
    test_func : function
        The statistical test function to apply. Must accept a list of groups as the first argument.
    alpha : float, optional
        Significance threshold for counting significant results.
    print_results : bool, optional
        Whether to print results row by row.
    **kwargs :
        Additional keyword arguments passed to the test function.

    Returns
    -------
    results : list of tuples
        Each tuple contains (row index, test statistic, p-value).
    total_significant : int
        Total number of rows with p-value < alpha.
    """

    results = []
    total_significant = 0
    details = {}  # To store additional outputs per row, if any

    for index, row in df.iterrows():
        # Extract groups: list of non-empty lists or arrays
        groups = [np.array(cell) for cell in row if isinstance(cell, (list, np.ndarray)) and len(cell) > 0]

        # Safety check: skip if fewer than 2 groups or any empty group
        if len(groups) < 2:
            if print_results:
                print(f"Row {index}: Skipped — less than 2 valid groups.")
            continue

        try:
            # Apply test function
            output = test_func(groups, **kwargs)
            # Support flexible return values
            if isinstance(output, tuple):
                stat, p_value, *extras = output
            else:
                raise ValueError("test_func must return at least (stat, p_value)")

            # NaN check
            if math.isnan(stat) or math.isnan(p_value):
                if print_results:
                    print(f"Row {index}: Skipped — NaN result.")
                continue

            # Store result
            results.append((index, stat, p_value))
            # Store extras if available
            if extras:
                details[index] = {
                    'extras': extras,
                    'stat': stat,
                    'p_value': p_value
                }
            # Count significant
            if p_value < alpha:
                total_significant += 1
                if print_results:
                    print(f"Row {index}: Stat = {stat:.4f}, p = {p_value:.4f}")

        except ValueError as e:
            # If test fails (e.g., not enough data points), skip row
            if print_results:
                print(f"Row {index}: Error — {e}")
            continue

    return results, total_significant, details


# ================================
# Everything below this line needs to be refactored
# ================================


def perform_anova_on_dataframe_rows_for_time_windowed(df):
    """
    Perform one-way ANOVA on rows of a DataFrame
    """
    results = []
    significant_results = []
    for index, row in df.iterrows():
        # Extract groups as lists
        groups = [group for group in row if isinstance(group, list)]
        # Perform OneWay ANOVA

        f_val, p_val = f_oneway(*groups)
        results.append({'Date': row['Date'], 'Round No.': row['Round No.'],
                        'Time Window': row['Time Window'],
                        'Cell': index, 'F Value': f_val, 'P Value': p_val})
        if p_val < 0.05:
            # Collect significant result data
            significant_results.append({
                'Date': row['Date'],
                'Round No.': row['Round No.'],
                'Time Window': row['Time Window'],
                'Cell': index,
                'P Value': p_val
            })
    results_df = pd.DataFrame(results)
    significant_results_df = pd.DataFrame(significant_results)
    return results_df, significant_results_df


def generate_sliding_time_windows(window_size, step_size, total_duration=2000):
    """
    Generate a list of time windows (tuples) using numpy for efficient computation.
    The windows are sliding across a specified total duration with overlap.

    Parameters:
    window_size (int): The size of each time window in milliseconds.
    step_size (int): The step size in milliseconds by which the window slides.
    total_duration (int): Total duration in milliseconds over which to generate windows (default 2000 ms).

    Returns:
    numpy.ndarray: Array of tuples, each representing a time window with a start and end time.
    """
    # Validate inputs
    if window_size <= 0 or step_size <= 0 or window_size > total_duration:
        raise ValueError("Invalid window size, step size, or total duration.")

    # Create start points using np.arange
    start_points = np.arange(0, total_duration - window_size + 1, step_size)

    # Create an array of windows using start points
    windows = np.array(list(zip(start_points, start_points + window_size)))

    return windows


def generate_time_windows_for_given_window_size(window_size):
    """
    Generate a list of time windows (tuples) representing ranges with a specified window size using numpy.

    Parameters:
    window_size (int): Must be a positive integer and should not exceed 2000 ms.

    Returns:
    numpy.ndarray: Array of tuples, each tuple represents a range.

    Raises:
    ValueError:
        If the window_size is not a positive integer or exceeds 2000 ms.
    """
    if not isinstance(window_size, int) or window_size <= 0 or window_size > 2000:
        raise ValueError("Window size must be a positive integer and not exceed 2000 ms.")

    starts = np.arange(0, 2000, window_size)
    ends = starts + window_size
    return np.array(list(zip(starts, ends)))


if __name__ == '__main__':
    date = "2023-09-26"
    round_no = 2
    analysis_df = prepare_binned_spike_data(date, round_no, 0.05)
    filtered_df = analysis_df[analysis_df['MonkeyGroup'] == 'Zombies']
    zombies_trial_df = aggregate_trial_level(filtered_df)
    results = run_permutation_anova(zombies_trial_df, category_col='MonkeyName', neuron_col='NeuronID', count_col='SpikeCount',
                          n_permutations=1000, alpha=0.05, verbose=True)

    # neuron_id = analysis_df['NeuronID'].unique()[0]  # or pick any neuron you like
    # neuron_df = analysis_df[analysis_df['NeuronID'] == neuron_id]
    # # Group by TimeBinIndex and StimulusGroup
    # grouped = neuron_df.groupby(['TimeBinIndex', 'MonkeyName'])['SpikeCount'].apply(list)
    # # Pivot table: rows = time bins, columns = stimulus groups, values = lists of spike counts
    # anova_input_df = grouped.unstack(fill_value=[]).reset_index(drop=True)
    # results, total_significant = perform_test_on_dataframe_rows(
    #     anova_input_df,
    #     test_func=permutation_anova_test,
    #     num_permutations=1000
    # )
    # print(results)
    # print(total_significant)

    '''
    Date Created : 2024-04-29
    ANOVA for selected cells from Ed (time windowed)
    
    Last Updated: 2024-07-03 
    Latest Updates: running ANOVA for BestFrans
    '''
    # # Get list of monkey names
    # zombies = [member.value for name, member in Zombies.__members__.items()]
    # bestfrans = [member.value for name, member in BestFrans.__members__.items()]
    #
    # # Get metadata for list of cells (time windowed)
    # time_windowed_cells = get_metadata_for_list_of_cells_with_time_window("BestFrans_Cells")
    #
    # # calculate spike count
    # spike_count = get_spike_count_for_single_neuron_with_time_window(time_windowed_cells)
    # bestfrans_columns = [col for col in zombies if col in spike_count.columns]
    # bestfrans_spike_count = spike_count[bestfrans_columns]
    # # adding in the Date and Round No. columns because above operation got rid of it
    # bestfrans_spike_count['Date'] = spike_count['Date']
    # bestfrans_spike_count['Round No.'] = spike_count['Round No.']
    #
    # anova_results, sig_results = perform_anova_on_dataframe_rows_for_time_windowed(bestfrans_spike_count)
    # sig_results.to_csv('sig_ANOVA_results_for_2nd_list_zombies.csv')
    # anova_results.to_csv('Windowed_ANOVA_sig_results_Zombies_new.csv')

    '''
    Date Created: 2024-04-29
    Last Updated: 2024-??-??
    ANOVA or PermANOVA on all rounds from metadata
    '''
    # reader = RecordingMetadataReader()
    # metadata_for_analysis = reader.get_metadata_for_preliminary_analysis()
    # total_sig_cells = 0
    # all_significant_results = []
    # for _, row in metadata_for_analysis.iterrows():
    #     date = row['Date'].strftime('%Y-%m-%d')
    #     round_no = row['Round No.']
    #     spike_count_dataframe = get_spike_count_for_each_trial(date, round_no)
    #     # anova_results, sig_results = perform_anova_on_rows(spike_count_dataframe)
    #     results, sig = perform_anova_permutation_test_on_rows(spike_count_dataframe, num_permutations=1000)
    #     for result in results:
    #         index, f_stat, p_value = result
    #         to_be_saved = [date, round_no, index, f_stat, p_value]
    #         all_significant_results.append(to_be_saved)
    #     total_sig_cells = total_sig_cells + sig
    #     # print(f'For {date} Round No. {round_no}')
    #     # all_significant_results.extend(sig_results)
    #     # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'P-Value'])
    #     # results_file_path = 'significant_anova_results.xlsx'
    #     # results_df.to_excel(results_file_path, index=False)
    # results_df = pd.DataFrame(all_significant_results, columns=['Date', 'Round No.', 'Cell', 'F-statistics', 'P-Value'])
    # results_file_path = 'significant_anova_results.xlsx'
    # results_df.to_excel(results_file_path, index=False)
    # print(total_sig_cells)
