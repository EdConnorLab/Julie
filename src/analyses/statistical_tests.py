import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind

from analyses.spike_count import prepare_binned_spike_data, aggregate_trial_level


# ================================
# Core statistical tests
# ================================

def anova_test(groups):
    """Standard one-way ANOVA."""
    return f_oneway(*groups)


def kruskal_test(groups):
    """Non-parametric Kruskal-Wallis test."""
    return kruskal(*groups, nan_policy='omit')


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
    observed_f_stat, _ = anova_test(groups)

    permutation_f_stats = []
    for _ in range(num_permutations):
        np.random.shuffle(data)
        new_groups = np.split(data, np.cumsum(original_group_sizes)[:-1])
        f_stat, _ = anova_test(new_groups)
        permutation_f_stats.append(f_stat)

    p_perm = np.mean([f_stat >= observed_f_stat for f_stat in permutation_f_stats])
    return observed_f_stat, p_perm, permutation_f_stats

def permutation_kruskal_test(groups, num_permutations=1000, random_state=None):
    """Permutation-based Kruskal–Wallis H-test."""

    rng = np.random.default_rng(random_state)
    groups = [np.asarray(g) for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required for Kruskal–Wallis test.")

    # Compute observed statistic
    observed_H, _ = kruskal_test(groups)

    # Combine all data and record original group sizes
    data = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]
    cuts = np.cumsum(group_sizes)[:-1]

    perm_H = []
    for _ in range(num_permutations):
        permuted = rng.permutation(data)
        perm_groups = np.split(permuted, cuts)
        H_stat, _ = kruskal_test(perm_groups)
        perm_H.append(H_stat)

    # finite-sample p-value
    perm_H = np.asarray(perm_H)
    p_perm = (np.sum(perm_H >= observed_H) + 1) / (num_permutations + 1)

    return observed_H, p_perm, perm_H.tolist()


# ================================
# Row-wise DataFrame tests
# ================================

def perform_statistical_test_on_dataframe_rows(df, test_func, alpha=0.05, print_results=False, **kwargs):
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

            details[index] = {
                'extras': [],
                'stat': np.nan,
                'p_value': np.nan
            }
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
                details[index] = {
                    'extras': [],
                    'stat': np.nan,
                    'p_value': np.nan
                }
                continue

            # Store result
            results.append((index, stat, p_value))
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
            details[index] = {
                'extras': [],
                'stat': np.nan,
                'p_value': np.nan
            }
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
