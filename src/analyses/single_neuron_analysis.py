import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

from analyses.spike_count import prepare_binned_spike_data
from analyses.statistical_tests import perform_statistical_test_on_dataframe_rows, permutation_anova_test, \
    permutation_kruskal_test


def run_permutation_kruskal_wallis(
    df,
    category_col='MonkeyName',
    neuron_col='NeuronID',
    count_col='SpikeCount',
    n_permutations=1000,
    alpha=0.05,
    plot=True,
    random_state=None
):
    """df has to be trial-level spike counts -- perform aggregate_trial_level before passing it in"""
    all_results = []
    unique_neurons = df[neuron_col].unique()

    for neuron in tqdm(unique_neurons, desc="Running permutation Kruskal-Wallis per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        grouped = neuron_df.groupby(category_col)[count_col].apply(list)

        # Safety check: skip neurons with fewer than 2 monkeys in data
        if len(grouped) < 2:
            print(f"Neuron {neuron}: Skipped — fewer than 2 monkeys.")
            continue

        permutation_kw_input_df = pd.DataFrame([grouped])

        # Run permutation Kruskal-Wallis (same flow as perm ANOVA)
        results, total_significant, details = perform_statistical_test_on_dataframe_rows(
            permutation_kw_input_df,
            test_func=permutation_kruskal_test,
            num_permutations=n_permutations,
            random_state=random_state,
            alpha=alpha
        )

        for result in results:
            index, h_stat, p_value = result
            all_results.append({
                'NeuronID': neuron,
                'H-statistic': h_stat,
                'p-value': p_value
            })

            if plot:
                detail = details.get(index, {})
                extras = detail.get('extras')
                if extras:
                    perm_h_stats, = extras  # because permutation_kruskal_test returns (H, p, perm_H_list)
                    plot_permutation_kw_distribution(
                        perm_h_stats=perm_h_stats,
                        observed_h_stat=detail['stat'],
                        neuron_id=neuron,
                        category_name=category_col,
                        output_dir="permutation_kw_plots"
                    )

    if all_results:
        results_df = pd.DataFrame(all_results)
        significant_df = results_df[results_df['p-value'] < alpha]

        print(f"\nPermKW Significant neurons (p < {alpha}):")
        print(significant_df.head())
        return results_df, significant_df
    else:
        print("No valid neurons found for permutation Kruskal-Wallis.")
        return pd.DataFrame(), pd.DataFrame()


def run_permutation_anova(df, category_col='MonkeyName', neuron_col='NeuronID', count_col='SpikeCount', n_permutations=1000, alpha=0.05, plot=True):
    """df has to be trial-level spike counts -- perform aggregate_trial_level before passing it in"""
    all_results = []
    unique_neurons = df[neuron_col].unique()
    for neuron in tqdm(unique_neurons, desc="Running permutation ANOVA per neuron"):
        neuron_df = df[df[neuron_col] == neuron]
        # neuron_df = neuron_df.sort_values([neuron_col, 'TimeBinIndex']) ## add this line if you want the list to be
        grouped = neuron_df.groupby(category_col)[count_col].apply(list)
        # Safety check: skip neurons with fewer than 2 monkeys in data
        if len(grouped) < 2:
            print(f"Neuron {neuron}: Skipped — fewer than 2 monkeys.")
            continue

        permutation_anova_input_df = pd.DataFrame([grouped])

        # Run permutation ANOVA
        results, total_significant, details = perform_statistical_test_on_dataframe_rows(
            permutation_anova_input_df,
            test_func=permutation_anova_test,
            num_permutations=n_permutations,
            alpha = alpha
        )

        for result in results:
            index, f_stat, p_value = result
            all_results.append({
                'NeuronID': neuron,
                'F-statistic': f_stat,
                'p-value': p_value
            })

            if plot:
                detail = details.get(index, {})
                extras = detail.get('extras')
                if extras:
                    perm_f_stats, = extras
                    plot_permutation_anova_distribution(
                        perm_f_stats=perm_f_stats,
                        observed_f_stat=detail['stat'],
                        neuron_id=neuron,
                        category_name=category_col,
                        output_dir="permutation_anova_plots"
                    )
    if all_results:
        results_df = pd.DataFrame(all_results)
        significant_df = results_df[results_df['p-value'] < 0.05]
        # print("\nPermANOVA Results:")
        # print(results_df)
        print("\nPermANOVA Significant neurons (p < 0.05):")
        print(significant_df.head())
        return results_df, significant_df
    else:
        print("No valid neurons found for permutation ANOVA.")
        return pd.DataFrame(),pd.DataFrame()

def plot_permutation_anova_distribution(perm_f_stats, observed_f_stat, neuron_id, category_name, output_dir="permutation_anova_plots"):
    os.makedirs(output_dir, exist_ok=True)
    if perm_f_stats is None or observed_f_stat is None or len(perm_f_stats) == 0 or np.isnan(observed_f_stat):
        print(f"[Plot skipped] Invalid data for {neuron_id}")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(perm_f_stats, bins=30, color='skyblue', alpha=0.7, label='Permutation null')
    plt.axvline(observed_f_stat, color='red', linestyle='--', label=f'Observed F = {observed_f_stat:.3f}')

    plt.xlabel('F-statistic')
    plt.ylabel('Frequency')
    plt.title(f'Neuron {neuron_id} | {category_name} Permutation ANOVA')
    plt.legend()

    filename = f"Neuron {neuron_id}_{category_name}_permutation_anova.png"
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    # plt.close()


def plot_permutation_kw_distribution(perm_h_stats, observed_h_stat, neuron_id, category_name, output_dir="permutation_kw_plots"):
    os.makedirs(output_dir, exist_ok=True)
    if perm_h_stats is None or observed_h_stat is None or len(perm_h_stats) == 0 or np.isnan(observed_h_stat):
        print(f"[Plot skipped] Invalid data for {neuron_id}")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(perm_h_stats, bins=30, color='skyblue', alpha=0.7, label='Permutation null')
    plt.axvline(observed_h_stat, color='red', linestyle='--', label=f'Observed H = {observed_h_stat:.3f}')

    plt.xlabel('H-statistic')
    plt.ylabel('Frequency')
    plt.title(f'Neuron {neuron_id} | {category_name} Permutation Kruskal-Wallis')
    plt.legend()

    filename = f"Neuron {neuron_id}_{category_name}_permutation_kruskal_wallis.png"
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    # plt.close()

def plot_permutation_anova_results_summary(anova_results_df):
    plt.figure(figsize=(12, 5))

    # Plot p-value distribution
    plt.subplot(1, 2, 1)
    sns.histplot(anova_results_df['p-value'], bins=20, kde=False)
    plt.title('Permutation ANOVA: P-value Distribution')
    plt.xlabel('P-value')
    plt.ylabel('Neuron Count')

    # Plot F-statistic distribution
    plt.subplot(1, 2, 2)
    sns.histplot(anova_results_df['F-statistic'], bins=20, kde=False)
    plt.title('Permutation ANOVA: F-statistic Distribution')
    plt.xlabel('F-statistic')
    plt.ylabel('Neuron Count')

    plt.tight_layout()
    plt.show()


def multiple_comparison_test(statistical_test_results):
    pval_cols = [c for c in statistical_test_results.columns if re.search(r"p-value", str(c), flags=re.IGNORECASE)]

    if len(pval_cols) == 1:
        pcol = pval_cols[0]

        if not statistical_test_results[pcol].isnull().all():
            reject, pvals_corrected, _, _ = multipletests(
                statistical_test_results[pcol].fillna(1).astype(float),
                method="fdr_bh"
            )
            statistical_test_results[f"{pcol}_corrected"] = pvals_corrected
            statistical_test_results[f"{pcol}_significant"] = reject
        else:
            statistical_test_results[f"{pcol}_corrected"] = None
            statistical_test_results[f"{pcol}_significant"] = None

    elif len(pval_cols) == 0:
        # no p-value column found
        statistical_test_results["pval_corrected"] = None
        statistical_test_results["pval_significant"] = None

    else:
        # more than one match -> force you to disambiguate so you don't correct the wrong column
        raise ValueError(f"Expected exactly one p-value column containing 'p-value', found {len(pval_cols)}: {pval_cols}")

    return statistical_test_results

def plot_significant_neuron_psth(analysis_df, summary_report, time_bin_size=0.05):
    # Filter significant neurons (either GLM or permutation)
    sig_neurons = summary_report[
        (summary_report['GLM_significant'] == True) | (summary_report['Permutation_significant'] == True)][
        'NeuronID'].unique()

    for neuron in sig_neurons:
        neuron_df = analysis_df[analysis_df['NeuronID'] == neuron]

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=neuron_df,
            x=neuron_df['TimeBinIndex'] * time_bin_size,
            y='SpikeCount',
            hue='MonkeyGroup',
            estimator='mean',
            errorbar='sd'
        )

        plt.title(f'Neuron: {neuron} PSTH (Significant)')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike Count')
        plt.legend(title='Stimulus Group')
        plt.tight_layout()
        plt.show()

def detect_significant_time_windows(time_perm_results, alpha=0.05, time_bin_size=0.05):
    """
    Detects significant time windows for each neuron based on permutation p-values.

    Parameters:
        time_perm_results (DataFrame): Time-resolved permutation test results.
        alpha (float): Significance threshold.
        time_bin_size (float): Size of each time bin in seconds.

    Returns:
        DataFrame: Neuron-wise significant time windows.
    """
    records = []

    unique_neurons = time_perm_results['NeuronID'].unique()

    for neuron in unique_neurons:
        neuron_df = time_perm_results[time_perm_results['NeuronID'] == neuron].sort_values('TimeBinIndex')

        # Boolean array: True if significant at this bin
        sig_mask = neuron_df['P-value'] < alpha
        time_bins = neuron_df['TimeBinIndex'].values

        # Detect contiguous significant windows
        if sig_mask.any():
            in_window = False
            window_start = None

            for idx, is_sig in zip(time_bins, sig_mask):
                if is_sig and not in_window:
                    in_window = True
                    window_start = idx
                elif not is_sig and in_window:
                    in_window = False
                    window_end = idx - 1
                    records.append({
                        'NeuronID': neuron,
                        'WindowStart_s': window_start * time_bin_size,
                        'WindowEnd_s': (window_end + 1) * time_bin_size
                    })

            # If ends with a significant window
            if in_window:
                records.append({
                    'NeuronID': neuron,
                    'WindowStart_s': window_start * time_bin_size,
                    'WindowEnd_s': (time_bins[-1] + 1) * time_bin_size
                })

        else:
            records.append({
                'NeuronID': neuron,
                'WindowStart_s': None,
                'WindowEnd_s': None
            })

    return pd.DataFrame(records)


if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1

    analysis_df = prepare_binned_spike_data(date, round_no, 0.05)
    # formula = "SpikeCount ~ C(MonkeyName)"  # Stimulus identity
    formula = "SpikeCount ~ C(MonkeyGroup)"  # Stimulus group 간 차이
