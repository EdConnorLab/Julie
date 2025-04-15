import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm  # progress bar
from statsmodels.stats.multitest import multipletests

from analyses.spike_count import prepare_binned_spike_data
from analyses.statistical_tests import perform_statistical_test_on_dataframe_rows, permutation_anova_test


def run_glm(df, formula="SpikeCount ~ C(MonkeyName)", neuron_col="NeuronID"):
    results = []
    unique_neurons = df[neuron_col].unique()

    for neuron in tqdm(unique_neurons, desc="Running GLM per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        # Skip neurons with too few spikes
        if neuron_df['SpikeCount'].sum() == 0:
            continue
        try:
            model = smf.glm(formula=formula, data=neuron_df, family=sm.families.Poisson()).fit()
            summary = model.summary2().tables[1].reset_index()
            summary['NeuronID'] = neuron
            results.append(summary)
        except Exception as e:
            print(f"Error processing neuron {neuron}: {e}")
            continue

    # Combine results
    if results:
        results_df = pd.concat(results, ignore_index=True)
        return results_df
    else:
        print("No valid neurons found for GLM.")
        return pd.DataFrame()

def plot_glm_coefficients(glm_results):
    coef_df = glm_results[glm_results['index'] != '(Intercept)'].copy()

    # Clean predictor names
    coef_df['Predictor'] = coef_df['index'].str.replace('C\\(MonkeyName\\)\\[T\\.', '', regex=True)
    coef_df['Predictor'] = coef_df['Predictor'].str.replace(']', '')

    # Pivot for heatmap
    pivot_df = coef_df.pivot(index='NeuronID', columns='Predictor', values='Coef.')

    # Plot
    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.3)))
    sns.heatmap(pivot_df, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Coefficient'})
    plt.title('GLM Coefficients (Stimulus Identity Effect per Neuron)')
    plt.xlabel('Stimulus Identity (MonkeyName)')
    plt.ylabel('Neuron ID')
    plt.tight_layout()
    plt.show()

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
    results_df = pd.DataFrame(all_results)
    significant_df = results_df[results_df['p-value'] < 0.05]
    print("\nResults:")
    print(results_df)
    print("\nSignificant neurons (p < 0.05):")
    print(significant_df)


    return results_df

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

def plot_permutation_anova_results_summary(results_df):
    plt.figure(figsize=(12, 5))

    # Plot p-value distribution
    plt.subplot(1, 2, 1)
    sns.histplot(results_df['p-value'], bins=20, kde=False)
    plt.title('Permutation ANOVA: P-value Distribution')
    plt.xlabel('P-value')
    plt.ylabel('Neuron Count')

    # Plot F-statistic distribution
    plt.subplot(1, 2, 2)
    sns.histplot(results_df['F-statistic'], bins=20, kde=False)
    plt.title('Permutation ANOVA: F-statistic Distribution')
    plt.xlabel('F-statistic')
    plt.ylabel('Neuron Count')

    plt.tight_layout()
    plt.show()

def merge_glm_and_permutation_anova(glm_results, perm_results):
    # GLM 결과 neuron 별로 p-value 정리
    glm_summary = glm_results.groupby('NeuronID')['P>|z|'].min().reset_index()
    glm_summary.rename(columns={'P>|z|': 'GLM_p-value'}, inplace=True)

    # permutation 결과는 이미 neuron 별로 되어 있음
    # Just changing the column name
    perm_results.rename(columns={'p-value': 'PermANOVA_p-value'}, inplace=True)

    merged = pd.merge(glm_summary, perm_results, on='NeuronID', how='outer')

    # Multiple comparison correction
    # for GLM
    if not merged['GLM_p-value'].isnull().all():
        reject_glm, glm_pvals_corrected, _, _ = multipletests(merged['GLM_p-value'].fillna(1), method='fdr_bh')
        merged['GLM_pval_corrected'] = glm_pvals_corrected
        merged['GLM_significant'] = reject_glm
    else:
        merged['GLM_pval_corrected'] = None
        merged['GLM_significant'] = None

    # for Permutation
    if not merged['PermANOVA_p-value'].isnull().all():
        reject_perm, perm_pvals_corrected, _, _ = multipletests(merged['PermANOVA_p-value'].fillna(1), method='fdr_bh')
        merged['Permutation_pval_corrected'] = perm_pvals_corrected
        merged['Permutation_significant'] = reject_perm
    else:
        merged['Permutation_pval_corrected'] = None
        merged['Permutation_significant'] = None

    return merged

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

def time_resolved_glm(df, neuron_col='NeuronID', time_col='TimeBinIndex', group_col='MonkeyGroup',
                      count_col='SpikeCount'):
    results = []

    unique_neurons = df[neuron_col].unique()
    time_bins = df[time_col].unique()

    for neuron in tqdm(unique_neurons, desc="Time-resolved GLM per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        for time_bin in time_bins:
            bin_df = neuron_df[neuron_df[time_col] == time_bin]

            # Skip empty bins
            if bin_df.empty or bin_df[count_col].sum() == 0:
                continue

            try:
                model = smf.glm(
                    formula=f"{count_col} ~ C({group_col})",
                    data=bin_df,
                    family=sm.families.Poisson()
                ).fit()

                for predictor in model.params.index:
                    if predictor == '(Intercept)':
                        continue  # Skip intercept
                    results.append({
                        'NeuronID': neuron,
                        'TimeBinIndex': time_bin,
                        'Predictor': predictor,
                        'Coef.': model.params[predictor],
                        'P-value': model.pvalues[predictor]
                    })

            except Exception as e:
                print(f"Error in neuron {neuron}, time bin {time_bin}: {e}")
                continue

    return pd.DataFrame(results)

def plot_time_resolved_glm(time_glm_results, time_bin_size=0.05):
    unique_neurons = time_glm_results['NeuronID'].unique()

    for neuron in unique_neurons:
        neuron_df = time_glm_results[time_glm_results['NeuronID'] == neuron]

        plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=neuron_df,
            x=neuron_df['TimeBinIndex'] * time_bin_size,
            y='Coef.',
            hue='Predictor'
        )

        plt.title(f'Neuron {neuron} - Time-Resolved GLM Coefficients')
        plt.xlabel('Time (s)')
        plt.ylabel('Coefficient (log spike count)')
        plt.legend(title='Stimulus Group')
        plt.tight_layout()
        plt.show()

def time_resolved_permutation_test(df, n_permutations=1000, neuron_col='NeuronID', time_col='TimeBinIndex',
                                   group_col='MonkeyGroup', count_col='SpikeCount'):
    results = []

    unique_neurons = df[neuron_col].unique()
    time_bins = df[time_col].unique()

    for neuron in tqdm(unique_neurons, desc="Time-resolved permutation test per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        for time_bin in time_bins:
            bin_df = neuron_df[neuron_df[time_col] == time_bin]

            if bin_df.empty or bin_df[count_col].sum() == 0:
                continue

            # Observed difference: max - min group mean
            group_means = bin_df.groupby(group_col)[count_col].mean()
            if group_means.shape[0] < 2:
                continue
            observed_diff = group_means.max() - group_means.min()

            # Permutation distribution
            perm_diffs = []
            for _ in range(n_permutations):
                shuffled = bin_df.copy()
                shuffled[group_col] = np.random.permutation(shuffled[group_col].values)
                perm_group_means = shuffled.groupby(group_col)[count_col].mean()
                perm_diff = perm_group_means.max() - perm_group_means.min()
                perm_diffs.append(perm_diff)

            p_value = np.mean(np.array(perm_diffs) >= observed_diff)

            results.append({
                'NeuronID': neuron,
                'TimeBinIndex': time_bin,
                'ObservedDifference': observed_diff,
                'P-value': p_value
            })

    return pd.DataFrame(results)

def plot_time_resolved_significance(time_glm_results, time_perm_results, neuron_id, time_bin_size=0.05):
    glm_df = time_glm_results[time_glm_results['NeuronID'] == neuron_id]
    perm_df = time_perm_results[time_perm_results['NeuronID'] == neuron_id]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # GLM coefficient plot
    sns.lineplot(
        data=glm_df,
        x=glm_df['TimeBinIndex'] * time_bin_size,
        y='Coef.',
        hue='Predictor',
        ax=ax1
    )
    ax1.set_ylabel('GLM Coefficient (log scale)')
    ax1.set_xlabel('Time (s)')

    # Permutation p-value overlay
    ax2 = ax1.twinx()
    sns.lineplot(
        data=perm_df,
        x=perm_df['TimeBinIndex'] * time_bin_size,
        y='P-value',
        color='black',
        label='Permutation p-value',
        ax=ax2
    )
    ax2.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
    ax2.set_ylabel('Permutation p-value')

    fig.suptitle(f'Time-Resolved Analysis: Neuron {neuron_id}')
    fig.legend(loc='upper right')
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

    # glm_results = run_glm(analysis_df, formula=formula)
    #
    # print(glm_results.head())
    # # ---- multiple comparison correction ---
    # # glm_results 에서 P-value 컬럼 선택
    # pvals = glm_results['P>|z|']
    #
    # # correction 방법: 'fdr_bh' (False Discovery Rate, Benjamini/Hochberg)
    # reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    #
    # # glm_results 에 corrected p-value 와 reject 여부 추가
    # glm_results['pval_corrected'] = pvals_corrected
    # glm_results['significant'] = reject
    # glm_results.to_excel('glm_results.xlsx')
    # plot_glm_coefficients(glm_results)
    #
    # # ---- permutation test
    # perm_results = permutation_test(
    #     analysis_df,
    #     n_permutations=1000,
    #     neuron_col='NeuronID',
    #     category_col='MonkeyGroup',  # stimulus group
    #     count_col='SpikeCount'
    # )
    #
    # print(perm_results.head())
    #
    # # ---- multiple comparison correction ---
    # reject, pvals_corrected, _, _ = multipletests(perm_results['P-value'], method='fdr_bh')
    # perm_results['pval_corrected'] = pvals_corrected
    # perm_results['significant'] = reject
    # plot_permutation_results(perm_results)
    #
    # # --- merge glm and permutation ---
    # summary_report = merge_glm_and_permutation(glm_results, perm_results)
    # plot_significant_neuron_psth(analysis_df, summary_report)
    #
    #
    #
    # ### Report
    #
    # generate_neuron_reports(
    #     analysis_df=analysis_df,
    #     glm_results=glm_results,
    #     perm_results=perm_results,
    #     output_dir='neuron_pdf_reports'
    # )

    # time_glm_results = time_resolved_glm(analysis_df)
    # print(time_glm_results.columns)
    # print(time_glm_results.head())
    # plot_time_resolved_glm(time_glm_results)
    time_perm_results = time_resolved_permutation_test(analysis_df, n_permutations=1000)

    print(time_perm_results.head())
    significant_windows = detect_significant_time_windows(time_perm_results, alpha=0.05, time_bin_size=0.05)
    # example_neuron = time_glm_results['NeuronID'].unique()[0]
    # plot_time_resolved_significance(time_glm_results, time_perm_results, example_neuron)
    print(significant_windows.head())
