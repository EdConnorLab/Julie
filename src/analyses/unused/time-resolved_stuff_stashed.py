import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

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
