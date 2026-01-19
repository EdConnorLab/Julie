import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.behavior_vector_linear_regression import \
    run_directional_vector_linear_regression_cell_level, \
    run_rsa_analysis, run_directional_vector_linear_regression_window_level, \
    expand_window_level_regression_results_with_spike_rates_per_stimulus

from analyses.spike_rate import compute_mean_spike_rate_for_cells, compute_mean_spike_rate_for_windows


def plot_single_neuron_profile(df, neuron_id):
    """
    Plot the full social encoding profile for a single neuron.
    """
    df = df[df['NeuronID'] == neuron_id]

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x='Source_Monkey', y='R-squared', hue='Behavior')
    plt.title(f"Neuron: {neuron_id} — social encoding profile")
    plt.ylabel("R²")
    plt.xlabel("Source Monkey")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    monkey_group = "Zombies"
    subject_monkey_index = 6
    monkey_list = get_monkeys_by_default_order(monkey_group)
    base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'
    behavior_files = {
        "AffiliationTo": "zombies_feature_df_affiliation.xlsx",
        "AffiliationFrom": "zombies_feature_df_affiliation.xlsx",
        "SubmissionTo": "zombies_feature_df_submission.xlsx",
        "SubmissionFrom": "zombies_feature_df_submission.xlsx",
        "AgonismTo": "zombies_feature_df_agonism.xlsx",
        "AgonismFrom": "zombies_feature_df_agonism.xlsx",
    }

    behavior_matrices = {
        name: pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy().T if 'From' in name
        else pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy()
        for name, fname in behavior_files.items()
    }
    # Directional Linear Regression (OLS) on Significant Cells (ONLY PERMUTATION SIGNIFICIANT)
    sig_neurons = pd.read_pickle(
        f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_neurons_pANOVAorGLM_passed.pkl')
    sig_neurons_perm_significant = sig_neurons[sig_neurons["Permutation_significant"]]


    mean_spike_rate_neurons = compute_mean_spike_rate_for_cells(sig_neurons_perm_significant)
    directional_neuron_results = []
    for name, mat in behavior_matrices.items():
        neuron_results_df = run_directional_vector_linear_regression_cell_level(
            mean_spike_rate_neurons, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True, permutation_test=True
        )
        directional_neuron_results.append(neuron_results_df)
    directional_all_neuron_results_df = pd.concat(directional_neuron_results, ignore_index=True)
    directional_all_neuron_results_sorted = directional_all_neuron_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    filtered_df = directional_all_neuron_results_df[
        (directional_all_neuron_results_df['p_value'] < 0.05) &
        (directional_all_neuron_results_df['R-squared'] > 0.5)
        ]
    directional_all_neuron_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_single_neurons_only_pANOVA_significant.pkl')
    print(directional_all_neuron_results_sorted)
    # mean_spike_rate = compute_mean_spike_rate_for_cells(filtered_df)

    # Directional Linear Regression (OLS) on Significant Cells
    sig_neurons = pd.read_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_neurons_pANOVAorGLM_passed.pkl')
    # mean_spike_rate_neurons = compute_mean_spike_rate_for_cells(sig_neurons)
    # directional_neuron_results = []
    # for name, mat in behavior_matrices.items():
    #     neuron_results_df = run_directional_vector_linear_regression_cell_level(
    #         mean_spike_rate_neurons, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True
    #     )
    #     directional_neuron_results.append(neuron_results_df)
    # directional_all_neuron_results_df = pd.concat(directional_neuron_results, ignore_index=True)
    # directional_all_neuron_results_sorted = directional_all_neuron_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    # filtered_df = directional_all_neuron_results_df[
    #     (directional_all_neuron_results_df['p_value'] < 0.05) &
    #     (directional_all_neuron_results_df['R-squared'] > 0.5)
    #     ]
    # directional_all_neuron_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_single_neurons.pkl')
    # print(directional_all_neuron_results_sorted)
    # mean_spike_rate = compute_mean_spike_rate_for_cells(filtered_df)

    # Marginal Linear Regression (OLS) on Significant Cells
    # marginal_neuron_results = []
    # for name, mat in behavior_matrices.items():
    #     neuron_results_df = run_directional_vector_linear_regression(
    #         mean_spike_rate_neurons, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True
    #     )
    #     marginal_neuron_results.append(neuron_results_df)
    # marginal_all_neuron_results_df = pd.concat(marginal_neuron_results, ignore_index=True)
    # marginal_all_neuron_results_sorted = marginal_all_neuron_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    # marginal_all_neuron_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_marg_ols_on_single_neurons.pkl')
    # print(marginal_all_neuron_results_sorted)

    # Directional Linear Regression (OLS) on Significant Windows
    sig_windows = pd.read_pickle(
        f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_windows_pANOVAorGLM_passed.pkl')
    mean_spike_rate_windows = compute_mean_spike_rate_for_windows(sig_windows)
    # all_window_results = []
    # for name, mat in behavior_matrices.items():
    #     results_df = run_directional_vector_linear_regression_window_level(
    #         mean_spike_rate_windows, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True, permutation_test=True, n_perm=1000
    #     )
    #     all_window_results.append(results_df)
    # all_window_results_df = pd.concat(all_window_results, ignore_index=True)
    # all_window_results_sorted = all_window_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    # all_window_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_windows.pkl')
    all_window_results_df = pd.read_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_windows.pkl')
    filtered_df = all_window_results_df[
        (all_window_results_df['p_value'] < 0.05) &
        (all_window_results_df['R-squared'] > 0.5) &
        (all_window_results_df['p_perm'] < 0.05)
        ]
    expanded_results = expand_window_level_regression_results_with_spike_rates_per_stimulus(mean_spike_rate_windows, filtered_df, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True)
    expanded_results.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_windows_expanded.pkl')
    # window_dir_results = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/Zombies_dir_ols_on_windows.pkl')
    # print(window_dir_results)
    # mean_spike_rate = compute_mean_spike_rate_for_windows(filtered_df)

    ### RSA
    # rsa_results = []
    # for name, mat in behavior_matrices.items():
    #     r, p, neural_rsm, social_rsm = run_rsa_analysis(
    #         spike_df=mean_spike_rate,
    #         behavior_matrix=mat,
    #         behavior_name=name,
    #         monkey_list=monkey_list,
    #         subject_idx=6,
    #         method='Euclidean',
    #         use_rate=True
    #     )
    #
    #     rsa_results.append({
    #         'Behavior': name,
    #         'RSA_r': r,
    #         'RSA_p': p
    #     })
    #     print(f"RSA for {name}: r={r:.3f}, p={p:.3f}")

if __name__ == "__main__":
    main()