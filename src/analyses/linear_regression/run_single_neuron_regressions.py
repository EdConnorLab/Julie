import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.directional_behavioral_vector_analysis import run_directional_vector_linear_regression
from analyses.spike_count import extract_spike_counts_from_cells
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

    # Directional Linear Regression (OLS) on Significant Cells
    sig_neurons = pd.read_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_neurons_pANOVAorGLM_passed.pkl')
    mean_spike_rate_neurons = compute_mean_spike_rate_for_cells(sig_neurons)
    neuron_results = []
    for name, mat in behavior_matrices.items():
        neuron_results_df = run_directional_vector_linear_regression(
            mean_spike_rate_neurons, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True
        )
        neuron_results.append(neuron_results_df)
    all_neuron_results_df = pd.concat(neuron_results, ignore_index=True)
    all_neuron_results_sorted = all_neuron_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    all_neuron_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_single_neurons.pkl')
    print(all_neuron_results_sorted)

    # Directional Linear Regression (OLS) on Significant Windows
    sig_windows = pd.read_pickle(
        f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_windows_pANOVAorGLM_passed.pkl')
    mean_spike_rate_windows = compute_mean_spike_rate_for_windows(sig_windows)
    all_window_results = []
    for name, mat in behavior_matrices.items():
        results_df = run_directional_vector_linear_regression(
            mean_spike_rate_windows, mat, name, monkey_group, monkey_list, subject_monkey_index, use_spikerate=True
        )
        all_window_results.append(results_df)
    all_window_results_df = pd.concat(all_window_results, ignore_index=True)
    all_window_results_sorted = all_window_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    all_window_results_sorted.to_pickle(f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_windows.pkl')
    print(all_window_results_sorted)


if __name__ == "__main__":
    main()