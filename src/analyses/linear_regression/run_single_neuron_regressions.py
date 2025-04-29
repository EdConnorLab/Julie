import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.directional_behavioral_vector_analysis import run_directional_vector_linear_regression
from analyses.spike_rate import compute_mean_spike_rate_for_cells


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
    monkey_group_name = "Zombies"
    monkey_list = get_monkeys_by_default_order(monkey_group_name)
    base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'
    behavior_files = {
        "AffliationTo": "zombies_feature_df_affiliation.xlsx",
        "AffliationFrom": "zombies_feature_df_affiliation.xlsx",
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
    sig_neurons = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl')
    mean_spike_rate = compute_mean_spike_rate_for_cells(sig_neurons)
    subject_monkey_index = 6
    all_results = []
    for name, mat in behavior_matrices.items():
        results_df = run_directional_vector_linear_regression(
            mean_spike_rate, mat, name, monkey_group_name, monkey_list, subject_monkey_index, use_spikerate=True
        )
        all_results.append(results_df)
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_sorted = all_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    significant_results = all_results_sorted[all_results_sorted['p_value'] < 0.05]
    significant_results.to_excel('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_neurons_only_significant.xlsx')
    all_results_sorted.to_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_single_neurons.pkl')
    # print(all_results_sorted)
    # significant_results = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_single_neurons.pkl')
    # significant_results = significant_results[significant_results['p_value'] < 0.05]



if __name__ == "__main__":
    main()