import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analyses.cache_utils import BehaviorMatrixCacheManager
from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.behavior_vector_linear_regression import \
    run_directional_vector_linear_regression_cell_level, \
    run_rsa_analysis, run_directional_vector_linear_regression_window_level, \
    expand_window_level_regression_results_with_spike_rates_per_stimulus
from analyses.preprocessing.preprocess_and_select_significant_neurons import PreprocessConfig

from analyses.spike_rate import compute_mean_spike_rate_for_cells_from_source, compute_mean_spike_rate_for_windows_from_source, \
    compute_mean_spike_rate_for_cells_from_source
from analyses.spike_source import SpikeSource, SISortedSpikeSource


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

# ----------------------------
# Behavior loading (cache_utils)
# ----------------------------

def load_behavior_matrices_cached() -> Dict[str, np.ndarray]:
    """
    Loads the 6 matrices using BehaviorMatrixCacheManager so Excel IO is cached.
    Mirrors your previous To/From transpose behavior.
    """
    bm_cache = BehaviorMatrixCacheManager()

    files = {
        "AffiliationTo": ("zombies_social_data/zombies_feature_df_affiliation.xlsx", False),
        "AffiliationFrom": ("zombies_social_data/zombies_feature_df_affiliation.xlsx", True),
        "SubmissionTo": ("zombies_social_data/zombies_feature_df_submission.xlsx", False),
        "SubmissionFrom": ("zombies_social_data/zombies_feature_df_submission.xlsx", True),
        "AgonismTo": ("zombies_social_data/zombies_feature_df_agonism.xlsx", False),
        "AgonismFrom": ("zombies_social_data/zombies_feature_df_agonism.xlsx", True),
    }

    out = {}
    for name, (xlsx_rel, transpose) in files.items():
        df = bm_cache.load_or_cache(
            xlsx_path=xlsx_rel,
            label=f"{name}",  # stable cache key
            transpose=transpose,
            drop_first_col=True,
            force_recompute=False,
        )
        out[name] = df.to_numpy()

    return out

def main():
    analysis_cache_dir = Path("/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/")
    analysis_results_dir = analysis_cache_dir.parent / "analysis_results"
    cfg = PreprocessConfig(
        group_name="Zombies",
        bin_size=0.05,
        analysis_cache_dir=analysis_cache_dir,
        save=True
    )

    source: SpikeSource = SISortedSpikeSource()
    monkey_group = cfg.group_name
    subject_monkey_index = 6
    monkey_list = get_monkeys_by_default_order(monkey_group)


    # Behavior matrices (cached)
    behavior_matrices = load_behavior_matrices_cached()

    # ----------------------------
    # 1) Cell-level directional regression on significant neurons (pKW output)
    # ----------------------------
    '''
    sig_neurons_path = cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_neurons_pKW_passed.pkl"
    sig_neurons = pd.read_pickle(sig_neurons_path)

    mean_spike_rate_neurons = compute_mean_spike_rate_for_cells_from_source(sig_neurons, source=source)

    directional_neuron_results = []
    for behavior_name, mat in behavior_matrices.items():
        neuron_results_df = run_directional_vector_linear_regression_cell_level(
            mean_spike_rate_neurons,
            mat,
            behavior_name,
            monkey_group,
            monkey_list,
            subject_monkey_index,
            use_spikerate=True,
            permutation_test=True,
            n_perm=1000
        )
        directional_neuron_results.append(neuron_results_df)

    directional_all_neuron_results_df = pd.concat(directional_neuron_results, ignore_index=True)
    directional_all_neuron_results_sorted = directional_all_neuron_results_df.sort_values(
        by=["p_value", "R-squared"], ascending=False
    )
    # filtered_df = directional_all_neuron_results_df[
    #     (directional_all_neuron_results_df['p_value'] < 0.05) &
    #     (directional_all_neuron_results_df['R-squared'] > 0.5)
    #     ]
    # mean_spike_rate = compute_mean_spike_rate_for_cells(filtered_df)

    out_cells = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_single_neurons_pKW.pkl"
    directional_all_neuron_results_sorted.to_pickle(out_cells)
    print(directional_all_neuron_results_sorted)
    '''

    # ----------------------------
    # 2) Window-level directional regression on significant windows (optional)
    # ----------------------------
    sig_windows_path = cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl"
    if sig_windows_path.exists():
        sig_windows = pd.read_pickle(sig_windows_path)
        mean_spike_rate_windows = compute_mean_spike_rate_for_windows_from_source(sig_windows, source=source)

        all_window_results = []
        for behavior_name, mat in behavior_matrices.items():
            results_df = run_directional_vector_linear_regression_window_level(
                mean_spike_rate_windows,
                mat,
                behavior_name,
                monkey_group,
                monkey_list,
                subject_monkey_index,
                use_spikerate=True,
                permutation_test=True,
                n_perm=1000
            )
            all_window_results.append(results_df)

        all_window_results_df = pd.concat(all_window_results, ignore_index=True)
        print('all window results..........................')
        print(all_window_results_df)
        out_windows = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW.pkl"
        all_window_results_df.to_pickle(out_windows)

        # expand (your existing function)
        filtered_df = all_window_results_df[
            (all_window_results_df["p_value"] < 0.05) &
            (all_window_results_df["R-squared"] > 0.5) &
            (all_window_results_df.get("p_perm", 0.0) < 0.05)
            ]

        expanded_results = expand_window_level_regression_results_with_spike_rates_per_stimulus(
            mean_spike_rate_windows,
            filtered_df,
            monkey_group,
            monkey_list,
            subject_monkey_index,
            use_spikerate=True
        )
        out_windows_expanded = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW_expanded.pkl"
        print('saving the results')
        expanded_results.to_pickle(out_windows_expanded)

    '''
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
    '''
if __name__ == "__main__":
    main()