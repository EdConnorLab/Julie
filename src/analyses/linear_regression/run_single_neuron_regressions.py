from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_access.cache_utils import BehaviorMatrixCacheManager
from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.behavior_vector_linear_regression import \
    run_directional_vector_linear_regression_cell_level, \
    run_directional_vector_linear_regression_window_level, \
    flatten_regression_results
from analyses.preprocessing.preprocess_and_select_significant_neurons import PreprocessConfig

from analyses.spike_rate import compute_mean_spike_rate_for_windows_from_source, compute_mean_spike_rate_for_cells_from_source
from data_access.spike_source import SpikeSource, SISortedSpikeSource


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

    # # ----------------------------
    # # 1) Cell-level directional regression on significant neurons (pKW output)
    # # ----------------------------
    #
    # sig_neurons_path = cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_neurons_pKW_passed.pkl"
    # sig_neurons = pd.read_pickle(sig_neurons_path)
    #
    # mean_spike_rate_neurons = compute_mean_spike_rate_for_cells_from_source(sig_neurons, source=source)
    #
    # directional_neuron_results = []
    # for behavior_name, mat in behavior_matrices.items():
    #     neuron_results_df = run_directional_vector_linear_regression_cell_level(
    #         mean_spike_rate_neurons,
    #         mat,
    #         behavior_name,
    #         monkey_group,
    #         monkey_list,
    #         subject_monkey_index,
    #         use_spikerate=True,
    #         permutation_test=True,
    #         n_perm=10000
    #     )
    #     directional_neuron_results.append(neuron_results_df)
    #
    # directional_all_neuron_results_df = pd.concat(directional_neuron_results, ignore_index=True)
    #
    #
    # out_cells = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_neurons_pKW.pkl"
    # directional_all_neuron_results_df.to_pickle(out_cells)
    # print(directional_all_neuron_results_df)
    #
    # flat_cell_results = flatten_regression_results(directional_all_neuron_results_df)
    # flat_cell_results_filename = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_neurons_pKW_flat.pkl"
    # print('saving the flat cell results')
    # flat_cell_results.to_pickle(flat_cell_results_filename)

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
                n_perm=10000,
                random_state=42
            )
            all_window_results.append(results_df)

        all_window_results_df = pd.concat(all_window_results, ignore_index=True)
        print('all window results..........................')
        print(all_window_results_df)
        out_windows = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW.pkl"
        all_window_results_df.to_pickle(out_windows)

        flat_window_results = flatten_regression_results(all_window_results_df)
        flat_window_filename = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW_flat.pkl"
        print('saving the flat window results')
        flat_window_results.to_pickle(flat_window_filename)

if __name__ == "__main__":
    main()
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
