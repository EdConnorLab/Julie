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
from analyses.linear_regression.behavior_vector_linear_regression_extensions import (
    winsorize_behavior_matrix,
    run_directional_vector_linear_regression_cell_level_with_loo,
    run_directional_vector_linear_regression_window_level_with_loo,
    summarize_loo_results,
)
from run_meta_analysis import run_meta_analysis_from_dataframe
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
    analysis_cache_dir = Path("/Cortana/analysis_cache/")
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

    # ── NEW: build winsorized matrices once ───────────────────────────────────
    # Each behavior type is winsorized independently (pooling within its own
    # matrix) so different behavior scales don't interfere with each other.
    winsorized_behavior_matrices = {
        name: winsorize_behavior_matrix(mat, percentile=95)
        for name, mat in behavior_matrices.items()
    }
    # ─────────────────────────────────────────────────────────────────────────

    # =========================================================================
    # SECTION A — LOO regression  (cells)
    # =========================================================================
    sig_neurons_path = (
            cfg.analysis_cache_dir /
            f"{source.name}_{cfg.group_name}_significant_neurons_pKW_passed.pkl"
    )
    if sig_neurons_path.exists():
        sig_neurons = pd.read_pickle(sig_neurons_path)
        mean_spike_rate_neurons = compute_mean_spike_rate_for_cells_from_source(
            sig_neurons, source=source
        )

        loo_neuron_results = []
        for behavior_name, mat in behavior_matrices.items():
            res = run_directional_vector_linear_regression_cell_level_with_loo(
                mean_spike_rate_neurons,
                mat,
                behavior_name,
                monkey_group,
                monkey_list,
                subject_monkey_index,
                use_spikerate=True,
                permutation_test=False,  # set True if you want p_perm (slow)
                n_perm=10000,
            )
            loo_neuron_results.append(res)

        loo_neuron_df = pd.concat(loo_neuron_results, ignore_index=True)

        # Save raw per-monkey LOO results
        out_loo_raw = analysis_results_dir / f"{source.name}_{monkey_group}_loo_neurons_raw.pkl"
        loo_neuron_df.to_pickle(out_loo_raw)
        print(f"LOO raw (neurons) → {out_loo_raw}")

        # Aggregate to one row per neuron × behavior → upload this to dashboard
        loo_summary = summarize_loo_results(loo_neuron_df)
        out_loo_summary = analysis_results_dir / f"{source.name}_{monkey_group}_loo_neurons_summary.csv"
        loo_summary.to_csv(out_loo_summary, index=False)
        print(f"LOO summary (neurons) → {out_loo_summary}")
        print(loo_summary.head(10))

    # =========================================================================
    # SECTION A — LOO regression  (windows)
    # =========================================================================
    sig_windows_path = (
            cfg.analysis_cache_dir /
            f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl"
    )
    if sig_windows_path.exists():
        sig_windows = pd.read_pickle(sig_windows_path)
        mean_spike_rate_windows = compute_mean_spike_rate_for_windows_from_source(
            sig_windows, source=source
        )

        loo_window_results = []
        for behavior_name, mat in behavior_matrices.items():
            res = run_directional_vector_linear_regression_window_level_with_loo(
                mean_spike_rate_windows,
                mat,
                behavior_name,
                monkey_group,
                monkey_list,
                subject_monkey_index,
                use_spikerate=True,
                permutation_test=False,
                n_perm=10000,
                random_state=42,
            )
            loo_window_results.append(res)

        loo_window_df = pd.concat(loo_window_results, ignore_index=True)

        out_loo_raw_w = analysis_results_dir / f"{source.name}_{monkey_group}_loo_windows_raw.pkl"
        loo_window_df.to_pickle(out_loo_raw_w)
        print(f"LOO raw (windows) → {out_loo_raw_w}")

        loo_summary_w = summarize_loo_results(loo_window_df)
        out_loo_summary_w = analysis_results_dir / f"{source.name}_{monkey_group}_loo_windows_summary.csv"
        loo_summary_w.to_csv(out_loo_summary_w, index=False)
        print(f"LOO summary (windows) → {out_loo_summary_w}")

    # =========================================================================
    # SECTION B — Winsorized regression  (cells)
    # =========================================================================
    if sig_neurons_path.exists():
        winsorized_neuron_results = []
        for behavior_name, mat in winsorized_behavior_matrices.items():
            res = run_directional_vector_linear_regression_cell_level(
                mean_spike_rate_neurons,
                mat,
                behavior_name,
                monkey_group,
                monkey_list,
                subject_monkey_index,
                use_spikerate=True,
                permutation_test=False,
                n_perm=10000,
            )
            winsorized_neuron_results.append(res)

        winsorized_neuron_df = pd.concat(winsorized_neuron_results, ignore_index=True)

        # Save raw (per-monkey) winsorized results
        out_wins_raw = (analysis_results_dir /
                        f"{source.name}_{monkey_group}_winsorized_neurons_raw.pkl")
        winsorized_neuron_df.to_pickle(out_wins_raw)
        print(f"Winsorized raw (neurons) → {out_wins_raw}")

        # ── Run meta-analysis on winsorized results immediately ───────────────
        # This gives you a single pooled effect per neuron × behavior
        # with heterogeneity stats, comparable to the standard meta-analysis.
        meta_winsorized = run_meta_analysis_from_dataframe(winsorized_neuron_df)
        out_meta_wins = (analysis_results_dir /
                         f"{source.name}_{monkey_group}_meta_winsorized_neurons.csv")
        meta_winsorized.to_csv(out_meta_wins, index=False)
        print(f"Meta-analysis (winsorized neurons) → {out_meta_wins}")

    # =========================================================================
    # SECTION B — Winsorized regression  (windows)
    # =========================================================================
    if sig_windows_path.exists():
        winsorized_window_results = []
        for behavior_name, mat in winsorized_behavior_matrices.items():
            res = run_directional_vector_linear_regression_window_level(
                mean_spike_rate_windows,
                mat,
                behavior_name,
                monkey_group,
                monkey_list,
                subject_monkey_index,
                use_spikerate=True,
                permutation_test=False,
                n_perm=10000,
                random_state=42,
            )
            winsorized_window_results.append(res)

        winsorized_window_df = pd.concat(winsorized_window_results, ignore_index=True)

        out_wins_raw_w = (analysis_results_dir /
                          f"{source.name}_{monkey_group}_winsorized_windows_raw.pkl")
        winsorized_window_df.to_pickle(out_wins_raw_w)
        print(f"Winsorized raw (windows) → {out_wins_raw_w}")

        meta_winsorized_w = run_meta_analysis_from_dataframe(winsorized_window_df)
        out_meta_wins_w = (analysis_results_dir /
                           f"{source.name}_{monkey_group}_meta_winsorized_windows.csv")
        meta_winsorized_w.to_csv(out_meta_wins_w, index=False)
        print(f"Meta-analysis (winsorized windows) → {out_meta_wins_w}")
    # Original Code
    # ----------------------------
    # 1) Cell-level directional regression on significant neurons (pKW output)
    # ----------------------------
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
    # sig_windows_path = cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl"
    # if sig_windows_path.exists():
    #     sig_windows = pd.read_pickle(sig_windows_path)
    #     mean_spike_rate_windows = compute_mean_spike_rate_for_windows_from_source(sig_windows, source=source)
    #
    #     all_window_results = []
    #     for behavior_name, mat in behavior_matrices.items():
    #         results_df = run_directional_vector_linear_regression_window_level(
    #             mean_spike_rate_windows,
    #             mat,
    #             behavior_name,
    #             monkey_group,
    #             monkey_list,
    #             subject_monkey_index,
    #             use_spikerate=True,
    #             permutation_test=True,
    #             n_perm=10000,
    #             random_state=42
    #         )
    #         all_window_results.append(results_df)
    #
    #     all_window_results_df = pd.concat(all_window_results, ignore_index=True)
    #     print('all window results..........................')
    #     print(all_window_results_df)
    #     out_windows = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW.pkl"
    #     all_window_results_df.to_pickle(out_windows)
    #
    #     flat_window_results = flatten_regression_results(all_window_results_df)
    #     flat_window_filename = analysis_results_dir / f"{source.name}_{monkey_group}_dir_ols_on_windows_pKW_flat.pkl"
    #     print('saving the flat window results')
    #     flat_window_results.to_pickle(flat_window_filename)

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
