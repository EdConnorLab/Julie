# run_rsa_social_pkl.py
"""
Social-behavior RSA on a hand-picked set of (NeuronID, window) pairs from
a pkl (e.g. the si_sorted_*_significant_windows_pKW_passed.pkl files).

Same neuron-selection logic as run_rsa_pkl.py.  Once the pseudo-population
result is built, this script feeds it through the existing social-RSA
pipeline (rsa_social + run_rsa_social helpers).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from analyses.population_analysis.rsa.core.visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from analyses.population_analysis.rsa.variants.pkl.rsa_pkl_config import PklSocialRSAConfig
from analyses.population_analysis.rsa.variants.pkl.rsa_pkl_core import load_pkl_windows, run_rsa_pkl_pseudopop
from analyses.population_analysis.rsa.social.rsa_social import (
    load_all_interaction_matrices,
    build_neural_similarity_matrix,
    build_rank_distance_matrix,
)
from analyses.population_analysis.rsa.core.rsa_plotting import plot_neural_rdm

# Re-use the helpers defined in run_rsa_social.py instead of duplicating them.
# (If your project layout doesn't put run_rsa_social.py on the import path,
# inline the two functions below or copy run_rsa_social into a module.)
from analyses.population_analysis.rsa.social.run_rsa_social import (
    BEHAVIOR_FILES,
    plot_social_matrix,
    run_dissimilarity_condition,
    _p_to_stars,
)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def main():
    cfg = PklSocialRSAConfig(
        # ── Pkl options ──
        pkl_path='/home/connorlab/Documents/JulieData/Cortana/analysis_cache/'
                 # 'si_sorted_Zombies_significant_windows_pKW_passed.pkl',
                    'si_sorted_Zombies_all_windows_pKW_tested.pkl',
        pkl_window_mode='common',          # 'per_neuron' | 'common'
        pkl_filter_significant=True,          # True → keep only p < pkl_p_threshold
        pkl_p_threshold=0.05,

        # ── Social-RSA options ──
        region='ALL',
        session=None,                          # always pseudo-pop in pkl mode
        window=(0.300, 0.600),                 # only used in pkl_window_mode='common'
        min_epoch_duration=2.2,                # must cover the longest pkl window
        min_reps_per_monkey=7,
        neural_metric='correlation',
        exclude_groups=['Stranger Things', 'Best Frans'],
        normalization=None,
        transform_social_behavior='log',        # None | 'rank' | 'log'
        n_permutations=2000,
        between_group_permutations=2000,
        n_bootstrap=2000,
        partial_out_rank=False,
        exclude_identities=[],

        # ── Neuron filters (off by default — the pkl is the filter) ──
        peak_latency_filter=False,
        neuron_id_filter_pkl=None,
        visual_responsiveness_filter=False,

        save_plots=True,
        save_dir='rsa_social_results_from_pkl',
    )
    cfg.validate()

    if cfg.session is not None:
        print("  ⚠  cfg.session is set but pkl mode always pools across sessions; "
              "ignoring cfg.session.")

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)

    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    df = filter_visually_responsive(df, cfg)
    df = apply_neuron_filters(df, cfg)

    print("\n--- Loading pkl windows ---")
    neuron_windows, pkl_df = load_pkl_windows(cfg)

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    print(f"\n{'='*60}")
    print(f"Social Behavior RSA (pkl-driven)")
    print(f"  Pkl path: {cfg.pkl_path}")
    print(f"  Window mode: {cfg.pkl_window_mode}")
    print(f"  Filter significant: {cfg.pkl_filter_significant} "
          f"(threshold p < {cfg.pkl_p_threshold})")
    print(f"  Region: {cfg.region}")
    if cfg.pkl_window_mode == 'common':
        print(f"  Common window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Normalization: {cfg.normalization}")
    print(f"  Neural metric: {cfg.neural_metric}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"  Between-group Δρ permutations: {cfg.between_group_permutations}")
    print(f"  Bootstrap CI: {cfg.n_bootstrap}")
    print(f"  Partial out rank: {cfg.partial_out_rank}")
    print(f"  Social behavior transform: {cfg.transform_social_behavior or 'raw'}")
    print(f"{'='*60}\n")

    # ── Pooled pseudo-population RSA ──
    result = run_rsa_pkl_pseudopop(df, info_df, cfg, neuron_windows)
    print(f"  Pool: {result['n_neurons']} neurons "
          f"(pkl requested {result['pkl_n_requested']}, used {result['pkl_n_used']}); "
          f"{result['n_identities']} identities")

    identities = result['identities']
    neural_rdm = result['neural_rdm']
    rate_matrix = result['rate_matrix']
    session_label = result['session']

    # ── Sensitivity: exclude specific identities (same logic as run_rsa_social) ──
    if cfg.exclude_identities:
        keep_mask = [i for i, mid in enumerate(identities)
                     if mid not in cfg.exclude_identities]
        removed = [mid for mid in identities if mid in cfg.exclude_identities]
        identities  = [identities[i] for i in keep_mask]
        neural_rdm  = neural_rdm[np.ix_(keep_mask, keep_mask)]
        rate_matrix = rate_matrix[keep_mask, :]
        print(f"\n  [Sensitivity] Excluded: {removed}  →  "
              f"{len(identities)} identities remaining")

    neural_sim = build_neural_similarity_matrix(rate_matrix)

    rank_confound = None
    if cfg.partial_out_rank:
        rank_confound = build_rank_distance_matrix(identities, info_df)
        print(f"  Partial out rank: ON (|rank_i - rank_j| as confound)")

    # ── Build save_dir (window-mode and sig-filter encoded into the path) ──
    sig_tag = 'sig' if cfg.pkl_filter_significant else 'all'
    tx_tag  = cfg.transform_social_behavior or 'raw'
    if cfg.pkl_window_mode == 'common':
        win_start = int(cfg.window[0] * 1000)
        win_end   = int(cfg.window[1] * 1000)
        win_tag   = f"{win_start}_{win_end}_common"
    else:
        win_tag = "per_neuron"
    save_dir_base = (f"{cfg.save_dir}/{cfg.region}_{win_tag}_{tx_tag}_{sig_tag}/filter")

    print(f"\n{'='*60}")
    print(f"  Identities ({len(identities)}): {identities}")
    print(f"  Saving to: {save_dir_base}/")
    print(f"{'='*60}")

    # ── Neural reference plots ──
    if cfg.save_plots:
        plot_neural_rdm(result, cfg, save_dir=save_dir_base)
        plot_social_matrix(
            neural_sim, identities, info_df,
            title=(f"Neural Similarity (Pearson r) | {cfg.region} "
                   f"[{win_tag}, {sig_tag}]"),
            group_colors=cfg.group_colors,
            cbar_label='Pearson r',
            save_path=f"{save_dir_base}/neural_similarity.png")

    # ── Dissimilarity RSA: neural 1-r vs social profile RDMs ──
    conditions = [
        (False, "asymmetric",  "asym_"),
        (True,  "symmetrized", "sym_"),
    ]

    dissim_results = {}
    for symmetrize, label, prefix in conditions:
        group_comp, between_comp, _ = run_dissimilarity_condition(
            neural_rdm, identities, interactions, info_df, cfg,
            symmetrize=symmetrize,
            condition_label=label, save_dir=save_dir_base, prefix=prefix,
            confound_matrix=rank_confound)
        dissim_results[label] = (group_comp, between_comp)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY (per-group): {session_label}")
    print(f"{'='*70}")

    cond_labels = [c[1] for c in conditions]
    _print_per_group_summary(
        "DISSIMILARITY (neural 1-r vs social profile RDMs)",
        dissim_results, cond_labels)

    print(f"\n{'='*70}")
    plt.show()


def _print_per_group_summary(mode_label, results_dict, cond_labels):
    """Per-group RSA summary table (same format as run_rsa_social._print_per_group_summary)."""
    print(f"\n  {mode_label}:")

    first_group_comp = list(results_dict.values())[0][0]
    mat_names = list(first_group_comp.keys())
    all_groups = set()
    for gc in first_group_comp.values():
        all_groups.update(gc.keys())
    all_groups = sorted(all_groups)

    has_ci = False
    for label in cond_labels:
        gc = results_dict[label][0]
        for mat_data in gc.values():
            for entry in mat_data.values():
                if not np.isnan(entry.get('ci_low', np.nan)):
                    has_ci = True
                    break
            if has_ci:
                break
        if has_ci:
            break
    col_w = 34 if has_ci else 20

    for group in all_groups:
        print(f"\n    {group}:")
        header = f"    {'Matrix':<28s}" + "".join(f"{c:>{col_w}s}" for c in cond_labels)
        print(header)
        print(f"    {'-'*28}" + "-" * col_w * len(cond_labels))

        for mat_name in mat_names:
            row = f"    {mat_name:<28s}"
            for label in cond_labels:
                gc = results_dict[label][0]
                entry = gc.get(mat_name, {}).get(group, {})
                rho = entry.get('rho', np.nan)
                p   = entry.get('p_val')
                ci_lo = entry.get('ci_low', np.nan)
                ci_hi = entry.get('ci_high', np.nan)
                if np.isnan(rho):
                    row += f"{'--':>{col_w}s}"
                else:
                    star = _p_to_stars(p)
                    ci_str = f" [{ci_lo:+.2f},{ci_hi:+.2f}]" if not np.isnan(ci_lo) else ""
                    row += f"  {rho:>+.4f} {star:<5s}{ci_str}"
            print(row)

    first_between = list(results_dict.values())[0][1]
    if first_between is not None:
        print(f"\n    Between-group Δρ:")
        all_pairs = set()
        for bc in first_between.values():
            all_pairs.update(bc.keys())
        all_pairs = sorted(all_pairs)
        pair_labels = [f"{a}−{b}" for a, b in all_pairs]
        for pi, (pair, plabel) in enumerate(zip(all_pairs, pair_labels)):
            print(f"\n    {plabel}:")
            header = f"    {'Matrix':<28s}" + "".join(f"{c:>20s}" for c in cond_labels)
            print(header)
            print(f"    {'-'*28}" + "-" * 20 * len(cond_labels))
            for mat_name in mat_names:
                row = f"    {mat_name:<28s}"
                for label in cond_labels:
                    bc = results_dict[label][1]
                    entry = bc.get(mat_name, {}).get(pair, {})
                    d = entry.get('delta_rho', np.nan)
                    p = entry.get('p_val')
                    if np.isnan(d):
                        row += f"{'--':>20s}"
                    else:
                        star = _p_to_stars(p)
                        row += f"  {d:>+.4f} {star:<12s}"
                print(row)


if __name__ == '__main__':
    main()
