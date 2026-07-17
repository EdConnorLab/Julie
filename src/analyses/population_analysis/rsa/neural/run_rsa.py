# run_rsa.py
"""
Main script for running RSA analysis on neural data.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from analyses.population_analysis.rsa.core.visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from analyses.population_analysis.rsa.core.rsa_config import RSAConfig
from analyses.population_analysis.rsa.core.rsa_core import (run_rsa_session, run_rsa_pseudopop,
                      variance_quartile_diagnostic,
                      random_subset_diagnostic, compute_containment_metric)
from analyses.population_analysis.rsa.core.rsa_plotting import (plot_neural_rdm, plot_model_rdms, plot_rsa_bar,
                          plot_mds, plot_neural_rdm_multi_sort,
                          plot_variance_quartile_rdms,
                          plot_random_subset_rdms, plot_containment_metric)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def main():
    cfg = RSAConfig(
        region='AMG',                          # 'AMG', 'ER', or 'ALL'
        session=None,                          # None = pseudo-population; 'session_id' = single session
        window=(0.300, 0.500),                 # analysis window (seconds)
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='euclidean',           # 'correlation' or 'euclidean' or 'cosine' or 'mahalanobis'
        model_factors=['group', 'familiarity'], # 'sex','age_continuous', 'rank'
        partial_out=  [],  # ['familiarity', 'group'],  # regress these out when testing other factors
        exclude_groups=[],    # ['Stranger Things'],
        exclude_identities=[],
        normalization='soft',                  # None or 'soft' or 'zscore'
        n_permutations=0,                      # set to e.g. 1000 for perm test
        rdm_sort_mode='by_factor',
        # ── Neuron filters (off by default) ──
        visual_responsiveness_filter=False,
        peak_latency_filter=False,
        peak_latency_range=(0.200, 0.500),
        peak_latency_search_window=(0.0, 1.00),
        # neuron_id_filter_pkl='/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pKW_passed.pkl',
        save_plots=True,
        save_dir=f'rsa_results_for_progress_report_Jun2026_euclidean',
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)  # uses cfg.data_path, cfg.region, cfg.session, cfg.min_epoch_duration

    # Exclude groups if requested
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")
    if cfg.exclude_identities:
        df = df[~df['MonkeyName'].isin(cfg.exclude_identities)].reset_index(drop=True)
        print(f"Excluded identities {cfg.exclude_identities}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    # Visual responsiveness filter (toggle via cfg.visual_responsiveness_filter)
    df = filter_visually_responsive(df, cfg)

    # Optional neuron-level filters: pkl NeuronID whitelist and/or peak-latency filter
    df = apply_neuron_filters(df, cfg)

    print(f"\n{'='*60}")
    print(f"RSA Analysis")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Normalization: {cfg.normalization}")
    print(f"  Neural Metric: {cfg.neural_metric}")
    print(f"  Factors: {cfg.model_factors}")
    print(f"  Partial out: {cfg.partial_out or 'none'}")
    print(f"  Exclude groups: {cfg.exclude_groups or 'none'}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"  Mode: {'pseudo-population' if cfg.session is None else f'single session ({cfg.session})'}")
    print(f"{'='*60}\n")

    if cfg.session is None:
        # ─── Pseudo-population mode (all sessions) ───
        result = run_rsa_pseudopop(df, info_df, cfg)
        _print_comparisons(result, cfg)

        # ─── Variance-quartile diagnostic ───
        diag = variance_quartile_diagnostic(
            result['raw_rate_matrix'], info_df, result['identities'], cfg)

        save_dir = f"{cfg.save_dir}/{cfg.region}_{cfg.normalization}_pseudopop/"
        plot_variance_quartile_rdms(diag, cfg, save_dir=save_dir)
        cont = compute_containment_metric(diag, info_df)
        plot_containment_metric(cont, cfg, save_dir=save_dir)
        rand = random_subset_diagnostic(
            result['raw_rate_matrix'], info_df, result['identities'], cfg,
            fraction=0.25, n_draws=4)
        plot_random_subset_rdms(rand, cfg, reference_entry=diag[0],
                                save_dir=save_dir)
        # ─── Overlap check: how many top-25% neurons did each random draw catch? ───
        top_set = set(diag[0]['neuron_indices'])  # diag[0] = top-25% by variance
        k = diag[0]['n_neurons']
        n_total = result['raw_rate_matrix'].shape[1]
        expected_chance = k * k / n_total
        print(f"\nOverlap with top-25% variance set (n={k}/{n_total}, "
              f"chance ≈ {expected_chance:.1f}):")
        for r in rand:
            overlap = len(set(r['neuron_indices']) & top_set)
            enrichment = overlap / expected_chance
            print(f"  Random draw {r['draw_index'] + 1}: {overlap:2d} neurons  "
                  f"({enrichment:.2f}× chance)")
        # ─── Selectivity-based overlap (identity-agnostic) ───
        # "Spread" = how much each neuron's max-firing identity stands out from
        # its average across identities, normalized by its variability.
        # Captures sparse tuning regardless of which identity each neuron prefers.
        rate_mat = result['raw_rate_matrix']  # (n_ids, n_neurons)
        spread = (rate_mat.max(axis=0) - rate_mat.mean(axis=0))

        K = 10
        top_selective = set(np.argsort(spread)[::-1][:K].tolist())

        chance_K = K * diag[0]['n_neurons'] / rate_mat.shape[1]
        print(f"\nOverlap with top-{K} most sparsely-tuned neurons "
              f"(chance ≈ {chance_K:.2f}):")
        for r in rand:
            overlap = len(set(r['neuron_indices'].tolist()) & top_selective)
            enrichment = overlap / chance_K if chance_K > 0 else float('nan')
            print(f"  Random draw {r['draw_index'] + 1}: {overlap}/{K}  "
                  f"({enrichment:.2f}× chance)")

        # How much do "high variance" and "sparsely tuned" overlap?
        top_var = set(diag[0]['neuron_indices'].tolist())
        print(f"\nOf the top-{K} sparsely-tuned neurons, "
              f"{len(top_selective & top_var)}/{K} are in the top-25% by variance.")



        plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
        plot_model_rdms(result, cfg, save_dir=save_dir)



    else:
        # ─── Single-session mode ───
        result = run_rsa_session(df, cfg.session, info_df, cfg)
        if result is not None:
            _print_comparisons(result, cfg)

            # ─── Variance-quartile diagnostic ───
            diag = variance_quartile_diagnostic(
                result['raw_rate_matrix'], info_df, result['identities'], cfg)

            save_dir = f"{cfg.save_dir}/{cfg.region}/{cfg.session}"
            plot_variance_quartile_rdms(diag, cfg, save_dir=save_dir)
            plot_neural_rdm(result, cfg, save_dir=save_dir)
            plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
            plot_model_rdms(result, cfg, save_dir=save_dir)
            for factor in cfg.model_factors:
                if factor in ('group', 'sex', 'age_bin', 'familiarity'):
                    plot_mds(result, cfg, color_by=factor, save_dir=save_dir)

    plt.show()


def _print_comparisons(result, cfg=None):
    print(f"  {result['n_identities']} identities, {result['n_neurons']} neurons")
    partial_out = getattr(cfg, 'partial_out', []) if cfg else []
    for factor, comp in result['comparisons'].items():
        p_str = f"p={comp['p_val']:.4f}" if comp['p_val'] is not None else "no perm test"
        confounds = [p for p in partial_out if p != factor]
        tag = f"  | partial out: {', '.join(confounds)}" if confounds else ""
        print(f"  {factor:20s}  ρ = {comp['rho']:+.4f}  ({p_str}){tag}")


def _print_summary_table(results, cfg):
    """Print a summary table of RSA rho values across sessions."""
    print(f"\n{'='*70}")
    print("RSA SUMMARY (Spearman ρ per session)")
    print(f"{'='*70}")

    factors = cfg.model_factors
    header = f"{'Session':>30s}" + "".join(f"{f:>14s}" for f in factors) + f"{'n_ids':>8s}{'n_neur':>8s}"
    print(header)
    print("-" * len(header))

    rho_matrix = []
    for r in results:
        rhos = [r['comparisons'][f]['rho'] for f in factors]
        rho_matrix.append(rhos)
        row = f"{r['session']:>30s}" + "".join(f"{rho:>14.4f}" for rho in rhos)
        row += f"{r['n_identities']:>8d}{r['n_neurons']:>8d}"
        print(row)

    rho_matrix = np.array(rho_matrix)
    means = np.nanmean(rho_matrix, axis=0)
    sems = np.nanstd(rho_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(rho_matrix), axis=0))
    print("-" * len(header))
    mean_row = f"{'MEAN':>30s}" + "".join(f"{m:>14.4f}" for m in means)
    sem_row = f"{'SEM':>30s}" + "".join(f"{s:>14.4f}" for s in sems)
    print(mean_row)
    print(sem_row)
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
