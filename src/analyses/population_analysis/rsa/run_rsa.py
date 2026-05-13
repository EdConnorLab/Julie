# run_rsa.py
"""
Main script for running RSA analysis on neural data.

Usage:
  python run_rsa.py

Configure via the RSAConfig at the top of main().
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsa_config import RSAConfig
from rsa_core import run_rsa_session, run_rsa_pseudopop
from rsa_plotting import (plot_model_rdms, plot_rsa_bar,
                          plot_mds, plot_neural_rdm_multi_sort)

# Adjust if needed:
# sys.path.insert(0, '/home/connorlab/Documents/GitHub/Julie/src')
from population_analysis.state_space.data_loading import load_and_filter

MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"


def main():
    cfg = RSAConfig(
        region='AMG',                          # 'AMG', 'ER', or 'ALL'
        session=None,                          # None = all sessions
        window=(0.300, 0.600),                 # analysis window (seconds)
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        neural_metric='correlation',           # 'correlation' or 'euclidean'
        model_factors=['group'],
        exclude_groups=['Stranger Things'],                     # e.g. ['Stranger Things'] to drop
        pca_before_rsa=True,                  # True = PCA denoise before building RDM
        pca_n_components=None,                 # fixed # PCs, or None → use var threshold
        pca_var_threshold=0.50,                # keep PCs explaining this fraction
        n_permutations=0,                      # set to e.g. 1000 for perm test
        pseudo_population=True,               # True = pool neurons across sessions
        save_plots=True,
        save_dir=f'rsa_results',
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)  # uses cfg.data_path, cfg.region, cfg.session, cfg.min_epoch_duration

    # Exclude groups if requested
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    print(f"\n{'='*60}")
    print(f"RSA Analysis")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Factors: {cfg.model_factors}")
    print(f"  Exclude groups: {cfg.exclude_groups or 'none'}")
    if cfg.pca_before_rsa:
        pca_desc = f"{cfg.pca_n_components} PCs" if cfg.pca_n_components else f"{cfg.pca_var_threshold:.0%} var"
        print(f"  PCA before RSA: {pca_desc}")
    else:
        print(f"  PCA before RSA: off")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"  Mode: {'pseudo-population' if cfg.pseudo_population else 'per-session'}")
    print(f"{'='*60}\n")

    if cfg.pseudo_population:
        # ─── Pseudo-population mode ───
        result = run_rsa_pseudopop(df, info_df, cfg)
        _print_comparisons(result)

        save_dir = f"{cfg.save_dir}/{cfg.region}_pseudopop"
        plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
        plot_model_rdms(result, cfg, save_dir=save_dir)
        plot_rsa_bar([result], cfg, save_dir=save_dir)
        for factor in cfg.model_factors:
            if factor in ('group', 'sex', 'age_bin', 'familiarity'):
                plot_mds(result, cfg, color_by=factor, save_dir=save_dir)

    else:
        # ─── Per-session mode ───
        sessions = sorted(df['session'].unique())
        results = []

        for sess in sessions:
            print(f"\n--- Session: {sess} ---")
            result = run_rsa_session(df, sess, info_df, cfg)
            if result is None:
                continue
            _print_comparisons(result)
            results.append(result)

            save_dir = f"{cfg.save_dir}/{cfg.region}/{sess}"
            plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
            plot_model_rdms(result, cfg, save_dir=save_dir)
            for factor in cfg.model_factors:
                if factor in ('group', 'sex', 'age_bin', 'familiarity'):
                    plot_mds(result, cfg, color_by=factor, save_dir=save_dir)

        if results:
            # Summary bar chart across sessions
            summary_dir = f"{cfg.save_dir}/{cfg.region}/summary"
            plot_rsa_bar(results, cfg, save_dir=summary_dir)

            # Print summary table
            _print_summary_table(results, cfg)

    plt.show()


def _print_comparisons(result):
    pca = result.get('pca_info')
    pca_str = f", PCA → {pca['n_components']} PCs ({pca['var_cumulative'][-1]:.1%})" if pca else ""
    print(f"  {result['n_identities']} identities, {result['n_neurons']} neurons{pca_str}")
    for factor, comp in result['comparisons'].items():
        p_str = f"p={comp['p_val']:.4f}" if comp['p_val'] is not None else "no perm test"
        print(f"  {factor:20s}  ρ = {comp['rho']:+.4f}  ({p_str})")


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
