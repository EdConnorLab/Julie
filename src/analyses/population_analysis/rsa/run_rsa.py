# run_rsa.py
"""
Main script for running RSA analysis on neural data.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from rsa_config import RSAConfig
from rsa_core import run_rsa_session, run_rsa_pseudopop
from rsa_plotting import (plot_neural_rdm, plot_model_rdms, plot_rsa_bar,
                          plot_mds, plot_neural_rdm_multi_sort)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def main():
    cfg = RSAConfig(
        region='AMG',                          # 'AMG', 'ER', or 'ALL'
        session=None,                          # None = pseudo-population; 'session_id' = single session
        window=(0.200, 0.600),                 # analysis window (seconds)
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',           # 'correlation' or 'euclidean' or 'cosine' or 'mahalanobis'
        model_factors=['group', 'familiarity', 'sex','age_continuous'],
        partial_out=['familiarity', 'group'],  # regress these out when testing other factors
        exclude_groups=['Stranger Things'],
        normalization='soft',                  # None or 'soft'
        n_permutations=0,                      # set to e.g. 1000 for perm test
        rdm_sort_mode='by_factor',
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

    # Visual responsiveness filter (toggle via cfg.visual_responsiveness_filter)
    df = filter_visually_responsive(df, cfg)

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

        save_dir = f"{cfg.save_dir}/{cfg.region}_pseudopop"
        plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
        plot_model_rdms(result, cfg, save_dir=save_dir)

    else:
        # ─── Single-session mode ───
        result = run_rsa_session(df, cfg.session, info_df, cfg)
        if result is not None:
            _print_comparisons(result, cfg)
            save_dir = f"{cfg.save_dir}/{cfg.region}/{cfg.session}"
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
