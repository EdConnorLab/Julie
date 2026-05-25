# run_rsa_pkl.py
"""
RSA analysis on a hand-picked set of (NeuronID, window) pairs from a pkl
(e.g. the si_sorted_*_significant_windows_pKW_passed.pkl files).

Two modes, controlled by cfg.pkl_window_mode:
  - 'per_neuron' : each neuron uses its own window from the pkl
  - 'common'     : all selected neurons use cfg.window (pkl is just a whitelist)

The p-value filter (cfg.pkl_filter_significant) is independent of the
window mode: set True to keep only rows with p < cfg.pkl_p_threshold.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from rsa_pkl_config import PklRSAConfig
from rsa_pkl_core import load_pkl_windows, run_rsa_pkl_pseudopop
from rsa_plotting import (plot_neural_rdm, plot_model_rdms, plot_rsa_bar,
                          plot_mds, plot_neural_rdm_multi_sort)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def main():
    cfg = PklRSAConfig(
        # ── Pkl options ──
        pkl_path='/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/'
                 'si_sorted_Zombies_significant_windows_pKW_passed.pkl',
        pkl_window_mode='per_neuron',          # 'per_neuron' | 'common'
        pkl_filter_significant=False,          # True → keep only p < pkl_p_threshold
        pkl_p_threshold=0.05,

        # ── Standard RSA options ──
        region='ALL',
        session=None,                          # pkl pool is always pseudo-pop
        window=(0.100, 0.450),                 # only used in pkl_window_mode='common'
        min_epoch_duration=2.2,                # must cover the longest pkl window
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=['group', 'familiarity', 'sex', 'age_continuous'],
        partial_out=[],
        exclude_groups=['Stranger Things'],
        normalization='soft',
        n_permutations=0,
        rdm_sort_mode='by_factor',

        # ── Neuron filters (off by default — the pkl is the filter) ──
        peak_latency_filter=False,
        # NOTE: do NOT also set cfg.neuron_id_filter_pkl unless you want a
        # second (intersecting) whitelist on top of the pkl driving this run.
        neuron_id_filter_pkl=None,
        visual_responsiveness_filter=False,

        save_plots=True,
        save_dir='rsa_results_from_pkl',
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

    # Standard upstream filters (most users will leave these off here)
    df = filter_visually_responsive(df, cfg)
    df = apply_neuron_filters(df, cfg)

    # ── Load the pkl ──
    print("\n--- Loading pkl windows ---")
    neuron_windows, pkl_df = load_pkl_windows(cfg)

    print(f"\n{'='*60}")
    print(f"RSA (pkl-driven)")
    print(f"  Pkl path: {cfg.pkl_path}")
    print(f"  Window mode: {cfg.pkl_window_mode}")
    print(f"  Filter significant: {cfg.pkl_filter_significant} "
          f"(threshold p < {cfg.pkl_p_threshold})")
    print(f"  Region: {cfg.region}")
    if cfg.pkl_window_mode == 'common':
        print(f"  Common window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Normalization: {cfg.normalization}")
    print(f"  Neural Metric: {cfg.neural_metric}")
    print(f"  Factors: {cfg.model_factors}")
    print(f"  Partial out: {cfg.partial_out or 'none'}")
    print(f"  Exclude groups: {cfg.exclude_groups or 'none'}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"{'='*60}\n")

    # ── Pooled RSA ──
    result = run_rsa_pkl_pseudopop(df, info_df, cfg, neuron_windows)
    _print_comparisons(result, cfg)

    # ── Plots ──
    sig_tag = 'sig' if cfg.pkl_filter_significant else 'all'
    save_dir = (f"{cfg.save_dir}/{cfg.region}_{cfg.normalization}_"
                f"{cfg.pkl_window_mode}_{sig_tag}/")
    if cfg.save_plots:
        plot_neural_rdm_multi_sort(result, cfg, save_dir=save_dir)
        plot_model_rdms(result, cfg, save_dir=save_dir)

    plt.show()


def _print_comparisons(result, cfg=None):
    print(f"  {result['n_identities']} identities, {result['n_neurons']} neurons "
          f"(pkl requested {result['pkl_n_requested']}, used {result['pkl_n_used']})")
    partial_out = getattr(cfg, 'partial_out', []) if cfg else []
    for factor, comp in result['comparisons'].items():
        p_str = f"p={comp['p_val']:.4f}" if comp['p_val'] is not None else "no perm test"
        confounds = [p for p in partial_out if p != factor]
        tag = f"  | partial out: {', '.join(confounds)}" if confounds else ""
        print(f"  {factor:20s}  ρ = {comp['rho']:+.4f}  ({p_str}){tag}")


if __name__ == '__main__':
    main()
