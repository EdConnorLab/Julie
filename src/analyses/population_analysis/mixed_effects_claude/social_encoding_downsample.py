# social_encoding_downsample.py
"""
Downsampling analysis for group-size-equalized comparisons.

Problem: Best Frans has only 5 monkeys; Zombies has 9, Instigators 14.
Comparing R² or p-values across groups with different sizes is confounded
because more monkeys = more feature diversity = higher chance of fitting.

Solution: Randomly subsample k monkeys from each larger group (where k =
size of the smallest group being compared), run the full analysis on each
subsample, and repeat many times.  Report the distribution of R² and
p-values across subsamples.

Key design choice: subsampling is RANDOM (no composition matching).
Composition matching (e.g., always include the alpha male) would bias
the subsamples toward maximally diverse social configurations, inflating
the chance of finding an effect.  Random subsampling is unbiased — some
draws will include the alpha, some won't — and the average over many
draws reflects the true expected effect at that group size.

Usage:
    python social_encoding_downsample.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from analyses.population_analysis.state_space.data_loading import load_and_filter
from social_encoding_config import SocialEncodingConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from social_encoding_analysis import (
    compute_trial_rates,
    load_social_features,
    reduce_features,
    build_trial_table,
    fit_mixed_model,
    permutation_test,
    _p_to_stars,
)


def _reduce_features_quiet(features, n_components=2):
    """Same as reduce_features but without print output (for repeated calls)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    pc_cols = [f"social_PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, index=features.index, columns=pc_cols)
    return pc_df, pca, scaler


# ═══════════════════════════════════════════════════════════════════════
# 1. Single downsampled run
# ═══════════════════════════════════════════════════════════════════════

def run_downsampled(df, group_name, cfg, monkey_subset, n_pcs):
    """Run social encoding on a subset of monkeys from one group.

    Parameters
    ----------
    df : full trial DataFrame (already has 'rate' column)
    group_name : str
    cfg : SocialEncodingConfig
    monkey_subset : list of monkey names to include
    n_pcs : int, number of PCA components

    Returns
    -------
    dict with marginal_r2 and perm_pvalue, or None if analysis fails
    """
    import io, contextlib

    # Load full feature matrix, then subset
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            features = load_social_features(group_name, cfg)
    except (FileNotFoundError, ValueError):
        return None

    if cfg.subject_name in features.index:
        features = features.drop(cfg.subject_name)

    # Subset to selected monkeys
    features = features.loc[features.index.isin(monkey_subset)]
    if len(features) < 3:  # need at least 3 monkeys for meaningful PCA
        return None

    # PCA (quiet version)
    actual_pcs = min(n_pcs, features.shape[0] - 1, features.shape[1])
    if actual_pcs < 1:
        return None

    features_pc, pca, scaler = _reduce_features_quiet(features, n_components=actual_pcs)

    # Build trial table (suppress print)
    with contextlib.redirect_stdout(io.StringIO()):
        trial_df = build_trial_table(df, features_pc, group_name, cfg)
    if trial_df is None or len(trial_df) == 0:
        return None

    # Fit model
    try:
        model_result = fit_mixed_model(trial_df, actual_pcs)
    except Exception:
        return None

    # Permutation test (suppress progress prints)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            perm_result = permutation_test(trial_df, features_pc, actual_pcs, cfg)
    except Exception:
        return None

    return {
        'marginal_r2': model_result['marginal_r2'],
        'perm_pvalue': perm_result['p_value'],
        'n_monkeys': len(monkey_subset),
        'n_trials': len(trial_df),
        'n_pcs': actual_pcs,
        'monkeys': list(monkey_subset),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Downsampling loop
# ═══════════════════════════════════════════════════════════════════════

def run_downsample_analysis(df, group_name, cfg, target_size, n_pcs,
                            n_subsamples=200, seed=42):
    """Repeat the analysis n_subsamples times, each with a random
    subset of target_size monkeys.

    If the group already has <= target_size monkeys, runs once on all.

    Parameters
    ----------
    df : trial DataFrame with 'rate' column
    group_name : str
    cfg : SocialEncodingConfig
    target_size : int, how many monkeys per subsample
    n_pcs : int
    n_subsamples : int
    seed : int

    Returns
    -------
    results_df : DataFrame with one row per subsample
    """
    rng = np.random.default_rng(seed)

    # Get all available monkeys for this group
    try:
        features = load_social_features(group_name, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"  SKIP {group_name}: {e}")
        return pd.DataFrame()

    if cfg.subject_name in features.index:
        features = features.drop(cfg.subject_name)

    all_monkeys = list(features.index)
    n_available = len(all_monkeys)

    if n_available <= target_size:
        # Group is already at or below target size — run once
        print(f"  {group_name}: {n_available} monkeys ≤ target {target_size}, "
              f"running once (no downsampling needed)")
        result = run_downsampled(df, group_name, cfg, all_monkeys, n_pcs)
        if result is None:
            return pd.DataFrame()
        result['subsample_id'] = 0
        result['group'] = group_name
        return pd.DataFrame([result])

    print(f"  {group_name}: subsampling {target_size} from {n_available} monkeys, "
          f"{n_subsamples} iterations")

    rows = []
    for i in range(n_subsamples):
        # Random subset
        subset = list(rng.choice(all_monkeys, size=target_size, replace=False))

        result = run_downsampled(df, group_name, cfg, subset, n_pcs)

        if result is not None:
            result['subsample_id'] = i
            result['group'] = group_name
            rows.append(result)

        if (i + 1) % 50 == 0:
            n_done = len(rows)
            if n_done > 0:
                mean_r2 = np.mean([r['marginal_r2'] for r in rows])
                mean_p = np.mean([r['perm_pvalue'] for r in rows])
                print(f"    iteration {i+1}/{n_subsamples}: "
                      f"mean R²={mean_r2:.4f}, mean p={mean_p:.3f}")
            else:
                print(f"    iteration {i+1}/{n_subsamples}: no successful fits yet")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 3. Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_downsample_r2(all_results, cfg, target_size, save_path=None):
    """Violin/box plot: distribution of marginal R² per group."""
    groups = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(6, 4))

    positions = range(len(groups))
    for i, group in enumerate(groups):
        rdf = all_results[group]
        if len(rdf) == 0:
            continue
        r2 = rdf['marginal_r2'].values
        color = cfg.group_colors.get(group, 'gray')

        # Box plot
        bp = ax.boxplot([r2], positions=[i], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color='k', lw=2),
                        whiskerprops=dict(color='k'),
                        capprops=dict(color='k'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))

        # Overlay individual points
        jitter = rng_jitter(len(r2))
        ax.scatter(np.full(len(r2), i) + jitter, r2,
                   color=color, alpha=0.15, s=10, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Marginal R²', fontsize=11)
    ax.set_title(f'Downsampled to {target_size} monkeys\n'
                 f'{cfg.region}, '
                 f'{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms',
                 fontsize=12)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_downsample_pvalues(all_results, cfg, target_size, save_path=None):
    """Histogram of permutation p-values per group, with significance line."""
    groups = list(all_results.keys())
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)

    for i, group in enumerate(groups):
        ax = axes[0, i]
        rdf = all_results[group]
        if len(rdf) == 0:
            ax.set_title(f'{group}: no data')
            continue

        p_vals = rdf['perm_pvalue'].values
        color = cfg.group_colors.get(group, 'gray')

        ax.hist(p_vals, bins=20, color=color, alpha=0.7, edgecolor='k', lw=0.3)
        ax.axvline(0.05, color='red', ls='--', lw=1.5, label='p = 0.05')

        frac_sig = np.mean(p_vals < 0.05)
        ax.set_title(f'{group}\n'
                     f'{frac_sig*100:.0f}% significant (p<0.05)',
                     fontsize=11)
        ax.set_xlabel('Permutation p-value', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle(f'Downsampled to {target_size} monkeys — '
                 f'{cfg.region}, '
                 f'{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms',
                 fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def rng_jitter(n, scale=0.08):
    """Small horizontal jitter for overlaid scatter points."""
    return np.random.default_rng(0).uniform(-scale, scale, n)


# ═══════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = SocialEncodingConfig(
        region='AMG',
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        exclude_groups=['Stranger Things'],
        feature_mode='summary',
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=500,            # fewer perms per subsample (speed)
        save_plots=True,
        save_dir='social_encoding_downsample',
    )
    cfg.validate()

    # ── Settings ──
    target_size = 5       # match Best Frans size
    n_subsamples = 100    # number of random subsets to draw
    n_pcs = 2
    groups = ['Zombies', 'Best Frans', 'Instigators']

    print(f"\n{'='*70}")
    print(f"Downsampling Analysis")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Feature mode: {cfg.feature_mode}")
    print(f"  PCs: {n_pcs}")
    print(f"  Target group size: {target_size}")
    print(f"  Subsamples: {n_subsamples}")
    print(f"  Permutations per subsample: {cfg.n_permutations}")
    print(f"{'='*70}\n")

    # ── Load data ──
    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)

    print("Computing trial-level firing rates ...")
    df = compute_trial_rates(df, cfg.window)

    # ── Run for each group ──
    all_results = {}
    for group in groups:
        print(f"\n{'='*60}")
        print(f"Group: {group}")
        print(f"{'='*60}")

        rdf = run_downsample_analysis(
            df, group, cfg,
            target_size=target_size,
            n_pcs=n_pcs,
            n_subsamples=n_subsamples,
            seed=cfg.rng_seed,
        )
        all_results[group] = rdf

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"DOWNSAMPLING SUMMARY — {target_size} monkeys, "
          f"{cfg.region}, {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"{'='*70}")
    print(f"\n{'Group':<16s} {'N subs':>7s} {'Mean R²':>9s} {'Med R²':>9s} "
          f"{'Mean p':>8s} {'Med p':>8s} {'% sig':>7s}")
    print(f"{'-'*66}")

    for group in groups:
        rdf = all_results[group]
        if len(rdf) == 0:
            print(f"{group:<16s} {'SKIP':>7s}")
            continue

        r2 = rdf['marginal_r2'].values
        p = rdf['perm_pvalue'].values
        frac_sig = np.mean(p < 0.05) * 100

        print(f"{group:<16s} {len(rdf):>7d} {r2.mean():>9.4f} {np.median(r2):>9.4f} "
              f"{p.mean():>8.3f} {np.median(p):>8.3f} {frac_sig:>6.1f}%")

    # ── Plots ──
    if cfg.save_plots:
        save_base = f"{cfg.save_dir}/{cfg.region}"
        os.makedirs(save_base, exist_ok=True)

        plot_downsample_r2(
            all_results, cfg, target_size,
            save_path=f"{save_base}/downsample_r2_{target_size}monkeys.png")
        plot_downsample_pvalues(
            all_results, cfg, target_size,
            save_path=f"{save_base}/downsample_pvalues_{target_size}monkeys.png")

        # Save raw results
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(f"{save_base}/downsample_results_{target_size}monkeys.csv",
                        index=False)
        print(f"\nResults saved to {save_base}/")

    plt.close('all')
    return all_results


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Downsampling analysis')
    # parser.add_argument('--region', default='AMG')
    # parser.add_argument('--window', default='300-600')
    # parser.add_argument('--mode', default='summary',
    #                     help='Feature mode: full_profile or summary')
    # parser.add_argument('--pcs', type=int, default=2)
    # parser.add_argument('--target-size', type=int, default=5,
    #                     help='Number of monkeys per subsample')
    # parser.add_argument('--n-subsamples', type=int, default=100)
    # parser.add_argument('--perms', type=int, default=500,
    #                     help='Permutations per subsample (default: 500)')
    #
    # args = parser.parse_args()

    # w_start, w_end = args.window.split('-')
    # window = (float(w_start) / 1000, float(w_end) / 1000)

    import matplotlib
    matplotlib.use('Agg')

    cfg = SocialEncodingConfig(
        region='ALL',
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        exclude_groups=['Stranger Things'],
        feature_mode='summary',
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=500,
        save_plots=True,
        save_dir='social_encoding_downsample',
    )
    cfg.validate()

    target_size = 5 # Best frans group have 5
    n_subsamples = 100
    n_pcs = 2
    groups = ['Zombies', 'Best Frans', 'Instigators']

    print(f"\n{'='*70}")
    print(f"Downsampling Analysis")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Feature mode: {cfg.feature_mode}")
    print(f"  PCs: {n_pcs}")
    print(f"  Target group size: {target_size}")
    print(f"  Subsamples: {n_subsamples}")
    print(f"  Permutations per subsample: {cfg.n_permutations}")
    print(f"{'='*70}\n")

    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)

    print("Computing trial-level firing rates ...")
    df = compute_trial_rates(df, cfg.window)

    all_results = {}
    for group in groups:
        print(f"\n{'='*60}")
        print(f"Group: {group}")
        print(f"{'='*60}")

        rdf = run_downsample_analysis(
            df, group, cfg,
            target_size=target_size,
            n_pcs=n_pcs,
            n_subsamples=n_subsamples,
            seed=cfg.rng_seed,
        )
        all_results[group] = rdf

    # Summary
    print(f"\n\n{'='*70}")
    print(f"DOWNSAMPLING SUMMARY — {target_size} monkeys, "
          f"{cfg.region}, {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"{'='*70}")
    print(f"\n{'Group':<16s} {'N subs':>7s} {'Mean R²':>9s} {'Med R²':>9s} "
          f"{'Mean p':>8s} {'Med p':>8s} {'% sig':>7s}")
    print(f"{'-'*66}")

    for group in groups:
        rdf = all_results[group]
        if len(rdf) == 0:
            print(f"{group:<16s} {'SKIP':>7s}")
            continue
        r2 = rdf['marginal_r2'].values
        p = rdf['perm_pvalue'].values
        frac_sig = np.mean(p < 0.05) * 100
        print(f"{group:<16s} {len(rdf):>7d} {r2.mean():>9.4f} {np.median(r2):>9.4f} "
              f"{p.mean():>8.3f} {np.median(p):>8.3f} {frac_sig:>6.1f}%")

    # Save
    if cfg.save_plots:
        save_base = f"{cfg.save_dir}/{cfg.region}"
        os.makedirs(save_base, exist_ok=True)

        plot_downsample_r2(
            all_results, cfg, target_size,
            save_path=f"{save_base}/downsample_r2_{target_size}monkeys.png")
        plot_downsample_pvalues(
            all_results, cfg, target_size,
            save_path=f"{save_base}/downsample_pvalues_{target_size}monkeys.png")

        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(f"{save_base}/downsample_results_{target_size}monkeys.csv",
                        index=False)
        print(f"\nResults saved to {save_base}/")

    plt.close('all')
