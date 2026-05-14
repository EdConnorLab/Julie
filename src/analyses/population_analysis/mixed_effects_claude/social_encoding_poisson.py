# social_encoding_poisson.py
"""
Poisson GLMM encoding model for social information in primate MTL.

Same analysis as social_encoding_analysis.py but uses a Poisson model
for spike counts instead of a Gaussian model for firing rates.

Why Poisson:
  - Spike counts are discrete, non-negative, and Poisson-distributed
  - Poisson variance scales with the mean (high-rate neurons have more
    trial-to-trial variability), which the Gaussian LMM ignores
  - Properly accounts for the noise structure → tighter estimates

Implementation:
  Uses GEE (Generalized Estimating Equations) with Poisson family and
  exchangeable working correlation within NeuronID clusters. This handles
  the clustering by neuron without full random-effects GLMM, which is
  poorly supported in Python. GEE gives population-averaged coefficients
  with robust (sandwich) standard errors.

  Test statistic for permutation: deviance-based pseudo-R².

Usage:
    python social_encoding_poisson.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable, Independence

from analyses.population_analysis.state_space.data_loading import load_and_filter
from social_encoding_config import SocialEncodingConfig

# Reuse feature loading, PCA, trial table, and plotting from the Gaussian version
from social_encoding_analysis import (
    BEHAVIOR_FILES,
    load_social_features,
    reduce_features,
    plot_social_pca,
    build_trial_table,
    per_neuron_encoding,
    _p_to_stars,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Spike count computation
# ═══════════════════════════════════════════════════════════════════════

def compute_spike_counts(df, window):
    """Compute per-trial spike count and window duration.

    Parameters
    ----------
    df : DataFrame with SpikeTimes, EpochStartStop columns
    window : tuple (start_s, end_s) relative to epoch start

    Returns
    -------
    df : copy with added 'spike_count' and 'log_duration' columns
    """
    win_start, win_end = window
    win_dur = win_end - win_start

    counts = np.zeros(len(df), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        epoch_start = row['EpochStartStop'][0]
        spk = np.asarray(row['SpikeTimes'])
        if len(spk) > 0:
            spk_rel = spk - epoch_start
            counts[i] = int(np.sum((spk_rel >= win_start) & (spk_rel < win_end)))

    df = df.copy()
    df['spike_count'] = counts
    df['log_duration'] = np.log(win_dur)    # offset for converting count → rate

    # Also add rate for per-neuron OLS (supplementary)
    df['rate'] = counts / win_dur
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Poisson GEE model
# ═══════════════════════════════════════════════════════════════════════

def fit_poisson_model(trial_df, n_pcs):
    """Fit Poisson GEE: spike_count ~ social_PCs + offset(log_duration),
    clustered by NeuronID.

    Uses exchangeable working correlation within neurons and robust
    (sandwich) standard errors.

    Returns
    -------
    result : dict with pseudo_r2, deviance, coefficients, etc.
    """
    pc_cols = [f"social_PC{i+1}" for i in range(n_pcs)]

    # Sort by NeuronID (required by GEE)
    trial_df = trial_df.sort_values('NeuronID').reset_index(drop=True)

    # Design matrices
    X_full = sm.add_constant(trial_df[pc_cols].values)
    X_null = sm.add_constant(np.ones(len(trial_df)))  # intercept only

    y = trial_df['spike_count'].values
    groups = trial_df['NeuronID'].values
    offset = trial_df['log_duration'].values

    # Full model
    try:
        model_full = GEE(
            endog=y,
            exog=X_full,
            groups=groups,
            family=Poisson(),
            offset=offset,
            cov_struct=Exchangeable(),
        )
        result_full = model_full.fit(maxiter=100)
    except Exception as e:
        warnings.warn(f"Exchangeable correlation failed ({e}), falling back to Independence")
        model_full = GEE(
            endog=y,
            exog=X_full,
            groups=groups,
            family=Poisson(),
            offset=offset,
            cov_struct=Independence(),
        )
        result_full = model_full.fit(maxiter=100)

    # Null model
    try:
        model_null = GEE(
            endog=y,
            exog=X_null,
            groups=groups,
            family=Poisson(),
            offset=offset,
            cov_struct=Exchangeable(),
        )
        result_null = model_null.fit(maxiter=100)
    except Exception:
        model_null = GEE(
            endog=y,
            exog=X_null,
            groups=groups,
            family=Poisson(),
            offset=offset,
            cov_struct=Independence(),
        )
        result_null = model_null.fit(maxiter=100)

    # Deviance-based pseudo-R²
    # For Poisson: deviance = 2 * sum(y*log(y/mu) - (y - mu))
    # We use the QIC or deviance comparison
    dev_full = _poisson_deviance(y, result_full.fittedvalues)
    dev_null = _poisson_deviance(y, result_null.fittedvalues)
    pseudo_r2 = 1 - (dev_full / dev_null) if dev_null > 0 else 0.0

    # Extract coefficients (skip intercept at index 0)
    coef_names = ['Intercept'] + pc_cols
    coefficients = dict(zip(coef_names, result_full.params))
    pvalues = dict(zip(coef_names, result_full.pvalues))

    return {
        'model_full': result_full,
        'model_null': result_null,
        'pseudo_r2': pseudo_r2,
        'deviance_full': dev_full,
        'deviance_null': dev_null,
        'coefficients': coefficients,
        'pvalues': pvalues,
    }


def _poisson_deviance(y, mu):
    """Compute Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Handle y=0 case: 0*log(0) = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return 2.0 * np.sum(terms)


# ═══════════════════════════════════════════════════════════════════════
# 3. Permutation test (Poisson version)
# ═══════════════════════════════════════════════════════════════════════

def permutation_test_poisson(trial_df, features_pc, n_pcs, cfg):
    """Shuffle monkey → social-feature mapping, refit Poisson GEE.

    Same logic as the Gaussian version but uses pseudo-R² from
    deviance as the test statistic.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    pc_cols = [f"social_PC{i+1}" for i in range(n_pcs)]

    obs_result = fit_poisson_model(trial_df, n_pcs)
    obs_r2 = obs_result['pseudo_r2']

    monkey_names = features_pc.index.values.copy()
    n_monkeys = len(monkey_names)

    null_r2 = np.zeros(cfg.n_permutations)

    for i in range(cfg.n_permutations):
        shuffled_idx = rng.permutation(n_monkeys)
        shuffled_pcs = features_pc.values[shuffled_idx]
        shuffled_pc_df = pd.DataFrame(
            shuffled_pcs, index=monkey_names, columns=pc_cols)

        perm_df = trial_df.drop(columns=pc_cols).copy()
        perm_df = perm_df.merge(
            shuffled_pc_df, left_on='MonkeyName', right_index=True, how='left')

        try:
            perm_result = fit_poisson_model(perm_df, n_pcs)
            null_r2[i] = perm_result['pseudo_r2']
        except Exception:
            null_r2[i] = np.nan

        if (i + 1) % 500 == 0:
            print(f"      permutation {i+1}/{cfg.n_permutations}")

    p_value = np.nanmean(null_r2 >= obs_r2)

    return {
        'observed_r2': obs_r2,
        'null_r2': null_r2,
        'p_value': p_value,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_r2_comparison_poisson(all_results, cfg, save_path=None):
    """Bar chart: pseudo-R² per group with permutation p-value stars."""
    groups = list(all_results.keys())
    r2_vals = [all_results[g]['model']['pseudo_r2'] for g in groups]
    p_vals = [all_results[g]['permutation']['p_value'] for g in groups]
    colors = [cfg.group_colors.get(g, 'gray') for g in groups]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(groups)), r2_vals, color=colors, edgecolor='k', alpha=0.8)

    for i, (r2, p) in enumerate(zip(r2_vals, p_vals)):
        star = _p_to_stars(p)
        ax.text(i, r2 + max(r2_vals) * 0.03, star, ha='center', fontsize=12)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Pseudo-R² (Poisson deviance)', fontsize=11)
    ax.set_title(f"Social Encoding (Poisson): {cfg.region}, "
                 f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms",
                 fontsize=12)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_null_distribution_poisson(perm_result, group_name, cfg, save_path=None):
    """Histogram of null pseudo-R² distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    null = perm_result['null_r2']
    obs = perm_result['observed_r2']
    p = perm_result['p_value']

    ax.hist(null[~np.isnan(null)], bins=50, color='gray', alpha=0.7,
            edgecolor='k', lw=0.3, label='Null distribution')
    ax.axvline(obs, color=cfg.group_colors.get(group_name, 'red'),
               lw=2, ls='--', label=f'Observed (R²={obs:.4f})')
    ax.set_xlabel('Pseudo-R² (Poisson deviance)')
    ax.set_ylabel('Count')
    ax.set_title(f"{group_name}: Poisson permutation test (p={p:.4f})")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_gaussian_vs_poisson(gaussian_results, poisson_results, cfg, save_path=None):
    """Side-by-side comparison of Gaussian and Poisson results."""
    groups = sorted(set(gaussian_results.keys()) & set(poisson_results.keys()))
    if not groups:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # R² comparison
    ax = axes[0]
    x = np.arange(len(groups))
    w = 0.35
    gauss_r2 = [gaussian_results[g]['model']['marginal_r2'] for g in groups]
    poiss_r2 = [poisson_results[g]['model']['pseudo_r2'] for g in groups]
    ax.bar(x - w/2, gauss_r2, w, label='Gaussian (marginal R²)',
           color='steelblue', edgecolor='k', alpha=0.8)
    ax.bar(x + w/2, poiss_r2, w, label='Poisson (pseudo-R²)',
           color='coral', edgecolor='k', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Effect size comparison', fontsize=12)
    ax.legend(fontsize=9)

    # P-value comparison
    ax = axes[1]
    gauss_p = [gaussian_results[g]['permutation']['p_value'] for g in groups]
    poiss_p = [poisson_results[g]['permutation']['p_value'] for g in groups]
    ax.bar(x - w/2, gauss_p, w, label='Gaussian',
           color='steelblue', edgecolor='k', alpha=0.8)
    ax.bar(x + w/2, poiss_p, w, label='Poisson',
           color='coral', edgecolor='k', alpha=0.8)
    ax.axhline(0.05, color='red', ls='--', lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('Permutation p-value', fontsize=11)
    ax.set_title('Significance comparison', fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle(f"{cfg.region}, {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms, "
                 f"mode={cfg.feature_mode}", fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_group_poisson(df, group_name, cfg):
    """Run Poisson encoding analysis for one group."""
    print(f"\n  --- Loading social features for {group_name} ---")
    try:
        features = load_social_features(group_name, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"  SKIP — {e}")
        return None

    if cfg.subject_name in features.index:
        features = features.drop(cfg.subject_name)

    print(f"    Feature matrix: {features.shape[0]} monkeys × {features.shape[1]} features")

    features_pc, pca, scaler = reduce_features(features, n_components=cfg.n_feature_pcs)

    trial_df = build_trial_table(df, features_pc, group_name, cfg)
    if trial_df is None or len(trial_df) == 0:
        print(f"  SKIP — no valid trials")
        return None

    # Fit Poisson model
    print(f"\n    Fitting Poisson GEE model ...")
    model_result = fit_poisson_model(trial_df, cfg.n_feature_pcs)
    print(f"    Pseudo-R² (deviance): {model_result['pseudo_r2']:.6f}")
    print(f"    Deviance full:  {model_result['deviance_full']:.2f}")
    print(f"    Deviance null:  {model_result['deviance_null']:.2f}")
    print(f"    Coefficients (log-rate scale):")
    for name, val in model_result['coefficients'].items():
        pv = model_result['pvalues'].get(name, np.nan)
        print(f"      {name:>15s}: {val:+.6f}  (p={pv:.4e})")

    # Per-neuron encoding (same as Gaussian — uses OLS on trial-averaged rates)
    print(f"\n    Per-neuron encoding (supplementary) ...")
    neuron_r2 = per_neuron_encoding(trial_df, cfg.n_feature_pcs)
    print(f"    Mean per-neuron R²: {neuron_r2['r2'].mean():.4f} "
          f"± {neuron_r2['r2'].std():.4f} (n={len(neuron_r2)} neurons)")

    # Permutation test
    perm_result = None
    if cfg.n_permutations > 0:
        print(f"\n    Running Poisson permutation test ({cfg.n_permutations} perms) ...")
        perm_result = permutation_test_poisson(
            trial_df, features_pc, cfg.n_feature_pcs, cfg)
        print(f"    Observed pseudo-R²: {perm_result['observed_r2']:.6f}")
        print(f"    Permutation p:      {perm_result['p_value']:.4f}")

    return {
        'model': model_result,
        'per_neuron': neuron_r2,
        'permutation': perm_result,
        'features_pc': features_pc,
        'pca': pca,
        'trial_df': trial_df,
    }


def main():
    cfg = SocialEncodingConfig(
        region='AMG',
        session=None,
        window=(0.300, 0.500),
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        exclude_groups=['Stranger Things'],
        feature_mode='full_profile',
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=500,
        save_plots=True,
        save_dir='social_encoding_poisson_results',
        groups=['Zombies', 'Best Frans', 'Instigators'],
    )
    cfg.validate()

    # ─── Load neural data ───
    print(f"\n{'='*60}")
    print(f"Social Encoding — POISSON GEE Model")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Feature mode: {cfg.feature_mode}")
    print(f"  PCs: {cfg.n_feature_pcs}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"{'='*60}\n")

    df = load_and_filter(cfg)

    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)

    # ─── Compute spike counts ───
    print("Computing spike counts ...")
    df = compute_spike_counts(df, cfg.window)

    print(f"  Spike count stats: mean={df['spike_count'].mean():.2f}, "
          f"var={df['spike_count'].var():.2f}, "
          f"var/mean={df['spike_count'].var()/df['spike_count'].mean():.2f} "
          f"(>1 = over-dispersed)")

    # ─── Run for each group ───
    all_results = {}
    for group in cfg.groups:
        print(f"\n{'='*60}")
        print(f"Group: {group}")
        print(f"{'='*60}")
        result = run_group_poisson(df, group, cfg)
        if result is not None:
            all_results[group] = result

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"SUMMARY (Poisson GEE)")
    print(f"{'='*60}")
    for group, res in all_results.items():
        perm_p = res['permutation']['p_value'] if res['permutation'] else 'N/A'
        print(f"\n  {group}:")
        print(f"    Pseudo-R²:       {res['model']['pseudo_r2']:.6f}")
        print(f"    Permutation p:   {perm_p}")
        print(f"    Per-neuron R²:   {res['per_neuron']['r2'].mean():.4f} "
              f"± {res['per_neuron']['r2'].std():.4f}")

    # ─── Plots ───
    if cfg.save_plots and len(all_results) > 0:
        save_base = f"{cfg.save_dir}/{cfg.region}"
        os.makedirs(save_base, exist_ok=True)

        plot_r2_comparison_poisson(all_results, cfg,
                                   save_path=f"{save_base}/r2_comparison_poisson.png")

        for group, res in all_results.items():
            plot_social_pca(
                res['features_pc'], res['pca'], group, cfg,
                save_path=f"{save_base}/social_pca_{group}.png")

            if res['permutation'] is not None:
                plot_null_distribution_poisson(
                    res['permutation'], group, cfg,
                    save_path=f"{save_base}/null_dist_poisson_{group}.png")

        print(f"\nPlots saved to {save_base}/")

    plt.show()
    return all_results


if __name__ == '__main__':
    main()
