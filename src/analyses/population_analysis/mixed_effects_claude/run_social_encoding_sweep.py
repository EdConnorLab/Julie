# run_social_encoding_sweep.py
"""
Run social encoding analysis across feature modes, groups, and PCs.

CLI usage:
    # Feature-mode sweep (4 modes × 3 groups)
    python run_social_encoding_sweep.py sweep --region AMG --window 200-600 --perms 5000

    # PCs sweep (full_profile, 2–5 PCs × 3 groups)
    python run_social_encoding_sweep.py pcs --region ALL --window 300-600 --min-pcs 2 --max-pcs 5

    # Both sweeps in sequence
    python run_social_encoding_sweep.py both --region AMG --window 200-600

See run_overnight.sh for running all region × window combinations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from social_encoding_config import SocialEncodingConfig
from social_encoding_analysis import (
    compute_trial_rates,
    load_social_features,
    reduce_features,
    plot_social_pca,
    build_trial_table,
    fit_mixed_model,
    permutation_test,
    per_neuron_encoding,
    _p_to_stars,
)


def run_sweep(region='AMG', window=(0.200, 0.600), n_permutations=5000):
    # ── Base config ──
    win_label = f"{int(window[0]*1000)}-{int(window[1]*1000)}ms"
    cfg = SocialEncodingConfig(
        region=region,
        session=None,
        window=window,
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        exclude_groups=['Stranger Things'],
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=n_permutations,
        save_plots=True,
        save_dir=f'social_encoding_sweep/{region}/{win_label}',
    )
    cfg.validate()

    # ── Load & prep data once ──
    print(f"\n{'='*70}")
    print(f"Social Encoding Sweep")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"{'='*70}\n")

    df = load_and_filter(cfg)

    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)

    print("Computing trial-level firing rates ...")
    df = compute_trial_rates(df, cfg.window)

    # ── Sweep grid ──
    feature_modes = ['full_profile', 'to_subject', 'from_subject', 'summary']
    groups = ['Zombies', 'Best Frans', 'Instigators']

    # Collect results
    rows = []

    for mode in feature_modes:
        cfg.feature_mode = mode

        # Adjust n_feature_pcs based on mode
        # to_subject / from_subject: only 3 features (one per behavior type) → max 2 PCs
        # summary: 6 features → 2 PCs fine
        # full_profile: ~27-30 features → 2 PCs fine
        if mode in ('to_subject', 'from_subject'):
            cfg.n_feature_pcs = 2  # 3 features, so max 2 PCs
        else:
            cfg.n_feature_pcs = 2

        for group in groups:
            print(f"\n{'─'*60}")
            print(f"  feature_mode={mode}, group={group}")
            print(f"{'─'*60}")

            try:
                features = load_social_features(group, cfg)
            except (FileNotFoundError, ValueError) as e:
                print(f"  SKIP — {e}")
                rows.append({
                    'feature_mode': mode,
                    'group': group,
                    'marginal_r2': np.nan,
                    'perm_pvalue': np.nan,
                    'mean_neuron_r2': np.nan,
                    'n_neurons': 0,
                    'n_monkeys': 0,
                    'n_trials': 0,
                    'n_features': 0,
                    'n_pcs': 0,
                    'pca_var_explained': np.nan,
                })
                continue

            # Exclude subject
            if cfg.subject_name in features.index:
                features = features.drop(cfg.subject_name)

            n_features = features.shape[1]
            n_monkeys_feat = features.shape[0]

            # Adjust PCs if fewer monkeys than requested PCs + 1
            actual_pcs = min(cfg.n_feature_pcs, n_monkeys_feat - 1, n_features)
            if actual_pcs < 1:
                print(f"  SKIP — not enough monkeys/features for PCA")
                rows.append({
                    'feature_mode': mode,
                    'group': group,
                    'marginal_r2': np.nan,
                    'perm_pvalue': np.nan,
                    'mean_neuron_r2': np.nan,
                    'n_neurons': 0,
                    'n_monkeys': n_monkeys_feat,
                    'n_trials': 0,
                    'n_features': n_features,
                    'n_pcs': 0,
                    'pca_var_explained': np.nan,
                })
                continue

            print(f"    Features: {n_monkeys_feat} monkeys × {n_features} features")

            features_pc, pca, scaler = reduce_features(features, n_components=actual_pcs)

            trial_df = build_trial_table(df, features_pc, group, cfg)
            if trial_df is None or len(trial_df) == 0:
                print(f"  SKIP — no valid trials")
                rows.append({
                    'feature_mode': mode,
                    'group': group,
                    'marginal_r2': np.nan,
                    'perm_pvalue': np.nan,
                    'mean_neuron_r2': np.nan,
                    'n_neurons': 0,
                    'n_monkeys': n_monkeys_feat,
                    'n_trials': 0,
                    'n_features': n_features,
                    'n_pcs': actual_pcs,
                    'pca_var_explained': np.nan,
                })
                continue

            # Fit model
            model_result = fit_mixed_model(trial_df, actual_pcs)
            print(f"    Marginal R²: {model_result['marginal_r2']:.4f}")

            # Per-neuron
            neuron_r2 = per_neuron_encoding(trial_df, actual_pcs)

            # Permutation test
            print(f"    Running permutation test ...")
            perm_result = permutation_test(trial_df, features_pc, actual_pcs, cfg)
            print(f"    Permutation p: {perm_result['p_value']:.4f}")

            # PCA scatter plot
            save_dir_pca = f"{cfg.save_dir}/pca_plots"
            plot_social_pca(features_pc, pca, group, cfg,
                            save_path=f"{save_dir_pca}/{mode}_{group}_pca.png")

            rows.append({
                'feature_mode': mode,
                'group': group,
                'marginal_r2': model_result['marginal_r2'],
                'perm_pvalue': perm_result['p_value'],
                'mean_neuron_r2': neuron_r2['r2'].mean(),
                'n_neurons': trial_df['NeuronID'].nunique(),
                'n_monkeys': trial_df['MonkeyName'].nunique(),
                'n_trials': len(trial_df),
                'n_features': n_features,
                'n_pcs': actual_pcs,
                'pca_var_explained': pca.explained_variance_ratio_.sum(),
            })

    # ── Summary table ──
    results_df = pd.DataFrame(rows)

    print(f"\n\n{'='*90}")
    print(f"SWEEP SUMMARY — {cfg.region}, {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms, {cfg.n_permutations} permutations")
    print(f"{'='*90}")
    print(f"\n{'Feature Mode':<16s} {'Group':<16s} {'R²':>8s} {'Perm p':>8s} "
          f"{'Neuron R²':>10s} {'Neurons':>8s} {'Monkeys':>8s} "
          f"{'Trials':>8s} {'Feats':>6s} {'PCs':>4s}")
    print(f"{'-'*96}")

    for _, row in results_df.iterrows():
        if np.isnan(row['marginal_r2']):
            print(f"{row['feature_mode']:<16s} {row['group']:<16s} {'SKIP':>8s}")
            continue

        star = _p_to_stars(row['perm_pvalue'])
        print(f"{row['feature_mode']:<16s} {row['group']:<16s} "
              f"{row['marginal_r2']:>8.4f} "
              f"{row['perm_pvalue']:>7.4f}{star:<1s} "
              f"{row['mean_neuron_r2']:>10.4f} "
              f"{int(row['n_neurons']):>8d} "
              f"{int(row['n_monkeys']):>8d} "
              f"{int(row['n_trials']):>8d} "
              f"{int(row['n_features']):>6d} "
              f"{int(row['n_pcs']):>4d}")

    print(f"{'='*96}")

    # ── Plot: grouped bar chart ──
    os.makedirs(cfg.save_dir, exist_ok=True)

    plot_sweep_comparison(results_df, cfg, save_path=f"{cfg.save_dir}/sweep_comparison.png")
    plot_sweep_perm_pvalues(results_df, cfg, save_path=f"{cfg.save_dir}/sweep_perm_pvalues.png")

    # Save results table
    results_df.to_csv(f"{cfg.save_dir}/sweep_results.csv", index=False)
    print(f"\nResults saved to {cfg.save_dir}/")

    plt.close('all')
    return results_df


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────

def plot_sweep_comparison(results_df, cfg, save_path=None):
    """Grouped bar chart: R² by group, grouped by feature_mode."""
    rdf = results_df.dropna(subset=['marginal_r2'])
    modes = rdf['feature_mode'].unique()
    groups = rdf['group'].unique()

    fig, ax = plt.subplots(figsize=(max(8, len(modes) * 3), 5))

    bar_width = 0.8 / len(groups)
    x = np.arange(len(modes))

    for g_idx, group in enumerate(groups):
        r2_vals = []
        perm_ps = []
        for mode in modes:
            subset = rdf[(rdf['feature_mode'] == mode) & (rdf['group'] == group)]
            if len(subset) > 0:
                r2_vals.append(subset['marginal_r2'].values[0])
                perm_ps.append(subset['perm_pvalue'].values[0])
            else:
                r2_vals.append(0)
                perm_ps.append(np.nan)

        offset = (g_idx - (len(groups) - 1) / 2) * bar_width
        color = cfg.group_colors.get(group, 'gray')
        ax.bar(x + offset, r2_vals, bar_width, label=group,
               color=color, edgecolor='k', alpha=0.8)

        for i, (r2, p) in enumerate(zip(r2_vals, perm_ps)):
            star = _p_to_stars(p)
            if r2 > 0:
                ax.text(x[i] + offset, r2 + 0.0002, star,
                        ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_xlabel('Feature Mode', fontsize=11)
    ax.set_ylabel('Marginal R²', fontsize=11)
    ax.set_title(f"Social Encoding: {cfg.region}, "
                 f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms\n"
                 f"(stars = permutation p-value)", fontsize=12)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sweep_perm_pvalues(results_df, cfg, save_path=None):
    """Dot plot of permutation p-values with significance lines."""
    rdf = results_df.dropna(subset=['perm_pvalue'])
    modes = rdf['feature_mode'].unique()
    groups = rdf['group'].unique()

    fig, ax = plt.subplots(figsize=(max(8, len(modes) * 3), 4))

    x = np.arange(len(modes))
    spread = 0.8 / len(groups)

    for g_idx, group in enumerate(groups):
        p_vals = []
        for mode in modes:
            subset = rdf[(rdf['feature_mode'] == mode) & (rdf['group'] == group)]
            if len(subset) > 0:
                p_vals.append(subset['perm_pvalue'].values[0])
            else:
                p_vals.append(np.nan)

        offset = (g_idx - (len(groups) - 1) / 2) * spread
        color = cfg.group_colors.get(group, 'gray')
        ax.scatter(x + offset, p_vals, color=color, s=80,
                   edgecolors='k', lw=0.5, label=group, zorder=5)

    ax.axhline(0.05, color='red', ls='--', lw=1, alpha=0.7, label='p = 0.05')
    ax.axhline(0.01, color='red', ls=':', lw=1, alpha=0.5, label='p = 0.01')

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_xlabel('Feature Mode', fontsize=11)
    ax.set_ylabel('Permutation p-value', fontsize=11)
    ax.set_title(f"Permutation p-values: {cfg.region}, "
                 f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# PCs sweep: test 1–max_pcs for full_profile, per group
# ══════════════════════════════════════════════════════════════════════

def run_pcs_sweep(region='ALL', window=(0.200, 0.600), min_pcs=2, max_pcs=5, n_permutations=5000):
    """Sweep number of PCs (min_pcs to max_pcs) for full_profile mode.

    Tests whether the social encoding result is driven by a specific
    number of PCs or just improves monotonically (overfitting risk).

    Key check: Zombies should improve with more PCs while Instigators
    should stay flat. If both improve, it's overfitting.
    """
    win_label = f"{int(window[0]*1000)}-{int(window[1]*1000)}ms"
    cfg = SocialEncodingConfig(
        region=region,
        session=None,
        window=window,
        min_epoch_duration=2.0,
        min_reps_per_monkey=3,
        exclude_groups=['Stranger Things'],
        feature_mode='full_profile',
        n_feature_pcs=2,           # will be overridden per iteration
        normalization=None,
        pseudo_population=True,
        n_permutations=n_permutations,
        save_plots=True,
        save_dir=f'social_encoding_pcs_sweep/{region}/{win_label}',
    )
    cfg.validate()

    groups = ['Zombies', 'Best Frans', 'Instigators']

    print(f"\n{'='*70}")
    print(f"PCs Sweep — full_profile, {region}, {win_label}")
    print(f"  PCs: {min_pcs} to {max_pcs}")
    print(f"  Permutations: {n_permutations}")
    print(f"{'='*70}\n")

    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)

    print("Computing trial-level firing rates ...")
    df = compute_trial_rates(df, cfg.window)

    # Preload features for each group (only need to do this once)
    group_features = {}
    for group in groups:
        try:
            features = load_social_features(group, cfg)
            if cfg.subject_name in features.index:
                features = features.drop(cfg.subject_name)
            group_features[group] = features
        except (FileNotFoundError, ValueError) as e:
            print(f"  {group}: SKIP — {e}")

    rows = []

    for n_pcs in range(min_pcs, max_pcs + 1):
        cfg.n_feature_pcs = n_pcs

        for group in groups:
            if group not in group_features:
                rows.append({
                    'n_pcs': n_pcs, 'group': group,
                    'marginal_r2': np.nan, 'perm_pvalue': np.nan,
                    'pca_var_explained': np.nan,
                })
                continue

            features = group_features[group]

            print(f"\n{'─'*50}")
            print(f"  n_pcs={n_pcs}, group={group}")
            print(f"{'─'*50}")

            # Cap PCs at (n_monkeys - 1) or n_features
            actual_pcs = min(n_pcs, features.shape[0] - 1, features.shape[1])
            if actual_pcs < n_pcs:
                print(f"    Capped to {actual_pcs} PCs (only {features.shape[0]} monkeys)")
            if actual_pcs < 1:
                rows.append({
                    'n_pcs': n_pcs, 'group': group,
                    'marginal_r2': np.nan, 'perm_pvalue': np.nan,
                    'pca_var_explained': np.nan,
                })
                continue

            features_pc, pca, scaler = reduce_features(features, n_components=actual_pcs)

            trial_df = build_trial_table(df, features_pc, group, cfg)
            if trial_df is None or len(trial_df) == 0:
                rows.append({
                    'n_pcs': n_pcs, 'group': group,
                    'marginal_r2': np.nan, 'perm_pvalue': np.nan,
                    'pca_var_explained': np.nan,
                })
                continue

            model_result = fit_mixed_model(trial_df, actual_pcs)
            print(f"    Marginal R²: {model_result['marginal_r2']:.4f}")

            perm_result = permutation_test(trial_df, features_pc, actual_pcs, cfg)
            print(f"    Permutation p: {perm_result['p_value']:.4f}")

            rows.append({
                'n_pcs': n_pcs,
                'group': group,
                'marginal_r2': model_result['marginal_r2'],
                'perm_pvalue': perm_result['p_value'],
                'pca_var_explained': pca.explained_variance_ratio_.sum(),
            })

    results_df = pd.DataFrame(rows)

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"PCs SWEEP SUMMARY — full_profile, {region}, {win_label}")
    print(f"{'='*70}")
    print(f"\n{'PCs':>4s}  {'Group':<16s}  {'R²':>8s}  {'Perm p':>8s}  {'PCA var':>8s}")
    print(f"{'-'*50}")
    for _, row in results_df.iterrows():
        if np.isnan(row['marginal_r2']):
            print(f"{int(row['n_pcs']):>4d}  {row['group']:<16s}  {'SKIP':>8s}")
            continue
        star = _p_to_stars(row['perm_pvalue'])
        print(f"{int(row['n_pcs']):>4d}  {row['group']:<16s}  "
              f"{row['marginal_r2']:>8.4f}  "
              f"{row['perm_pvalue']:>7.4f}{star:<1s}  "
              f"{row['pca_var_explained']:>8.3f}")

    # ── Plot ──
    os.makedirs(cfg.save_dir, exist_ok=True)

    plot_pcs_sweep(results_df, cfg, save_path=f"{cfg.save_dir}/pcs_sweep.png")

    results_df.to_csv(f"{cfg.save_dir}/pcs_sweep_results.csv", index=False)
    print(f"\nResults saved to {cfg.save_dir}/")

    plt.close('all')
    return results_df


def plot_pcs_sweep(results_df, cfg, save_path=None):
    """Line plot: permutation p-value vs number of PCs, one line per group."""
    rdf = results_df.dropna(subset=['perm_pvalue'])
    groups = rdf['group'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R² vs PCs
    ax = axes[0]
    for group in groups:
        gdf = rdf[rdf['group'] == group].sort_values('n_pcs')
        color = cfg.group_colors.get(group, 'gray')
        ax.plot(gdf['n_pcs'], gdf['marginal_r2'], 'o-',
                color=color, label=group, lw=2, markersize=8)
    ax.set_xlabel('Number of PCs', fontsize=11)
    ax.set_ylabel('Marginal R²', fontsize=11)
    ax.set_title('Effect size vs. model complexity', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, int(rdf['n_pcs'].max()) + 1))

    # Right: p-value vs PCs
    ax = axes[1]
    for group in groups:
        gdf = rdf[rdf['group'] == group].sort_values('n_pcs')
        color = cfg.group_colors.get(group, 'gray')
        ax.plot(gdf['n_pcs'], gdf['perm_pvalue'], 'o-',
                color=color, label=group, lw=2, markersize=8)
    ax.axhline(0.05, color='red', ls='--', lw=1, alpha=0.7, label='p = 0.05')
    ax.set_xlabel('Number of PCs', fontsize=11)
    ax.set_ylabel('Permutation p-value', fontsize=11)
    ax.set_title('Significance vs. model complexity', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, int(rdf['n_pcs'].max()) + 1))

    fig.suptitle(f"full_profile PCs sweep — {cfg.region}, "
                 f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Social encoding sweep analysis',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('mode', choices=['sweep', 'pcs', 'both'],
                        help='sweep = feature-mode sweep\n'
                             'pcs   = PCs sweep (full_profile only)\n'
                             'both  = run sweep then pcs sweep')
    parser.add_argument('--region', default='AMG',
                        help='Brain region: AMG, ER, or ALL (default: AMG)')
    parser.add_argument('--window', default='200-600',
                        help='Firing rate window in ms, e.g. 200-600 (default: 200-600)')
    parser.add_argument('--perms', type=int, default=5000,
                        help='Number of permutations (default: 5000)')
    parser.add_argument('--min-pcs', type=int, default=2,
                        help='Min PCs for pcs sweep (default: 2)')
    parser.add_argument('--max-pcs', type=int, default=5,
                        help='Max PCs for pcs sweep (default: 5)')

    args = parser.parse_args()

    # Parse window string → tuple in seconds
    w_start, w_end = args.window.split('-')
    window = (float(w_start) / 1000, float(w_end) / 1000)

    # Use non-interactive backend for overnight runs
    import matplotlib
    matplotlib.use('Agg')

    if args.mode in ('sweep', 'both'):
        run_sweep(region=args.region, window=window, n_permutations=args.perms)

    if args.mode in ('pcs', 'both'):
        run_pcs_sweep(region=args.region, window=window,
                      min_pcs=args.min_pcs, max_pcs=args.max_pcs,
                      n_permutations=args.perms)
