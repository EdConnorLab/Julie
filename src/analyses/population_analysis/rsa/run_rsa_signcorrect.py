# run_rsa_signcorrect.py
"""
Sign-corrected RSA runner (corrected permutation test).

Uses ALL trials (no split-half). The permutation test re-estimates
neuron signs on every shuffle, so p-values account for the
selection bias inherent in sign correction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from rsa_config import SocialRSAConfig
from rsa_social import load_all_interaction_matrices
from rsa_sign_corrected import (
    run_sign_corrected_pseudopop,
    print_sign_corrected_results,
)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"

BEHAVIOR_FILES = {
    'Zombies': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_submission.xlsx',
    },
    'Best Frans': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_submission.xlsx',
    },
    'Instigators': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_submission.xlsx',
    },
}


def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def plot_slope_distribution(slopes, signs, save_path=None, title='Per-neuron social-axis slopes'):
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = ~np.isnan(slopes)
    s = slopes[valid]
    si = signs[valid]
    bins = np.linspace(np.nanmin(s), np.nanmax(s), 40) if s.size else np.linspace(-1, 1, 20)
    ax.hist(s[si > 0], bins=bins, color='#d62728', alpha=0.75,
            label=f'positive slope (n={int((si > 0).sum())})', edgecolor='k')
    ax.hist(s[si < 0], bins=bins, color='#1f77b4', alpha=0.75,
            label=f'negative slope (n={int((si < 0).sum())})', edgecolor='k')
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('OLS slope: firing rate vs social axis score')
    ax.set_ylabel('# neurons')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_variants_comparison(comparisons, n_neurons_info, group_colors, title, save_path=None):
    variants = list(comparisons.keys())
    n_var = len(variants)
    if n_var == 0:
        return None
    fig, axes = plt.subplots(1, n_var, figsize=(8 * n_var, 5.5), sharey=True)
    if n_var == 1:
        axes = [axes]
    for ax, var in zip(axes, variants):
        by_social = comparisons[var]
        mat_names = sorted(by_social.keys())
        all_groups = sorted({g for sr in by_social.values() for g in sr.keys()})
        n_g = len(all_groups)
        bar_w = 0.8 / max(n_g, 1)
        x = np.arange(len(mat_names))
        for gi, group in enumerate(all_groups):
            rhos, stars = [], []
            for mat in mat_names:
                e = by_social[mat].get(group, {})
                rhos.append(e.get('rho', np.nan))
                stars.append(_stars(e.get('p_val')))
            rhos = np.asarray(rhos)
            offset = (gi - (n_g - 1) / 2) * bar_w
            color = group_colors.get(group, 'gray')
            ax.bar(x + offset, rhos, bar_w, color=color, edgecolor='k', alpha=0.85, label=group)
            for i, s in enumerate(stars):
                if not np.isnan(rhos[i]) and s:
                    y_off = 0.01 if rhos[i] >= 0 else -0.04
                    ax.text(x[i] + offset, rhos[i] + y_off, s, ha='center', fontsize=9, fontweight='bold')
        ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=8)
        n_info = n_neurons_info.get(var, '')
        ax.set_title(f"{var}  ({n_info})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Spearman rho', fontsize=11)
    axes[-1].legend(fontsize=9, loc='best')
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def main():
    # ── CONFIG ──────────────────────────────────────────
    REGION = 'ER'
    MODE = 'both'
    SIGN_AXIS = 'submission'
    AXIS_AGGREGATE = 'column_sum'
    N_PERMUTATIONS = 2000
    SEED = 42
    # ────────────────────────────────────────────────────

    cfg = SocialRSAConfig(
        region=REGION,
        session=None,                          # None = pseudo-population
        window=(0.400, 0.700),
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=[],
        exclude_groups=['Stranger Things'],
        normalization=None,
        transform_social_behavior=None,        # None | 'rank' | 'log'
        exclude_identities=['7124', 'G942'],                 # e.g. ['7124'] to drop alpha
        n_permutations=N_PERMUTATIONS,
        save_plots=True,
        save_dir='rsa_signcorrect_results',
        rng_seed=SEED,
        visual_responsiveness_filter=False,
        vr_response_window=(0.400, 0.700),
        vr_alpha=0.15,
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)

    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    df = filter_visually_responsive(df, cfg)

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    print(f"\n{'█'*70}")
    print(f"█  SIGN-CORRECTED RSA (corrected permutation)")
    print(f"█  region={cfg.region}, sign axis={SIGN_AXIS} ({AXIS_AGGREGATE})")
    print(f"█  mode={MODE}, window={cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"█  permutations={N_PERMUTATIONS}")
    print(f"█  visual_responsive_filter={cfg.visual_responsiveness_filter}")
    print(f"{'█'*70}")

    result = run_sign_corrected_pseudopop(
        df, info_df, interactions, cfg,
        mode=MODE,
        sign_axis=SIGN_AXIS,
        axis_aggregate=AXIS_AGGREGATE,
        n_permutations=N_PERMUTATIONS,
        rng_seed=SEED)

    print_sign_corrected_results(result['comparisons'])

    win_start = int(cfg.window[0] * 1000)
    win_end   = int(cfg.window[1] * 1000)
    tx_tag    = cfg.transform_social_behavior or 'raw'
    save_root = f"{cfg.save_dir}/{cfg.region}_{win_start}_{win_end}_{tx_tag}"
    plot_slope_distribution(
        result['slopes'], result['signs'],
        title=f"Sign-corrected RSA — {SIGN_AXIS}-axis slopes "
              f"(region={cfg.region}, n={result['n_neurons']})",
        save_path=f"{save_root}/slope_distribution_{SIGN_AXIS}.png")

    n_info = {
        'flipped': f"n={result['n_neurons']}",
        'pos': f"n={result['n_neurons_pos']}",
        'neg': f"n={result['n_neurons_neg']}",
    }
    plot_variants_comparison(
        result['comparisons'], n_info, cfg.group_colors,
        title=f"Sign-corrected RSA (corrected perm) — {SIGN_AXIS} axis, "
              f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms",
        save_path=f"{save_root}/variants_{SIGN_AXIS}.png")

    plt.show()


if __name__ == '__main__':
    main()
