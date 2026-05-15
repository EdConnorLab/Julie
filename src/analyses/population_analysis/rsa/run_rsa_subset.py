# run_rsa_subset.py
"""
Subset / group-size-control RSA runner.

Caps every group at TARGET_SIZE monkeys (default 5 — matches Best Frans)
and runs the dissimilarity-mode social RSA N_DRAWS times with random
subsamples of the larger groups. Each draw runs an inner permutation
test; per-group ρ and between-group Δρ are aggregated across draws
with p-values derived from pooled nulls.

Edit the CONFIG block at the top of main() and run from PyCharm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from rsa_config import SocialRSAConfig
from rsa_core import run_rsa_pseudopop
from rsa_social import (
    load_all_interaction_matrices,
    build_rank_distance_matrix,
)
from rsa_subset import (
    run_subsampled_dissimilarity_rsa,
    print_subsample_summary,
    print_subsample_between_groups,
)


# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────

def _stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def plot_per_group_dots(summary, group_colors, title, save_path=None):
    """
    Dot plot with 95% CI error bars: per-group mean ρ across matrices.

    X-axis: social matrices. Dots per group, horizontally jittered.
    Stars above each dot indicate p-value from pooled-null permutation.
    """
    pg = summary['per_group']
    mat_names = sorted(pg.keys())
    all_groups = sorted({g for by_g in pg.values() for g in by_g.keys()})
    n_groups = len(all_groups)

    fig, ax = plt.subplots(figsize=(max(10, len(mat_names) * 2.0), 5.5))
    jitter = 0.7 / max(n_groups, 1)
    x = np.arange(len(mat_names))

    for gi, group in enumerate(all_groups):
        means, lo_err, hi_err, stars = [], [], [], []
        for mat in mat_names:
            e = pg[mat].get(group, {})
            m = e.get('mean', np.nan)
            ci_lo = e.get('ci_low', np.nan)
            ci_hi = e.get('ci_high', np.nan)
            p = e.get('p_val')
            means.append(m)
            lo_err.append(m - ci_lo if not np.isnan(m) else np.nan)
            hi_err.append(ci_hi - m if not np.isnan(m) else np.nan)
            stars.append(_stars(p))

        means = np.asarray(means)
        # ── FIX: clamp error bars to be non-negative ──
        lo_err = np.maximum(0.0, np.asarray(lo_err))
        hi_err = np.maximum(0.0, np.asarray(hi_err))
        # ──────────────────────────────────────────────

        offset = (gi - (n_groups - 1) / 2) * jitter
        xpos = x + offset
        valid = ~np.isnan(means)

        color = group_colors.get(group, 'gray')
        ax.errorbar(xpos[valid], means[valid],
                    yerr=[lo_err[valid], hi_err[valid]],
                    fmt='o', color=color, markersize=8,
                    markeredgecolor='k', markeredgewidth=0.5,
                    capsize=4, linewidth=1.5,
                    label=group, zorder=3)
        for i, s in enumerate(stars):
            if valid[i] and s:
                y_star = means[i] + hi_err[i] + 0.02
                ax.text(xpos[i], y_star, s,
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color=color)

    ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ  (mean across draws, 95% CI)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_between_group_deltas(summary, title, save_path=None):
    """
    Grouped bar chart of between-group Δρ across matrices.

    One bar group per matrix, one bar per group-pair, colored by pair.
    Stars indicate p-value.
    """
    bw = summary['between']
    mat_names = sorted(bw.keys())
    all_pairs = sorted({p for by_p in bw.values() for p in by_p.keys()})
    n_pairs = len(all_pairs)

    pair_palette = ['#7570b3', '#1b9e77', '#d95f02', '#e7298a']

    fig, ax = plt.subplots(figsize=(max(10, len(mat_names) * 2.0), 5.5))
    bar_w = 0.8 / max(n_pairs, 1)
    x = np.arange(len(mat_names))

    for pi, pair in enumerate(all_pairs):
        deltas, stars = [], []
        for mat in mat_names:
            e = bw[mat].get(pair, {})
            deltas.append(e.get('mean', np.nan))
            stars.append(_stars(e.get('p_val')))
        deltas = np.asarray(deltas)
        offset = (pi - (n_pairs - 1) / 2) * bar_w
        ax.bar(x + offset, deltas, bar_w,
               color=pair_palette[pi % len(pair_palette)],
               edgecolor='k', alpha=0.85,
               label=f"{pair[0]} − {pair[1]}")
        for i, s in enumerate(stars):
            if not np.isnan(deltas[i]) and s:
                y_off = 0.01 if deltas[i] >= 0 else -0.04
                ax.text(x[i] + offset, deltas[i] + y_off, s,
                        ha='center', fontsize=9, fontweight='bold')

    ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Δρ  (mean across draws)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    # ── CONFIG (edit here) ──────────────────────────────
    REGION = 'ALL'
    TARGET_SIZE = 5
    N_DRAWS = 200
    PERMS_PER_DRAW = 200
    PARTIAL_OUT_RANK = True
    RUN_ASYMMETRIC = True
    RUN_SYMMETRIZED = True
    SEED = 42
    # ────────────────────────────────────────────────────

    cfg = SocialRSAConfig(
        region=REGION,
        session=None,                          # None = pseudo-population
        window=(0.300, 0.600),
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=[],
        exclude_groups=['Stranger Things'],
        normalization=None,
        partial_out_rank=PARTIAL_OUT_RANK,
        rank_transform_behavior=False,
        n_permutations=0,
        between_group_permutations=0,
        n_bootstrap=0,
        save_plots=True,
        save_dir='rsa_subset_results',
        rng_seed=SEED,
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    # Visual responsiveness filter (toggle via cfg.visual_responsiveness_filter)
    df = filter_visually_responsive(df, cfg)

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    print("\n--- Building standard pseudo-population result ---")
    pp_result = run_rsa_pseudopop(df, info_df, cfg)

    print(f"\n{'█'*70}")
    print(f"█  SUBSET RSA  —  region={cfg.region}, target_size={TARGET_SIZE}")
    print(f"█  n_draws={N_DRAWS}, perms/draw={PERMS_PER_DRAW}, "
          f"partial rank: {cfg.partial_out_rank}")
    print(f"{'█'*70}")

    conditions = []
    if RUN_ASYMMETRIC:  conditions.append((False, 'asymmetric'))
    if RUN_SYMMETRIZED: conditions.append((True,  'symmetrized'))

    confound_fn = None
    if cfg.partial_out_rank:
        confound_fn = lambda ids, info: build_rank_distance_matrix(ids, info)

    save_root = f"{cfg.save_dir}/{cfg.region}"

    for symmetrize, label in conditions:
        print(f"\n  --- Condition: {label} ---")
        summary = run_subsampled_dissimilarity_rsa(
            pp_result, info_df, interactions, cfg,
            target_size=TARGET_SIZE,
            n_draws=N_DRAWS,
            n_permutations_per_draw=PERMS_PER_DRAW,
            rng_seed=cfg.rng_seed,
            symmetrize=symmetrize, log_transform=False,
            behavior_types=('affiliation', 'agonism', 'submission'),
            confound_matrix_fn=confound_fn,
            ci_level=0.95)

        print_subsample_summary(
            summary, title=f"Per-group ρ  (subsampled, {label})")
        print_subsample_between_groups(
            summary, title=f"Between-group Δρ  (subsampled, {label})")

        plot_per_group_dots(
            summary, cfg.group_colors,
            title=f"Subset RSA per-group ρ — {label} "
                  f"(n={TARGET_SIZE}, {N_DRAWS} draws)",
            save_path=f"{save_root}/per_group_{label}.png")
        plot_between_group_deltas(
            summary,
            title=f"Subset RSA between-group Δρ — {label}",
            save_path=f"{save_root}/between_group_{label}.png")

    plt.show()


if __name__ == '__main__':
    main()
