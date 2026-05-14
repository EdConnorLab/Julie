# run_rsa_signcorrect.py
"""
Sign-corrected RSA runner (AMG-focused).

The sign-correction motivation is specific to amygdala: an opponent
population where roughly half the neurons code submission-axis with
positive slope and half with negative slope cancel out in a naive
population RDM. We use a trial split-half to estimate per-neuron sign
on half A, then build the population RDM on half B with negatives
flipped (or split into pos/neg subpopulations).

Defaults: REGION='AMG', PARTIAL_OUT_RANK=False (the sign axis
correlates with rank, so partialing rank would partly undo the very
axis used for flipping).

Edit the CONFIG block at the top of main() and run from PyCharm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from rsa_config import SocialRSAConfig
from rsa_social import (
    load_all_interaction_matrices,
    build_social_rdms,
    compare_neural_to_social_by_group,
    build_rank_distance_matrix,
    print_group_comparisons,
)
from rsa_sign_corrected import run_sign_corrected_pseudopop


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
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def plot_slope_distribution(slopes, signs, save_path=None,
                             title='Per-neuron submission-axis slopes'):
    """
    Histogram of OLS slopes across neurons, colored by assigned sign.
    Visualizes how balanced the opponent population is.
    """
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
    ax.set_xlabel('OLS slope: firing rate vs within-group rank of submission-received')
    ax.set_ylabel('# neurons')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_variants_side_by_side(comps_by_variant, n_neurons_by_variant,
                                group_colors, title, save_path=None):
    """
    Side-by-side panels (one per RDM variant: flipped / pos / neg),
    each showing per-group ρ across matrices. Shared y-axis.
    """
    variants = list(comps_by_variant.keys())
    n_var = len(variants)
    if n_var == 0:
        return None

    fig, axes = plt.subplots(1, n_var, figsize=(8 * n_var, 5.5), sharey=True)
    if n_var == 1:
        axes = [axes]

    for ax, var in zip(axes, variants):
        group_comp = comps_by_variant[var]
        mat_names = list(group_comp.keys())
        all_groups = sorted({g for by_g in group_comp.values() for g in by_g.keys()})
        n_g = len(all_groups)
        bar_w = 0.8 / max(n_g, 1)
        x = np.arange(len(mat_names))

        for gi, group in enumerate(all_groups):
            rhos, stars = [], []
            for mat in mat_names:
                e = group_comp[mat].get(group, {})
                rhos.append(e.get('rho', np.nan))
                stars.append(_stars(e.get('p_val')))
            rhos = np.asarray(rhos)
            offset = (gi - (n_g - 1) / 2) * bar_w
            color = group_colors.get(group, 'gray')
            ax.bar(x + offset, rhos, bar_w, color=color, edgecolor='k',
                   alpha=0.85, label=group)
            for i, s in enumerate(stars):
                if not np.isnan(rhos[i]) and s:
                    y_off = 0.01 if rhos[i] >= 0 else -0.04
                    ax.text(x[i] + offset, rhos[i] + y_off, s,
                            ha='center', fontsize=9, fontweight='bold')

        ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=8)
        n_used = n_neurons_by_variant.get(var, 0)
        ax.set_title(f"{var}  (n_neurons={n_used})", fontsize=11)
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Spearman ρ', fontsize=11)
    axes[-1].legend(fontsize=9, loc='best')
    fig.suptitle(title, fontsize=12, y=1.02)
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
    REGION = 'AMG'
    MODE = 'both'                # 'flip' | 'subpop' | 'both'
    SIGN_AXIS = 'submission'
    AXIS_AGGREGATE = 'column_sum'   # 'column_sum' (received) or 'row_sum' (given)
    PARTIAL_OUT_RANK = False        # default OFF — see module docstring
    N_PERMUTATIONS = 1000
    SEED = 42
    # ────────────────────────────────────────────────────

    cfg = SocialRSAConfig(
        region=REGION,
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=[],
        exclude_groups=['Stranger Things'],
        normalization=None,
        partial_out_rank=PARTIAL_OUT_RANK,
        rank_transform_behavior=False,
        n_permutations=N_PERMUTATIONS,
        between_group_permutations=0,
        n_bootstrap=0,
        pseudo_population=True,
        save_plots=True,
        save_dir='rsa_signcorrect_results',
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

    print(f"\n{'█'*70}")
    print(f"█  SIGN-CORRECTED RSA  —  region={cfg.region}")
    print(f"█  sign axis : {SIGN_AXIS} ({AXIS_AGGREGATE})")
    print(f"█  mode      : {MODE}")
    print(f"█  partial rank: {cfg.partial_out_rank}")
    print(f"{'█'*70}")

    sc_result = run_sign_corrected_pseudopop(
        df, info_df, interactions, cfg,
        mode=MODE,
        sign_axis=SIGN_AXIS,
        axis_aggregate=AXIS_AGGREGATE,
        rng_seed=cfg.rng_seed)

    identities = sc_result['identities']

    social_rdms = build_social_rdms(
        identities, interactions,
        behavior_types=['affiliation', 'agonism', 'submission'],
        symmetrize=False, profile_metric='correlation',
        log_transform=False, include_combined=True,
        rank_transform=cfg.rank_transform_behavior)

    rank_confound = None
    if cfg.partial_out_rank:
        rank_confound = build_rank_distance_matrix(identities, info_df)

    save_root = f"{cfg.save_dir}/{cfg.region}"
    comps_by_variant = {}
    n_neurons_by_variant = {}

    for tag, rdm_key, n_key in [
        ('flipped_combined', 'neural_rdm_flipped', 'n_neurons'),
        ('subpop_pos_only',  'neural_rdm_pos',     'n_neurons_pos'),
        ('subpop_neg_only',  'neural_rdm_neg',     'n_neurons_neg'),
    ]:
        rdm = sc_result.get(rdm_key)
        if rdm is None:
            print(f"\n  [{tag}] no RDM (not enough neurons)")
            continue
        print(f"\n  --- {tag} ---")
        group_comp = compare_neural_to_social_by_group(
            rdm, social_rdms, identities, info_df,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
            rng_seed=cfg.rng_seed,
            confound_matrix=rank_confound)
        print_group_comparisons(group_comp)
        comps_by_variant[tag] = group_comp
        n_neurons_by_variant[tag] = sc_result.get(n_key, 0)

    # ── Summary figures ──
    plot_slope_distribution(
        sc_result['slopes'], sc_result['signs'],
        title=f"Sign-corrected RSA — submission-axis slopes "
              f"(region={cfg.region}, n={sc_result['n_neurons']})",
        save_path=f"{save_root}/slope_distribution.png")

    if comps_by_variant:
        plot_variants_side_by_side(
            comps_by_variant, n_neurons_by_variant, cfg.group_colors,
            title=f"Sign-corrected RSA — neural RDM variants  (region={cfg.region})",
            save_path=f"{save_root}/variants_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
