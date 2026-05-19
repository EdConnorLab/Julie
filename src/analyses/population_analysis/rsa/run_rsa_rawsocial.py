# run_rsa_rawsocial.py
"""
Raw-social-matrix RSA runner.

Builds two RDM families that DO NOT symmetrize the social matrix:
  - signed asymmetry (pair-level M[i,j] - M[j,i])
  - concat[row, col] per-monkey vector (correlation distance)

Both preserve directed structure. The signed-asymmetry RDM is
particularly rank-correlated (dominant→subordinate flow), so by
default we report BOTH raw and rank-partialed versions side by side.

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
    compare_neural_to_social_by_group,
    build_rank_distance_matrix,
    print_group_comparisons,
)
from rsa_raw_social import build_raw_social_rdms


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


def plot_rho_per_group_per_matrix(group_comp, group_colors,
                                   title, save_path=None):
    """
    Grouped bar chart: bars colored by group, one bar group per matrix.
    Stars on top show p-values from compare_neural_to_social_by_group.
    """
    mat_names = list(group_comp.keys())
    all_groups = sorted({g for by_g in group_comp.values() for g in by_g.keys()})
    n_g = len(all_groups)

    fig, ax = plt.subplots(figsize=(max(10, len(mat_names) * 1.8), 5.5))
    bar_w = 0.8 / max(n_g, 1)
    x = np.arange(len(mat_names))

    for gi, group in enumerate(all_groups):
        rhos, stars, ns = [], [], []
        for mat in mat_names:
            e = group_comp[mat].get(group, {})
            rho = e.get('rho', np.nan)
            rhos.append(rho)
            stars.append(_stars(e.get('p_val')))
            ns.append(e.get('n_pairs', 0))
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
            if not np.isnan(rhos[i]):
                ax.text(x[i] + offset, -0.02, f'n={ns[i]}',
                        ha='center', fontsize=6, color='gray', va='top')

    ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mat_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_versions_side_by_side(comps_by_version, group_colors,
                                title, save_path=None):
    """
    Side-by-side panels: one per version (raw vs rank-partialed), each
    showing the per-group bar chart. Shared y-axis to make differences
    in ρ across versions visually obvious.
    """
    versions = list(comps_by_version.keys())
    n_ver = len(versions)
    if n_ver == 0:
        return None

    fig, axes = plt.subplots(1, n_ver, figsize=(8 * n_ver, 5.5), sharey=True)
    if n_ver == 1:
        axes = [axes]

    for ax, ver in zip(axes, versions):
        group_comp = comps_by_version[ver]
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
        ax.set_title(ver, fontsize=11)
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
    REGION = 'ALL'
    RUN_RAW_NO_PARTIAL = True
    RUN_RANK_PARTIALED = True
    TRANSFORM_SOCIAL_BEHAVIOR = None       # None | 'rank' | 'log'
    N_PERMUTATIONS = 1000
    N_BOOTSTRAP = 1000
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
        partial_out_rank=False,            # set per run below
        transform_social_behavior=TRANSFORM_SOCIAL_BEHAVIOR,
        n_permutations=N_PERMUTATIONS,
        between_group_permutations=0,
        n_bootstrap=N_BOOTSTRAP,
        save_plots=True,
        save_dir='rsa_rawsocial_results',
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
    identities = pp_result['identities']
    neural_rdm = pp_result['neural_rdm']

    print(f"\n{'█'*70}")
    print(f"█  RAW SOCIAL RSA  —  region={cfg.region}")
    print(f"█  signed asymmetry  +  concat[row,col] (correlation distance)")
    print(f"{'█'*70}")

    tx = cfg.transform_social_behavior
    raw_rdms = build_raw_social_rdms(
        identities, interactions,
        behavior_types=('affiliation', 'agonism', 'submission'),
        log_transform=(tx == 'log'),
        rank_transform=(tx == 'rank'),
        concat_metric='correlation',
        include_combined=True)

    versions = []
    if RUN_RAW_NO_PARTIAL: versions.append(('raw_no_partial',  False))
    if RUN_RANK_PARTIALED: versions.append(('rank_partialed',  True))

    rank_confound_mat = build_rank_distance_matrix(identities, info_df)
    win_start = int(cfg.window[0] * 1000)
    win_end   = int(cfg.window[1] * 1000)
    tx_tag    = tx or 'raw'
    save_root = f"{cfg.save_dir}/{cfg.region}_{win_start}_{win_end}_{tx_tag}"
    comps_by_version = {}

    for tag, do_partial in versions:
        confound = rank_confound_mat if do_partial else None
        partial_str = "(rank partialed)" if do_partial else "(no partial)"
        print(f"\n  --- {tag} {partial_str} ---")
        group_comp = compare_neural_to_social_by_group(
            neural_rdm, raw_rdms, identities, info_df,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
            rng_seed=cfg.rng_seed,
            confound_matrix=confound)
        print_group_comparisons(group_comp)
        comps_by_version[tag] = group_comp

        plot_rho_per_group_per_matrix(
            group_comp, cfg.group_colors,
            title=f"Raw social RSA — {tag} {partial_str}",
            save_path=f"{save_root}/per_group_{tag}.png")

    if len(comps_by_version) > 1:
        plot_versions_side_by_side(
            comps_by_version, cfg.group_colors,
            title=f"Raw social RSA — comparison",
            save_path=f"{save_root}/comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
