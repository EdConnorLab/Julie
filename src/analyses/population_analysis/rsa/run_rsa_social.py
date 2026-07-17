# run_rsa_social.py
"""
RSA analysis comparing neural dissimilarity (1-r) to social behavioral profile RDMs.

Per-group Spearman ρ, optional bootstrap 95% CI, and optional between-group Δρ test.
Runs two symmetry conditions (asymmetric / symmetrized) and saves flat plots to:
  {save_dir}/{region}_{win_start}_{win_end}_{rank_tag}/asym_*.png  sym_*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from rsa_config import SocialRSAConfig
from rsa_core import run_rsa_session, run_rsa_pseudopop
from rsa_social import (
    load_all_interaction_matrices,
    build_neural_similarity_matrix,
    build_social_rdms,
    build_dominance_interactions,
    compare_neural_to_social_by_group,
    compare_between_groups,
    print_group_comparisons,
    print_between_group_comparisons,
    plot_scatter_multi,
    build_rank_distance_matrix,
)
from rsa_plotting import plot_neural_rdm

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

def _p_to_stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def plot_social_matrix(mat, identities, info_df, title, group_colors,
                        cbar_label='Value', save_path=None):
    """Plot a social matrix (similarity or dissimilarity) with group-colored labels."""
    fig, ax = plt.subplots(figsize=(10, 9))

    info = info_df.set_index(info_df['Name'].astype(str))
    groups = [info.loc[m, 'Group Name'] if m in info.index else 'Unknown'
              for m in identities]

    sort_idx = np.array(sorted(range(len(identities)),
                               key=lambda i: (groups[i], identities[i])))
    mat_sorted = mat[np.ix_(sort_idx, sort_idx)]
    ids_sorted = [identities[i] for i in sort_idx]
    groups_sorted = [groups[i] for i in sort_idx]

    masked_mat = np.ma.masked_where(np.isnan(mat_sorted), mat_sorted)
    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(masked_mat, cmap=cmap, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)

    ax.set_xticks(range(len(ids_sorted)))
    ax.set_yticks(range(len(ids_sorted)))
    ax.set_xticklabels(ids_sorted, rotation=90, fontsize=7)
    ax.set_yticklabels(ids_sorted, fontsize=7)

    for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        g = groups_sorted[i]
        if g in group_colors:
            tx.set_color(group_colors[g])
            ty.set_color(group_colors[g])

    boundaries = []
    prev = groups_sorted[0]
    for i, g in enumerate(groups_sorted):
        if g != prev:
            boundaries.append(i - 0.5)
            prev = g
    for b in boundaries:
        ax.axhline(b, color='k', lw=1, alpha=0.7)
        ax.axvline(b, color='k', lw=1, alpha=0.7)

    handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()
               if g in set(groups_sorted)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=8)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_rsa_bar(comparisons, title, save_path=None):
    """Bar chart of RSA rho values."""
    names = list(comparisons.keys())
    rhos = [comparisons[n]['rho'] for n in names]
    n_pairs = [comparisons[n]['n_pairs'] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    colors = []
    for name in names:
        if 'combined' in name:
            colors.append('#ff7f0e')
        elif 'affiliation' in name:
            colors.append('#2ca02c')
        elif 'agonism' in name:
            colors.append('#d62728')
        elif 'submission' in name:
            colors.append('#1f77b4')
        else:
            colors.append('steelblue')

    ax.bar(range(len(names)), rhos, color=colors, edgecolor='k', alpha=0.8)

    for i, name in enumerate(names):
        p = comparisons[name].get('p_val')
        star = _p_to_stars(p)
        y_offset = 0.01 if rhos[i] >= 0 else -0.03
        ax.text(i, rhos[i] + y_offset, star, ha='center', fontsize=10)
        ax.text(i, -0.02, f'n={n_pairs[i]}', ha='center', fontsize=7,
                color='gray', va='top')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Spearman ρ')
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_rsa_by_group(group_comparisons, group_colors, title, save_path=None):
    """Grouped bar chart: bars colored by group."""
    rdm_names = list(group_comparisons.keys())
    all_groups = set()
    for gc in group_comparisons.values():
        all_groups.update(gc.keys())
    all_groups = sorted(all_groups)
    n_groups = len(all_groups)

    fig, ax = plt.subplots(figsize=(max(10, len(rdm_names) * 2.5), 5.5))

    bar_width = 0.8 / n_groups
    x = np.arange(len(rdm_names))

    for g_idx, group in enumerate(all_groups):
        rhos = []
        for rdm_name in rdm_names:
            gc = group_comparisons[rdm_name].get(group, {})
            rhos.append(gc.get('rho', np.nan))

        offset = (g_idx - (n_groups - 1) / 2) * bar_width
        color = group_colors.get(group, 'gray')
        ax.bar(x + offset, rhos, bar_width, label=group,
               color=color, edgecolor='k', alpha=0.8)

        for i, rdm_name in enumerate(rdm_names):
            gc = group_comparisons[rdm_name].get(group, {})
            p = gc.get('p_val')
            rho = gc.get('rho', np.nan)
            n = gc.get('n_pairs', 0)
            if not np.isnan(rho):
                star = _p_to_stars(p)
                y_offset = 0.01 if rho >= 0 else -0.03
                ax.text(x[i] + offset, rho + y_offset, star,
                        ha='center', fontsize=8)
                ax.text(x[i] + offset, -0.02, f'n={n}',
                        ha='center', fontsize=6, color='gray', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(rdm_names, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Spearman ρ')
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_rsa_ci(group_comparisons, group_colors, title, save_path=None):
    """
    Dot plot with bootstrap 95% CI error bars.

    X-axis: social matrices. Dots per group, horizontally jittered.
    Horizontal dashed line at ρ = 0. Stars for significance.
    """
    rdm_names = list(group_comparisons.keys())
    all_groups = set()
    for gc in group_comparisons.values():
        all_groups.update(gc.keys())
    all_groups = sorted(all_groups)
    n_groups = len(all_groups)

    fig, ax = plt.subplots(figsize=(max(10, len(rdm_names) * 2.2), 5.5))

    jitter_width = 0.7 / n_groups
    x = np.arange(len(rdm_names))

    for g_idx, group in enumerate(all_groups):
        rhos = []
        ci_lows = []
        ci_highs = []
        stars_list = []
        n_pairs_list = []

        for rdm_name in rdm_names:
            gc = group_comparisons[rdm_name].get(group, {})
            rho = gc.get('rho', np.nan)
            ci_lo = gc.get('ci_low', np.nan)
            ci_hi = gc.get('ci_high', np.nan)
            p = gc.get('p_val')
            n = gc.get('n_pairs', 0)
            rhos.append(rho)
            ci_lows.append(ci_lo)
            ci_highs.append(ci_hi)
            stars_list.append(_p_to_stars(p))
            n_pairs_list.append(n)

        rhos = np.array(rhos)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        offset = (g_idx - (n_groups - 1) / 2) * jitter_width
        color = group_colors.get(group, 'gray')
        xpos = x + offset

        # Error bars
        yerr_lo = rhos - ci_lows
        yerr_hi = ci_highs - rhos
        valid = ~np.isnan(rhos) & ~np.isnan(ci_lows)

        ax.errorbar(xpos[valid], rhos[valid],
                     yerr=[yerr_lo[valid], yerr_hi[valid]],
                     fmt='o', color=color, markeredgecolor='k',
                     markeredgewidth=0.5, markersize=7,
                     capsize=4, capthick=1.5, linewidth=1.5,
                     label=group, zorder=3)

        # Stars and n labels
        for i in range(len(rdm_names)):
            if not np.isnan(rhos[i]) and not np.isnan(ci_highs[i]):
                if stars_list[i].strip():
                    ax.text(xpos[i], ci_highs[i] + 0.02, stars_list[i],
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(rdm_names, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.set_title(title, fontsize=12)

    # Light gray grid
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_rdm_grid(neural_rdm, social_rdms, identities, info_df, group_colors,
                  title, save_path=None, group_comp=None):
    """
    Plot the neural RDM and every social RDM in one figure (shared identity
    ordering, sorted by group) so they can be compared side by side.
    Each panel keeps its own colour scale.

    If group_comp (output of compare_neural_to_social_by_group) is given, each
    social panel's title shows the RSA Spearman ρ against the neural RDM and its
    permutation significance (* p<.05, ** p<.01, *** p<.001). Panels significant
    at p<.05 are drawn in bold red so the significant RDMs stand out. Cleanest
    with a single group (one ρ per panel); with multiple groups the per-group ρ
    values are listed compactly.
    """
    def _stars(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    def _annot(name):
        """Return (subtitle, is_significant) for a social panel from group_comp."""
        if not group_comp or name not in group_comp:
            return '', False
        items = [(g, d) for g, d in group_comp[name].items()
                 if d and not (isinstance(d.get('rho'), float)
                               and np.isnan(d.get('rho', np.nan)))]
        if not items:
            return '', False
        if len(items) == 1:
            d = items[0][1]
            s = _stars(d.get('p_val'))
            return f"ρ = {d.get('rho', np.nan):+.2f} {s}".rstrip(), bool(s)
        parts, any_sig = [], False
        for g, d in items:
            s = _stars(d.get('p_val'))
            any_sig = any_sig or bool(s)
            parts.append(f"{g[:4]} {d.get('rho', np.nan):+.2f}{s}")
        return "  ".join(parts), any_sig

    info = info_df.set_index(info_df['Name'].astype(str))
    groups = [info.loc[m, 'Group Name'] if m in info.index else 'Unknown'
              for m in identities]
    sort_idx = np.array(sorted(range(len(identities)),
                               key=lambda i: (groups[i], identities[i])))
    ids_sorted = [identities[i] for i in sort_idx]
    groups_sorted = [groups[i] for i in sort_idx]

    panels = [('Neural RDM (1 - r)', neural_rdm)]
    panels += list(social_rdms.items())

    n_panels = len(panels)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.0))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color='lightgray')

    for ax_idx, (name, mat) in enumerate(panels):
        ax = axes[ax_idx]
        mat_sorted = mat[np.ix_(sort_idx, sort_idx)]
        masked = np.ma.masked_where(np.isnan(mat_sorted), mat_sorted)
        im = ax.imshow(masked, cmap=cmap, aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.7)
        sub, sig = _annot(name)
        ax.set_title(f"{name}\n{sub}" if sub else name, fontsize=9,
                     color='crimson' if sig else 'black',
                     fontweight='bold' if sig else 'normal')
        ax.set_xticks(range(len(ids_sorted)))
        ax.set_yticks(range(len(ids_sorted)))
        ax.set_xticklabels(ids_sorted, rotation=90, fontsize=5)
        ax.set_yticklabels(ids_sorted, fontsize=5)
        for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                          ax.get_yticklabels())):
            g = groups_sorted[i]
            if g in group_colors:
                tx.set_color(group_colors[g])
                ty.set_color(group_colors[g])
        prev = groups_sorted[0]
        for i, g in enumerate(groups_sorted):
            if g != prev:
                ax.axhline(i - 0.5, color='k', lw=0.8, alpha=0.6)
                ax.axvline(i - 0.5, color='k', lw=0.8, alpha=0.6)
                prev = g

    for ax_idx in range(n_panels, len(axes)):
        axes[ax_idx].axis('off')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_rho_heatmap(group_comp, group_colors, title, save_path=None):
    """
    Summary heatmap of RSA Spearman ρ: social RDMs (rows) × groups (columns).

    Cell colour = ρ (diverging, centred at 0); annotation = ρ and permutation
    stars (* p<.05, ** p<.01, *** p<.001); significant cells are outlined.
    Handles 1..N groups (designed for up to 3), so it's the multi-group
    companion to plot_rdm_grid — each group gets its own column instead of
    being crammed into a panel title.
    """
    def _stars(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    rdm_names = list(group_comp.keys())
    groups = sorted({g for gd in group_comp.values() for g in gd})
    if not rdm_names or not groups:
        return None

    R = np.full((len(rdm_names), len(groups)), np.nan)
    P = np.empty((len(rdm_names), len(groups)), dtype=object)
    for i, name in enumerate(rdm_names):
        for j, g in enumerate(groups):
            d = group_comp.get(name, {}).get(g)
            if d and not (isinstance(d.get('rho'), float)
                          and np.isnan(d.get('rho', np.nan))):
                R[i, j] = d['rho']
            P[i, j] = d.get('p_val') if d else None

    vmax = np.nanmax(np.abs(R)) if np.isfinite(R).any() else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(max(4.5, len(groups) * 1.7 + 2.0),
                                    max(5.0, len(rdm_names) * 0.5 + 1.0)))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(np.ma.masked_invalid(R), cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ (neural vs social)')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    for tick, g in zip(ax.get_xticklabels(), groups):
        tick.set_color(group_colors.get(g, 'black'))
    ax.set_yticks(range(len(rdm_names)))
    ax.set_yticklabels(rdm_names, fontsize=8)

    for i in range(len(rdm_names)):
        for j in range(len(groups)):
            if np.isnan(R[i, j]):
                continue
            s = _stars(P[i, j])
            txt = f"{R[i, j]:+.2f}\n{s}" if s else f"{R[i, j]:+.2f}"
            col = 'white' if abs(R[i, j]) > 0.6 * vmax else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color=col, fontweight='bold' if s else 'normal')
            if s:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='k', lw=2.0))

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────────
# Run one dissimilarity condition
# ──────────────────────────────────────────────────────────

def run_dissimilarity_condition(neural_rdm, identities, interactions, info_df, cfg,
                                 symmetrize, condition_label, save_dir,
                                 prefix='', confound_matrix=None):
    """
    Run dissimilarity-based analysis for one condition.
    Neural RDM (1-r) vs social behavioral profile RDMs.

    Transform applied to behavioral profiles is controlled by
    cfg.transform_social_behavior (None | 'rank' | 'log').

    Plots are saved flat into save_dir with filename prefix (e.g. 'asym_' or 'sym_').
    Always includes dominance RDMs (log1p(agonism) + log1p(submission.T)).
    """
    print(f"\n{'─'*50}")
    print(f"  DISSIMILARITY: {condition_label}")
    print(f"{'─'*50}")

    log_transform  = (cfg.transform_social_behavior == 'log')
    rank_transform = (cfg.transform_social_behavior == 'rank')

    # Standard behavior types
    social_rdms = build_social_rdms(
        identities, interactions,
        behavior_types=['affiliation', 'agonism', 'submission'],
        symmetrize=symmetrize, profile_metric='correlation',
        log_transform=log_transform, include_combined=True,
        rank_transform=rank_transform,
        exclude_ids=cfg.exclude_identities)

    # Dominance: log already baked in during construction, so log_transform=False
    interactions_with_dom = build_dominance_interactions(
        interactions, exclude_ids=cfg.exclude_identities)
    social_rdms_dominance = build_social_rdms(
        identities, interactions_with_dom,
        behavior_types=['dominance'],
        symmetrize=symmetrize, profile_metric='correlation',
        log_transform=False, include_combined=False,
        rank_transform=rank_transform,
        exclude_ids=None)   # already stripped inside build_dominance_interactions
    social_rdms.update(social_rdms_dominance)

    # Per-group
    print(f"\n  Per-group:")
    group_comp = compare_neural_to_social_by_group(
        neural_rdm, social_rdms, identities, info_df,
        n_permutations=cfg.n_permutations,
        n_bootstrap=cfg.n_bootstrap,
        rng_seed=cfg.rng_seed,
        confound_matrix=confound_matrix)
    print_group_comparisons(group_comp)

    # Between-group Δρ
    between_comp = None
    if cfg.between_group_permutations > 0:
        print(f"\n  Between-group Δρ ({cfg.between_group_permutations} permutations):")
        between_comp = compare_between_groups(
            neural_rdm, social_rdms, identities, info_df,
            n_permutations=cfg.between_group_permutations,
            rng_seed=cfg.rng_seed)
        print_between_group_comparisons(between_comp)

    # Plots
    if cfg.save_plots:
        # One combined figure: neural RDM + all social RDMs, same identity
        # ordering, for direct visual comparison (replaces the per-RDM PNGs).
        plot_rdm_grid(
            neural_rdm, social_rdms, identities, info_df, cfg.group_colors,
            title=f"Neural + Social RDMs ({condition_label})",
            group_comp=group_comp,
            save_path=f"{save_dir}/{prefix}all_rdms.png")

        # Multi-group summary: ρ heatmap (social RDMs × groups), stars + outlines
        plot_rho_heatmap(
            group_comp, cfg.group_colors,
            title=f"RSA ρ: social RDM × group ({condition_label})",
            save_path=f"{save_dir}/{prefix}rsa_rho_heatmap.png")

        # Scatter shows ρ and the permutation p-value already computed in
        # group_comp (compare_neural_to_social_by_group, cfg.n_permutations).
        plot_scatter_multi(
            neural_rdm, social_rdms, identities, info_df,
            cfg.group_colors,
            neural_label='Neural dissimilarity (1 - r)',
            social_label_prefix='Social dissimilarity',
            title=f"Neural vs Social dissimilarity ({condition_label})",
            group_comp=group_comp,
            save_path=f"{save_dir}/{prefix}scatter.png")

    return group_comp, between_comp, social_rdms

from datetime import datetime

def save_config_summary(cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    config_dict = {
        k: v
        for k, v in vars(cfg).items()
        if not callable(v) and not k.startswith("_")
    }

    with open(os.path.join(save_dir, "config_summary.txt"), "w") as f:
        f.write("RSA ANALYSIS CONFIG\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Region: {cfg.region}\n")
        f.write(f"Window: {cfg.window}\n")
        f.write(f"Mode: {'pseudo-population' if cfg.session is None else cfg.session}\n")
        f.write("\nFull config:\n")
        for key, value in sorted(vars(cfg).items()):
            f.write(f"{key}: {value}\n")

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    cfg = SocialRSAConfig(
        region='ER',
        session=None,                          # None = pseudo-population; 'session_id' = single session
        window=(0.300, 0.600),
        min_epoch_duration=1.3,
        min_reps_per_monkey=5,
        neural_metric='correlation',  # 'correlation', 'euclidean', 'cosine'
        model_factors=[],
        exclude_groups=['Stranger Things', 'Best Frans', 'Instigators'],
        normalization='soft',                   # None | 'soft' | 'zscore'
        transform_social_behavior='log',        # None | 'rank' | 'log'
        n_permutations=5000,
        between_group_permutations=2000,
        n_bootstrap=2000,
        save_plots=True,
        save_dir='rsa_social_visual_responsive',
        visual_responsiveness_filter=True,
        partial_out_rank=False,
        # exclude_identities=['7124','G942'],
        # ── Neuron filters (off by default) ──
        peak_latency_filter=False,
        peak_latency_range=(0.200, 0.700),
        peak_latency_search_window=(0.0, 1.00),
        # neuron_id_filter_pkl='/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pKW_passed.pkl',
        exclude_identities=[],
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

    # Optional neuron-level filters: pkl NeuronID whitelist and/or peak-latency filter
    df = apply_neuron_filters(df, cfg)

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    print(f"\n{'='*60}")
    print(f"Social Behavior RSA Analysis")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Normalization: {cfg.normalization}")
    print(f"  Neural metric: {cfg.neural_metric}")
    print(f"  Mode: {'pseudo-population' if cfg.session is None else f'single session ({cfg.session})'}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"  Between-group Δρ permutations: {cfg.between_group_permutations}")
    print(f"  Bootstrap CI: {cfg.n_bootstrap}")
    print(f"  Partial out rank: {cfg.partial_out_rank}")
    print(f"  Social behavior transform: {cfg.transform_social_behavior or 'raw'}")
    print(f"  Exclude groups: {cfg.exclude_groups}")
    print(f"  Exclude identities: {cfg.exclude_identities}")
    print(f"{'='*60}\n")

    if cfg.session is None:
        result = run_rsa_pseudopop(df, info_df, cfg)
        # pseudopop_raw_rate_matrix = result["raw_rate_matrix"]
        # pd.DataFrame(pseudopop_raw_rate_matrix).to_csv(f'{cfg.region}_pseudopop_raw_rate_matrix.csv', index=False)
        results_list = [result]
    else:
        r = run_rsa_session(df, cfg.session, info_df, cfg)
        results_list = [r] if r is not None else []

    for result in results_list:
        identities = result['identities']
        neural_rdm = result['neural_rdm']
        rate_matrix = result['rate_matrix']
        session_label = result['session']

        # ── Exclude specific identities (sensitivity analysis) ──
        if cfg.exclude_identities:
            keep_mask = [i for i, mid in enumerate(identities)
                         if mid not in cfg.exclude_identities]
            removed = [mid for mid in identities if mid in cfg.exclude_identities]
            identities  = [identities[i] for i in keep_mask]
            neural_rdm  = neural_rdm[np.ix_(keep_mask, keep_mask)]
            rate_matrix = rate_matrix[keep_mask, :]
            print(f"\n  [Sensitivity] Excluded: {removed}  →  {len(identities)} identities remaining")

        # Build rank distance confound matrix if requested
        rank_confound = None
        if cfg.partial_out_rank:
            rank_confound = build_rank_distance_matrix(identities, info_df)
            print(f"  Partial out rank: ON (|rank_i - rank_j| as confound)")

        win_start = int(cfg.window[0] * 1000)
        win_end   = int(cfg.window[1] * 1000)
        tx_tag    = cfg.transform_social_behavior or 'raw'
        save_dir_base = f"{cfg.save_dir}/{cfg.region}_{win_start}_{win_end}_beh_{tx_tag}_neur_{cfg.neural_metric}_{cfg.normalization}"
        save_config_summary(cfg, save_dir_base)

        print(f"\n{'='*60}")
        print(f"  Identities ({len(identities)}): {identities}")
        print(f"  Saving to: {save_dir_base}/")
        print(f"{'='*60}")

        # Neural reference plots (condition-independent, no prefix)
        if cfg.save_plots:
            plot_neural_rdm(result, cfg, save_dir=save_dir_base)

        # ── Dissimilarity RSA: neural 1-r vs social behavioral profile RDMs ──
        conditions = [
            (False, "asymmetric",  "asym_"),
            # (True,  "symmetrized", "sym_"),   # symmetric runs disabled for now
        ]

        dissim_results = {}
        for symmetrize, label, prefix in conditions:
            group_comp, between_comp, _ = run_dissimilarity_condition(
                neural_rdm, identities, interactions, info_df, cfg,
                symmetrize=symmetrize,
                condition_label=label, save_dir=save_dir_base, prefix=prefix,
                confound_matrix=rank_confound)
            dissim_results[label] = (group_comp, between_comp)

        # ══════════════════════════════════════════════════
        # Summary (per-group)
        # ══════════════════════════════════════════════════
        print(f"\n{'='*70}")
        print(f"SUMMARY (per-group): {session_label}")
        print(f"{'='*70}")

        cond_labels = [c[1] for c in conditions]  # label string is index 1 in (symmetrize, label, prefix)

        def _print_per_group_summary(mode_label, results_dict, cond_labels):
            """Print a per-group summary table for one mode (similarity or dissimilarity)."""
            print(f"\n  {mode_label}:")

            # Get groups and matrix names from first condition
            first_group_comp = list(results_dict.values())[0][0]
            mat_names = list(first_group_comp.keys())
            all_groups = set()
            for gc in first_group_comp.values():
                all_groups.update(gc.keys())
            all_groups = sorted(all_groups)

            # Check if CIs are available
            has_ci = False
            for label in cond_labels:
                gc = results_dict[label][0]
                for mat_data in gc.values():
                    for entry in mat_data.values():
                        if not np.isnan(entry.get('ci_low', np.nan)):
                            has_ci = True
                            break
                    if has_ci:
                        break
                if has_ci:
                    break

            col_w = 34 if has_ci else 20

            for group in all_groups:
                print(f"\n    {group}:")
                header = f"    {'Matrix':<28s}" + "".join(f"{c:>{col_w}s}" for c in cond_labels)
                print(header)
                print(f"    {'-'*28}" + "-" * col_w * len(cond_labels))

                for mat_name in mat_names:
                    row = f"    {mat_name:<28s}"
                    for label in cond_labels:
                        gc = results_dict[label][0]  # group_comp
                        entry = gc.get(mat_name, {}).get(group, {})
                        rho = entry.get('rho', np.nan)
                        p = entry.get('p_val')
                        ci_lo = entry.get('ci_low', np.nan)
                        ci_hi = entry.get('ci_high', np.nan)
                        if np.isnan(rho):
                            row += f"{'--':>{col_w}s}"
                        else:
                            star = _p_to_stars(p)
                            ci_str = f" [{ci_lo:+.2f},{ci_hi:+.2f}]" if not np.isnan(ci_lo) else ""
                            row += f"  {rho:>+.4f} {star:<5s}{ci_str}"
                    print(row)

            # Between-group Δρ (if computed)
            first_between = list(results_dict.values())[0][1]
            if first_between is not None:
                print(f"\n    Between-group Δρ:")
                all_pairs = set()
                for bc in first_between.values():
                    all_pairs.update(bc.keys())
                all_pairs = sorted(all_pairs)
                pair_labels = [f"{a}−{b}" for a, b in all_pairs]

                for pi, (pair, plabel) in enumerate(zip(all_pairs, pair_labels)):
                    print(f"\n    {plabel}:")
                    header = f"    {'Matrix':<28s}" + "".join(f"{c:>20s}" for c in cond_labels)
                    print(header)
                    print(f"    {'-'*28}" + "-" * 20 * len(cond_labels))

                    for mat_name in mat_names:
                        row = f"    {mat_name:<28s}"
                        for label in cond_labels:
                            bc = results_dict[label][1]  # between_comp
                            entry = bc.get(mat_name, {}).get(pair, {})
                            d = entry.get('delta_rho', np.nan)
                            p = entry.get('p_val')
                            if np.isnan(d):
                                row += f"{'--':>20s}"
                            else:
                                star = _p_to_stars(p)
                                row += f"  {d:>+.4f} {star:<12s}"
                        print(row)

        _print_per_group_summary(
            "DISSIMILARITY (neural 1-r vs social profile RDMs)",
            dissim_results, cond_labels)

        print(f"\n{'='*70}")

    plt.show()


if __name__ == '__main__':
    main()
