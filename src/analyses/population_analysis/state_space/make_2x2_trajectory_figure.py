# make_2x2_trajectory_figure.py
"""
Generate a single 2x2 figure of state-space trajectories:
    rows    = methods (group-level pooling, identity-level averaging)
    columns = regions (AMG, ER)

Reuses the existing pipeline (load_and_filter -> build_matrix_by_condition ->
preprocess -> run_pca -> compute_trajectories). Each subplot plots group-mean
2D trajectories; legend is shared across all four panels.

Run this *in place of* run_trajectory_by_condition.py when you want the
combined figure. No need to modify plotting.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyses.social_rank_analysis import load_group_matrices, davids_score
from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from analyses.population_analysis.state_space.preprocessing import preprocess
from pca_runner import run_pca
from plotting import compute_trajectories

# Reuse condition_map builder from your existing script
from run_trajectory_by_condition import build_condition_map

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"

GRANT_LABELS = {
    'Best Frans':      'Best Frans (Neighbor)',
    'Zombies':         'Zombies (Home)',
    'Stranger Things': 'Stranger Things (Unfamiliar)',
    'Instigators':     'Instigators (Unfamiliar)',
}

# Order legend / colors consistently across panels
LEGEND_ORDER = [
    'Zombies (Home)',
    'Best Frans (Neighbor)',
    'Instigators (Unfamiliar)',
    'Stranger Things (Unfamiliar)',
]


def run_one(region, analysis, mean_center=True, soft_normalize=True,
            min_reps=5, n_components=6, bin_width=0.050, min_epoch=2.0):
    """Run the pipeline for one (region, analysis) combination.

    Returns
    -------
    group_trajs : dict {group_label: (n_bins, n_comp)}  with GRANT_LABELS applied
    var         : ndarray of explained-variance ratios
    cfg         : TrajectoryConfig used for this run
    """
    cfg = TrajectoryConfig(
        region=region, session=None, trial_averaged=True, peak_align=False,
        n_components=n_components, bin_width=bin_width,
        min_epoch_duration=min_epoch, analysis=analysis,
    )
    cfg.validate()

    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)

    if analysis == 'identity':
        monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
        common = sorted(set.intersection(*monkeys_per_session))
        name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
        known = [m for m in common if m in name_to_group]
        condition_map = {m: m for m in known}
        group_map = {m: name_to_group[m] for m in known}
        exclude_groups = []
        print(f"[{region}|{analysis}] {len(known)} identities")
    else:
        condition_map, exclude_groups = build_condition_map(analysis, info_df)
        group_map = None

    if exclude_groups:
        df = df[~df['MonkeyGroup'].isin(exclude_groups)]

    pca_matrix, row_meta_df, info = build_matrix_by_condition(
        df, cfg, condition_map, group_map=group_map, min_reps=min_reps)

    pca_matrix = preprocess(pca_matrix, info, cfg,
                            soft_normalize=soft_normalize,
                            mean_center=mean_center)
    pca_result = run_pca(pca_matrix, info, cfg)
    _, group_trajs, _ = compute_trajectories(
        pca_result, row_meta_df, info, cfg)

    # Apply GRANT_LABELS
    group_trajs_grant = {GRANT_LABELS.get(g, g): t for g, t in group_trajs.items()}
    cfg.group_colors = {GRANT_LABELS.get(k, k): v
                        for k, v in cfg.group_colors.items()}

    return group_trajs_grant, pca_result['var'], cfg


def _draw_panel(ax, group_trajs, cfg):
    """Plot 2D group-mean trajectories onto a single Axes."""
    for g in LEGEND_ORDER:
        if g not in group_trajs:
            continue
        traj = group_trajs[g]
        color = cfg.group_colors.get(g, 'gray')
        ax.plot(traj[:, 0], traj[:, 1],
                color=color, linewidth=2.5, label=g)
        ax.scatter(traj[0, 0], traj[0, 1],
                   color=color, s=80, marker='o', edgecolors='black', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=color, s=80, marker='s', edgecolors='black', zorder=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')


def make_2x2(save_path=None):
    # Bump font sizes so they survive being placed in a Word/PowerPoint doc
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })

    # Run all four combinations
    panels = {}
    for region in ['AMG', 'ER']:
        for analysis, method_label in [('group', 'Group-level pooling'),
                                        ('identity', 'Identity-level averaging')]:
            print(f"\n=== Running {region} | {method_label} ===")
            group_trajs, var, cfg = run_one(region=region, analysis=analysis)
            panels[(method_label, region)] = (group_trajs, cfg)

    # Build the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),constrained_layout=True)
    method_rows = ['Group-level pooling', 'Identity-level averaging']
    region_cols = ['AMG', 'ER']

    for r, method in enumerate(method_rows):
        for c, region in enumerate(region_cols):
            group_trajs, cfg = panels[(method, region)]
            ax = axes[r, c]
            _draw_panel(ax, group_trajs, cfg)

    # Single shared legend on the right
    handles = []
    for g in LEGEND_ORDER:
        for (_, _), (gt, cfg_) in panels.items():
            if g in gt:
                color = cfg_.group_colors.get(g, 'gray')
                handles.append(Line2D([0], [0], color=color, linewidth=3, label=g))
                break

    start_handle = Line2D([0], [0], marker='o', color='gray', linestyle='None',
                          markersize=9, markeredgecolor='black', label='Start')
    end_handle = Line2D([0], [0], marker='s', color='gray', linestyle='None',
                        markersize=9, markeredgecolor='black', label='End')

    fig.legend(
        handles=handles + [start_handle, end_handle],
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # right side of figure
        frameon=False,
        ncol=1
    )
    # fig.text(0.30, 0.96, "AMG",
    #          ha="center", va="center",
    #          fontsize=16, fontweight="bold")
    #
    # fig.text(0.70, 0.96, "ER",
    #          ha="center", va="center",
    #          fontsize=16, fontweight="bold")
    #
    # fig.text(0.02, 0.73, "Group-level\npooling",
    #          rotation=90, ha="left", va="center",
    #          fontsize=15, fontweight="bold")
    #
    # fig.text(0.02, 0.28, "Identity-level\naveraging",
    #          rotation=90, ha="left", va="center",
    #          fontsize=15, fontweight="bold")
    fig.subplots_adjust(
        left=0.12,
        right=0.82,
        top=0.92,
        bottom=0.08,
        wspace=0.18,
        hspace=0.18,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    plt.show()
    return fig


if __name__ == '__main__':
    SAVE = '/home/connorlab/Documents/JulieData/Cortana/analysis_results/state_space_trajectory/progress_report_Jun2026/figure1_2x2.png'
    make_2x2(save_path=SAVE)
