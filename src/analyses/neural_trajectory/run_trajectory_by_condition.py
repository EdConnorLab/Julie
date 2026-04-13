# run_trajectory_by_condition.py
"""
PCA with condition axis = {group | familiarity | sex | rank}.

Switch ANALYSIS below to choose the comparison. For binary comparisons
(familiarity, sex) consider setting MEAN_CENTER = False — with only 2
conditions, mean-centering forces the two trajectories into exact mirror
images through the origin (as a mathematical artifact)
"""
# import sys
# sys.path.insert(0, '/home/connorlab/Documents/GitHub/Julie/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.neural_trajectory.plotting import plot_pc_vs_time_all_shaded, plot_pc_vs_time_per_group
from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from preprocessing import preprocess
from pca_runner import run_pca
from plotting import (compute_trajectories,
                      plot_group_mean_3d_mpl, plot_group_mean_2d_mpl,
                      plot_pc_vs_time_group_mean,
                      plot_per_group_3d_mpl, plot_all_shaded_3d_mpl)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def build_condition_map(analysis, info_df):
    """
    Returns (condition_map, exclude_groups).

    condition_map : dict {MonkeyName: label}
    exclude_groups : list of MonkeyGroup values to drop from df before binning
    """
    names = info_df['Name'].astype(str)

    if analysis == 'group':
        m = dict(zip(names, info_df['Group Name']))
        return m, []

    if analysis == 'familiarity':
        fam = {'Zombies': 'familiar', 'Best Frans': 'familiar',
               'Stranger Things': 'unfamiliar', 'Instigators': 'unfamiliar'}
        m = {n: fam[g] for n, g in zip(names, info_df['Group Name']) if g in fam}
        return m, []

    if analysis == 'sex':
        m = {n: s for n, s in zip(names, info_df['Sex']) if pd.notna(s)}
        return m, []

    if analysis == 'rank':
        sub = info_df[info_df['Group Name'] != 'Stranger Things'].copy()
        sub = sub.dropna(subset=['Rank'])
        m = dict(zip(sub['Name'].astype(str),
                     'rank' + sub['Rank'].astype(int).astype(str)))
        return m, ['Stranger Things']

    if analysis == 'identity':
        m = dict(zip(names, info_df['Name']))
        return m, []

    raise ValueError(f"Unknown analysis: {analysis}")


def plot_condition_pcs(pca_result, info, cfg, title='', n_pcs=6):
    """Plot PC-vs-time for each condition, one subplot per PC."""
    scores_3d = pca_result['scores_3d']      # (n_conditions, n_bins, n_comp)
    conditions = info['conditions']
    var = pca_result['var']
    t = np.arange(info['n_bins']) * cfg.bin_width

    n_pcs = min(n_pcs, scores_3d.shape[2])
    ncols = 3
    nrows = int(np.ceil(n_pcs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    for pc, ax in enumerate(axes.flat):
        if pc >= n_pcs:
            ax.axis('off'); continue
        for ci, cname in enumerate(conditions):
            ax.plot(t, scores_3d[ci, :, pc], label=str(cname), lw=1.5)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_title(f'PC{pc+1} ({var[pc]:.1%})')
        ax.set_xlabel('Time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave room on the right
    fig.savefig('plot_condition_pcs.png')
    return fig


def plot_condition_3d(pca_result, info, title=''):
    """3D trajectory in PC1–3 space."""
    scores_3d = pca_result['scores_3d']
    conditions = info['conditions']

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for ci, cname in enumerate(conditions):
        tr = scores_3d[ci]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=2, label=str(cname))
        ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], marker='o', s=50)   # start
        ax.scatter(tr[-1, 0], tr[-1, 1], tr[-1, 2], marker='s', s=50)  # end
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('plot_condition_3d.png')
    return fig


def main():
    cfg = TrajectoryConfig(
        region='ER', session=None, trial_averaged=True, peak_align=False,
        n_components=6, bin_width=0.050, min_epoch_duration=2.0,
        analysis='group' # 'identity' | 'group' | 'familiarity' | 'sex' | 'rank'
    )
    cfg.validate()

    # --------- User settings ----------
    MEAN_CENTER = False
    SOFT_NORMALIZE = True
    MIN_REPS_PER_COND = 5
    PLOT_SAVE_DIR = f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/population_trajectory/{cfg.analysis}'
    SAVE = True
    # ----------------------------------

    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)

    if cfg.analysis == 'identity':
        monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
        common = sorted(set.intersection(*monkeys_per_session))
        name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
        # Drop monkeys not in monkeyinfo.csv
        known = [m for m in common if m in name_to_group]
        missing = set(common) - set(known)
        if missing:
            print(f"Dropping {len(missing)} monkeys not in monkeyinfo.csv: {sorted(missing)}")
        condition_map = {m: m for m in known}
        group_map = {m: name_to_group[m] for m in known}
        exclude_groups = []
        print(f"identity analysis: {len(known)} monkeys")
    else:
        condition_map, exclude_groups = build_condition_map(cfg.analysis, info_df)
        group_map = None  # each condition is its own group

    if exclude_groups:
        df = df[~df['MonkeyGroup'].isin(exclude_groups)]

    pca_matrix, row_meta_df, info = build_matrix_by_condition(
        df, cfg, condition_map, group_map=group_map, min_reps=MIN_REPS_PER_COND)

    pca_matrix = preprocess(pca_matrix, info, cfg, soft_normalize=SOFT_NORMALIZE, mean_center=MEAN_CENTER)
    pca_result = run_pca(pca_matrix, info, cfg)

    cond_trajs, group_trajs, c2g = compute_trajectories(
        pca_result, row_meta_df, info, cfg)

    var = pca_result['var']
    suffix = f"[{cfg.region}] {cfg.analysis} MC: {MEAN_CENTER} SN: {SOFT_NORMALIZE}"

    # plot_group_mean_3d_mpl(group_trajs, var, cfg, suffix, save=SAVE, save_dir = PLOT_SAVE_DIR)
    plot_group_mean_2d_mpl(group_trajs, var, cfg, suffix=suffix, save=SAVE, save_dir = PLOT_SAVE_DIR)
    plot_pc_vs_time_group_mean(group_trajs, var, cfg, row_meta_df, info,
                               pcs=range(1, cfg.n_components + 1),
                               suffix=suffix, save=SAVE)
    # plot_pc_vs_time_per_group(group_trajs, c2g, var, cfg, row_meta_df, info, pcs=range(1, cfg.n_components + 1),
    #                            suffix=suffix+"_per_group", save=SAVE)
    # plot_pc_vs_time_all_shaded(group_trajs, c2g, var, cfg, row_meta_df, info, pcs=range(1, cfg.n_components + 1),
    #                                 suffix=suffix+"_per_group", save=SAVE)

    # Only meaningful when there's sub-grouping (identity mode)
    if cfg.analysis == 'identity':
        plot_per_group_3d_mpl(cond_trajs, c2g, var, cfg, suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_all_shaded_3d_mpl(cond_trajs, c2g, var, cfg, suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_pc_vs_time_per_group(cond_trajs, c2g, var, cfg, row_meta_df, info,
        #                                pcs=range(1, cfg.n_components + 1), suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_per_group_2d_mpl(cond_trajs, c2g, var, cfg, suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_all_shaded_2d_mpl(cond_trajs, c2g, var, cfg, suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)

    plt.show()

if __name__ == '__main__':
    main()
