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

from population_analysis.state_space.plotting import plot_per_group_2d_mpl, plot_all_shaded_2d_mpl, plot_all_shaded_3d_plotly
from social_rank_analysis import load_group_matrices, davids_score
from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from population_analysis.state_space.preprocessing import preprocess
from pca_runner import run_pca, plot_scree
from plotting import (compute_trajectories)

MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"


def _compute_ds_combined(groups=('Zombies', 'Best Frans')):
    """Compute DS_combined per monkey by importing social_rank_analysis.
    Returns {MonkeyName: float}. DS is computed within each group."""
    ds_all = {}
    for g in groups:
        mats = load_group_matrices(g)
        combined = mats['agonism'] + mats['submission'].T
        ds = davids_score(combined)
        for name, score in ds.items():
            ds_all[str(name)] = float(score)
    return ds_all


def _tertile_within_group(scores, group_labels, bin_names=('DS_low', 'DS_mid', 'DS_high')):
    """Rank-based tertile split within each group. Stable for small n."""
    out = pd.Series(index=scores.index, dtype=object)
    for g in group_labels.unique():
        mask = group_labels == g
        s = scores[mask]
        ranks = s.rank(method='first')
        n = len(ranks)
        bins = np.ceil(ranks / n * 3).astype(int).clip(1, 3)  # 1, 2, or 3
        out.loc[mask] = bins.map({1: bin_names[0], 2: bin_names[1], 3: bin_names[2]})
    return out
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
        sub = info_df[~info_df['Group Name'].isin(['Stranger Things', 'Instigators'])].copy()
        sub = sub.dropna(subset=['Rank'])
        sub = sub[sub['Rank'].astype(int).between(1, 5)]  # <-- add this line
        m = dict(zip(sub['Name'].astype(str),
                     'rank' + sub['Rank'].astype(int).astype(str)))
        return m, ['Stranger Things', 'Instigators']

    if analysis == 'identity':
        m = dict(zip(names, info_df['Name']))
        return m, []
    if analysis == 'adult_females':
        # Adult females in Zombies + Best Frans, binned by within-group DS tertile
        sub = info_df[info_df['Group Name'].isin(['Zombies', 'Best Frans'])].copy()
        sub = sub[sub['Sex'] == 'F']
        sub = sub[sub['Age'] >= 4]
        sub['Name'] = sub['Name'].astype(str)

        ds_map = _compute_ds_combined(groups=('Zombies', 'Best Frans'))
        sub['DS'] = sub['Name'].map(ds_map)
        sub = sub.dropna(subset=['DS'])

        sub['DS_bin'] = _tertile_within_group(sub['DS'], sub['Group Name'])
        print(f"adult_females: {len(sub)} monkeys across "
              f"{sub['Group Name'].nunique()} groups")
        print(sub[['Name', 'Group Name', 'DS', 'DS_bin']]
              .sort_values(['Group Name', 'DS'], ascending=[True, False])
              .to_string(index=False))

        m = dict(zip(sub['Name'], sub['DS_bin']))
        return m, ['Stranger Things', 'Instigators']

    if analysis == 'demographic':
        # alpha / adult_female / juvenile across Zombies + Best Frans
        sub = info_df[info_df['Group Name'].isin(['Zombies', 'Best Frans'])].copy()
        sub['Name'] = sub['Name'].astype(str)

        def _label(row):
            if pd.notna(row['Age']) and row['Age'] < 4:
                return 'juvenile'
            if pd.notna(row['Rank']) and int(row['Rank']) == 1:
                return 'alpha'
            if row['Sex'] == 'F':
                return 'adult_female'
            return None  # adult males who aren't alpha — excluded

        sub['demo'] = sub.apply(_label, axis=1)
        sub = sub.dropna(subset=['demo'])
        print(f"demographic: {sub['demo'].value_counts().to_dict()}")

        m = dict(zip(sub['Name'], sub['demo']))
        return m, ['Stranger Things', 'Instigators']

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
        analysis= 'identity' # 'adult_females' | 'identity' | 'group' | 'rank' |     cannot use 'familiarity' | 'sex'  because there are only 2 groups
    )
    cfg.validate()

    # --------- User settings ----------
    MEAN_CENTER = True
    SOFT_NORMALIZE = True
    MIN_REPS_PER_COND = 5
    PLOT_SAVE_DIR = f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/state_space_trajectory/{cfg.analysis}'
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

    raw_matrix = pca_matrix.copy()

    pca_matrix = preprocess(pca_matrix, info, cfg, soft_normalize=SOFT_NORMALIZE, mean_center=MEAN_CENTER)
    pca_result = run_pca(pca_matrix, info, cfg)
    plot_scree(pca_result)
    cond_trajs, group_trajs, c2g = compute_trajectories(
        pca_result, row_meta_df, info, cfg)

    var = pca_result['var']
    suffix = f"[{cfg.region}] {cfg.analysis} MC {MEAN_CENTER} SN {SOFT_NORMALIZE}"

    # plot_group_mean_3d_mpl(group_trajs, var, cfg, suffix, save=SAVE, save_dir = PLOT_SAVE_DIR)
    # plot_group_mean_2d_mpl(group_trajs, var, cfg, suffix=suffix, save=SAVE, save_dir = PLOT_SAVE_DIR)
    # plot_pc_vs_time_group_mean(group_trajs, var, cfg, row_meta_df, info,
    #                            pcs=range(1, cfg.n_components + 1),
    #                            suffix=suffix, save=SAVE)
    # plot_pc_vs_time_per_group(group_trajs, c2g, var, cfg, row_meta_df, info, pcs=range(1, cfg.n_components + 1),
    #                            suffix=suffix+"_per_group", save=SAVE)
    # plot_pc_vs_time_all_shaded(group_trajs, c2g, var, cfg, row_meta_df, info, pcs=range(1, cfg.n_components + 1),
    #                                 suffix=suffix+"_per_group", save=SAVE)


    # PSTH Plot for... trial-averaged by group; Note that SEM here is across all neurons pooled across sessions,
    # not across trials. Standard single-neuron PSTHs use SEM across trials

    # if cfg.analysis == 'group':
    #     plot_psth_group_mean(raw_matrix, pca_matrix, row_meta_df, info, cfg,
    #                          smooth_sigma_s=0.050, suffix=suffix,
    #                          save=SAVE, save_dir=PLOT_SAVE_DIR)

    # Only meaningful when there's sub-grouping (identity mode)
    if cfg.analysis == 'identity':
        plot_all_shaded_3d_plotly(cond_trajs, c2g, var, cfg, suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_per_group_3d_mpl(cond_trajs, c2g, var, cfg, suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)

        # plot_all_shaded_3d_mpl(cond_trajs, c2g, var, cfg, suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        # plot_pc_vs_time_per_group(cond_trajs, c2g, var, cfg, row_meta_df, info,
        #                                  pcs=range(1, cfg.n_components + 1), suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        plot_per_group_2d_mpl(cond_trajs, c2g, var, cfg, suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)
        plot_all_shaded_2d_mpl(cond_trajs, c2g, var, cfg, suffix=suffix, save=SAVE, save_dir=PLOT_SAVE_DIR)

    plt.show()

if __name__ == '__main__':
    main()
