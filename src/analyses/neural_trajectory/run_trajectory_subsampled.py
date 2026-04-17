# run_trajectory_subsampled.py
"""
Subsampling variant of run_trajectory_by_condition.py for the 'group'
analysis: equalizes group sizes by randomly picking the same number of
stimulus monkeys from each group, then runs the full_dominance_first pipeline N times
with different random subsets.

Use this to check whether Best Frans (or any group) looks distinct
because of a real effect or because of small-sample noise / outliers.

Output: per-PC plots with one thin line per iteration and a thick
line showing the across-iteration mean trajectory per group.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from neural_trajectory.preprocessing import preprocess
from pca_runner import run_pca


MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


def subsample_condition_map(info_df, k, rng, analysis, exclude_names=()):
    """Pick k monkeys from each group uniformly at random."""
    info_df = info_df[~info_df['Name'].astype(str).isin(exclude_names)]
    m = {}
    for group, sub in info_df.groupby('Group Name'):
        names = sub['Name'].astype(str).tolist()
        if len(names) < k:
            raise ValueError(f"Group {group} has only {len(names)} monkeys, "
                             f"cannot sample {k}.")
        picked = rng.choice(names, size=k, replace=False)
        for n in picked:
            m[n] = group if analysis == 'group' else n
    return m


def run_one(df, cfg, condition_map, mean_center, min_reps, soft_normalize):
    pca_matrix, _, info = build_matrix_by_condition(
        df, cfg, condition_map, min_reps=min_reps)
    pca_matrix = preprocess(pca_matrix, info, cfg, mean_center=mean_center, soft_normalize=soft_normalize)
    pca_result = run_pca(pca_matrix, info, cfg)
    # scores_3d: (n_conditions, n_bins, n_comp)
    return pca_result['scores_3d'], info['conditions'], pca_result['var']


def _align_sign(ref, curr):
    """
    PCA sign is arbitrary. For each component, flip the whole curr
    trajectory set if that makes it closer to ref.
    ref and curr: (n_conditions, n_bins, n_comp).
    """
    out = curr.copy()
    for pc in range(curr.shape[2]):
        pos = np.sum((curr[:, :, pc] - ref[:, :, pc]) ** 2)
        neg = np.sum((-curr[:, :, pc] - ref[:, :, pc]) ** 2)
        if neg < pos:
            out[:, :, pc] = -curr[:, :, pc]
    return out


def plot_overlay(all_scores, conditions, var, cfg, n_bins, title='', n_pcs=6, save=False):
    """
    all_scores: ndarray (n_iter, n_conditions, n_bins, n_comp)
    """
    t = np.arange(n_bins) * cfg.bin_width
    mean_scores = all_scores.mean(axis=0)
    n_pcs = min(n_pcs, all_scores.shape[-1])
    ncols = 3
    nrows = int(np.ceil(n_pcs / ncols))

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    for pc, ax in enumerate(axes.flat):
        if pc >= n_pcs:
            ax.axis('off'); continue
        for ci, cname in enumerate(conditions):
            color = cmap(ci)
            # thin lines: each iteration
            for it in range(all_scores.shape[0]):
                ax.plot(t, all_scores[it, ci, :, pc],
                        color=color, lw=0.5, alpha=0.25)
            # thick line: mean across iterations
            ax.plot(t, mean_scores[ci, :, pc],
                    color=color, lw=2.2, label=str(cname))
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_title(f'PC{pc+1} (~{var[pc]:.1%})')
        ax.set_xlabel('Time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave room on the right
    if save:
        fig.savefig(f'{title}.png')
    return fig


def main():
    cfg = TrajectoryConfig(
        region='AMG',
        session=None,
        trial_averaged=True,
        peak_align=False,
        n_components=6,
        bin_width=0.050,
        min_epoch_duration=2.0,
        analysis = 'group' # 'group' | 'identity'
    )
    cfg.validate()

    # --------- User settings ----------
    cfg.analysis = 'group'      # 'group' | 'identity'
    K_PER_GROUP = 5         # smallest group = Best Frans (5)
    N_ITER = 30             # number of random subsamples
    MEAN_CENTER = True
    MIN_REPS_PER_COND = 5
    SEED = 15
    EXCLUDE_NAMES = ()      # e.g., ('19J',) to drop the suspected outlier
    SOFT_NORMALIZE = True
    # ----------------------------------

    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)

    rng = np.random.default_rng(SEED)

    all_scores = []
    conditions_ref = None
    var_ref = None
    ref_scores = None

    for it in range(N_ITER):
        cond_map = subsample_condition_map(info_df, K_PER_GROUP, rng, analysis=cfg.analysis,
                                           exclude_names=EXCLUDE_NAMES)
        scores_3d, conditions, var = run_one(
            df, cfg, cond_map, MEAN_CENTER, MIN_REPS_PER_COND, soft_normalize=SOFT_NORMALIZE)

        if conditions_ref is None:
            conditions_ref = conditions
            var_ref = var
            ref_scores = scores_3d
        else:
            assert conditions == conditions_ref, \
                "Condition order drifted between iterations."
            scores_3d = _align_sign(ref_scores, scores_3d)

        all_scores.append(scores_3d)
        print(f"  iter {it+1}/{N_ITER} done")

    all_scores = np.stack(all_scores, axis=0)  # (N_ITER, n_cond, n_bins, n_comp)
    n_bins = all_scores.shape[2]

    title = (f"[{cfg.region}] {cfg.analysis} subsampled k={K_PER_GROUP} "
             f"× {N_ITER} iters, mean-centered = {MEAN_CENTER}, soft_normalized = {SOFT_NORMALIZE}"
             + (f" (excl {','.join(EXCLUDE_NAMES)})" if EXCLUDE_NAMES else ""))
    plot_overlay(all_scores, conditions_ref, var_ref, cfg, n_bins, title=title, save=True)
    plt.show()


if __name__ == '__main__':
    main()
