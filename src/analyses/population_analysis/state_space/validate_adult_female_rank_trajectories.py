"""
Validation for adult_females DS-bin analysis:
  1. Leave-one-monkey-out  — is any single monkey driving the separation?
  2. Label shuffle test    — is observed bin separation above chance?
"""
import numpy as np
import pandas as pd

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from preprocessing import preprocess
from pca_runner import run_pca
from run_trajectory_by_condition import build_condition_map

MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"


def separation_metric(pca_result, top_k=6):
    """Mean pairwise Euclidean distance between bin-mean trajectories,
    summed over time, in the top-k PC space."""
    scores = pca_result['scores_3d'][:, :, :top_k]  # (n_cond, n_bins, k)
    n = scores.shape[0]
    total, n_pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(scores[i] - scores[j], axis=1)  # per-time-bin
            total += dist.sum()
            n_pairs += 1
    return total / n_pairs


def run_once(df, cfg, condition_map, min_reps=5, mean_center=True, soft_normalize=True):
    pca_matrix, row_meta_df, info = build_matrix_by_condition(
        df, cfg, condition_map, min_reps=min_reps)
    pca_matrix = preprocess(pca_matrix, info, cfg,
                            soft_normalize=soft_normalize, mean_center=mean_center)
    pca_result = run_pca(pca_matrix, info, cfg)
    return pca_result, info, row_meta_df


def pc_bin_means(pca_result, info, pc=1):
    """Time-averaged score for one PC, per condition."""
    scores = pca_result['scores_3d']
    means = scores[:, :, pc - 1].mean(axis=1)
    return dict(zip(info['conditions'], means))


def leave_one_out(df, cfg, condition_map, min_reps=5):
    """Drop each monkey one at a time; report PC1 and PC2 bin means."""
    print("\n--- Leave-one-monkey-out ---")
    base = run_once(df, cfg, condition_map, min_reps=min_reps)
    print(f"  baseline PC1 means: {_fmt(pc_bin_means(base[0], base[1], pc=1))}")
    print(f"  baseline PC2 means: {_fmt(pc_bin_means(base[0], base[1], pc=2))}")

    rows = []
    for monkey in sorted(condition_map.keys()):
        loo_map = {m: b for m, b in condition_map.items() if m != monkey}
        try:
            res = run_once(df, cfg, loo_map, min_reps=min_reps)
        except ValueError as e:
            print(f"  dropped {monkey}: pipeline failed ({e})")
            continue
        pc1 = pc_bin_means(res[0], res[1], pc=1)
        pc2 = pc_bin_means(res[0], res[1], pc=2)
        rows.append(dict(dropped=monkey, was_bin=condition_map[monkey],
                         **{f'PC1_{k}': v for k, v in pc1.items()},
                         **{f'PC2_{k}': v for k, v in pc2.items()}))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    return out


def shuffle_test(df, cfg, condition_map, n_shuffles=200, min_reps=5, seed=0):
    """Shuffle bin labels across monkeys; compare separation metric to null."""
    print(f"\n--- Shuffle test ({n_shuffles} iters) ---")
    rng = np.random.default_rng(seed)

    real = run_once(df, cfg, condition_map, min_reps=min_reps)
    real_sep = separation_metric(real[0], top_k=cfg.n_components)
    print(f"  observed separation: {real_sep:.3f}")

    monkeys = list(condition_map.keys())
    labels = list(condition_map.values())
    null = []
    for i in range(n_shuffles):
        perm = list(labels)
        rng.shuffle(perm)
        shuffled_map = dict(zip(monkeys, perm))
        try:
            res = run_once(df, cfg, shuffled_map, min_reps=min_reps)
            null.append(separation_metric(res[0], top_k=cfg.n_components))
        except ValueError:
            continue
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{n_shuffles}")
    null = np.array(null)
    p = (np.sum(null >= real_sep) + 1) / (len(null) + 1)
    print(f"  null mean: {null.mean():.3f}  |  95th pct: {np.percentile(null, 95):.3f}")
    print(f"  p-value: {p:.4f}  (based on {len(null)} valid shuffles)")
    return dict(observed=real_sep, null=null, p=p)


def _fmt(d):
    return "  ".join(f"{k}={v:+.3f}" for k, v in d.items())


def main():
    cfg = TrajectoryConfig(
        region='ER',           # change to 'AMG' to test that region
        session=None, trial_averaged=True, peak_align=False,
        n_components=6, bin_width=0.050, min_epoch_duration=2.0,
        analysis='adult_females')
    cfg.validate()

    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    condition_map, exclude_groups = build_condition_map(cfg.analysis, info_df)
    if exclude_groups:
        df = df[~df['MonkeyGroup'].isin(exclude_groups)]

    leave_one_out(df, cfg, condition_map)
    shuffle_test(df, cfg, condition_map, n_shuffles=200)


if __name__ == '__main__':
    main()