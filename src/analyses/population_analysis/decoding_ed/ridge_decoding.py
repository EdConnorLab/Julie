"""
ridge_decoding.py
=================

Ridge-regression neural decoding of stimulus-monkey features from
population activity in a specified time window.

Unlike the LASSO variant, Ridge keeps ALL neurons (no sparsity), which
tends to decode better when n_samples << n_features because it
regularises smoothly rather than zeroing out coefficients.

Pipeline
--------
  1.  load_and_filter  →  all neurons from one region.
  2.  build_matrix_by_condition (identity mapping) → trial-averaged
      firing-rate tensor (n_cond, n_bins, n_neurons).  Soft-normalize,
      NO mean-centering.
  3.  Slice the requested time window and average across those bins →
      (n_cond, n_neurons).
  4.  For each feature, LOO-CV Ridge decoding:
        • continuous (dominance, sociability, age)  → RidgeCV  → R²
        • binary    (sex)                           → LogisticRegressionCV (L2)  → accuracy
        • multiclass (group)                        → LogisticRegressionCV (L2, multinomial) → accuracy
  5.  Permutation test (shuffle labels, repeat LOO, build null).
  6.  Summary figure + null-distribution histograms.

Usage
-----
    python ridge_decoding.py

Adjust REGION, GROUPS, TIME_WINDOW_S, and N_PERM in the __main__ block.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition


# Re-use the feature-table builder from the LASSO script
from lasso_decoding import build_decoding_table
from population_analysis.state_space.preprocessing import preprocess


# ---------------------------------------------------------------------------
# Neural data → windowed mean-rate matrix
# ---------------------------------------------------------------------------
def build_windowed_rate_matrix(df, cfg, monkey_list,
                               time_window_s=(0.2, 0.5),
                               min_reps=1):
    """
    Build a (n_monkeys, n_neurons_total) mean-rate matrix restricted
    to a specific time window.

    Parameters
    ----------
    time_window_s : (float, float)
        Start and end of the analysis window in seconds relative to
        epoch onset.  Bins whose START falls in [t_start, t_end) are
        included.  With bin_width = 0.050 s and window (0.2, 0.5),
        that is bins 4–9 inclusive (200–500 ms).

    Returns
    -------
    X : (n_monkeys, n_neurons)
    conditions : list of monkey names (row order)
    info : dict from build_matrix_by_condition
    """
    condition_map = {m: m for m in monkey_list}

    pca_matrix, row_meta, info = build_matrix_by_condition(
        df, cfg, condition_map, min_reps=min_reps)

    # Soft-normalize, no mean-centering
    pca_matrix = preprocess(pca_matrix, info, cfg,
                            soft_normalize=True, mean_center=False)

    n_cond = info['n_conditions']
    n_bins = info['n_bins']
    n_neu = pca_matrix.shape[1]

    R = pca_matrix.reshape(n_cond, n_bins, n_neu)

    # Determine bin indices for the requested window
    t_start, t_end = time_window_s
    bin_starts = np.arange(n_bins) * cfg.bin_width  # start time of each bin
    mask = (bin_starts >= t_start) & (bin_starts < t_end)
    selected_bins = np.where(mask)[0]

    if len(selected_bins) == 0:
        raise ValueError(
            f"No bins fall in window [{t_start}, {t_end}) s. "
            f"Available range: [0, {n_bins * cfg.bin_width:.3f}) s"
        )

    X = R[:, selected_bins, :].mean(axis=1)  # (n_cond, n_neurons)

    conditions = info['conditions']
    print(f"Time window: {t_start}–{t_end} s  "
          f"(bins {selected_bins[0]}–{selected_bins[-1]}, "
          f"{len(selected_bins)} bins)")
    print(f"Windowed rate matrix: {X.shape}  "
          f"({n_cond} monkeys × {n_neu} neurons)")
    return X, conditions, info


# ---------------------------------------------------------------------------
# LOO decoding helpers
# ---------------------------------------------------------------------------
def _loo_ridge_continuous(X, y):
    """
    LOO cross-validated Ridge for a continuous target.

    Inner CV (RidgeCV with GCV) picks alpha; outer LOO evaluates.

    Returns (R², y_pred array).
    """
    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)

    alphas = np.logspace(-3, 5, 50)

    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler().fit(X[train_ix])
        X_tr = scaler.transform(X[train_ix])
        X_te = scaler.transform(X[test_ix])

        model = RidgeCV(alphas=alphas, scoring='r2')
        model.fit(X_tr, y[train_ix])
        y_pred[test_ix] = model.predict(X_te)

    r2 = r2_score(y, y_pred)
    return r2, y_pred


def _loo_ridge_categorical(X, y):
    """
    LOO cross-validated logistic Ridge for a categorical target.

    Returns (accuracy, y_pred array).
    """
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    classes = le.classes_
    n_classes = len(classes)

    loo = LeaveOneOut()
    y_pred = np.full_like(y_enc, -1)

    multi_class = 'ovr' if n_classes == 2 else 'multinomial'
    solver = 'lbfgs'  # supports L2 + multinomial

    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler().fit(X[train_ix])
        X_tr = scaler.transform(X[train_ix])
        X_te = scaler.transform(X[test_ix])

        if len(np.unique(y_enc[train_ix])) < n_classes:
            vals, counts = np.unique(y_enc[train_ix], return_counts=True)
            y_pred[test_ix] = vals[counts.argmax()]
            continue

        model = LogisticRegressionCV(
            penalty='l2', solver=solver, multi_class=multi_class,
            cv=min(5, len(train_ix)), max_iter=10_000, random_state=0,
        )
        model.fit(X_tr, y_enc[train_ix])
        y_pred[test_ix] = model.predict(X_te)

    acc = accuracy_score(y_enc, y_pred)
    return acc, le.inverse_transform(y_pred)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def permutation_test(X, y, decode_fn, n_perm=1000, rng_seed=42):
    """Shuffle y, re-decode, collect null metric values."""
    rng = np.random.default_rng(rng_seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null[i], _ = decode_fn(X, y_shuf)
        if (i + 1) % 200 == 0:
            print(f"    perm {i + 1}/{n_perm}")
    return null


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_summary(results, region, time_window_s, save_path=None):
    """Bar chart of decoding performance, green = p<0.05."""
    names = list(results.keys())
    metrics = [results[n]['metric'] for n in names]
    pvals = [results[n]['p_value'] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ca02c' if p < 0.05 else '#888888' for p in pvals]
    ax.bar(range(len(names)), metrics, color=colors, edgecolor='k', lw=0.8)

    y_top = max(abs(min(metrics)), max(metrics)) if metrics else 1
    for i, (m, p) in enumerate(zip(metrics, pvals)):
        label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(i, m + 0.03 * y_top, label,
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('R² (continuous) / Accuracy (categorical)')
    ax.set_title(f'Ridge Decoding — {region}  '
                 f'[{time_window_s[0]*1000:.0f}–{time_window_s[1]*1000:.0f} ms]')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_null_distributions(results, region, time_window_s, save_dir=None):
    """One histogram per feature: null distribution + observed."""
    n_feat = len(results)
    fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 3.5),
                             squeeze=False)
    for ax, (name, res) in zip(axes.flat, results.items()):
        null = res['null']
        obs = res['metric']
        ax.hist(null, bins=30, color='#aaaaaa', edgecolor='k', lw=0.3)
        ax.axvline(obs, color='red', lw=2, label=f"obs = {obs:.3f}")
        ax.set_title(name)
        metric_label = 'R²' if res['kind'] == 'continuous' else 'Accuracy'
        ax.set_xlabel(metric_label)
        ax.set_ylabel('count')
        ax.legend(fontsize=7)
    fig.suptitle(f'Permutation Nulls — Ridge — {region}  '
                 f'[{time_window_s[0]*1000:.0f}–{time_window_s[1]*1000:.0f} ms]',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_dir:
        path = os.path.join(save_dir, 'null_distributions.png')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"  saved → {path}")
    return fig


def plot_pred_vs_true(results, save_dir=None):
    """Scatter of predicted vs true for each continuous feature."""
    continuous = {k: v for k, v in results.items()
                  if v['kind'] == 'continuous'}
    if not continuous:
        return
    n = len(continuous)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (name, res) in zip(axes.flat, continuous.items()):
        y_true = res['y_true']
        y_pred = res['y_pred']
        ax.scatter(y_true, y_pred, c='steelblue', edgecolor='k', lw=0.3)
        lims = [min(y_true.min(), y_pred.min()) - 0.2,
                max(y_true.max(), y_pred.max()) + 0.2]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel('True (z-scored)')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}  (R²={res["metric"]:.3f}, '
                     f'p={res["p_value"]:.3f})')
        ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, 'pred_vs_true.png')
        fig.savefig(path, dpi=200)
        print(f"  saved → {path}")
    return fig


def plot_confusion(results, save_dir=None):
    """Confusion-style printout for categorical features."""
    categorical = {k: v for k, v in results.items()
                   if v['kind'] == 'categorical'}
    for name, res in categorical.items():
        y_true = res['y_true']
        y_pred = res['y_pred']
        labels = sorted(set(y_true))
        print(f"\n  {name} confusion (LOO):")
        print(f"    {'true  pred':<15}", end='')
        for l in labels:
            print(f"{l:<15}", end='')
        print()
        for lt in labels:
            print(f"    {lt:<15}", end='')
            for lp in labels:
                count = sum((yt == lt) and (yp == lp)
                            for yt, yp in zip(y_true, y_pred))
                print(f"{count:<15}", end='')
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==================== USER SETTINGS ====================
    REGION = 'AMG'
    GROUPS = ('Zombies', 'Best Frans', 'Instigators')
    TIME_WINDOW_S = (0.200, 0.500)    # seconds from epoch onset
    N_PERM = 1000
    MIN_REPS = 1
    SAVE_DIR = (
        "/home/connorlab/Documents/GitHub/Julie/Cortana/"
        f"analysis_results/ridge_decoding/{REGION}"
    )
    # =======================================================

    cfg = TrajectoryConfig(
        region=REGION,
        session=None,
        trial_averaged=True,
        peak_align=False,
        bin_width=0.050,
        min_epoch_duration=2.0,
    )
    cfg.validate()

    # 1. Feature table
    print("=" * 60)
    print("Building feature table ...")
    print("=" * 60)
    tbl = build_decoding_table(groups=GROUPS, exclude_subject=True)
    print(tbl[['group', 'dominance_z', 'sociability_z', 'age_z', 'sex']])
    print()

    # 2. Neural data
    print("=" * 60)
    print("Loading neural data ...")
    print("=" * 60)
    df = load_and_filter(cfg)

    monkeys_in_data = set(df['MonkeyName'].unique())
    monkeys_in_tbl = set(tbl.index)
    common = sorted(monkeys_in_data & monkeys_in_tbl)
    only_data = monkeys_in_data - monkeys_in_tbl
    only_tbl = monkeys_in_tbl - monkeys_in_data
    print(f"\nMonkeys in neural data:    {len(monkeys_in_data)}")
    print(f"Monkeys in feature table:  {len(monkeys_in_tbl)}")
    print(f"Intersection (usable):     {len(common)}")
    if only_data:
        print(f"  in data but no features: {sorted(only_data)}")
    if only_tbl:
        print(f"  in table but no data:    {sorted(only_tbl)}")
    print()

    if len(common) < 5:
        raise RuntimeError(
            f"Only {len(common)} monkeys overlap — too few for decoding."
        )

    X, conditions, info = build_windowed_rate_matrix(
        df, cfg, common, time_window_s=TIME_WINDOW_S, min_reps=MIN_REPS)

    tbl = tbl.loc[conditions]
    print(f"\nFinal decoding set: {len(conditions)} monkeys, "
          f"{X.shape[1]} neurons")
    print(tbl[['group', 'dominance_z', 'sociability_z', 'age_z', 'sex']])
    print()

    # 3. Features
    features = {
        'dominance':   ('dominance_z',   'continuous'),
        'sociability': ('sociability_z', 'continuous'),
        'age':         ('age_z',         'continuous'),
        'sex':         ('sex',           'categorical'),
        'group':       ('group',         'categorical'),
    }

    # 4. Decode
    results = {}
    for feat_name, (col, kind) in features.items():
        y = tbl[col].values
        print("=" * 60)
        print(f"Decoding: {feat_name}  ({kind})  — Ridge")
        print("=" * 60)

        if kind == 'continuous':
            y_float = y.astype(float)
            metric, y_pred = _loo_ridge_continuous(X, y_float)
            print(f"  LOO R² = {metric:.4f}")

            print(f"  Running {N_PERM} permutations ...")
            null = permutation_test(
                X, y_float, _loo_ridge_continuous, N_PERM)
            p = (np.sum(null >= metric) + 1) / (N_PERM + 1)
            print(f"  p = {p:.4f}")

            results[feat_name] = dict(
                metric=metric, kind=kind, null=null, p_value=p,
                y_true=y_float, y_pred=y_pred,
            )

        else:
            metric, y_pred = _loo_ridge_categorical(X, y)
            chance = 1.0 / len(np.unique(y))
            print(f"  LOO accuracy = {metric:.4f}  "
                  f"(chance = {chance:.4f})")

            print(f"  Running {N_PERM} permutations ...")
            null = permutation_test(
                X, y, _loo_ridge_categorical, N_PERM)
            p = (np.sum(null >= metric) + 1) / (N_PERM + 1)
            print(f"  p = {p:.4f}")

            results[feat_name] = dict(
                metric=metric, kind=kind, null=null, p_value=p,
                y_true=y, y_pred=y_pred, chance=chance,
            )

    # 5. Plots
    print("\n" + "=" * 60)
    print("Plotting ...")
    print("=" * 60)
    plot_summary(results, REGION, TIME_WINDOW_S,
                 save_path=os.path.join(SAVE_DIR, 'summary.png'))
    plot_null_distributions(results, REGION, TIME_WINDOW_S,
                            save_dir=SAVE_DIR)
    plot_pred_vs_true(results, save_dir=SAVE_DIR)
    plot_confusion(results, save_dir=SAVE_DIR)

    # 6. Save
    summary_rows = []
    for feat_name, res in results.items():
        row = {
            'feature': feat_name,
            'kind': res['kind'],
            'metric': res['metric'],
            'p_value': res['p_value'],
            'window_ms': f"{TIME_WINDOW_S[0]*1000:.0f}-{TIME_WINDOW_S[1]*1000:.0f}",
        }
        if res['kind'] == 'categorical':
            row['chance'] = res.get('chance')
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(SAVE_DIR, 'decoding_summary.csv')
    os.makedirs(SAVE_DIR, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved → {csv_path}")
    print(summary_df.to_string(index=False))

    plt.show()


if __name__ == '__main__':
    main()
