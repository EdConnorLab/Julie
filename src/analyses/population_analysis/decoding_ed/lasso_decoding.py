"""
lasso_decoding.py
=================

Sparse neural decoding of stimulus-monkey features using LASSO.

For each feature (dominance, sociability, age, sex, group), fits a
LASSO model predicting that feature from the mean firing-rate vector
across all neurons in one brain region (pseudo-population pooled
across sessions).

Pipeline
--------
  1.  load_and_filter  →  all neurons from one region.
  2.  build_matrix_by_condition (identity mapping) → trial-averaged
      firing-rate tensor (n_cond, n_bins, n_neurons).  Soft-normalize
      but do NOT mean-center (condition differences are the signal).
  3.  Average across time → (n_cond, n_neurons) mean-rate matrix.
  4.  For each feature, LOO-CV LASSO decoding:
        • continuous (dominance, sociability, age)  → LassoCV  → R²
        • binary    (sex)                           → LogisticRegressionCV (L1)  → accuracy
        • multiclass (group)                        → LogisticRegressionCV (L1, multinomial) → accuracy
  5.  Permutation test (shuffle labels, repeat LOO, build null).
  6.  Summary figure + per-feature null-distribution histograms.

Usage
-----
    python lasso_decoding.py

Adjust REGION, GROUPS, and N_PERM in the __main__ block.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from preprocessing import preprocess

sys.path.insert(0,
                '/population/neural_trajectory')
from social_rank_analysis import (
    load_group_matrices,
    davids_score,
    behavioral_pca,
)

MONKEY_INFO_PATH = (
    "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"
)


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def build_decoding_table(groups=('Zombies', 'Best Frans', 'Instigators'),
                         exclude_subject=True):
    """
    Returns a DataFrame indexed by monkey name with columns:
        group, dominance_z, sociability_z, age_z, sex

    Extends the TDR regressor table to include sex and Instigators.
    Continuous variables are z-scored within group (as in run_tdr_multi).
    """
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()
    info_idx = info_df.set_index('Name')

    # Behavioral PCA across all available groups for PC2 (sociability)
    all_mats = {g: load_group_matrices(g)
                for g in groups if g in ('Zombies', 'Best Frans', 'Instigators')}
    pca_scores, pca_obj, loadings = behavioral_pca(all_mats,
                                                   z_score_within_group=True)

    rows = []
    for g in groups:
        mats = all_mats[g]
        ago = mats['agonism']
        sub = mats['submission']
        combined = ago + sub.T
        ds = davids_score(combined)

        for m in ds.index:
            zero_ago = (ago.loc[m].sum() == 0) and (ago[m].sum() == 0)
            zero_sub = (sub.loc[m].sum() == 0) and (sub[m].sum() == 0)
            if zero_ago and zero_sub:
                continue
            if m not in info_idx.index:
                print(f"[warn] {m} in matrices but not in monkeyinfo.csv; skipping")
                continue

            age = info_idx.loc[m, 'Age']
            if pd.isna(age):
                print(f"[warn] {m} has no age; skipping")
                continue

            sex = info_idx.loc[m, 'Sex'] if 'Sex' in info_idx.columns else np.nan
            if pd.isna(sex):
                print(f"[warn] {m} has no sex; skipping")
                continue

            if m in pca_scores.index:
                soc = pca_scores.loc[m, 'PC2']
            else:
                print(f"[warn] {m} not in PCA scores; skipping")
                continue

            rows.append({
                'name': m,
                'group': g,
                'dominance_raw': ds[m],
                'sociability_raw': soc,
                'age_raw': float(age),
                'sex': str(sex).strip(),
                'note': (info_idx.loc[m, 'Note']
                         if 'Note' in info_idx.columns else ''),
            })

    tbl = pd.DataFrame(rows).set_index('name')

    if exclude_subject:
        subject_mask = (tbl['note'].fillna('')
                        .str.contains('subject', case=False))
        if subject_mask.any():
            dropped = tbl.index[subject_mask].tolist()
            print(f"Excluding subject monkey(s): {dropped}")
            tbl = tbl[~subject_mask]

    # Within-group z-scoring for continuous variables
    for col_raw, col_z in [('dominance_raw', 'dominance_z'),
                           ('sociability_raw', 'sociability_z'),
                           ('age_raw', 'age_z')]:
        parts = []
        for g, sub in tbl.groupby('group'):
            x = sub[col_raw]
            if x.std(ddof=0) == 0:
                z = x * 0.0
            else:
                z = (x - x.mean()) / x.std(ddof=0)
            parts.append(z)
        tbl[col_z] = pd.concat(parts)

    return tbl


# ---------------------------------------------------------------------------
# Neural data → mean-rate matrix
# ---------------------------------------------------------------------------
def build_mean_rate_matrix(df, cfg, monkey_list, min_reps=1):
    """
    Build a (n_monkeys, n_neurons_total) mean-rate matrix.

    Steps:
      1. build_matrix_by_condition with identity mapping.
      2. Soft-normalize (no mean-centering — condition differences
         are the decoding signal).
      3. Reshape to (n_cond, n_bins, n_neurons) and average over time.

    Returns
    -------
    X : (n_monkeys, n_neurons) mean firing-rate matrix
    conditions : list of monkey names in row order
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

    R = pca_matrix.reshape(n_cond, n_bins, n_neu)  # (cond, time, neurons)
    X = R.mean(axis=1)                               # (cond, neurons)

    conditions = info['conditions']
    print(f"Mean-rate matrix: {X.shape}  "
          f"({n_cond} monkeys × {n_neu} neurons)")
    return X, conditions, info


# ---------------------------------------------------------------------------
# LOO decoding helpers
# ---------------------------------------------------------------------------
def _loo_lasso_continuous(X, y):
    """
    LOO cross-validated LASSO for a continuous target.

    Inner CV (LassoCV) picks the regularisation strength;
    outer LOO evaluates generalisation.

    Returns (R², y_pred array).
    """
    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)

    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler().fit(X[train_ix])
        X_tr = scaler.transform(X[train_ix])
        X_te = scaler.transform(X[test_ix])

        model = LassoCV(cv=min(5, len(train_ix)),
                         max_iter=10_000, random_state=0)
        model.fit(X_tr, y[train_ix])
        y_pred[test_ix] = model.predict(X_te)

    r2 = r2_score(y, y_pred)
    return r2, y_pred


def _loo_lasso_categorical(X, y):
    """
    LOO cross-validated logistic LASSO for a categorical target.

    Handles binary (2 classes) and multiclass (>2, multinomial).

    Returns (accuracy, y_pred array).
    """
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    classes = le.classes_
    n_classes = len(classes)

    loo = LeaveOneOut()
    y_pred = np.full_like(y_enc, -1)

    multi_class = 'ovr' if n_classes == 2 else 'multinomial'
    solver = 'liblinear' if n_classes == 2 else 'saga'

    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler().fit(X[train_ix])
        X_tr = scaler.transform(X[train_ix])
        X_te = scaler.transform(X[test_ix])

        # Check that train fold contains all classes
        if len(np.unique(y_enc[train_ix])) < n_classes:
            # Degenerate fold — predict majority class
            vals, counts = np.unique(y_enc[train_ix], return_counts=True)
            y_pred[test_ix] = vals[counts.argmax()]
            continue

        model = LogisticRegressionCV(
            penalty='l1', solver=solver, multi_class=multi_class,
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
    """
    Shuffle y, re-decode, collect null metric values.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,) — raw labels (string or float)
    decode_fn : callable(X, y) → (metric, y_pred)
    n_perm : int

    Returns
    -------
    null_metrics : (n_perm,)
    """
    rng = np.random.default_rng(rng_seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null[i], _ = decode_fn(X, y_shuf)
        if (i + 1) % 200 == 0:
            print(f"    perm {i + 1}/{n_perm}")
    return null


# ---------------------------------------------------------------------------
# Full-data LASSO for coefficient inspection
# ---------------------------------------------------------------------------
def fit_full_lasso_continuous(X, y):
    """Fit LassoCV on ALL data, return model + scaler (for coef inspection)."""
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = LassoCV(cv=min(5, len(y)), max_iter=10_000, random_state=0)
    model.fit(X_s, y)
    return model, scaler


def fit_full_lasso_categorical(X, y):
    """Fit LogisticRegressionCV on ALL data."""
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    n_classes = len(le.classes_)
    multi_class = 'ovr' if n_classes == 2 else 'multinomial'
    solver = 'liblinear' if n_classes == 2 else 'saga'

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = LogisticRegressionCV(
        penalty='l1', solver=solver, multi_class=multi_class,
        cv=min(5, len(y)), max_iter=10_000, random_state=0)
    model.fit(X_s, y_enc)
    return model, scaler, le


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_summary(results, save_path=None):
    """
    Bar chart of decoding performance for each feature.
    Green = significant (p < 0.05), grey = not significant.
    """
    names = list(results.keys())
    metrics = [results[n]['metric'] for n in names]
    pvals = [results[n]['p_value'] for n in names]
    kinds = [results[n]['kind'] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ca02c' if p < 0.05 else '#888888' for p in pvals]
    bars = ax.bar(range(len(names)), metrics, color=colors, edgecolor='k',
                  linewidth=0.8)

    for i, (m, p) in enumerate(zip(metrics, pvals)):
        label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(i, m + 0.02 * max(abs(min(metrics)), max(metrics)),
                label, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('R² (continuous) / Accuracy (categorical)')
    ax.set_title('LASSO Decoding Summary')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_null_distributions(results, save_dir=None):
    """One histogram per feature: null distribution + observed value."""
    n_feat = len(results)
    fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 3.5),
                             squeeze=False)
    for ax, (name, res) in zip(axes.flat, results.items()):
        null = res['null']
        obs = res['metric']
        ax.hist(null, bins=30, color='#aaaaaa', edgecolor='k', linewidth=0.3)
        ax.axvline(obs, color='red', lw=2, label=f"observed = {obs:.3f}")
        ax.set_title(name)
        metric_label = 'R²' if res['kind'] == 'continuous' else 'Accuracy'
        ax.set_xlabel(metric_label)
        ax.set_ylabel('count')
        ax.legend(fontsize=7)
    fig.suptitle('Permutation Null Distributions', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_dir:
        path = os.path.join(save_dir, 'null_distributions.png')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"  saved → {path}")
    return fig


def plot_selected_neurons(results, neuron_ids, save_dir=None):
    """
    For each continuous feature, bar chart of LASSO coefficients
    (full_dominance_first-data fit) showing which neurons were selected.
    """
    continuous = {k: v for k, v in results.items()
                  if v['kind'] == 'continuous'}
    if not continuous:
        return

    for name, res in continuous.items():
        coef = res.get('full_coef')
        if coef is None:
            continue
        nonzero = np.where(coef != 0)[0]
        if len(nonzero) == 0:
            print(f"  {name}: all coefficients are zero (fully regularised)")
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(nonzero) * 0.3), 4))
        ax.bar(range(len(nonzero)), coef[nonzero], color='steelblue',
               edgecolor='k', linewidth=0.3)
        ax.set_xticks(range(len(nonzero)))
        ax.set_xticklabels([neuron_ids[i] for i in nonzero],
                           rotation=90, fontsize=5)
        ax.set_ylabel('LASSO coefficient')
        ax.set_title(f'{name}: {len(nonzero)} / {len(coef)} neurons selected')
        fig.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f'coef_{name}.png')
            fig.savefig(path, dpi=200)
            print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==================== USER SETTINGS ====================
    REGION = 'AMG'
    GROUPS = ('Zombies', 'Best Frans', 'Instigators')
    N_PERM = 1000
    MIN_REPS = 1          # min trials per condition per session
    SAVE_DIR = (
        "/home/connorlab/Documents/GitHub/Julie/Cortana/"
        f"analysis_results/lasso_decoding/{REGION}"
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

    # Intersect: monkeys that have BOTH neural data and features
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
            f"Only {len(common)} monkeys overlap — too few for decoding. "
            "Check group coverage in neural data."
        )

    X, conditions, info = build_mean_rate_matrix(
        df, cfg, common, min_reps=MIN_REPS)

    # Align feature table to matrix row order
    tbl = tbl.loc[conditions]
    print(f"\nFinal decoding set: {len(conditions)} monkeys, "
          f"{X.shape[1]} neurons")
    print(tbl[['group', 'dominance_z', 'sociability_z', 'age_z', 'sex']])
    print()

    # 3. Define features to decode
    features = {
        'dominance':   ('dominance_z',   'continuous'),
        'sociability': ('sociability_z', 'continuous'),
        'age':         ('age_z',         'continuous'),
        'sex':         ('sex',           'categorical'),
        'group':       ('group',         'categorical'),
    }

    # 4. Decode each feature
    results = {}
    for feat_name, (col, kind) in features.items():
        y = tbl[col].values
        print("=" * 60)
        print(f"Decoding: {feat_name}  ({kind})")
        print("=" * 60)

        if kind == 'continuous':
            metric, y_pred = _loo_lasso_continuous(X, y.astype(float))
            print(f"  LOO R² = {metric:.4f}")

            # Full-data fit for coefficient inspection
            model, scaler = fit_full_lasso_continuous(X, y.astype(float))
            full_coef = model.coef_
            n_sel = np.sum(full_coef != 0)
            print(f"  Full-data alpha = {model.alpha_:.4e}, "
                  f"{n_sel}/{len(full_coef)} neurons selected")

            # Permutation test
            print(f"  Running {N_PERM} permutations ...")
            null = permutation_test(
                X, y.astype(float), _loo_lasso_continuous, N_PERM)
            p = (np.sum(null >= metric) + 1) / (N_PERM + 1)
            print(f"  p = {p:.4f}")

            results[feat_name] = dict(
                metric=metric, kind=kind, null=null, p_value=p,
                y_true=y, y_pred=y_pred, full_coef=full_coef,
            )

        else:  # categorical
            metric, y_pred = _loo_lasso_categorical(X, y)
            chance = 1.0 / len(np.unique(y))
            print(f"  LOO accuracy = {metric:.4f}  "
                  f"(chance = {chance:.4f})")

            # Permutation test
            print(f"  Running {N_PERM} permutations ...")
            null = permutation_test(
                X, y, _loo_lasso_categorical, N_PERM)
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
    plot_summary(results, save_path=os.path.join(SAVE_DIR, 'summary.png'))
    plot_null_distributions(results, save_dir=SAVE_DIR)
    plot_selected_neurons(results, info['neuron_ids'], save_dir=SAVE_DIR)

    # 6. Save numerical results
    summary_rows = []
    for feat_name, res in results.items():
        row = {
            'feature': feat_name,
            'kind': res['kind'],
            'metric': res['metric'],
            'p_value': res['p_value'],
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