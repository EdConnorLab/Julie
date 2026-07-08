# decode_utils.py
"""
Shared utilities for all decoding / RSA scripts.

Centralises data loading, firing-rate computation, neuron filtering,
pseudo-population building, classification, regression, and permutation
testing so that individual analysis scripts contain only their
analysis-specific logic.
"""

import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Tuple, List

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report, r2_score)

import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_trials(pkl_dir: str) -> pd.DataFrame:
    """Load and concatenate all per-neuron .pkl DataFrames from *pkl_dir*."""
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")
    dfs = []
    for f in pkl_files:
        with open(f, 'rb') as fh:
            obj = pickle.load(fh)
        if isinstance(obj, pd.DataFrame):
            dfs.append(obj)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(pkl_files)} files -> {len(combined)} total trials")
    return combined


def load_and_prepare(cfg) -> pd.DataFrame:
    """
    One-call data pipeline:
      1. Load all trials from cfg.data_path
      2. Exclude monkeys listed in cfg.exclude_monkeys
      3. (Optional) filter by region / session
      4. Compute firing rates (windowed or full-epoch)

    Returns the prepared DataFrame with a 'firing_rate' column.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(cfg.data_path)

    # Exclude monkeys
    for name in cfg.exclude_monkeys:
        before = len(df)
        df = df[df['MonkeyName'] != name]
        dropped = before - len(df)
        if dropped:
            print(f"  Excluded '{name}': {dropped} trials removed")

    # Region filter (if the column exists)
    if cfg.region != 'ALL' and 'Region' in df.columns:
        df = df[df['Region'] == cfg.region]
        print(f"  Region filter '{cfg.region}': {len(df)} trials")

    # Session filter
    if cfg.session is not None and 'Session' in df.columns:
        df = df[df['Session'] == cfg.session]
        print(f"  Session filter '{cfg.session}': {len(df)} trials")

    # Minimum epoch duration filter
    if cfg.min_epoch_duration > 0:
        before = len(df)
        df = _filter_short_epochs(df, cfg.min_epoch_duration)
        print(f"  Epoch duration >= {cfg.min_epoch_duration}s: kept {len(df)}/{before}")

    print(f"  Total after filtering: {len(df)} trials")

    # Compute firing rates
    print("\nComputing firing rates ...")
    if cfg.response_window is not None:
        start_sec, end_sec = cfg.response_window
        print(f"  Response window: {start_sec}–{end_sec} s relative to epoch onset")
        df = compute_windowed_firing_rates(df, start_sec, end_sec)
    else:
        print("  Using full epoch duration")
        df = compute_firing_rates(df)

    return df


def _filter_short_epochs(df: pd.DataFrame, min_dur: float) -> pd.DataFrame:
    """Drop rows whose epoch is shorter than *min_dur* seconds."""
    keep = []
    for _, row in df.iterrows():
        epoch = row['EpochStartStop']
        if isinstance(epoch, str):
            epoch = eval(epoch)
        duration = epoch[1] - epoch[0]
        keep.append(duration >= min_dur)
    return df[keep].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRING RATES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_firing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Spike count in full epoch / epoch duration."""
    rates = []
    for _, row in df.iterrows():
        spikes = np.array(row['SpikeTimes'])
        epoch = row['EpochStartStop']
        if isinstance(epoch, str):
            epoch = eval(epoch)
        t_start, t_stop = epoch[0], epoch[1]
        duration = t_stop - t_start
        n_spikes = np.sum((spikes >= t_start) & (spikes <= t_stop))
        rate = n_spikes / duration if duration > 0 else 0.0
        rates.append(rate)
    df = df.copy()
    df['firing_rate'] = rates
    return df


def compute_windowed_firing_rates(df: pd.DataFrame,
                                  start_sec: float,
                                  end_sec: float) -> pd.DataFrame:
    """
    Spike count in [epoch_onset + start_sec, epoch_onset + end_sec] / window duration.

    The window is clipped to the actual epoch boundaries so it never extends
    beyond the available data.
    """
    window_dur = end_sec - start_sec
    if window_dur <= 0:
        raise ValueError(f"Invalid window: start={start_sec}, end={end_sec}")

    rates = []
    for _, row in df.iterrows():
        spikes = np.array(row['SpikeTimes'])
        epoch = row['EpochStartStop']
        if isinstance(epoch, str):
            epoch = eval(epoch)
        epoch_onset = epoch[0]
        epoch_offset = epoch[1]

        win_start = epoch_onset + start_sec
        win_end = min(epoch_onset + end_sec, epoch_offset)
        dur = win_end - win_start
        if dur <= 0:
            rates.append(0.0)
            continue
        n_spikes = np.sum((spikes >= win_start) & (spikes <= win_end))
        rates.append(n_spikes / dur)
    df = df.copy()
    df['firing_rate'] = rates
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# NEURON FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def filter_neurons_by_min_trials(df: pd.DataFrame,
                                 target_identities: list,
                                 min_trials: int,
                                 verbose: bool = True) -> Tuple[pd.DataFrame, list]:
    """
    Keep only neurons that have >= *min_trials* for EVERY identity in
    *target_identities*.  Returns (filtered_df, kept_neuron_ids).
    """
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(target_identities)
    kept = []
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        if all(len(neuron_df[neuron_df['MonkeyName'] == mid]) >= min_trials
               for mid in identities):
            kept.append(nid)

    if verbose:
        print(f"\n  min_trials filter (>= {min_trials} per identity):")
        print(f"    Kept: {len(kept)} / {len(neuron_ids)} neurons")
        print(f"    Dropped: {len(neuron_ids) - len(kept)}")

    return df[df['NeuronID'].isin(kept)], kept


# ═══════════════════════════════════════════════════════════════════════════════
# PSEUDO-POPULATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_pseudo_population(df: pd.DataFrame,
                            target_identities: list,
                            n_draws: int = 10,
                            seed: int = 42,
                            verbose: bool = True):
    """
    Build pseudo-population trial matrices by randomly pairing trials
    across neurons for each stimulus identity.

    Returns
    -------
    X_draws : list[np.ndarray]   – (n_trials, n_neurons) per draw
    y_draws : list[np.ndarray]   – identity labels per draw
    neuron_ids : list             – ordered neuron IDs (columns of X)
    valid_identities : list       – identities with > 0 trials everywhere
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(target_identities)

    if verbose:
        print(f"\nNeurons: {len(neuron_ids)}")
        print(f"Target identities: {len(identities)}")

    # Collect firing rates per (neuron, identity)
    neuron_identity_rates = {}
    min_trials_per_id = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        ndf = df[df['NeuronID'] == nid]
        for mid in identities:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = rates
            if len(rates) < min_trials_per_id[mid]:
                min_trials_per_id[mid] = len(rates)

    # Exclude identities with 0 trials for any neuron
    valid_identities = []
    for mid in identities:
        if int(min_trials_per_id[mid]) > 0:
            valid_identities.append(mid)
        elif verbose:
            print(f"  WARNING: '{mid}' has 0 trials for some neuron(s) — excluding")

    if verbose:
        print(f"Valid identities: {len(valid_identities)}")
        print("\nTrials per identity (min across neurons):")
        for mid in valid_identities:
            print(f"  {mid}: {int(min_trials_per_id[mid])}")

    # Draw pseudo-populations
    X_draws, y_draws = [], []
    for _ in range(n_draws):
        X_rows, y_rows = [], []
        for mid in valid_identities:
            nt = int(min_trials_per_id[mid])
            block = np.zeros((nt, len(neuron_ids)))
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=nt, replace=False)
                block[:, j] = rates[chosen]
            X_rows.append(block)
            y_rows.append(np.full(nt, mid))
        X_draws.append(np.vstack(X_rows))
        y_draws.append(np.concatenate(y_rows))

    return X_draws, y_draws, neuron_ids, valid_identities


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def balance_classes(X, y_enc, seed):
    """Downsample all classes to match the smallest class."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_enc, return_counts=True)
    min_count = min(counts)
    balanced_idx = []
    for c in classes:
        c_idx = np.where(y_enc == c)[0]
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        balanced_idx.extend(chosen)
    balanced_idx = np.array(sorted(balanced_idx))
    return X[balanced_idx], y_enc[balanced_idx]


def run_classification(X, y, cfg, *, do_balance=False, return_predictions=False):
    """
    Stratified k-fold logistic regression classification.

    Parameters
    ----------
    do_balance : bool
        If True, downsample to equal class sizes before CV.
    return_predictions : bool
        If True, also return (all_y_true, all_y_pred, LabelEncoder).

    Returns
    -------
    fold_accs : np.ndarray
    (optionally) all_y_true, all_y_pred, le
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if do_balance:
        X_use, y_use = balance_classes(X, y_enc, cfg.random_seed)
    else:
        X_use, y_use = X, y_enc

    min_class_count = min(np.bincount(y_use))
    n_folds = min(cfg.n_cv_folds, min_class_count)
    if n_folds < 2:
        print(f"    WARNING: smallest class has {min_class_count} samples, cannot CV")
        empty = (np.array([0.0]), np.array([]), np.array([]), le)
        return empty if return_predictions else np.array([0.0])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=cfg.random_seed)
    fold_accs = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X_use, y_use):
        X_train, X_test = X_use[train_idx], X_use[test_idx]
        y_train, y_test = y_use[train_idx], y_use[test_idx]

        # Standardise
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # PCA
        if cfg.n_pca is not None and cfg.n_pca < X_train.shape[1]:
            n_comp = min(cfg.n_pca, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=cfg.random_seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

        clf = LogisticRegression(
            C=cfg.logreg_C, penalty='l2', solver='lbfgs',
            max_iter=cfg.logreg_max_iter, multi_class='multinomial',
            random_state=cfg.random_seed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    fold_accs = np.array(fold_accs)
    if return_predictions:
        return fold_accs, np.array(all_y_true), np.array(all_y_pred), le
    return fold_accs


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def run_regression(X, y_continuous, cfg, seed_offset: int = 0):
    """K-fold Ridge regression.  Returns mean R²."""
    valid_mask = ~np.isnan(y_continuous)
    X_valid = X[valid_mask]
    y_valid = y_continuous[valid_mask]
    seed = cfg.random_seed + seed_offset

    if len(y_valid) < 10:
        print("    WARNING: too few valid samples for regression")
        return 0.0

    kf = KFold(n_splits=min(cfg.n_cv_folds, len(y_valid)),
               shuffle=True, random_state=seed)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X_valid):
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if cfg.n_pca is not None and cfg.n_pca < X_train.shape[1]:
            n_comp = min(cfg.n_pca, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        fold_r2s.append(r2_score(y_test, y_pred))

    return np.mean(fold_r2s)


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_classification_permutation(X, y, cfg, n_perms: int, seed_offset: int = 0):
    """Permutation null distribution for classification accuracy."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    perm_accs = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i + 1}/{n_perms}")
        y_shuf = rng.permutation(y)
        fold_accs = run_classification(X, y_shuf, cfg, do_balance=False)
        perm_accs.append(np.mean(fold_accs))
    return np.array(perm_accs)


def run_regression_permutation(X, y_continuous, cfg, n_perms: int, seed_offset: int = 0):
    """Permutation null distribution for regression R²."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    valid_mask = ~np.isnan(y_continuous)
    y_valid = y_continuous[valid_mask]

    perm_r2s = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i + 1}/{n_perms}")
        y_shuf = rng.permutation(y_valid)
        y_full_shuf = y_continuous.copy()
        y_full_shuf[valid_mask] = y_shuf
        r2 = run_regression(X, y_full_shuf, cfg, seed_offset=seed_offset + i)
        perm_r2s.append(r2)
    return np.array(perm_r2s)


def compute_p_value(observed: float, null_dist: np.ndarray) -> float:
    """One-tailed p-value: fraction of null >= observed (with +1 correction)."""
    return (np.sum(null_dist >= observed) + 1) / (len(null_dist) + 1)


def compute_p_value_twotailed(observed: float, null_dist: np.ndarray) -> float:
    """Two-tailed p-value based on |null| >= |observed|."""
    return (np.sum(np.abs(null_dist) >= np.abs(observed)) + 1) / (len(null_dist) + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load monkey metadata CSV.
    Expected columns: Name, Sex, Age, Rank, Group Name.
    Returns DataFrame with standardised column names.
    """
    meta = pd.read_csv(csv_path)
    meta = meta.rename(columns={
        'Name': 'MonkeyName',
        'Group Name': 'MonkeyGroup',
        'Group Category': 'GroupCategory',
    })
    meta['Rank'] = pd.to_numeric(meta['Rank'], errors='coerce')

    # Exclude subject monkey if flagged
    if 'Note' in meta.columns:
        is_subject = meta['Note'].fillna('').str.contains('subject', case=False)
        excluded = meta[is_subject]['MonkeyName'].tolist()
        if excluded:
            print(f"  Excluding subject monkey: {excluded}")
        meta = meta[~is_subject]

    print(f"Loaded metadata for {len(meta)} monkeys")
    return meta


def load_behavior_matrix(xlsx_path: str) -> pd.DataFrame:
    """Load an actor-receiver behaviour matrix from Excel."""
    df = pd.read_excel(xlsx_path)
    df.set_index(df.columns[0], inplace=True)
    df.index = [str(x).strip() for x in df.index]
    cleaned_cols = []
    for col in df.columns:
        col = str(col).strip()
        if col.startswith('Behavior Towards '):
            col = col.replace('Behavior Towards ', '')
        cleaned_cols.append(col.strip())
    df.columns = cleaned_cols
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE LABEL MAPPERS  (used by decode_social)
# ═══════════════════════════════════════════════════════════════════════════════

def map_labels(identity_labels, lookup: dict, key: str):
    """Generic: identity -> lookup[identity][key]."""
    return np.array([lookup.get(mid, {}).get(key, None) for mid in identity_labels])


def map_continuous(identity_labels, meta_lookup: dict, variable: str):
    """Map identities to a continuous variable (Age, Rank, …)."""
    values = []
    for mid in identity_labels:
        val = meta_lookup.get(mid, {}).get(variable, None)
        values.append(float(val) if val is not None else np.nan)
    return np.array(values)


def median_split(values, variable_name: str, verbose: bool = True):
    """Split continuous values into High/Low at median."""
    valid_mask = ~np.isnan(values)
    median_val = np.median(values[valid_mask])
    labels = np.full(len(values), None, dtype=object)
    labels[valid_mask & (values <= median_val)] = f'Low {variable_name}'
    labels[valid_mask & (values > median_val)] = f'High {variable_name}'
    if verbose:
        print(f"  Median split at {median_val:.2f}: "
              f"{np.sum(labels == f'Low {variable_name}')} low, "
              f"{np.sum(labels == f'High {variable_name}')} high")
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_permutation_histogram(ax, observed, null_dist, chance=None,
                               title='', xlabel='Accuracy'):
    """Draw a permutation-test histogram on an existing Axes."""
    ax.hist(null_dist, bins=50, color='gray', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Permuted')
    ax.axvline(observed, color='red', linewidth=2,
               label=f'Observed = {observed:.3f}')
    if chance is not None:
        ax.axvline(chance, color='blue', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()


def plot_confusion_matrix(ax, cm, labels, title=''):
    """Draw a normalised confusion matrix on an existing Axes."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)
    return im
