"""
Social variable decoding from pseudo-population neural activity.

Decodes the following social variables:
  1. Familiarity  — binary: familiar vs unfamiliar (from MonkeyGroup)
  2. Group membership — 4-class (from MonkeyGroup)
  3. Sex — binary (from CSV)
  4. Age — classification (median split) + Ridge regression (from CSV)
  5. Dominance rank — classification (median split) + Ridge regression (from CSV)

Requires a metadata CSV with columns: MonkeyName, Sex, Age, Rank
  - Sex: M or F
  - Age: numeric (years)
  - Rank: numeric (ordinal, e.g. 1 = most dominant)

Usage: Hardcode paths below for PyCharm, or uncomment argparse block for CLI.
"""

import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
N_CV_FOLDS = 5
N_PERMUTATIONS = 1000
N_PCA_COMPONENTS = 50
N_PSEUDO_DRAWS = 10
RANDOM_SEED = 42

# Familiarity mapping: which groups are familiar to the subject?
# (From CSV: Group Category = Familiar/Unfamiliar)
FAMILIAR_GROUPS = ['Zombies', 'Best Frans']
UNFAMILIAR_GROUPS = ['Instigators', 'Stranger Things']

# Current year for age computation from birth year
CURRENT_YEAR = 2023


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_all_trials(pkl_dir):
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")
    dfs = []
    for f in pkl_files:
        with open(f, 'rb') as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(pkl_files)} files -> {len(combined)} total trials")
    return combined


def compute_firing_rates(df):
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


def load_metadata(csv_path):
    """Load monkey metadata CSV. Expects columns: Name, Sex, Birth Year, Rank, Group Name."""
    meta = pd.read_csv(csv_path)

    # Standardize column names
    meta = meta.rename(columns={'Name': 'MonkeyName', 'Group Name': 'MonkeyGroup',
                                'Group Category': 'GroupCategory'})

    # Compute age from birth year
    meta['Age'] = CURRENT_YEAR - meta['Birth Year']

    # Clean up rank — convert to numeric, NaN for missing
    meta['Rank'] = pd.to_numeric(meta['Rank'], errors='coerce')

    # Exclude subject monkey if present
    if 'Note' in meta.columns:
        is_subject = meta['Note'].fillna('').str.contains('subject', case=False)
        excluded = meta[is_subject]['MonkeyName'].tolist()
        if excluded:
            print(f"  Excluding subject monkey: {excluded}")
        meta = meta[~is_subject]

    print(f"Loaded metadata for {len(meta)} monkeys")
    print(f"  Sex: {meta['Sex'].value_counts().to_dict()}")
    print(f"  Age range: {meta['Age'].min():.0f} - {meta['Age'].max():.0f} years")
    n_with_rank = meta['Rank'].notna().sum()
    print(f"  Rank available: {n_with_rank}/{len(meta)} monkeys")
    print(f"  Groups: {meta['MonkeyGroup'].value_counts().to_dict()}")
    return meta


def filter_neurons_by_min_trials(df, target_identities, min_trials):
    """Keep only neurons with >= min_trials for EVERY target identity."""
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(target_identities)
    kept = []
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        if all(len(neuron_df[neuron_df['MonkeyName'] == mid]) >= min_trials
               for mid in identities):
            kept.append(nid)
    print(f"\n  min_trials filter (>= {min_trials} per identity):")
    print(f"    Kept: {len(kept)} / {len(neuron_ids)} neurons")
    print(f"    Dropped: {len(neuron_ids) - len(kept)}")
    df_filtered = df[df['NeuronID'].isin(kept)]
    return df_filtered, kept


# ─── PSEUDO-POPULATION BUILDER ───────────────────────────────────────────────

def build_pseudo_population_by_identity(df, identities, n_draws=N_PSEUDO_DRAWS,
                                         seed=RANDOM_SEED):
    """
    Build pseudo-population trial matrices aligned by identity.
    Returns X_draws (list of arrays), identity labels per row.
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(df['NeuronID'].unique())

    # Collect rates
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
    valid_ids = [mid for mid in identities if int(min_trials_per_id[mid]) > 0]
    if len(valid_ids) < len(identities):
        excluded = set(identities) - set(valid_ids)
        print(f"  Excluded {len(excluded)} identities with 0 trials: {excluded}")

    total_trials = sum(int(min_trials_per_id[mid]) for mid in valid_ids)
    print(f"  Valid identities: {len(valid_ids)}, total trials per draw: {total_trials}")

    X_draws = []
    identity_labels_draws = []

    for _ in range(n_draws):
        X_rows = []
        id_labels = []
        for mid in valid_ids:
            nt = int(min_trials_per_id[mid])
            block = np.zeros((nt, len(neuron_ids)))
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=nt, replace=False)
                block[:, j] = rates[chosen]
            X_rows.append(block)
            id_labels.extend([mid] * nt)
        X_draws.append(np.vstack(X_rows))
        identity_labels_draws.append(np.array(id_labels))

    return X_draws, identity_labels_draws, neuron_ids, valid_ids


# ─── LABEL MAPPERS ───────────────────────────────────────────────────────────

def map_familiarity_labels(identity_labels, group_lookup):
    """Map identity labels to familiar/unfamiliar."""
    labels = []
    for mid in identity_labels:
        group = group_lookup.get(mid, None)
        if group in FAMILIAR_GROUPS:
            labels.append('Familiar')
        elif group in UNFAMILIAR_GROUPS:
            labels.append('Unfamiliar')
        else:
            labels.append(None)
    return np.array(labels)


def map_group_labels(identity_labels, group_lookup):
    """Map identity labels to group name."""
    return np.array([group_lookup.get(mid, None) for mid in identity_labels])


def map_sex_labels(identity_labels, meta_lookup):
    """Map identity labels to sex."""
    return np.array([meta_lookup.get(mid, {}).get('Sex', None) for mid in identity_labels])


def map_continuous_variable(identity_labels, meta_lookup, variable):
    """Map identity labels to a continuous variable (Age or Rank)."""
    values = []
    for mid in identity_labels:
        val = meta_lookup.get(mid, {}).get(variable, None)
        values.append(float(val) if val is not None else np.nan)
    return np.array(values)


def median_split_labels(values, variable_name):
    """Split continuous values into High/Low at median."""
    valid_mask = ~np.isnan(values)
    median_val = np.median(values[valid_mask])
    labels = np.full(len(values), None, dtype=object)
    labels[valid_mask & (values <= median_val)] = f'Low {variable_name}'
    labels[valid_mask & (values > median_val)] = f'High {variable_name}'
    print(f"  Median split at {median_val:.2f}: "
          f"{np.sum(labels == f'Low {variable_name}')} low, "
          f"{np.sum(labels == f'High {variable_name}')} high")
    return labels


# ─── CLASSIFICATION DECODING ─────────────────────────────────────────────────

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


def run_classification(X, y, n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    """Stratified k-fold classification with class balancing.
    Downsamples to equal class sizes before CV so accuracy is interpretable
    at a true 1/n_classes chance level."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Balance classes by downsampling
    X_bal, y_bal = balance_classes(X, y_enc, seed)

    min_class_count = min(np.bincount(y_bal))
    n_folds = min(N_CV_FOLDS, min_class_count)
    if n_folds < 2:
        print(f"    WARNING: smallest class has {min_class_count} samples after balancing, cannot CV")
        return 0.0, 0.0, le

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    fold_bal_accs = []

    for train_idx, test_idx in skf.split(X_bal, y_bal):
        X_train, X_test = X_bal[train_idx], X_bal[test_idx]
        y_train, y_test = y_bal[train_idx], y_bal[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if n_pca is not None and n_pca < X_train.shape[1]:
            n_comp = min(n_pca, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

        clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                 max_iter=2000, multi_class='multinomial',
                                 random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))

    return np.mean(fold_accs), np.mean(fold_bal_accs), le


def run_classification_permutation(X, y, n_perms, n_pca, seed):
    rng = np.random.default_rng(seed)
    perm_accs = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i+1}/{n_perms}")
        y_shuf = rng.permutation(y)
        acc, _, _ = run_classification(X, y_shuf, n_pca=n_pca, seed=seed + i)
        perm_accs.append(acc)
    return np.array(perm_accs)


# ─── REGRESSION DECODING ─────────────────────────────────────────────────────

def run_regression(X, y_continuous, n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    """K-fold Ridge regression. Returns mean R^2."""
    valid_mask = ~np.isnan(y_continuous)
    X_valid = X[valid_mask]
    y_valid = y_continuous[valid_mask]

    if len(y_valid) < 10:
        print("    WARNING: too few valid samples for regression")
        return 0.0

    kf = KFold(n_splits=min(N_CV_FOLDS, len(y_valid)), shuffle=True, random_state=seed)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X_valid):
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if n_pca is not None and n_pca < X_train.shape[1]:
            n_comp = min(n_pca, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        fold_r2s.append(r2_score(y_test, y_pred))

    return np.mean(fold_r2s)


def run_regression_permutation(X, y_continuous, n_perms, n_pca, seed):
    rng = np.random.default_rng(seed)
    valid_mask = ~np.isnan(y_continuous)
    y_valid = y_continuous[valid_mask]

    perm_r2s = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i+1}/{n_perms}")
        y_shuf = rng.permutation(y_valid)
        y_full_shuf = y_continuous.copy()
        y_full_shuf[valid_mask] = y_shuf
        r2 = run_regression(X, y_full_shuf, n_pca=n_pca, seed=seed + i)
        perm_r2s.append(r2)
    return np.array(perm_r2s)


# ─── MAIN DECODING LOOP ──────────────────────────────────────────────────────

def decode_variable(variable_name, X_draws, labels_draws, n_pca, is_regression=False,
                    run_perms=False, n_perms=N_PERMUTATIONS,
                    continuous_draws=None):
    """
    Run decoding for a single variable across pseudo-population draws.

    For classification: labels_draws is a list of label arrays.
    For regression: continuous_draws is a list of continuous value arrays.
    """
    print(f"\n  --- {variable_name} {'(regression)' if is_regression else '(classification)'} ---")

    draw_scores = []

    for d in range(len(X_draws)):
        if is_regression:
            y = continuous_draws[d]
            valid = ~np.isnan(y)
            if np.sum(valid) < 10:
                print(f"    Draw {d+1}: too few valid samples, skipping")
                continue
            score = run_regression(X_draws[d], y, n_pca=n_pca, seed=RANDOM_SEED + d)
            draw_scores.append(score)
        else:
            y = labels_draws[d]
            valid = y != None  # noqa
            if np.sum(valid) < 10:
                print(f"    Draw {d+1}: too few valid samples, skipping")
                continue
            X_valid = X_draws[d][valid]
            y_valid = y[valid]
            acc, bal_acc, le = run_classification(X_valid, y_valid, n_pca=n_pca,
                                                   seed=RANDOM_SEED + d)
            draw_scores.append(acc)

    if not draw_scores:
        print("    No valid draws!")
        return None

    mean_score = np.mean(draw_scores)
    std_score = np.std(draw_scores)
    metric = 'R^2' if is_regression else 'accuracy'
    print(f"    Mean {metric}: {mean_score:.4f} +/-{std_score:.4f}")

    # Permutation test (on first valid draw)
    perm_scores = None
    p_value = None
    if run_perms:
        print(f"    Running permutation test ({n_perms} permutations)...")
        if is_regression:
            perm_scores = run_regression_permutation(
                X_draws[0], continuous_draws[0], n_perms=n_perms,
                n_pca=n_pca, seed=RANDOM_SEED + 9999
            )
        else:
            y = labels_draws[0]
            valid = y != None  # noqa
            perm_scores = run_classification_permutation(
                X_draws[0][valid], y[valid], n_perms=n_perms,
                n_pca=n_pca, seed=RANDOM_SEED + 9999
            )
        p_value = (np.sum(perm_scores >= mean_score) + 1) / (n_perms + 1)
        print(f"    Permutation p-value: {p_value:.4f}")

    # Determine chance level
    if is_regression:
        chance = 0.0
    else:
        y_example = labels_draws[0]
        valid = y_example != None  # noqa
        y_valid = y_example[valid]
        unique, counts = np.unique(y_valid, return_counts=True)
        chance = 1.0 / len(unique)

    return {
        'variable': variable_name,
        'type': 'regression' if is_regression else 'classification',
        'mean_score': mean_score,
        'std_score': std_score,
        'draw_scores': draw_scores,
        'chance': chance,
        'perm_scores': perm_scores,
        'p_value': p_value,
    }


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_all_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Separate classification and regression results
    clf_results = {k: v for k, v in all_results.items()
                   if v is not None and v['type'] == 'classification'}
    reg_results = {k: v for k, v in all_results.items()
                   if v is not None and v['type'] == 'regression'}

    # ── 1. Classification summary bar plot ──
    if clf_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(clf_results.keys())
        means = [clf_results[n]['mean_score'] for n in names]
        stds = [clf_results[n]['std_score'] for n in names]
        chances = [clf_results[n]['chance'] for n in names]

        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                  '#937860', '#DA8BC3', '#8C8C8C']
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors[:len(names)], edgecolor='black', linewidth=0.8,
                      alpha=0.7)

        # Plot chance levels
        for i, ch in enumerate(chances):
            ax.plot([i - 0.3, i + 0.3], [ch, ch], 'k--', linewidth=1)

        # Add p-values
        for i, name in enumerate(names):
            r = clf_results[name]
            label = f'{means[i]:.3f}'
            if r['p_value'] is not None:
                p = r['p_value']
                p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                label += f'\n{p_str}'
            ax.text(i, means[i] + stds[i] + 0.01, label,
                    ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Social Variable Decoding (Classification)')
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) + 0.1)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'social_decoding_classification.png'), dpi=200)
        plt.close()
        print(f"  Saved: social_decoding_classification.png")

    # ── 2. Regression summary bar plot ──
    if reg_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(reg_results.keys())
        means = [reg_results[n]['mean_score'] for n in names]
        stds = [reg_results[n]['std_score'] for n in names]

        colors = ['#55A868', '#C44E52']
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors[:len(names)], edgecolor='black', linewidth=0.8,
                      alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='Chance (R^2 = 0)')

        for i, name in enumerate(names):
            r = reg_results[name]
            label = f'{means[i]:.3f}'
            if r['p_value'] is not None:
                p = r['p_value']
                p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                label += f'\n{p_str}'
            y_pos = max(means[i] + stds[i], 0) + 0.01
            ax.text(i, y_pos, label, ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('R^2')
        ax.set_title('Social Variable Decoding (Regression)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'social_decoding_regression.png'), dpi=200)
        plt.close()
        print(f"  Saved: social_decoding_regression.png")

    # ── 3. Permutation distributions ──
    for name, r in all_results.items():
        if r is None or r['perm_scores'] is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(r['perm_scores'], bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(r['mean_score'], color='red', linewidth=2,
                   label=f'Observed = {r["mean_score"]:.3f}')

        metric = 'R^2' if r['type'] == 'regression' else 'Accuracy'
        if r['type'] == 'classification':
            ax.axvline(r['chance'], color='blue', linestyle='--', linewidth=1,
                       label=f'Chance = {r["chance"]:.3f}')

        p = r['p_value']
        p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        ax.set_title(f'{name} ({p_str})')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        fname = f'permutation_{name.replace(" ", "_").replace("/", "_")}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()
        print(f"  Saved: {fname}")

    print(f"\nAll figures saved to {output_dir}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ─── HARDCODED CONFIG (for PyCharm) ──────────────────────────────────────
    pkl_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered"
    metadata_csv = r"/home/connorlab/Downloads/monkeyinfo.csv"
    output_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/social_decoding_age_sex_balanced_final"
    skip_perm = False       # set False to run permutation tests
    n_perm = 1000
    n_draws = 10
    n_pca = 50             # set to 0 to skip PCA
    min_trials = 10         # set > 0 to filter neurons
    # ─────────────────────────────────────────────────────────────────────────

    n_pca = n_pca if n_pca > 0 else None

    # ── Load data ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)
    df = df[df['MonkeyName'] != 'NewMonkey']
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Load metadata ──
    print("\nLoading metadata...")
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict()
                   for _, row in meta.iterrows()}

    # ── Group lookup ──
    group_lookup = dict(zip(df['MonkeyName'], df['MonkeyGroup']))

    # ── Compute firing rates ──
    print("\nComputing trial-level firing rates (full epoch)...")
    df = compute_firing_rates(df)

    # ── All identities ──
    all_identities = sorted(df['MonkeyName'].unique())
    print(f"\nTotal identities: {len(all_identities)}")

    # Check metadata coverage
    missing = [mid for mid in all_identities if mid not in meta_lookup]
    if missing:
        print(f"  WARNING: {len(missing)} identities missing from metadata CSV: {missing}")
        print(f"  These will be excluded from Sex/Age/Rank decoding.")

    # ── Filter neurons ──
    if min_trials > 0:
        df, _ = filter_neurons_by_min_trials(df, all_identities, min_trials)

    # ── Build pseudo-population ──
    print(f"\n{'=' * 60}")
    print("BUILDING PSEUDO-POPULATION")
    print(f"{'=' * 60}")
    X_draws, id_labels_draws, neuron_ids, valid_ids = build_pseudo_population_by_identity(
        df, all_identities, n_draws=n_draws
    )
    n_neurons = len(neuron_ids)
    print(f"Neurons: {n_neurons}")
    print(f"Feature matrix shape: {X_draws[0].shape}")

    # ── Prepare labels for each variable ──
    all_results = {}

    # --- 1. Familiarity ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 1: FAMILIARITY")
    print(f"{'=' * 60}")
    fam_labels_draws = []
    for id_labels in id_labels_draws:
        labels = map_familiarity_labels(id_labels, group_lookup)
        fam_labels_draws.append(labels)

    # Check label distribution
    example_labels = fam_labels_draws[0]
    valid = example_labels != None  # noqa
    unique, counts = np.unique(example_labels[valid], return_counts=True)
    print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Familiarity'] = decode_variable(
        'Familiarity', X_draws,
        labels_draws=[np.where(l == None, None, l) for l in fam_labels_draws],
        n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
    )

    # --- 1b. Familiarity (females only — controlling for sex confound) ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 1b: FAMILIARITY (FEMALES ONLY)")
    print(f"{'=' * 60}")

    female_monkeys = [mid for mid in all_identities
                      if meta_lookup.get(mid, {}).get('Sex') == 'F']
    print(f"  Female monkeys: {len(female_monkeys)}")
    fam_counts = {'Familiar': 0, 'Unfamiliar': 0}
    for mid in female_monkeys:
        grp = group_lookup.get(mid)
        if grp in FAMILIAR_GROUPS:
            fam_counts['Familiar'] += 1
        elif grp in UNFAMILIAR_GROUPS:
            fam_counts['Unfamiliar'] += 1
    print(f"  Familiar: {fam_counts['Familiar']}, Unfamiliar: {fam_counts['Unfamiliar']}")

    if min(fam_counts.values()) >= 3:
        df_female = df[df['MonkeyName'].isin(female_monkeys)]
        if min_trials > 0:
            df_female, _ = filter_neurons_by_min_trials(df_female, female_monkeys, min_trials)

        X_fem_draws, fem_id_labels_draws, fem_neuron_ids, fem_valid_ids = \
            build_pseudo_population_by_identity(
                df_female, female_monkeys, n_draws=n_draws
            )
        print(f"  Female pseudo-population: {X_fem_draws[0].shape}")

        fem_fam_labels_draws = []
        for id_labels in fem_id_labels_draws:
            labels = map_familiarity_labels(id_labels, group_lookup)
            fem_fam_labels_draws.append(labels)

        example_labels = fem_fam_labels_draws[0]
        valid = example_labels != None  # noqa
        unique, counts = np.unique(example_labels[valid], return_counts=True)
        print(f"  Classes: {dict(zip(unique, counts))}")

        all_results['Familiarity (F only)'] = decode_variable(
            'Familiarity (F only)', X_fem_draws,
            labels_draws=[np.where(l == None, None, l) for l in fem_fam_labels_draws],
            n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
        )
    else:
        print("  Too few females in one category, skipping")
        all_results['Familiarity (F only)'] = None

    # --- 2. Group membership ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 2: GROUP MEMBERSHIP")
    print(f"{'=' * 60}")
    group_labels_draws = []
    for id_labels in id_labels_draws:
        labels = map_group_labels(id_labels, group_lookup)
        group_labels_draws.append(labels)

    example_labels = group_labels_draws[0]
    valid = example_labels != None  # noqa
    unique, counts = np.unique(example_labels[valid], return_counts=True)
    print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Group'] = decode_variable(
        'Group', X_draws, labels_draws=group_labels_draws,
        n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
    )

    # --- 3. Sex ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 3: SEX")
    print(f"{'=' * 60}")
    sex_labels_draws = []
    for id_labels in id_labels_draws:
        labels = map_sex_labels(id_labels, meta_lookup)
        sex_labels_draws.append(labels)

    example_labels = sex_labels_draws[0]
    valid = example_labels != None  # noqa
    if np.sum(valid) > 0:
        unique, counts = np.unique(example_labels[valid], return_counts=True)
        print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Sex'] = decode_variable(
        'Sex', X_draws, labels_draws=sex_labels_draws,
        n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
    )

    # --- 4. Age ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 4: AGE")
    print(f"{'=' * 60}")

    # 4a. Classification: Adult (>= 4 years) vs Juvenile (< 4 years)
    ADULT_AGE_CUTOFF = 4
    print(f"  Adult/Juvenile cutoff: >= {ADULT_AGE_CUTOFF} years = Adult")

    age_cont_draws = []
    age_class_draws = []
    for id_labels in id_labels_draws:
        ages = map_continuous_variable(id_labels, meta_lookup, 'Age')
        age_cont_draws.append(ages)
        # Biological age class
        labels = np.full(len(ages), None, dtype=object)
        valid_mask = ~np.isnan(ages)
        labels[valid_mask & (ages >= ADULT_AGE_CUTOFF)] = 'Adult'
        labels[valid_mask & (ages < ADULT_AGE_CUTOFF)] = 'Juvenile'
        age_class_draws.append(labels)

    example_labels = age_class_draws[0]
    valid = example_labels != None  # noqa
    unique, counts = np.unique(example_labels[valid], return_counts=True)
    print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Age (class)'] = decode_variable(
        'Age (class)', X_draws, labels_draws=age_class_draws,
        n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
    )

    # 4b. Age classification — females only (controlling for adult male confound)
    print(f"\n{'=' * 60}")
    print("VARIABLE 4b: AGE (FEMALES ONLY)")
    print(f"{'=' * 60}")

    female_monkeys = [mid for mid in all_identities
                      if meta_lookup.get(mid, {}).get('Sex') == 'F']
    # Count adults vs juveniles among females
    fem_age_counts = {'Adult': 0, 'Juvenile': 0}
    for mid in female_monkeys:
        age = meta_lookup.get(mid, {}).get('Age')
        if age is not None and not np.isnan(float(age)):
            if float(age) >= ADULT_AGE_CUTOFF:
                fem_age_counts['Adult'] += 1
            else:
                fem_age_counts['Juvenile'] += 1
    print(f"  Female monkeys: {len(female_monkeys)}")
    print(f"  Adult: {fem_age_counts['Adult']}, Juvenile: {fem_age_counts['Juvenile']}")

    if min(fem_age_counts.values()) >= 3:
        # Reuse df_female if it exists from familiarity control, otherwise build it
        df_female_age = df[df['MonkeyName'].isin(female_monkeys)]
        if min_trials > 0:
            df_female_age, _ = filter_neurons_by_min_trials(
                df_female_age, female_monkeys, min_trials)

        X_fem_age_draws, fem_age_id_draws, _, _ = build_pseudo_population_by_identity(
            df_female_age, female_monkeys, n_draws=n_draws
        )
        print(f"  Female pseudo-population: {X_fem_age_draws[0].shape}")

        fem_age_class_draws = []
        for id_labels in fem_age_id_draws:
            ages = map_continuous_variable(id_labels, meta_lookup, 'Age')
            labels = np.full(len(ages), None, dtype=object)
            valid_mask = ~np.isnan(ages)
            labels[valid_mask & (ages >= ADULT_AGE_CUTOFF)] = 'Adult'
            labels[valid_mask & (ages < ADULT_AGE_CUTOFF)] = 'Juvenile'
            fem_age_class_draws.append(labels)

        example_labels = fem_age_class_draws[0]
        valid = example_labels != None  # noqa
        unique, counts = np.unique(example_labels[valid], return_counts=True)
        print(f"  Classes: {dict(zip(unique, counts))}")

        all_results['Age (F only)'] = decode_variable(
            'Age (F only)', X_fem_age_draws, labels_draws=fem_age_class_draws,
            n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
        )
    else:
        print("  Too few females in one age category, skipping")
        all_results['Age (F only)'] = None

    # 4c. Sex x Age: Adult Male, Adult Female, Juvenile (3 classes)
    print(f"\n{'=' * 60}")
    print("VARIABLE 4c: SEX x AGE")
    print(f"{'=' * 60}")

    sex_age_draws = []
    for id_labels in id_labels_draws:
        ages = map_continuous_variable(id_labels, meta_lookup, 'Age')
        sexes = map_sex_labels(id_labels, meta_lookup)
        labels = np.full(len(ages), None, dtype=object)
        for i in range(len(ages)):
            if sexes[i] is None or np.isnan(ages[i]):
                continue
            if ages[i] < ADULT_AGE_CUTOFF:
                labels[i] = 'Juvenile'
            elif sexes[i] == 'M':
                labels[i] = 'Adult Male'
            else:
                labels[i] = 'Adult Female'
        sex_age_draws.append(labels)

    example_labels = sex_age_draws[0]
    valid = example_labels != None  # noqa
    unique, counts = np.unique(example_labels[valid], return_counts=True)
    print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Sex x Age'] = decode_variable(
        'Sex x Age', X_draws, labels_draws=sex_age_draws,
        n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
    )

    # 4d. Regression (continuous age)
    all_results['Age (regression)'] = decode_variable(
        'Age (regression)', X_draws, labels_draws=None,
        n_pca=n_pca, is_regression=True,
        continuous_draws=age_cont_draws,
        run_perms=(not skip_perm), n_perms=n_perm
    )

    # --- 5. Dominance Rank (Zombies + Best Frans only) ---
    print(f"\n{'=' * 60}")
    print("VARIABLE 5: DOMINANCE RANK (familiar groups only)")
    print(f"{'=' * 60}")

    # Build separate pseudo-population for ranked monkeys only
    ranked_monkeys = [mid for mid in all_identities
                      if meta_lookup.get(mid, {}).get('Rank') is not None
                      and not np.isnan(float(meta_lookup[mid]['Rank']))
                      and group_lookup.get(mid) in FAMILIAR_GROUPS]
    print(f"  Monkeys with rank in familiar groups: {len(ranked_monkeys)}")
    print(f"    {', '.join(sorted(ranked_monkeys))}")

    if len(ranked_monkeys) >= 4:
        # Filter df to only ranked monkeys for pseudo-population
        df_ranked = df[df['MonkeyName'].isin(ranked_monkeys)]
        if min_trials > 0:
            df_ranked, _ = filter_neurons_by_min_trials(df_ranked, ranked_monkeys, min_trials)

        X_rank_draws, rank_id_labels_draws, rank_neuron_ids, rank_valid_ids = \
            build_pseudo_population_by_identity(
                df_ranked, ranked_monkeys, n_draws=n_draws
            )
        print(f"  Rank pseudo-population: {X_rank_draws[0].shape}")

        # 5a. Classification (median split)
        rank_cont_draws = []
        rank_class_draws = []
        for id_labels in rank_id_labels_draws:
            ranks = map_continuous_variable(id_labels, meta_lookup, 'Rank')
            rank_cont_draws.append(ranks)
            rank_labels = median_split_labels(ranks, 'Rank')
            rank_class_draws.append(rank_labels)

        all_results['Rank (class)'] = decode_variable(
            'Rank (class)', X_rank_draws, labels_draws=rank_class_draws,
            n_pca=n_pca, run_perms=(not skip_perm), n_perms=n_perm
        )

        # 5b. Regression
        all_results['Rank (regression)'] = decode_variable(
            'Rank (regression)', X_rank_draws, labels_draws=None,
            n_pca=n_pca, is_regression=True,
            continuous_draws=rank_cont_draws,
            run_perms=(not skip_perm), n_perms=n_perm
        )
    else:
        print("  Too few ranked monkeys, skipping rank decoding")
        all_results['Rank (class)'] = None
        all_results['Rank (regression)'] = None

    # ── Overall summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {n_neurons}")
    if min_trials > 0:
        print(f"  (filtered: min_trials >= {min_trials})")
    print()

    for name, r in all_results.items():
        if r is None:
            print(f"  {name:25s}: FAILED")
            continue
        metric = 'R^2' if r['type'] == 'regression' else 'acc'
        chance_str = f"chance={r['chance']:.3f}" if r['type'] == 'classification' else 'chance=0'
        p_str = f", p={r['p_value']:.4f}" if r['p_value'] is not None else ""
        print(f"  {name:25s}: {metric}={r['mean_score']:.4f} +/-{r['std_score']:.4f} "
              f"({chance_str}{p_str})")

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    plot_all_results(all_results, output_dir)

    save_path = os.path.join(output_dir, 'social_decoding_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
