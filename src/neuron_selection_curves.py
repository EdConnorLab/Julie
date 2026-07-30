"""
Neuron Selection Curve Analysis.

For each variable of interest, rank neurons by their individual sensitivity
(using one half of trials), then evaluate RSA or decoding performance at
increasing neuron fractions (using the other half of trials).

Demonstrates sparse vs distributed coding:
  - Social variables (submission, affiliation): expected to peak at small fraction
  - Perceptual variables (identity, age): expected to rise steadily

Usage: Hardcode paths below for PyCharm.
"""

import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, f_oneway, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
SUBJECT_MONKEY = '81G'
CURRENT_YEAR = 2023
ADULT_AGE_CUTOFF = 4

# Neuron fractions to evaluate
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

# Number of split-half repetitions to stabilize results
N_SPLIT_REPS = 20

# Decoding settings
N_CV_FOLDS = 5
N_PCA_COMPONENTS = 50

# RSA settings
N_RSA_DRAWS = 20  # pseudo-population draws for neural RDM


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


def load_behavior_matrix(xlsx_path):
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


def load_metadata(csv_path):
    meta = pd.read_csv(csv_path)
    meta = meta.rename(columns={'Name': 'MonkeyName', 'Group Name': 'MonkeyGroup'})
    meta['Age'] = CURRENT_YEAR - meta['Birth Year']
    if 'Note' in meta.columns:
        is_subject = meta['Note'].fillna('').str.contains('subject', case=False)
        meta = meta[~is_subject]
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}
    return meta, meta_lookup


def filter_neurons_by_min_trials(df, target_identities, min_trials):
    neuron_ids = sorted(df['NeuronID'].unique())
    kept = []
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        if all(len(neuron_df[neuron_df['MonkeyName'] == mid]) >= min_trials
               for mid in target_identities):
            kept.append(nid)
    print(f"  min_trials filter (>= {min_trials}): kept {len(kept)}/{len(neuron_ids)}")
    df_filtered = df[df['NeuronID'].isin(kept)]
    return df_filtered, kept


# ─── TRIAL SPLITTING ─────────────────────────────────────────────────────────

def split_trials_half(df, seed):
    """
    Split each neuron x identity's trials into two halves.
    Returns two DataFrames with a 'half' column.
    """
    rng = np.random.default_rng(seed)
    half_labels = []

    for (nid, mname), group in df.groupby(['NeuronID', 'MonkeyName']):
        n = len(group)
        perm = rng.permutation(n)
        labels = np.zeros(n, dtype=int)
        labels[perm[:n // 2]] = 0  # ranking half
        labels[perm[n // 2:]] = 1  # evaluation half
        half_labels.extend(labels)

    df = df.copy()
    df['half'] = half_labels
    df_rank = df[df['half'] == 0].copy()
    df_eval = df[df['half'] == 1].copy()
    return df_rank, df_eval


# ─── NEURON RANKING FUNCTIONS ────────────────────────────────────────────────

def rank_neurons_identity(df_rank, identities):
    """
    Rank neurons by how well they discriminate individual identities.
    Uses one-way ANOVA F-statistic across identity groups.
    """
    neuron_ids = sorted(df_rank['NeuronID'].unique())
    scores = {}

    for nid in neuron_ids:
        ndf = df_rank[df_rank['NeuronID'] == nid]
        groups = []
        for mid in identities:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            if len(rates) >= 2:
                groups.append(rates)

        if len(groups) >= 2:
            try:
                f_stat, _ = f_oneway(*groups)
                scores[nid] = f_stat if not np.isnan(f_stat) else 0.0
            except:
                scores[nid] = 0.0
        else:
            scores[nid] = 0.0

    # Sort descending by F-statistic
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, scores


def rank_neurons_age(df_rank, meta_lookup, identities):
    """
    Rank neurons by sensitivity to adult vs juvenile (females only).
    Uses point-biserial correlation between firing rate and age class.
    """
    neuron_ids = sorted(df_rank['NeuronID'].unique())

    # Determine age class per identity (females only)
    female_age_class = {}
    for mid in identities:
        info = meta_lookup.get(mid, {})
        if info.get('Sex') == 'F' and info.get('Age') is not None:
            age = float(info['Age'])
            female_age_class[mid] = 1 if age >= ADULT_AGE_CUTOFF else 0

    if len(set(female_age_class.values())) < 2:
        print("    WARNING: not enough age variation in females")
        return list(neuron_ids), {nid: 0.0 for nid in neuron_ids}

    scores = {}
    for nid in neuron_ids:
        ndf = df_rank[df_rank['NeuronID'] == nid]
        rates_all = []
        labels_all = []
        for mid, age_label in female_age_class.items():
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            rates_all.extend(rates)
            labels_all.extend([age_label] * len(rates))

        if len(set(labels_all)) >= 2 and len(rates_all) >= 5:
            try:
                r, _ = pointbiserialr(labels_all, rates_all)
                scores[nid] = abs(r) if not np.isnan(r) else 0.0
            except:
                scores[nid] = 0.0
        else:
            scores[nid] = 0.0

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, scores


def rank_neurons_social_rsa(df_rank, stimulus_monkeys, behavior_rdm):
    """
    Rank neurons by how well their individual response pattern matches
    the behavioral RDM. For each neuron, compute a single-neuron RDM
    (|mean_rate_i - mean_rate_j| for all pairs) and correlate with
    the behavioral RDM.
    """
    neuron_ids = sorted(df_rank['NeuronID'].unique())
    monkeys = sorted(stimulus_monkeys)
    n_monkeys = len(monkeys)
    behav_vec = behavior_rdm[np.triu_indices(n_monkeys, k=1)]

    scores = {}
    for nid in neuron_ids:
        ndf = df_rank[df_rank['NeuronID'] == nid]
        mean_rates = []
        valid = True
        for mid in monkeys:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            if len(rates) == 0:
                valid = False
                break
            mean_rates.append(np.mean(rates))

        if not valid:
            scores[nid] = 0.0
            continue

        # Single-neuron RDM: absolute difference in mean rates
        mean_rates = np.array(mean_rates)
        neuron_rdm = np.abs(mean_rates[:, None] - mean_rates[None, :])
        neuron_vec = neuron_rdm[np.triu_indices(n_monkeys, k=1)]

        if np.std(neuron_vec) == 0 or np.std(behav_vec) == 0:
            scores[nid] = 0.0
            continue

        rho, _ = spearmanr(neuron_vec, behav_vec)
        scores[nid] = abs(rho) if not np.isnan(rho) else 0.0

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, scores


# ─── EVALUATION FUNCTIONS ────────────────────────────────────────────────────

def eval_identity_decoding(df_eval, identities, neuron_subset, n_pca, seed):
    """
    Evaluate identity decoding accuracy using a neuron subset.
    Uses pseudo-population with balanced trial counts.
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(neuron_subset)

    if len(neuron_ids) == 0:
        return 0.0

    # Collect rates and find min trials
    neuron_identity_rates = {}
    min_trials = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        ndf = df_eval[df_eval['NeuronID'] == nid]
        for mid in identities:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = rates
            if len(rates) < min_trials[mid]:
                min_trials[mid] = len(rates)

    valid_ids = [mid for mid in identities if int(min_trials[mid]) >= 2]
    if len(valid_ids) < 3:
        return 0.0

    # Build pseudo-population matrix
    X_rows = []
    y_rows = []
    for mid in valid_ids:
        nt = int(min_trials[mid])
        block = np.zeros((nt, len(neuron_ids)))
        for j, nid in enumerate(neuron_ids):
            rates = neuron_identity_rates[(nid, mid)]
            chosen = rng.choice(len(rates), size=nt, replace=False)
            block[:, j] = rates[chosen]
        X_rows.append(block)
        y_rows.append(np.full(nt, mid))

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    # Decode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    min_class = min(np.bincount(y_enc))
    n_folds = min(N_CV_FOLDS, min_class)
    if n_folds < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

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
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_accs.append(accuracy_score(y_test, y_pred))
        except ValueError:
            continue

    if not fold_accs:
        return 0.5

    return np.mean(fold_accs)


def eval_age_decoding(df_eval, meta_lookup, identities, neuron_subset, n_pca, seed):
    """
    Evaluate adult vs juvenile decoding (females only) using a neuron subset.
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(neuron_subset)

    if len(neuron_ids) == 0:
        return 0.0

    # Determine age class for female identities
    female_age = {}
    for mid in identities:
        info = meta_lookup.get(mid, {})
        if info.get('Sex') == 'F' and info.get('Age') is not None:
            age = float(info['Age'])
            female_age[mid] = 'Adult' if age >= ADULT_AGE_CUTOFF else 'Juvenile'

    female_ids = sorted(female_age.keys())
    if len(set(female_age.values())) < 2:
        return 0.0

    # Build pseudo-population
    neuron_identity_rates = {}
    min_trials = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        ndf = df_eval[df_eval['NeuronID'] == nid]
        for mid in female_ids:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = rates
            if len(rates) < min_trials[mid]:
                min_trials[mid] = len(rates)

    valid_ids = [mid for mid in female_ids if int(min_trials[mid]) >= 2]
    if len(valid_ids) < 4:
        return 0.0

    X_rows = []
    y_rows = []
    for mid in valid_ids:
        nt = int(min_trials[mid])
        block = np.zeros((nt, len(neuron_ids)))
        for j, nid in enumerate(neuron_ids):
            rates = neuron_identity_rates[(nid, mid)]
            chosen = rng.choice(len(rates), size=nt, replace=False)
            block[:, j] = rates[chosen]
        X_rows.append(block)
        y_rows.append(np.full(nt, female_age[mid]))

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    # Balance classes
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes, counts = np.unique(y_enc, return_counts=True)
    min_count = min(counts)
    balanced_idx = []
    for c in classes:
        c_idx = np.where(y_enc == c)[0]
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        balanced_idx.extend(chosen)
    balanced_idx = np.array(sorted(balanced_idx))
    X = X[balanced_idx]
    y_enc = y_enc[balanced_idx]

    min_class = min(np.bincount(y_enc))
    n_folds = min(N_CV_FOLDS, min_class)
    if n_folds < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

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
                                 max_iter=2000, random_state=seed)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_accs.append(accuracy_score(y_test, y_pred))
        except ValueError:
            continue

    if not fold_accs:
        return 0.5
    return np.mean(fold_accs)


def eval_rsa(df_eval, stimulus_monkeys, neuron_subset, behavioral_rdm,
             n_draws=N_RSA_DRAWS, seed=RANDOM_SEED):
    """
    Evaluate RSA correlation using a neuron subset.
    Build neural RDM from subset, correlate with behavioral RDM.
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(neuron_subset)
    monkeys = sorted(stimulus_monkeys)
    n_monkeys = len(monkeys)

    if len(neuron_ids) == 0:
        return 0.0

    # Collect rates
    neuron_identity_rates = {}
    min_trials = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        ndf = df_eval[df_eval['NeuronID'] == nid]
        for mid in monkeys:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = rates
            if len(rates) < min_trials[mid]:
                min_trials[mid] = len(rates)

    valid_monkeys = [m for m in monkeys if int(min_trials[m]) >= 2]
    if len(valid_monkeys) < 4:
        return 0.0

    # Rebuild behavioral RDM for valid monkeys only
    valid_idx = [monkeys.index(m) for m in valid_monkeys]
    behav_sub = behavioral_rdm[np.ix_(valid_idx, valid_idx)]

    # Build neural RDM across draws
    rdm_draws = []
    for _ in range(n_draws):
        pop_vectors = np.zeros((len(valid_monkeys), len(neuron_ids)))
        for i, mid in enumerate(valid_monkeys):
            nt = int(min_trials[mid])
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=nt, replace=False)
                pop_vectors[i, j] = np.mean(rates[chosen])

        corr_mat = np.corrcoef(pop_vectors)
        # Handle NaN from zero-variance neurons
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        rdm = 1 - corr_mat
        rdm_draws.append(rdm)

    mean_rdm = np.mean(rdm_draws, axis=0)

    # RSA correlation
    n_v = len(valid_monkeys)
    neural_vec = mean_rdm[np.triu_indices(n_v, k=1)]
    behav_vec = behav_sub[np.triu_indices(n_v, k=1)]

    if np.std(neural_vec) == 0 or np.std(behav_vec) == 0:
        return 0.0

    rho, _ = spearmanr(neural_vec, behav_vec)
    return rho if not np.isnan(rho) else 0.0


# ─── MAIN CURVE BUILDER ──────────────────────────────────────────────────────

def run_selection_curve(variable_name, rank_fn, eval_fn, df, fractions, n_reps, seed):
    """
    Run the full_dominance_first neuron selection curve for one variable.

    1. Split trials in half (repeated n_reps times)
    2. Rank neurons on ranking half
    3. Evaluate at each fraction on evaluation half
    4. Average across repetitions

    Returns: dict with fraction -> (mean_score, std_score)
    """
    print(f"\n  {variable_name}: running {n_reps} split-half repetitions...")

    all_curves = []  # list of dicts: {fraction: score}

    for rep in range(n_reps):
        if (rep + 1) % 5 == 0:
            print(f"    Rep {rep+1}/{n_reps}")

        rep_seed = seed + rep * 1000

        # Split trials
        df_rank, df_eval = split_trials_half(df, seed=rep_seed)

        # Rank neurons (on ranking half)
        ranked_neurons, neuron_scores = rank_fn(df_rank)
        total_neurons = len(ranked_neurons)

        # Evaluate at each fraction (on evaluation half)
        curve = {}
        for frac in fractions:
            n_use = max(1, int(frac * total_neurons))
            subset = ranked_neurons[:n_use]
            score = eval_fn(df_eval, subset, rep_seed + 500)
            curve[frac] = score

        all_curves.append(curve)

    # Aggregate across reps
    results = {}
    for frac in fractions:
        scores = [c[frac] for c in all_curves]
        results[frac] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'sem': np.std(scores) / np.sqrt(len(scores)),
            'all_scores': scores,
        }

    return results


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_selection_curves(all_curves, fractions, output_dir, n_neurons_total):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'Identity (37-class)': '#4C72B0',
        'Age (F only, Adult vs Juv)': '#DD8452',
        'Submission profile (RSA)': '#C44E52',
        'Affiliation strength (RSA)': '#55A868',
    }
    linestyles = {
        'Identity (37-class)': '-',
        'Age (F only, Adult vs Juv)': '-',
        'Submission profile (RSA)': '--',
        'Affiliation strength (RSA)': '--',
    }

    x_neurons = [int(f * n_neurons_total) for f in fractions]
    x_pct = [f * 100 for f in fractions]

    for var_name, curve_data in all_curves.items():
        means = [curve_data[f]['mean'] for f in fractions]
        sems = [curve_data[f]['sem'] for f in fractions]

        color = colors.get(var_name, '#8C8C8C')
        ls = linestyles.get(var_name, '-')

        ax.plot(x_pct, means, marker='o', markersize=5, color=color,
                linestyle=ls, linewidth=2, label=var_name)
        ax.fill_between(x_pct,
                         [m - s for m, s in zip(means, sems)],
                         [m + s for m, s in zip(means, sems)],
                         alpha=0.15, color=color)

    ax.set_xlabel('Fraction of neurons used (%)')
    ax.set_ylabel('Performance (accuracy or Spearman rho)')
    ax.set_title(f'Neuron Selection Curves (n={n_neurons_total} total neurons)')
    ax.legend(loc='best', fontsize=9)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

    # Add secondary x-axis with neuron counts
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pcts = [5, 20, 50, 100]
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([f'{int(p/100 * n_neurons_total)}' for p in tick_pcts])
    ax2.set_xlabel('Number of neurons')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'neuron_selection_curves.png'), dpi=200)
    plt.close()
    print(f"  Saved: neuron_selection_curves.png")

    # ── Individual variable plots with more detail ──
    for var_name, curve_data in all_curves.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        means = [curve_data[f]['mean'] for f in fractions]
        sems = [curve_data[f]['sem'] for f in fractions]
        stds = [curve_data[f]['std'] for f in fractions]

        color = colors.get(var_name, '#8C8C8C')

        ax.plot(x_pct, means, marker='o', markersize=6, color=color, linewidth=2)
        ax.fill_between(x_pct,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=color, label='+/- 1 SD')
        ax.fill_between(x_pct,
                         [m - s for m, s in zip(means, sems)],
                         [m + s for m, s in zip(means, sems)],
                         alpha=0.3, color=color, label='+/- 1 SEM')

        # Mark peak
        peak_idx = np.argmax(means)
        ax.scatter([x_pct[peak_idx]], [means[peak_idx]],
                   s=150, color='red', zorder=5, marker='*',
                   label=f'Peak at {x_pct[peak_idx]:.0f}% ({int(fractions[peak_idx]*n_neurons_total)} neurons)')

        ax.set_xlabel('Fraction of neurons used (%)')
        ax.set_ylabel('Performance')
        ax.set_title(f'Neuron Selection Curve: {var_name}')
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        plt.tight_layout()
        fname = f"selection_curve_{var_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()

    print(f"\nAll figures saved to {output_dir}")


def plot_normalized_curves(all_curves, fractions, output_dir, n_neurons_total):
    """
    Plot all curves normalized to proportion of each variable's own peak.
    All curves range from 0 to 1, making shape comparison clean.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'Identity (37-class)': '#4C72B0',
        'Age (F only, Adult vs Juv)': '#DD8452',
        'Submission profile (RSA)': '#C44E52',
        'Affiliation strength (RSA)': '#55A868',
    }
    linestyles = {
        'Identity (37-class)': '-',
        'Age (F only, Adult vs Juv)': '-',
        'Submission profile (RSA)': '--',
        'Affiliation strength (RSA)': '--',
    }

    x_pct = [f * 100 for f in fractions]

    for var_name, curve_data in all_curves.items():
        means = np.array([curve_data[f]['mean'] for f in fractions])
        sems = np.array([curve_data[f]['sem'] for f in fractions])

        # Normalize by absolute peak value
        # For negative curves (e.g. affiliation), use the most extreme value
        peak_val = np.max(np.abs(means))
        if peak_val == 0:
            continue

        # Normalize so peak absolute performance = 1.0
        norm_means = np.abs(means) / peak_val
        norm_sems = sems / peak_val

        color = colors.get(var_name, '#8C8C8C')
        ls = linestyles.get(var_name, '-')

        # Mark peak
        peak_idx = np.argmax(norm_means)
        peak_label = f"{var_name} (peak: {x_pct[peak_idx]:.0f}%)"

        ax.plot(x_pct, norm_means, marker='o', markersize=5, color=color,
                linestyle=ls, linewidth=2, label=peak_label)
        ax.fill_between(x_pct,
                         norm_means - norm_sems,
                         norm_means + norm_sems,
                         alpha=0.15, color=color)

        # Star at peak
        ax.scatter([x_pct[peak_idx]], [norm_means[peak_idx]],
                   s=120, color=color, zorder=5, marker='*', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Fraction of neurons used (%)', fontsize=12)
    ax.set_ylabel('Normalized performance\n(proportion of peak)', fontsize=12)
    ax.set_title(f'Neuron Selection Curves — Normalized (n={n_neurons_total} neurons)', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # Annotate sparse vs distributed regions
    ax.annotate('Sparse coding:\npeak at small fraction,\nthen declines',
                xy=(10, 0.95), fontsize=8, fontstyle='italic', color='gray',
                ha='left', va='top')
    ax.annotate('Distributed coding:\nrises steadily,\npeaks at large fraction',
                xy=(70, 0.3), fontsize=8, fontstyle='italic', color='gray',
                ha='center', va='top')

    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pcts = [5, 20, 50, 100]
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([f'{int(p/100 * n_neurons_total)}' for p in tick_pcts])
    ax2.set_xlabel('Number of neurons')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'neuron_selection_curves_normalized.png'), dpi=200)
    plt.close()
    print(f"  Saved: neuron_selection_curves_normalized.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ─── HARDCODED CONFIG (for PyCharm) ──────────────────────────────────────
    pkl_dir = r"/home/connorlab/Documents/JulieData/Cortana/sorted_spike_cache_filtered"                    # <-- EDIT
    metadata_csv = r"/home/connorlab/Downloads/monkeyinfo.csv"        # <-- EDIT
    output_dir = r"/home/connorlab/Documents/JulieData/Cortana/neuron_selection_curves_output"       # <-- EDIT
    min_trials = 10

    # Behavioral matrices for RSA-based curves
    root_dir = "/home/connorlab/Documents/GitHub/Julie/social_data/"
    submission_xlsx_zombies = root_dir + "zombies_social_data/zombies_feature_df_affiliation.xlsx"   # <-- EDIT
    affiliation_xlsx_zombies =root_dir + "zombies_social_data/zombies_feature_df_submission.xlsx"       # <-- EDIT
    # ─────────────────────────────────────────────────────────────────────────

    n_pca = N_PCA_COMPONENTS

    # ── Load data ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)
    df = df[df['MonkeyName'] != 'NewMonkey']
    print(f"After excluding NewMonkey: {len(df)} trials")

    print("\nComputing firing rates...")
    df = compute_firing_rates(df)

    meta, meta_lookup = load_metadata(metadata_csv)

    # ── Load behavioral matrices ──
    print(f"\nLoading behavioral matrices...")
    sub_mat = load_behavior_matrix(submission_xlsx_zombies)
    aff_mat = load_behavior_matrix(affiliation_xlsx_zombies)
    print(f"  Submission: {sub_mat.shape}")
    print(f"  Affiliation: {aff_mat.shape}")

    # ── Identities ──
    all_identities = sorted(df['MonkeyName'].unique())
    zombies_stimulus = [m for m in sub_mat.index
                        if m != SUBJECT_MONKEY and m in set(df['MonkeyName'].unique())]
    zombies_stimulus = sorted(zombies_stimulus)

    print(f"\nAll identities: {len(all_identities)}")
    print(f"Zombies stimulus monkeys: {len(zombies_stimulus)}")

    # ── Filter neurons ──
    if min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, all_identities, min_trials)
    neuron_ids = sorted(df['NeuronID'].unique())
    n_total = len(neuron_ids)
    print(f"Total neurons: {n_total}")

    # ── Build behavioral RDMs for Zombies (for RSA ranking/evaluation) ──
    print(f"\nBuilding behavioral RDMs for Zombies...")

    # Submission profile RDM
    n_z = len(zombies_stimulus)
    sub_profiles = []
    all_in_mat = [m for m in sub_mat.index if m in sub_mat.columns]
    for mid in zombies_stimulus:
        row_vals = [float(sub_mat.loc[mid, o]) if mid in sub_mat.index and o in sub_mat.columns else 0.0
                    for o in all_in_mat]
        col_vals = [float(sub_mat.loc[o, mid]) if o in sub_mat.index and mid in sub_mat.columns else 0.0
                    for o in all_in_mat]
        sub_profiles.append(np.array(row_vals + col_vals))
    sub_profile_rdm = squareform(pdist(np.array(sub_profiles), metric='euclidean'))

    # Affiliation strength RDM (direct interaction, symmetric)
    aff_strength_rdm = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            if i == j:
                continue
            mi, mj = zombies_stimulus[i], zombies_stimulus[j]
            val_ij = float(aff_mat.loc[mi, mj]) if mi in aff_mat.index and mj in aff_mat.columns else 0.0
            val_ji = float(aff_mat.loc[mj, mi]) if mj in aff_mat.index and mi in aff_mat.columns else 0.0
            aff_strength_rdm[i, j] = -0.5 * (val_ij + val_ji)
    off_diag = aff_strength_rdm[np.triu_indices(n_z, k=1)]
    aff_strength_rdm = aff_strength_rdm - np.min(off_diag)
    np.fill_diagonal(aff_strength_rdm, 0)

    print(f"  Submission profile RDM: {sub_profile_rdm.shape}")
    print(f"  Affiliation strength RDM: {aff_strength_rdm.shape}")

    # ── Define variables and their ranking/evaluation functions ──
    all_curves = {}

    # 1. Identity (37-class decoding)
    print(f"\n{'=' * 60}")
    print("VARIABLE: Identity (37-class)")
    print(f"{'=' * 60}")
    all_curves['Identity (37-class)'] = run_selection_curve(
        'Identity (37-class)',
        rank_fn=lambda df_r: rank_neurons_identity(df_r, all_identities),
        eval_fn=lambda df_e, subset, seed: eval_identity_decoding(
            df_e, all_identities, subset, n_pca, seed),
        df=df,
        fractions=FRACTIONS,
        n_reps=N_SPLIT_REPS,
        seed=RANDOM_SEED
    )

    # 2. Age (females only, Adult vs Juvenile)
    print(f"\n{'=' * 60}")
    print("VARIABLE: Age (F only, Adult vs Juvenile)")
    print(f"{'=' * 60}")
    all_curves['Age (F only, Adult vs Juv)'] = run_selection_curve(
        'Age (F only, Adult vs Juv)',
        rank_fn=lambda df_r: rank_neurons_age(df_r, meta_lookup, all_identities),
        eval_fn=lambda df_e, subset, seed: eval_age_decoding(
            df_e, meta_lookup, all_identities, subset, n_pca, seed),
        df=df,
        fractions=FRACTIONS,
        n_reps=N_SPLIT_REPS,
        seed=RANDOM_SEED + 100000
    )

    # 3. Submission profile RSA (Zombies)
    print(f"\n{'=' * 60}")
    print("VARIABLE: Submission profile (RSA, Zombies)")
    print(f"{'=' * 60}")
    df_zombies = df[df['MonkeyName'].isin(zombies_stimulus)]
    all_curves['Submission profile (RSA)'] = run_selection_curve(
        'Submission profile (RSA)',
        rank_fn=lambda df_r: rank_neurons_social_rsa(df_r, zombies_stimulus, sub_profile_rdm),
        eval_fn=lambda df_e, subset, seed: eval_rsa(
            df_e, zombies_stimulus, subset, sub_profile_rdm, seed=seed),
        df=df_zombies,
        fractions=FRACTIONS,
        n_reps=N_SPLIT_REPS,
        seed=RANDOM_SEED + 200000
    )

    # 4. Affiliation strength RSA (Zombies)
    print(f"\n{'=' * 60}")
    print("VARIABLE: Affiliation strength (RSA, Zombies)")
    print(f"{'=' * 60}")
    all_curves['Affiliation strength (RSA)'] = run_selection_curve(
        'Affiliation strength (RSA)',
        rank_fn=lambda df_r: rank_neurons_social_rsa(df_r, zombies_stimulus, aff_strength_rdm),
        eval_fn=lambda df_e, subset, seed: eval_rsa(
            df_e, zombies_stimulus, subset, aff_strength_rdm, seed=seed),
        df=df_zombies,
        fractions=FRACTIONS,
        n_reps=N_SPLIT_REPS,
        seed=RANDOM_SEED + 300000
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total neurons: {n_total}")
    print(f"Split-half repetitions: {N_SPLIT_REPS}")
    print(f"Fractions tested: {FRACTIONS}")
    print()

    for var_name, curve_data in all_curves.items():
        means = [curve_data[f]['mean'] for f in FRACTIONS]
        peak_idx = np.argmax(means)
        peak_frac = FRACTIONS[peak_idx]
        peak_val = means[peak_idx]
        full_val = means[-1]  # 100%
        print(f"  {var_name}:")
        print(f"    Peak: {peak_val:.4f} at {peak_frac*100:.0f}% ({int(peak_frac*n_total)} neurons)")
        print(f"    Full population: {full_val:.4f}")
        print(f"    Peak vs Full: {'peak > full_dominance_first (SPARSE)' if peak_val > full_val else 'full_dominance_first >= peak (DISTRIBUTED)'}")

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    plot_selection_curves(all_curves, FRACTIONS, output_dir, n_total)
    plot_normalized_curves(all_curves, FRACTIONS, output_dir, n_total)

    save_path = os.path.join(output_dir, 'selection_curve_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_curves, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
