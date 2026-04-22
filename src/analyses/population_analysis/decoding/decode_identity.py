"""
Pseudo-population Logistic Regression decoding of stimulus identity.

Uses DecodeIdentityConfig for all tuneable parameters and
data_loading.load_and_filter for data ingestion / region filtering.
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from decode_config import DecodeIdentityConfig
from population_analysis.state_space import load_and_filter


# ═══════════════════════════════════════════════════════════════════════════
# FIRING RATES
# ═══════════════════════════════════════════════════════════════════════════

def compute_firing_rates(df):
    """Compute trial-level firing rate = spike count in epoch / epoch duration."""
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


# ═══════════════════════════════════════════════════════════════════════════
# NEURON FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def filter_neurons_by_min_trials(df, min_trials):
    """Keep only neurons that have >= min_trials for EVERY target identity."""
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(df['MonkeyName'].unique())

    kept = []
    dropped = 0
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        if all(len(neuron_df[neuron_df['MonkeyName'] == mid]) >= min_trials
               for mid in identities):
            kept.append(nid)
        else:
            dropped += 1

    print(f"\n  min_trials filter (>= {min_trials} per identity):")
    print(f"    Kept: {len(kept)} / {len(neuron_ids)} neurons")
    print(f"    Dropped: {dropped}")

    return df[df['NeuronID'].isin(kept)], kept


# ═══════════════════════════════════════════════════════════════════════════
# PSEUDO-POPULATION
# ═══════════════════════════════════════════════════════════════════════════

def build_pseudo_population(df, cfg):
    """
    Build pseudo-population trial matrices by randomly pairing trials
    across neurons for each stimulus identity.
    """
    rng = np.random.default_rng(cfg.random_seed)

    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(df['MonkeyName'].unique())

    print(f"\nNeurons: {len(neuron_ids)}")
    print(f"Stimulus identities: {len(identities)}")

    # Collect firing rates per (neuron, identity)
    neuron_identity_rates = {}
    min_trials_per_identity = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        for mid in identities:
            trials = neuron_df[neuron_df['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = trials
            if len(trials) < min_trials_per_identity[mid]:
                min_trials_per_identity[mid] = len(trials)

    # Exclude identities with 0 trials for any neuron
    valid_identities = []
    for mid in identities:
        if int(min_trials_per_identity[mid]) > 0:
            valid_identities.append(mid)
        else:
            print(f"  WARNING: '{mid}' has 0 trials for some neuron(s) — excluding")

    print(f"Valid identities: {len(valid_identities)}")
    print("\nTrials per identity (min across neurons):")
    for mid in valid_identities:
        print(f"  {mid}: {int(min_trials_per_identity[mid])}")

    total_min = min(int(min_trials_per_identity[mid]) for mid in valid_identities)
    print(f"\nGlobal minimum trials per identity: {total_min}")

    # Draw pseudo-populations
    X_draws, y_draws = [], []
    for _ in range(cfg.n_pseudo_draws):
        X_rows, y_rows = [], []
        for mid in valid_identities:
            n_trials = int(min_trials_per_identity[mid])
            row_block = np.zeros((n_trials, len(neuron_ids)))
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=n_trials, replace=False)
                row_block[:, j] = rates[chosen]
            X_rows.append(row_block)
            y_rows.append(np.full(n_trials, mid))
        X_draws.append(np.vstack(X_rows))
        y_draws.append(np.concatenate(y_rows))

    return X_draws, y_draws, neuron_ids, valid_identities


# ═══════════════════════════════════════════════════════════════════════════
# DECODING
# ═══════════════════════════════════════════════════════════════════════════

def run_decoding(X, y, cfg):
    """Run stratified k-fold cross-validated logistic regression decoding."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    min_class_count = min(np.bincount(y_enc))
    n_folds = min(cfg.n_cv_folds, min_class_count)
    if n_folds < 2:
        print(f"    WARNING: only {min_class_count} samples in smallest class, "
              f"cannot cross-validate")
        return np.array([0.0]), np.array([]), np.array([]), le

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=cfg.random_seed)

    fold_accs, all_y_true, all_y_pred = [], [], []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

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

        # Logistic Regression
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

    return np.array(fold_accs), np.array(all_y_true), np.array(all_y_pred), le


# ═══════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_permutation_test(X_draws, y_draws, cfg):
    """
    Permutation test averaged across pseudo-population draws.
    Returns array of null-distribution mean accuracies (length n_permutations).
    """
    rng = np.random.default_rng(cfg.random_seed)
    perm_mean_accs = []

    for i in range(cfg.n_permutations):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i + 1}/{cfg.n_permutations}")
        draw_accs = []
        for d in range(len(X_draws)):
            y_perm = rng.permutation(y_draws[d])
            fold_accs, _, _, _ = run_decoding(X_draws[d], y_perm, cfg)
            draw_accs.append(np.mean(fold_accs))
        perm_mean_accs.append(np.mean(draw_accs))

    return np.array(perm_mean_accs)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(mean_acc, std_acc, chance, cm, le, perm_accs, p_value,
                 n_neurons, n_classes, output_dir, min_trials_filter=0):
    os.makedirs(output_dir, exist_ok=True)
    filter_str = f", min_trials>={min_trials_filter}" if min_trials_filter > 0 else ""

    # ── 1. Accuracy / permutation histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    if perm_accs is not None:
        ax.hist(perm_accs, bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(mean_acc, color='red', linewidth=2,
                   label=f'Observed = {mean_acc:.3f}')
        ax.axvline(chance, color='blue', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        ax.set_title(f'Identity Decoding — LogReg ({n_neurons} neurons, '
                     f'{n_classes} classes{filter_str}, {p_str})')
    else:
        ax.bar(['LogReg'], [mean_acc], yerr=[std_acc], capsize=5,
               color='#4C72B0', edgecolor='black', linewidth=0.8)
        ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Identity Decoding — LogReg ({n_neurons} neurons, '
                     f'{n_classes} classes{filter_str})')
        ax.set_ylim(0, max(mean_acc + std_acc, chance) * 1.5)

    ax.set_xlabel('Accuracy')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_identity.png'), dpi=200)
    plt.close()
    print(f"  Saved: accuracy_identity.png")

    # ── 2. Confusion matrix ──
    labels = le.classes_
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Confusion Matrix — LogReg (normalized, '
                 f'{n_neurons} neurons{filter_str})')
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

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_identity_logreg.png'), dpi=200)
    plt.close()
    print(f"  Saved: confusion_identity_logreg.png")

    # ── 3. Permutation histogram (separate file) ──
    if perm_accs is not None:
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(perm_accs, bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(mean_acc, color='red', linewidth=2,
                   label=f'Observed = {mean_acc:.3f}')
        ax.axvline(chance, color='blue', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'Permutation Test — LogReg ({n_neurons} neurons, '
                     f'{n_classes} classes{filter_str}, {p_str})')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'permutation_identity_logreg.png'), dpi=200)
        plt.close()
        print(f"  Saved: permutation_identity_logreg.png")

    print(f"\nFigures saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Instantiate & validate config ──
    cfg = DecodeIdentityConfig(
        region='ALL',           # 'AMG', 'ER', or 'ALL'
        min_trials=0,           # set > 0 to filter neurons
        n_pseudo_draws=10,
        n_pca=50,               # None or 0 to skip PCA
        skip_perm=True,
        n_permutations=1000,
        output_dir='./decode_identity_output',
    )
    cfg.validate()

    # ── Load & filter data via shared loader ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_and_filter(cfg)

    # ── Compute firing rates ──
    print("\nComputing trial-level firing rates (full_dominance_first epoch)...")
    df = compute_firing_rates(df)

    # ── Filter neurons ──
    if cfg.min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, cfg.min_trials)
    else:
        kept_neurons = sorted(df['NeuronID'].unique())

    # ── Build pseudo-population ──
    print("\n" + "=" * 60)
    print("BUILDING PSEUDO-POPULATION")
    print("=" * 60)
    X_draws, y_draws, neuron_ids, valid_identities = build_pseudo_population(df, cfg)

    n_classes = len(valid_identities)
    n_neurons = len(neuron_ids)
    chance = 1.0 / n_classes
    print(f"\nChance level: {chance:.4f} ({n_classes} classes)")
    print(f"Feature matrix shape: {X_draws[0].shape}")

    # ── Decode ──
    print(f"\n{'=' * 60}")
    print("DECODING: LOGREG")
    print(f"{'=' * 60}")

    draw_accs = []
    best_acc = -1
    best_results = None

    for d in range(len(X_draws)):
        print(f"\n  Draw {d + 1}/{len(X_draws)}")
        fold_accs, y_true, y_pred, le = run_decoding(X_draws[d], y_draws[d], cfg)
        mean_acc = np.mean(fold_accs)
        draw_accs.append(mean_acc)
        print(f"    Mean accuracy: {mean_acc:.4f} (±{np.std(fold_accs):.4f})")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_results = (fold_accs, y_true, y_pred, le)

    overall_mean = np.mean(draw_accs)
    overall_std = np.std(draw_accs)
    print(f"\n  Overall across draws: {overall_mean:.4f} (±{overall_std:.4f})")

    fold_accs, y_true, y_pred, le = best_results
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Classification report (best draw):")
    print(classification_report(y_true, y_pred, target_names=le.classes_,
                                zero_division=0))

    # ── Permutation test ──
    perm_accs = None
    p_value = None
    if not cfg.skip_perm:
        print(f"\n  Running permutation test ({cfg.n_permutations} permutations)...")
        perm_accs = run_permutation_test(X_draws, y_draws, cfg)
        p_value = (np.sum(perm_accs >= overall_mean) + 1) / (cfg.n_permutations + 1)
        print(f"  Permutation p-value: {p_value:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Region: {cfg.region}")
    print(f"Neurons: {n_neurons}")
    if cfg.min_trials > 0:
        print(f"  (filtered: min_trials >= {cfg.min_trials} per identity)")
    print(f"Identities: {n_classes} — {', '.join(valid_identities)}")
    print(f"Chance: {chance:.4f}")
    p_str = f", p = {p_value:.4f}" if p_value is not None else ""
    print(f"  logreg: {overall_mean:.4f} ± {overall_std:.4f}{p_str}")

    # ── Save plots & results ──
    plot_results(overall_mean, overall_std, chance, cm, le,
                 perm_accs, p_value, n_neurons, n_classes,
                 cfg.output_dir, min_trials_filter=cfg.min_trials)

    save_dict = {
        'config': cfg,
        'mean_acc': overall_mean,
        'std_acc': overall_std,
        'draw_accs': draw_accs,
        'confusion_matrix': cm,
        'label_encoder': le,
        'chance_level': chance,
        'perm_accs': perm_accs,
        'p_value': p_value,
        'n_neurons': n_neurons,
        'n_classes': n_classes,
        'valid_identities': valid_identities,
    }
    save_path = os.path.join(cfg.output_dir, 'decoding_identity.pkl')
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()