"""
Pseudo-population decoding of stimulus identity within a selected MonkeyGroup.
Includes min_trials filter to keep only neurons with sufficient trials per identity.

Usage: Hardcode paths below for PyCharm, or uncomment argparse block for CLI.
"""

import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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


def list_groups_and_select(df):
    """Print all MonkeyGroups with their members and let user select."""
    group_members = df.groupby('MonkeyGroup')['MonkeyName'].unique()

    print("\n" + "=" * 60)
    print("AVAILABLE MONKEY GROUPS")
    print("=" * 60)

    group_names = sorted(group_members.index)
    for i, g in enumerate(group_names):
        members = sorted(group_members[g])
        print(f"  [{i}] {g} ({len(members)} members): {', '.join(members)}")

    print()
    while True:
        try:
            choice = input("Select group number (or type group name): ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(group_names):
                    selected = group_names[idx]
                    break
                else:
                    print(f"  Invalid index. Choose 0-{len(group_names)-1}")
            else:
                if choice in group_names:
                    selected = choice
                    break
                else:
                    print(f"  '{choice}' not found. Try again.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            exit(0)

    members = sorted(group_members[selected])
    print(f"\nSelected: {selected} -> {len(members)} identities: {', '.join(members)}")
    return selected, members


def filter_neurons_by_min_trials(df, target_identities, min_trials):
    """
    Keep only neurons that have >= min_trials for EVERY target identity.
    Returns filtered df and list of kept neuron IDs.
    """
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(target_identities)

    kept = []
    dropped = 0
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        passes = True
        for mid in identities:
            n_trials = len(neuron_df[neuron_df['MonkeyName'] == mid])
            if n_trials < min_trials:
                passes = False
                break
        if passes:
            kept.append(nid)
        else:
            dropped += 1

    print(f"\n  min_trials filter (>= {min_trials} per identity):")
    print(f"    Kept: {len(kept)} / {len(neuron_ids)} neurons")
    print(f"    Dropped: {dropped}")

    df_filtered = df[df['NeuronID'].isin(kept)]
    return df_filtered, kept


def build_pseudo_population(df, target_identities, n_draws=N_PSEUDO_DRAWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(target_identities)

    print(f"\nNeurons: {len(neuron_ids)}")
    print(f"Target identities: {len(identities)}")

    neuron_identity_rates = {}
    min_trials_per_identity = defaultdict(lambda: np.inf)

    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        for mid in identities:
            trials = neuron_df[neuron_df['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = trials
            if len(trials) < min_trials_per_identity[mid]:
                min_trials_per_identity[mid] = len(trials)

    valid_identities = []
    for mid in identities:
        min_t = int(min_trials_per_identity[mid])
        if min_t > 0:
            valid_identities.append(mid)
        else:
            print(f"  WARNING: '{mid}' has 0 trials for some neuron(s) -- excluding")

    print(f"Valid identities: {len(valid_identities)}")
    print("\nTrials per identity (min across neurons):")
    for mid in valid_identities:
        print(f"  {mid}: {int(min_trials_per_identity[mid])}")

    total_min = min(int(min_trials_per_identity[mid]) for mid in valid_identities)
    print(f"\nGlobal minimum trials per identity: {total_min}")

    X_draws = []
    y_draws = []

    for draw_i in range(n_draws):
        X_rows = []
        y_rows = []
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


def run_decoding(X, y, n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Adjust folds if too few samples per class
    min_class_count = min(np.bincount(y_enc))
    n_folds = min(N_CV_FOLDS, min_class_count)
    if n_folds < 2:
        print(f"    WARNING: only {min_class_count} samples in smallest class, cannot cross-validate")
        return np.array([0.0]), np.array([]), np.array([]), le

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accs = []
    all_y_true = []
    all_y_pred = []

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

        clf = LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs',
            max_iter=2000, multi_class='multinomial',
            random_state=seed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(fold_accs), np.array(all_y_true), np.array(all_y_pred), le


def run_permutation_test(X, y, n_perms=N_PERMUTATIONS, n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    perm_accs = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")
        y_shuffled = rng.permutation(y)
        fold_accs, _, _, _ = run_decoding(X, y_shuffled, n_pca=n_pca, seed=seed + i)
        perm_accs.append(np.mean(fold_accs))
    return np.array(perm_accs)


def plot_results(group_name, n_neurons, mean_acc, std_acc, chance,
                 cm, le, output_dir, min_trials_filter=0,
                 perm_accs=None, p_value=None):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(le.classes_)

    filter_str = f", min_trials>={min_trials_filter}" if min_trials_filter > 0 else ""

    # ── 1. Accuracy plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    if perm_accs is not None:
        ax.hist(perm_accs, bins=50, color='gray', alpha=0.7, edgecolor='black',
                linewidth=0.5, label='Permuted')
        ax.axvline(mean_acc, color='red', linewidth=2,
                   label=f'Observed = {mean_acc:.3f}')
        ax.axvline(chance, color='blue', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        ax.set_title(f'{group_name} Identity Decoding ({n_neurons} neurons, '
                     f'{n_classes} classes{filter_str}, {p_str})')
    else:
        ax.bar(['LogReg'], [mean_acc], yerr=[std_acc], capsize=5,
               color='#4C72B0', edgecolor='black', linewidth=0.8)
        ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                   label=f'Chance = {chance:.3f}')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{group_name} Identity Decoding ({n_neurons} neurons, '
                     f'{n_classes} classes{filter_str})')
        ax.set_ylim(0, max(mean_acc + std_acc, chance) * 1.5)

    ax.set_xlabel('Accuracy')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'accuracy_{group_name}.png'), dpi=200)
    plt.close()
    print(f"  Saved: accuracy_{group_name}.png")

    # ── 2. Confusion matrix ──
    fig, ax = plt.subplots(figsize=(8, 7))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Confusion Matrix -- {group_name} (normalized, {n_neurons} neurons{filter_str})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    labels = le.classes_
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'confusion_{group_name}.png'), dpi=200)
    plt.close()
    print(f"  Saved: confusion_{group_name}.png")

    print(f"\nFigures saved to {output_dir}")


def main():
    # ─── HARDCODED CONFIG (for PyCharm) ──────────────────────────────────────
    # Comment out this block and uncomment argparse below for command line.
    pkl_dir = r"/sorted_spike_cache_filtered"
    output_dir = r"/group_decoding_output_with_min_trials"
    skip_perm = False       # set False to run permutation test
    n_perm = 1000
    n_draws = 10
    n_pca = 50             # set to 0 to skip PCA
    min_trials = 10         # set > 0 to filter neurons (e.g. 10, 15, 20)
    # ─────────────────────────────────────────────────────────────────────────

    # # ─── ARGPARSE (for command line) ─────────────────────────────────────────
    # parser = argparse.ArgumentParser(description='Group-level identity decoding')
    # parser.add_argument('--pkl_dir', required=True)
    # parser.add_argument('--output_dir', default='./decoding_group_output')
    # parser.add_argument('--skip_perm', action='store_true')
    # parser.add_argument('--n_perm', type=int, default=N_PERMUTATIONS)
    # parser.add_argument('--n_draws', type=int, default=N_PSEUDO_DRAWS)
    # parser.add_argument('--n_pca', type=int, default=N_PCA_COMPONENTS)
    # parser.add_argument('--min_trials', type=int, default=0,
    #                     help='Min trials per identity per neuron (0 = no filter)')
    # args = parser.parse_args()
    # pkl_dir = args.pkl_dir
    # output_dir = args.output_dir
    # skip_perm = args.skip_perm
    # n_perm = args.n_perm
    # n_draws = args.n_draws
    # n_pca = args.n_pca
    # min_trials = args.min_trials
    # ─────────────────────────────────────────────────────────────────────────

    n_pca = n_pca if n_pca > 0 else None

    # ── Load ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)

    # Exclude NewMonkey
    df = df[df['MonkeyName'] != 'NewMonkey']
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Select group ──
    group_name, members = list_groups_and_select(df)

    # ── Compute firing rates ──
    print("\nComputing trial-level firing rates (full_dominance_first epoch)...")
    df = compute_firing_rates(df)

    # ── Filter neurons by min_trials ──
    if min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, members, min_trials)
    else:
        kept_neurons = sorted(df['NeuronID'].unique())

    # ── Build pseudo-population for selected group ──
    print("\n" + "=" * 60)
    print(f"BUILDING PSEUDO-POPULATION -- {group_name}")
    print("=" * 60)
    X_draws, y_draws, neuron_ids, valid_identities = build_pseudo_population(
        df, target_identities=members, n_draws=n_draws
    )

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
        print(f"\n  Draw {d+1}/{len(X_draws)}")
        fold_accs, y_true, y_pred, le = run_decoding(X_draws[d], y_draws[d], n_pca=n_pca)
        mean_acc = np.mean(fold_accs)
        draw_accs.append(mean_acc)
        print(f"    Mean accuracy: {mean_acc:.4f} (+/-{np.std(fold_accs):.4f})")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_results = (fold_accs, y_true, y_pred, le)

    overall_mean = np.mean(draw_accs)
    overall_std = np.std(draw_accs)
    print(f"\n  Overall across draws: {overall_mean:.4f} (+/-{overall_std:.4f})")

    fold_accs, y_true, y_pred, le = best_results
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Classification report (best draw):")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    # ── Permutation test ──
    perm_accs = None
    p_value = None
    if not skip_perm:
        print(f"\n  Running permutation test ({n_perm} permutations)...")
        perm_accs = run_permutation_test(
            X_draws[0], y_draws[0], n_perms=n_perm, n_pca=n_pca
        )
        p_value = (np.sum(perm_accs >= overall_mean) + 1) / (n_perm + 1)
        print(f"  Permutation p-value: {p_value:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Group: {group_name}")
    print(f"Neurons: {n_neurons}")
    if min_trials > 0:
        print(f"  (filtered: min_trials >= {min_trials} per identity)")
    print(f"Identities: {n_classes} -- {', '.join(valid_identities)}")
    print(f"Chance: {chance:.4f}")
    p_str = f", p = {p_value:.4f}" if p_value is not None else ""
    print(f"  logreg: {overall_mean:.4f} +/- {overall_std:.4f}{p_str}")

    # ── Save plots (always) & results ──
    os.makedirs(output_dir, exist_ok=True)

    plot_results(group_name, n_neurons, overall_mean, overall_std, chance,
                 cm, le, output_dir, min_trials_filter=min_trials,
                 perm_accs=perm_accs, p_value=p_value)

    results = {
        'group': group_name,
        'members': valid_identities,
        'n_neurons': n_neurons,
        'n_classes': n_classes,
        'chance': chance,
        'mean_acc': overall_mean,
        'std_acc': overall_std,
        'draw_accs': draw_accs,
        'confusion_matrix': cm,
        'label_encoder': le,
        'perm_accs': perm_accs,
        'p_value': p_value,
        'min_trials_filter': min_trials,
    }
    save_path = os.path.join(output_dir, f'decoding_{group_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
