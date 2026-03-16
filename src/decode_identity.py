"""
Pseudo-population MLP/Logistic Regression decoding of stimulus identity

Each pkl file should be a DataFrame with columns:
    SpikeTimes, NeuronID, EpochStartStop, MonkeyName, MonkeyGroup
"""

import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
N_PCA_COMPONENTS = 50       # None will skip PCA
MLP_HIDDEN = (128, 64)
MLP_MAX_ITER = 500
RANDOM_SEED = 42
N_PSEUDO_DRAWS = 10         # number of pseudo-population resamples to average over


def load_all_trials(pkl_dir):
    """Load all pkl files and concatenate into a single DataFrame."""
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")
    
    dfs = []
    for f in pkl_files:
        with open(f, 'rb') as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
        else:
            print(f"  Skipping {os.path.basename(f)} — not a DataFrame")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(pkl_files)} files → {len(combined)} total trials")
    return combined


def compute_firing_rates(df):
    """Compute trial-level firing rate = spike count in epoch / epoch duration."""
    rates = []
    for _, row in df.iterrows():
        spikes = np.array(row['SpikeTimes'])
        epoch = row['EpochStartStop']
        
        # Handle tuple or list for epoch boundaries
        if isinstance(epoch, str):
            epoch = eval(epoch)
        t_start, t_stop = epoch[0], epoch[1]
        duration = t_stop - t_start
        
        # Count spikes within epoch
        n_spikes = np.sum((spikes >= t_start) & (spikes <= t_stop))
        rate = n_spikes / duration if duration > 0 else 0.0
        rates.append(rate)
    
    df = df.copy()
    df['firing_rate'] = rates
    return df


def build_pseudo_population(df, n_draws=N_PSEUDO_DRAWS, seed=RANDOM_SEED):
    """
    Build pseudo-population trial matrices by randomly pairing trials across neurons for each stimulus identity.
    
    Returns:
        X_draws: list of (n_trials_per_draw, n_neurons) arrays
        y_draws: list of (n_trials_per_draw,) label arrays
        neuron_ids: list of neuron ID strings (column order)
    """
    rng = np.random.default_rng(seed)
    
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(df['MonkeyName'].unique())
    
    print(f"\nNeurons: {len(neuron_ids)}")
    print(f"Stimulus identities: {len(identities)}")
    
    # For each neuron × identity, collect firing rates
    neuron_identity_rates = {}
    min_trials_per_identity = defaultdict(lambda: np.inf)
    
    for nid in neuron_ids: # nid is neuron id
        neuron_df = df[df['NeuronID'] == nid]
        for mid in identities: # mid is monkey id
            trials = neuron_df[neuron_df['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = trials
            if len(trials) < min_trials_per_identity[mid]:
                min_trials_per_identity[mid] = len(trials)
    
    # Check for identities with zero trials for any neuron
    valid_identities = []
    for mid in identities:
        min_t = int(min_trials_per_identity[mid])
        if min_t > 0:
            valid_identities.append(mid)
        else:
            print(f"  WARNING: '{mid}' has 0 trials for some neuron(s) — excluding")
    
    print(f"Valid identities (with trials for all neurons): {len(valid_identities)}")
    
    # Report trial counts
    print("\nTrials per identity (min across neurons):")
    for mid in valid_identities:
        print(f"  {mid}: {int(min_trials_per_identity[mid])}")
    
    total_min = min(int(min_trials_per_identity[mid]) for mid in valid_identities)
    print(f"\nGlobal minimum trials per identity: {total_min}")
    
    # Build pseudo-population draws
    X_draws = []
    y_draws = []
    
    for draw_i in range(n_draws):
        X_rows = [] # firing rate
        y_rows = [] # monkey identity
        
        for mid in valid_identities:
            n_trials = int(min_trials_per_identity[mid]) # due to limitations on the number of trials across neurons, every neuron will contribute overall min trials for particular monkey
            
            row_block = np.zeros((n_trials, len(neuron_ids)))
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)] # neuron_identity_rates has f.r. in such a way that (Neuron1, monkeyA) → f.r. [5,7,6,4]
                # Randomly sample (without replacement) to match min trial count
                chosen = rng.choice(len(rates), size=n_trials, replace=False)
                row_block[:, j] = rates[chosen]
            
            X_rows.append(row_block)
            y_rows.append(np.full(n_trials, mid))
        
        X_draws.append(np.vstack(X_rows))
        y_draws.append(np.concatenate(y_rows))
    
    return X_draws, y_draws, neuron_ids, valid_identities


def run_decoding(X, y, model_name='logreg', n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    """
    Run stratified k-fold cross-validated decoding.
    
    Returns: fold accuracies, predicted labels, true labels (for confusion matrix)
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)


    # Adjust folds if too few samples per class
    min_class_count = min(np.bincount(y_enc))
    n_folds = min(N_CV_FOLDS, min_class_count)
    # print(f"\nNumber of folds: {n_folds}")
    if n_folds < 2:
        print(f"    WARNING: only {min_class_count} samples in smallest class, cannot cross-validate")
        return np.array([0.0]), np.array([]), np.array([]), le

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_accs = []
    all_y_true = []
    all_y_pred = []
    
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # PCA (safe component count)
        if n_pca is not None and n_pca < X_train.shape[1]:
            n_comp = min(n_pca, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
        
        # Model
        if model_name == 'logreg':
            clf = LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs',
                max_iter=2000, multi_class='multinomial',
                random_state=seed
            )
        elif model_name == 'mlp':
            clf = MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN,
                activation='relu',
                max_iter=MLP_MAX_ITER,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    return np.array(fold_accs), np.array(all_y_true), np.array(all_y_pred), le


def run_permutation_test(X, y, model_name='logreg', n_perms=N_PERMUTATIONS, 
                         n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    """Permutation test: shuffle labels and re-run decoding."""
    rng = np.random.default_rng(seed)
    
    perm_accs = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")
        y_shuffled = rng.permutation(y)
        fold_accs, _, _, _ = run_decoding(X, y_shuffled, model_name=model_name, 
                                           n_pca=n_pca, seed=seed + i)
        perm_accs.append(np.mean(fold_accs))
    
    return np.array(perm_accs)


def plot_results(results, n_neurons, n_classes, output_dir, min_trials_filter=0):
    """Generate summary figures (decode_group-style naming and formatting)."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(results.keys())
    chance = results[models[0]]['chance_level']
    filter_str = f", min_trials>={min_trials_filter}" if min_trials_filter > 0 else ""

    # ── 1. Accuracy comparison bar plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [results[m]['mean_acc'] for m in models]
    sds = [results[m]['std_acc'] for m in models]

    bars = ax.bar(models, means, yerr=sds, capsize=5, color=['#4C72B0', '#DD8452'],
                  edgecolor='black', linewidth=0.8)
    ax.axhline(chance, color='gray', linestyle='--', linewidth=1, label=f'Chance = {chance:.3f}')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Identity Decoding ({n_neurons} neurons, {n_classes} classes{filter_str})')
    ax.legend()
    ax.set_ylim(0, max(means) * 1.3)

    for bar, m in zip(bars, models):
        if results[m].get('p_value') is not None:
            p = results[m]['p_value']
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sds[models.index(m)] + 0.005,
                    p_str, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_identity.png'), dpi=200)
    plt.close()
    print(f"  Saved: accuracy_identity.png")

    # ── 2. Confusion matrix per model (normalized, with cell annotations) ──
    for model_name in models:
        cm = results[model_name]['confusion_matrix']
        le = results[model_name]['label_encoder']
        labels = le.classes_

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Confusion Matrix -- {model_name} (normalized, '
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
        fig.savefig(os.path.join(output_dir, f'confusion_identity_{model_name}.png'), dpi=200)
        plt.close()
        print(f"  Saved: confusion_identity_{model_name}.png")

    # ── 3. Permutation distribution per model ──
    for model_name in models:
        if results[model_name].get('perm_accs') is not None:
            perm_accs = results[model_name]['perm_accs']
            mean_acc = results[model_name]['mean_acc']
            p_value = results[model_name]['p_value']
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
            ax.set_title(f'Permutation Test -- {model_name} ({n_neurons} neurons, '
                         f'{n_classes} classes{filter_str}, {p_str})')
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'permutation_identity_{model_name}.png'), dpi=200)
            plt.close()
            print(f"  Saved: permutation_identity_{model_name}.png")

    print(f"\nFigures saved to {output_dir}")


def filter_neurons_by_min_trials(df, min_trials):
    """
    Keep only neurons that have >= min_trials for EVERY target identity.
    Returns filtered df and list of kept neuron IDs.
    """
    neuron_ids = sorted(df['NeuronID'].unique())
    identities = sorted(df['MonkeyName'].unique())

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

def main():

    # ───  CONFIG  ──────────────────────────────────────
    pkl_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered"
    output_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/decode_identity_output"
    skip_perm = False       # set False to run permutation tests
    n_perm = 1000
    n_draws = 10
    n_pca = 50             # set to 0 to skip PCA
    min_trials = 0         # set > 0 to filter neurons
    
    n_pca = n_pca if n_pca > 0 else None
    
    # ── Load data ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)

    # Exclude NewMonkey
    df = df[df['MonkeyName'] != 'NewMonkey']
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Compute firing rates ──
    print("\nComputing trial-level firing rates (full epoch)...")
    df = compute_firing_rates(df)

    # ── Filter neurons by min_trials ──
    if min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, min_trials)
    else:
        kept_neurons = sorted(df['NeuronID'].unique())

    # ── Build pseudo-population ──
    print("\n" + "=" * 60)
    print("BUILDING PSEUDO-POPULATION")
    print("=" * 60)
    X_draws, y_draws, neuron_ids, valid_identities = build_pseudo_population(
        df, n_draws=n_draws
    )
    
    chance_level = 1.0 / len(valid_identities)
    print(f"\nChance level: {chance_level:.4f} ({len(valid_identities)} classes)")
    print(f"Feature matrix shape: {X_draws[0].shape}")
    
    # ── Decoding (averaged over pseudo-population draws) ──
    results = {}
    
    for model_name in ['logreg', 'mlp']:
        print(f"\n{'=' * 60}")
        print(f"DECODING: {model_name.upper()}")
        print(f"{'=' * 60}")
        
        draw_accs = []
        best_acc = -1
        best_results = None
        
        for d in range(len(X_draws)):
            print(f"\n  Draw {d+1}/{len(X_draws)}")
            fold_accs, y_true, y_pred, le = run_decoding(
                X_draws[d], y_draws[d], model_name=model_name, n_pca=n_pca
            )
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

        rng = np.random.default_rng(RANDOM_SEED)
        p_value = None
        # Permutation test: average null accuracy across pseudo-population draws
        perm_mean_accs = []
        if not skip_perm:
            print(f"\n  Running permutation test ({n_perm} permutations)...")
            for i in range(n_perm):
                perm_draw_accs = []
                for d in range(len(X_draws)):
                    y_perm = rng.permutation(y_draws[d])
                    fold_accs, _, _, _ = run_decoding(X_draws[d], y_perm, model_name=model_name,
                    n_pca=n_pca, seed=RANDOM_SEED)
                    perm_draw_accs.append(np.mean(fold_accs))
                perm_mean_accs.append(np.mean(perm_draw_accs))
            perm_mean_accs = np.array(perm_mean_accs)
            p_value = (np.sum(perm_mean_accs >= overall_mean) + 1) / (n_perm + 1)

        # perm_accs = None
        # p_value = None
        # if not skip_perm:
        #     print(f"\n  Running permutation test ({n_perm} permutations)...")
        #     perm_accs = run_permutation_test(
        #         X_draws[0], y_draws[0], model_name=model_name,
        #         n_perms=n_perm, n_pca=n_pca
        #     )
        #     p_value = (np.sum(perm_accs >= overall_mean) + 1) / (n_perm + 1)
        #     print(f"  Permutation p-value: {p_value:.4f}")
        
        results[model_name] = {
            'mean_acc': overall_mean,
            'std_acc': overall_std,
            'draw_accs': draw_accs,
            'fold_accs': fold_accs,
            'confusion_matrix': cm,
            'label_encoder': le,
            'chance_level': chance_level,
            'perm_mean_accs': perm_mean_accs,
            'p_value': p_value,
        }
        
        print(f"\n  Classification report (best draw):")
        print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Neurons: {len(neuron_ids)}")
    if min_trials > 0:
        print(f"  (filtered: min_trials >= {min_trials} per identity)")
    print(f"Identities: {len(valid_identities)} -- {', '.join(valid_identities)}")
    print(f"Chance: {chance_level:.4f}")
    for m in results:
        r = results[m]
        p_str = f", p = {r['p_value']:.4f}" if r['p_value'] is not None else ""
        print(f"  {m}: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}{p_str}")
    
    # ── Save results & plots ──
    plot_results(results, n_neurons=len(neuron_ids), n_classes=len(valid_identities),
                 output_dir=output_dir, min_trials_filter=min_trials)
    
    # Save raw results
    save_dict = {
        'results': results,
        'n_neurons': len(neuron_ids),
        'n_classes': len(valid_identities),
        'valid_identities': valid_identities,
        'min_trials_filter': min_trials,
    }
    save_path = os.path.join(output_dir, 'decoding_identity.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
