"""
Pairwise identity decoding across monkey groups.

For each group, decode every pair of identities (binary, 50% chance).
This makes decoding accuracy directly comparable across groups of different sizes.

Usage: Hardcode paths below for PyCharm, or uncomment argparse block for CLI.
"""

import glob
import os
import pickle
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib

from analyses.spike_rate import add_trial_spike_rate_columns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
N_CV_FOLDS = 5
N_PERMUTATIONS = 500
N_PCA_COMPONENTS = 50
N_PSUEDO_DRAWS = 5 # TODO: try increasing this?
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
    df_filtered = df[df['NeuronID'].isin(kept)]
    return df_filtered, kept


def build_pairwise_pseudo_population(df, id_a, id_b, neuron_ids, n_draws, seed, min_trials_per_id):
    """Build pseudo-population for a single pair of identities."""
    rng = np.random.default_rng(seed)

    # Collect firing rates per neuron per identity
    rates_a = {}
    rates_b = {}
    min_trials_a = np.inf
    min_trials_b = np.inf
    kept_neurons = []

    for nid in neuron_ids:
        ndf = df[df['NeuronID'] == nid]
        ra = ndf[ndf['MonkeyName'] == id_a]['SpikeRate'].values
        rb = ndf[ndf['MonkeyName'] == id_b]['SpikeRate'].values


        # Per-pair filtering: skip this neuron if it lacks trials for either identity
        if len(ra) < min_trials_per_id or len(rb) < min_trials_per_id:
            continue

        rates_a[nid] = ra
        rates_b[nid] = rb
        kept_neurons.append(nid)
        min_trials_a = min(min_trials_a, len(ra))
        min_trials_b = min(min_trials_b, len(rb))

    if not kept_neurons:
        return None, None, 0, 0

    min_trials_a = int(min_trials_a)
    min_trials_b = int(min_trials_b)

    if min_trials_a < 2 or min_trials_b < 2:
        return None, None, min_trials_a, min_trials_b

    X_draws = []
    y_draws = []

    for _ in range(n_draws):
        block_a = np.zeros((min_trials_a, len(kept_neurons)))
        block_b = np.zeros((min_trials_b, len(kept_neurons)))

        for j, nid in enumerate(kept_neurons):
            idx_a = rng.choice(len(rates_a[nid]), size=min_trials_a, replace=False)
            idx_b = rng.choice(len(rates_b[nid]), size=min_trials_b, replace=False)
            block_a[:, j] = rates_a[nid][idx_a]
            block_b[:, j] = rates_b[nid][idx_b]

        X = np.vstack([block_a, block_b])
        y = np.concatenate([np.zeros(min_trials_a), np.ones(min_trials_b)])
        X_draws.append(X)
        y_draws.append(y)

    return X_draws, y_draws, min_trials_a, min_trials_b


def run_binary_decoding(X, y, n_pca=N_PCA_COMPONENTS, seed=RANDOM_SEED):
    """Binary decoding with stratified CV."""
    y_int = y.astype(int)

    min_class_count = min(np.bincount(y_int))
    n_folds = min(N_CV_FOLDS, min_class_count)
    if n_folds < 2:
        return np.array([0.5])  # can't cross-validate

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y_int):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

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
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_accs.append(accuracy_score(y_test, y_pred))

    return np.array(fold_accs)

# running permutation only on the first draw
# def run_pairwise_permutation(X, y, observed_acc, n_perms, n_pca, seed):
#     """Permutation test for a single pair."""
#     rng = np.random.default_rng(seed)
#     perm_accs = []
#     for i in range(n_perms):
#         y_shuf = rng.permutation(y)
#         fold_accs = run_binary_decoding(X, y_shuf, n_pca=n_pca, seed=seed + i)
#         perm_accs.append(np.mean(fold_accs))
#     perm_accs = np.array(perm_accs)
#     p_value = (np.sum(perm_accs >= observed_acc) + 1) / (n_perms + 1)
#     return p_value, perm_accs

def run_pairwise_permutation(X_draws, y_draws, observed_acc, n_perms, n_pca, seed):
    """Permutation test across all draws."""
    rng = np.random.default_rng(seed)
    perm_accs = []
    for i in range(n_perms):
        draw_accs = []
        for d, (X, y) in enumerate(zip(X_draws, y_draws)):
            y_shuf = rng.permutation(y)
            fold_accs = run_binary_decoding(X, y_shuf, n_pca=n_pca, seed=seed + i * 100 + d)
            draw_accs.append(np.mean(fold_accs))
        perm_accs.append(np.mean(draw_accs))
    perm_accs = np.array(perm_accs)
    p_value = (np.sum(perm_accs >= observed_acc) + 1) / (n_perms + 1)
    return p_value, perm_accs

def decode_group_pairwise(df, group_name, members, neuron_ids, n_draws, n_pca,
                          run_perms=False, n_perms=N_PERMUTATIONS, min_trials_per_id=3):
    """Run pairwise decoding for all pairs within a group."""
    # Generate all pairs from each group
    pairs = list(itertools.combinations(sorted(members), 2))
    print(f"\n{'=' * 60}")
    print(f"PAIRWISE DECODING: {group_name} ({len(members)} members, {len(pairs)} pairs)")
    print(f"{'=' * 60}")

    pair_results = {}

    for pair_index, (id_a, id_b) in enumerate(pairs):
        X_draws, y_draws, n_a, n_b = build_pairwise_pseudo_population(df, id_a, id_b, neuron_ids, n_draws=n_draws,
                                                                      seed=RANDOM_SEED + pair_index * 1000,
                                                                      min_trials_per_id=min_trials_per_id)

        if X_draws is None:
            print(f"  [{pair_index+1}/{len(pairs)}] {id_a} vs {id_b}: SKIPPED (too few trials)")
            pair_results[(id_a, id_b)] = {
                'accuracy': np.nan, 'std': np.nan, 'n_trials': (n_a, n_b),
                'p_value': np.nan, 'significant': False
            }
            continue

        # Average accuracy across draws
        draw_accs = []
        for d in range(len(X_draws)):
            fold_accs = run_binary_decoding(X_draws[d], y_draws[d], n_pca=n_pca,
                                            seed=RANDOM_SEED + pair_index * 1000 + d)
            draw_accs.append(np.mean(fold_accs))

        mean_acc = np.mean(draw_accs)
        std_acc = np.std(draw_accs)

        # Permutation test (on first draw)
        p_value = np.nan
        if run_perms:
            # p_value, _ = run_pairwise_permutation(
            #     X_draws[0], y_draws[0], mean_acc, n_perms=n_perms,
            #     n_pca=n_pca, seed=RANDOM_SEED + pair_index * 2000
            # )
            p_value, _ = run_pairwise_permutation(
                X_draws, y_draws, mean_acc, n_perms=n_perms,
                n_pca=n_pca, seed=RANDOM_SEED + pair_index * 2000
            )

        pair_results[(id_a, id_b)] = {
            'accuracy': mean_acc,
            'std': std_acc,
            'n_trials': (n_a, n_b),
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }

        p_str = f", p={p_value:.4f}" if not np.isnan(p_value) else ""
        sig_str = " *" if pair_results[(id_a, id_b)]['significant'] else ""
        print(f"  [{pair_index+1}/{len(pairs)}] {id_a} vs {id_b}: "
              f"{mean_acc:.3f} +/-{std_acc:.3f} "
              f"(n={n_a}+{n_b}){p_str}{sig_str}")

    return pair_results


def plot_pairwise_comparison(all_group_results, output_dir):
    """
    Plot comparison of pairwise decoding across groups.
    1. Box/strip plot of pairwise accuracies per group
    2. Pairwise accuracy matrices per group
    """
    os.makedirs(output_dir, exist_ok=True)

    group_names = list(all_group_results.keys())
    n_groups = len(group_names)

    # ── 1. Box plot comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    group_accs = []
    group_labels = []
    for gi, gname in enumerate(group_names):
        accs = [r['accuracy'] for r in all_group_results[gname]['pair_results'].values()
                if not np.isnan(r['accuracy'])]
        group_accs.append(accs)

        n_sig = sum(1 for r in all_group_results[gname]['pair_results'].values()
                    if r['significant'])
        n_total = sum(1 for r in all_group_results[gname]['pair_results'].values()
                      if not np.isnan(r['accuracy']))
        group_labels.append(f"{gname}\n({n_sig}/{n_total} sig)")

    bp = ax.boxplot(group_accs, labels=group_labels, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], colors[:n_groups]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual points with jitter
    for gi, accs in enumerate(group_accs):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(accs))
        ax.scatter(np.full(len(accs), gi + 1) + jitter, accs,
                   color=colors[gi % len(colors)], alpha=0.6, s=20, zorder=3)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
    ax.set_ylabel('Pairwise Decoding Accuracy')
    ax.set_title('Pairwise Identity Decoding Across Groups')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pairwise_group_comparison.png'), dpi=200)
    plt.close()
    print(f"  Saved: pairwise_group_comparison.png")

    # ── 2. Summary bar plot (mean pairwise accuracy per group) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.nanmean(a) for a in group_accs]
    sems = [np.nanstd(a) / np.sqrt(len(a)) if len(a) > 0 else 0 for a in group_accs]

    bars = ax.bar(range(n_groups), means, yerr=sems, capsize=5,
                  color=colors[:n_groups], edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g.split('\n')[0] for g in group_labels], rotation=15, ha='right')
    ax.set_ylabel('Mean Pairwise Accuracy')
    ax.set_title('Mean Pairwise Decoding Accuracy by Group')

    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.set_ylim(0.3, max(means) + max(sems) + 0.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pairwise_mean_comparison.png'), dpi=200)
    plt.close()
    print(f"  Saved: pairwise_mean_comparison.png")

    # ── 3. Pairwise accuracy matrices per group ──
    for gname in group_names:
        pair_results = all_group_results[gname]['pair_results']
        members = all_group_results[gname]['members']
        members_sorted = sorted(members)
        n = len(members_sorted)

        mat = np.full((n, n), np.nan)
        for (id_a, id_b), res in pair_results.items():
            i = members_sorted.index(id_a)
            j = members_sorted.index(id_b)
            mat[i, j] = res['accuracy']
            mat[j, i] = res['accuracy']

        # Fill diagonal with NaN (self vs self)
        np.fill_diagonal(mat, np.nan)

        fig, ax = plt.subplots(figsize=(8, 7))
        # Mask NaN for display
        masked = np.ma.masked_where(np.isnan(mat), mat)
        im = ax.imshow(masked, cmap='RdYlGn', vmin=0.3, vmax=0.9,
                       interpolation='nearest')
        ax.set_title(f'Pairwise Decoding Accuracy — {gname}')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(members_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(members_sorted, fontsize=9)

        # Add text
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    val = mat[i, j]
                    color = 'white' if val < 0.45 or val > 0.8 else 'black'
                    # Mark significant pairs
                    pair_key = (members_sorted[min(i,j)], members_sorted[max(i,j)])
                    sig = pair_results.get(pair_key, {}).get('significant', False)
                    star = '*' if sig else ''
                    ax.text(j, i, f'{val:.2f}{star}', ha='center', va='center',
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label='Accuracy')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'pairwise_matrix_{gname}.png'), dpi=200)
        plt.close()
        print(f"  Saved: pairwise_matrix_{gname}.png")

    print(f"\nAll figures saved to {output_dir}")


def main():
    pkl_dir = r"/sorted_spike_cache_filtered"
    output_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/population_decoding_analysis/pairwise_decoding_output"
    skip_perm = False       # set False to run permutation tests per pair
    n_perm = 1000
    n_pca = 50             # set to 0 to skip PCA
    min_trials = 7
    n_pca = n_pca if n_pca > 0 else None

    # ── Load ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)
    df = df[df['MonkeyName'] != 'NewMonkey']
    print(f"After excluding NewMonkey: {len(df)} trials")

    # ── Compute firing rates ──
    print("\nComputing trial-level firing rates (full_dominance_first epoch)...")
    df = add_trial_spike_rate_columns(df)

    # ── Get groups ──
    group_members = df.groupby('MonkeyGroup')['MonkeyName'].unique()
    group_names = sorted(group_members.index)

    print(f"\nGroups found: {', '.join(group_names)}")

    # ── Get all unique identities across all groups for neuron filtering ──
    all_identities = sorted(df['MonkeyName'].unique())
    neuron_ids = sorted(df['NeuronID'].unique())

    # if min_trials > 0:
    #     # Filter neurons that have enough trials for ALL identities
    #     df, neuron_ids = filter_neurons_by_min_trials(df, all_identities, min_trials)
    #     print(f"  After min_trials filter: {len(neuron_ids)} neurons")
    # else:
    #     neuron_ids = sorted(df['NeuronID'].unique())

    print(f"\nUsing {len(neuron_ids)} neurons")

    # ── Run pairwise decoding per group ──
    all_group_results = {}

    for gname in group_names:
        members = sorted(group_members[gname])
        pair_results = decode_group_pairwise(
            df, gname, members, neuron_ids,
            n_draws=N_PSUEDO_DRAWS, n_pca=n_pca,
            run_perms=(not skip_perm), n_perms=n_perm, min_trials_per_id=min_trials,
        )

        # Group summary
        accs = [r['accuracy'] for r in pair_results.values() if not np.isnan(r['accuracy'])]
        n_sig = sum(1 for r in pair_results.values() if r['significant'])

        print(f"\n  {gname} summary:")
        print(f"    Mean pairwise accuracy: {np.nanmean(accs):.3f} +/-{np.nanstd(accs):.3f}")
        print(f"    Pairs above chance: {sum(1 for a in accs if a > 0.5)}/{len(accs)}")
        if not skip_perm:
            print(f"    Significantly above chance: {n_sig}/{len(accs)}")

        all_group_results[gname] = {
            'members': members,
            'pair_results': pair_results,
            'mean_acc': np.nanmean(accs),
            'std_acc': np.nanstd(accs),
            'n_pairs': len(accs),
            'n_above_chance': sum(1 for a in accs if a > 0.5),
            'n_significant': n_sig,
        }

    # ── Overall summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {len(neuron_ids)}")
    # if min_trials > 0:
    #     print(f"  (filtered: min_trials >= {min_trials} per identity)")
    print(f"Chance: 0.5000 (binary)")
    print()

    for gname in group_names:
        r = all_group_results[gname]
        sig_str = f", {r['n_significant']}/{r['n_pairs']} sig" if not skip_perm else ""
        print(f"  {gname:20s}: {r['mean_acc']:.3f} +/-{r['std_acc']:.3f} "
              f"({r['n_above_chance']}/{r['n_pairs']} above chance{sig_str})")

    # ── Save plots ──
    plot_pairwise_comparison(all_group_results, output_dir)

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pairwise_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_group_results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
