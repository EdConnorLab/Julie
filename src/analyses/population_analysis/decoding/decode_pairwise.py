# decode_pairwise.py
"""
Pairwise identity decoding across monkey groups.

For each group, decode every pair of identities (binary, 50% chance).
Uses shared DecodeConfig and decode_utils.
"""

import itertools
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decode_config import DecodeConfig
from decode_utils import (
    load_and_prepare, filter_neurons_by_min_trials, run_classification,
    compute_p_value,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRWISE PSEUDO-POPULATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_pairwise_pseudo_population(df, id_a, id_b, neuron_ids, cfg,
                                     seed_offset=0, min_trials_per_id=3):
    """Build pseudo-population for a single pair of identities."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)

    rates_a, rates_b = {}, {}
    min_trials_a, min_trials_b = np.inf, np.inf
    kept_neurons = []

    for nid in neuron_ids:
        ndf = df[df['NeuronID'] == nid]
        ra = ndf[ndf['MonkeyName'] == id_a]['firing_rate'].values
        rb = ndf[ndf['MonkeyName'] == id_b]['firing_rate'].values

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

    X_draws, y_draws = [], []
    for _ in range(cfg.n_pseudo_draws):
        block_a = np.zeros((min_trials_a, len(kept_neurons)))
        block_b = np.zeros((min_trials_b, len(kept_neurons)))
        for j, nid in enumerate(kept_neurons):
            block_a[:, j] = rates_a[nid][rng.choice(len(rates_a[nid]),
                                                      size=min_trials_a, replace=False)]
            block_b[:, j] = rates_b[nid][rng.choice(len(rates_b[nid]),
                                                      size=min_trials_b, replace=False)]
        X_draws.append(np.vstack([block_a, block_b]))
        y_draws.append(np.concatenate([np.zeros(min_trials_a),
                                        np.ones(min_trials_b)]))

    return X_draws, y_draws, min_trials_a, min_trials_b


def run_pairwise_permutation(X_draws, y_draws, observed_acc, cfg, n_perms,
                             seed_offset=0):
    """Permutation test across all draws for a pair."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    perm_accs = []
    for i in range(n_perms):
        draw_accs = []
        for d, (X, y) in enumerate(zip(X_draws, y_draws)):
            y_shuf = rng.permutation(y)
            fold_accs = run_classification(X, y_shuf, cfg)
            draw_accs.append(np.mean(fold_accs))
        perm_accs.append(np.mean(draw_accs))
    perm_accs = np.array(perm_accs)
    p_value = compute_p_value(observed_acc, perm_accs)
    return p_value, perm_accs


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP PAIRWISE DECODING
# ═══════════════════════════════════════════════════════════════════════════════

def decode_group_pairwise(df, group_name, members, neuron_ids, cfg,
                          run_perms=False, min_trials_per_id=3):
    """Run pairwise decoding for all pairs within a group."""
    pairs = list(itertools.combinations(sorted(members), 2))
    print(f"\n{'=' * 60}")
    print(f"PAIRWISE DECODING: {group_name} ({len(members)} members, {len(pairs)} pairs)")
    print(f"{'=' * 60}")

    pair_results = {}

    for pi, (id_a, id_b) in enumerate(pairs):
        X_draws, y_draws, n_a, n_b = build_pairwise_pseudo_population(
            df, id_a, id_b, neuron_ids, cfg,
            seed_offset=pi * 1000, min_trials_per_id=min_trials_per_id)

        if X_draws is None:
            print(f"  [{pi + 1}/{len(pairs)}] {id_a} vs {id_b}: SKIPPED (too few trials)")
            pair_results[(id_a, id_b)] = {
                'accuracy': np.nan, 'std': np.nan, 'n_trials': (n_a, n_b),
                'p_value': np.nan, 'significant': False
            }
            continue

        draw_accs = []
        for d in range(len(X_draws)):
            fold_accs = run_classification(
                X_draws[d], y_draws[d], cfg)
            draw_accs.append(np.mean(fold_accs))

        mean_acc = np.mean(draw_accs)
        std_acc = np.std(draw_accs)

        p_value = np.nan
        if run_perms:
            p_value, _ = run_pairwise_permutation(
                X_draws, y_draws, mean_acc, cfg,
                n_perms=cfg.n_permutations, seed_offset=pi * 2000)

        pair_results[(id_a, id_b)] = {
            'accuracy': mean_acc, 'std': std_acc,
            'n_trials': (n_a, n_b),
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }

        p_str = f", p={p_value:.4f}" if not np.isnan(p_value) else ""
        sig_str = " *" if pair_results[(id_a, id_b)]['significant'] else ""
        print(f"  [{pi + 1}/{len(pairs)}] {id_a} vs {id_b}: "
              f"{mean_acc:.3f} ±{std_acc:.3f} (n={n_a}+{n_b}){p_str}{sig_str}")

    return pair_results


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pairwise_comparison(all_group_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    group_names = list(all_group_results.keys())
    n_groups = len(group_names)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    # 1. Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    group_accs, group_labels = [], []
    for gname in group_names:
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

    # 2. Summary bar plot
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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend()
    ax.set_ylim(0.3, max(means) + max(sems) + 0.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pairwise_mean_comparison.png'), dpi=200)
    plt.close()
    print(f"  Saved: pairwise_mean_comparison.png")

    # 3. Pairwise accuracy matrices per group
    for gname in group_names:
        pair_results = all_group_results[gname]['pair_results']
        members_sorted = sorted(all_group_results[gname]['members'])
        n = len(members_sorted)
        mat = np.full((n, n), np.nan)
        for (id_a, id_b), res in pair_results.items():
            i, j = members_sorted.index(id_a), members_sorted.index(id_b)
            mat[i, j] = res['accuracy']
            mat[j, i] = res['accuracy']
        np.fill_diagonal(mat, np.nan)

        fig, ax = plt.subplots(figsize=(8, 7))
        masked = np.ma.masked_where(np.isnan(mat), mat)
        im = ax.imshow(masked, cmap='RdYlGn', vmin=0.3, vmax=0.9,
                       interpolation='nearest')
        ax.set_title(f'Pairwise Decoding Accuracy — {gname}')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(members_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(members_sorted, fontsize=9)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    val = mat[i, j]
                    color = 'white' if val < 0.45 or val > 0.8 else 'black'
                    pair_key = (members_sorted[min(i, j)], members_sorted[max(i, j)])
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = DecodeConfig(
        min_trials=7,
        n_pseudo_draws=5,
        n_pca=50,
        skip_perm=True,
        n_permutations=1000,
        response_window=(0.2, 0.5),       # e.g. (0.0, 0.5)
        output_dir='./pairwise_decoding_output',
    )
    cfg.validate()

    # ── Load & prepare ──
    df = load_and_prepare(cfg)

    # ── Get groups ──
    group_members = df.groupby('MonkeyGroup')['MonkeyName'].unique()
    group_names = sorted(group_members.index)
    print(f"\nGroups found: {', '.join(group_names)}")

    neuron_ids = sorted(df['NeuronID'].unique())
    print(f"Using {len(neuron_ids)} neurons")

    # ── Run pairwise decoding per group ──
    all_group_results = {}

    for gname in group_names:
        members = sorted(group_members[gname])
        pair_results = decode_group_pairwise(
            df, gname, members, neuron_ids, cfg,
            run_perms=(not cfg.skip_perm),
            min_trials_per_id=cfg.min_trials)

        accs = [r['accuracy'] for r in pair_results.values()
                if not np.isnan(r['accuracy'])]
        n_sig = sum(1 for r in pair_results.values() if r['significant'])

        print(f"\n  {gname} summary:")
        print(f"    Mean pairwise accuracy: {np.nanmean(accs):.3f} ±{np.nanstd(accs):.3f}")
        print(f"    Pairs above chance: {sum(1 for a in accs if a > 0.5)}/{len(accs)}")
        if not cfg.skip_perm:
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

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {len(neuron_ids)}")
    if cfg.response_window:
        print(f"Response window: {cfg.response_window[0]}–{cfg.response_window[1]} s")
    print(f"Chance: 0.5000 (binary)\n")

    for gname in group_names:
        r = all_group_results[gname]
        sig_str = f", {r['n_significant']}/{r['n_pairs']} sig" if not cfg.skip_perm else ""
        print(f"  {gname:20s}: {r['mean_acc']:.3f} ±{r['std_acc']:.3f} "
              f"({r['n_above_chance']}/{r['n_pairs']} above chance{sig_str})")

    # ── Save ──
    plot_pairwise_comparison(all_group_results, cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, 'pairwise_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'config': cfg, 'results': all_group_results}, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
