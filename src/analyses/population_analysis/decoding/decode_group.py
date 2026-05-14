# decode_group.py
"""
Pseudo-population decoding of stimulus identity within a selected MonkeyGroup.
Uses shared DecodeConfig and decode_utils.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from decode_config import DecodeConfig
from decode_utils import (
    load_and_prepare, filter_neurons_by_min_trials, build_pseudo_population,
    run_classification, run_classification_permutation, compute_p_value,
    plot_permutation_histogram, plot_confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def list_groups_and_select(df):
    """Print all MonkeyGroups with their members and let user select."""
    group_members = df.groupby('MonkeyGroup')['MonkeyName'].unique()
    group_names = sorted(group_members.index)

    print("\n" + "=" * 60)
    print("AVAILABLE MONKEY GROUPS")
    print("=" * 60)
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
                    print(f"  Invalid index. Choose 0-{len(group_names) - 1}")
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


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(group_name, n_neurons, mean_acc, std_acc, chance,
                 cm, le, output_dir, min_trials_filter=0,
                 perm_accs=None, p_value=None):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(le.classes_)
    filter_str = f", min_trials>={min_trials_filter}" if min_trials_filter > 0 else ""

    # 1. Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    if perm_accs is not None:
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        plot_permutation_histogram(
            ax, mean_acc, perm_accs, chance=chance,
            title=f'{group_name} Identity Decoding ({n_neurons} neurons, '
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
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'accuracy_{group_name}.png'), dpi=200)
    plt.close()
    print(f"  Saved: accuracy_{group_name}.png")

    # 2. Confusion matrix
    labels = le.classes_
    fig, ax = plt.subplots(figsize=(8, 7))
    im = plot_confusion_matrix(
        ax, cm, labels,
        title=f'Confusion Matrix — {group_name} (normalized, '
              f'{n_neurons} neurons{filter_str})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'confusion_{group_name}.png'), dpi=200)
    plt.close()
    print(f"  Saved: confusion_{group_name}.png")

    print(f"\nFigures saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = DecodeConfig(
        min_trials=10,
        n_pseudo_draws=10,
        n_pca=50,
        skip_perm=True,
        n_permutations=1000,
        response_window=(0.2, 0.5),       # e.g. (0.0, 0.5)
        output_dir='./group_decoding_output',
    )
    cfg.validate()

    # ── Load & prepare ──
    df = load_and_prepare(cfg)

    # ── Select group ──
    group_name, members = list_groups_and_select(df)

    # ── Filter neurons ──
    if cfg.min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, members, cfg.min_trials)
    else:
        kept_neurons = sorted(df['NeuronID'].unique())

    # ── Build pseudo-population ──
    print(f"\n{'=' * 60}")
    print(f"BUILDING PSEUDO-POPULATION — {group_name}")
    print(f"{'=' * 60}")
    X_draws, y_draws, neuron_ids, valid_identities = build_pseudo_population(
        df, members, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed
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
        print(f"\n  Draw {d + 1}/{len(X_draws)}")
        fold_accs, y_true, y_pred, le = run_classification(
            X_draws[d], y_draws[d], cfg, return_predictions=True)
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
        perm_accs = run_classification_permutation(
            X_draws[0], y_draws[0], cfg, n_perms=cfg.n_permutations)
        p_value = compute_p_value(overall_mean, perm_accs)
        print(f"  Permutation p-value: {p_value:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Group: {group_name}")
    print(f"Neurons: {n_neurons}")
    if cfg.min_trials > 0:
        print(f"  (filtered: min_trials >= {cfg.min_trials} per identity)")
    if cfg.response_window:
        print(f"Response window: {cfg.response_window[0]}–{cfg.response_window[1]} s")
    print(f"Identities: {n_classes} — {', '.join(valid_identities)}")
    print(f"Chance: {chance:.4f}")
    p_str = f", p = {p_value:.4f}" if p_value is not None else ""
    print(f"  logreg: {overall_mean:.4f} ± {overall_std:.4f}{p_str}")

    # ── Save ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    plot_results(group_name, n_neurons, overall_mean, overall_std, chance,
                 cm, le, cfg.output_dir, min_trials_filter=cfg.min_trials,
                 perm_accs=perm_accs, p_value=p_value)

    results = {
        'config': cfg,
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
    }
    save_path = os.path.join(cfg.output_dir, f'decoding_{group_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
