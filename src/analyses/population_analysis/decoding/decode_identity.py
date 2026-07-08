# decode_identity.py
"""
Pseudo-population Logistic Regression decoding of stimulus identity.
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
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(mean_acc, std_acc, chance, cm, le, perm_accs, p_value,
                 n_neurons, n_classes, output_dir, min_trials_filter=0):
    os.makedirs(output_dir, exist_ok=True)
    filter_str = f", min_trials>={min_trials_filter}" if min_trials_filter > 0 else ""

    # 1. Accuracy / permutation histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    if perm_accs is not None:
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        plot_permutation_histogram(
            ax, mean_acc, perm_accs, chance=chance,
            title=f'Identity Decoding — LogReg ({n_neurons} neurons, '
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
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_identity.png'), dpi=200)
    plt.close()
    print(f"  Saved: accuracy_identity.png")

    # 2. Confusion matrix
    labels = le.classes_
    fig, ax = plt.subplots(figsize=(14, 12))
    im = plot_confusion_matrix(
        ax, cm, labels,
        title=f'Confusion Matrix — LogReg (normalized, {n_neurons} neurons{filter_str})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_identity_logreg.png'), dpi=200)
    plt.close()
    print(f"  Saved: confusion_identity_logreg.png")

    # 3. Separate permutation histogram
    if perm_accs is not None:
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_permutation_histogram(
            ax, mean_acc, perm_accs, chance=chance,
            title=f'Permutation Test — LogReg ({n_neurons} neurons, '
                  f'{n_classes} classes{filter_str}, {p_str})')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'permutation_identity_logreg.png'), dpi=200)
        plt.close()
        print(f"  Saved: permutation_identity_logreg.png")

    print(f"\nFigures saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = DecodeConfig(
        region='ER',
        min_trials=7,
        n_pseudo_draws=10,
        n_pca=50,
        skip_perm=True,
        n_permutations=1000,
        response_window=(0.200,0.500),       # e.g. (0.0, 0.5) for 0–500 ms
        output_dir='./decode_identity_output',
    )
    cfg.validate()

    # ── Load & prepare ──
    df = load_and_prepare(cfg)

    # ── Filter neurons ──
    all_identities = sorted(df['MonkeyName'].unique())
    if cfg.min_trials > 0:
        df, kept_neurons = filter_neurons_by_min_trials(df, all_identities, cfg.min_trials)
    else:
        kept_neurons = sorted(df['NeuronID'].unique())

    # ── Build pseudo-population ──
    print(f"\n{'=' * 60}")
    print("BUILDING PSEUDO-POPULATION")
    print(f"{'=' * 60}")
    X_draws, y_draws, neuron_ids, valid_identities = build_pseudo_population(
        df, all_identities, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed
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
        # Average permutation across draws
        rng = np.random.default_rng(cfg.random_seed)
        perm_mean_accs = []
        for i in range(cfg.n_permutations):
            if (i + 1) % 100 == 0:
                print(f"    Permutation {i + 1}/{cfg.n_permutations}")
            draw_perm = []
            for d in range(len(X_draws)):
                y_perm = rng.permutation(y_draws[d])
                fa = run_classification(X_draws[d], y_perm, cfg)
                draw_perm.append(np.mean(fa))
            perm_mean_accs.append(np.mean(draw_perm))
        perm_accs = np.array(perm_mean_accs)
        p_value = compute_p_value(overall_mean, perm_accs)
        print(f"  Permutation p-value: {p_value:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Region: {cfg.region}")
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
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, 'decoding_identity.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
