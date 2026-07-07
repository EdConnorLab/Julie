"""
Standalone script: Predicted monkey positions in social PC space.

Three decoding / geometry analyses per group:

  1. FULL-POP DECODING — LOO ridge regression from the full population
     vector (n_neurons features) to [PC1, PC2].

  2. NEURAL-PCA DECODING — PCA the population vectors down to a few
     components first, then LOO ridge to [PC1, PC2].

  3. PROCRUSTES / RSA — Compare pairwise distance structure between
     neural space and social PC space.  Reports Procrustes disparity,
     RSA (Spearman ρ between distance matrices), and permutation p-values.

  4. RATE SURFACE — LMM fixed-effect prediction surface as contour
     behind monkey scatter (unchanged from before).

Plots 1 & 2 are shown side-by-side in a single figure per group.

Requires: social_encoding_analysis.py and social_encoding_config.py
          on the Python path.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ── Import from existing pipeline ────────────────────────────────────
from social_encoding_analysis import (
    compute_trial_rates,
    load_social_features,
    reduce_features,
    build_trial_table,
    fit_mixed_model,
)
from social_encoding_config import SocialEncodingConfig
from analyses.population_analysis.state_space.data_loading import load_and_filter


# ═════════════════════════════════════════════════════════════════════
# 1. BUILD POPULATION VECTORS
# ═════════════════════════════════════════════════════════════════════

def build_population_vectors(trial_df):
    """Return (pop_matrix, monkey_names, neuron_ids).

    pop_matrix : (n_monkeys, n_neurons)  — mean firing rate of each
                 neuron to each monkey identity, averaged across trials.
    """
    avg = (trial_df
           .groupby(['MonkeyName', 'NeuronID'])['rate']
           .mean()
           .reset_index())

    pivot = avg.pivot(index='MonkeyName', columns='NeuronID', values='rate')
    pivot = pivot.dropna(axis=1)

    monkey_names = pivot.index.values
    neuron_ids = pivot.columns.values
    pop_matrix = pivot.values

    return pop_matrix, monkey_names, neuron_ids


# ═════════════════════════════════════════════════════════════════════
# 2. LOO DECODING METHODS
# ═════════════════════════════════════════════════════════════════════

def loo_decode(pop_matrix, targets):
    """LOO ridge regression: full pop_matrix → targets."""
    n = pop_matrix.shape[0]
    predicted = np.full_like(targets, np.nan)
    alphas = np.logspace(-3, 5, 50)

    for i in range(n):
        train_idx = np.arange(n) != i
        X_train, y_train = pop_matrix[train_idx], targets[train_idx]
        X_test = pop_matrix[i:i+1]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_train_sc, y_train)
        predicted[i] = ridge.predict(X_test_sc)[0]

    return predicted


def loo_decode_neural_pca(pop_matrix, targets, n_neural_pcs=None):
    """LOO ridge, but first reduce pop_matrix with PCA (fit on training fold).

    n_neural_pcs : int or None
        Number of neural PCs to keep.  None = min(n_monkeys-2, 5).
    """
    n = pop_matrix.shape[0]
    if n_neural_pcs is None:
        n_neural_pcs = min(n - 2, 5)

    predicted = np.full_like(targets, np.nan)
    alphas = np.logspace(-3, 5, 50)

    for i in range(n):
        train_idx = np.arange(n) != i
        X_train, y_train = pop_matrix[train_idx], targets[train_idx]
        X_test = pop_matrix[i:i+1]

        # Standardize
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # PCA on training set only
        n_comp = min(n_neural_pcs, X_train_sc.shape[0], X_train_sc.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train_sc)
        X_test_pca = pca.transform(X_test_sc)

        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_train_pca, y_train)
        predicted[i] = ridge.predict(X_test_pca)[0]

    return predicted, n_neural_pcs


# ═════════════════════════════════════════════════════════════════════
# 2b. DECODING DIAGNOSTICS:  permutation test + jackknife
# ═════════════════════════════════════════════════════════════════════

def _decode_dispatch(pop_matrix, targets, method, n_neural_pcs):
    """Dispatch to the right decoder."""
    if method == 'fullpop':
        return loo_decode(pop_matrix, targets)
    elif method == 'neuralpca':
        pred, _ = loo_decode_neural_pca(pop_matrix, targets,
                                        n_neural_pcs=n_neural_pcs)
        return pred
    else:
        raise ValueError(f"Unknown method: {method}")


def decoding_permutation_test(pop_matrix, targets, method='fullpop',
                               n_neural_pcs=5, n_perms=1000, seed=42):
    """Shuffle target monkey labels, redo LOO ridge, get null r distribution.

    Returns dict with observed r and null r distributions for PC1 and PC2.
    """
    rng = np.random.default_rng(seed)
    n = pop_matrix.shape[0]

    # Observed
    pred_obs = _decode_dispatch(pop_matrix, targets, method, n_neural_pcs)
    r_obs = np.array([
        np.corrcoef(targets[:, k], pred_obs[:, k])[0, 1] for k in range(2)
    ])

    # Null
    null_r = np.full((n_perms, 2), np.nan)
    for i in range(n_perms):
        perm = rng.permutation(n)
        targets_perm = targets[perm]
        pred_perm = _decode_dispatch(pop_matrix, targets_perm,
                                     method, n_neural_pcs)
        for k in range(2):
            null_r[i, k] = np.corrcoef(targets_perm[:, k],
                                        pred_perm[:, k])[0, 1]

    # Two-tailed p (since we care about both positive and negative r)
    p_vals = np.array([
        np.mean(np.abs(null_r[:, k]) >= np.abs(r_obs[k])) for k in range(2)
    ])

    return {
        'r_obs': r_obs,
        'null_r': null_r,
        'p_vals': p_vals,
    }


def jackknife_decoding(pop_matrix, targets, monkey_names,
                       method='fullpop', n_neural_pcs=5):
    """For each monkey i, drop it entirely; rerun LOO on the remaining n-1.

    Returns array of shape (n_monkeys, 2): correlation r between actual
    and decoded for PC1 and PC2 after dropping each monkey.
    """
    n = pop_matrix.shape[0]
    jack_r = np.full((n, 2), np.nan)

    for i in range(n):
        keep_idx = np.arange(n) != i
        pop_sub = pop_matrix[keep_idx]
        tgt_sub = targets[keep_idx]

        if pop_sub.shape[0] < 3:    # need at least 3 for LOO with split
            continue

        pred_sub = _decode_dispatch(pop_sub, tgt_sub, method, n_neural_pcs)
        for k in range(2):
            jack_r[i, k] = np.corrcoef(tgt_sub[:, k], pred_sub[:, k])[0, 1]

    return jack_r


def plot_decoding_diagnostics(perm_result, jack_r, monkey_names,
                              group_name, cfg, method_label='Full-Population',
                              save_path=None):
    """2×2 figure: top row = permutation null for PC1, PC2;
       bottom row = jackknife r per monkey for PC1, PC2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    color = cfg.group_colors.get(group_name, 'gray')

    # ── Top row: permutation null distributions ─────────────────────
    for k, pc_label in enumerate(['PC1', 'PC2']):
        ax = axes[0, k]
        ax.hist(perm_result['null_r'][:, k], bins=40,
                color='lightgray', edgecolor='k', lw=0.3,
                label='Null (shuffled labels)')
        r_obs = perm_result['r_obs'][k]
        p_val = perm_result['p_vals'][k]
        ax.axvline(r_obs, color=color, lw=2.5, ls='--',
                   label=f'Observed r = {r_obs:+.3f}')
        ax.axvline(-r_obs, color=color, lw=1.0, ls=':',
                   alpha=0.5, label='−|observed|')
        ax.set_xlabel(f'r(actual, decoded) — {pc_label}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Permutation Test: {pc_label}\n'
                     f'two-tailed p = {p_val:.4f}', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    # ── Bottom row: jackknife r per monkey ──────────────────────────
    for k, pc_label in enumerate(['PC1', 'PC2']):
        ax = axes[1, k]
        x_pos = np.arange(len(monkey_names))
        ax.bar(x_pos, jack_r[:, k], color=color, edgecolor='k', lw=0.5,
               alpha=0.8)
        ax.axhline(perm_result['r_obs'][k], color='k', lw=1.5, ls='--',
                   label=f'Full r = {perm_result["r_obs"][k]:+.3f}')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monkey_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'r after dropping monkey — {pc_label}', fontsize=10)
        ax.set_title(f'Jackknife: {pc_label}', fontsize=10)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'{group_name}: {method_label} Decoding Diagnostics',
                 fontsize=13, y=1.00)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


def print_diagnostics_summary(perm_result, jack_r, monkey_names, label=''):
    """Print permutation and jackknife results."""
    print(f"\n    {label} diagnostics")
    print(f"    Permutation test (n_perms = {len(perm_result['null_r'])}):")
    for k, pc in enumerate(['PC1', 'PC2']):
        print(f"      {pc}: r_obs = {perm_result['r_obs'][k]:+.3f}  "
              f"p (two-tailed) = {perm_result['p_vals'][k]:.4f}")
    print(f"    Jackknife: range of r after dropping any single monkey")
    for k, pc in enumerate(['PC1', 'PC2']):
        vals = jack_r[:, k]
        print(f"      {pc}: min = {np.nanmin(vals):+.3f}  "
              f"max = {np.nanmax(vals):+.3f}  "
              f"std = {np.nanstd(vals):.3f}")
        # Most-influential monkey
        full_r = perm_result['r_obs'][k]
        deltas = np.abs(jack_r[:, k] - full_r)
        i_max = int(np.nanargmax(deltas))
        print(f"        most influential: {monkey_names[i_max]} "
              f"(Δr = {jack_r[i_max, k] - full_r:+.3f})")


# ═════════════════════════════════════════════════════════════════════
# 3. PROCRUSTES + RSA
# ═════════════════════════════════════════════════════════════════════

def procrustes_analysis(X, Y):
    """Procrustes: best rotation/scaling of Y onto X.

    Returns
    -------
    disparity : float  — sum of squared differences after alignment
    Y_aligned : (n, 2) — Y after optimal transform
    """
    # Center
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    # Scale to unit Frobenius norm
    sx = np.sqrt((X_c ** 2).sum())
    sy = np.sqrt((Y_c ** 2).sum())
    X_c /= sx
    Y_c /= sy

    # Optimal rotation via SVD
    U, _, Vt = np.linalg.svd(X_c.T @ Y_c)
    R = (U @ Vt).T

    Y_rot = Y_c @ R

    # Disparity (lower = better match)
    disparity = ((X_c - Y_rot) ** 2).sum()

    # Scale Y_aligned back to X's scale for plotting
    Y_aligned = Y_rot * sx + X.mean(axis=0)

    return disparity, Y_aligned


def rsa_correlation(X, Y):
    """Spearman correlation between pairwise distance vectors."""
    d_x = pdist(X, metric='euclidean')
    d_y = pdist(Y, metric='euclidean')
    rho, p = spearmanr(d_x, d_y)
    return rho, p, d_x, d_y


def permutation_test_geometry(pop_matrix, social_pc,
                               n_neural_pcs_rsa=None,
                               n_perms=5000, seed=42):
    """Permutation test for both Procrustes disparity and RSA rho.

    Procrustes uses a 2D neural projection (must match social PC dimensionality).
    RSA uses pairwise distances in `n_neural_pcs_rsa`-dimensional neural space.

    Parameters
    ----------
    n_neural_pcs_rsa : int or None
        Number of neural PCs to use for RSA distance computation.
        None = use full neural space (no PCA reduction).
        Higher values let RSA detect signal in low-variance neural dimensions
        that Procrustes (restricted to top 2 neural PCs) cannot see.
    """
    rng = np.random.default_rng(seed)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(pop_matrix)

    # ── 2D neural projection for Procrustes (matches social PC dim) ──
    n_comp_proc = min(2, X_sc.shape[0] - 1, X_sc.shape[1])
    pca_proc = PCA(n_components=n_comp_proc)
    neural_2d = pca_proc.fit_transform(X_sc)
    neural_var_2d = pca_proc.explained_variance_ratio_

    # ── Higher-D neural representation for RSA ──
    if n_neural_pcs_rsa is None:
        neural_for_rsa = X_sc                # full neural space
        n_used_rsa = X_sc.shape[1]
        rsa_label = ' '
        # rsa_label = f'full neural space ({n_used_rsa} dims)'
    else:
        n_comp_rsa = min(n_neural_pcs_rsa, X_sc.shape[0] - 1, X_sc.shape[1])
        pca_rsa = PCA(n_components=n_comp_rsa)
        neural_for_rsa = pca_rsa.fit_transform(X_sc)
        n_used_rsa = n_comp_rsa
        rsa_label = f'{n_comp_rsa} neural PCs'

    # Observed metrics
    obs_disp, obs_aligned = procrustes_analysis(social_pc, neural_2d)
    obs_rho, _, d_neural_obs, d_social_obs = rsa_correlation(
        neural_for_rsa, social_pc)

    # Permutation
    null_disp = np.full(n_perms, np.nan)
    null_rho = np.full(n_perms, np.nan)

    for i in range(n_perms):
        perm_idx = rng.permutation(len(social_pc))
        social_perm = social_pc[perm_idx]
        null_disp[i], _ = procrustes_analysis(social_perm, neural_2d)
        null_rho[i], _, _, _ = rsa_correlation(neural_for_rsa, social_perm)

    p_disp = np.nanmean(null_disp <= obs_disp)
    p_rho = np.nanmean(null_rho >= obs_rho)

    return {
        'neural_2d': neural_2d,
        'neural_var': neural_var_2d,
        'aligned_neural': obs_aligned,
        'n_used_rsa': n_used_rsa,
        'rsa_label': rsa_label,
        'd_neural_rsa': d_neural_obs,
        'd_social_rsa': d_social_obs,
        'disparity': obs_disp,
        'p_disparity': p_disp,
        'null_disparity': null_disp,
        'rsa_rho': obs_rho,
        'p_rho': p_rho,
        'null_rho': null_rho,
    }


# ═════════════════════════════════════════════════════════════════════
# 4. PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════

def _draw_scatter_on_ax(ax, positions, monkey_names, pca,
                        group_name, cfg, subtitle=''):
    """Draw a labeled monkey scatter on a given axes."""
    color = cfg.group_colors.get(group_name, 'gray')
    pc1, pc2 = positions[:, 0], positions[:, 1]

    ax.scatter(pc1, pc2, c=color, s=120,
               edgecolors='k', lw=0.8, zorder=5)

    for i, name in enumerate(monkey_names):
        ax.annotate(name, (pc1[i], pc2[i]),
                    textcoords='offset points', xytext=(7, 7),
                    fontsize=8, fontweight='bold', color='k')

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Social PC1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"Social PC2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(subtitle, fontsize=11)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls=':', alpha=0.3)


def compute_decoding_metrics(actual, predicted, monkey_names, label=''):
    """Print per-monkey and aggregate decoding error.  Returns errors."""
    errors = np.linalg.norm(actual - predicted, axis=1)

    print(f"\n    {label}")
    print(f"    {'Monkey':<10s} {'Error (L2)':>10s}")
    print(f"    {'-'*22}")
    for i, name in enumerate(monkey_names):
        print(f"    {name:<10s} {errors[i]:>10.3f}")
    print(f"    {'Mean':<10s} {errors.mean():>10.3f}")
    print(f"    {'Median':<10s} {np.median(errors):>10.3f}")

    for pc_idx, pc_label in enumerate(['PC1', 'PC2']):
        r = np.corrcoef(actual[:, pc_idx], predicted[:, pc_idx])[0, 1]
        print(f"    Corr(actual, decoded) {pc_label}: r = {r:.3f}")

    return errors


# ═════════════════════════════════════════════════════════════════════
# 5. DECODING FIGURES
# ═════════════════════════════════════════════════════════════════════

def _shared_limits(actual_pc, predicted_pc, pad_frac=0.15):
    """Compute shared axis limits across actual and predicted."""
    all_x = np.concatenate([actual_pc[:, 0], predicted_pc[:, 0]])
    all_y = np.concatenate([actual_pc[:, 1], predicted_pc[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * pad_frac
    pad_y = (all_y.max() - all_y.min()) * pad_frac
    return (all_x.min() - pad_x, all_x.max() + pad_x,
            all_y.min() - pad_y, all_y.max() + pad_y)


def plot_decoding_actual_vs_decoded(actual_pc, predicted_pc, monkey_names,
                                     pca, group_name, cfg,
                                     method_label='Full-Population',
                                     save_path=None):
    """Two subplots: (left) actual positions, (right) decoded positions.
    Each subplot auto-scales to its own data for better visibility."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    _draw_scatter_on_ax(ax_left, actual_pc, monkey_names,
                        pca, group_name, cfg,
                        subtitle='Actual Social PC Positions')

    _draw_scatter_on_ax(ax_right, predicted_pc, monkey_names,
                        pca, group_name, cfg,
                        subtitle=f'Decoded Positions (LOO)')

    # Let each subplot auto-scale independently
    # (decoded positions are often compressed — independent scaling
    #  reveals the relative structure even if the absolute spread differs)
    for ax, data in [(ax_left, actual_pc), (ax_right, predicted_pc)]:
        pad_x = (data[:, 0].max() - data[:, 0].min()) * 0.2
        pad_y = (data[:, 1].max() - data[:, 1].min()) * 0.2
        # Guard against near-zero range
        pad_x = max(pad_x, 0.1)
        pad_y = max(pad_y, 0.1)
        ax.set_xlim(data[:, 0].min() - pad_x, data[:, 0].max() + pad_x)
        ax.set_ylim(data[:, 1].min() - pad_y, data[:, 1].max() + pad_y)

    # Note about different scales
    ax_right.text(0.98, 0.02, '(note: different scale from left panel)',
                  transform=ax_right.transAxes, fontsize=7,
                  ha='right', va='bottom', style='italic', color='gray')

    fig.suptitle(f"{group_name}: {method_label} Decoding",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


def plot_decoded_vs_actual_overlay(actual_pc, predicted_pc, monkey_names,
                                    pca, group_name, cfg,
                                    method_label='Full-Population',
                                    save_path=None):
    """Single plot: actual (filled) + decoded (hollow), connected by arrows."""
    fig, ax = plt.subplots(figsize=(7, 6))
    color = cfg.group_colors.get(group_name, 'gray')

    pc1_act, pc2_act = actual_pc[:, 0], actual_pc[:, 1]
    pc1_pred, pc2_pred = predicted_pc[:, 0], predicted_pc[:, 1]

    # Arrows: decoded → actual
    for i in range(len(monkey_names)):
        arrow = FancyArrowPatch(
            (pc1_pred[i], pc2_pred[i]),
            (pc1_act[i], pc2_act[i]),
            arrowstyle='->', mutation_scale=12,
            color='gray', lw=1.0, alpha=0.6, zorder=2)
        ax.add_patch(arrow)

    ax.scatter(pc1_act, pc2_act, c=color, s=120,
               edgecolors='k', lw=0.8, zorder=5, label='Actual')
    ax.scatter(pc1_pred, pc2_pred, facecolors='none',
               edgecolors=color, s=120, lw=2.0, zorder=5,
               label='Decoded (LOO)')

    for i, name in enumerate(monkey_names):
        ax.annotate(name, (pc1_act[i], pc2_act[i]),
                    textcoords='offset points', xytext=(7, 7),
                    fontsize=8, fontweight='bold', color='k')

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Social PC1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"Social PC2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(f"{group_name}: {method_label} — Decoded vs Actual",
                 fontsize=12)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════
# 6. PROCRUSTES / RSA FIGURE
# ═════════════════════════════════════════════════════════════════════
from adjustText import adjust_text

def plot_procrustes(social_pc, geo_result, monkey_names,
                    pca, group_name, cfg, label_mapping=None, save_path=None):
    """Three-panel figure: (a) Procrustes overlay, (b) RSA scatter,
       (c) null distributions for both metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    color = cfg.group_colors.get(group_name, 'gray')

    # ── Panel A: Procrustes overlay ──────────────────────────────────
    ax = axes[0]
    aligned = geo_result['aligned_neural']
    ax.scatter(social_pc[:, 0], social_pc[:, 1],
               c=color, s=120, edgecolors='k', lw=0.8, zorder=5,
               label='Social PCs')
    ax.scatter(aligned[:, 0], aligned[:, 1],
               facecolors='none', edgecolors=color, s=120, lw=2.0,
               zorder=5, label='Neural (aligned)')

    texts = []
    for i in range(len(monkey_names)):
        display_name = label_mapping.get(monkey_names[i], monkey_names[i]) \
            if label_mapping else monkey_names[i]
        arrow = FancyArrowPatch(
            (aligned[i, 0], aligned[i, 1]),
            (social_pc[i, 0], social_pc[i, 1]),
            arrowstyle='->', mutation_scale=12,
            color='gray', lw=1.0, alpha=0.6, zorder=2)
        ax.add_patch(arrow)
        texts.append(ax.text(
            social_pc[i, 0] + 0.1, social_pc[i, 1] + 0.1,
            display_name, fontsize=8, fontweight='bold'
        ))

    adjust_text(
        texts,
        x=social_pc[:, 0],
        y=social_pc[:, 1],
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand=(1.2, 1.4),
    )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Social PC1 ({var1:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"Social PC2 ({var2:.1f}% var)", fontsize=10)
    ax.set_title(f"Procrustes Alignment\n"
                 f"disparity = {geo_result['disparity']:.3f}  "
                 f"(p = {geo_result['p_disparity']:.4f})", fontsize=10)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # ── Panel B: RSA — pairwise distance scatter ─────────────────────
    ax = axes[1]
    d_neural = geo_result['d_neural_rsa']
    d_social = geo_result['d_social_rsa']
    ax.scatter(d_social, d_neural, c=color, s=50, edgecolors='k',
               lw=0.5, alpha=0.7)
    m, b = np.polyfit(d_social, d_neural, 1)
    x_line = np.array([d_social.min(), d_social.max()])
    ax.plot(x_line, m * x_line + b, 'k--', lw=1, alpha=0.6)
    ax.set_xlabel("Social PC pairwise distance", fontsize=10)
    ax.set_ylabel("Neural pairwise distance", fontsize=9)
    ax.set_title(f"RSA\n {geo_result['n_used_rsa']} neurons"
                 f"ρ = {geo_result['rsa_rho']:.3f}  "
                 f"(p = {geo_result['p_rho']:.4f})", fontsize=10)

    # ── Panel C: Null distributions ──────────────────────────────────
    ax = axes[2]
    null_d = geo_result['null_disparity']
    ax.hist(null_d[~np.isnan(null_d)], bins=50, color='steelblue',
            alpha=0.5, edgecolor='k', lw=0.3, label='Null (disparity)')
    ax.axvline(geo_result['disparity'], color='steelblue', lw=2,
               ls='--', label=f"Obs disp={geo_result['disparity']:.3f}")
    ax.set_xlabel("Procrustes disparity", fontsize=10)
    ax.set_ylabel("Count", fontsize=10, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax.twiny()
    null_r = geo_result['null_rho']
    ax2.hist(null_r[~np.isnan(null_r)], bins=50, color='coral',
             alpha=0.4, edgecolor='k', lw=0.3, label='Null (RSA ρ)')
    ax2.axvline(geo_result['rsa_rho'], color='coral', lw=2,
                ls='--', label=f"Obs ρ={geo_result['rsa_rho']:.3f}")
    ax2.set_xlabel("RSA ρ", fontsize=10, color='coral')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    ax.set_title("Permutation Null Distributions", fontsize=10)

    fig.suptitle(f"Unfamiliar Group (Group I): Geometry Comparison (Procrustes and RSA)",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig

# ═════════════════════════════════════════════════════════════════════
# 7. RATE SURFACE (unchanged)
# ═════════════════════════════════════════════════════════════════════

def plot_rate_surface(features_pc, model_result, pca, group_name, cfg,
                      save_path=None):
    """Contour map of fixed-effect predicted rate over PC1 × PC2."""
    fig, ax = plt.subplots(figsize=(7, 6))

    coeffs = model_result['coefficients']
    intercept = coeffs.get('Intercept', 0.0)
    b1 = coeffs.get('social_PC1', 0.0)
    b2 = coeffs.get('social_PC2', 0.0)

    pc1_vals = features_pc['social_PC1'].values
    pc2_vals = features_pc['social_PC2'].values
    pad = 0.3
    x_range = pc1_vals.max() - pc1_vals.min()
    y_range = pc2_vals.max() - pc2_vals.min()
    x_grid = np.linspace(pc1_vals.min() - pad * x_range,
                         pc1_vals.max() + pad * x_range, 200)
    y_grid = np.linspace(pc2_vals.min() - pad * y_range,
                         pc2_vals.max() + pad * y_range, 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = intercept + b1 * X_grid + b2 * Y_grid

    cf = ax.contourf(X_grid, Y_grid, Z_grid, levels=30,
                     cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Predicted firing rate (Hz)\n(fixed effects only)',
                   fontsize=10)

    color = cfg.group_colors.get(group_name, 'gray')
    ax.scatter(pc1_vals, pc2_vals, c=color, s=120,
               edgecolors='k', lw=0.8, zorder=5)

    for i, name in enumerate(features_pc.index):
        ax.annotate(name, (pc1_vals[i], pc2_vals[i]),
                    textcoords='offset points', xytext=(7, 7),
                    fontsize=8, fontweight='bold', color='white',
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground='black')])

    pv1 = model_result['pvalues'].get('social_PC1', np.nan)
    pv2 = model_result['pvalues'].get('social_PC2', np.nan)
    txt = (f"β₁(PC1) = {b1:+.4f}  (p={pv1:.2e})\n"
           f"β₂(PC2) = {b2:+.4f}  (p={pv2:.2e})")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Social PC1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"Social PC2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(f"{group_name}: LMM Rate Surface over Social PC Space",
                 fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    cfg = SocialEncodingConfig(
        region='AMG',
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=2.0,
        min_reps_per_monkey=5,
        exclude_groups=['Stranger Things'],
        exclude_individuals=['70G', '79G', '144H'],
        feature_mode='summary',
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=0,
        save_plots=True,
        save_dir='social_pc_predictions_grant_figures',
        groups=['Instigators'],
    )
    cfg.validate()

    N_NEURAL_PCS = 5       # for Option A (neural-PCA decoding)
    N_GEO_PERMS  = 5000    # for Procrustes / RSA permutation test
    N_DECODE_PERMS = 1000  # for decoding permutation test (per method)

    # For RSA in the geometry analysis:
    #   None = use FULL neural space (recommended — tests whether social
    #          structure is preserved across ALL neural dimensions, not just top)
    #   int  = restrict RSA to that many neural PCs
    # Note: Procrustes is always 2D (matches social PC dimensionality).
    N_NEURAL_PCS_RSA = None

    print(f"\n{'='*60}")
    print(f"Social PC Prediction Plots")
    print(f"  Region: {cfg.region}  |  Window: "
          f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Feature mode: {cfg.feature_mode}  |  PCs: {cfg.n_feature_pcs}")
    print(f"  Neural PCs for decoding: {N_NEURAL_PCS}")
    print(f"  Geometry permutations: {N_GEO_PERMS}")
    print(f"{'='*60}\n")

    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
    if cfg.exclude_individuals:
        df = df[~df['MonkeyName'].isin(cfg.exclude_individuals)].reset_index(drop=True)
    df = compute_trial_rates(df, cfg.window)

    save_base = (f"{cfg.save_dir}/{cfg.region}_{cfg.feature_mode}/"
                 f"{cfg.window[0]*1000:.0f}_{cfg.window[1]*1000:.0f}")
    os.makedirs(save_base, exist_ok=True)

    for group in cfg.groups:
        print(f"\n{'='*60}")
        print(f"Group: {group}")
        print(f"{'='*60}")

        # Social features → PCA
        features = load_social_features(group, cfg)
        if cfg.subject_name in features.index:
            features = features.drop(cfg.subject_name)
        features_pc, pca, scaler = reduce_features(
            features, n_components=cfg.n_feature_pcs)

        # Trial table + LMM
        trial_df = build_trial_table(df, features_pc, group, cfg)
        if trial_df is None or len(trial_df) == 0:
            print(f"  SKIP — no valid trials for {group}")
            continue

        model_result = fit_mixed_model(trial_df, cfg.n_feature_pcs)
        print(f"  R² (Nakagawa marginal): "
              f"{model_result['marginal_r2_nakagawa']:.6f}")

        # Population vectors
        pop_matrix, monkey_names, neuron_ids = build_population_vectors(trial_df)
        actual_pc = features_pc.loc[monkey_names][
            ['social_PC1', 'social_PC2']].values
        print(f"  Population matrix: {pop_matrix.shape[0]} monkeys × "
              f"{pop_matrix.shape[1]} neurons")

        # ── DECODING 1: Full population vector ──────────────────────
        print(f"\n  --- Decoding: Full-population LOO ridge ---")
        pred_full = loo_decode(pop_matrix, actual_pc)
        compute_decoding_metrics(actual_pc, pred_full, monkey_names,
                                 label='[Full-Pop]')

        plot_decoding_actual_vs_decoded(
            actual_pc, pred_full, monkey_names, pca, group, cfg,
            method_label='Full-Population',
            save_path=f"{save_base}/decoded_fullpop_{group}.png")

        plot_decoded_vs_actual_overlay(
            actual_pc, pred_full, monkey_names, pca, group, cfg,
            method_label='Full-Population',
            save_path=f"{save_base}/overlay_fullpop_{group}.png")

        # Diagnostics: permutation test + jackknife
        print(f"\n  --- Diagnostics: Full-Population ---")
        perm_full = decoding_permutation_test(
            pop_matrix, actual_pc, method='fullpop',
            n_perms=N_DECODE_PERMS, seed=42)
        jack_full = jackknife_decoding(
            pop_matrix, actual_pc, monkey_names, method='fullpop')
        print_diagnostics_summary(perm_full, jack_full, monkey_names,
                                  label='[Full-Pop]')
        plot_decoding_diagnostics(
            perm_full, jack_full, monkey_names, group, cfg,
            method_label='Full-Population',
            save_path=f"{save_base}/diagnostics_fullpop_{group}.png")

        # ── DECODING 2: Neural-PCA first ────────────────────────────
        print(f"\n  --- Decoding: Neural-PCA ({N_NEURAL_PCS} PCs) + LOO ridge ---")
        pred_pca, n_used = loo_decode_neural_pca(
            pop_matrix, actual_pc, n_neural_pcs=N_NEURAL_PCS)
        compute_decoding_metrics(actual_pc, pred_pca, monkey_names,
                                 label=f'[Neural-PCA, {n_used} PCs]')

        plot_decoding_actual_vs_decoded(
            actual_pc, pred_pca, monkey_names, pca, group, cfg,
            method_label=f'Neural-PCA ({n_used} PCs)',
            save_path=f"{save_base}/decoded_neuralpca_{group}.png")

        plot_decoded_vs_actual_overlay(
            actual_pc, pred_pca, monkey_names, pca, group, cfg,
            method_label=f'Neural-PCA ({n_used} PCs)',
            save_path=f"{save_base}/overlay_neuralpca_{group}.png")

        # Diagnostics: permutation test + jackknife
        print(f"\n  --- Diagnostics: Neural-PCA ---")
        perm_pca = decoding_permutation_test(
            pop_matrix, actual_pc, method='neuralpca',
            n_neural_pcs=N_NEURAL_PCS,
            n_perms=N_DECODE_PERMS, seed=42)
        jack_pca = jackknife_decoding(
            pop_matrix, actual_pc, monkey_names, method='neuralpca',
            n_neural_pcs=N_NEURAL_PCS)
        print_diagnostics_summary(perm_pca, jack_pca, monkey_names,
                                  label=f'[Neural-PCA, {n_used} PCs]')
        plot_decoding_diagnostics(
            perm_pca, jack_pca, monkey_names, group, cfg,
            method_label=f'Neural-PCA ({n_used} PCs)',
            save_path=f"{save_base}/diagnostics_neuralpca_{group}.png")

        # ── GEOMETRY: Procrustes + RSA ──────────────────────────────
        print(f"\n  --- Geometry comparison (Procrustes + RSA) ---")

        geo_result = permutation_test_geometry(
            pop_matrix, actual_pc,
            n_neural_pcs_rsa=N_NEURAL_PCS_RSA,
            n_perms=N_GEO_PERMS, seed=42)
        print(f"    Neural PCA var explained (2D for Procrustes): "
              f"{geo_result['neural_var'].round(3)}")
        print(f"    RSA neural representation: {geo_result['rsa_label']}")
        print(f"    Procrustes disparity: {geo_result['disparity']:.4f}  "
              f"(p = {geo_result['p_disparity']:.4f})")
        print(f"    RSA Spearman ρ:       {geo_result['rsa_rho']:.4f}  "
              f"(p = {geo_result['p_rho']:.4f})")

        # label_for_r01 = {
        #     '7124': 'Z0',
        #     '69X': 'Z1',
        #     '72X': 'Z2',
        #     '94B': 'Z3',
        #     '110E': 'Z4',
        #     '67G': 'Z5',
        #     '81G': 'Z6',
        #     '143H': 'Z7',
        #     '87J': 'Z8',
        #     '151J': 'Z9',
        # }

        # label_for_r01 = {
        #     'G942': '0',
        #     '35Y': '1',
        #     '49Y': '2',
        #     '42Z': '3',
        #     '48Z': '4',
        #     '59E': '5',
        #     '68E': '6',
        #     '86I': '7',
        #     '167I': '8',
        #     '114J': '9',
        # }

        plot_procrustes(
            actual_pc, geo_result, monkey_names, pca, group, cfg, label_mapping=None,
            save_path=f"{save_base}/procrustes_rsa_{group}_R01.png")

        # ── RATE SURFACE ────────────────────────────────────────────
        print(f"\n  --- Rate surface ---")
        plot_rate_surface(
            features_pc, model_result, pca, group, cfg,
            save_path=f"{save_base}/rate_surface_{group}.png")

    plt.show()
    print(f"\nAll plots saved to {save_base}/")


if __name__ == '__main__':
    main()
