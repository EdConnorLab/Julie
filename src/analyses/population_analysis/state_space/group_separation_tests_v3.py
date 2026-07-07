# group_separation_tests_v3.py
"""
v3 changes from v2 — addressing the two pathologies that masked real signal:

  (a) CENTROID DISTANCE now uses top N_PC_CENTROID PCs (default 3), not all 6.
      With a flat scree plot, PCs 4–6 add noise faster than signal and dilute
      the omnibus distance. Pairwise z-values were stronger than the omnibus
      in v2 because of this. Using top 3 PCs concentrates the test on the
      dimensions where group structure actually lives.

  (b) LOO DECODER now operates on top N_PC_DECODER PCs (default 6), not raw
      neurons. The raw-neuron decoder had 123-170 features for 14-33 samples
      and was severely underdetermined — bootstrap CIs spanned ~30 accuracy
      points because every LOO split made wildly different decisions on the
      held-out point. Decoding in low-D PC space gives a well-posed problem.

All other v2 mechanics are unchanged: pairwise breakdown, null-z, FDR across
pairs, stratified bootstrap CIs, time-resolved windowed analysis, vlines
for CI display.

A DECODE_ON setting lets you toggle decoder to 'raw' for comparison if you
want to confirm v3 PC-decoder is tighter than v2 raw-neuron decoder.
"""

import os
import pickle
from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from analyses.population_analysis.state_space.preprocessing import preprocess
from pca_runner import run_pca

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"


# ═══════════════════════════════════════════════════════════════════════
# CORE STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def centroid_distances(features, labels):
    """
    Omnibus = mean of pairwise centroid distances.
    Returns (omnibus_mean, pairwise_dict {(g1,g2): dist}).
    """
    groups = np.unique(labels)
    centroids = {g: features[labels == g].mean(axis=0) for g in groups}
    pairwise = {}
    for g1, g2 in combinations(groups, 2):
        pairwise[(g1, g2)] = float(np.linalg.norm(centroids[g1] - centroids[g2]))
    omnibus = float(np.mean(list(pairwise.values())))
    return omnibus, pairwise


def loo_decode(features, labels, seed=42):
    """
    Leave-one-identity-out classifier accuracy.
    Multi-class if >2 unique labels, binary otherwise.
    """
    n = len(features)
    correct = 0
    valid = 0
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=1.0,
                                   random_state=seed, solver='lbfgs')),
    ])
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_tr = features[train_idx]
        y_tr = labels[train_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(features[i:i + 1])[0]
        if pred == labels[i]:
            correct += 1
        valid += 1
    return correct / valid if valid > 0 else np.nan


def pairwise_loo_decode(features, labels, g1, g2, seed=42):
    """Binary LOO decoder restricted to the two named groups."""
    mask = (labels == g1) | (labels == g2)
    return loo_decode(features[mask], labels[mask], seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# PERMUTATION / BOOTSTRAP / EFFECT SIZE
# ═══════════════════════════════════════════════════════════════════════

def null_z(obs, null_dist):
    """Standardized effect size: (obs - mean(null)) / std(null)."""
    null_dist = np.asarray(null_dist)
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.std(null_dist) == 0:
        return np.nan
    return float((obs - null_dist.mean()) / null_dist.std())


def p_from_null(obs, null_dist, tail='upper'):
    """One-sided permutation p-value (fraction of null >= obs)."""
    null_dist = np.asarray(null_dist)
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.isnan(obs):
        return np.nan
    if tail == 'upper':
        return float((null_dist >= obs).sum() / len(null_dist))
    elif tail == 'lower':
        return float((null_dist <= obs).sum() / len(null_dist))
    else:
        raise ValueError(tail)


def fdr_bh(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values. NaNs are preserved."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite = ~np.isnan(p)
    if finite.sum() == 0:
        return out
    p_finite = p[finite]
    n = len(p_finite)
    order = np.argsort(p_finite)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = p_finite * n / ranks
    # enforce monotonicity from the largest p downward
    sorted_idx = order[::-1]
    cummin = np.inf
    for i in sorted_idx:
        cummin = min(cummin, adj[i])
        adj[i] = cummin
    adj = np.clip(adj, 0, 1)
    out[finite] = adj
    return out


def stratified_bootstrap(features, labels, statistic_fn, n_boot=1000,
                        ci=95, seed=42):
    """
    Bootstrap by resampling identities WITH REPLACEMENT within each group.
    Returns (lo, hi, boot_distribution).
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    group_idx_lookup = {g: np.where(labels == g)[0] for g in groups}

    boot_values = np.full(n_boot, np.nan)
    for b in range(n_boot):
        boot_idx = []
        for g in groups:
            idx_g = group_idx_lookup[g]
            boot_idx.append(rng.choice(idx_g, size=len(idx_g), replace=True))
        boot_idx = np.concatenate(boot_idx)
        try:
            boot_values[b] = statistic_fn(features[boot_idx], labels[boot_idx])
        except Exception:
            pass

    finite = boot_values[~np.isnan(boot_values)]
    if len(finite) == 0:
        return np.nan, np.nan, boot_values
    alpha = (100 - ci) / 2
    lo = float(np.percentile(finite, alpha))
    hi = float(np.percentile(finite, 100 - alpha))
    return lo, hi, boot_values


# ═══════════════════════════════════════════════════════════════════════
# TIME-AVERAGED PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_centroid_test(features, labels, n_perms=1000, n_boot=1000, seed=42):
    """
    Centroid distance: omnibus + pairwise.
    For each: observed, null (perm), null-z, p, FDR-adjusted p, bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    pairs = list(combinations(groups, 2))

    # Observed
    obs_omni, obs_pair = centroid_distances(features, labels)

    # Permutation null
    null_omni = np.zeros(n_perms)
    null_pair = {p: np.zeros(n_perms) for p in pairs}
    for i in range(n_perms):
        shuf = rng.permutation(labels)
        omni_i, pair_i = centroid_distances(features, shuf)
        null_omni[i] = omni_i
        for p in pairs:
            # pair_i may be keyed in the order that np.unique returned;
            # combinations(np.unique(...)) yields the same canonical order
            null_pair[p][i] = pair_i.get(p, pair_i.get((p[1], p[0]), np.nan))

    # Bootstrap CIs
    omni_lo, omni_hi, _ = stratified_bootstrap(
        features, labels,
        lambda f, l: centroid_distances(f, l)[0],
        n_boot=n_boot, seed=seed)
    pair_ci = {}
    for p in pairs:
        g1, g2 = p
        # restrict bootstrap to the two groups
        mask = (labels == g1) | (labels == g2)
        sub_f = features[mask]
        sub_l = labels[mask]
        lo, hi, _ = stratified_bootstrap(
            sub_f, sub_l,
            lambda f, l: centroid_distances(f, l)[0],
            n_boot=n_boot, seed=seed)
        pair_ci[p] = (lo, hi)

    # Effect sizes and p-values
    omni_z = null_z(obs_omni, null_omni)
    omni_p = p_from_null(obs_omni, null_omni, tail='upper')

    pair_results = {}
    raw_p = []
    for p in pairs:
        z = null_z(obs_pair[p], null_pair[p])
        pv = p_from_null(obs_pair[p], null_pair[p], tail='upper')
        pair_results[p] = dict(obs=obs_pair[p], null=null_pair[p],
                               z=z, p=pv, ci=pair_ci[p])
        raw_p.append(pv)
    # FDR across pairs
    adj_p = fdr_bh(raw_p)
    for p, padj in zip(pairs, adj_p):
        pair_results[p]['p_adj'] = float(padj)

    return dict(
        omnibus=dict(obs=obs_omni, null=null_omni, z=omni_z, p=omni_p,
                     ci=(omni_lo, omni_hi)),
        pairwise=pair_results,
        pairs=pairs,
    )


def run_decoder_test(features, labels, n_perms=1000, n_boot=1000, seed=42):
    """
    LOO decoder: omnibus (multi-class) + pairwise (binary, restricted).
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    pairs = list(combinations(groups, 2))

    # Observed
    obs_omni = loo_decode(features, labels, seed=seed)
    obs_pair = {p: pairwise_loo_decode(features, labels, p[0], p[1], seed=seed)
                for p in pairs}

    # Permutation null
    null_omni = np.zeros(n_perms)
    null_pair = {p: np.zeros(n_perms) for p in pairs}
    for i in range(n_perms):
        shuf = rng.permutation(labels)
        null_omni[i] = loo_decode(features, shuf, seed=seed)
        for p in pairs:
            null_pair[p][i] = pairwise_loo_decode(features, shuf, p[0], p[1],
                                                  seed=seed)

    # Bootstrap CIs
    omni_lo, omni_hi, _ = stratified_bootstrap(
        features, labels,
        lambda f, l: loo_decode(f, l, seed=seed),
        n_boot=n_boot, seed=seed)
    pair_ci = {}
    for p in pairs:
        g1, g2 = p
        mask = (labels == g1) | (labels == g2)
        lo, hi, _ = stratified_bootstrap(
            features[mask], labels[mask],
            lambda f, l: loo_decode(f, l, seed=seed),
            n_boot=n_boot, seed=seed)
        pair_ci[p] = (lo, hi)

    # Effect sizes and p-values
    omni_z = null_z(obs_omni, null_omni)
    omni_p = p_from_null(obs_omni, null_omni, tail='upper')

    pair_results = {}
    raw_p = []
    for p in pairs:
        z = null_z(obs_pair[p], null_pair[p])
        pv = p_from_null(obs_pair[p], null_pair[p], tail='upper')
        pair_results[p] = dict(obs=obs_pair[p], null=null_pair[p],
                               z=z, p=pv, ci=pair_ci[p])
        raw_p.append(pv)
    adj_p = fdr_bh(raw_p)
    for p, padj in zip(pairs, adj_p):
        pair_results[p]['p_adj'] = float(padj)

    return dict(
        omnibus=dict(obs=obs_omni, null=null_omni, z=omni_z, p=omni_p,
                     ci=(omni_lo, omni_hi)),
        pairwise=pair_results,
        pairs=pairs,
    )


# ═══════════════════════════════════════════════════════════════════════
# TIME-RESOLVED PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def downsample_to_windows(data_3d, n_windows):
    """
    data_3d: (n_ids, n_bins, n_features)
    Returns (n_ids, n_windows, n_features), averaged within window.
    """
    n_ids, n_bins, n_feat = data_3d.shape
    if n_bins % n_windows != 0:
        # truncate so it divides evenly
        keep = (n_bins // n_windows) * n_windows
        data_3d = data_3d[:, :keep, :]
        n_bins = keep
    bins_per_win = n_bins // n_windows
    reshaped = data_3d.reshape(n_ids, n_windows, bins_per_win, n_feat)
    return reshaped.mean(axis=2)


def run_time_resolved(features_3d, labels, n_perms=500, n_boot=500, seed=42,
                      kind='centroid'):
    """
    Compute centroid distance or LOO decoder accuracy per time window.
    features_3d: (n_ids, n_windows, n_features)
    Returns dict with per-window obs, null (mean, 5/95%ile), z, p, CI.
    """
    n_ids, n_windows, _ = features_3d.shape
    rng = np.random.default_rng(seed)

    obs = np.full(n_windows, np.nan)
    null_mean = np.full(n_windows, np.nan)
    null_lo = np.full(n_windows, np.nan)
    null_hi = np.full(n_windows, np.nan)
    z = np.full(n_windows, np.nan)
    p = np.full(n_windows, np.nan)
    ci_lo = np.full(n_windows, np.nan)
    ci_hi = np.full(n_windows, np.nan)

    if kind == 'centroid':
        stat = lambda f, l: centroid_distances(f, l)[0]
    elif kind == 'decoder':
        stat = lambda f, l: loo_decode(f, l, seed=seed)
    else:
        raise ValueError(kind)

    for w in range(n_windows):
        f_w = features_3d[:, w, :]
        obs[w] = stat(f_w, labels)

        # permutation null
        null_w = np.zeros(n_perms)
        for i in range(n_perms):
            shuf = rng.permutation(labels)
            null_w[i] = stat(f_w, shuf)
        null_mean[w] = np.nanmean(null_w)
        null_lo[w] = np.nanpercentile(null_w, 5)
        null_hi[w] = np.nanpercentile(null_w, 95)
        z[w] = null_z(obs[w], null_w)
        p[w] = p_from_null(obs[w], null_w, tail='upper')

        # bootstrap CI on observed
        lo, hi, _ = stratified_bootstrap(f_w, labels, stat,
                                        n_boot=n_boot, seed=seed)
        ci_lo[w] = lo
        ci_hi[w] = hi

    # FDR across windows
    p_adj = fdr_bh(p)

    return dict(obs=obs, null_mean=null_mean, null_lo=null_lo, null_hi=null_hi,
                z=z, p=p, p_adj=p_adj, ci_lo=ci_lo, ci_hi=ci_hi,
                n_windows=n_windows, kind=kind)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def _pair_label(p):
    return f'{str(p[0])[:8]} vs {str(p[1])[:8]}'


def plot_time_averaged(centroid_res, decoder_res, region, output_path,
                       centroid_label='top 3 PCs',
                       decoder_label='top 6 PCs'):
    """
    2 rows × 2 cols
      Row 1: centroid  | omnibus null hist (left) | pairwise bar with CIs + null-z (right)
      Row 2: decoder   | omnibus null hist (left) | pairwise bar with CIs + null-z (right)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ─── row 1: centroid ─────────────────────────────────────────
    ax = axes[0, 0]
    om = centroid_res['omnibus']
    ax.hist(om['null'], bins=30, color='steelblue', alpha=0.6,
            edgecolor='black', linewidth=0.4)
    ax.axvline(om['obs'], color='red', lw=2,
               label=f"obs = {om['obs']:.3f}")
    ax.axvspan(om['ci'][0], om['ci'][1], color='red', alpha=0.15,
               label=f"95% CI [{om['ci'][0]:.3f}, {om['ci'][1]:.3f}]")
    p_str = "p < 0.001" if om['p'] < 1e-3 else f"p = {om['p']:.3f}"
    ax.set_title(f"Centroid distance (omnibus, {centroid_label})\n"
                 f"z = {om['z']:.2f},  {p_str}")
    ax.set_xlabel('mean pairwise centroid dist')
    ax.set_ylabel('# permutations')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    pairs = centroid_res['pairs']
    labels_str = [_pair_label(p) for p in pairs]
    obs_vals = [centroid_res['pairwise'][p]['obs'] for p in pairs]
    null_means = [centroid_res['pairwise'][p]['null'].mean() for p in pairs]
    ci_los = [centroid_res['pairwise'][p]['ci'][0] for p in pairs]
    ci_his = [centroid_res['pairwise'][p]['ci'][1] for p in pairs]
    z_vals = [centroid_res['pairwise'][p]['z'] for p in pairs]
    padj_vals = [centroid_res['pairwise'][p]['p_adj'] for p in pairs]

    x = np.arange(len(pairs))
    ax.bar(x, obs_vals, color='steelblue', alpha=0.55,
           edgecolor='black', linewidth=0.5, label='observed')
    # CI as vertical line with caps — works even when CI doesn't bracket obs
    ax.vlines(x, ci_los, ci_his, color='black', lw=2, alpha=0.85,
              label='95% CI (boot)')
    cap_half = 0.15
    ax.hlines(ci_los, x - cap_half, x + cap_half, color='black', lw=1.5, alpha=0.85)
    ax.hlines(ci_his, x - cap_half, x + cap_half, color='black', lw=1.5, alpha=0.85)
    ax.scatter(x, null_means, color='gray', marker='_', s=200,
               label='null mean', zorder=3)
    # Determine a stable y position for z-labels: top of CI per bar, with
    # an axis-relative cap so labels don't get pushed off the plot
    y_max_data = max(np.nanmax(ci_his), np.nanmax(obs_vals))
    label_y_cap = y_max_data * 1.18
    for i, (z_, pa) in enumerate(zip(z_vals, padj_vals)):
        star = '***' if pa < 0.001 else ('**' if pa < 0.01 else
               ('*' if pa < 0.05 else 'n.s.'))
        # Color and weight the label based on direction and significance
        if pa < 0.05:
            txt_color = 'darkred' if z_ > 0 else 'darkblue'
        else:
            txt_color = 'black' if z_ > 0 else 'steelblue'
        y_top = min(max(ci_his[i], obs_vals[i]) * 1.04, label_y_cap)
        ax.text(i, y_top, f'z = {z_:+.1f}\n{star}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=txt_color,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='lightgray',
                          alpha=0.9, linewidth=0.5))
    # Extend ylim a bit so labels are not clipped
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, label_y_cap * 1.10))
    ax.set_xticks(x)
    ax.set_xticklabels(labels_str, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('centroid distance')
    ax.set_title('Centroid distance — pairwise (FDR-adjusted)')
    ax.legend(fontsize=8, loc='lower right')

    # ─── row 2: decoder ─────────────────────────────────────────
    ax = axes[1, 0]
    om = decoder_res['omnibus']
    ax.hist(om['null'], bins=30, color='seagreen', alpha=0.6,
            edgecolor='black', linewidth=0.4)
    ax.axvline(om['obs'], color='red', lw=2,
               label=f"obs = {om['obs']:.3f}")
    ax.axvspan(om['ci'][0], om['ci'][1], color='red', alpha=0.15,
               label=f"95% CI [{om['ci'][0]:.3f}, {om['ci'][1]:.3f}]")
    p_str = "p < 0.001" if om['p'] < 1e-3 else f"p = {om['p']:.3f}"
    ax.set_title(f"LOO decoder (multi-class, {decoder_label})\n"
                 f"z = {om['z']:.2f},  {p_str}")
    ax.set_xlabel('LOO accuracy')
    ax.set_ylabel('# permutations')
    # chance level = 1 / n_groups
    n_groups = len({g for pair in pairs for g in pair})
    ax.axvline(1.0 / n_groups, color='black', ls=':', lw=1, alpha=0.5,
               label=f'chance (1/{n_groups})')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    obs_vals = [decoder_res['pairwise'][p]['obs'] for p in pairs]
    null_means = [decoder_res['pairwise'][p]['null'].mean() for p in pairs]
    ci_los = [decoder_res['pairwise'][p]['ci'][0] for p in pairs]
    ci_his = [decoder_res['pairwise'][p]['ci'][1] for p in pairs]
    z_vals = [decoder_res['pairwise'][p]['z'] for p in pairs]
    padj_vals = [decoder_res['pairwise'][p]['p_adj'] for p in pairs]

    ax.bar(x, obs_vals, color='seagreen', alpha=0.55,
           edgecolor='black', linewidth=0.5, label='observed')
    ax.vlines(x, ci_los, ci_his, color='black', lw=2, alpha=0.85,
              label='95% CI (boot)')
    cap_half = 0.15
    ax.hlines(ci_los, x - cap_half, x + cap_half, color='black', lw=1.5, alpha=0.85)
    ax.hlines(ci_his, x - cap_half, x + cap_half, color='black', lw=1.5, alpha=0.85)
    ax.scatter(x, null_means, color='gray', marker='_', s=200,
               label='null mean', zorder=3)
    ax.axhline(0.5, color='black', ls=':', lw=1, alpha=0.5,
               label='chance (binary)')
    # Bounded y position for z-labels; decoder y-axis is [0, 1.1]
    label_y_cap = 1.05
    for i, (z_, pa) in enumerate(zip(z_vals, padj_vals)):
        star = '***' if pa < 0.001 else ('**' if pa < 0.01 else
               ('*' if pa < 0.05 else 'n.s.'))
        if pa < 0.05:
            txt_color = 'darkred' if z_ > 0 else 'darkblue'
        else:
            txt_color = 'black' if z_ > 0 else 'steelblue'
        y_top = min(max(ci_his[i], obs_vals[i]) * 1.02 + 0.02, label_y_cap)
        ax.text(i, y_top, f'z = {z_:+.1f}\n{star}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=txt_color,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='lightgray',
                          alpha=0.9, linewidth=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels_str, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('binary LOO accuracy')
    ax.set_title('LOO decoder — pairwise (FDR-adjusted)')
    ax.set_ylim(0, 1.18)  # a bit taller so z-labels are not clipped
    ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(f'Time-averaged group separation — {region}', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_time_resolved(centroid_tr, decoder_tr, bin_width, region, output_path,
                       centroid_label='top 3 PCs',
                       decoder_label='top 6 PCs'):
    """
    1 row × 2 cols
      Left: centroid distance vs time, observed + bootstrap CI + null band
      Right: decoder accuracy vs time, same structure
    """
    n_windows = centroid_tr['n_windows']
    # window centers in seconds
    epoch_dur = 2.0  # assumed; could pass in
    win_dur = epoch_dur / n_windows
    t = (np.arange(n_windows) + 0.5) * win_dur

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── centroid panel
    ax = axes[0]
    ax.fill_between(t, centroid_tr['null_lo'], centroid_tr['null_hi'],
                    color='gray', alpha=0.25, label='null 5-95%')
    ax.plot(t, centroid_tr['null_mean'], color='gray', lw=1, ls='--',
            label='null mean')
    ax.fill_between(t, centroid_tr['ci_lo'], centroid_tr['ci_hi'],
                    color='steelblue', alpha=0.3, label='obs 95% CI (boot)')
    ax.plot(t, centroid_tr['obs'], color='steelblue', lw=2,
            marker='o', label='observed')
    for i, pa in enumerate(centroid_tr['p_adj']):
        if not np.isnan(pa) and pa < 0.05:
            ax.scatter(t[i], centroid_tr['obs'][i], color='red',
                       marker='*', s=80, zorder=5)
    ax.set_xlabel('time from stim onset (s)')
    ax.set_ylabel(f'centroid distance ({centroid_label})')
    ax.set_title('Centroid distance over time')
    ax.legend(fontsize=8)

    # ── decoder panel
    ax = axes[1]
    ax.fill_between(t, decoder_tr['null_lo'], decoder_tr['null_hi'],
                    color='gray', alpha=0.25, label='null 5-95%')
    ax.plot(t, decoder_tr['null_mean'], color='gray', lw=1, ls='--',
            label='null mean')
    ax.fill_between(t, decoder_tr['ci_lo'], decoder_tr['ci_hi'],
                    color='seagreen', alpha=0.3, label='obs 95% CI (boot)')
    ax.plot(t, decoder_tr['obs'], color='seagreen', lw=2,
            marker='o', label='observed')
    for i, pa in enumerate(decoder_tr['p_adj']):
        if not np.isnan(pa) and pa < 0.05:
            ax.scatter(t[i], decoder_tr['obs'][i], color='red',
                       marker='*', s=80, zorder=5)
    ax.axhline(0.25, color='black', ls=':', lw=1, alpha=0.5,
               label='chance (4-way)')
    ax.set_xlabel('time from stim onset (s)')
    ax.set_ylabel(f'LOO accuracy ({decoder_label})')
    ax.set_title('LOO decoder over time')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    fig.suptitle(f'Time-resolved group separation — {region}  '
                 f'(red ★ = FDR-adjusted p < 0.05)', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# PER-REGION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def analyze_region(region, n_perms=1000, n_boot=1000, n_perms_tr=500,
                   n_boot_tr=500, n_windows=10, mean_center=True,
                   soft_normalize=True, n_components=6,
                   n_pc_centroid=3, n_pc_decoder=6, decode_on='pc',
                   min_reps=5, seed=42, output_dir=None):
    print(f"\n{'═' * 76}")
    print(f"  GROUP SEPARATION TESTS v3 — region = {region}")
    print(f"{'═' * 76}")

    # Sanity: PCA must produce enough components for the chosen tests
    n_components = max(n_components, n_pc_centroid, n_pc_decoder)

    cfg = TrajectoryConfig(
        region=region, session=None, trial_averaged=True, peak_align=False,
        n_components=n_components, bin_width=0.050, min_epoch_duration=2.0,
        analysis='identity',
    )
    cfg.validate()

    if output_dir is None:
        output_dir = f'./group_separation_tests_v3_{region}'
    os.makedirs(output_dir, exist_ok=True)

    # Load + build matrix in identity mode (mirrors run_trajectory_by_condition.py)
    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)

    monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
    common = sorted(set.intersection(*monkeys_per_session))
    name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
    known = [m for m in common if m in name_to_group]
    condition_map = {m: m for m in known}
    group_map = {m: name_to_group[m] for m in known}

    pca_matrix, row_meta_df, info = build_matrix_by_condition(
        df, cfg, condition_map, group_map=group_map, min_reps=min_reps)

    pca_matrix_pp = preprocess(pca_matrix, info, cfg,
                               soft_normalize=soft_normalize,
                               mean_center=mean_center)
    pca_result = run_pca(pca_matrix_pp, info, cfg)

    # Per-identity feature arrays
    identities = list(info['conditions'])
    n_bins = info['n_bins']
    n_neurons = pca_matrix_pp.shape[1]
    n_ids = len(identities)

    scores_3d = pca_result['scores_3d']                 # (n_ids, n_bins, n_components)
    neuron_3d = pca_matrix_pp.reshape(n_ids, n_bins, n_neurons)

    pc_mean_full = scores_3d.mean(axis=1)               # (n_ids, n_components)
    neuron_mean = neuron_3d.mean(axis=1)                # (n_ids, n_neurons)

    # === v3 feature slicing ===
    centroid_features = pc_mean_full[:, :n_pc_centroid]                  # for centroid test
    if decode_on == 'pc':
        decoder_features = pc_mean_full[:, :n_pc_decoder]                # for LOO decoder
        decoder_label = f'top {n_pc_decoder} PCs'
    elif decode_on == 'raw':
        decoder_features = neuron_mean
        decoder_label = f'raw neurons ({n_neurons}D)'
    else:
        raise ValueError(f"decode_on must be 'pc' or 'raw', got {decode_on}")

    # Time-windowed (downsampled for time-resolved tests)
    pc_windowed_full = downsample_to_windows(scores_3d, n_windows)
    neuron_windowed = downsample_to_windows(neuron_3d, n_windows)
    centroid_features_tr = pc_windowed_full[:, :, :n_pc_centroid]        # (n_ids, n_windows, n_pc_centroid)
    if decode_on == 'pc':
        decoder_features_tr = pc_windowed_full[:, :, :n_pc_decoder]
    else:
        decoder_features_tr = neuron_windowed

    group_labels = np.array([group_map[i] for i in identities])

    var = pca_result['var']
    cum_centroid = float(var[:n_pc_centroid].sum())
    cum_decoder = float(var[:n_pc_decoder].sum()) if decode_on == 'pc' else None

    print(f"\n  Identities ({n_ids}):")
    for g in sorted(np.unique(group_labels)):
        members = [i for i, l in zip(identities, group_labels) if l == g]
        print(f"    {g:<20s} ({len(members)}): {', '.join(members)}")
    print(f"\n  Settings:")
    print(f"    mean_center = {mean_center}")
    print(f"    soft_normalize = {soft_normalize}")
    print(f"    PCA n_components (fit) = {n_components}")
    print(f"    centroid test = top {n_pc_centroid} PCs "
          f"(cum var = {cum_centroid:.1%})")
    if decode_on == 'pc':
        print(f"    decoder test = top {n_pc_decoder} PCs "
              f"(cum var = {cum_decoder:.1%})")
    else:
        print(f"    decoder test = raw neurons ({n_neurons}D)")
    print(f"    n_perms (time-avg) = {n_perms}, n_boot (time-avg) = {n_boot}")
    print(f"    n_perms (time-res) = {n_perms_tr}, n_boot (time-res) = {n_boot_tr}")
    print(f"    n_windows = {n_windows}")
    print(f"  PCA variance: " +
          ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))

    # Distinct permutations sanity check
    _, inv = np.unique(group_labels, return_inverse=True)
    grp_sizes = np.bincount(inv)
    n_distinct = factorial(n_ids) // int(np.prod([factorial(int(s)) for s in grp_sizes]))
    print(f"  Distinct label permutations possible: {n_distinct}")
    if n_distinct < n_perms:
        print(f"  ⚠  Asked for {n_perms}, only {n_distinct} distinct — null is discrete.")

    # ── TIME-AVERAGED TESTS ───────────────────────────────────────────
    print(f"\n{'─' * 76}")
    print(f"  TIME-AVERAGED TESTS")
    print(f"{'─' * 76}")

    print(f"\n  [1/2] Centroid distance — features = top {n_pc_centroid} PCs "
          f"({cum_centroid:.1%} cum var)")
    centroid_res = run_centroid_test(centroid_features, group_labels,
                                     n_perms=n_perms, n_boot=n_boot, seed=seed)
    _print_test_table(centroid_res, label='centroid distance')

    print(f"\n  [2/2] LOO decoder — features = {decoder_label}")
    decoder_res = run_decoder_test(decoder_features, group_labels,
                                   n_perms=n_perms, n_boot=n_boot, seed=seed)
    _print_test_table(decoder_res, label='LOO accuracy')

    # ── TIME-RESOLVED TESTS ───────────────────────────────────────────
    print(f"\n{'─' * 76}")
    print(f"  TIME-RESOLVED TESTS ({n_windows} windows × {2.0/n_windows*1000:.0f} ms)")
    print(f"{'─' * 76}")

    print(f"\n  Centroid distance (omnibus) per window — top {n_pc_centroid} PCs...")
    centroid_tr = run_time_resolved(centroid_features_tr, group_labels,
                                    n_perms=n_perms_tr, n_boot=n_boot_tr,
                                    seed=seed, kind='centroid')
    _print_timeres_table(centroid_tr, label='centroid')

    print(f"\n  LOO decoder (omnibus) per window — {decoder_label}...")
    decoder_tr = run_time_resolved(decoder_features_tr, group_labels,
                                   n_perms=n_perms_tr, n_boot=n_boot_tr,
                                   seed=seed, kind='decoder')
    _print_timeres_table(decoder_tr, label='LOO acc')

    # ── PLOTS ─────────────────────────────────────────────────────────
    centroid_label = f'top {n_pc_centroid} PCs'
    plot_time_averaged(
        centroid_res, decoder_res, region,
        os.path.join(output_dir, f'time_averaged_{region}.png'),
        centroid_label=centroid_label,
        decoder_label=decoder_label)
    plot_time_resolved(
        centroid_tr, decoder_tr, cfg.bin_width, region,
        os.path.join(output_dir, f'time_resolved_{region}.png'),
        centroid_label=centroid_label,
        decoder_label=decoder_label)
    print(f"\n  Figures saved to: {output_dir}")

    # ── SAVE NUMERICS ─────────────────────────────────────────────────
    results = dict(
        region=region,
        identities=identities,
        group_labels=group_labels.tolist(),
        group_map=group_map,
        n_neurons=n_neurons, n_bins=n_bins, n_components=n_components,
        n_pc_centroid=n_pc_centroid, n_pc_decoder=n_pc_decoder,
        decode_on=decode_on,
        n_windows=n_windows,
        mean_center=mean_center, soft_normalize=soft_normalize,
        pca_variance=pca_result['var'].tolist(),
        time_averaged=dict(centroid=centroid_res, decoder=decoder_res),
        time_resolved=dict(centroid=centroid_tr, decoder=decoder_tr),
    )
    pkl_path = os.path.join(output_dir, f'group_separation_tests_v3_{region}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Pickle saved: {pkl_path}")

    return results


def _print_test_table(res, label='value'):
    om = res['omnibus']
    print(f"    OMNIBUS:  obs={om['obs']:.4f}  CI95=[{om['ci'][0]:.4f}, {om['ci'][1]:.4f}]  "
          f"null_mean={om['null'].mean():.4f}  z={om['z']:.2f}  p={om['p']:.4f}")
    print(f"    PAIRWISE (FDR-adjusted):")
    print(f"      {'pair':<28s} {'obs':>8s} {'CI_lo':>8s} {'CI_hi':>8s} "
          f"{'null_mu':>9s} {'z':>6s} {'p':>8s} {'p_adj':>8s}")
    for p in res['pairs']:
        r = res['pairwise'][p]
        print(f"      {_pair_label(p):<28s} {r['obs']:>8.4f} "
              f"{r['ci'][0]:>8.4f} {r['ci'][1]:>8.4f} "
              f"{r['null'].mean():>9.4f} {r['z']:>6.2f} "
              f"{r['p']:>8.4f} {r['p_adj']:>8.4f}")


def _print_timeres_table(res, label='value'):
    n_windows = res['n_windows']
    win_dur = 2.0 / n_windows
    print(f"    {'win':>4s} {'time (s)':>14s} {'obs':>8s} {'CI_lo':>8s} {'CI_hi':>8s} "
          f"{'null_mu':>9s} {'z':>6s} {'p':>8s} {'p_adj':>8s}")
    for w in range(n_windows):
        t0 = w * win_dur
        t1 = (w + 1) * win_dur
        sig = '*' if (not np.isnan(res['p_adj'][w]) and res['p_adj'][w] < 0.05) else ' '
        print(f"    {w:>4d} {f'{t0:.2f}-{t1:.2f}':>14s} "
              f"{res['obs'][w]:>8.4f} {res['ci_lo'][w]:>8.4f} {res['ci_hi'][w]:>8.4f} "
              f"{res['null_mean'][w]:>9.4f} {res['z'][w]:>6.2f} "
              f"{res['p'][w]:>8.4f} {res['p_adj'][w]:>8.4f} {sig}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── User settings ────────────────────────────────────────
    N_PERMS_TA = 1000          # time-averaged permutations
    N_BOOT_TA = 1000           # time-averaged bootstrap
    N_PERMS_TR = 500           # time-resolved permutations (per window)
    N_BOOT_TR = 500            # time-resolved bootstrap (per window)
    N_WINDOWS = 10             # time-resolved granularity (2.0s / 10 = 200 ms)
    MEAN_CENTER = True         # matches trajectory plots
    SOFT_NORMALIZE = True
    N_COMPONENTS = 6           # PCA fit dimensionality (must be ≥ N_PC_DECODER and ≥ N_PC_CENTROID)
    # ── v3 feature-selection knobs ──
    N_PC_CENTROID = 3          # centroid test: top N PCs (default 3 — concentrates signal)
    N_PC_DECODER = 15           # LOO decoder: top N PCs (default 6 — well-posed for n=14-33)
    DECODE_ON = 'pc'           # 'pc' (default, recommended) or 'raw' (v2 behavior, for comparison)
    # ────────────────────────────────
    MIN_REPS = 5
    SEED = 42
    OUTPUT_ROOT = './group_separation_tests_v3'
    REGIONS = ['AMG', 'ER']
    # ─────────────────────────────────────────────────────────

    all_results = {}
    for region in REGIONS:
        out_dir = os.path.join(OUTPUT_ROOT, region)
        all_results[region] = analyze_region(
            region=region,
            n_perms=N_PERMS_TA, n_boot=N_BOOT_TA,
            n_perms_tr=N_PERMS_TR, n_boot_tr=N_BOOT_TR,
            n_windows=N_WINDOWS,
            mean_center=MEAN_CENTER, soft_normalize=SOFT_NORMALIZE,
            n_components=N_COMPONENTS,
            n_pc_centroid=N_PC_CENTROID, n_pc_decoder=N_PC_DECODER,
            decode_on=DECODE_ON,
            min_reps=MIN_REPS,
            seed=SEED, output_dir=out_dir,
        )

    # ── Cross-region summary table (headline numbers only) ──
    print(f"\n\n{'═' * 80}")
    decoder_desc = (f"top {N_PC_DECODER} PCs" if DECODE_ON == 'pc' else 'raw neurons')
    print(f"  CROSS-REGION SUMMARY (centroid: top {N_PC_CENTROID} PCs; "
          f"decoder: {decoder_desc})")
    print(f"{'═' * 80}")
    print(f"  {'region':<6} {'metric':<35} {'obs':>9} {'null_mu':>9} "
          f"{'z':>7} {'p':>8}")
    print('-' * 80)
    for region in REGIONS:
        r = all_results[region]
        for test_name, key, sub in [
            (f'centroid (top {N_PC_CENTROID} PCs)', 'centroid', r['time_averaged']),
            (f'LOO decoder ({decoder_desc})', 'decoder', r['time_averaged']),
        ]:
            om = sub[key]['omnibus']
            print(f"  {region:<6} {test_name:<35} "
                  f"{om['obs']:>9.4f} {om['null'].mean():>9.4f} "
                  f"{om['z']:>7.2f} {om['p']:>8.4f}")
        print()


if __name__ == '__main__':
    main()
