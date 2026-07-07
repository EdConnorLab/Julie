# state_space_group_separation.py
"""
STATE-SPACE group-separation test (categorical).  Belongs in the state-space folder.

This is the population-geometry question: in PCA state space, do the social
groups occupy separable regions? Two complementary statistics, each with a
permutation null and a leakage-free delete-one-identity CI:

  - centroid distance (mean pairwise distance between group centroids)
  - leave-one-identity-out multi-class decoder accuracy

plus a DIMENSIONALITY SWEEP with a max-statistic correction, so the number of
PCs is not a hand-tuned forking path: we report the whole z-vs-#PCs curve and a
single sweep-corrected p-value ("is there ANY dimensionality at which groups
separate, accounting for the search").

CI note: a trial-within-identity bootstrap (resample each identity's trials to
rebuild its mean, identity set fixed) would fold in within-identity trial noise,
but needs trial-level features this pipeline averages away (trial_averaged=True).
The delete-one CI is the correct, runnable choice; `trial_bootstrap_ci` (bottom)
is a ready hook if you later expose trial-level features.
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
# GROUP-POOLING TOGGLE  (defensive 3-box analysis)
# ═══════════════════════════════════════════════════════════════════════
# The original 4-box categorical test includes an Instigators-vs-Stranger Things
# contrast that the subject cannot socially distinguish — neither group is known
# to her. SOCIAL_GROUP pools those two into a single "Strangers" label so the
# categorical test only asks questions the subject's brain can plausibly answer:
# Zombies (home) vs Best Frans (neighbor) vs Strangers (both unknown groups).
#
# main() runs BOTH versions per region by default, so the choice is committed
# up-front rather than picked after seeing which is significant.
SOCIAL_GROUP = {
    'Zombies':         'Zombies',
    'Best Frans':      'Best Frans',
    'Instigators':     'Strangers',
    'Stranger Things': 'Strangers',
}


# ═══════════════════════════════════════════════════════════════════════
# CORE STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def centroid_distances(features, labels):
    """Omnibus = mean of pairwise centroid distances. Returns (omnibus, pairwise dict)."""
    groups = np.unique(labels)
    centroids = {g: features[labels == g].mean(axis=0) for g in groups}
    pairwise = {}
    for g1, g2 in combinations(groups, 2):
        pairwise[(g1, g2)] = float(np.linalg.norm(centroids[g1] - centroids[g2]))
    omnibus = float(np.mean(list(pairwise.values())))
    return omnibus, pairwise


def loo_decode(features, labels, seed=42):
    """Leave-one-identity-out classifier accuracy. Multi-class if >2 labels."""
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
        if len(np.unique(labels[train_idx])) < 2:
            continue
        pipe.fit(features[train_idx], labels[train_idx])
        if pipe.predict(features[i:i + 1])[0] == labels[i]:
            correct += 1
        valid += 1
    return correct / valid if valid > 0 else np.nan


def pairwise_loo_decode(features, labels, g1, g2, seed=42):
    mask = (labels == g1) | (labels == g2)
    return loo_decode(features[mask], labels[mask], seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# PERMUTATION / EFFECT SIZE / FDR
# ═══════════════════════════════════════════════════════════════════════

def null_z(obs, null_dist):
    null_dist = np.asarray(null_dist)
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.std(null_dist) == 0:
        return np.nan
    return float((obs - null_dist.mean()) / null_dist.std())


def p_from_null(obs, null_dist, tail='upper'):
    null_dist = np.asarray(null_dist)
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.isnan(obs):
        return np.nan
    if tail == 'upper':
        return float((null_dist >= obs).sum() / len(null_dist))
    elif tail == 'lower':
        return float((null_dist <= obs).sum() / len(null_dist))
    raise ValueError(tail)


def fdr_bh(p_values):
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
    sorted_idx = order[::-1]
    cummin = np.inf
    for i in sorted_idx:
        cummin = min(cummin, adj[i])
        adj[i] = cummin
    adj = np.clip(adj, 0, 1)
    out[finite] = adj
    return out


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE-FREE DELETE-ONE-IDENTITY CI
# ═══════════════════════════════════════════════════════════════════════

def jackknife_ci(features, labels, statistic_fn, obs, clip=None):
    """
    Delete-one-identity (jackknife) 95% CI around `obs`. Each replicate drops one
    identity and recomputes on the rest — identities intact, no LOO leakage.
    `clip=(a,b)` bounds the CI (use (0,1) for accuracy).
    """
    n = len(features)
    vals = np.full(n, np.nan)
    for i in range(n):
        keep = np.arange(n) != i
        try:
            vals[i] = statistic_fn(features[keep], labels[keep])
        except Exception:
            pass
    fin = vals[~np.isnan(vals)]
    if len(fin) < 2:
        return np.nan, np.nan, np.nan
    m = fin.mean()
    ne = len(fin)
    se = float(np.sqrt((ne - 1) / ne * np.sum((fin - m) ** 2)))
    lo, hi = obs - 1.96 * se, obs + 1.96 * se
    if clip is not None:
        lo, hi = max(clip[0], lo), min(clip[1], hi)
    return float(lo), float(hi), se


# ═══════════════════════════════════════════════════════════════════════
# TIME-AVERAGED CENTROID + DECODER TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_centroid_test(features, labels, n_perms=1000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    pairs = list(combinations(groups, 2))

    obs_omni, obs_pair = centroid_distances(features, labels)

    null_omni = np.zeros(n_perms)
    null_pair = {p: np.zeros(n_perms) for p in pairs}
    for i in range(n_perms):
        shuf = rng.permutation(labels)
        omni_i, pair_i = centroid_distances(features, shuf)
        null_omni[i] = omni_i
        for p in pairs:
            null_pair[p][i] = pair_i.get(p, pair_i.get((p[1], p[0]), np.nan))

    omni_lo, omni_hi, _ = jackknife_ci(
        features, labels, lambda f, l: centroid_distances(f, l)[0], obs_omni)
    pair_ci = {}
    for p in pairs:
        mask = (labels == p[0]) | (labels == p[1])
        lo, hi, _ = jackknife_ci(features[mask], labels[mask],
                                 lambda f, l: centroid_distances(f, l)[0],
                                 obs_pair[p])
        pair_ci[p] = (lo, hi)

    omni_z = null_z(obs_omni, null_omni)
    omni_p = p_from_null(obs_omni, null_omni, tail='upper')

    pair_results, raw_p = {}, []
    for p in pairs:
        z = null_z(obs_pair[p], null_pair[p])
        pv = p_from_null(obs_pair[p], null_pair[p], tail='upper')
        pair_results[p] = dict(obs=obs_pair[p], null=null_pair[p], z=z, p=pv, ci=pair_ci[p])
        raw_p.append(pv)
    for p, padj in zip(pairs, fdr_bh(raw_p)):
        pair_results[p]['p_adj'] = float(padj)

    return dict(omnibus=dict(obs=obs_omni, null=null_omni, z=omni_z, p=omni_p,
                             ci=(omni_lo, omni_hi)),
                pairwise=pair_results, pairs=pairs)


def run_decoder_test(features, labels, n_perms=1000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    pairs = list(combinations(groups, 2))

    obs_omni = loo_decode(features, labels, seed=seed)
    obs_pair = {p: pairwise_loo_decode(features, labels, p[0], p[1], seed=seed) for p in pairs}

    null_omni = np.zeros(n_perms)
    null_pair = {p: np.zeros(n_perms) for p in pairs}
    for i in range(n_perms):
        shuf = rng.permutation(labels)
        null_omni[i] = loo_decode(features, shuf, seed=seed)
        for p in pairs:
            null_pair[p][i] = pairwise_loo_decode(features, shuf, p[0], p[1], seed=seed)

    omni_lo, omni_hi, _ = jackknife_ci(
        features, labels, lambda f, l: loo_decode(f, l, seed=seed), obs_omni, clip=(0, 1))
    pair_ci = {}
    for p in pairs:
        mask = (labels == p[0]) | (labels == p[1])
        lo, hi, _ = jackknife_ci(features[mask], labels[mask],
                                 lambda f, l: loo_decode(f, l, seed=seed),
                                 obs_pair[p], clip=(0, 1))
        pair_ci[p] = (lo, hi)

    omni_z = null_z(obs_omni, null_omni)
    omni_p = p_from_null(obs_omni, null_omni, tail='upper')

    pair_results, raw_p = {}, []
    for p in pairs:
        z = null_z(obs_pair[p], null_pair[p])
        pv = p_from_null(obs_pair[p], null_pair[p], tail='upper')
        pair_results[p] = dict(obs=obs_pair[p], null=null_pair[p], z=z, p=pv, ci=pair_ci[p])
        raw_p.append(pv)
    for p, padj in zip(pairs, fdr_bh(raw_p)):
        pair_results[p]['p_adj'] = float(padj)

    return dict(omnibus=dict(obs=obs_omni, null=null_omni, z=omni_z, p=omni_p,
                             ci=(omni_lo, omni_hi)),
                pairwise=pair_results, pairs=pairs)


# ═══════════════════════════════════════════════════════════════════════
# DIMENSIONALITY SWEEP (+ max-statistic correction over PC counts)
# ═══════════════════════════════════════════════════════════════════════

def run_dim_sweep(pc_means, labels, dims, kind='centroid', n_perms=500, seed=42):
    rng = np.random.default_rng(seed)
    if kind == 'centroid':
        stat = lambda f, l: centroid_distances(f, l)[0]
    elif kind == 'decoder':
        stat = lambda f, l: loo_decode(f, l, seed=seed)
    else:
        raise ValueError(kind)

    dims = list(dims)
    obs = np.array([stat(pc_means[:, :d], labels) for d in dims])

    perms = [rng.permutation(labels) for _ in range(n_perms)]
    null = np.zeros((n_perms, len(dims)))
    for pi, sh in enumerate(perms):
        for di, d in enumerate(dims):
            null[pi, di] = stat(pc_means[:, :d], sh)

    mu = null.mean(0)
    sd = null.std(0)
    sd_safe = np.where(sd == 0, np.nan, sd)
    z = (obs - mu) / sd_safe
    p = np.array([(null[:, di] >= obs[di]).mean() for di in range(len(dims))])

    znull = (null - mu) / sd_safe
    maxz_null = np.nanmax(znull, axis=1)
    obs_maxz = np.nanmax(z)
    p_sweep = float((maxz_null >= obs_maxz).mean())
    best_dim = int(dims[int(np.nanargmax(z))])

    return dict(dims=dims, obs=obs, z=z, p=p,
                p_sweep_corrected=p_sweep, best_dim=best_dim, kind=kind)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def _pair_label(p):
    return f'{str(p[0])[:8]} vs {str(p[1])[:8]}'


def _star(pa):
    if np.isnan(pa):
        return 'n.s.'
    return '***' if pa < 0.001 else ('**' if pa < 0.01 else ('*' if pa < 0.05 else 'n.s.'))


def plot_time_averaged(centroid_res, decoder_res, region, output_path,
                       centroid_label='top 3 PCs', decoder_label='top 6 PCs'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    om = centroid_res['omnibus']
    ax.hist(om['null'], bins=30, color='steelblue', alpha=0.6, edgecolor='black', linewidth=0.4)
    ax.axvline(om['obs'], color='red', lw=2, label=f"obs = {om['obs']:.3f}")
    ax.axvspan(om['ci'][0], om['ci'][1], color='red', alpha=0.15,
               label=f"delete-one CI [{om['ci'][0]:.3f}, {om['ci'][1]:.3f}]")
    p_str = "p < 0.001" if om['p'] < 1e-3 else f"p = {om['p']:.3f}"
    ax.set_title(f"Centroid distance (omnibus, {centroid_label})\nz = {om['z']:.2f},  {p_str}")
    ax.set_xlabel('mean pairwise centroid dist'); ax.set_ylabel('# permutations')
    ax.legend(fontsize=8)

    _pairwise_panel(axes[0, 1], centroid_res, ylabel='centroid distance',
                    title='Centroid distance — pairwise (FDR-adjusted)', is_acc=False)

    ax = axes[1, 0]
    om = decoder_res['omnibus']
    ax.hist(om['null'], bins=30, color='seagreen', alpha=0.6, edgecolor='black', linewidth=0.4)
    ax.axvline(om['obs'], color='red', lw=2, label=f"obs = {om['obs']:.3f}")
    ax.axvspan(om['ci'][0], om['ci'][1], color='red', alpha=0.15,
               label=f"delete-one CI [{om['ci'][0]:.3f}, {om['ci'][1]:.3f}]")
    p_str = "p < 0.001" if om['p'] < 1e-3 else f"p = {om['p']:.3f}"
    n_groups = len({g for pair in decoder_res['pairs'] for g in pair})
    ax.set_title(f"LOO decoder (multi-class, {decoder_label})\nz = {om['z']:.2f},  {p_str}")
    ax.set_xlabel('LOO accuracy'); ax.set_ylabel('# permutations')
    ax.axvline(1.0 / n_groups, color='black', ls=':', lw=1, alpha=0.5, label=f'chance (1/{n_groups})')
    ax.legend(fontsize=8)

    _pairwise_panel(axes[1, 1], decoder_res, ylabel='binary LOO accuracy',
                    title='LOO decoder — pairwise (FDR-adjusted)', is_acc=True)

    fig.suptitle(f'State-space group separation — {region}', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _pairwise_panel(ax, res, ylabel, title, is_acc):
    pairs = res['pairs']
    x = np.arange(len(pairs))
    obs_vals = [res['pairwise'][p]['obs'] for p in pairs]
    null_means = [res['pairwise'][p]['null'].mean() for p in pairs]
    ci_los = [res['pairwise'][p]['ci'][0] for p in pairs]
    ci_his = [res['pairwise'][p]['ci'][1] for p in pairs]
    z_vals = [res['pairwise'][p]['z'] for p in pairs]
    padj = [res['pairwise'][p]['p_adj'] for p in pairs]
    color = 'seagreen' if is_acc else 'steelblue'

    ax.bar(x, obs_vals, color=color, alpha=0.55, edgecolor='black', linewidth=0.5, label='observed')
    ax.vlines(x, ci_los, ci_his, color='black', lw=2, alpha=0.85, label='delete-one CI')
    cap = 0.15
    ax.hlines(ci_los, x - cap, x + cap, color='black', lw=1.5, alpha=0.85)
    ax.hlines(ci_his, x - cap, x + cap, color='black', lw=1.5, alpha=0.85)
    ax.scatter(x, null_means, color='gray', marker='_', s=200, label='null mean', zorder=3)
    if is_acc:
        ax.axhline(0.5, color='black', ls=':', lw=1, alpha=0.5, label='chance (binary)')
        cap_y = 1.05
    else:
        cap_y = max(np.nanmax(ci_his), np.nanmax(obs_vals)) * 1.18
    for i, (z_, pa) in enumerate(zip(z_vals, padj)):
        txt_color = ('darkred' if z_ > 0 else 'darkblue') if pa < 0.05 else ('black' if z_ > 0 else 'steelblue')
        y_top = min((max(ci_his[i], obs_vals[i]) * (1.02 if is_acc else 1.04)) + (0.02 if is_acc else 0), cap_y)
        ax.text(i, y_top, f'z = {z_:+.1f}\n{_star(pa)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=txt_color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='lightgray', alpha=0.9, linewidth=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels([_pair_label(p) for p in pairs], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, 1.18) if is_acc else ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], cap_y * 1.10))
    ax.legend(fontsize=8, loc='lower right')


def plot_dim_sweep(centroid_sweep, decoder_sweep, region, output_path):
    z05 = 1.645
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sw, name, color in [
        (axes[0], centroid_sweep, 'Centroid distance', 'steelblue'),
        (axes[1], decoder_sweep, 'LOO decoder', 'seagreen')]:
        dims = sw['dims']
        ax.plot(dims, sw['z'], color=color, marker='o', lw=2, label='null-z (obs)')
        ax.axhline(z05, color='red', ls='--', lw=1, alpha=0.7, label='z = 1.645 (uncorr. p=0.05)')
        ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.6)
        ax.scatter([sw['best_dim']], [np.nanmax(sw['z'])], color='red', s=80, zorder=5)
        ax.set_xlabel('# PCs included'); ax.set_ylabel('null-z (omnibus)')
        psc = sw['p_sweep_corrected']
        psc_str = "p < 0.001" if psc < 1e-3 else f"p = {psc:.3f}"
        ax.set_title(f"{name} vs dimensionality\nbest dim = {sw['best_dim']}, sweep-corrected {psc_str}")
        ax.legend(fontsize=8)
    fig.suptitle(f'Dimensionality sweep — {region}  '
                 f'(sweep-corrected p accounts for searching over PC count)', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# PER-REGION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def analyze_region(region, dims_sweep, n_pc_centroid_report, n_pc_decoder_report,
                   n_perms=1000, n_perms_sweep=500, mean_center=True, soft_normalize=True,
                   min_reps=5, seed=42, output_dir=None, pool_strangers=False):
    print(f"\n{'═' * 76}")
    print(f"  STATE-SPACE GROUP SEPARATION — region = {region}")
    print(f"{'═' * 76}")

    n_components = max(max(dims_sweep), n_pc_centroid_report, n_pc_decoder_report)

    cfg = TrajectoryConfig(
        region=region, session=None, trial_averaged=True, peak_align=False,
        n_components=n_components, bin_width=0.050, min_epoch_duration=2.0,
        analysis='identity',
    )
    cfg.validate()

    if output_dir is None:
        output_dir = f'./state_space_group_separation_{region}'
    os.makedirs(output_dir, exist_ok=True)

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
                               soft_normalize=soft_normalize, mean_center=mean_center)
    pca_result = run_pca(pca_matrix_pp, info, cfg)

    identities = list(info['conditions'])
    n_bins = info['n_bins']
    n_neurons = pca_matrix_pp.shape[1]
    n_ids = len(identities)

    scores_3d = pca_result['scores_3d']
    pc_mean_full = scores_3d.mean(axis=1)
    raw_group_labels = np.array([group_map[i] for i in identities])

    # apply pooling only to the LABELS used for the categorical test — PCA stays
    # fit on the underlying neural data, untouched. This isolates the question
    # "do these labels separate?" from any effect of pooling on dimensionality.
    if pool_strangers:
        missing = sorted(set(raw_group_labels) - set(SOCIAL_GROUP))
        if missing:
            raise KeyError(f"SOCIAL_GROUP map missing entries for: {missing} "
                           f"— edit the map at the top of the file.")
        group_labels = np.array([SOCIAL_GROUP[g] for g in raw_group_labels])
        label_scheme = '3box_pooled_strangers'
    else:
        group_labels = raw_group_labels
        label_scheme = '4box_original'
    var = pca_result['var']

    centroid_features = pc_mean_full[:, :n_pc_centroid_report]
    decoder_features = pc_mean_full[:, :n_pc_decoder_report]

    print(f"\n  Label scheme: {label_scheme}  "
          f"({'4 groups, original' if not pool_strangers else '3 groups; Instigators+Stranger Things → Strangers'})")
    print(f"  Identities ({n_ids}):")
    for g in sorted(np.unique(group_labels)):
        members = [i for i, l in zip(identities, group_labels) if l == g]
        print(f"    {g:<18s} ({len(members)}): {', '.join(members)}")
    print(f"\n  soft_normalize={soft_normalize}  mean_center={mean_center}  PCA n_comp={n_components}")
    print(f"  report panel: centroid=top {n_pc_centroid_report} PCs, decoder=top {n_pc_decoder_report} PCs (fixed)")
    print(f"  dim sweep: {dims_sweep}")
    print(f"  PCA var: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))
    _, inv = np.unique(group_labels, return_inverse=True)
    grp_sizes = np.bincount(inv)
    n_distinct = factorial(n_ids) // int(np.prod([factorial(int(s)) for s in grp_sizes]))
    print(f"  distinct label permutations: {n_distinct}")
    if n_distinct < n_perms:
        print(f"  ⚠  asked {n_perms} perms, only {n_distinct} distinct — null is discrete.")

    print(f"\n  [1] Centroid test (top {n_pc_centroid_report} PCs)")
    centroid_res = run_centroid_test(centroid_features, group_labels, n_perms=n_perms, seed=seed)
    om = centroid_res['omnibus']
    print(f"      omnibus obs={om['obs']:.4f}  z={om['z']:.2f}  p={om['p']:.4f}  "
          f"CI[{om['ci'][0]:.3f},{om['ci'][1]:.3f}]")

    print(f"  [2] LOO decoder (top {n_pc_decoder_report} PCs)")
    decoder_res = run_decoder_test(decoder_features, group_labels, n_perms=n_perms, seed=seed)
    om = decoder_res['omnibus']
    print(f"      omnibus obs={om['obs']:.4f}  z={om['z']:.2f}  p={om['p']:.4f}  "
          f"CI[{om['ci'][0]:.3f},{om['ci'][1]:.3f}]")

    print(f"\n  [3] Dimensionality sweep (n_perms_sweep={n_perms_sweep})")
    centroid_sweep = run_dim_sweep(pc_mean_full, group_labels, dims_sweep,
                                   kind='centroid', n_perms=n_perms_sweep, seed=seed)
    decoder_sweep = run_dim_sweep(pc_mean_full, group_labels, dims_sweep,
                                  kind='decoder', n_perms=n_perms_sweep, seed=seed)
    print(f"      centroid: best dim={centroid_sweep['best_dim']}, "
          f"sweep-corrected p={centroid_sweep['p_sweep_corrected']:.4f}")
    print(f"      decoder : best dim={decoder_sweep['best_dim']}, "
          f"sweep-corrected p={decoder_sweep['p_sweep_corrected']:.4f}")
    for nm, sw in [('centroid', centroid_sweep), ('decoder', decoder_sweep)]:
        print(f"        {nm}: " + "  ".join(f"d{d}:z={z:+.2f}" for d, z in zip(sw['dims'], sw['z'])))

    plot_time_averaged(centroid_res, decoder_res, region,
                       os.path.join(output_dir, f'time_averaged_{region}_{label_scheme}.png'),
                       centroid_label=f'top {n_pc_centroid_report} PCs',
                       decoder_label=f'top {n_pc_decoder_report} PCs')
    plot_dim_sweep(centroid_sweep, decoder_sweep, region,
                   os.path.join(output_dir, f'dim_sweep_{region}_{label_scheme}.png'))
    print(f"\n  Figures saved to: {output_dir}  (tagged {label_scheme})")

    results = dict(
        region=region, identities=identities,
        label_scheme=label_scheme, pool_strangers=pool_strangers,
        group_labels=group_labels.tolist(),
        raw_group_labels=raw_group_labels.tolist(), group_map=group_map,
        n_neurons=n_neurons, n_bins=n_bins, n_components=n_components,
        n_pc_centroid_report=n_pc_centroid_report, n_pc_decoder_report=n_pc_decoder_report,
        mean_center=mean_center, soft_normalize=soft_normalize,
        pca_variance=var.tolist(),
        time_averaged=dict(centroid=centroid_res, decoder=decoder_res),
        dim_sweep=dict(centroid=centroid_sweep, decoder=decoder_sweep),
    )
    with open(os.path.join(output_dir,
              f'state_space_group_separation_{region}_{label_scheme}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"  Pickle saved ({label_scheme}).")
    return results


# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL: trial-within-identity bootstrap (needs trial-level features)
# ═══════════════════════════════════════════════════════════════════════

def trial_bootstrap_ci(trial_features_by_identity, labels, statistic_fn,
                       obs, n_boot=1000, clip=None, seed=42):
    """Preferred CI if trial-level features are available. Identity SET fixed
    (no LOO leakage); resample each identity's TRIALS with replacement to rebuild
    its mean. Not called by default — wire into run_*_test when trial-level data
    is exposed (cfg.trial_averaged=False path)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        feats = np.vstack([
            trial_features_by_identity[k][
                rng.integers(0, len(trial_features_by_identity[k]),
                             len(trial_features_by_identity[k]))].mean(axis=0)
            for k in range(n)])
        try:
            boot[b] = statistic_fn(feats, labels)
        except Exception:
            pass
    fin = boot[~np.isnan(boot)]
    if len(fin) < 2:
        return np.nan, np.nan
    lo, hi = np.percentile(fin, 2.5), np.percentile(fin, 97.5)
    if clip is not None:
        lo, hi = max(clip[0], lo), min(clip[1], hi)
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    N_PERMS = 1000
    N_PERMS_SWEEP = 500
    DIMS_SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    N_PC_CENTROID_REPORT = 3   # fixed a-priori dimensionality for the report panel
    N_PC_DECODER_REPORT = 6
    MEAN_CENTER = True
    SOFT_NORMALIZE = True
    MIN_REPS = 5
    SEED = 42
    OUTPUT_ROOT = './state_space_group_separation'
    REGIONS = ['AMG', 'ER']

    # Run BOTH label schemes per region. Committing to reporting both up-front
    # closes the door on choosing whichever happens to be significant.
    POOL_OPTIONS = [
        (False, '4box_original'),
        (True,  '3box_pooled_strangers'),
    ]

    all_results = {}
    for region in REGIONS:
        all_results[region] = {}
        for pool_strangers, tag in POOL_OPTIONS:
            all_results[region][tag] = analyze_region(
                region=region, dims_sweep=DIMS_SWEEP,
                n_pc_centroid_report=N_PC_CENTROID_REPORT,
                n_pc_decoder_report=N_PC_DECODER_REPORT,
                n_perms=N_PERMS, n_perms_sweep=N_PERMS_SWEEP,
                mean_center=MEAN_CENTER, soft_normalize=SOFT_NORMALIZE,
                min_reps=MIN_REPS, seed=SEED,
                output_dir=os.path.join(OUTPUT_ROOT, region),
                pool_strangers=pool_strangers)

    print(f"\n\n{'═' * 92}")
    print("  CROSS-REGION SUMMARY (state space) — both label schemes")
    print(f"{'═' * 92}")
    print(f"  {'region':<6} {'scheme':<22} {'test':<28} {'obs':>8} {'z':>7} {'p':>9}")
    print('-' * 92)
    for region in REGIONS:
        for _, tag in POOL_OPTIONS:
            r = all_results[region][tag]
            c = r['time_averaged']['centroid']['omnibus']
            d = r['time_averaged']['decoder']['omnibus']
            cs = r['dim_sweep']['centroid']
            ds = r['dim_sweep']['decoder']
            print(f"  {region:<6} {tag:<22} {'centroid (report dim)':<28} "
                  f"{c['obs']:>8.4f} {c['z']:>7.2f} {c['p']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'decoder (report dim)':<28} "
                  f"{d['obs']:>8.4f} {d['z']:>7.2f} {d['p']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'centroid (sweep-corr.)':<28} "
                  f"{'—':>8} {'—':>7} {cs['p_sweep_corrected']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'decoder (sweep-corr.)':<28} "
                  f"{'—':>8} {'—':>7} {ds['p_sweep_corrected']:>9.4f}")
        print()


if __name__ == '__main__':
    main()
