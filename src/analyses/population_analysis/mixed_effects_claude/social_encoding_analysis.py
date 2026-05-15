# social_encoding_analysis.py
"""
Trial-level mixed-effects encoding model for social information in primate MTL.

Tests whether trial-level firing rates are predicted by social-network
features of the viewed monkey.  Compares model fit across groups
(e.g. Zombies = home group vs Instigators = unfamiliar).

Pseudo-population: neurons pooled across sessions; NeuronID treated
as a random effect in the LMM.

Usage:
    python social_encoding_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

from analyses.population_analysis.state_space.data_loading import load_and_filter
from social_encoding_config import SocialEncodingConfig
from analyses.population_analysis.rsa.rsa_social import load_interaction_matrix


# ──────────────────────────────────────────────────────────
# Paths (same structure as run_rsa_social.py)
# ──────────────────────────────────────────────────────────

BEHAVIOR_FILES = {
    'Zombies': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_submission.xlsx',
    },
    'Best Frans': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_submission.xlsx',
    },
    'Instigators': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_submission.xlsx',
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 1. Trial-level firing rate
# ═══════════════════════════════════════════════════════════════════════

def compute_trial_rates(df, window):
    """Compute per-trial firing rate within [epoch_start + win_start, epoch_start + win_end].

    Parameters
    ----------
    df : DataFrame with SpikeTimes, EpochStartStop columns
    window : tuple (start_s, end_s) relative to epoch start

    Returns
    -------
    df : copy with added 'rate' column (spikes/s)
    """
    win_start, win_end = window
    win_dur = win_end - win_start

    rates = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        epoch_start = row['EpochStartStop'][0]
        spk = np.asarray(row['SpikeTimes'])
        if len(spk) > 0:
            spk_rel = spk - epoch_start
            rates[i] = np.sum((spk_rel >= win_start) & (spk_rel < win_end)) / win_dur
        else:
            rates[i] = 0.0

    df = df.copy()
    df['rate'] = rates
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Social features
# ═══════════════════════════════════════════════════════════════════════

def load_social_features(group_name, cfg):
    """Build a feature matrix: one row per monkey, columns = behavioral profile.

    Uses load_interaction_matrix from rsa_social.py for consistency.

    Modes:
      'full_profile'  – concatenate full row from each behavior matrix
      'to_subject'    – only the column for the subject monkey
      'summary'       – row-sum (given) and col-sum (received) per behavior type

    Returns
    -------
    features : DataFrame (index = monkey name, columns = feature names)
    """
    if group_name not in BEHAVIOR_FILES:
        raise FileNotFoundError(f"No behavior files defined for group '{group_name}'")

    matrices = {}
    monkey_ids = None
    for btype in cfg.behavior_types:
        fpath = BEHAVIOR_FILES[group_name].get(btype)
        if fpath is None or not os.path.exists(fpath):
            warnings.warn(f"Missing {btype} file for {group_name}: {fpath}")
            continue
        matrix, mids = load_interaction_matrix(fpath)
        matrices[btype] = (matrix, mids)
        if monkey_ids is None:
            monkey_ids = mids

    if not matrices:
        raise FileNotFoundError(f"No behavior matrices loaded for {group_name}")

    if cfg.feature_mode == 'full_profile':
        pieces = []
        for btype, (matrix, mids) in matrices.items():
            cols = [f"{btype}_to_{mid}" for mid in mids]
            pieces.append(pd.DataFrame(matrix, index=mids, columns=cols))
        features = pd.concat(pieces, axis=1)

    elif cfg.feature_mode == 'to_subject':
        pieces = []
        for btype, (matrix, mids) in matrices.items():
            if cfg.subject_name in mids:
                subj_idx = mids.index(cfg.subject_name)
                col_vals = matrix[:, subj_idx]
                pieces.append(pd.DataFrame(
                    {f"{btype}_to_subject": col_vals}, index=mids))
            else:
                warnings.warn(f"Subject '{cfg.subject_name}' not in {btype} matrix for {group_name}")
        if not pieces:
            raise ValueError(f"Subject '{cfg.subject_name}' not found in any matrix")
        features = pd.concat(pieces, axis=1)

    elif cfg.feature_mode == 'from_subject':
        # Row for 81G: how the subject behaves toward each group member
        pieces = []
        for btype, (matrix, mids) in matrices.items():
            if cfg.subject_name in mids:
                subj_idx = mids.index(cfg.subject_name)
                row_vals = matrix[subj_idx, :]  # subject's row
                pieces.append(pd.DataFrame(
                    {f"{btype}_from_subject": row_vals}, index=mids))
            else:
                warnings.warn(f"Subject '{cfg.subject_name}' not in {btype} matrix for {group_name}")
        if not pieces:
            raise ValueError(f"Subject '{cfg.subject_name}' not found in any matrix")
        features = pd.concat(pieces, axis=1)

    elif cfg.feature_mode == 'summary':
        rows = []
        for mid in monkey_ids:
            row = {}
            for btype, (matrix, mids) in matrices.items():
                idx = mids.index(mid)
                row[f"{btype}_given"] = matrix[idx, :].sum()
                row[f"{btype}_received"] = matrix[:, idx].sum()
            rows.append(row)
        features = pd.DataFrame(rows, index=monkey_ids)

    else:
        raise ValueError(f"Unknown feature_mode: {cfg.feature_mode}")

    features.index.name = 'MonkeyName'
    return features


def reduce_features(features, n_components=2):
    """PCA on social feature matrix.

    Returns
    -------
    pc_df : DataFrame (index = monkey name, columns = social_PC1, ...)
    pca : fitted PCA object
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    pc_cols = [f"social_PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, index=features.index, columns=pc_cols)

    print(f"    PCA: {features.shape[1]} features → {n_components} PCs "
          f"({pca.explained_variance_ratio_.round(3)} var explained)")

    # PCA loadings (features × PCs)
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose: rows = features
        index=features.columns,
        columns=pc_cols
    )

    print("\nPCA loadings:")
    print(loadings.round(3))
    return pc_df, pca, scaler


def plot_social_pca(features_pc, pca, group_name, cfg, save_path=None):
    """Scatter plot of monkeys in social PC space (PC1 vs PC2).

    Each dot = one monkey, labeled with its name.
    """
    if features_pc.shape[1] < 2:
        warnings.warn(f"Need ≥2 PCs for scatter plot, got {features_pc.shape[1]}")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    color = cfg.group_colors.get(group_name, 'gray')

    pc1 = features_pc['social_PC1'].values
    pc2 = features_pc['social_PC2'].values
    names = features_pc.index.values

    ax.scatter(pc1, pc2, c=color, s=100, edgecolors='k', lw=0.8, zorder=5)

    # Label each monkey
    for i, name in enumerate(names):
        ax.annotate(name, (pc1[i], pc2[i]),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=9, fontweight='bold')

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Social PC1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"Social PC2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(f"{group_name}: Social Feature Space\n"
                 f"(mode: {cfg.feature_mode})", fontsize=12)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. Merge trial data with social features
# ═══════════════════════════════════════════════════════════════════════

def build_trial_table(df, features_pc, group_name, cfg):
    """Filter to one group, merge with social-feature PCs.

    Returns
    -------
    trial_df : DataFrame with columns: rate, social_PC1, ..., NeuronID, MonkeyName
    """
    gdf = df[df['MonkeyGroup'] == group_name].copy()

    # Exclude subject from stimulus set
    gdf = gdf[gdf['MonkeyName'] != cfg.subject_name]

    # Only monkeys in the feature matrix
    valid_monkeys = set(features_pc.index)
    gdf = gdf[gdf['MonkeyName'].isin(valid_monkeys)]

    if len(gdf) == 0:
        warnings.warn(f"No trials for group '{group_name}' after filtering")
        return None

    # Require minimum reps per monkey per neuron
    trial_counts = gdf.groupby(['NeuronID', 'MonkeyName']).size()
    valid_pairs = trial_counts[trial_counts >= cfg.min_reps_per_monkey].index
    gdf = gdf.set_index(['NeuronID', 'MonkeyName'])
    gdf = gdf.loc[gdf.index.isin(valid_pairs)].reset_index()

    # Merge social PCs
    gdf = gdf.merge(features_pc, left_on='MonkeyName', right_index=True, how='left')

    n_neurons = gdf['NeuronID'].nunique()
    n_monkeys = gdf['MonkeyName'].nunique()
    print(f"    {group_name}: {len(gdf)} trials, {n_neurons} neurons, {n_monkeys} monkeys")
    return gdf


# ═══════════════════════════════════════════════════════════════════════
# 4. Mixed-effects model
# ═══════════════════════════════════════════════════════════════════════

def fit_mixed_model(trial_df, n_pcs):
    """Fit LMM:  rate ~ social_PC1 + social_PC2 + ... + (1 | NeuronID)

    Social PCs are monkey-level (level-2) predictors.
    NeuronID random intercept absorbs baseline firing-rate differences
    across the pseudo-population.

    Note: we do NOT include (1 | MonkeyName) because the social PCs are
    the monkey-level predictors — a monkey random intercept would absorb
    exactly the variance we are trying to explain.

    Returns
    -------
    result : dict
    """
    pc_cols = [f"social_PC{i+1}" for i in range(n_pcs)]
    formula = "rate ~ " + " + ".join(pc_cols)

    model_full = smf.mixedlm(formula, data=trial_df, groups=trial_df['NeuronID'])
    result_full = model_full.fit(reml=False)

    model_null = smf.mixedlm("rate ~ 1", data=trial_df, groups=trial_df['NeuronID'])
    result_null = model_null.fit(reml=False)

    # Marginal R² approximation (variance reduction in residuals)
    marginal_r2 = 1 - (result_full.scale / result_null.scale)

    return {
        'model_full': result_full,
        'model_null': result_null,
        'marginal_r2': marginal_r2,
        'coefficients': result_full.fe_params,
        'pvalues': result_full.pvalues,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Permutation test
# ═══════════════════════════════════════════════════════════════════════

def permutation_test(trial_df, features_pc, n_pcs, cfg):
    """Shuffle monkey → social-feature mapping and refit.

    Null hypothesis: social features are unrelated to neural responses.
    Permute which feature vector is assigned to which monkey identity,
    keeping trial structure and neuron assignments intact.

    With 9 monkeys there are 362,880 possible permutations.

    Returns
    -------
    dict with observed_r2, null_r2 array, p_value
    """
    rng = np.random.default_rng(cfg.rng_seed)
    pc_cols = [f"social_PC{i+1}" for i in range(n_pcs)]

    obs_result = fit_mixed_model(trial_df, n_pcs)
    obs_r2 = obs_result['marginal_r2']

    monkey_names = features_pc.index.values.copy()
    n_monkeys = len(monkey_names)

    null_r2 = np.zeros(cfg.n_permutations)

    for i in range(cfg.n_permutations):
        shuffled_idx = rng.permutation(n_monkeys)
        shuffled_pcs = features_pc.values[shuffled_idx]
        shuffled_pc_df = pd.DataFrame(
            shuffled_pcs, index=monkey_names, columns=pc_cols)

        perm_df = trial_df.drop(columns=pc_cols).copy()
        perm_df = perm_df.merge(
            shuffled_pc_df, left_on='MonkeyName', right_index=True, how='left')

        try:
            perm_result = fit_mixed_model(perm_df, n_pcs)
            null_r2[i] = perm_result['marginal_r2']
        except Exception:
            null_r2[i] = np.nan

        if (i + 1) % 500 == 0:
            print(f"      permutation {i+1}/{cfg.n_permutations}")

    p_value = np.nanmean(null_r2 >= obs_r2)

    return {
        'observed_r2': obs_r2,
        'null_r2': null_r2,
        'p_value': p_value,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. Per-neuron encoding strength (supplementary)
# ═══════════════════════════════════════════════════════════════════════

def per_neuron_encoding(trial_df, n_pcs):
    """For each neuron: trial-average rate per monkey, then OLS R² against social PCs.

    Returns DataFrame with NeuronID, r2, n_monkeys.
    Interpret with caution — only ~9 data points per neuron.
    """
    pc_cols = [f"social_PC{i+1}" for i in range(n_pcs)]

    avg = trial_df.groupby(['NeuronID', 'MonkeyName']).agg(
        mean_rate=('rate', 'mean'),
        n_trials=('rate', 'count'),
        **{pc: (pc, 'first') for pc in pc_cols}
    ).reset_index()

    results = []
    for nid, ndf in avg.groupby('NeuronID'):
        if len(ndf) < 4:
            continue
        X = ndf[pc_cols].values
        y = ndf['mean_rate'].values

        X_aug = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        y_pred = X_aug @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results.append({'NeuronID': nid, 'r2': r2, 'n_monkeys': len(ndf)})

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# 7. Plotting
# ═══════════════════════════════════════════════════════════════════════

def _p_to_stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:   return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    return 'n.s.'


def plot_r2_comparison(all_results, cfg, save_path=None):
    """Bar chart: marginal R² per group, with permutation p-value stars."""
    groups = list(all_results.keys())
    r2_vals = [all_results[g]['model']['marginal_r2'] for g in groups]
    p_vals = [all_results[g]['permutation']['p_value'] for g in groups]
    colors = [cfg.group_colors.get(g, 'gray') for g in groups]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(range(len(groups)), r2_vals, color=colors, edgecolor='k', alpha=0.8)

    for i, (r2, p) in enumerate(zip(r2_vals, p_vals)):
        star = _p_to_stars(p)
        ax.text(i, r2 + 0.002, star, ha='center', fontsize=12)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Marginal R² (social features)', fontsize=11)
    ax.set_title(f"Social Encoding: {cfg.region}, "
                 f"{cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms",
                 fontsize=12)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_null_distribution(perm_result, group_name, cfg, save_path=None):
    """Histogram of null R² distribution with observed value marked."""
    fig, ax = plt.subplots(figsize=(6, 4))

    null = perm_result['null_r2']
    obs = perm_result['observed_r2']
    p = perm_result['p_value']

    ax.hist(null[~np.isnan(null)], bins=50, color='gray', alpha=0.7,
            edgecolor='k', lw=0.3, label='Null distribution')
    ax.axvline(obs, color=cfg.group_colors.get(group_name, 'red'),
               lw=2, ls='--', label=f'Observed (R²={obs:.4f})')
    ax.set_xlabel('Marginal R²')
    ax.set_ylabel('Count')
    ax.set_title(f"{group_name}: permutation test (p={p:.4f})")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_neuron_r2_distribution(all_results, cfg, save_path=None):
    """Overlaid histograms of per-neuron R² for each group."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for group, res in all_results.items():
        r2 = res['per_neuron']['r2'].values
        color = cfg.group_colors.get(group, 'gray')
        ax.hist(r2, bins=20, color=color, alpha=0.5, edgecolor='k', lw=0.3,
                label=f"{group} (mean={r2.mean():.3f})")

    ax.set_xlabel('Per-neuron R² (OLS, trial-averaged)')
    ax.set_ylabel('Count')
    ax.set_title('Per-neuron social encoding strength (supplementary)')
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8. Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_group(df, group_name, cfg):
    """Run social encoding analysis for one group.

    Returns dict with model, permutation, per_neuron results,
    or None if group cannot be analyzed.
    """
    print(f"\n  --- Loading social features for {group_name} ---")
    try:
        features = load_social_features(group_name, cfg)
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        return None

    # Exclude subject from feature matrix
    if cfg.subject_name in features.index:
        features = features.drop(cfg.subject_name)

    print(f"    Feature matrix: {features.shape[0]} monkeys × {features.shape[1]} features")

    features_pc, pca, scaler = reduce_features(features, n_components=cfg.n_feature_pcs)

    # Build trial table
    trial_df = build_trial_table(df, features_pc, group_name, cfg)
    if trial_df is None or len(trial_df) == 0:
        print(f"  SKIP — no valid trials")
        return None

    # Fit mixed-effects model
    print(f"\n    Fitting mixed-effects model ...")
    model_result = fit_mixed_model(trial_df, cfg.n_feature_pcs)
    print(f"    Marginal R²:  {model_result['marginal_r2']:.4f}")
    print(f"    Coefficients:")
    for name, val in model_result['coefficients'].items():
        pv = model_result['pvalues'].get(name, np.nan)
        print(f"      {name:>15s}: {val:+.6f}  (p={pv:.4e})")

    # Per-neuron encoding (supplementary)
    print(f"\n    Per-neuron encoding (supplementary) ...")
    neuron_r2 = per_neuron_encoding(trial_df, cfg.n_feature_pcs)
    print(f"    Mean per-neuron R²: {neuron_r2['r2'].mean():.4f} "
          f"± {neuron_r2['r2'].std():.4f} (n={len(neuron_r2)} neurons)")

    # Permutation test
    perm_result = None
    if cfg.n_permutations > 0:
        print(f"\n    Running permutation test ({cfg.n_permutations} perms) ...")
        perm_result = permutation_test(
            trial_df, features_pc, cfg.n_feature_pcs, cfg)
        print(f"    Observed R²:      {perm_result['observed_r2']:.4f}")
        print(f"    Permutation p:    {perm_result['p_value']:.4f}")

    return {
        'model': model_result,
        'per_neuron': neuron_r2,
        'permutation': perm_result,
        'features_pc': features_pc,
        'pca': pca,
        'trial_df': trial_df,
    }


def main():
    cfg = SocialEncodingConfig(
        region='ALL',
        session=None,
        window=(0.300, 0.500),
        min_epoch_duration=2.0,
        min_reps_per_monkey=2,
        exclude_groups=['Stranger Things'],
        feature_mode='summary',
        n_feature_pcs=2,
        normalization=None,
        pseudo_population=True,
        n_permutations=500,
        save_plots=True,
        save_dir='social_encoding_results',
        groups=['Zombies', 'Best Frans','Instigators'],
    )
    cfg.validate()

    # ─── Load neural data ───
    print(f"\n{'='*60}")
    print(f"Social Encoding Model (Trial-Level Mixed Effects)")
    print(f"  Region: {cfg.region}")
    print(f"  Window: {cfg.window[0]*1000:.0f}–{cfg.window[1]*1000:.0f} ms")
    print(f"  Feature mode: {cfg.feature_mode}")
    print(f"  PCs: {cfg.n_feature_pcs}")
    print(f"  Normalization: {cfg.normalization}")
    print(f"  Permutations: {cfg.n_permutations}")
    print(f"{'='*60}\n")

    df = load_and_filter(cfg)

    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    # ─── Compute trial-level firing rates ───
    print("\nComputing trial-level firing rates ...")
    df = compute_trial_rates(df, cfg.window)

    # ─── Run for each group ───
    all_results = {}
    for group in cfg.groups:
        print(f"\n{'='*60}")
        print(f"Group: {group}")
        print(f"{'='*60}")
        result = run_group(df, group, cfg)
        if result is not None:
            all_results[group] = result

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for group, res in all_results.items():
        perm_p = res['permutation']['p_value'] if res['permutation'] else 'N/A'
        print(f"\n  {group}:")
        print(f"    Marginal R²:     {res['model']['marginal_r2']:.4f}")
        print(f"    Permutation p:   {perm_p}")
        print(f"    Per-neuron R²:   {res['per_neuron']['r2'].mean():.4f} "
              f"± {res['per_neuron']['r2'].std():.4f}")

    # ─── Plots ───
    if cfg.save_plots and len(all_results) > 0:
        save_base = f"{cfg.save_dir}/{cfg.region}_{cfg.feature_mode}/{cfg.window[0]*1000:.0f}_{cfg.window[1]*1000:.0f}"
        os.makedirs(save_base, exist_ok=True)

        plot_r2_comparison(all_results, cfg,
                           save_path=f"{save_base}/r2_comparison.png")

        for group, res in all_results.items():
            # Social PCA scatter
            plot_social_pca(
                res['features_pc'], res['pca'], group, cfg,
                save_path=f"{save_base}/social_pca_{group}.png")

            if res['permutation'] is not None:
                plot_null_distribution(
                    res['permutation'], group, cfg,
                    save_path=f"{save_base}/null_dist_{group}.png")

        plot_neuron_r2_distribution(all_results, cfg,
                                     save_path=f"{save_base}/neuron_r2_dist.png")

        print(f"\nPlots saved to {save_base}/")

    plt.show()
    return all_results


if __name__ == '__main__':
    main()
