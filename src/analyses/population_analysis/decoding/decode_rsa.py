# decode_rsa.py
"""
Representational Similarity Analysis (RSA):
Compare neural population geometry with social behavioural structure.

Supports standard Mantel and partial Mantel (multiple control RDMs).
Uses shared DecodeConfig and decode_utils.
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from decode_config import DecodeConfig
from decode_utils import (
    load_and_prepare, load_metadata, load_behavior_matrix,
    filter_neurons_by_min_trials, compute_p_value_twotailed,
)

SUBJECT_MONKEY = '81G'


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL RDM
# ═══════════════════════════════════════════════════════════════════════════════

def build_neural_rdm(df, stimulus_monkeys, n_draws=50, seed=42):
    """
    Build neural RDM via pseudo-population subsampling:
      1. Subsample trials per draw → trial-averaged population vector per monkey
      2. Pairwise correlation distance (1 − Pearson r)
      3. Average across draws
    """
    rng = np.random.default_rng(seed)
    neuron_ids = sorted(df['NeuronID'].unique())
    n_neurons = len(neuron_ids)
    n_monkeys = len(stimulus_monkeys)

    neuron_identity_rates = {}
    min_trials = defaultdict(lambda: np.inf)
    for nid in neuron_ids:
        ndf = df[df['NeuronID'] == nid]
        for mid in stimulus_monkeys:
            rates = ndf[ndf['MonkeyName'] == mid]['firing_rate'].values
            neuron_identity_rates[(nid, mid)] = rates
            if len(rates) < min_trials[mid]:
                min_trials[mid] = len(rates)

    print(f"\n  Building neural RDM ({n_draws} draws)")
    print(f"  Neurons: {n_neurons}, Monkeys: {n_monkeys}")
    for mid in stimulus_monkeys:
        print(f"    {mid}: {int(min_trials[mid])} trials (min across neurons)")

    rdm_draws = []
    for _ in range(n_draws):
        pop_vectors = np.zeros((n_monkeys, n_neurons))
        for i, mid in enumerate(stimulus_monkeys):
            nt = int(min_trials[mid])
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=nt, replace=True)
                pop_vectors[i, j] = np.mean(rates[chosen])
        dists = pdist(pop_vectors, metric='correlation')
        rdm_draws.append(squareform(dists))

    mean_rdm = np.mean(rdm_draws, axis=0)
    std_rdm = np.std(rdm_draws, axis=0)
    return mean_rdm, std_rdm, rdm_draws, neuron_ids


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOURAL RDMs
# ═══════════════════════════════════════════════════════════════════════════════

def build_behavioral_rdms(behavior_matrices, stimulus_monkeys, log_transform=True):
    """
    From each behavioural matrix build 3 RDM types:
      1. Symmetrised strength → dissimilarity
      2. Asymmetry |A→B − B→A|
      3. Profile distance (Euclidean)
    """
    rdms = {}
    for label, mat in behavior_matrices.items():
        behavior_type = label.split('_')[0]
        group = '_'.join(label.split('_')[1:])
        available = [m for m in stimulus_monkeys if m in mat.index and m in mat.columns]
        if len(available) < 3:
            print(f"  {label}: only {len(available)} monkeys in matrix, skipping")
            continue

        sub_mat = mat.loc[available, available].values.astype(float)
        if log_transform:
            sub_mat = np.log1p(sub_mat)

        # 1. Strength
        sym = (sub_mat + sub_mat.T) / 2.0
        sym_dissim = np.max(sym) - sym if np.max(sym) > 0 else np.zeros_like(sym)
        np.fill_diagonal(sym_dissim, 0)
        rdms[f"{behavior_type} strength ({group})"] = {
            'rdm': sym_dissim, 'monkeys': available,
            'description': f'Symmetrized {behavior_type.lower()} dissimilarity',
            'type': 'strength',
        }

        # 2. Asymmetry
        asym = np.abs(sub_mat - sub_mat.T)
        np.fill_diagonal(asym, 0)
        rdms[f"{behavior_type} asymmetry ({group})"] = {
            'rdm': asym, 'monkeys': available,
            'description': f'{behavior_type} directional asymmetry',
            'type': 'asymmetry',
        }

        # 3. Profile distance
        rdms[f"{behavior_type} profile ({group})"] = {
            'rdm': squareform(pdist(sub_mat, metric='euclidean')),
            'monkeys': available,
            'description': f'{behavior_type} behavioural profile distance',
            'type': 'profile',
        }
    return rdms


def _build_distance_rdm(metadata_csv, stimulus_monkeys, variable, description, rdm_type):
    """Build |value_i − value_j| RDM for a numeric metadata variable."""
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}
    available, vals = [], []
    for m in stimulus_monkeys:
        v = meta_lookup.get(m, {}).get(variable)
        if v is not None and not np.isnan(float(v)):
            available.append(m)
            vals.append(float(v))
    if len(available) < 3:
        return {}
    vals = np.array(vals)
    return {
        f'{variable} distance': {
            'rdm': np.abs(vals[:, None] - vals[None, :]),
            'monkeys': available,
            'description': description,
            'type': rdm_type,
        }
    }


def build_rank_rdm(metadata_csv, stimulus_monkeys):
    return _build_distance_rdm(metadata_csv, stimulus_monkeys,
                               'Rank', '|rank_i − rank_j|', 'rank')


def build_age_rdm(metadata_csv, stimulus_monkeys):
    return _build_distance_rdm(metadata_csv, stimulus_monkeys,
                               'Age', '|age_i − age_j|', 'age')


def build_sex_rdm(metadata_csv, stimulus_monkeys):
    """0 = same sex, 1 = different sex."""
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}
    available, sexes = [], []
    for m in stimulus_monkeys:
        s = meta_lookup.get(m, {}).get('Sex')
        if s is not None:
            available.append(m)
            sexes.append(str(s).strip().upper())
    if len(available) < 3:
        return {}
    n = len(available)
    dist = np.array([[0.0 if sexes[i] == sexes[j] else 1.0
                       for j in range(n)] for i in range(n)])
    return {
        'Sex distance': {
            'rdm': dist, 'monkeys': available,
            'description': '0 = same sex, 1 = different sex', 'type': 'sex',
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RSA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_upper_triangle(matrix):
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def align_rdms(neural_rdm, neural_monkeys, model_rdm_info):
    model_monkeys = model_rdm_info['monkeys']
    shared = [m for m in neural_monkeys if m in model_monkeys]
    if len(shared) < 3:
        return None, None, shared
    neural_idx = [neural_monkeys.index(m) for m in shared]
    model_idx = [model_monkeys.index(m) for m in shared]
    neural_vec = extract_upper_triangle(neural_rdm[np.ix_(neural_idx, neural_idx)])
    model_vec = extract_upper_triangle(model_rdm_info['rdm'][np.ix_(model_idx, model_idx)])
    return neural_vec, model_vec, shared


def _residualize(x, z):
    """Regress z out of x via OLS.  z: 1-D or 2-D (n_pairs, n_controls)."""
    z = np.atleast_2d(z)
    if z.shape[0] != x.shape[0]:
        z = z.T
    keep = np.std(z, axis=0) > 1e-12
    z = z[:, keep]
    if z.shape[1] == 0:
        return x
    z_c = z - z.mean(axis=0)
    beta = np.linalg.lstsq(z_c, x - x.mean(), rcond=None)[0]
    return x - z_c @ beta


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD MANTEL
# ═══════════════════════════════════════════════════════════════════════════════

def mantel_test(neural_rdm, neural_monkeys, model_rdm_info, n_perms=10000,
                seed=42):
    rng = np.random.default_rng(seed)
    neural_vec, model_vec, shared = align_rdms(neural_rdm, neural_monkeys, model_rdm_info)
    if neural_vec is None or len(shared) < 3:
        return np.nan, np.nan, None, shared
    observed_rho, _ = spearmanr(neural_vec, model_vec)
    n_shared = len(shared)
    neural_idx = [neural_monkeys.index(m) for m in shared]
    perm_rhos = []
    for _ in range(n_perms):
        perm = rng.permutation(n_shared)
        sub = neural_rdm[np.ix_(neural_idx, neural_idx)]
        perm_vec = extract_upper_triangle(sub[np.ix_(perm, perm)])
        rho, _ = spearmanr(perm_vec, model_vec)
        perm_rhos.append(rho)
    perm_rhos = np.array(perm_rhos)
    p_value = compute_p_value_twotailed(observed_rho, perm_rhos)
    return observed_rho, p_value, perm_rhos, shared


def run_rsa(rdm_draws, neural_monkeys, model_rdm_info, n_perms=10000,
            run_perms=True, seed=42):
    draw_rhos = []
    shared = None
    for rdm in rdm_draws:
        nv, mv, shared = align_rdms(rdm, neural_monkeys, model_rdm_info)
        if nv is None:
            continue
        rho, _ = spearmanr(nv, mv)
        draw_rhos.append(rho)
    if not draw_rhos:
        return None
    mean_rho, std_rho = np.mean(draw_rhos), np.std(draw_rhos)
    p_value, perm_rhos = None, None
    if run_perms:
        mean_rdm = np.mean(rdm_draws, axis=0)
        _, p_value, perm_rhos, shared = mantel_test(
            mean_rdm, neural_monkeys, model_rdm_info, n_perms=n_perms, seed=seed)
    return {
        'mean_rho': mean_rho, 'std_rho': std_rho, 'draw_rhos': draw_rhos,
        'p_value': p_value, 'perm_rhos': perm_rhos,
        'n_monkeys': len(shared) if shared else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIAL MANTEL (multiple controls)
# ═══════════════════════════════════════════════════════════════════════════════

def partial_mantel_test(neural_rdm, neural_monkeys, model_rdm_info,
                        control_rdm_infos, n_perms=10000, seed=42):
    rng = np.random.default_rng(seed)
    model_monkeys = model_rdm_info['monkeys']
    shared = [m for m in neural_monkeys if m in model_monkeys]
    for ctrl in control_rdm_infos:
        shared = [m for m in shared if m in ctrl['monkeys']]
    if len(shared) < 4:
        return np.nan, np.nan, None, shared

    neural_idx = [neural_monkeys.index(m) for m in shared]
    model_idx = [model_monkeys.index(m) for m in shared]
    neural_sub = neural_rdm[np.ix_(neural_idx, neural_idx)]
    model_sub = model_rdm_info['rdm'][np.ix_(model_idx, model_idx)]

    neural_vec = extract_upper_triangle(neural_sub)
    model_vec = extract_upper_triangle(model_sub)

    control_vecs = []
    for ctrl in control_rdm_infos:
        ci = [ctrl['monkeys'].index(m) for m in shared]
        control_vecs.append(extract_upper_triangle(ctrl['rdm'][np.ix_(ci, ci)]))
    control_matrix = np.column_stack(control_vecs)

    neural_resid = _residualize(neural_vec, control_matrix)
    model_resid = _residualize(model_vec, control_matrix)
    observed_rho, _ = spearmanr(neural_resid, model_resid)

    n_shared = len(shared)
    perm_rhos = []
    for _ in range(n_perms):
        perm = rng.permutation(n_shared)
        perm_vec = extract_upper_triangle(neural_sub[np.ix_(perm, perm)])
        perm_resid = _residualize(perm_vec, control_matrix)
        rho, _ = spearmanr(perm_resid, model_resid)
        perm_rhos.append(rho)

    perm_rhos = np.array(perm_rhos)
    p_value = compute_p_value_twotailed(observed_rho, perm_rhos)
    return observed_rho, p_value, perm_rhos, shared


def run_rsa_partial(rdm_draws, neural_monkeys, model_rdm_info,
                    control_rdm_infos, n_perms=10000, run_perms=True, seed=42):
    model_monkeys = model_rdm_info['monkeys']
    shared = [m for m in neural_monkeys if m in model_monkeys]
    for ctrl in control_rdm_infos:
        shared = [m for m in shared if m in ctrl['monkeys']]
    if len(shared) < 4:
        return None

    model_idx = [model_monkeys.index(m) for m in shared]
    neural_idx = [neural_monkeys.index(m) for m in shared]
    model_vec = extract_upper_triangle(
        model_rdm_info['rdm'][np.ix_(model_idx, model_idx)])

    control_vecs = []
    for ctrl in control_rdm_infos:
        ci = [ctrl['monkeys'].index(m) for m in shared]
        control_vecs.append(extract_upper_triangle(ctrl['rdm'][np.ix_(ci, ci)]))
    control_matrix = np.column_stack(control_vecs)
    model_resid = _residualize(model_vec, control_matrix)

    draw_rhos = []
    for rdm in rdm_draws:
        nsub = rdm[np.ix_(neural_idx, neural_idx)]
        nvec = extract_upper_triangle(nsub)
        nresid = _residualize(nvec, control_matrix)
        rho, _ = spearmanr(nresid, model_resid)
        draw_rhos.append(rho)

    if not draw_rhos:
        return None

    mean_rho, std_rho = np.mean(draw_rhos), np.std(draw_rhos)
    p_value, perm_rhos = None, None
    if run_perms:
        mean_rdm = np.mean(rdm_draws, axis=0)
        _, p_value, perm_rhos, shared = partial_mantel_test(
            mean_rdm, neural_monkeys, model_rdm_info, control_rdm_infos,
            n_perms=n_perms, seed=seed)

    return {
        'mean_rho': mean_rho, 'std_rho': std_rho, 'draw_rhos': draw_rhos,
        'p_value': p_value, 'perm_rhos': perm_rhos,
        'n_monkeys': len(shared) if shared else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rdms(neural_rdm, neural_monkeys, model_rdms, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Neural RDM
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(neural_rdm, cmap='viridis', interpolation='nearest')
    ax.set_title('Neural RDM (correlation distance)')
    n = len(neural_monkeys)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(neural_monkeys, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(neural_monkeys, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = neural_rdm[i, j]
            color = 'white' if val > np.median(neural_rdm) else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'neural_rdm.png'), dpi=200)
    plt.close()
    print(f"  Saved: neural_rdm.png")

    # Model RDMs grid
    model_names = list(model_rdms.keys())
    n_models = len(model_names)
    if n_models == 0:
        return
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()
    for i, name in enumerate(model_names):
        ax = axes[i]
        info = model_rdms[name]
        im = ax.imshow(info['rdm'], cmap='viridis', interpolation='nearest')
        ax.set_title(name, fontsize=9)
        monkeys = info['monkeys']
        ax.set_xticks(range(len(monkeys)))
        ax.set_yticks(range(len(monkeys)))
        ax.set_xticklabels(monkeys, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(monkeys, fontsize=7)
        plt.colorbar(im, ax=ax)
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_rdms.png'), dpi=200)
    plt.close()
    print(f"  Saved: model_rdms.png")


def plot_rsa_results(all_results, model_rdms, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    valid = {k: v for k, v in all_results.items() if v is not None}
    if not valid:
        return

    names = list(valid.keys())
    rhos = [valid[n]['mean_rho'] for n in names]
    stds = [valid[n]['std_rho'] for n in names]

    color_map = {
        'affiliation': '#55A868', 'submission': '#4C72B0', 'agonism': '#C44E52',
        'rank': '#8172B3', 'age': '#DD8452',
    }
    colors = []
    for n in names:
        nl = n.lower()
        matched = next((c for k, c in color_map.items() if k in nl), '#8C8C8C')
        colors.append(matched)

    # Summary bar
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    ax.bar(x, rhos, yerr=stds, capsize=3, color=colors,
           edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('RSA: Neural–Behavioural Similarity')
    for i, n in enumerate(names):
        r = valid[n]
        if r.get('p_value') is not None:
            p = r['p_value']
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if star:
                y_pos = rhos[i] + stds[i] + 0.02 if rhos[i] >= 0 else rhos[i] - stds[i] - 0.06
                ax.text(i, y_pos, star, ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rsa_summary.png'), dpi=200)
    plt.close()
    print(f"  Saved: rsa_summary.png")

    # Permutation distributions
    for name, r in valid.items():
        if r.get('perm_rhos') is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(r['perm_rhos'], bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(r['mean_rho'], color='red', linewidth=2,
                   label=f"Observed ρ = {r['mean_rho']:.3f}")
        ax.axvline(0, color='blue', linestyle='--', linewidth=1)
        p = r['p_value']
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        ax.set_title(f'{name}\n{p_str}')
        ax.set_xlabel('Spearman ρ')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        fname = f"permutation_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()
    print(f"\nAll figures saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = DecodeConfig(
        exclude_monkeys=['NewMonkey', SUBJECT_MONKEY],
        min_trials=10,
        n_pseudo_draws=50,
        n_pca=None,             # RSA doesn't use PCA/logreg
        skip_perm=True,
        n_permutations=5000,
        response_window=(0.2, 0.5),   # e.g. (0.0, 0.5)
        output_dir='./rsa_output',
    )
    cfg.validate()

    metadata_csv = '/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv'
    root_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/'
    behavior_xlsx = {
        'Affiliation_zombies': root_dir + 'zombies_social_data/zombies_feature_df_affiliation.xlsx',
        'Submission_zombies': root_dir + 'zombies_social_data/zombies_feature_df_submission.xlsx',
        'Agonism_zombies': root_dir + 'zombies_social_data/zombies_feature_df_agonism.xlsx',
        'Affiliation_bestfrans': root_dir + 'bestfrans_social_data/bestfrans_feature_df_affiliation.xlsx',
        'Submission_bestfrans': root_dir + 'bestfrans_social_data/bestfrans_feature_df_submission.xlsx',
        'Agonism_bestfrans': root_dir + 'bestfrans_social_data/bestfrans_feature_df_agonism.xlsx',
        'Affiliation_instigators': root_dir + 'instigators_social_data/instigators_feature_df_affiliation.xlsx',
        'Submission_instigators': root_dir + 'instigators_social_data/instigators_feature_df_submission.xlsx',
        'Agonism_instigators': root_dir + 'instigators_social_data/instigators_feature_df_agonism.xlsx',
    }
    ADULT_AGE_THRESHOLD = 5

    # ── Load neural data ──
    df = load_and_prepare(cfg)

    # ── Load metadata ──
    meta = load_metadata(metadata_csv)
    adult_females = meta[
        (meta['Sex'].str.strip().str.upper() == 'F') &
        (meta['Age'] >= ADULT_AGE_THRESHOLD)
    ]['MonkeyName'].tolist()
    print(f"\nAdult females (age >= {ADULT_AGE_THRESHOLD}): {adult_females}")

    # ── Load behavioural matrices ──
    print(f"\n{'=' * 60}")
    print("LOADING BEHAVIOURAL MATRICES")
    print(f"{'=' * 60}")
    behavior_matrices = {}
    for label, path in behavior_xlsx.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {label}")
            continue
        mat = load_behavior_matrix(path)
        for name in [SUBJECT_MONKEY]:
            if name in mat.index:
                mat = mat.drop(name, axis=0)
            if name in mat.columns:
                mat = mat.drop(name, axis=1)

        is_bestfrans = 'bestfrans' in label.lower()
        if not is_bestfrans:
            keep = [m for m in mat.index if m in adult_females]
            mat = mat.loc[keep, keep]
            print(f"  {label}: {mat.shape}, monkeys (adult F): {list(mat.index)}")
        else:
            print(f"  {label}: {mat.shape}, monkeys (all): {list(mat.index)}")
        behavior_matrices[label] = mat

    # ── Determine stimulus monkeys ──
    all_behavior_monkeys = set()
    for mat in behavior_matrices.values():
        all_behavior_monkeys.update(mat.index)
    neural_monkeys = set(df['MonkeyName'].unique())
    stimulus_monkeys = sorted(all_behavior_monkeys & neural_monkeys)
    print(f"\nStimulus monkeys with behavioural + neural data: {len(stimulus_monkeys)}")

    # ── Filter neurons ──
    df_social = df[df['MonkeyName'].isin(stimulus_monkeys)]
    if cfg.min_trials > 0:
        df_social, _ = filter_neurons_by_min_trials(df_social, stimulus_monkeys, cfg.min_trials)

    # ── Build neural RDM ──
    print(f"\n{'=' * 60}")
    print("BUILDING NEURAL RDM")
    print(f"{'=' * 60}")
    mean_rdm, std_rdm, rdm_draws, neuron_ids = build_neural_rdm(
        df_social, stimulus_monkeys, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed)
    n_neurons = len(neuron_ids)

    # ── Build behavioural RDMs ──
    print(f"\n{'=' * 60}")
    print("BUILDING BEHAVIOURAL RDMs")
    print(f"{'=' * 60}")
    model_rdms = build_behavioral_rdms(behavior_matrices, stimulus_monkeys, log_transform=True)
    model_rdms.update(build_rank_rdm(metadata_csv, stimulus_monkeys))
    model_rdms.update(build_age_rdm(metadata_csv, stimulus_monkeys))

    sex_rdm_info = build_sex_rdm(metadata_csv, stimulus_monkeys).get('Sex distance')
    age_rdm_info = build_age_rdm(metadata_csv, stimulus_monkeys).get('Age distance')

    print(f"\nTotal model RDMs: {len(model_rdms)}")
    for name, info in model_rdms.items():
        print(f"  {name}: {len(info['monkeys'])} monkeys — {info['description']}")

    # ── Run RSA ──
    print(f"\n{'=' * 60}")
    print("RUNNING RSA")
    print(f"{'=' * 60}")
    all_results = {}
    run_perms = not cfg.skip_perm

    for name, info in model_rdms.items():
        is_bestfrans = 'bestfrans' in name.lower()
        is_zombies = 'zombies' in name.lower()
        is_instigators = 'instigators' in name.lower()

        if is_bestfrans and sex_rdm_info is not None:
            controls = [sex_rdm_info]
            ctrl_labels = ['sex']
            if age_rdm_info is not None:
                controls.append(age_rdm_info)
                ctrl_labels.append('age')
            print(f"\n  {name} (partial Mantel, controlling for {' + '.join(ctrl_labels)})...")
            result = run_rsa_partial(rdm_draws, stimulus_monkeys, info, controls,
                                     n_perms=cfg.n_permutations, run_perms=run_perms,
                                     seed=cfg.random_seed)
        elif (is_zombies or is_instigators) and age_rdm_info is not None:
            print(f"\n  {name} (partial Mantel, controlling for age)...")
            result = run_rsa_partial(rdm_draws, stimulus_monkeys, info, [age_rdm_info],
                                     n_perms=cfg.n_permutations, run_perms=run_perms,
                                     seed=cfg.random_seed)
        else:
            print(f"\n  {name}...")
            result = run_rsa(rdm_draws, stimulus_monkeys, info,
                             n_perms=cfg.n_permutations, run_perms=run_perms,
                             seed=cfg.random_seed)
        all_results[name] = result
        if result:
            p_str = f", p={result['p_value']:.4f}" if result['p_value'] is not None else ""
            print(f"    ρ = {result['mean_rho']:.4f} ±{result['std_rho']:.4f}{p_str}")
        else:
            print(f"    FAILED")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {n_neurons}")
    if cfg.response_window:
        print(f"Response window: {cfg.response_window[0]}–{cfg.response_window[1]} s")
    print()

    for rdm_type in ['strength', 'asymmetry', 'profile', 'rank', 'age']:
        typed = {k: v for k, v in all_results.items()
                 if v is not None and model_rdms[k]['type'] == rdm_type}
        if typed:
            print(f"  [{rdm_type.upper()}]")
            for name, r in typed.items():
                p_str = f", p={r['p_value']:.4f}" if r['p_value'] is not None else ""
                print(f"    {name:45s}: ρ={r['mean_rho']:.4f} ±{r['std_rho']:.4f}{p_str}")
            print()

    # ── Save ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    plot_rdms(mean_rdm, stimulus_monkeys, model_rdms, cfg.output_dir)
    plot_rsa_results(all_results, model_rdms, cfg.output_dir)

    save_path = os.path.join(cfg.output_dir, 'rsa_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'config': cfg,
            'neural_rdm_mean': mean_rdm, 'neural_rdm_std': std_rdm,
            'neural_monkeys': stimulus_monkeys, 'n_neurons': n_neurons,
            'model_rdms': model_rdms, 'rsa_results': all_results,
        }, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
