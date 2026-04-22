"""
Representational Similarity Analysis (RSA):
Compare neural population geometry with social behavioral structure.

Builds a neural RDM (representational dissimilarity matrix) from population
responses to each stimulus monkey, then correlates it with behavioral RDMs
derived from actor-receiver matrices (affiliation, submission, agonism).

Behavioral RDMs are constructed in three ways:
  1. Symmetrized interaction strength: (A->B + B->A) / 2, converted to dissimilarity
  2. Interaction asymmetry: |A->B - B->A|
  3. Behavioral profile distance: Euclidean distance between each monkey's
     row in the behavioral matrix (similar social role = similar profile)

Also includes rank distance and age distance RDMs.

Permutation testing via Mantel test (shuffle identity labels).
"""

import glob
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib

from population_analysis.decoding.decode_pairwise_with_sliding_window import compute_windowed_firing_rates

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
N_PSEUDO_DRAWS = 50
N_PERMUTATIONS = 10000
RANDOM_SEED = 42
SUBJECT_MONKEY = '81G'
CURRENT_YEAR = 2023


# ─── DATA LOADING ────────────────────────────────────────────────────────────

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


def compute_firing_rates(df):
    rates = []
    for _, row in df.iterrows():
        spikes = np.array(row['SpikeTimes'])
        epoch = row['EpochStartStop']
        if isinstance(epoch, str):
            epoch = eval(epoch)
        t_start, t_stop = epoch[0], epoch[1]
        duration = t_stop - t_start
        n_spikes = np.sum((spikes >= t_start) & (spikes <= t_stop))
        rate = n_spikes / duration if duration > 0 else 0.0
        rates.append(rate)
    df = df.copy()
    df['firing_rate'] = rates
    return df


def load_behavior_matrix(xlsx_path):
    """Load a behavior matrix from Excel."""
    df = pd.read_excel(xlsx_path)
    df.set_index(df.columns[0], inplace=True)
    df.index = [str(x).strip() for x in df.index]
    cleaned_cols = []
    for col in df.columns:
        col = str(col).strip()
        if col.startswith('Behavior Towards '):
            col = col.replace('Behavior Towards ', '')
        cleaned_cols.append(col.strip())
    df.columns = cleaned_cols
    return df


def load_metadata(csv_path):
    """Load monkey metadata CSV."""
    meta = pd.read_csv(csv_path)
    meta = meta.rename(columns={'Name': 'MonkeyName', 'Group Name': 'MonkeyGroup'})
    meta['Age'] = meta['Age']
    meta['Rank'] = pd.to_numeric(meta['Rank'], errors='coerce')
    if 'Note' in meta.columns:
        is_subject = meta['Note'].fillna('').str.contains('subject', case=False)
        meta = meta[~is_subject]
    return meta


def filter_neurons_by_min_trials(df, target_identities, min_trials):
    neuron_ids = sorted(df['NeuronID'].unique())
    kept = []
    for nid in neuron_ids:
        neuron_df = df[df['NeuronID'] == nid]
        if all(len(neuron_df[neuron_df['MonkeyName'] == mid]) >= min_trials
               for mid in target_identities):
            kept.append(nid)
    print(f"\n  min_trials filter (>= {min_trials} per identity):")
    print(f"    Kept: {len(kept)} / {len(neuron_ids)} neurons")
    df_filtered = df[df['NeuronID'].isin(kept)]
    return df_filtered, kept


# ─── NEURAL RDM ──────────────────────────────────────────────────────────────

def build_neural_rdm(df, stimulus_monkeys, n_draws=N_PSEUDO_DRAWS, seed=RANDOM_SEED):
    """
    Build neural RDM:
    1. For each draw, subsample trials and compute trial-averaged population vector per monkey
    2. Compute pairwise correlation distance (1 - Pearson r)
    3. Average RDM across draws
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

    for d in range(n_draws):
        pop_vectors = np.zeros((n_monkeys, n_neurons))
        for i, mid in enumerate(stimulus_monkeys):
            nt = int(min_trials[mid])
            for j, nid in enumerate(neuron_ids):
                rates = neuron_identity_rates[(nid, mid)]
                chosen = rng.choice(len(rates), size=nt, replace=True)
                pop_vectors[i, j] = np.mean(rates[chosen])

        dists = pdist(pop_vectors, metric='correlation')
        rdm = squareform(dists)
        rdm_draws.append(rdm)

    mean_rdm = np.mean(rdm_draws, axis=0)
    std_rdm = np.std(rdm_draws, axis=0)

    return mean_rdm, std_rdm, rdm_draws, neuron_ids


# ─── BEHAVIORAL RDMs ─────────────────────────────────────────────────────────

def build_behavioral_rdms(behavior_matrices, stimulus_monkeys, log_transform=True):
    """
    From each behavioral matrix, construct three types of RDMs:
    1. Symmetrized strength -> dissimilarity (max - value)
    2. Asymmetry: |A->B - B->A|
    3. Profile distance: Euclidean distance between row vectors

    If log_transform=True, applies log(x + 1) to compress range before
    building RDMs. Helps with sparse matrices (e.g. agonism) and outliers.
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

        # Log-transform to compress range (handles sparsity + outliers)
        if log_transform:
            sub_mat = np.log1p(sub_mat)  # log(x + 1), safe for zeros

        # 1. Symmetrized interaction strength -> dissimilarity
        sym = (sub_mat + sub_mat.T) / 2.0
        if np.max(sym) > 0:
            sym_dissim = np.max(sym) - sym
        else:
            sym_dissim = np.zeros_like(sym)
        np.fill_diagonal(sym_dissim, 0)

        rdm_name = f"{behavior_type} strength ({group})"
        rdms[rdm_name] = {
            'rdm': sym_dissim,
            'monkeys': available,
            'description': f'Symmetrized {behavior_type.lower()} dissimilarity (max - value)',
            'type': 'strength',
        }

        # 2. Asymmetry: |A->B - B->A|
        asym = np.abs(sub_mat - sub_mat.T)
        np.fill_diagonal(asym, 0)

        rdm_name = f"{behavior_type} asymmetry ({group})"
        rdms[rdm_name] = {
            'rdm': asym,
            'monkeys': available,
            'description': f'{behavior_type} directional asymmetry',
            'type': 'asymmetry',
        }

        # 3. Profile distance
        profile_dists = squareform(pdist(sub_mat, metric='euclidean'))

        rdm_name = f"{behavior_type} profile ({group})"
        rdms[rdm_name] = {
            'rdm': profile_dists,
            'monkeys': available,
            'description': f'{behavior_type} behavioral profile distance',
            'type': 'profile',
        }

    return rdms


def build_rank_rdm(metadata_csv, stimulus_monkeys):
    """Build rank distance RDM: |rank_i - rank_j|."""
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}

    available = []
    ranks = []
    for m in stimulus_monkeys:
        r = meta_lookup.get(m, {}).get('Rank')
        if r is not None and not np.isnan(float(r)):
            available.append(m)
            ranks.append(float(r))

    if len(available) < 3:
        print(f"  Rank RDM: only {len(available)} monkeys with rank, skipping")
        return {}

    ranks = np.array(ranks)
    rank_dist = np.abs(ranks[:, None] - ranks[None, :])

    return {
        'Rank distance': {
            'rdm': rank_dist,
            'monkeys': available,
            'description': '|rank_i - rank_j|',
            'type': 'rank',
        }
    }


def build_age_rdm(metadata_csv, stimulus_monkeys):
    """Build age distance RDM: |age_i - age_j|."""
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}

    available = []
    ages = []
    for m in stimulus_monkeys:
        a = meta_lookup.get(m, {}).get('Age')
        if a is not None and not np.isnan(float(a)):
            available.append(m)
            ages.append(float(a))

    if len(available) < 3:
        return {}

    ages = np.array(ages)
    age_dist = np.abs(ages[:, None] - ages[None, :])

    return {
        'Age distance': {
            'rdm': age_dist,
            'monkeys': available,
            'description': '|age_i - age_j|',
            'type': 'age',
        }
    }


def build_sex_rdm(metadata_csv, stimulus_monkeys):
    """Build sex distance RDM: 0 = same sex, 1 = different sex."""
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}

    available = []
    sexes = []
    for m in stimulus_monkeys:
        s = meta_lookup.get(m, {}).get('Sex')
        if s is not None:
            available.append(m)
            sexes.append(str(s).strip().upper())

    if len(available) < 3:
        print(f"  Sex RDM: only {len(available)} monkeys with sex info, skipping")
        return {}

    n = len(available)
    sex_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sex_dist[i, j] = 0.0 if sexes[i] == sexes[j] else 1.0

    return {
        'Sex distance': {
            'rdm': sex_dist,
            'monkeys': available,
            'description': '0 = same sex, 1 = different sex',
            'type': 'sex',
        }
    }


# ─── PARTIAL MANTEL ──────────────────────────────────────────────────────────

def _residualize(x, z):
    """Regress z out of x, return residuals. z can be a 1-D vector or 2-D (n_pairs, n_controls)."""
    z = np.atleast_2d(z)
    if z.shape[0] == 1 and z.shape[1] != x.shape[0]:
        z = z.T  # ensure shape is (n_pairs, n_controls)
    if z.shape[0] != x.shape[0]:
        z = z.T
    # Drop constant columns (no variance to partial out)
    keep = np.std(z, axis=0) > 1e-12
    z = z[:, keep]
    if z.shape[1] == 0:
        return x  # nothing to partial out
    # OLS: x = z @ beta + residual
    z_centered = z - z.mean(axis=0)
    beta = np.linalg.lstsq(z_centered, x - x.mean(), rcond=None)[0]
    return x - z_centered @ beta


def partial_mantel_test(neural_rdm, neural_monkeys, model_rdm_info, control_rdm_infos,
                        n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Partial Mantel test: Spearman correlation between neural and model RDM
    vectors after regressing out one or more control RDMs (e.g. sex + age).

    control_rdm_infos: list of RDM info dicts to control for.

    Permutations shuffle monkey labels on the neural RDM (same as standard
    Mantel), then residualize before computing permuted rho.
    """
    rng = np.random.default_rng(seed)

    # Find monkeys shared across neural, model, and ALL controls
    model_monkeys = model_rdm_info['monkeys']
    shared = [m for m in neural_monkeys if m in model_monkeys]
    for ctrl in control_rdm_infos:
        shared = [m for m in shared if m in ctrl['monkeys']]

    if len(shared) < 4:
        print(f"    Partial Mantel: only {len(shared)} shared monkeys, skipping")
        return np.nan, np.nan, None, shared

    neural_idx = [neural_monkeys.index(m) for m in shared]
    model_idx = [model_monkeys.index(m) for m in shared]

    neural_sub = neural_rdm[np.ix_(neural_idx, neural_idx)]
    model_sub = model_rdm_info['rdm'][np.ix_(model_idx, model_idx)]

    neural_vec = extract_upper_triangle(neural_sub)
    model_vec = extract_upper_triangle(model_sub)

    # Stack all control vectors into a matrix (n_pairs, n_controls)
    control_vecs = []
    for ctrl in control_rdm_infos:
        ctrl_idx = [ctrl['monkeys'].index(m) for m in shared]
        ctrl_sub = ctrl['rdm'][np.ix_(ctrl_idx, ctrl_idx)]
        control_vecs.append(extract_upper_triangle(ctrl_sub))
    control_matrix = np.column_stack(control_vecs)

    # Residualize both neural and model vectors w.r.t. all controls
    neural_resid = _residualize(neural_vec, control_matrix)
    model_resid = _residualize(model_vec, control_matrix)

    observed_rho, _ = spearmanr(neural_resid, model_resid)

    # Permutation test
    n_shared = len(shared)
    perm_rhos = []
    for _ in range(n_perms):
        perm_order = rng.permutation(n_shared)
        neural_perm = neural_sub[np.ix_(perm_order, perm_order)]
        neural_perm_vec = extract_upper_triangle(neural_perm)
        neural_perm_resid = _residualize(neural_perm_vec, control_matrix)
        rho, _ = spearmanr(neural_perm_resid, model_resid)
        perm_rhos.append(rho)

    perm_rhos = np.array(perm_rhos)
    p_value = (np.sum(np.abs(perm_rhos) >= np.abs(observed_rho)) + 1) / (n_perms + 1)

    return observed_rho, p_value, perm_rhos, shared


def run_rsa_partial(rdm_draws, neural_monkeys, model_rdm_info, control_rdm_infos,
                    n_perms=N_PERMUTATIONS, run_perms=True, seed=RANDOM_SEED):
    """Run partial RSA (controlling for one or more confound RDMs) across neural RDM draws.

    control_rdm_infos: list of RDM info dicts to control for.
    """
    # Find monkeys shared across neural, model, and ALL controls
    model_monkeys = model_rdm_info['monkeys']
    shared = [m for m in neural_monkeys if m in model_monkeys]
    for ctrl in control_rdm_infos:
        shared = [m for m in shared if m in ctrl['monkeys']]

    if len(shared) < 4:
        return None

    model_idx = [model_monkeys.index(m) for m in shared]
    neural_idx = [neural_monkeys.index(m) for m in shared]

    model_sub = model_rdm_info['rdm'][np.ix_(model_idx, model_idx)]
    model_vec = extract_upper_triangle(model_sub)

    # Stack all control vectors
    control_vecs = []
    for ctrl in control_rdm_infos:
        ctrl_idx = [ctrl['monkeys'].index(m) for m in shared]
        ctrl_sub = ctrl['rdm'][np.ix_(ctrl_idx, ctrl_idx)]
        control_vecs.append(extract_upper_triangle(ctrl_sub))
    control_matrix = np.column_stack(control_vecs)

    model_resid = _residualize(model_vec, control_matrix)

    draw_rhos = []
    for rdm in rdm_draws:
        neural_sub = rdm[np.ix_(neural_idx, neural_idx)]
        neural_vec = extract_upper_triangle(neural_sub)
        neural_resid = _residualize(neural_vec, control_matrix)
        rho, _ = spearmanr(neural_resid, model_resid)
        draw_rhos.append(rho)

    if not draw_rhos:
        return None

    mean_rho = np.mean(draw_rhos)
    std_rho = np.std(draw_rhos)

    # Permutation test on mean RDM
    mean_rdm = np.mean(rdm_draws, axis=0)
    p_value = None
    perm_rhos = None

    if run_perms:
        _, p_value, perm_rhos, shared = partial_mantel_test(
            mean_rdm, neural_monkeys, model_rdm_info, control_rdm_infos,
            n_perms=n_perms, seed=seed
        )

    n_shared = len(shared) if shared is not None else 0

    return {
        'mean_rho': mean_rho,
        'std_rho': std_rho,
        'draw_rhos': draw_rhos,
        'p_value': p_value,
        'perm_rhos': perm_rhos,
        'n_monkeys': n_shared,
    }


# ─── RSA CORRELATION ─────────────────────────────────────────────────────────

def extract_upper_triangle(matrix):
    """Extract upper triangle (excluding diagonal) as flat vector."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def align_rdms(neural_rdm, neural_monkeys, model_rdm_info):
    """Align neural and model RDMs to shared monkeys."""
    model_monkeys = model_rdm_info['monkeys']
    model_rdm = model_rdm_info['rdm']

    shared = [m for m in neural_monkeys if m in model_monkeys]
    if len(shared) < 3:
        return None, None, shared

    neural_idx = [neural_monkeys.index(m) for m in shared]
    model_idx = [model_monkeys.index(m) for m in shared]

    neural_sub = neural_rdm[np.ix_(neural_idx, neural_idx)]
    model_sub = model_rdm[np.ix_(model_idx, model_idx)]

    neural_vec = extract_upper_triangle(neural_sub)
    model_vec = extract_upper_triangle(model_sub)

    return neural_vec, model_vec, shared


def mantel_test(neural_rdm, neural_monkeys, model_rdm_info, n_perms=N_PERMUTATIONS,
                seed=RANDOM_SEED):
    """
    Mantel test: Spearman correlation between upper triangles of neural and model RDMs.
    Permutations shuffle monkey labels (rows + columns).
    Two-tailed p-value.
    """
    rng = np.random.default_rng(seed)

    neural_vec, model_vec, shared = align_rdms(neural_rdm, neural_monkeys, model_rdm_info)

    if neural_vec is None or len(shared) < 3:
        return np.nan, np.nan, None, shared

    observed_rho, _ = spearmanr(neural_vec, model_vec)

    n_shared = len(shared)
    neural_idx = [neural_monkeys.index(m) for m in shared]

    perm_rhos = []
    for i in range(n_perms):
        perm_order = rng.permutation(n_shared)

        neural_sub = neural_rdm[np.ix_(neural_idx, neural_idx)]
        neural_perm = neural_sub[np.ix_(perm_order, perm_order)]
        neural_perm_vec = extract_upper_triangle(neural_perm)

        rho, _ = spearmanr(neural_perm_vec, model_vec)
        perm_rhos.append(rho)

    perm_rhos = np.array(perm_rhos)

    # Two-tailed
    p_value = (np.sum(np.abs(perm_rhos) >= np.abs(observed_rho)) + 1) / (n_perms + 1)

    return observed_rho, p_value, perm_rhos, shared


# ─── RSA ACROSS DRAWS ────────────────────────────────────────────────────────

def run_rsa(rdm_draws, neural_monkeys, model_rdm_info, n_perms=N_PERMUTATIONS,
            run_perms=True, seed=RANDOM_SEED):
    """Run RSA across multiple neural RDM draws."""
    draw_rhos = []
    shared = None
    for rdm in rdm_draws:
        neural_vec, model_vec, shared = align_rdms(rdm, neural_monkeys, model_rdm_info)
        if neural_vec is None:
            continue
        rho, _ = spearmanr(neural_vec, model_vec)
        draw_rhos.append(rho)

    if not draw_rhos:
        return None

    mean_rho = np.mean(draw_rhos)
    std_rho = np.std(draw_rhos)

    # Permutation test on mean RDM
    mean_rdm = np.mean(rdm_draws, axis=0)
    p_value = None
    perm_rhos = None

    if run_perms:
        _, p_value, perm_rhos, shared = mantel_test(
            mean_rdm, neural_monkeys, model_rdm_info,
            n_perms=n_perms, seed=seed
        )

    n_shared = len(shared) if shared is not None else 0

    return {
        'mean_rho': mean_rho,
        'std_rho': std_rho,
        'draw_rhos': draw_rhos,
        'p_value': p_value,
        'perm_rhos': perm_rhos,
        'n_monkeys': n_shared,
    }


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_rdms(neural_rdm, neural_monkeys, model_rdms, output_dir):
    """Plot neural RDM and all model RDMs."""
    os.makedirs(output_dir, exist_ok=True)

    # Neural RDM
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(neural_rdm, cmap='viridis', interpolation='nearest')
    ax.set_title('Neural RDM (correlation distance)')
    ax.set_xticks(range(len(neural_monkeys)))
    ax.set_yticks(range(len(neural_monkeys)))
    ax.set_xticklabels(neural_monkeys, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(neural_monkeys, fontsize=9)

    n = len(neural_monkeys)
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
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, name in enumerate(model_names):
        ax = axes[i]
        info = model_rdms[name]
        rdm = info['rdm']
        monkeys = info['monkeys']

        im = ax.imshow(rdm, cmap='viridis', interpolation='nearest')
        ax.set_title(name, fontsize=9)
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
    """Plot RSA summary and permutation distributions."""
    os.makedirs(output_dir, exist_ok=True)

    valid = {k: v for k, v in all_results.items() if v is not None}
    if not valid:
        print("  No valid results to plot")
        return

    names = list(valid.keys())
    rhos = [valid[n]['mean_rho'] for n in names]
    stds = [valid[n]['std_rho'] for n in names]

    colors = []
    for n in names:
        nl = n.lower()
        if 'affiliation' in nl:
            colors.append('#55A868')
        elif 'submission' in nl:
            colors.append('#4C72B0')
        elif 'agonism' in nl:
            colors.append('#C44E52')
        elif 'rank' in nl:
            colors.append('#8172B3')
        elif 'age' in nl:
            colors.append('#DD8452')
        else:
            colors.append('#8C8C8C')

    # ── 1. Summary bar plot ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, rhos, yerr=stds, capsize=3, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman rho (neural RDM vs model RDM)')
    ax.set_title('RSA: Neural-Behavioral Similarity')

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

    # ── 2. Permutation distributions ──
    for name, r in valid.items():
        if r.get('perm_rhos') is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(r['perm_rhos'], bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(r['mean_rho'], color='red', linewidth=2,
                   label=f"Observed rho = {r['mean_rho']:.3f}")
        ax.axvline(0, color='blue', linestyle='--', linewidth=1)

        p = r['p_value']
        p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        ax.set_title(f'{name}\n{p_str}')
        ax.set_xlabel('Spearman rho')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()

        fname = f"permutation_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()

    print(f"\nAll figures saved to {output_dir}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ─── HARDCODED CONFIG (for PyCharm) ──────────────────────────────────────
    pkl_dir = r"/sorted_spike_cache_filtered"
    metadata_csv = r"/home/connorlab/Downloads/monkeyinfo.csv"
    output_dir = r"/home/connorlab/Documents/GitHub/Julie/Cortana/decoding_rsa_output"
    skip_perm = True
    n_perm = 5000
    n_draws = 50
    min_trials = 10

    root_dir = "/social_data/"
    behavior_xlsx = {
        'Affiliation_zombies': root_dir + "zombies_social_data/zombies_feature_df_affiliation.xlsx",
        'Submission_zombies': root_dir + "zombies_social_data/zombies_feature_df_submission.xlsx",
        'Agonism_zombies': root_dir + "zombies_social_data/zombies_feature_df_agonism.xlsx",
        'Affiliation_bestfrans': root_dir + "bestfrans_social_data/bestfrans_feature_df_affiliation.xlsx",
        'Submission_bestfrans': root_dir + "bestfrans_social_data/bestfrans_feature_df_submission.xlsx",
        'Agonism_bestfrans': root_dir + "bestfrans_social_data/bestfrans_feature_df_agonism.xlsx",
        'Affiliation_instigators': root_dir + "instigators_social_data/instigators_feature_df_affiliation.xlsx",
        'Submission_instigators': root_dir + "instigators_social_data/instigators_feature_df_submission.xlsx",
        'Agonism_instigators': root_dir + "instigators_social_data/instigators_feature_df_agonism.xlsx",
    }
    ADULT_AGE_THRESHOLD = 5  # monkeys aged >= 5 are adults
    # ─────────────────────────────────────────────────────────────────────────

    # ── Load neural data ──
    print("=" * 60)
    print("LOADING NEURAL DATA")
    print("=" * 60)
    df = load_all_trials(pkl_dir)
    df = df[df['MonkeyName'] != 'NewMonkey']
    df = df[df['MonkeyName'] != SUBJECT_MONKEY]
    print(f"After exclusions: {len(df)} trials")

    # print("\nComputing trial-level firing rates (full_dominance_first epoch)...")
    # df = compute_firing_rates(df)
    window = (0, 500)
    df = compute_windowed_firing_rates(df, 100, 300)
    print(f'compute windowed firing rates for window {window}')
    df['firing_rate'] = df['SpikeRate']

    # ── Load metadata for demographic filtering ──
    meta = load_metadata(metadata_csv)
    adult_females = meta[
        (meta['Sex'].str.strip().str.upper() == 'F') &
        (meta['Age'] >= ADULT_AGE_THRESHOLD)
    ]['MonkeyName'].tolist()
    print(f"\nAdult females (age >= {ADULT_AGE_THRESHOLD}): {adult_females}")

    # ── Load behavioral matrices ──
    print(f"\n{'=' * 60}")
    print("LOADING BEHAVIORAL MATRICES")
    print(f"{'=' * 60}")

    behavior_matrices = {}
    for label, path in behavior_xlsx.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {label}")
            continue
        mat = load_behavior_matrix(path)
        # Remove subject monkey from matrix
        if SUBJECT_MONKEY in mat.index:
            mat = mat.drop(SUBJECT_MONKEY, axis=0)
        if SUBJECT_MONKEY in mat.columns:
            mat = mat.drop(SUBJECT_MONKEY, axis=1)

        # Filter to adult females for zombies and instigators
        # Keep all monkeys for bestfrans (handled via partial Mantel instead)
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
    print(f"\nStimulus monkeys with behavioral + neural data: {len(stimulus_monkeys)}")
    print(f"  {', '.join(stimulus_monkeys)}")

    # ── Filter neurons ──
    df_social = df[df['MonkeyName'].isin(stimulus_monkeys)]
    if min_trials > 0:
        df_social, _ = filter_neurons_by_min_trials(df_social, stimulus_monkeys, min_trials)

    # ── Build neural RDM ──
    print(f"\n{'=' * 60}")
    print("BUILDING NEURAL RDM")
    print(f"{'=' * 60}")
    mean_rdm, std_rdm, rdm_draws, neuron_ids = build_neural_rdm(
        df_social, stimulus_monkeys, n_draws=n_draws
    )
    n_neurons = len(neuron_ids)
    print(f"  Neural RDM shape: {mean_rdm.shape}")

    # ── Build behavioral RDMs ──
    print(f"\n{'=' * 60}")
    print("BUILDING BEHAVIORAL RDMs")
    print(f"{'=' * 60}")
    model_rdms = build_behavioral_rdms(behavior_matrices, stimulus_monkeys, log_transform=True)

    # Add rank and age RDMs
    rank_rdm = build_rank_rdm(metadata_csv, stimulus_monkeys)
    model_rdms.update(rank_rdm)

    age_rdm = build_age_rdm(metadata_csv, stimulus_monkeys)
    model_rdms.update(age_rdm)

    # Build sex RDM (used as control for partial Mantel on bestfrans)
    sex_rdm_dict = build_sex_rdm(metadata_csv, stimulus_monkeys)
    sex_rdm_info = sex_rdm_dict.get('Sex distance', None)
    if sex_rdm_info is not None:
        print(f"  Sex RDM built: {len(sex_rdm_info['monkeys'])} monkeys")
    else:
        print("  WARNING: Could not build sex RDM — partial Mantel will be skipped for bestfrans")

    # Build age RDM (used as control for partial Mantel on zombies/instigators)
    age_rdm_dict = build_age_rdm(metadata_csv, stimulus_monkeys)
    age_rdm_info = age_rdm_dict.get('Age distance', None)
    if age_rdm_info is not None:
        print(f"  Age RDM built (for partial control): {len(age_rdm_info['monkeys'])} monkeys")

    print(f"\nTotal model RDMs: {len(model_rdms)}")
    for name, info in model_rdms.items():
        print(f"  {name}: {len(info['monkeys'])} monkeys -- {info['description']}")

    # ── Run RSA ──
    print(f"\n{'=' * 60}")
    print("RUNNING RSA")
    print(f"{'=' * 60}")

    all_results = {}

    for name, info in model_rdms.items():
        is_bestfrans = 'bestfrans' in name.lower()
        is_zombies = 'zombies' in name.lower()
        is_instigators = 'instigators' in name.lower()

        if is_bestfrans and sex_rdm_info is not None:
            # Partial Mantel: control for sex + age in bestfrans (uses all monkeys)
            controls = [sex_rdm_info]
            ctrl_labels = ['sex']
            if age_rdm_info is not None:
                controls.append(age_rdm_info)
                ctrl_labels.append('age')
            print(f"\n  {name} (partial Mantel, controlling for {' + '.join(ctrl_labels)})...")
            result = run_rsa_partial(
                rdm_draws, stimulus_monkeys, info, controls,
                n_perms=n_perm, run_perms=(not skip_perm), seed=RANDOM_SEED
            )
        elif (is_zombies or is_instigators) and age_rdm_info is not None:
            # Partial Mantel: control for age in zombies/instigators (adult females only)
            print(f"\n  {name} (partial Mantel, controlling for age)...")
            result = run_rsa_partial(
                rdm_draws, stimulus_monkeys, info, [age_rdm_info],
                n_perms=n_perm, run_perms=(not skip_perm), seed=RANDOM_SEED
            )
        else:
            # Fallback: standard Mantel (rank, age, or if control RDM unavailable)
            print(f"\n  {name}...")
            result = run_rsa(
                rdm_draws, stimulus_monkeys, info,
                n_perms=n_perm, run_perms=(not skip_perm), seed=RANDOM_SEED
            )
        all_results[name] = result

        if result is not None:
            p_str = f", p={result['p_value']:.4f}" if result['p_value'] is not None else ""
            print(f"    rho = {result['mean_rho']:.4f} +/-{result['std_rho']:.4f}{p_str}")
        else:
            print(f"    FAILED (insufficient shared monkeys)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {n_neurons}")
    print(f"Stimulus monkeys: {len(stimulus_monkeys)}")
    if min_trials > 0:
        print(f"  (filtered: min_trials >= {min_trials})")
    print(f"Neural RDM draws: {n_draws}")
    print()

    for rdm_type in ['strength', 'asymmetry', 'profile', 'rank', 'age']:
        type_results = {k: v for k, v in all_results.items()
                        if v is not None and model_rdms[k]['type'] == rdm_type}
        if type_results:
            print(f"  [{rdm_type.upper()}]")
            for name, r in type_results.items():
                p_str = f", p={r['p_value']:.4f}" if r['p_value'] is not None else ""
                partial = ""
                if 'bestfrans' in name.lower() and sex_rdm_info is not None:
                    partial = " [partial, sex+age-controlled]"
                elif ('zombies' in name.lower() or 'instigators' in name.lower()) and age_rdm_info is not None:
                    partial = " [partial, age-controlled]"
                print(f"    {name:45s}: rho={r['mean_rho']:.4f} +/-{r['std_rho']:.4f}{p_str}{partial}")
            print()

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    plot_rdms(mean_rdm, stimulus_monkeys, model_rdms, output_dir)
    plot_rsa_results(all_results, model_rdms, output_dir)

    save_data = {
        'neural_rdm_mean': mean_rdm,
        'neural_rdm_std': std_rdm,
        'neural_monkeys': stimulus_monkeys,
        'n_neurons': n_neurons,
        'model_rdms': model_rdms,
        'rsa_results': all_results,
    }
    save_path = os.path.join(output_dir, 'rsa_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
