# rsa_core.py
"""
Core RSA functions:
  - compute_firing_rates:  spike times → (n_identities, n_neurons) mean rate matrix
  - build_neural_rdm:      response matrix → neural RDM
  - build_model_rdms:      monkey info → dict of model RDMs
  - compare_rdms:          neural RDM × model RDM → Spearman rho (+ optional permutation p)
  - run_rsa_session:       full per-session pipeline
  - run_rsa_pseudopop:     pseudo-population pipeline (pool neurons across sessions)
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from itertools import combinations


# ──────────────────────────────────────────────────────────
# 0. Normalization
# ──────────────────────────────────────────────────────────

def normalize_rates(rate_matrix, method='none', soft_const=5.0):
    """
    Normalize each neuron (column) of the rate matrix.

    Parameters
    ----------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
    method : str
        'none'   → no normalization (return as-is)
        'soft'   → divide by (range + const); conservative equalization
        'zscore' → subtract mean, divide by std; full equalization
    soft_const : float
        Additive constant for soft normalization (spikes/s).

    Returns
    -------
    normalized : ndarray, same shape
    """
    if method == 'none':
        return rate_matrix

    if method == 'soft':
        rng = rate_matrix.max(axis=0) - rate_matrix.min(axis=0) # finding range for each neuron
        denom = rng + soft_const
        normalized = rate_matrix / denom
        n_flat = np.sum(rng < 1e-6)
        if n_flat > 0:
            print(f"  Soft-norm: {n_flat}/{rate_matrix.shape[1]} neurons had near-zero range")
        return normalized

    if method == 'zscore':
        mu = rate_matrix.mean(axis=0)
        sd = rate_matrix.std(axis=0)
        # Neurons with zero std (no modulation) → leave as zero
        n_flat = np.sum(sd < 1e-10)
        if n_flat > 0:
            print(f"  Z-score: {n_flat}/{rate_matrix.shape[1]} neurons had near-zero std (set to 0)")
        sd[sd < 1e-10] = 1.0  # avoid div-by-zero; numerator will be ~0 anyway
        normalized = (rate_matrix - mu) / sd
        return normalized

    raise ValueError(f"Unknown normalization method: {method}")


# ──────────────────────────────────────────────────────────
# 1. Firing-rate computation
# ──────────────────────────────────────────────────────────

def compute_firing_rates(df, identities, window, min_reps):
    """
    Compute trial-averaged mean firing rate per neuron per stimulus identity
    within a time window.

    Parameters
    ----------
    df : DataFrame
        Filtered to a single session. Must have columns:
        MonkeyName, NeuronID, SpikeTimes, EpochStartStop, TaskField
    identities : list of str
        Ordered stimulus identities to include.
    window : tuple (start_s, end_s)
        Time window relative to epoch start.
    min_reps : int
        Minimum repetitions per identity; identities with fewer are dropped.

    Returns
    -------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
        Trial-averaged firing rate (spikes/s).
    valid_identities : list of str
        Identities that survived the min_reps filter (same order as rows).
    neuron_ids : list
        Neuron IDs corresponding to columns.
    """
    win_start, win_end = window
    win_dur = win_end - win_start

    neuron_ids = sorted(df['NeuronID'].unique())
    neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    # Get trial-level metadata (one row per trial)
    trial_meta = (df.groupby('TaskField')
                    .first()[['MonkeyName']]
                    .reset_index())

    valid_identities = []
    rate_rows = []

    for monkey in identities:
        trials = trial_meta[trial_meta['MonkeyName'] == monkey]['TaskField'].values
        if len(trials) < min_reps:
            continue

        # Accumulate spike counts across trials for this identity
        sum_counts = np.zeros(n_neurons, dtype=float)
        n_trials = 0

        for tf in trials:
            trial_rows = df[df['TaskField'] == tf]
            for _, row in trial_rows.iterrows():
                n_idx = neuron_to_idx[row['NeuronID']]
                epoch_start = row['EpochStartStop'][0]
                spk = row['SpikeTimes'] - epoch_start
                count = np.sum((spk >= win_start) & (spk < win_end))
                sum_counts[n_idx] += count
            n_trials += 1

        avg_rate = (sum_counts / n_trials) / win_dur  # spikes per second
        rate_rows.append(avg_rate)
        valid_identities.append(monkey)

    rate_matrix = np.array(rate_rows)  # (n_identities, n_neurons)
    return rate_matrix, valid_identities, neuron_ids


# ──────────────────────────────────────────────────────────
# 2. Neural RDM
# ──────────────────────────────────────────────────────────

def build_neural_rdm(rate_matrix, metric='correlation'):
    """
    Build neural RDM from response matrix.

    Parameters
    ----------
    rate_matrix : ndarray, shape (n_conditions, n_neurons)
    metric : str
        'correlation' → 1 - Pearson r
        'euclidean'   → Euclidean distance

    Returns
    -------
    rdm : ndarray, shape (n_conditions, n_conditions)
        Symmetric dissimilarity matrix (diagonal = 0).
    """
    if metric == 'correlation':
        dists = pdist(rate_matrix, metric='correlation')
    elif metric == 'euclidean':
        dists = pdist(rate_matrix, metric='euclidean')
    elif metric == 'cosine':
        dists = pdist(rate_matrix, metric='cosine')

    elif metric == 'mahalanobis':
        # Covariance across neurons
        cov = np.cov(rate_matrix, rowvar=False)
        # Small regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        # Inverse covariance matrix
        VI = np.linalg.inv(cov)

        dists = pdist(rate_matrix, metric='mahalanobis', VI=VI)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    rdm = squareform(dists)
    return rdm


# ──────────────────────────────────────────────────────────
# 3. Model RDMs
# ──────────────────────────────────────────────────────────

def _categorical_rdm(labels):
    """Binary same(0)/different(1) RDM from categorical labels."""
    n = len(labels)
    rdm = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rdm[i, j] = rdm[j, i] = 0.0 if labels[i] == labels[j] else 1.0
    return rdm


def _ordinal_distance_rdm(values):
    """
    |rank_i - rank_j| distance RDM from ordinal values.
    NaN entries are masked by setting their distances to NaN.
    """
    n = len(values)
    rdm = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        rdm[i, i] = 0.0
        for j in range(i + 1, n):
            if np.isnan(values[i]) or np.isnan(values[j]):
                continue
            d = abs(values[i] - values[j])
            rdm[i, j] = rdm[j, i] = d
    return rdm


def _continuous_distance_rdm(values):
    """Absolute difference RDM from continuous values."""
    return _ordinal_distance_rdm(values)  # same math


def build_model_rdms(identities, info_df, factors):
    """
    Build model RDMs for a list of stimulus identities.

    Parameters
    ----------
    identities : list of str
        Ordered stimulus monkey names (must match rows of neural RDM).
    info_df : DataFrame
        monkeyinfo.csv with columns: Name, Sex, Age, Group Name, Rank, etc.
    factors : list of str
        Which model RDMs to build. Options:
        'group', 'familiarity', 'sex', 'age_bin', 'age_continuous', 'rank'

    Returns
    -------
    model_rdms : dict {factor_name: ndarray (n, n)}
    model_labels : dict {factor_name: list of labels per identity}
        For inspection / plotting.
    """
    # Build a lookup from monkey name → info
    info = info_df.set_index(info_df['Name'].astype(str))

    model_rdms = {}
    model_labels = {}

    for factor in factors:
        if factor == 'group':
            labels = [info.loc[m, 'Group Name'] if m in info.index else np.nan
                      for m in identities]
            model_rdms['group'] = _categorical_rdm(labels)
            model_labels['group'] = labels

        elif factor == 'familiarity':
            fam_map = {'Zombies': 'familiar', 'Best Frans': 'familiar',
                       'Instigators': 'unfamiliar', 'Stranger Things': 'unfamiliar'}
            labels = []
            for m in identities:
                if m in info.index:
                    g = info.loc[m, 'Group Name']
                    labels.append(fam_map.get(g, np.nan))
                else:
                    labels.append(np.nan)
            model_rdms['familiarity'] = _categorical_rdm(labels)
            model_labels['familiarity'] = labels

        elif factor == 'sex':
            labels = [info.loc[m, 'Sex'] if m in info.index else np.nan
                      for m in identities]
            model_rdms['sex'] = _categorical_rdm(labels)
            model_labels['sex'] = labels

        elif factor == 'age_bin':
            labels = []
            for m in identities:
                if m in info.index and pd.notna(info.loc[m, 'Age']):
                    labels.append('adult' if info.loc[m, 'Age'] >= 4 else 'juvenile')
                else:
                    labels.append(np.nan)
            model_rdms['age_bin'] = _categorical_rdm(labels)
            model_labels['age_bin'] = labels

        elif factor == 'age_continuous':
            vals = []
            for m in identities:
                if m in info.index and pd.notna(info.loc[m, 'Age']):
                    vals.append(float(info.loc[m, 'Age']))
                else:
                    vals.append(np.nan)
            model_rdms['age_continuous'] = _continuous_distance_rdm(vals)
            model_labels['age_continuous'] = vals

        elif factor == 'rank':
            vals = []
            for m in identities:
                if m in info.index and pd.notna(info.loc[m, 'Rank']):
                    vals.append(float(info.loc[m, 'Rank']))
                else:
                    vals.append(np.nan)
            model_rdms['rank'] = _ordinal_distance_rdm(vals)
            model_labels['rank'] = vals

        else:
            raise ValueError(f"Unknown factor: {factor}")

    return model_rdms, model_labels


# ──────────────────────────────────────────────────────────
# 4. RDM comparison
# ──────────────────────────────────────────────────────────

def _upper_triangle(rdm):
    """Extract upper triangle as a flat vector (excluding diagonal)."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def compare_rdms(neural_rdm, model_rdm, n_permutations=0, rng_seed=42):
    """
    Spearman correlation between vectorized upper triangles of two RDMs.

    Handles NaN entries in the model RDM by excluding those pairs.

    Parameters
    ----------
    neural_rdm : ndarray (n, n)
    model_rdm : ndarray (n, n)
    n_permutations : int
        If > 0, run a permutation test (shuffle rows+cols of neural RDM).
    rng_seed : int

    Returns
    -------
    result : dict with keys:
        'rho'   : float, Spearman correlation
        'p_val' : float or None (None if n_permutations == 0)
        'null_distribution' : ndarray or None
    """
    v_neural = _upper_triangle(neural_rdm)
    v_model = _upper_triangle(model_rdm)

    # Mask NaNs (from missing rank/age data)
    mask = ~(np.isnan(v_neural) | np.isnan(v_model))
    if mask.sum() < 3:
        return dict(rho=np.nan, p_val=np.nan, null_distribution=None)

    rho, _ = spearmanr(v_neural[mask], v_model[mask])

    p_val = None
    null_dist = None

    if n_permutations > 0:
        rng = np.random.default_rng(rng_seed)
        n = neural_rdm.shape[0]
        null_dist = np.zeros(n_permutations)

        for perm_i in range(n_permutations):
            perm = rng.permutation(n)
            shuffled = neural_rdm[np.ix_(perm, perm)]
            v_shuf = _upper_triangle(shuffled)
            null_dist[perm_i], _ = spearmanr(v_shuf[mask], v_model[mask])

        # Two-tailed p-value
        p_val = np.mean(np.abs(null_dist) >= np.abs(rho))

    return dict(rho=rho, p_val=p_val, null_distribution=null_dist)


# ──────────────────────────────────────────────────────────
# 5. Per-session pipeline
# ──────────────────────────────────────────────────────────

def run_rsa_session(df, session, info_df, cfg):
    """
    Run RSA for a single recording session.

    Parameters
    ----------
    df : DataFrame (already filtered by region, min epoch duration)
    session : str
    info_df : DataFrame from monkeyinfo.csv
    cfg : RSAConfig

    Returns
    -------
    result : dict with keys:
        'session', 'n_neurons', 'n_identities', 'identities',
        'neural_rdm', 'model_rdms', 'model_labels',
        'comparisons' : dict {factor: {rho, p_val, ...}}
    """
    sess_df = df[df['session'] == session].copy()

    # All identities present in this session
    all_monkeys = sorted(sess_df['MonkeyName'].unique())
    # Keep only those in monkeyinfo
    known = set(info_df['Name'].astype(str))
    identities = [m for m in all_monkeys if m in known]

    rate_matrix, valid_ids, neuron_ids = compute_firing_rates(
        sess_df, identities, cfg.window, cfg.min_reps_per_monkey)

    if len(valid_ids) < 3:
        print(f"  Session {session}: only {len(valid_ids)} identities, skipping")
        return None

    # Optional normalization
    if cfg.normalization != None:
        rate_matrix = normalize_rates(rate_matrix, method=cfg.normalization,
                                      soft_const=cfg.soft_normalize_const)

    neural_rdm = build_neural_rdm(rate_matrix, metric=cfg.neural_metric)
    model_rdms, model_labels = build_model_rdms(valid_ids, info_df, cfg.model_factors)

    comparisons = {}
    for factor in cfg.model_factors:
        comparisons[factor] = compare_rdms(
            neural_rdm, model_rdms[factor],
            n_permutations=cfg.n_permutations,
            rng_seed=cfg.rng_seed)

    return dict(
        session=session,
        n_neurons=len(neuron_ids),
        n_identities=len(valid_ids),
        identities=valid_ids,
        neuron_ids=neuron_ids,
        rate_matrix=rate_matrix,
        neural_rdm=neural_rdm,
        model_rdms=model_rdms,
        model_labels=model_labels,
        comparisons=comparisons,
    )


# ──────────────────────────────────────────────────────────
# 6. Pseudo-population pipeline
# ──────────────────────────────────────────────────────────

def run_rsa_pseudopop(df, info_df, cfg):
    """
    Pool neurons across sessions to build a pseudo-population,
    then compute RSA once on the combined response matrix.

    Only stimulus identities present in ALL sessions are kept.

    Parameters
    ----------
    df : DataFrame (already filtered by region, min epoch duration)
    info_df : DataFrame
    cfg : RSAConfig

    Returns
    -------
    result : dict (same structure as run_rsa_session, but session='pseudo_pop')
    """
    sessions = sorted(df['session'].unique())

    # Find common identities across all sessions
    known = set(info_df['Name'].astype(str))
    per_session_ids = []
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        monkeys = set(sess_df['MonkeyName'].unique()) & known
        per_session_ids.append(monkeys)
    common_ids = sorted(set.intersection(*per_session_ids))

    if len(common_ids) < 3:
        raise ValueError(f"Only {len(common_ids)} common identities across sessions")

    # Build rate matrix per session, then hstack
    session_matrices = []
    all_neuron_ids = []

    for sess in sessions:
        sess_df = df[df['session'] == sess]
        rate_mat, valid_ids, neuron_ids = compute_firing_rates(
            sess_df, common_ids, cfg.window, min_reps=1)
        # valid_ids should == common_ids since they were present
        # Reorder to match common_ids if needed
        if valid_ids != common_ids:
            # Some identities may have been dropped for min_reps in this session
            # For pseudo-pop we use min_reps=1 above, so this shouldn't happen
            print(f"  Warning: session {sess} dropped some common identities")
            continue
        session_matrices.append(rate_mat)
        all_neuron_ids.extend(f"{sess}__{nid}" for nid in neuron_ids)

    combined_matrix = np.hstack(session_matrices)  # (n_identities, n_neurons_total)
    print(f"Pseudo-population matrix: {combined_matrix.shape}")

    # Optional normalization
    if cfg.normalization != None:
        combined_matrix = normalize_rates(combined_matrix, method=cfg.normalization,
                                          soft_const=cfg.soft_normalize_const)

    neural_rdm = build_neural_rdm(combined_matrix, metric=cfg.neural_metric)
    model_rdms, model_labels = build_model_rdms(common_ids, info_df, cfg.model_factors)

    comparisons = {}
    for factor in cfg.model_factors:
        comparisons[factor] = compare_rdms(
            neural_rdm, model_rdms[factor],
            n_permutations=cfg.n_permutations,
            rng_seed=cfg.rng_seed)

    return dict(
        session='pseudo_pop',
        n_neurons=combined_matrix.shape[1],
        n_identities=len(common_ids),
        identities=common_ids,
        neuron_ids=all_neuron_ids,
        rate_matrix=combined_matrix,
        neural_rdm=neural_rdm,
        model_rdms=model_rdms,
        model_labels=model_labels,
        comparisons=comparisons,
    )
