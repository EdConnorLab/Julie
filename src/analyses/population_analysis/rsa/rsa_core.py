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

from analyses.population_analysis.rsa.rsa_utils import upper_triangle as _upper_triangle, partial_spearman, finite_mask


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
        'none' → no normalization (return as-is)
        'soft' → divide by (range + const); conservative equalization
    soft_const : float
        Additive constant for soft normalization (spikes/s).

    Returns
    -------
    normalized : ndarray, same shape
    """
    if method == 'none':
        return rate_matrix

    if method == 'zscore':
        mu = rate_matrix.mean(axis=0)
        sd = rate_matrix.std(axis=0, ddof=0)
        n_flat = np.sum(sd < 1e-6)
        if n_flat > 0:
            print(f"  Z-score: {n_flat}/{rate_matrix.shape[1]} neurons had near-zero std (left as-is)")
        safe_sd = np.where(sd < 1e-6, 1.0, sd)   # avoid division by zero; flat neurons become 0s after mean-sub
        return (rate_matrix - mu) / safe_sd

    if method == 'soft':
        rng = rate_matrix.max(axis=0) - rate_matrix.min(axis=0)
        denom = rng + soft_const
        normalized = rate_matrix / denom
        n_flat = np.sum(rng < 1e-6)
        if n_flat > 0:
            print(f"  Soft-norm: {n_flat}/{rate_matrix.shape[1]} neurons had near-zero range")
        return normalized

    raise ValueError(f"Unknown normalization method: {method}")


def variance_quartile_diagnostic(rate_matrix, info_df, identities, cfg,
                                 quartiles=(0.25, 0.50, 0.75, 1.0)):
    """
    Diagnostic: rebuild the neural RDM using only the top-Q% of neurons
    ranked by pre-normalization std, and re-run RSA comparisons for each
    quartile cutoff.

    Use this to check whether RSA structure is driven by a small subset of
    high-variance neurons (pattern reappears with top 25%) or is distributed
    (only emerges at 75–100%).

    Parameters
    ----------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
        **Raw** (pre-normalization) firing-rate matrix.
    info_df : DataFrame
    identities : list of str
    cfg : RSAConfig
    quartiles : tuple of float
        Cumulative fractions of neurons to include, sorted by descending std.
        E.g. (0.25, 0.50, 0.75, 1.0) → top 25%, top 50%, top 75%, all.

    Returns
    -------
    diag_results : list of dict
        One entry per quartile with keys:
        'quantile', 'n_neurons', 'comparisons'
    """
    n_neurons = rate_matrix.shape[1]
    neuron_std = rate_matrix.std(axis=0, ddof=0)
    # Rank neurons: highest std first
    ranked_idx = np.argsort(neuron_std)[::-1]

    # Build model RDMs once (shared across quartiles)
    all_factors = list(cfg.model_factors)
    for pf in getattr(cfg, 'partial_out', []):
        if pf not in all_factors:
            all_factors.append(pf)
    model_rdms, model_labels = build_model_rdms(identities, info_df, all_factors)

    partial_out = getattr(cfg, 'partial_out', [])
    diag_results = []

    for q in quartiles:
        k = max(1, int(np.ceil(n_neurons * q)))
        sel = ranked_idx[:k]
        sub_matrix = rate_matrix[:, sel]

        # Apply the same normalization the main pipeline uses
        if cfg.normalization is not None:
            sub_matrix = normalize_rates(sub_matrix, method=cfg.normalization,
                                         soft_const=cfg.soft_normalize_const)

        rdm = build_neural_rdm(sub_matrix, metric=cfg.neural_metric)

        comparisons = {}
        for factor in cfg.model_factors:
            confound_factors = [p for p in partial_out if p != factor]
            if confound_factors:
                confound_rdm_list = [model_rdms[p] for p in confound_factors]
                comparisons[factor] = compare_rdms_partial(
                    rdm, model_rdms[factor], confound_rdm_list,
                    n_permutations=0, rng_seed=cfg.rng_seed)
            else:
                comparisons[factor] = compare_rdms(
                    rdm, model_rdms[factor],
                    n_permutations=0, rng_seed=cfg.rng_seed)

        diag_results.append(dict(
            quantile=q,
            n_neurons=k,
            neuron_indices=sel,
            neural_rdm=rdm,
            identities=identities,
            model_rdms=model_rdms,
            model_labels=model_labels,
            comparisons=comparisons,
        ))

    return diag_results


def random_subset_diagnostic(rate_matrix, info_df, identities, cfg,
                             fraction=0.25, n_draws=4, rng_seed=None):
    """
    Diagnostic: rebuild the neural RDM using randomly-chosen subsets of
    neurons, to compare against the top-X% by variance subset.

    If the top-X%-by-variance RDM looks structurally different from random
    subsets of the same size, the high-variance neurons are privileged
    signal carriers (sparse / sparse-distributed code).  If random subsets
    look comparable to the top-variance subset, the code is more distributed.

    Parameters
    ----------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
        **Raw** (pre-normalization) firing-rate matrix.
    info_df : DataFrame
    identities : list of str
    cfg : RSAConfig
    fraction : float
        Fraction of neurons in each random subset (default 0.25 to match
        the top-25% quartile of variance_quartile_diagnostic).
    n_draws : int
        Number of independent random subsets (default 4 → 4 exemplar panels).
    rng_seed : int or None
        Seed for reproducibility.  None → use cfg.rng_seed.

    Returns
    -------
    random_results : list of dict
        One entry per random draw with keys compatible with the existing
        plotting helpers (same shape as variance_quartile_diagnostic entries):
        'draw_index', 'fraction', 'n_neurons', 'neuron_indices',
        'neural_rdm', 'identities', 'model_rdms', 'model_labels', 'comparisons'.
    """
    n_neurons = rate_matrix.shape[1]
    k = max(1, int(np.ceil(n_neurons * fraction)))

    if rng_seed is None:
        rng_seed = cfg.rng_seed
    rng = np.random.default_rng(rng_seed)

    # Build model RDMs once (shared across draws)
    all_factors = list(cfg.model_factors)
    for pf in getattr(cfg, 'partial_out', []):
        if pf not in all_factors:
            all_factors.append(pf)
    model_rdms, model_labels = build_model_rdms(identities, info_df, all_factors)

    partial_out = getattr(cfg, 'partial_out', [])
    random_results = []

    for draw_i in range(n_draws):
        sel = rng.choice(n_neurons, size=k, replace=False)
        sub_matrix = rate_matrix[:, sel]

        # Apply same normalization the main pipeline uses
        if cfg.normalization is not None:
            sub_matrix = normalize_rates(sub_matrix, method=cfg.normalization,
                                         soft_const=cfg.soft_normalize_const)

        rdm = build_neural_rdm(sub_matrix, metric=cfg.neural_metric)

        comparisons = {}
        for factor in cfg.model_factors:
            confound_factors = [p for p in partial_out if p != factor]
            if confound_factors:
                confound_rdm_list = [model_rdms[p] for p in confound_factors]
                comparisons[factor] = compare_rdms_partial(
                    rdm, model_rdms[factor], confound_rdm_list,
                    n_permutations=0, rng_seed=cfg.rng_seed)
            else:
                comparisons[factor] = compare_rdms(
                    rdm, model_rdms[factor],
                    n_permutations=0, rng_seed=cfg.rng_seed)

        random_results.append(dict(
            draw_index=draw_i,
            fraction=fraction,
            n_neurons=k,
            neuron_indices=sel,
            neural_rdm=rdm,
            identities=identities,
            model_rdms=model_rdms,
            model_labels=model_labels,
            comparisons=comparisons,
        ))

    return random_results

def print_variance_diagnostic(diag_results, cfg):
    """Pretty-print the variance-quartile diagnostic table."""
    factors = cfg.model_factors
    header = f"{'Top %':>8s}  {'n_neur':>6s}" + "".join(f"  {f:>16s}" for f in factors)
    print(f"\n  Variance-quartile diagnostic")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for entry in diag_results:
        pct = f"{entry['quantile']*100:.0f}%"
        row = f"  {pct:>8s}  {entry['n_neurons']:>6d}"
        for f in factors:
            rho = entry['comparisons'][f]['rho']
            row += f"  {rho:>+16.4f}"
        print(row)
    print()


def compute_containment_metric(diag_results, info_df):
    """
    For each variance quartile and each social group, compute:

        containment = mean(within-group 1-r) - mean(between-group 1-r)

    Positive  → high dissimilarity *contained* within that group's block
    Near zero → dissimilarity spread equally within and between (no block)
    Negative  → within-group more similar than between-group

    Parameters
    ----------
    diag_results : list of dict
        Output of variance_quartile_diagnostic (or random_subset_diagnostic).
    info_df : DataFrame
        Must have 'Name' and 'Group Name' columns.

    Returns
    -------
    list of dict, one per quartile entry.
    """
    name_to_group = dict(zip(info_df['Name'].astype(str),
                             info_df['Group Name']))

    results = []
    for entry in diag_results:
        rdm = entry['neural_rdm']
        identities = entry['identities']
        groups = [name_to_group.get(mid, 'Unknown') for mid in identities]
        unique_groups = sorted(set(g for g in groups if g != 'Unknown'))

        quartile_result = dict(
            quantile=entry.get('quantile', entry.get('fraction')),
            n_neurons=entry['n_neurons'],
            groups={},
        )

        for g in unique_groups:
            g_idx = [i for i, grp in enumerate(groups) if grp == g]
            o_idx = [i for i, grp in enumerate(groups) if grp != g]

            within_vals = [rdm[g_idx[i], g_idx[j]]
                           for i in range(len(g_idx))
                           for j in range(i + 1, len(g_idx))]
            between_vals = [rdm[i, j] for i in g_idx for j in o_idx]

            w = np.mean(within_vals) if within_vals else np.nan
            b = np.mean(between_vals) if between_vals else np.nan

            quartile_result['groups'][g] = dict(
                within_mean=w,
                between_mean=b,
                containment=w - b,
                n_within_pairs=len(within_vals),
                n_between_pairs=len(between_vals),
            )

        results.append(quartile_result)
    return results
# ──────────────────────────────────────────────────────────
# 1. Firing-rate computation
# ──────────────────────────────────────────────────────────

def compute_firing_rates(df, identities, window, min_reps):
    """
    Compute trial-averaged mean firing rate per neuron per stimulus identity
    within a time window.

    Each neuron is averaged over the trials on which IT was recorded
    (so neurons missing from some trials are not biased toward zero).
    Identities are filtered by the number of *trials shown for that identity*
    (regardless of which neurons recorded them).

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
        Minimum trials per identity; identities with fewer are dropped.

    Returns
    -------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
        Trial-averaged firing rate (spikes/s). NaN where a neuron was
        not recorded on any kept trial for that identity.
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

    # Per-trial spike count for every (TaskField, NeuronID) row in df
    # Vectorized over rows; avoids the previous double iterrows loop.
    epoch_starts = df['EpochStartStop'].map(lambda x: x[0]).to_numpy()
    spike_arrays = df['SpikeTimes'].to_numpy(dtype=object)
    counts = np.empty(len(df), dtype=np.int64)
    for i, (spk, t0) in enumerate(zip(spike_arrays, epoch_starts)):
        s = np.asarray(spk) - t0
        counts[i] = np.count_nonzero((s >= win_start) & (s < win_end))

    counts_df = pd.DataFrame({
        'TaskField': df['TaskField'].to_numpy(),
        'NeuronID':  df['NeuronID'].to_numpy(),
        'Count':     counts,
    })

    # Identity → list of TaskFields (trials)
    trial_meta = (df.groupby('TaskField')['MonkeyName'].first()
                    .reset_index())

    valid_identities = []
    rate_rows = []

    for monkey in identities:
        trials = trial_meta.loc[trial_meta['MonkeyName'] == monkey,
                                'TaskField'].to_numpy()
        if len(trials) < min_reps:
            continue

        sub = counts_df[counts_df['TaskField'].isin(trials)]
        # sum counts and trial counts per neuron (handles missing-neuron trials)
        sum_counts   = np.zeros(n_neurons, dtype=float)
        trial_counts = np.zeros(n_neurons, dtype=float)
        if len(sub):
            agg = sub.groupby('NeuronID')['Count'].agg(['sum', 'count'])
            for nid, row in agg.iterrows():
                j = neuron_to_idx[nid]
                sum_counts[j]   = row['sum']
                trial_counts[j] = row['count']

        with np.errstate(invalid='ignore', divide='ignore'):
            avg_rate = np.where(trial_counts > 0,
                                sum_counts / trial_counts / win_dur,
                                np.nan)
        rate_rows.append(avg_rate)
        valid_identities.append(monkey)

    rate_matrix = np.asarray(rate_rows)  # (n_identities, n_neurons)

    # If a neuron was not recorded on ANY kept trial for some identity, its
    # rate is NaN. Fill with 0 to keep the downstream RDM well-defined and
    # warn so the user knows.
    if rate_matrix.size and np.isnan(rate_matrix).any():
        n_nan = int(np.isnan(rate_matrix).sum())
        print(f"  compute_firing_rates: {n_nan} (identity, neuron) cells "
              f"had no recorded trials — filling with 0")
        rate_matrix = np.nan_to_num(rate_matrix, nan=0.0)

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


def _absdiff_rdm(values):
    """
    |value_i - value_j| distance RDM. NaN entries propagate (those pairs
    end up NaN). Used for both ordinal ranks and continuous values
    (math is identical; when `values` are integer ranks this is rank distance).
    """
    v = np.asarray(values, dtype=float)
    rdm = np.abs(v[:, None] - v[None, :])
    # Wherever either operand was NaN, np.abs already returns NaN; ensure
    # the diagonal is 0 even if a value is NaN.
    diag = np.diag_indices_from(rdm)
    rdm[diag] = np.where(np.isnan(v), np.nan, 0.0)
    # Replace diagonal NaNs with 0 (a monkey is identical to itself
    # whether or not its metadata is known)
    rdm[diag] = 0.0
    return rdm


# Backwards-compat aliases
_ordinal_distance_rdm    = _absdiff_rdm
_continuous_distance_rdm = _absdiff_rdm


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

    mask = finite_mask(v_neural, v_model)
    if mask.sum() < 3:
        return dict(rho=np.nan, p_val=None, null_distribution=None)

    rho, _ = spearmanr(v_neural[mask], v_model[mask])

    p_val = None
    null_dist = None

    if n_permutations > 0:
        rng = np.random.default_rng(rng_seed)
        n = neural_rdm.shape[0]
        null_dist = np.zeros(n_permutations)
        v_model_obs = v_model  # alias

        for perm_i in range(n_permutations):
            perm = rng.permutation(n)
            v_shuf = _upper_triangle(neural_rdm[np.ix_(perm, perm)])
            # Re-mask each permutation in case the shuffled neural triangle
            # has different NaN positions than the original.
            m = finite_mask(v_shuf, v_model_obs)
            if m.sum() < 3:
                null_dist[perm_i] = np.nan
                continue
            null_dist[perm_i], _ = spearmanr(v_shuf[m], v_model_obs[m])

        p_val = float(np.nanmean(np.abs(null_dist) >= np.abs(rho)))

    return dict(rho=rho, p_val=p_val, null_distribution=null_dist)


def compare_rdms_partial(neural_rdm, target_rdm, confound_rdms,
                         n_permutations=0, rng_seed=42):
    """
    Partial Spearman correlation between neural RDM and a target model RDM,
    controlling for one or more confound model RDMs.

    Method: rank-transform all vectors, then OLS-regress out confounds from
    both the neural and target vectors, and Pearson-correlate the residuals.

    Permutation test: shuffle rows+cols of the neural RDM, recompute the
    partial correlation each time.

    Parameters
    ----------
    neural_rdm : ndarray (n, n)
    target_rdm : ndarray (n, n)
        The model RDM whose unique contribution you want to test.
    confound_rdms : list of ndarray (n, n)
        Model RDMs to partial out (e.g. familiarity, group).
    n_permutations : int
    rng_seed : int

    Returns
    -------
    result : dict with keys:
        'rho'   : float, partial Spearman correlation
        'p_val' : float or None
        'null_distribution' : ndarray or None
    """
    v_neural = _upper_triangle(neural_rdm)
    v_target = _upper_triangle(target_rdm)
    v_confounds = [_upper_triangle(c) for c in confound_rdms]

    mask = finite_mask(v_neural, v_target, *v_confounds)
    if mask.sum() < 3 + len(confound_rdms):
        return dict(rho=np.nan, p_val=None, null_distribution=None)

    rho = partial_spearman(v_neural, v_target, v_confounds, mask)

    p_val = None
    null_dist = None

    if n_permutations > 0:
        rng = np.random.default_rng(rng_seed)
        n = neural_rdm.shape[0]
        null_dist = np.zeros(n_permutations)

        for perm_i in range(n_permutations):
            perm = rng.permutation(n)
            v_shuf = _upper_triangle(neural_rdm[np.ix_(perm, perm)])
            m = finite_mask(v_shuf, v_target, *v_confounds)
            if m.sum() < 3 + len(confound_rdms):
                null_dist[perm_i] = np.nan
                continue
            null_dist[perm_i] = partial_spearman(v_shuf, v_target, v_confounds, m)

        p_val = float(np.nanmean(np.abs(null_dist) >= np.abs(rho)))

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
    raw_rate_matrix = rate_matrix.copy()
    if cfg.normalization is not None:
        rate_matrix = normalize_rates(rate_matrix, method=cfg.normalization,
                                      soft_const=cfg.soft_normalize_const)

    neural_rdm = build_neural_rdm(rate_matrix, metric=cfg.neural_metric)

    # Ensure partial_out factors are built even if not in model_factors
    all_factors = list(cfg.model_factors)
    for pf in getattr(cfg, 'partial_out', []):
        if pf not in all_factors:
            all_factors.append(pf)
    model_rdms, model_labels = build_model_rdms(valid_ids, info_df, all_factors)

    # Run comparisons: partial or standard
    partial_out = getattr(cfg, 'partial_out', [])
    comparisons = {}
    for factor in cfg.model_factors:
        confound_factors = [p for p in partial_out if p != factor]
        if confound_factors:
            confound_rdms = [model_rdms[p] for p in confound_factors]
            comparisons[factor] = compare_rdms_partial(
                neural_rdm, model_rdms[factor], confound_rdms,
                n_permutations=cfg.n_permutations,
                rng_seed=cfg.rng_seed)
        else:
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
        raw_rate_matrix=raw_rate_matrix,
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
    min_reps = cfg.min_reps_per_monkey

    # An identity must appear in EVERY session (the pseudo-population hstacks
    # neurons across sessions, so each kept identity needs a rate in every
    # session's block). Among those, keep identities whose POOLED trial count
    # across sessions is >= min_reps.
    #
    # This is a pooled (total) sufficiency floor, not a per-session one:
    # requiring >= min_reps in *every* session is an intersection that
    # collapses under sparse data (e.g. ER), so per-neuron rates from a thin
    # session are still possible — those identities are warned about below.
    known = set(info_df['Name'].astype(str))
    per_session_present = []
    sess_counts = {}   # sess -> {monkey -> n_trials}
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        counts = sess_df.groupby('MonkeyName')['TaskField'].nunique().to_dict()
        sess_counts[sess] = counts
        per_session_present.append(set(counts) & known)
    present_all = (set.intersection(*per_session_present)
                   if per_session_present else set())

    common_ids, thin = [], []
    for m in sorted(present_all):
        per_sess = [sess_counts[s].get(m, 0) for s in sessions]
        if sum(per_sess) >= min_reps:
            common_ids.append(m)
            if min(per_sess) < min_reps:
                thin.append((m, min(per_sess)))

    if len(common_ids) < 3:
        raise ValueError(
            f"Only {len(common_ids)} identities present in all sessions with "
            f">= {min_reps} pooled reps. Lower cfg.min_reps_per_monkey, or "
            f"widen the region/session selection.")

    if thin:
        preview = ', '.join(f"{m}(min {c}/session)" for m, c in thin[:8])
        print(f"  Note: {len(thin)}/{len(common_ids)} identities have a session "
              f"with < {min_reps} reps (rates there rest on few trials): "
              f"{preview}{' ...' if len(thin) > 8 else ''}")

    # Build rate matrix per session, then hstack
    session_matrices = []
    all_neuron_ids = []

    for sess in sessions:
        sess_df = df[df['session'] == sess]
        # min_reps=1 here: identity sufficiency is handled above via the pooled
        # filter, so don't let a thin session drop an identity (which would
        # misalign the hstack).
        rate_mat, valid_ids, neuron_ids = compute_firing_rates(
            sess_df, common_ids, cfg.window, min_reps=1)
        # common_ids are present in every session, so none should drop here.
        if valid_ids != common_ids:
            print(f"  Warning: session {sess} dropped some common identities")
            continue
        session_matrices.append(rate_mat)
        all_neuron_ids.extend(f"{sess}__{nid}" for nid in neuron_ids)

    combined_matrix = np.hstack(session_matrices)  # (n_identities, n_neurons_total)
    print(f"Pseudo-population matrix: {combined_matrix.shape}")

    # Optional normalization
    raw_combined_matrix = combined_matrix.copy()
    if cfg.normalization is not None:
        combined_matrix = normalize_rates(combined_matrix, method=cfg.normalization,
                                          soft_const=cfg.soft_normalize_const)

    neural_rdm = build_neural_rdm(combined_matrix, metric=cfg.neural_metric)

    # Ensure partial_out factors are built even if not in model_factors
    all_factors = list(cfg.model_factors)
    for pf in getattr(cfg, 'partial_out', []):
        if pf not in all_factors:
            all_factors.append(pf)
    model_rdms, model_labels = build_model_rdms(common_ids, info_df, all_factors)

    # Run comparisons: partial or standard
    partial_out = getattr(cfg, 'partial_out', [])
    comparisons = {}
    for factor in cfg.model_factors:
        confound_factors = [p for p in partial_out if p != factor]
        if confound_factors:
            confound_rdms = [model_rdms[p] for p in confound_factors]
            comparisons[factor] = compare_rdms_partial(
                neural_rdm, model_rdms[factor], confound_rdms,
                n_permutations=cfg.n_permutations,
                rng_seed=cfg.rng_seed)
        else:
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
        raw_rate_matrix=raw_combined_matrix,
        neural_rdm=neural_rdm,
        model_rdms=model_rdms,
        model_labels=model_labels,
        comparisons=comparisons,
    )
