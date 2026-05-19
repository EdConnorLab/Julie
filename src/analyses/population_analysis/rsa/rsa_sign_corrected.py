# rsa_sign_corrected.py
"""
Sign-corrected RSA for opponent neuron populations (e.g. amygdala).

Corrected version — NO split-half.
-  Uses ALL trials for both sign estimation and RDM construction.
-  Permutation test re-estimates signs on every shuffle, so the null
   distribution captures inflation from the sign-selection procedure.

Approach
--------
1. Compute trial-averaged firing rates per identity using all trials.
2. Regress each neuron's rate vector against a social axis score
   (e.g. submission received) → sign = +1 or −1.
3. Build neural RDM(s):
   - 'flip'   : multiply negative-slope neurons by −1, build one RDM
   - 'subpop' : separate pos/neg neurons into two independent RDMs
   - 'both'   : return all three
4. Compare to social RDMs with corrected permutation test.

Corrected permutation test
--------------------------
For each permutation:
  a) Shuffle the social axis scores across identities.
  b) Re-estimate neuron signs from the SHUFFLED scores.
  c) Rebuild neural RDM(s) using the new signs.
  d) Compare to the REAL (unshuffled) social RDM → null ρ.

This null captures both data reuse AND selection bias, so the
p-value is valid despite not splitting trials.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform

from rsa_core import (
    compute_firing_rates,
    build_neural_rdm,
    normalize_rates,
)


# ──────────────────────────────────────────────────────────
# Social axis scores (unchanged from original)
# ──────────────────────────────────────────────────────────

def compute_submission_scores(identities, info_df, interactions,
                               behavior_type='submission',
                               aggregate='column_sum',
                               rank_within_group=True):
    """
    Compute a scalar social-axis score per identity.

    Parameters
    ----------
    identities : list of str
    info_df : DataFrame  (monkeyinfo.csv)
    interactions : dict  (load_all_interaction_matrices output)
    behavior_type : str
        Which interaction matrix to use ('submission' by default).
    aggregate : str
        'column_sum' = total received from groupmates (high = dominant)
        'row_sum'    = total given to groupmates (high = subordinate)
    rank_within_group : bool
        If True, replace raw scores with within-group ranks.

    Returns
    -------
    scores : ndarray (n_identities,)
        NaN where identity not found in any group's matrix.
    """
    from scipy.stats import rankdata as _rankdata

    info = info_df.set_index(info_df['Name'].astype(str))

    scores = np.full(len(identities), np.nan)
    by_group = {}

    for k, m in enumerate(identities):
        if m not in info.index:
            continue
        group = info.loc[m, 'Group Name']
        if group not in interactions or behavior_type not in interactions[group]:
            continue
        matrix, monkey_ids = interactions[group][behavior_type]
        monkey_ids_str = [str(x) for x in monkey_ids]
        if m not in monkey_ids_str:
            continue
        idx = monkey_ids_str.index(m)
        if aggregate == 'column_sum':
            val = float(matrix[:, idx].sum())
        elif aggregate == 'row_sum':
            val = float(matrix[idx, :].sum())
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")
        scores[k] = val
        by_group.setdefault(group, {'k': [], 'val': []})
        by_group[group]['k'].append(k)
        by_group[group]['val'].append(val)

    if rank_within_group:
        for group, bucket in by_group.items():
            ranks = _rankdata(bucket['val']).astype(float)
            for k, r in zip(bucket['k'], ranks):
                scores[k] = r
    else:
        for group, bucket in by_group.items():
            for k, v in zip(bucket['k'], bucket['val']):
                scores[k] = v

    return scores


# ──────────────────────────────────────────────────────────
# Sign estimation (unchanged)
# ──────────────────────────────────────────────────────────

def estimate_neuron_signs(rate_matrix, scores):
    """
    OLS slope sign for each neuron's firing rate per identity
    against the scalar `scores`.

    Returns
    -------
    signs : ndarray (n_neurons,)  ∈ {+1, -1}
    slopes : ndarray (n_neurons,)
    """
    mask = ~np.isnan(scores)
    if mask.sum() < 3:
        n_neurons = rate_matrix.shape[1]
        return np.ones(n_neurons), np.full(n_neurons, np.nan)

    x = scores[mask]
    R = rate_matrix[mask, :]
    x_centered = x - x.mean()
    var_x = float(np.sum(x_centered ** 2))
    if var_x < 1e-12:
        n_neurons = rate_matrix.shape[1]
        return np.ones(n_neurons), np.zeros(n_neurons)

    R_centered = R - R.mean(axis=0, keepdims=True)
    slopes = (x_centered[:, None] * R_centered).sum(axis=0) / var_x
    signs = np.where(slopes < 0, -1.0, 1.0)
    return signs, slopes


# ──────────────────────────────────────────────────────────
# Helper: build sign-corrected RDMs from rate matrix + signs
# ──────────────────────────────────────────────────────────

def _build_sign_corrected_rdms(rate_matrix, signs, metric, mode):
    """
    Given a rate matrix and per-neuron signs, build RDM(s).

    Returns dict with keys depending on mode:
      'flip'   → {'flipped': rdm}
      'subpop' → {'pos': rdm_or_None, 'neg': rdm_or_None}
      'both'   → all of the above
    """
    out = {}

    if mode in ('flip', 'both'):
        flipped = rate_matrix * signs[None, :]
        out['flipped'] = build_neural_rdm(flipped, metric=metric)

    if mode in ('subpop', 'both'):
        pos_mask = signs > 0
        neg_mask = signs < 0
        out['pos'] = (build_neural_rdm(rate_matrix[:, pos_mask], metric=metric)
                      if pos_mask.sum() >= 2 else None)
        out['neg'] = (build_neural_rdm(rate_matrix[:, neg_mask], metric=metric)
                      if neg_mask.sum() >= 2 else None)

    return out


# ──────────────────────────────────────────────────────────
# Corrected permutation test
# ──────────────────────────────────────────────────────────

def _upper_triangle(rdm):
    """Extract upper triangle of a square matrix as 1-D vector."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def compare_with_corrected_permutation(rate_matrix, scores, social_rdms,
                                        identities, info_df, metric,
                                        mode, n_permutations, rng_seed):
    """
    Sign-corrected RSA with permutation test that re-estimates signs
    on every shuffle.

    Parameters
    ----------
    rate_matrix : ndarray (n_identities, n_neurons)
        Full trial-averaged rates (ALL trials).
    scores : ndarray (n_identities,)
        Social axis scores for sign estimation.
    social_rdms : dict {name: ndarray (n, n)}
        Social RDMs to test against.
    identities : list of str
    info_df : DataFrame
    metric : str
        Neural distance metric ('correlation', etc.).
    mode : str  ('flip', 'subpop', 'both')
    n_permutations : int
    rng_seed : int

    Returns
    -------
    results : dict
        {rdm_variant: {social_matrix: {group: {rho, p_val, n_pairs}}}}
    observed_rdms : dict
        The actual neural RDMs produced by sign correction.
    signs : ndarray
    slopes : ndarray
    """
    rng = np.random.default_rng(rng_seed)
    n = len(identities)

    # ── Identify group membership for per-group testing ──
    info_lookup = info_df.set_index(info_df['Name'].astype(str))
    id_to_group = {}
    for m in identities:
        if m in info_lookup.index:
            id_to_group[m] = info_lookup.loc[m, 'Group Name']

    # Build group masks for pairwise comparisons
    group_pairs = {}  # {group_name: list of (i, j) upper-triangle indices}
    for i in range(n):
        for j in range(i + 1, n):
            gi = id_to_group.get(identities[i])
            gj = id_to_group.get(identities[j])
            if gi is not None and gj is not None and gi == gj:
                group_pairs.setdefault(gi, []).append((i, j))

    # ── Observed: estimate signs, build RDMs, compute ρ ──
    signs, slopes = estimate_neuron_signs(rate_matrix, scores)
    observed_rdms = _build_sign_corrected_rdms(rate_matrix, signs, metric, mode)

    # Compute observed ρ for each variant × social matrix × group
    def _compute_rho(neural_rdm, social_rdm, pairs):
        """Spearman ρ on within-group pairs only."""
        if neural_rdm is None or len(pairs) < 3:
            return np.nan
        n_vec = np.array([neural_rdm[i, j] for i, j in pairs])
        s_vec = np.array([social_rdm[i, j] for i, j in pairs])
        # Drop NaN pairs (cross-group in social RDM)
        valid = ~(np.isnan(n_vec) | np.isnan(s_vec))
        if valid.sum() < 3:
            return np.nan
        rho, _ = spearmanr(n_vec[valid], s_vec[valid])
        return rho

    # Structure: {variant: {social_name: {group: observed_rho}}}
    obs = {}
    variants = list(observed_rdms.keys())
    for var in variants:
        obs[var] = {}
        for soc_name, soc_rdm in social_rdms.items():
            obs[var][soc_name] = {}
            for group, pairs in group_pairs.items():
                obs[var][soc_name][group] = _compute_rho(
                    observed_rdms[var], soc_rdm, pairs)

    # ── Permutation: shuffle scores → re-estimate signs → rebuild RDMs ──
    null_dist = {var: {sn: {g: [] for g in group_pairs}
                       for sn in social_rdms}
                 for var in variants}

    valid_score_mask = ~np.isnan(scores)
    valid_indices = np.where(valid_score_mask)[0]

    for perm_i in range(n_permutations):
        # Shuffle the scores among valid (non-NaN) identities
        shuffled_scores = scores.copy()
        shuffled_valid = scores[valid_score_mask].copy()
        rng.shuffle(shuffled_valid)
        shuffled_scores[valid_score_mask] = shuffled_valid

        # Re-estimate signs with shuffled scores
        perm_signs, _ = estimate_neuron_signs(rate_matrix, shuffled_scores)

        # Rebuild neural RDMs with new signs
        perm_rdms = _build_sign_corrected_rdms(
            rate_matrix, perm_signs, metric, mode)

        # Compute ρ for each variant × social × group
        for var in variants:
            for soc_name, soc_rdm in social_rdms.items():
                for group, pairs in group_pairs.items():
                    rho = _compute_rho(perm_rdms[var], soc_rdm, pairs)
                    null_dist[var][soc_name][group].append(rho)

    # ── Compute p-values ──
    results = {}
    for var in variants:
        results[var] = {}
        for soc_name in social_rdms:
            results[var][soc_name] = {}
            for group, pairs in group_pairs.items():
                obs_rho = obs[var][soc_name][group]
                null_arr = np.array(null_dist[var][soc_name][group])

                if np.isnan(obs_rho):
                    p_val = np.nan
                else:
                    # Two-tailed
                    p_val = float(np.mean(np.abs(null_arr) >= np.abs(obs_rho)))

                results[var][soc_name][group] = {
                    'rho': obs_rho,
                    'p_val': p_val,
                    'n_pairs': len(pairs),
                }

    return results, observed_rdms, signs, slopes


# ──────────────────────────────────────────────────────────
# Top-level pipeline (corrected)
# ──────────────────────────────────────────────────────────

def run_sign_corrected_pseudopop(df, info_df, interactions, cfg,
                                  mode='both', sign_axis='submission',
                                  axis_aggregate='column_sum',
                                  n_permutations=2000,
                                  rng_seed=42):
    """
    Pseudo-population RSA with sign correction and corrected
    permutation test.

    No split-half: all trials are used for both sign estimation
    and RDM construction. The permutation test re-estimates signs
    on every shuffle to account for selection bias.

    Parameters
    ----------
    df : DataFrame  (already filtered by region, min epoch, etc.)
    info_df : DataFrame
    interactions : dict
    cfg : RSAConfig / SocialRSAConfig
    mode : str  ('flip', 'subpop', 'both')
    sign_axis : str  ('submission', 'agonism', 'affiliation')
    axis_aggregate : str  ('column_sum' or 'row_sum')
    n_permutations : int
    rng_seed : int

    Returns
    -------
    result : dict with keys:
        'identities', 'scores', 'signs', 'slopes',
        'n_neurons', 'n_neurons_pos', 'n_neurons_neg',
        'observed_rdms'  : {variant: neural RDM}
        'comparisons'    : {variant: {social_name: {group: {rho, p_val, n_pairs}}}}
    """
    sessions = sorted(df['session'].unique())
    known = set(info_df['Name'].astype(str))

    # Find identities present in all sessions
    per_session_ids = []
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        per_session_ids.append(set(sess_df['MonkeyName'].unique()) & known)
    common_ids = sorted(set.intersection(*per_session_ids))

    # Sensitivity analysis: drop excluded identities
    exclude_ids = set(getattr(cfg, 'exclude_identities', []) or [])
    if exclude_ids:
        removed = [m for m in common_ids if m in exclude_ids]
        common_ids = [m for m in common_ids if m not in exclude_ids]
        print(f"  [Sensitivity] Excluded identities: {removed}")

    # Diagnostics
    from collections import Counter
    info_lookup = info_df.set_index(info_df['Name'].astype(str))
    surviving_groups = Counter(
        info_lookup.loc[m, 'Group Name'] for m in common_ids
        if m in info_lookup.index)
    print(f"\n  Identity coverage across {len(sessions)} sessions:")
    print(f"    common to all sessions: {len(common_ids)}")
    print(f"    per-group counts: {dict(surviving_groups)}")

    if len(common_ids) < 3:
        raise ValueError(f"Only {len(common_ids)} common identities")

    # ── Compute firing rates using ALL trials (no split-half) ──
    all_rate_matrices = []
    all_neuron_ids = []

    for sess in sessions:
        sess_df = df[df['session'] == sess]
        rate_mat, valid_ids, neuron_ids = compute_firing_rates(
            sess_df, common_ids, cfg.window, min_reps=1)
        if valid_ids != common_ids:
            print(f"  Warning: session {sess} dropped some identities")
            continue
        all_rate_matrices.append(rate_mat)
        all_neuron_ids.extend(f"{sess}__{nid}" for nid in neuron_ids)

    rate_matrix = np.hstack(all_rate_matrices)  # (n_identities, n_neurons)
    print(f"  Pseudo-population: {rate_matrix.shape[1]} neurons")

    # Optional normalization
    if cfg.normalization is not None:
        rate_matrix = normalize_rates(
            rate_matrix, method=cfg.normalization,
            soft_const=getattr(cfg, 'soft_normalize_const', 5.0))

    # ── Social axis scores ──
    scores = compute_submission_scores(
        common_ids, info_df, interactions,
        behavior_type=sign_axis, aggregate=axis_aggregate,
        rank_within_group=True)

    # ── Build social RDMs to test against ──
    from rsa_social import build_social_rdms
    tx = getattr(cfg, 'transform_social_behavior', None)
    social_rdms = build_social_rdms(
        common_ids, interactions,
        behavior_types=['affiliation', 'agonism', 'submission'],
        symmetrize=False, profile_metric='correlation',
        log_transform=(tx == 'log'),
        include_combined=True,
        rank_transform=(tx == 'rank'),
        exclude_ids=list(exclude_ids) if exclude_ids else None)

    # ── Run corrected comparison ──
    print(f"  Running corrected permutation test ({n_permutations} permutations)...")
    comparisons, observed_rdms, signs, slopes = \
        compare_with_corrected_permutation(
            rate_matrix, scores, social_rdms,
            common_ids, info_df,
            metric=cfg.neural_metric,
            mode=mode,
            n_permutations=n_permutations,
            rng_seed=rng_seed)

    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
    print(f"  Signs: {n_pos} positive, {n_neg} negative")

    return {
        'identities': common_ids,
        'scores': scores,
        'signs': signs,
        'slopes': slopes,
        'n_neurons': rate_matrix.shape[1],
        'n_neurons_pos': n_pos,
        'n_neurons_neg': n_neg,
        'neuron_ids': all_neuron_ids,
        'rate_matrix': rate_matrix,
        'observed_rdms': observed_rdms,
        'comparisons': comparisons,
    }


# ──────────────────────────────────────────────────────────
# Pretty-print results
# ──────────────────────────────────────────────────────────

def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def print_sign_corrected_results(comparisons):
    """Print results in a readable table, one section per RDM variant."""
    for variant, by_social in comparisons.items():
        print(f"\n  --- {variant} ---\n")

        # Collect all groups
        all_groups = sorted({
            g for soc_results in by_social.values()
            for g in soc_results.keys()})

        header = f"  {'RDM/RSM':<45s}" + "".join(
            f"{g:>25s}" for g in all_groups)
        print(header)
        print("  " + "-" * (45 + 25 * len(all_groups)))

        for soc_name in sorted(by_social.keys()):
            row = f"  {soc_name:<45s}"
            for group in all_groups:
                entry = by_social[soc_name].get(group, {})
                rho = entry.get('rho', np.nan)
                p = entry.get('p_val', np.nan)
                n = entry.get('n_pairs', 0)
                star = _stars(p)
                if np.isnan(rho):
                    row += f"{'—':>25s}"
                else:
                    row += f"{rho:+.4f} (n={n}) p={p:.3f}{star:3s}".rjust(25)
            print(row)
