# rsa_social.py
"""
Social behavior RDM/RSM construction for RSA.

DISSIMILARITY APPROACHES (original):
  Approach 1 (behavioral profile RDM):
      Each monkey's row in the interaction matrix is its behavioral profile.
      Correlation distance (1-r) between profiles gives the RDM.
      Question: "Do these two monkeys behave similarly toward others?"

  Approach 2 (direct interaction RDM):
      Raw interaction count → dissimilarity via max - count.
      Question: "Do these two monkeys interact a lot?" (inverted)

SIMILARITY APPROACH (new):
  Neural similarity (Pearson r) vs raw interaction counts (untransformed).
  No conversion, no manipulation of social data.
  Question: "Do monkeys who interact more have more correlated neural responses?"

All approaches:
  - Cross-group pairs are NaN.
  - Optional symmetrize: (A[i,j] + A[j,i]) / 2.
  - Optional log_transform: log(1 + count).
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────
# Loading interaction matrices
# ──────────────────────────────────────────────────────────

def load_interaction_matrix(filepath):
    """
    Load a single interaction matrix from an xlsx file.

    Expected format:
        - First column: 'Focal Name' with monkey IDs as row labels
        - Remaining columns: 'Behavior Towards <ID>' with interaction counts

    Returns
    -------
    matrix : ndarray, shape (n_monkeys, n_monkeys)
    monkey_ids : list of str
    """
    df = pd.read_excel(filepath)

    monkey_ids = [str(x) for x in df.iloc[:, 0].values]

    col_ids = []
    for col in df.columns[1:]:
        col_id = str(col).replace('Behavior Towards ', '')
        col_ids.append(col_id)

    if monkey_ids != col_ids:
        print(f"  Warning: row IDs {monkey_ids} != col IDs {col_ids}")
        print(f"  Proceeding with row IDs as canonical order")

    matrix = df.iloc[:, 1:].values.astype(float)
    return matrix, monkey_ids


def load_all_interaction_matrices(behavior_files):
    """
    Load interaction matrices for all groups and behavior types.

    Parameters
    ----------
    behavior_files : dict  {group_name: {behavior_type: filepath}}

    Returns
    -------
    interactions : dict  {group_name: {behavior_type: (matrix, monkey_ids)}}
    """
    interactions = {}
    for group, files in behavior_files.items():
        interactions[group] = {}
        for btype, fpath in files.items():
            matrix, monkey_ids = load_interaction_matrix(fpath)
            interactions[group][btype] = (matrix, monkey_ids)
            print(f"  Loaded {group} {btype}: {len(monkey_ids)} monkeys, "
                  f"shape {matrix.shape}")
    return interactions


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _build_monkey_lookup(interactions, behavior_type, symmetrize=False,
                          log_transform=False):
    """Build lookup tables from interaction data."""
    monkey_to_group = {}
    group_matrices = {}

    for group_name, btypes in interactions.items():
        if behavior_type not in btypes:
            continue
        matrix, monkey_ids = btypes[behavior_type]
        matrix = matrix.copy().astype(float)

        if symmetrize:
            matrix = (matrix + matrix.T) / 2.0

        if log_transform:
            matrix = np.log1p(matrix)

        group_matrices[group_name] = (matrix, monkey_ids)
        for idx, mid in enumerate(monkey_ids):
            monkey_to_group[mid] = (group_name, idx)

    return monkey_to_group, group_matrices


def _upper_triangle(mat):
    """Extract upper triangle as flat vector."""
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def _compute_pairwise_distance(profile_i, profile_j, metric):
    """Compute distance between two profile vectors."""
    if metric == 'correlation':
        std_i = np.std(profile_i)
        std_j = np.std(profile_j)
        if std_i < 1e-10 or std_j < 1e-10:
            return np.nan
        r = np.corrcoef(profile_i, profile_j)[0, 1]
        return 1.0 - r
    elif metric == 'euclidean':
        return np.sqrt(np.sum((profile_i - profile_j) ** 2))
    elif metric == 'cosine':
        norm_i = np.linalg.norm(profile_i)
        norm_j = np.linalg.norm(profile_j)
        if norm_i < 1e-10 or norm_j < 1e-10:
            return np.nan
        return 1.0 - np.dot(profile_i, profile_j) / (norm_i * norm_j)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _get_group_indices(identities, info_df):
    """Return {group_name: [indices into identities]}."""
    info = info_df.set_index(info_df['Name'].astype(str))
    id_groups = {}
    for m in identities:
        if m in info.index:
            id_groups[m] = info.loc[m, 'Group Name']

    groups_present = sorted(set(id_groups.values()))
    group_indices = {}
    for g in groups_present:
        group_indices[g] = [i for i, m in enumerate(identities) if id_groups.get(m) == g]
    return group_indices


def build_rank_distance_matrix(identities, info_df):
    """
    Build a |rank_i - rank_j| distance matrix for all identity pairs.
    Cross-group pairs are NaN. Diagonal is 0.

    Uses the 'Rank' column from monkeyinfo (David's score results).

    Parameters
    ----------
    identities : list of str
    info_df : DataFrame with 'Name', 'Group Name', 'Rank' columns

    Returns
    -------
    rank_dist : ndarray (n, n)
    """
    n = len(identities)
    rank_dist = np.full((n, n), np.nan)
    np.fill_diagonal(rank_dist, 0.0)

    info = info_df.set_index(info_df['Name'].astype(str))

    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = identities[i], identities[j]
            if mi not in info.index or mj not in info.index:
                continue
            gi = info.loc[mi, 'Group Name']
            gj = info.loc[mj, 'Group Name']
            if gi != gj:
                continue
            ri = info.loc[mi, 'Rank']
            rj = info.loc[mj, 'Rank']
            if pd.notna(ri) and pd.notna(rj):
                d = abs(float(ri) - float(rj))
                rank_dist[i, j] = rank_dist[j, i] = d

    return rank_dist


def _partial_spearman(v_x, v_y, v_confound, mask):
    """
    Partial Spearman correlation between v_x and v_y, controlling for v_confound.

    Rank-transforms masked values, OLS-regresses out confound from both,
    then Pearson-correlates residuals.
    """
    r_x = rankdata(v_x[mask])
    r_y = rankdata(v_y[mask])
    r_c = rankdata(v_confound[mask])

    C = np.column_stack([np.ones(mask.sum()), r_c])

    betas_x = np.linalg.lstsq(C, r_x, rcond=None)[0]
    resid_x = r_x - C @ betas_x

    betas_y = np.linalg.lstsq(C, r_y, rcond=None)[0]
    resid_y = r_y - C @ betas_y

    std_x = np.std(resid_x)
    std_y = np.std(resid_y)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return np.corrcoef(resid_x, resid_y)[0, 1]


# ══════════════════════════════════════════════════════════
#  SIMILARITY-BASED APPROACH (NEW)
#  Neural Pearson r  vs  raw interaction counts
# ══════════════════════════════════════════════════════════

def build_neural_similarity_matrix(rate_matrix):
    """
    Build neural similarity matrix: Pearson r between all pairs of
    stimulus response patterns.

    Parameters
    ----------
    rate_matrix : ndarray, shape (n_identities, n_neurons)

    Returns
    -------
    sim_matrix : ndarray, shape (n_identities, n_identities)
        Pearson r values. Diagonal = 1.
    """
    # pdist returns 1-r, so similarity = 1 - pdist
    dists = pdist(rate_matrix, metric='correlation')
    sim_vec = 1.0 - dists
    sim_matrix = squareform(sim_vec)
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def build_interaction_similarity_matrix(identities, interactions, behavior_type,
                                         symmetrize=False, log_transform=False):
    """
    Build a similarity matrix from raw interaction counts.
    No dissimilarity conversion — values are raw counts (or log(1+count)).

    Cross-group pairs are NaN.

    Parameters
    ----------
    identities : list of str
    interactions : dict
    behavior_type : str
    symmetrize : bool
    log_transform : bool

    Returns
    -------
    sim_matrix : ndarray, shape (n_identities, n_identities)
        Raw interaction counts. NaN for cross-group pairs. Diagonal = NaN.
    """
    n = len(identities)
    sim_matrix = np.full((n, n), np.nan)

    monkey_to_group, group_matrices = _build_monkey_lookup(
        interactions, behavior_type, symmetrize, log_transform)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]

            if id_i not in monkey_to_group or id_j not in monkey_to_group:
                continue

            group_i, idx_i = monkey_to_group[id_i]
            group_j, idx_j = monkey_to_group[id_j]

            if group_i != group_j:
                continue

            matrix, _ = group_matrices[group_i]
            sim_val = matrix[idx_i, idx_j]
            sim_matrix[i, j] = sim_matrix[j, i] = sim_val

    return sim_matrix


def build_combined_interaction_similarity_matrix(identities, interactions,
                                                  behavior_types=('affiliation', 'agonism', 'submission'),
                                                  symmetrize=False, log_transform=False):
    """
    Build a combined similarity matrix by summing (or averaging) raw interaction
    counts across behavior types.

    Returns
    -------
    sim_matrix : ndarray (n, n)
        Sum of interaction counts across all behavior types. NaN for cross-group.
    """
    n = len(identities)
    combined = np.full((n, n), np.nan)
    count_matrix = np.zeros((n, n))  # track how many behavior types contributed

    for btype in behavior_types:
        sim = build_interaction_similarity_matrix(
            identities, interactions, btype,
            symmetrize=symmetrize, log_transform=log_transform)

        valid = ~np.isnan(sim)
        # Initialize combined entries on first valid behavior type
        for i in range(n):
            for j in range(i + 1, n):
                if valid[i, j]:
                    if np.isnan(combined[i, j]):
                        combined[i, j] = combined[j, i] = 0.0
                    combined[i, j] += sim[i, j]
                    combined[j, i] += sim[j, i]
                    count_matrix[i, j] += 1
                    count_matrix[j, i] += 1

    return combined


def build_all_similarity_matrices(identities, interactions,
                                   behavior_types=('affiliation', 'agonism', 'submission'),
                                   symmetrize=False, log_transform=False):
    """
    Build social similarity matrices for each behavior type + combined.

    Returns
    -------
    social_sims : dict {name: ndarray (n, n)}
    """
    social_sims = {}

    for btype in behavior_types:
        sim = build_interaction_similarity_matrix(
            identities, interactions, btype,
            symmetrize=symmetrize, log_transform=log_transform)
        social_sims[f'{btype}'] = sim

    if len(behavior_types) > 1:
        combined = build_combined_interaction_similarity_matrix(
            identities, interactions, behavior_types=behavior_types,
            symmetrize=symmetrize, log_transform=log_transform)
        social_sims['combined'] = combined

    return social_sims


# ──────────────────────────────────────────────────────────
# Bootstrap CI helper
# ──────────────────────────────────────────────────────────

def _bootstrap_rho_ci(v_neural, v_social, n_bootstrap, rng_seed=42,
                       ci_level=0.95):
    """
    Bootstrap confidence interval for Spearman ρ.

    Resamples (neural, social) pairs with replacement and recomputes ρ.

    Returns
    -------
    ci_low, ci_high, boot_distribution
    """
    rng = np.random.default_rng(rng_seed + 999)  # offset from permutation rng
    n = len(v_neural)
    boot_rhos = np.zeros(n_bootstrap)

    for bi in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(v_neural[idx], v_social[idx])
        boot_rhos[bi] = r

    alpha = 1.0 - ci_level
    ci_low = float(np.nanpercentile(boot_rhos, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(boot_rhos, 100 * (1 - alpha / 2)))
    return ci_low, ci_high, boot_rhos


def compare_similarity_matrices(neural_sim, social_sims, n_permutations=0,
                                 rng_seed=42):
    """
    Spearman correlation between neural similarity matrix and social similarity
    matrices. NaN entries are masked out.

    Parameters
    ----------
    neural_sim : ndarray (n, n)  — Pearson r values
    social_sims : dict {name: ndarray (n, n)}  — raw interaction counts

    Returns
    -------
    comparisons : dict {name: {rho, p_val, n_pairs, n_total_pairs, null_distribution}}
    """
    comparisons = {}
    for name, social_sim in social_sims.items():
        v_neural = _upper_triangle(neural_sim)
        v_social = _upper_triangle(social_sim)

        mask = ~(np.isnan(v_neural) | np.isnan(v_social))
        n_valid = int(mask.sum())
        n_total = len(v_neural)

        if n_valid < 3:
            comparisons[name] = dict(rho=np.nan, p_val=np.nan, n_pairs=n_valid,
                                     n_total_pairs=n_total, null_distribution=None)
            continue

        rho, _ = spearmanr(v_neural[mask], v_social[mask])

        p_val = None
        null_dist = None
        if n_permutations > 0:
            rng = np.random.default_rng(rng_seed)
            n = neural_sim.shape[0]
            null_dist = np.zeros(n_permutations)

            for perm_i in range(n_permutations):
                perm = rng.permutation(n)
                shuffled = neural_sim[np.ix_(perm, perm)]
                v_shuf = _upper_triangle(shuffled)
                null_dist[perm_i], _ = spearmanr(v_shuf[mask], v_social[mask])

            p_val = np.mean(np.abs(null_dist) >= np.abs(rho))

        comparisons[name] = dict(rho=rho, p_val=p_val, n_pairs=n_valid,
                                  n_total_pairs=n_total, null_distribution=null_dist)

        p_str = f", p={p_val:.4f}" if p_val is not None else ""
        print(f"  {name:30s}  ρ = {rho:+.4f}  "
              f"(n_pairs={n_valid}/{n_total}{p_str})")

    return comparisons


def compare_similarity_by_group(neural_sim, social_sims, identities,
                                 info_df, n_permutations=0, n_bootstrap=0,
                                 rng_seed=42, confound_matrix=None):
    """
    Compare similarity matrices per group.

    Parameters
    ----------
    confound_matrix : ndarray (n, n) or None
        If provided, partial out this matrix (e.g. rank distance) from
        the neural–social correlation within each group.

    Returns
    -------
    group_comparisons : dict {name: {group: {rho, p_val, n_pairs, ...}}}
    """
    group_indices = _get_group_indices(identities, info_df)
    group_comparisons = {}

    for name, social_sim in social_sims.items():
        group_comparisons[name] = {}

        for group_name, indices in group_indices.items():
            if len(indices) < 3:
                group_comparisons[name][group_name] = {
                    'rho': np.nan, 'p_val': np.nan, 'n_pairs': 0,
                    'ci_low': np.nan, 'ci_high': np.nan,
                    'null_distribution': None
                }
                continue

            idx = np.array(indices)
            neural_sub = neural_sim[np.ix_(idx, idx)]
            social_sub = social_sim[np.ix_(idx, idx)]

            v_neural = _upper_triangle(neural_sub)
            v_social = _upper_triangle(social_sub)

            mask = ~(np.isnan(v_neural) | np.isnan(v_social))
            v_confound = None
            if confound_matrix is not None:
                confound_sub = confound_matrix[np.ix_(idx, idx)]
                v_confound = _upper_triangle(confound_sub)
                mask &= ~np.isnan(v_confound)

            n_valid = int(mask.sum())
            min_required = 4 if confound_matrix is not None else 3

            if n_valid < min_required:
                group_comparisons[name][group_name] = {
                    'rho': np.nan, 'p_val': np.nan, 'n_pairs': n_valid,
                    'ci_low': np.nan, 'ci_high': np.nan,
                    'null_distribution': None
                }
                continue

            if v_confound is not None:
                rho = _partial_spearman(v_neural, v_social, v_confound, mask)
            else:
                rho, _ = spearmanr(v_neural[mask], v_social[mask])

            p_val = None
            null_dist = None
            if n_permutations > 0:
                rng = np.random.default_rng(rng_seed)
                n_sub = neural_sub.shape[0]
                null_dist = np.zeros(n_permutations)

                for perm_i in range(n_permutations):
                    perm = rng.permutation(n_sub)
                    shuffled = neural_sub[np.ix_(perm, perm)]
                    v_shuf = _upper_triangle(shuffled)
                    if v_confound is not None:
                        null_dist[perm_i] = _partial_spearman(
                            v_shuf, v_social, v_confound, mask)
                    else:
                        null_dist[perm_i], _ = spearmanr(v_shuf[mask], v_social[mask])

                p_val = np.mean(np.abs(null_dist) >= np.abs(rho))

            ci_low, ci_high = np.nan, np.nan
            if n_bootstrap > 0 and v_confound is None:
                ci_low, ci_high, _ = _bootstrap_rho_ci(
                    v_neural[mask], v_social[mask],
                    n_bootstrap, rng_seed=rng_seed)

            group_comparisons[name][group_name] = {
                'rho': rho, 'p_val': p_val, 'n_pairs': n_valid,
                'ci_low': ci_low, 'ci_high': ci_high,
                'null_distribution': null_dist
            }

    return group_comparisons


# ══════════════════════════════════════════════════════════
#  DISSIMILARITY APPROACHES (original)
# ══════════════════════════════════════════════════════════

def build_behavioral_profile_rdm(identities, interactions, behavior_type,
                                  symmetrize=False, metric='correlation',
                                  log_transform=False, rank_transform=False,
                                  use_columns=False):
    """
    Build a behavioral profile RDM (dissimilarity).

    Parameters
    ----------
    use_columns : bool
        False (default) = row profile: how this monkey directs behavior toward others.
        True = column profile: how this monkey is treated by others.
    rank_transform : bool
        If True, replace each profile vector with within-vector ranks.
        Removes magnitude effects from extreme counts.
    """
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)

    monkey_to_group, group_matrices = _build_monkey_lookup(
        interactions, behavior_type, symmetrize, log_transform)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]

            if id_i not in monkey_to_group or id_j not in monkey_to_group:
                continue

            group_i, idx_i = monkey_to_group[id_i]
            group_j, idx_j = monkey_to_group[id_j]

            if group_i != group_j:
                continue

            matrix, _ = group_matrices[group_i]

            if use_columns:
                profile_i = matrix[:, idx_i].copy()
                profile_j = matrix[:, idx_j].copy()
            else:
                profile_i = matrix[idx_i, :].copy()
                profile_j = matrix[idx_j, :].copy()

            if rank_transform:
                profile_i = rankdata(profile_i).astype(float)
                profile_j = rankdata(profile_j).astype(float)

            d = _compute_pairwise_distance(profile_i, profile_j, metric)
            rdm[i, j] = rdm[j, i] = d

    return rdm


def build_combined_profile_rdm(identities, interactions,
                                behavior_types=('affiliation', 'agonism', 'submission'),
                                symmetrize=False, metric='correlation',
                                log_transform=False, rank_transform=False,
                                use_columns=False):
    """
    Build a combined behavioral profile RDM by concatenating profiles
    across behavior types.

    Parameters
    ----------
    use_columns : bool
        False = row (directed), True = column (received).
    rank_transform : bool
        If True, rank each per-behavior profile before concatenation.
    """
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)

    all_lookups = {}
    for btype in behavior_types:
        monkey_to_group, group_matrices = _build_monkey_lookup(
            interactions, btype, symmetrize, log_transform)
        all_lookups[btype] = (monkey_to_group, group_matrices)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]

            group_name = None
            valid = True
            for btype in behavior_types:
                m2g, gm = all_lookups[btype]
                if id_i not in m2g or id_j not in m2g:
                    valid = False
                    break
                gi, _ = m2g[id_i]
                gj, _ = m2g[id_j]
                if gi != gj:
                    valid = False
                    break
                if group_name is None:
                    group_name = gi
                elif gi != group_name:
                    valid = False
                    break

            if not valid:
                continue

            profile_i_parts = []
            profile_j_parts = []
            for btype in behavior_types:
                m2g, gm = all_lookups[btype]
                _, idx_i = m2g[id_i]
                _, idx_j = m2g[id_j]
                matrix, _ = gm[group_name]

                if use_columns:
                    pi = matrix[:, idx_i].copy()
                    pj = matrix[:, idx_j].copy()
                else:
                    pi = matrix[idx_i, :].copy()
                    pj = matrix[idx_j, :].copy()

                if rank_transform:
                    pi = rankdata(pi).astype(float)
                    pj = rankdata(pj).astype(float)

                profile_i_parts.append(pi)
                profile_j_parts.append(pj)

            combined_i = np.concatenate(profile_i_parts)
            combined_j = np.concatenate(profile_j_parts)

            d = _compute_pairwise_distance(combined_i, combined_j, metric)
            rdm[i, j] = rdm[j, i] = d

    return rdm


def build_interaction_rdm(identities, interactions, behavior_type,
                           symmetrize=False, log_transform=False):
    """
    Raw interaction counts → dissimilarity via max - count.
    """
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    raw_sim = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)
    np.fill_diagonal(raw_sim, 0.0)

    monkey_to_group, group_matrices = _build_monkey_lookup(
        interactions, behavior_type, symmetrize, log_transform)

    group_max = {}
    for group_name, (matrix, monkey_ids) in group_matrices.items():
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        group_max[group_name] = np.max(matrix[mask])

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]

            if id_i not in monkey_to_group or id_j not in monkey_to_group:
                continue

            group_i, idx_i = monkey_to_group[id_i]
            group_j, idx_j = monkey_to_group[id_j]

            if group_i != group_j:
                continue

            matrix, _ = group_matrices[group_i]
            sim_val = matrix[idx_i, idx_j]
            raw_sim[i, j] = raw_sim[j, i] = sim_val

            dissim = group_max[group_i] - sim_val
            rdm[i, j] = rdm[j, i] = dissim

    return rdm, raw_sim


def build_social_rdms(identities, interactions, behavior_types=None,
                       symmetrize=False, profile_metric='correlation',
                       log_transform=False, include_combined=True,
                       rank_transform=False):
    """
    Build dissimilarity-based social RDMs.

    Builds two profile types per behavior:
      - {btype}_profile:   row-based (how this monkey directs behavior toward others)
      - {btype}_received:  column-based (how this monkey is treated by others)
    When symmetrize=True, received profiles are skipped (identical to directed).

    Parameters
    ----------
    rank_transform : bool
        If True, rank-transform each profile vector before computing distances.
    """
    if behavior_types is None:
        all_types = set()
        for group_data in interactions.values():
            all_types.update(group_data.keys())
        behavior_types = sorted(all_types)

    # When symmetrized, rows == columns, so received profiles are redundant
    build_received = not symmetrize

    social_rdms = {}

    for btype in behavior_types:
        # Directed profile (rows): how this monkey behaves toward others
        social_rdms[f'{btype}_profile'] = build_behavioral_profile_rdm(
            identities, interactions, btype,
            symmetrize=symmetrize, metric=profile_metric,
            log_transform=log_transform, rank_transform=rank_transform,
            use_columns=False)

        # Received profile (columns): how this monkey is treated by others
        if build_received:
            social_rdms[f'{btype}_received'] = build_behavioral_profile_rdm(
                identities, interactions, btype,
                symmetrize=symmetrize, metric=profile_metric,
                log_transform=log_transform, rank_transform=rank_transform,
                use_columns=True)

        # --- interaction approach commented out (uncomment to include) ---
        # interaction_rdm, _ = build_interaction_rdm(
        #     identities, interactions, btype,
        #     symmetrize=symmetrize, log_transform=log_transform)
        # social_rdms[f'{btype}_interaction'] = interaction_rdm

    if include_combined and len(behavior_types) > 1:
        social_rdms['combined_profile'] = build_combined_profile_rdm(
            identities, interactions,
            behavior_types=behavior_types,
            symmetrize=symmetrize, metric=profile_metric,
            log_transform=log_transform, rank_transform=rank_transform,
            use_columns=False)

        if build_received:
            social_rdms['combined_received'] = build_combined_profile_rdm(
                identities, interactions,
                behavior_types=behavior_types,
                symmetrize=symmetrize, metric=profile_metric,
                log_transform=log_transform, rank_transform=rank_transform,
                use_columns=True)

    return social_rdms


# ──────────────────────────────────────────────────────────
# Dissimilarity-based comparisons (original)
# ──────────────────────────────────────────────────────────

def compare_neural_to_social(neural_rdm, social_rdms, n_permutations=0,
                              rng_seed=42):
    """Compare neural RDM to each social RDM (both dissimilarity)."""
    from rsa_core import compare_rdms

    comparisons = {}
    for name, social_rdm in social_rdms.items():
        result = compare_rdms(neural_rdm, social_rdm,
                              n_permutations=n_permutations,
                              rng_seed=rng_seed)

        v_neural = _upper_triangle(neural_rdm)
        v_social = _upper_triangle(social_rdm)
        mask = ~(np.isnan(v_neural) | np.isnan(v_social))
        result['n_pairs'] = int(mask.sum())
        result['n_total_pairs'] = len(v_neural)

        comparisons[name] = result
        p_str = f", p={result['p_val']:.4f}" if result['p_val'] is not None else ""
        print(f"  {name:30s}  ρ = {result['rho']:+.4f}  "
              f"(n_pairs={result['n_pairs']}/{result['n_total_pairs']}{p_str})")

    return comparisons


def compare_neural_to_social_by_group(neural_rdm, social_rdms, identities,
                                       info_df, n_permutations=0, n_bootstrap=0,
                                       rng_seed=42, confound_matrix=None):
    """Compare neural RDM to each social RDM per group (both dissimilarity).

    Parameters
    ----------
    confound_matrix : ndarray (n, n) or None
        If provided, partial out this matrix (e.g. rank distance).
    """
    group_indices = _get_group_indices(identities, info_df)
    group_comparisons = {}

    for rdm_name, social_rdm in social_rdms.items():
        group_comparisons[rdm_name] = {}

        for group_name, indices in group_indices.items():
            if len(indices) < 3:
                group_comparisons[rdm_name][group_name] = {
                    'rho': np.nan, 'p_val': np.nan, 'n_pairs': 0,
                    'ci_low': np.nan, 'ci_high': np.nan,
                    'null_distribution': None
                }
                continue

            idx = np.array(indices)
            neural_sub = neural_rdm[np.ix_(idx, idx)]
            social_sub = social_rdm[np.ix_(idx, idx)]

            v_neural = _upper_triangle(neural_sub)
            v_social = _upper_triangle(social_sub)

            mask = ~(np.isnan(v_neural) | np.isnan(v_social))
            v_confound = None
            if confound_matrix is not None:
                confound_sub = confound_matrix[np.ix_(idx, idx)]
                v_confound = _upper_triangle(confound_sub)
                mask &= ~np.isnan(v_confound)

            n_valid = int(mask.sum())
            min_required = 4 if confound_matrix is not None else 3

            if n_valid < min_required:
                group_comparisons[rdm_name][group_name] = {
                    'rho': np.nan, 'p_val': np.nan, 'n_pairs': n_valid,
                    'ci_low': np.nan, 'ci_high': np.nan,
                    'null_distribution': None
                }
                continue

            if v_confound is not None:
                rho = _partial_spearman(v_neural, v_social, v_confound, mask)
            else:
                rho, _ = spearmanr(v_neural[mask], v_social[mask])

            p_val = None
            null_dist = None
            if n_permutations > 0:
                rng = np.random.default_rng(rng_seed)
                n_sub = neural_sub.shape[0]
                null_dist = np.zeros(n_permutations)

                for perm_i in range(n_permutations):
                    perm = rng.permutation(n_sub)
                    shuffled = neural_sub[np.ix_(perm, perm)]
                    v_shuf = _upper_triangle(shuffled)
                    if v_confound is not None:
                        null_dist[perm_i] = _partial_spearman(
                            v_shuf, v_social, v_confound, mask)
                    else:
                        null_dist[perm_i], _ = spearmanr(v_shuf[mask], v_social[mask])

                p_val = np.mean(np.abs(null_dist) >= np.abs(rho))

            ci_low, ci_high = np.nan, np.nan
            if n_bootstrap > 0 and v_confound is None:
                ci_low, ci_high, _ = _bootstrap_rho_ci(
                    v_neural[mask], v_social[mask],
                    n_bootstrap, rng_seed=rng_seed)

            group_comparisons[rdm_name][group_name] = {
                'rho': rho, 'p_val': p_val, 'n_pairs': n_valid,
                'ci_low': ci_low, 'ci_high': ci_high,
                'null_distribution': null_dist
            }

    return group_comparisons


# ──────────────────────────────────────────────────────────
# Between-group Δρ permutation test
# ──────────────────────────────────────────────────────────

def compare_between_groups(neural_mat, social_mats, identities, info_df,
                           n_permutations=0, rng_seed=42):
    """
    Test whether RSA correlation (Spearman ρ) differs between groups.

    For each pair of groups and each social matrix:
    - Compute observed Δρ = ρ_groupA − ρ_groupB
    - Permutation test: pool (neural, social) pairs from both groups,
      shuffle group assignment, recompute Δρ.

    Works identically for similarity matrices (Pearson r vs counts)
    and dissimilarity matrices (1−r vs social RDM).

    Parameters
    ----------
    neural_mat : ndarray (n, n)
    social_mats : dict {name: ndarray (n, n)}
    identities : list of str
    info_df : DataFrame
    n_permutations : int
    rng_seed : int

    Returns
    -------
    between_comparisons : dict
        {mat_name: {(groupA, groupB): {delta_rho, p_val, rho_a, rho_b,
                                        n_pairs_a, n_pairs_b}}}
    """
    group_indices = _get_group_indices(identities, info_df)
    groups = sorted(group_indices.keys())

    between_comparisons = {}

    for mat_name, social_mat in social_mats.items():
        between_comparisons[mat_name] = {}

        for gi, g_a in enumerate(groups):
            for g_b in groups[gi + 1:]:
                idx_a = np.array(group_indices[g_a])
                idx_b = np.array(group_indices[g_b])

                # Extract valid (neural, social) pairs per group
                def _get_valid_pairs(idx):
                    nsub = neural_mat[np.ix_(idx, idx)]
                    ssub = social_mat[np.ix_(idx, idx)]
                    vn = _upper_triangle(nsub)
                    vs = _upper_triangle(ssub)
                    mask = ~(np.isnan(vn) | np.isnan(vs))
                    return vn[mask], vs[mask]

                vn_a, vs_a = _get_valid_pairs(idx_a)
                vn_b, vs_b = _get_valid_pairs(idx_b)

                if len(vn_a) < 3 or len(vn_b) < 3:
                    between_comparisons[mat_name][(g_a, g_b)] = {
                        'delta_rho': np.nan, 'p_val': np.nan,
                        'rho_a': np.nan, 'rho_b': np.nan,
                        'n_pairs_a': len(vn_a), 'n_pairs_b': len(vn_b),
                    }
                    continue

                rho_a, _ = spearmanr(vn_a, vs_a)
                rho_b, _ = spearmanr(vn_b, vs_b)
                delta = rho_a - rho_b

                p_val = None
                if n_permutations > 0:
                    all_neural = np.concatenate([vn_a, vn_b])
                    all_social = np.concatenate([vs_a, vs_b])
                    n_a = len(vn_a)

                    rng = np.random.default_rng(rng_seed)
                    null_deltas = np.zeros(n_permutations)

                    for pi in range(n_permutations):
                        perm = rng.permutation(len(all_neural))
                        pn_a = all_neural[perm[:n_a]]
                        ps_a = all_social[perm[:n_a]]
                        pn_b = all_neural[perm[n_a:]]
                        ps_b = all_social[perm[n_a:]]

                        r_a, _ = spearmanr(pn_a, ps_a)
                        r_b, _ = spearmanr(pn_b, ps_b)
                        null_deltas[pi] = r_a - r_b

                    p_val = float(np.mean(np.abs(null_deltas) >= np.abs(delta)))

                between_comparisons[mat_name][(g_a, g_b)] = {
                    'delta_rho': delta, 'p_val': p_val,
                    'rho_a': rho_a, 'rho_b': rho_b,
                    'n_pairs_a': len(vn_a), 'n_pairs_b': len(vn_b),
                }

    return between_comparisons


# ──────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────

def print_between_group_comparisons(between_comparisons):
    """Pretty-print between-group Δρ results."""
    mat_names = list(between_comparisons.keys())
    if not mat_names:
        return

    all_pairs = set()
    for bc in between_comparisons.values():
        all_pairs.update(bc.keys())
    all_pairs = sorted(all_pairs)

    pair_labels = [f"{a}−{b}" for a, b in all_pairs]
    pair_header = "".join(f"{lbl:>30s}" for lbl in pair_labels)
    print(f"\n  {'Matrix':<30s}{pair_header}")
    print(f"  {'-' * (30 + 30 * len(all_pairs))}")

    for mat_name in mat_names:
        row = f"  {mat_name:<30s}"
        for pair in all_pairs:
            bc = between_comparisons[mat_name].get(pair, {})
            d = bc.get('delta_rho', np.nan)
            p = bc.get('p_val')
            if np.isnan(d):
                row += f"{'--':>30s}"
            else:
                p_str = f" p={p:.3f}" if p is not None else ""
                row += f"  Δρ={d:>+.4f}{p_str:>16s}"
        print(row)

    print()


def print_group_comparisons(group_comparisons):
    """Pretty-print per-group RSA results."""
    rdm_names = list(group_comparisons.keys())
    all_groups = set()
    for gc in group_comparisons.values():
        all_groups.update(gc.keys())
    all_groups = sorted(all_groups)

    # Check if any result has CIs
    has_ci = False
    for gc in group_comparisons.values():
        for entry in gc.values():
            if not np.isnan(entry.get('ci_low', np.nan)):
                has_ci = True
                break
        if has_ci:
            break

    col_width = 40 if has_ci else 25
    group_header = "".join(f"{g:>{col_width}s}" for g in all_groups)
    print(f"\n  {'RDM/RSM':<30s}{group_header}")
    print(f"  {'-' * (30 + col_width * len(all_groups))}")

    for rdm_name in rdm_names:
        row = f"  {rdm_name:<30s}"
        for group in all_groups:
            gc = group_comparisons[rdm_name].get(group, {})
            rho = gc.get('rho', np.nan)
            p = gc.get('p_val')
            n = gc.get('n_pairs', 0)
            ci_lo = gc.get('ci_low', np.nan)
            ci_hi = gc.get('ci_high', np.nan)
            if np.isnan(rho):
                row += f"{'--':>{col_width}s}"
            else:
                p_str = f" p={p:.3f}" if p is not None else ""
                ci_str = f" [{ci_lo:+.3f},{ci_hi:+.3f}]" if not np.isnan(ci_lo) else ""
                row += f"  {rho:>+.4f} (n={n:>2d}){p_str}{ci_str}"
        print(row)

    print()


# ──────────────────────────────────────────────────────────
# Scatter plots
# ──────────────────────────────────────────────────────────

def plot_scatter_multi(neural_mat, social_mats, identities, info_df,
                        group_colors, neural_label='Neural similarity (r)',
                        social_label_prefix='Social',
                        title='Neural vs Social', save_path=None):
    """
    Multi-panel scatter plot. Works for both similarity and dissimilarity matrices.

    Parameters
    ----------
    neural_mat : ndarray (n, n)
    social_mats : dict {name: ndarray (n, n)}
    neural_label : str — y-axis label
    social_label_prefix : str — prefix for x-axis
    """
    mat_names = list(social_mats.keys())
    n_panels = len(mat_names)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    group_indices = _get_group_indices(identities, info_df)

    for ax_idx, (mat_name, social_mat) in enumerate(social_mats.items()):
        ax = axes[ax_idx]

        for group_name, indices in group_indices.items():
            if len(indices) < 2:
                continue

            idx = np.array(indices)
            neural_sub = neural_mat[np.ix_(idx, idx)]
            social_sub = social_mat[np.ix_(idx, idx)]

            v_neural = _upper_triangle(neural_sub)
            v_social = _upper_triangle(social_sub)

            mask = ~(np.isnan(v_neural) | np.isnan(v_social))
            if mask.sum() == 0:
                continue

            color = group_colors.get(group_name, 'gray')
            ax.scatter(v_social[mask], v_neural[mask],
                       color=color, alpha=0.6, s=25, edgecolors='k', lw=0.2,
                       label=group_name, zorder=5)

        # Overall rho
        v_neural_all = _upper_triangle(neural_mat)
        v_social_all = _upper_triangle(social_mat)
        mask_all = ~(np.isnan(v_neural_all) | np.isnan(v_social_all))
        if mask_all.sum() >= 3:
            rho, _ = spearmanr(v_social_all[mask_all], v_neural_all[mask_all])
            from numpy.polynomial.polynomial import polyfit
            b, m = polyfit(v_social_all[mask_all], v_neural_all[mask_all], 1)
            x_range = np.array([np.nanmin(v_social_all[mask_all]),
                                np.nanmax(v_social_all[mask_all])])
            ax.plot(x_range, b + m * x_range, 'k--', alpha=0.4, lw=1)
            ax.set_title(f"{mat_name}\nρ = {rho:.3f}", fontsize=9)
        else:
            ax.set_title(mat_name, fontsize=9)

        ax.set_xlabel(f"{social_label_prefix}: {mat_name}", fontsize=8)
        ax.set_ylabel(neural_label, fontsize=8)
        ax.tick_params(labelsize=7)

    if len(axes) > 0:
        axes[0].legend(fontsize=7, loc='best')

    for ax_idx in range(n_panels, len(axes)):
        axes[ax_idx].axis('off')

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
