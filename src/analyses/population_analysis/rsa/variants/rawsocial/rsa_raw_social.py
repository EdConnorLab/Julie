# rsa_raw_social.py
"""
RAW social matrix RDMs (no symmetrization).

The standard pipeline (rsa_social.build_social_rdms) symmetrizes the
interaction matrix via (M + M.T)/2 and then collapses to row marginals
("profile") or column marginals ("received"). Both are forms of
marginalization that discard the directed asymmetry M[i,j] ≠ M[j,i].

This module provides two RAW alternatives that preserve direction:

  1. Pair-level signed asymmetry RDM
     For pair (i,j), entry = M[i,j] - M[j,i]. Captures how unbalanced
     the relationship is (and which direction it flows). Antisymmetric
     across the diagonal; the upper triangle carries signed values.
     Not a metric — feed directly into Spearman ρ against the neural
     RDM upper triangle.

  2. Per-monkey concat[row, col] vector RDM
     Each monkey i is represented by the vector
         concat(M[i, :≠i], M[:≠i, i])
     of length 2(n-1), preserving both outgoing and incoming
     interactions without averaging them. Pair dissimilarity =
     correlation distance (or any metric supported by
     _compute_pairwise_distance).

Both work per-group (cross-group entries are NaN).
"""

import numpy as np
from scipy.stats import rankdata

from analyses.population_analysis.rsa.social.rsa_social import (
    _build_monkey_lookup,
    _compute_pairwise_distance,
)


# ──────────────────────────────────────────────────────────
# 1. Pair-level signed asymmetry
# ──────────────────────────────────────────────────────────

def build_signed_asymmetry_rdm(identities, interactions, behavior_type,
                                log_transform=False, antisymmetric=True):
    """
    Pair-level signed asymmetry: entry[i,j] = M[i,j] - M[j,i].

    Parameters
    ----------
    identities : list of str
    interactions : dict
    behavior_type : str
    log_transform : bool
        If True, log1p the raw matrix before differencing.
    antisymmetric : bool
        If True (default), entry[j,i] = -entry[i,j] (antisymmetric).
        If False, entry[j,i] = entry[i,j] (mirror; treats asymmetry as
        unsigned magnitude). Spearman ρ on the upper triangle is the
        same either way since only [i<j] entries are used.

    Returns
    -------
    rdm : ndarray (n, n)
        Diagonal = 0. Cross-group pairs = NaN.
    """
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)

    # symmetrize=False so the raw matrix is preserved
    monkey_to_group, group_matrices = _build_monkey_lookup(
        interactions, behavior_type,
        symmetrize=False, log_transform=log_transform)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]
            if id_i not in monkey_to_group or id_j not in monkey_to_group:
                continue
            g_i, idx_i = monkey_to_group[id_i]
            g_j, idx_j = monkey_to_group[id_j]
            if g_i != g_j:
                continue

            mat, _ = group_matrices[g_i]
            diff = float(mat[idx_i, idx_j] - mat[idx_j, idx_i])
            rdm[i, j] = diff
            rdm[j, i] = -diff if antisymmetric else diff

    return rdm


def build_signed_asymmetry_rdms(identities, interactions,
                                 behavior_types=('affiliation', 'agonism', 'submission'),
                                 log_transform=False, include_combined=True):
    """
    Build one signed-asymmetry RDM per behavior type, plus optional
    summed "combined" RDM (sum of per-behavior signed asymmetries).

    Returns
    -------
    rdms : dict {f'{btype}_signed_asym': ndarray (n,n), ...,
                 'combined_signed_asym': ndarray (n,n)}
    """
    out = {}
    per_btype = []
    for btype in behavior_types:
        m = build_signed_asymmetry_rdm(identities, interactions, btype,
                                        log_transform=log_transform)
        out[f'{btype}_signed_asym'] = m
        per_btype.append(m)

    if include_combined and len(per_btype) > 1:
        # Sum signed asymmetries across behaviors (preserve signs and NaNs)
        stack = np.stack(per_btype, axis=0)
        with np.errstate(invalid='ignore'):
            combined = np.nansum(stack, axis=0)
        # Restore NaN where ALL inputs were NaN
        all_nan = np.all(np.isnan(stack), axis=0)
        combined[all_nan] = np.nan
        np.fill_diagonal(combined, 0.0)
        out['combined_signed_asym'] = combined

    return out


# ──────────────────────────────────────────────────────────
# 2. Per-monkey concat[row, col] vector RDM
# ──────────────────────────────────────────────────────────

def _directed_vector(matrix, idx):
    """concat(row[i, :≠i], col[:≠i, i]) of length 2(n-1)."""
    n = matrix.shape[0]
    keep = np.arange(n) != idx
    row = matrix[idx, :][keep]
    col = matrix[:, idx][keep]
    return np.concatenate([row, col])


def build_concat_directed_rdm(identities, interactions, behavior_type,
                               metric='correlation', log_transform=False,
                               rank_transform=False):
    """
    Per-monkey vector = concat(row, col); pairwise distance between
    vectors gives the RDM. No symmetrization.

    Parameters
    ----------
    metric : str  ('correlation', 'euclidean', 'cosine')
    log_transform : bool
    rank_transform : bool
        If True, rank-transform each monkey's concatenated vector
        before distance.

    Returns
    -------
    rdm : ndarray (n, n)
    """
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)

    monkey_to_group, group_matrices = _build_monkey_lookup(
        interactions, behavior_type,
        symmetrize=False, log_transform=log_transform)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]
            if id_i not in monkey_to_group or id_j not in monkey_to_group:
                continue
            g_i, idx_i = monkey_to_group[id_i]
            g_j, idx_j = monkey_to_group[id_j]
            if g_i != g_j:
                continue

            mat, _ = group_matrices[g_i]
            vi = _directed_vector(mat, idx_i)
            vj = _directed_vector(mat, idx_j)
            if rank_transform:
                vi = rankdata(vi).astype(float)
                vj = rankdata(vj).astype(float)

            d = _compute_pairwise_distance(vi, vj, metric)
            rdm[i, j] = rdm[j, i] = d

    return rdm


def build_concat_directed_combined_rdm(identities, interactions,
                                        behavior_types=('affiliation', 'agonism', 'submission'),
                                        metric='correlation', log_transform=False,
                                        rank_transform=False):
    """Concatenate concat[row,col] vectors across behavior types."""
    n = len(identities)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)

    lookups = {}
    for btype in behavior_types:
        lookups[btype] = _build_monkey_lookup(
            interactions, btype, symmetrize=False, log_transform=log_transform)

    for i in range(n):
        for j in range(i + 1, n):
            id_i, id_j = identities[i], identities[j]

            group_name = None
            valid = True
            for btype in behavior_types:
                m2g, _ = lookups[btype]
                if id_i not in m2g or id_j not in m2g:
                    valid = False
                    break
                g_i, _ = m2g[id_i]
                g_j, _ = m2g[id_j]
                if g_i != g_j:
                    valid = False
                    break
                if group_name is None:
                    group_name = g_i
                elif g_i != group_name:
                    valid = False
                    break
            if not valid:
                continue

            parts_i = []
            parts_j = []
            for btype in behavior_types:
                m2g, gm = lookups[btype]
                _, idx_i = m2g[id_i]
                _, idx_j = m2g[id_j]
                mat, _ = gm[group_name]
                vi = _directed_vector(mat, idx_i)
                vj = _directed_vector(mat, idx_j)
                if rank_transform:
                    vi = rankdata(vi).astype(float)
                    vj = rankdata(vj).astype(float)
                parts_i.append(vi)
                parts_j.append(vj)

            ci = np.concatenate(parts_i)
            cj = np.concatenate(parts_j)
            rdm[i, j] = rdm[j, i] = _compute_pairwise_distance(ci, cj, metric)

    return rdm


def build_concat_directed_rdms(identities, interactions,
                                behavior_types=('affiliation', 'agonism', 'submission'),
                                metric='correlation', log_transform=False,
                                rank_transform=False, include_combined=True):
    """One concat[row,col] RDM per behavior type + optional combined."""
    out = {}
    for btype in behavior_types:
        out[f'{btype}_concat_dir'] = build_concat_directed_rdm(
            identities, interactions, btype,
            metric=metric, log_transform=log_transform,
            rank_transform=rank_transform)
    if include_combined and len(behavior_types) > 1:
        out['combined_concat_dir'] = build_concat_directed_combined_rdm(
            identities, interactions,
            behavior_types=list(behavior_types),
            metric=metric, log_transform=log_transform,
            rank_transform=rank_transform)
    return out


# ──────────────────────────────────────────────────────────
# Convenience: build BOTH raw forms in one call
# ──────────────────────────────────────────────────────────

def build_raw_social_rdms(identities, interactions,
                           behavior_types=('affiliation', 'agonism', 'submission'),
                           log_transform=False, rank_transform=False,
                           concat_metric='correlation', include_combined=True):
    """
    Build both raw-matrix RDM families:
      - signed asymmetry (pair-level)
      - concat[row,col] directed vector (per-monkey)

    Returns one merged dict that can be passed to
    rsa_social.compare_neural_to_social_by_group.
    """
    rdms = {}
    rdms.update(build_signed_asymmetry_rdms(
        identities, interactions,
        behavior_types=list(behavior_types),
        log_transform=log_transform,
        include_combined=include_combined))
    rdms.update(build_concat_directed_rdms(
        identities, interactions,
        behavior_types=list(behavior_types),
        metric=concat_metric, log_transform=log_transform,
        rank_transform=rank_transform,
        include_combined=include_combined))
    return rdms
