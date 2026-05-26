# rsa_utils.py
"""
Shared helpers used across the RSA modules.

Centralizes:
  - _upper_triangle  : flatten a square matrix's upper triangle (k=1)
  - partial_spearman : partial Spearman ρ via rank-OLS residuals
  - stars            : p-value → significance stars (***, **, *)
  - permute_matrix   : RDM row/col permutation
  - finite_mask      : ~(NaN | NaN | ...) across N vectors
"""
import numpy as np
from scipy.stats import rankdata


def upper_triangle(mat):
    """Flat 1-D vector of upper triangle, excluding diagonal."""
    n = mat.shape[0]
    return mat[np.triu_indices(n, k=1)]


# Backwards-compat private alias (some modules imported the underscore name)
_upper_triangle = upper_triangle


def finite_mask(*vectors):
    """Boolean mask: True where every input vector is finite (not NaN)."""
    mask = np.ones_like(vectors[0], dtype=bool)
    for v in vectors:
        mask &= ~np.isnan(v)
    return mask


def permute_rdm(rdm, perm):
    """Return rdm[perm, :][:, perm] — symmetric row+col permutation."""
    return rdm[np.ix_(perm, perm)]


def partial_spearman(v_x, v_y, v_confounds, mask):
    """
    Partial Spearman ρ between v_x and v_y, controlling for one or more
    confound vectors.

    Method: rank-transform masked values, OLS-regress confounds out of both,
    Pearson-correlate residuals.

    Parameters
    ----------
    v_x, v_y    : 1-D ndarray (same length as `mask`)
    v_confounds : single 1-D ndarray, list of 1-D ndarrays, or None
    mask        : boolean array selecting the entries to use

    Returns
    -------
    float (partial ρ)  — 0.0 if either residual is degenerate.
    """
    if v_confounds is None:
        v_confounds = []
    elif isinstance(v_confounds, np.ndarray):
        v_confounds = [v_confounds]

    r_x = rankdata(v_x[mask])
    r_y = rankdata(v_y[mask])

    cols = [np.ones(mask.sum())]
    for vc in v_confounds:
        cols.append(rankdata(vc[mask]))
    C = np.column_stack(cols)

    betas_x, *_ = np.linalg.lstsq(C, r_x, rcond=None)
    resid_x = r_x - C @ betas_x
    betas_y, *_ = np.linalg.lstsq(C, r_y, rcond=None)
    resid_y = r_y - C @ betas_y

    if np.std(resid_x) < 1e-12 or np.std(resid_y) < 1e-12:
        return 0.0
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def stars(p):
    """p → '***' (<.001), '**' (<.01), '*' (<.05), else ''."""
    if p is None:
        return ''
    if isinstance(p, float) and np.isnan(p):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''
