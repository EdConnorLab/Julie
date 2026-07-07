"""
Common setup shared by all figure scripts and replication code.

Edit DEFAULT_DATA_FILE below to point to your .xlsx file, OR override it inside
each figure script (each one has its own DATA_FILE at the top).

Contains:
- Constants (monkey names, subject index, hardcoded column indices, behavior matrices)
- Data loading (build X_mean from xlsx)
- R^2 computation (vectorized + scalar)
- Helper builders for valid sinks and y vectors per (behavior, source)
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path

# ============================================================
# DEFAULT DATA PATH - change this OR override DATA_FILE in each fig script
# ============================================================
# Looks for the file next to this script. Replace with an absolute path if needed.
DEFAULT_DATA_FILE = str(
  '/home/connorlab/Documents/GitHub/Julie/Cortana/old/Ed and ANOVA/used_for_R01'
  '/zombies_spike_counts_for_all_anova_passed_time_windowed_cells_old--usedforgrant.xlsx'
)

# ============================================================
# CONSTANTS - DO NOT CHANGE unless you also change the data
# ============================================================
NMONKEYS = 10
SUBJECT = 6                                                                       # 81G
MONKEY_NAME = ['7124','69X','72X','94B','110E','67G','81G','143H','87J','151J']
BEH_NAMES = ['aff_to','aff_from','sub_to','sub_from','agn_to','agn_from']

# Behavior matrices - copied verbatim from the social-data xlsx files
AFF_TO = np.array([
    [0,19,9,38,84,27,13,14,4,9],[15,0,9,41,4,13,19,71,12,0],[18,10,0,21,7,17,49,18,3,1],
    [38,43,24,0,18,6,31,26,29,4],[90,3,8,8,0,22,8,9,1,18],[23,12,17,1,23,0,23,16,3,10],
    [17,18,43,34,10,23,0,34,2,9],[11,70,18,17,8,17,33,0,5,3],[3,6,4,31,2,1,2,4,0,9],
    [8,0,0,1,26,8,2,1,7,0]], dtype=float)
SUB_TO = np.array([
    [0,0,0,1,0,1,0,0,2,0],[4,0,0,13,0,0,0,2,1,0],[9,2,0,10,0,0,4,3,0,0],
    [7,0,2,0,0,0,0,1,6,0],[11,8,3,5,0,2,4,2,4,18],[90,16,14,43,0,0,10,7,9,6],
    [29,12,1,17,0,0,0,23,17,1],[41,6,1,28,0,0,2,0,5,0],[16,2,0,8,0,1,0,0,0,2],
    [7,2,3,1,2,0,5,2,5,0]], dtype=float)
AGN_TO = np.array([
    [0,4,2,7,4,7,3,9,3,2],[0,0,0,2,1,1,8,4,2,0],[0,1,0,0,0,7,0,1,0,2],
    [1,8,0,0,9,5,6,10,11,2],[0,0,0,0,0,1,0,1,3,10],[0,0,0,0,2,0,0,0,3,1],
    [0,0,3,2,0,4,0,1,0,6],[0,1,1,0,1,0,7,0,0,1],[0,0,0,2,2,5,2,3,0,6],
    [0,0,0,0,5,4,1,0,6,0]], dtype=float)
AFF_FROM = AFF_TO.T.copy()
SUB_FROM = SUB_TO.T.copy()
AGN_FROM = AGN_TO.T.copy()
ALL_BEH = [AFF_TO, AFF_FROM, SUB_TO, SUB_FROM, AGN_TO, AGN_FROM]

# Mapping from "k" (0..8 across non-subject monkeys) to full monkey index (0..9)
K_TO_FULL = [0,1,2,3,4,5,7,8,9]
FULL_TO_K = {f:k for k,f in enumerate(K_TO_FULL)}


# ============================================================
# DATA LOADING
# ============================================================
def get_monkey_columns(df):
    """Auto-detect column indices for the 9 non-subject monkeys."""
    non_subject = [m for i, m in enumerate(MONKEY_NAME) if i != SUBJECT]
    return [df.columns.get_loc(m) for m in non_subject]


def load_data(path=None, ncells=74):
    """
    Load neural data and build the X_mean matrix.

    Args:
        path: Path to xlsx file. If None, uses DEFAULT_DATA_FILE.
        ncells: Number of rows to use (74 = matches his script exactly).
                Set to None or <=0 to use all rows.

    Returns:
        X_mean: (ncells, 9) ndarray of mean spike counts.
                Columns ordered by k: [7124, 69X, 72X, 94B, 110E, 67G, 143H, 87J, 151J]
        df: pandas DataFrame (in case caller wants metadata)
    """
    if path is None:
        path = DEFAULT_DATA_FILE
    df = pd.read_excel(path)
    if ncells is None or ncells <= 0:
        ncells = len(df)
    monkey_cols = get_monkey_columns(df)
    arr = df.values
    X_mean = np.zeros((ncells, 9))
    for i in range(ncells):
        for k in range(9):
            s = arr[i, monkey_cols[k]]
            nums = [int(t) for t in re.findall(r'\b\d+\b', str(s))]
            X_mean[i, k] = sum(nums) / len(nums) if nums else 0.0
    return X_mean, df.iloc[:ncells]


# ============================================================
# REGRESSION HELPERS
# ============================================================
def build_y_per(valid_k_per_source):
    """Build dict {(behavior_idx, source_idx): y_vector}."""
    y_per = {}
    for ib, B in enumerate(ALL_BEH):
        for source in range(NMONKEYS):
            fs = [K_TO_FULL[k] for k in valid_k_per_source[source]]
            y_per[(ib, source)] = np.array([B[source, f] for f in fs], dtype=float)
    return y_per


def build_valid_k_per_source():
    """For each source monkey, the list of k-indices (into non-subject monkey ordering)
    that are valid sinks (excludes source and subject)."""
    out = []
    for source in range(NMONKEYS):
        if source == SUBJECT:
            out.append(list(range(9)))                                            # all 9 non-subject
        else:
            out.append([k for k in range(9) if k != FULL_TO_K[source]])
    return out


def vec_r2(X, y):
    """Vectorized R^2 (Pearson r squared) per row of X against y.

    For OLS with intercept evaluated on training data, this equals sklearn's
    r2_score(y, y_pred). Verified at floating-point precision.

    Args:
        X: (..., n) ndarray
        y: (n,) ndarray

    Returns:
        R^2 values with leading shape of X.
    """
    n = X.shape[-1]
    sx = X.sum(-1); sy = y.sum()
    sxy = (X*y).sum(-1); sxx = (X*X).sum(-1); syy = (y*y).sum()
    num = n*sxy - sx*sy
    den = (n*sxx - sx*sx) * (n*syy - sy*sy)
    return np.where(den > 1e-20, (num*num)/np.maximum(den, 1e-20), 0.0)


def scalar_r2(x, y):
    return float(vec_r2(x.reshape(1, -1), y)[0])


def compute_observed_R2(X_mean, ncells):
    """Compute the full (6, 10, ncells) array of observed R^2 values.

    Returns:
        obs_R2: shape (6 behaviors, 10 source monkeys, ncells)
        valid_k_per_source: list of lists (one per source)
        y_per: dict {(behavior, source): y vector}
        X_per_source: dict {source: (ncells, n_valid) X submatrix}
    """
    valid_k = build_valid_k_per_source()
    y_per = build_y_per(valid_k)
    X_per_source = {s: X_mean[:, valid_k[s]].copy() for s in range(NMONKEYS)}

    obs_R2 = np.zeros((6, NMONKEYS, ncells))
    for ib in range(6):
        for s in range(NMONKEYS):
            obs_R2[ib, s] = vec_r2(X_per_source[s], y_per[(ib, s)])
    return obs_R2, valid_k, y_per, X_per_source


# ============================================================
# Style helpers (so all figures look consistent)
# ============================================================
import matplotlib as mpl

def set_plot_style():
    mpl.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,
        'figure.dpi':150,'savefig.dpi':150,
        'axes.spines.top':False,'axes.spines.right':False,
    })

COLORS = {
    'blue':    '#4C72B0',
    'red':     '#C44E52',
    'gray':    '#888888',
    'green':   '#55A868',
    'dark':    '#444444',
    'success': '#2E8B57',
    'danger':  '#B22222',
}
