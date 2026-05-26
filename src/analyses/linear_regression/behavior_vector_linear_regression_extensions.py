"""
Additions to behavior_vector_linear_regression.py.

Adds
----
1. winsorize_behavior_matrix
       Pool all non-zero values in a behavior matrix and cap at a
       chosen percentile. Call this BEFORE passing the matrix to
       any regression function.

2. run_directional_vector_linear_regression_cell_level_with_loo
       Cell-level regression with leave-one-out sensitivity check.
       Removes the single most extreme behavioral observation,
       refits, and flags whether direction / significance changed.

3. run_directional_vector_linear_regression_window_level_with_loo
       Same as above for the window-level variant.

4. summarize_loo_results
       Aggregate per-monkey LOO rows to one row per (NeuronID, Behavior)
       with stability rates. Output is dashboard-compatible.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Winsorization
# ─────────────────────────────────────────────────────────────────────────────

def winsorize_behavior_matrix(behavior_matrix: np.ndarray,
                               percentile: float = 95) -> np.ndarray:
    """
    Cap extreme values in a behavior interaction matrix.

    All non-zero values across the entire matrix are pooled together and
    the ``percentile``-th value is used as an upper cap.  Pooling across
    the whole matrix (rather than per-row) gives a stable threshold even
    when individual rows contain only ~8 values.

    Parameters
    ----------
    behavior_matrix : np.ndarray  shape (n_monkeys, n_monkeys)
        Raw interaction count matrix (e.g. SubmissionTo counts).
    percentile : float
        Upper cap percentile.  Default 95.

    Returns
    -------
    np.ndarray  same shape, values clipped at the ``percentile``-th value.

    Example
    -------
    >>> winsorized = winsorize_behavior_matrix(submission_matrix, percentile=95)
    >>> results    = run_directional_vector_linear_regression_cell_level(
    ...                 spike_df, winsorized, "SubmissionTo", ...)
    """
    nonzero = behavior_matrix.flatten()
    nonzero = nonzero[nonzero > 0]
    if len(nonzero) == 0:
        return behavior_matrix.copy()
    cap = np.percentile(nonzero, percentile)
    return np.clip(behavior_matrix, a_min=None, a_max=cap)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Private LOO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _most_extreme_idx(behavior_vec: np.ndarray) -> int:
    """Index of the observation furthest from the mean (outlier candidate)."""
    return int(np.argmax(np.abs(behavior_vec - behavior_vec.mean())))


def _fit_loo(behavior_vec: np.ndarray,
             neural_response: np.ndarray,
             stimulus_monkeys: list) -> dict:
    """
    Remove the single most extreme behavioral observation, refit OLS, return stats.

    Returns a dict with:
        excluded_stimulus_monkey, coef_loo, p_value_loo, r_squared_loo
    All numeric fields are NaN if refitting fails (e.g. zero variance after removal).
    """
    drop = _most_extreme_idx(behavior_vec)
    excluded = stimulus_monkeys[drop]

    keep = np.ones(len(behavior_vec), dtype=bool)
    keep[drop] = False

    bvec = behavior_vec[keep]
    neural = neural_response[keep]

    if np.var(bvec) < 1e-3 or np.var(neural) < 1e-3:
        return {'excluded_stimulus_monkey': excluded,
                'coef_loo': np.nan,
                'p_value_loo': np.nan,
                'r_squared_loo': np.nan}

    m = sm.OLS(zscore(neural), sm.add_constant(zscore(bvec))).fit()
    return {'excluded_stimulus_monkey': excluded,
            'coef_loo': m.params[1],
            'p_value_loo': m.pvalues[1],
            'r_squared_loo': m.rsquared}


def _stability_flags(coef: float, p: float,
                     coef_loo: float, p_loo: float,
                     alpha: float = 0.05) -> dict:
    """
    Compare full_dominance_first and LOO regression results.

    Returns
    -------
    dict with:
        direction_stable     bool  sign(coef) == sign(coef_loo)
        significance_stable  bool  (p < alpha) unchanged after LOO
        stability_flag       bool  both of the above are True
    """
    if np.isnan(coef_loo) or np.isnan(p_loo):
        return {'direction_stable': False,
                'significance_stable': False,
                'stability_flag': False}

    dir_ok = bool(np.sign(coef) == np.sign(coef_loo))
    sig_ok = bool((p < alpha) == (p_loo < alpha))
    return {'direction_stable': dir_ok,
            'significance_stable': sig_ok,
            'stability_flag': dir_ok and sig_ok}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cell-level regression + LOO
# ─────────────────────────────────────────────────────────────────────────────

def run_directional_vector_linear_regression_cell_level_with_loo(
        spike_df,
        behavior_matrix,
        behavior_name: str,
        group_name: str,
        monkey_list: list,
        subject_idx: int,
        use_spikerate: bool = False,
        permutation_test: bool = False,
        n_perm: int = 10000,
        random_state: int = 42) -> pd.DataFrame:
    """
    Cell-level directional regression with leave-one-out sensitivity analysis.

    Runs the standard regression (identical to
    ``run_directional_vector_linear_regression_cell_level``) and then removes
    the single stimulus monkey whose behavior value is furthest from the mean,
    refits on the remaining n-1 observations, and records whether the result
    is stable.

    Additional output columns (beyond the standard ones)
    ----------------------------------------------------
    excluded_stimulus_monkey : str    monkey removed in LOO
    coef_loo                 : float  slope after LOO
    p_value_loo              : float  p-value after LOO
    r_squared_loo            : float  R² after LOO
    direction_stable         : bool   True if sign(coef) == sign(coef_loo)
    significance_stable      : bool   True if significance (p<0.05) unchanged
    stability_flag           : bool   True if both direction and significance stable

    Parameters  (same as run_directional_vector_linear_regression_cell_level)
    ----------
    spike_df        : pd.DataFrame
    behavior_matrix : np.ndarray  shape (n_monkeys, n_monkeys)
    behavior_name   : str
    group_name      : str
    monkey_list     : list[str]
    subject_idx     : int
    use_spikerate   : bool
    permutation_test: bool
    n_perm          : int
    random_state    : int
    """
    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'

    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'] != 'NewMonkey')
    ]
    mean_spikes = filtered_df.groupby(
        ['NeuronID', 'MonkeyName'], as_index=False
    )[value_col].mean()
    spike_matrix = (mean_spikes
                    .pivot(index='NeuronID', columns='MonkeyName', values=value_col)
                    .reindex(columns=monkey_list))

    for neuron_id, row in tqdm(spike_matrix.iterrows(),
                               total=len(spike_matrix),
                               desc='Cell-level LOO regression'):
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue

            raw_behavior = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_behavior, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            behavior_vec = raw_behavior[mask]

            stimulus_monkeys = [m for i, m in enumerate(monkey_list)
                                if i not in (src_idx, subject_idx)]
            neural_response = row[stimulus_monkeys].values.astype(float)

            if np.var(neural_response) < 1e-3 or np.var(behavior_vec) < 1e-3:
                continue
            if len(neural_response) != len(behavior_vec):
                print(f'Length mismatch for {neuron_id} (source: {src_monkey})')
                continue

            try:
                # ── full regression ──────────────────────────────────────────
                x = zscore(behavior_vec)
                y = zscore(neural_response)
                model = sm.OLS(y, sm.add_constant(x)).fit()
                r_squared = model.rsquared

                if permutation_test:
                    rng = np.random.default_rng(random_state)
                    null_dist = np.array([
                        sm.OLS(y, sm.add_constant(rng.permutation(x))).fit().rsquared
                        for _ in range(n_perm)
                    ])
                    p_perm = (np.sum(null_dist >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None

                # ── LOO ──────────────────────────────────────────────────────
                loo = _fit_loo(behavior_vec, neural_response, stimulus_monkeys)
                stab = _stability_flags(model.params[1], model.pvalues[1],
                                        loo['coef_loo'], loo['p_value_loo'])

                results.append({
                    # Standard columns — identical to existing function
                    'NeuronID':          neuron_id,
                    'Behavior':          behavior_name,
                    'Source_Monkey':     src_monkey,
                    'Stimulus_Monkeys':  stimulus_monkeys,
                    'Behavior_Vector':   behavior_vec,
                    'Neural_Response':   neural_response,
                    'R_squared':         r_squared,
                    'coef':              model.params[1],
                    'intercept':         model.params[0],
                    'p_value':           model.pvalues[1],
                    'p_perm':            p_perm,
                    # LOO columns
                    **loo,
                    **stab,
                })

            except Exception as e:
                print(f'Error for Neuron {neuron_id}, Monkey {src_monkey}: {e}')
                continue

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Window-level regression + LOO
# ─────────────────────────────────────────────────────────────────────────────

def run_directional_vector_linear_regression_window_level_with_loo(
        spike_df,
        behavior_matrix,
        behavior_name: str,
        group_name: str,
        monkey_list: list,
        subject_idx: int,
        use_spikerate: bool = False,
        permutation_test: bool = False,
        n_perm: int = 10000,
        random_state: int = 42) -> pd.DataFrame:
    """
    Window-level directional regression with LOO sensitivity analysis.

    Mirrors ``run_directional_vector_linear_regression_window_level`` with
    the same LOO extensions as the cell-level version above.
    """
    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'

    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'] != 'NewMonkey')
    ]
    mean_spikes = filtered_df.groupby(
        ['NeuronID', 'MonkeyName', 'WindowStart_ms', 'WindowEnd_ms'],
        as_index=False
    )[value_col].mean()

    mean_spikes['NeuronWindowID'] = (
        mean_spikes['NeuronID'].astype(str) + '_' +
        mean_spikes['WindowStart_ms'].astype(str) + '_' +
        mean_spikes['WindowEnd_ms'].astype(str)
    )
    spike_matrix = mean_spikes.pivot(
        index='NeuronWindowID', columns='MonkeyName', values=value_col
    )
    id_map = mean_spikes[
        ['NeuronWindowID', 'NeuronID', 'WindowStart_ms', 'WindowEnd_ms']
    ].drop_duplicates()

    for unit_id, row in tqdm(spike_matrix.iterrows(),
                             total=len(spike_matrix),
                             desc='Window-level LOO regression',
                             dynamic_ncols=True):
        meta = id_map[id_map['NeuronWindowID'] == unit_id].iloc[0]
        neuron_id = meta['NeuronID']
        win_start = meta['WindowStart_ms']
        win_end   = meta['WindowEnd_ms']

        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue

            raw_behavior = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_behavior, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            behavior_vec = raw_behavior[mask]

            stimulus_monkeys = [m for i, m in enumerate(monkey_list)
                                if i not in (src_idx, subject_idx)]
            neural_response = row[stimulus_monkeys].values.astype(float)

            if np.var(neural_response) < 1e-3 or np.var(behavior_vec) < 1e-3:
                continue
            if len(neural_response) != len(behavior_vec):
                print(f'Length mismatch for {unit_id} (source: {src_monkey})')
                continue

            try:
                x = zscore(behavior_vec)
                y = zscore(neural_response)
                model = sm.OLS(y, sm.add_constant(x)).fit()
                r_squared = model.rsquared

                if permutation_test:
                    rng = np.random.default_rng(random_state)
                    null_dist = np.array([
                        sm.OLS(y, sm.add_constant(rng.permutation(x))).fit().rsquared
                        for _ in range(n_perm)
                    ])
                    p_perm = (np.sum(null_dist >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None

                loo = _fit_loo(behavior_vec, neural_response, stimulus_monkeys)
                stab = _stability_flags(model.params[1], model.pvalues[1],
                                        loo['coef_loo'], loo['p_value_loo'])

                results.append({
                    'NeuronID':          neuron_id,
                    'WindowStart_ms':    win_start,
                    'WindowEnd_ms':      win_end,
                    'Behavior':          behavior_name,
                    'Source_Monkey':     src_monkey,
                    'Stimulus_Monkeys':  stimulus_monkeys,
                    'Behavior_Vector':   behavior_vec,
                    'Neural_Response':   neural_response,
                    'R_squared':         r_squared,
                    'coef':              model.params[1],
                    'intercept':         model.params[0],
                    'p_value':           model.pvalues[1],
                    'p_perm':            p_perm,
                    **loo,
                    **stab,
                })

            except Exception as e:
                print(f'Error for {unit_id}, Monkey {src_monkey}: {e}')
                continue

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Aggregate LOO results → dashboard-compatible CSV
# ─────────────────────────────────────────────────────────────────────────────

def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    q = np.empty(n)
    q[order] = p_values[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


def summarize_loo_results(loo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-monkey LOO rows to one row per (NeuronID, Behavior).

    This is the file you upload to the dashboard.

    Columns in output
    -----------------
    NeuronID                  : str
    Behavior                  : str
    region                    : str   (parsed from NeuronID prefix)
    n_source_monkeys          : int   number of source-monkey regressions
    coef                      : float mean coefficient across source monkeys
    p_value                   : float mean p-value across source monkeys
    coef_loo                  : float mean LOO coefficient
    p_value_loo               : float mean LOO p-value
    stability_rate            : float fraction of source monkeys with stability_flag=True
    direction_stable_rate     : float fraction with direction_stable=True
    significance_stable_rate  : float fraction with significance_stable=True
    fdr_q                     : float BH-corrected q-value on mean p_value

    Parameters
    ----------
    loo_df : pd.DataFrame
        Output of run_directional_vector_linear_regression_cell/window_level_with_loo.

    Returns
    -------
    pd.DataFrame sorted by p_value ascending.
    """
    records = []
    for (neuron_id, behavior), grp in loo_df.groupby(['NeuronID', 'Behavior']):
        records.append({
            'NeuronID':                  neuron_id,
            'Behavior':                  behavior,
            'region':                    str(neuron_id).split('_')[0],
            'n_source_monkeys':          len(grp),
            'coef':                      grp['coef'].mean(),
            'p_value':                   grp['p_value'].mean(),
            'coef_loo':                  grp['coef_loo'].mean(),
            'p_value_loo':               grp['p_value_loo'].mean(),
            'stability_rate':            grp['stability_flag'].mean(),
            'direction_stable_rate':     grp['direction_stable'].mean(),
            'significance_stable_rate':  grp['significance_stable'].mean(),
        })

    summary = pd.DataFrame(records)
    summary['fdr_q'] = _bh_fdr(summary['p_value'].fillna(1.0).values)
    return summary.sort_values('p_value').reset_index(drop=True)
