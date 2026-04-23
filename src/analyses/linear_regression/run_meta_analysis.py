"""
run_meta_analysis.py

DerSimonian-Laird random-effects meta-analysis across reference monkeys.

For each neuron × behavior pair the per-monkey regression coefficients are
pooled using inverse-variance weighting, allowing for between-monkey
heterogeneity (tau²).  The key biological output beyond a pooled p-value is
tau² itself: a neuron with low tau² encodes the behavior consistently
regardless of which monkey is the observer (general social encoder), while
high tau² indicates context-specific encoding.

Standard error derivation
-------------------------
Both the predictor (behavior_vec) and response (neural_response) are z-scored
with scipy.stats.zscore, which uses the population standard deviation (divides
by n, not n-1).  After z-scoring:

    sum(x_z²) = n          (exact, because population std normalises to this)
    SST(y_z)  = n

    MSE  = SSE / (n-2)
         = n · (1 - R²) / (n - 2)

    SE(β̂) = sqrt(MSE / sum(x_z²))
           = sqrt((1 - R²) / (n - 2))

This is derived analytically from the stored R_squared and the length of the
Behavior_Vector — no need to rerun regressions.

Usage — standalone (CLI)
------------------------
    python run_meta_analysis.py \\
        --input  /path/to/regression_results.pkl \\
        --output /path/to/meta_analysis_results.csv

Usage — in-pipeline (after running regressions in memory)
----------------------------------------------------------
    from run_meta_analysis import run_meta_analysis_from_dataframe

    neuron_results_df = run_directional_vector_linear_regression_cell_level(...)
    meta_df = run_meta_analysis_from_dataframe(neuron_results_df)
    meta_df.to_csv('meta_analysis_results.csv', index=False)
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2


# ─────────────────────────────────────────────────────────────────────────────
# SE from R²
# ─────────────────────────────────────────────────────────────────────────────

def compute_se_from_r2(r_squared: float, n: int) -> float:
    """
    Standard error of the OLS slope when both x and y are population-z-scored.

    SE = sqrt((1 - R²) / (n - 2))

    Parameters
    ----------
    r_squared : float   R² from the OLS fit
    n         : int     number of observations (stimulus monkeys, typically 8)

    Returns
    -------
    float  (NaN if n <= 2 or R² >= 1)
    """
    if n <= 2 or r_squared >= 1.0:
        return np.nan
    return float(np.sqrt((1.0 - r_squared) / (n - 2)))


# ─────────────────────────────────────────────────────────────────────────────
# DerSimonian-Laird estimator
# ─────────────────────────────────────────────────────────────────────────────

def dersimonian_laird(coefs: np.ndarray, ses: np.ndarray) -> dict:
    """
    DerSimonian-Laird random-effects meta-analysis.

    Pools k per-monkey regression coefficients using inverse-variance
    weighting with an empirical between-monkey variance (tau²).

    Parameters
    ----------
    coefs : array-like  shape (k,)   per-monkey regression coefficients
    ses   : array-like  shape (k,)   corresponding standard errors

    Returns
    -------
    dict with keys:
        pooled_coef  float   pooled random-effects coefficient
        pooled_se    float   SE of pooled coefficient
        pooled_z     float   z-statistic
        pooled_p     float   two-tailed p-value
        tau2         float   between-monkey variance  (0 = homogeneous)
        I2           float   % variance due to heterogeneity (0-100)
        Q            float   Cochran's Q
        Q_p          float   p-value for Q (test of homogeneity)
        k            int     number of monkey regressions used
    """
    coefs = np.asarray(coefs, dtype=float)
    ses   = np.asarray(ses,   dtype=float)

    # Remove invalid entries (NaN, zero or negative SE)
    valid = ~(np.isnan(coefs) | np.isnan(ses) | (ses <= 0))
    coefs, ses = coefs[valid], ses[valid]
    k = int(valid.sum())

    nan_result = {col: np.nan for col in
                  ['pooled_coef', 'pooled_se', 'pooled_z', 'pooled_p',
                   'tau2', 'I2', 'Q', 'Q_p']}
    nan_result['k'] = k
    if k < 2:
        return nan_result

    # Step 1 — fixed-effect weights and pooled estimate
    w = 1.0 / ses ** 2
    beta_fe = np.sum(w * coefs) / np.sum(w)

    # Step 2 — Cochran's Q (heterogeneity statistic)
    Q   = float(np.sum(w * (coefs - beta_fe) ** 2))
    Q_p = float(1 - chi2.cdf(Q, df=k - 1))

    # Step 3 — tau² (method-of-moments, floor at 0)
    c    = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = float(max(0.0, (Q - (k - 1)) / c))

    # Step 4 — I²
    I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100) if Q > 0 else 0.0

    # Step 5 — random-effects weights and pooled estimate
    w_re    = 1.0 / (ses ** 2 + tau2)
    beta_re = float(np.sum(w_re * coefs) / np.sum(w_re))
    se_re   = float(np.sqrt(1.0 / np.sum(w_re)))
    z       = beta_re / se_re
    p_re    = float(2 * (1 - norm.cdf(abs(z))))

    return {
        'pooled_coef': beta_re,
        'pooled_se':   se_re,
        'pooled_z':    z,
        'pooled_p':    p_re,
        'tau2':        tau2,
        'I2':          I2,
        'Q':           Q,
        'Q_p':         Q_p,
        'k':           k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BH FDR correction
# ─────────────────────────────────────────────────────────────────────────────

def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    n     = len(p_values)
    order = np.argsort(p_values)
    q     = np.empty(n)
    q[order] = p_values[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_meta_analysis_from_dataframe(regression_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run DerSimonian-Laird meta-analysis from an in-memory regression DataFrame.

    Expects one row per (NeuronID, Behavior, Source_Monkey) with columns:
        NeuronID, Behavior, coef, R_squared, Behavior_Vector

    Behavior_Vector must be a numpy array (to derive n from its length).

    Returns
    -------
    pd.DataFrame  one row per (NeuronID, Behavior) with:
        NeuronID, Behavior, region,
        pooled_coef, pooled_se, pooled_z, pooled_p,
        tau2, I2, Q, Q_p, k, fdr_q
    """
    records = []

    for (neuron_id, behavior), grp in regression_df.groupby(['NeuronID', 'Behavior']):
        coefs    = grp['coef'].values
        r2_vals  = grp['R_squared'].values
        n_vals   = grp['Behavior_Vector'].apply(
            lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 8
        ).values
        ses = np.array([compute_se_from_r2(r2, n)
                        for r2, n in zip(r2_vals, n_vals)])

        meta   = dersimonian_laird(coefs, ses)
        region = str(neuron_id).split('_')[0]

        records.append({'NeuronID': neuron_id, 'Behavior': behavior,
                        'region': region, **meta})

    if not records:
        return pd.DataFrame()

    meta_df = pd.DataFrame(records)

    # BH FDR on pooled_p
    p_vals          = meta_df['pooled_p'].fillna(1.0).values
    meta_df['fdr_q'] = _bh_fdr(p_vals)

    meta_df = meta_df.sort_values('pooled_p').reset_index(drop=True)

    # Print summary
    print(f"\n{'='*60}")
    print('META-ANALYSIS  (DerSimonian-Laird random-effects)')
    print(f"{'='*60}")
    print(f"Neuron × behavior pairs : {len(meta_df)}")
    print(f"pooled_p < 0.05         : {(meta_df['pooled_p'] < 0.05).sum()}")
    print(f"FDR q   < 0.05          : {(meta_df['fdr_q']   < 0.05).sum()}")
    print(f"FDR q   < 0.10          : {(meta_df['fdr_q']   < 0.10).sum()}")
    top_cols = ['NeuronID', 'Behavior', 'region',
                'pooled_coef', 'pooled_se', 'pooled_p',
                'tau2', 'I2', 'fdr_q', 'k']
    print(f"\nTop 10 results:")
    print(meta_df[top_cols].head(10).to_string(index=False))

    return meta_df


def run_meta_analysis_from_file(input_path: str,
                                output_path: str = None) -> pd.DataFrame:
    """
    Load regression results from a .pkl or .csv file and run meta-analysis.

    For .csv files, Behavior_Vector is parsed from its string representation.
    n is inferred from the parsed array length (or defaults to 8).

    Parameters
    ----------
    input_path  : str   path to regression results (.pkl or .csv)
    output_path : str   optional path to save meta-analysis CSV

    Returns
    -------
    pd.DataFrame
    """
    path = Path(input_path)

    if path.suffix == '.pkl':
        df = pd.read_pickle(path)

    elif path.suffix == '.csv':
        df = pd.read_csv(path)
        # Parse Behavior_Vector if it was saved as a string
        if 'Behavior_Vector' in df.columns:
            sample = df['Behavior_Vector'].iloc[0]
            if isinstance(sample, str):
                df['Behavior_Vector'] = df['Behavior_Vector'].apply(
                    lambda s: np.fromstring(
                        s.strip().strip('[]'), sep=' '
                    ) if s.strip().startswith('[') else np.array(ast.literal_eval(s))
                )
    else:
        raise ValueError(f'Unsupported format: {path.suffix}. Use .pkl or .csv')

    meta_df = run_meta_analysis_from_dataframe(df)

    if output_path is not None:
        meta_df.to_csv(output_path, index=False)
        print(f'Saved → {output_path}')

    return meta_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DerSimonian-Laird meta-analysis on per-monkey regression results'
    )
    parser.add_argument('--input',  required=True,
                        help='Regression results file (.pkl or .csv)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (optional)')
    args = parser.parse_args()
    run_meta_analysis_from_file(args.input, args.output)


if __name__ == '__main__':
    main()
