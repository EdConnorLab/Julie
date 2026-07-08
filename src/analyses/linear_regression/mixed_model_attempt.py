"""
Option A: Mixed-effects model for single-neuron social encoding analysis.

Requires: statsmodels, numpy, pandas, scipy
Install: pip install statsmodels

This script pools across reference monkeys using a linear mixed-effects model:
    firing_rate ~ behavioral_frequency + (behavioral_frequency | reference_monkey)

For each neuron-behavior pair, instead of 9 separate regressions (one per
reference monkey), we run a single model with all ~72 observations, treating
reference monkey as a random effect.

Usage:
    python option_a_mixed_model.py <path_to_results.txt>
"""

import sys
import csv
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings


def parse_data(filepath):
    """Parse the regression results file and return raw data."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for r in reader:
            bv = np.array([int(x) for x in r['Behavior_Vector'].strip().strip('[]').split()])
            nr_str = r['Neural_Response'].replace('\n', ' ').strip()
            nr = np.array([float(x) for x in nr_str.strip('[]').split()])
            stim = [s.strip().strip("'\"") for s in r['Stimulus_Monkeys'].strip('[]').split(',')]
            rows.append({
                'NeuronID': r['NeuronID'],
                'Behavior': r['Behavior'],
                'Source_Monkey': r['Source_Monkey'],
                'Stimulus_Monkeys': stim,
                'Behavior_Vector': bv,
                'Neural_Response': nr,
            })
    return rows


def build_long_format(rows, neuron_id, behavior):
    """
    Build a long-format DataFrame for one neuron-behavior pair.

    Each row = one (stimulus_identity, reference_monkey) observation.
    Columns: stimulus_id, ref_monkey, firing_rate, beh_freq, beh_freq_z
    """
    subset = [r for r in rows if r['NeuronID'] == neuron_id and r['Behavior'] == behavior]

    long_rows = []
    for r in subset:
        ref = r['Source_Monkey']
        for i, stim_id in enumerate(r['Stimulus_Monkeys']):
            long_rows.append({
                'stimulus_id': stim_id,
                'ref_monkey': ref,
                'firing_rate': r['Neural_Response'][i],
                'beh_freq': float(r['Behavior_Vector'][i]),
            })

    df = pd.DataFrame(long_rows)

    # Z-score behavioral frequency within each reference monkey
    # (preserves relative ordering but normalizes scale across ref monkeys)
    df['beh_freq_z'] = df.groupby('ref_monkey')['beh_freq'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # Z-score firing rate (same for all ref monkeys within a neuron)
    fr_mean = df['firing_rate'].mean()
    fr_std = df['firing_rate'].std()
    df['firing_rate_z'] = (df['firing_rate'] - fr_mean) / fr_std if fr_std > 0 else 0

    return df


def run_mixed_model(df):
    """
    Run a linear mixed-effects model:
        firing_rate_z ~ beh_freq_z + (beh_freq_z | ref_monkey)

    Returns dict with fixed effect estimate, p-value, etc.
    """
    import statsmodels.formula.api as smf

    try:
        # Random intercept + random slope for reference monkey
        model = smf.mixedlm(
            "firing_rate_z ~ beh_freq_z",
            df,
            groups=df["ref_monkey"],
            re_formula="~beh_freq_z"
        )
        result = model.fit(reml=True)

        coef = result.fe_params['beh_freq_z']
        p_val = result.pvalues['beh_freq_z']

        return {
            'coef': coef,
            'se': result.bse['beh_freq_z'],
            'z_stat': result.tvalues['beh_freq_z'],
            'p_value': p_val,
            'converged': result.converged,
            'n_obs': len(df),
        }

    except Exception as e:
        # If full_dominance_first model fails (singular fit), try random intercept only
        try:
            model = smf.mixedlm(
                "firing_rate_z ~ beh_freq_z",
                df,
                groups=df["ref_monkey"],
            )
            result = model.fit(reml=True)

            return {
                'coef': result.fe_params['beh_freq_z'],
                'se': result.bse['beh_freq_z'],
                'z_stat': result.tvalues['beh_freq_z'],
                'p_value': result.pvalues['beh_freq_z'],
                'converged': result.converged,
                'n_obs': len(df),
                'note': 'random_intercept_only',
            }
        except Exception as e2:
            return {'error': str(e2)}


def main(filepath):
    print("Parsing data...")
    rows = parse_data(filepath)

    neurons = sorted(set(r['NeuronID'] for r in rows))
    behaviors = sorted(set(r['Behavior'] for r in rows))

    print(f"Running mixed models for {len(neurons)} neurons x {len(behaviors)} behaviors...")

    results = []
    for nid in neurons:
        for beh in behaviors:
            subset = [r for r in rows if r['NeuronID'] == nid and r['Behavior'] == beh]
            if not subset:
                continue

            df = build_long_format(rows, nid, beh)

            if df['beh_freq'].std() < 1e-10 or df['firing_rate'].std() < 1e-10:
                continue

            res = run_mixed_model(df)
            res['NeuronID'] = nid
            res['Behavior'] = beh
            res['region'] = nid.split('_')[0]
            results.append(res)

    # Filter out errors
    results = [r for r in results if 'error' not in r]
    results.sort(key=lambda x: x['p_value'])

    # FDR correction (Benjamini-Hochberg)
    n_tests = len(results)
    for i, r in enumerate(results):
        rank = i + 1
        r['fdr_q'] = min(r['p_value'] * n_tests / rank, 1.0)
    for i in range(n_tests - 2, -1, -1):
        results[i]['fdr_q'] = min(results[i]['fdr_q'], results[i + 1]['fdr_q'])

    # Print results
    print(f"\n{'=' * 70}")
    print(f"OPTION A: Mixed-Effects Model Results")
    print(f"{'=' * 70}")
    print(f"Total tests: {n_tests}")
    print(f"p < 0.05: {sum(1 for r in results if r['p_value'] < 0.05)}")
    print(f"FDR q < 0.05: {sum(1 for r in results if r['fdr_q'] < 0.05)}")
    print(f"FDR q < 0.10: {sum(1 for r in results if r['fdr_q'] < 0.10)}")
    print(f"Expected FP at 0.05: {n_tests * 0.05:.1f}")

    print(f"\nTop 30 results:")
    print(f"{'Neuron':>35s} | {'Behavior':>18s} |   coef |  z-stat |  p-val |  FDR_q | n_obs")
    for r in results[:30]:
        sig = "*" if r['fdr_q'] < 0.05 else ("~" if r['fdr_q'] < 0.10 else " ")
        note = r.get('note', '')
        print(
            f"{r['NeuronID'][-35:]:>35s} | {r['Behavior']:>18s} | {r['coef']:+.3f} | {r['z_stat']:+.3f} | {r['p_value']:.4f} | {r['fdr_q']:.4f} | {r['n_obs']:3d} {sig} {note}")

    # Save full_dominance_first results
    out_df = pd.DataFrame([{k: v for k, v in r.items()} for r in results])
    out_path = filepath.replace('.txt', '_mixed_model_results.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("U/home/connorlab/Documents/GitHub/Julie/src/analyses/linear_regression/mixed_model_attempt.pysage: python option_a_mixed_model.py <path_to_results.txt>")
        sys.exit(1)

    warnings.filterwarnings('ignore')
    main(sys.argv[1])