"""
run_tdr_shuffle_test.py
=======================

Cross-validated permutation test for TDR significance.

The naive shuffle test (comparing raw beta norms) lacks power with
only 14 conditions because random label assignments can produce
large beta norms by overfitting.

This script uses a CROSS-VALIDATED approach:
    1. Split each neuron's trials into two halves (train / test).
    2. Fit regression on the TRAIN half -> get betas per neuron.
    3. Use those betas to PREDICT firing rates on the TEST half.
    4. Measure prediction quality as cross-validated R-squared.
    5. Real labels should generalize across trial splits;
       shuffled labels should NOT (they overfit to train noise).

The test statistic is the mean cross-validated R-squared across all
neurons and timepoints.  This is compared against the distribution
of the same statistic under shuffled labels.

Usage:
    python run_tdr_shuffle_test.py

    Adjust N_SHUFFLES, PREDICTORS, N_SPLITS below.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from config import TrajectoryConfig
from data_loading import load_and_filter
from run_tdr_multi import (
    build_regressor_table,
    churchland_preprocess_tensor,
    pca_denoising_projector,
    denoise_betas,
    check_collinearity,
    PLOT_SAVE_DIR,
)

sys.path.insert(0,
                '/population/neural_trajectory')
from run_tdr_rank import (
    build_rank_tensor,
    MONKEY_INFO_PATH,
)


# ---------------------------------------------------------------------------
# Core: single-neuron split-half regression
# ---------------------------------------------------------------------------
def _collect_neuron_trials(df_filt, session, nid, monkey_list, pred_lookup,
                           n_bins, bin_width, analysis_window):
    """
    Collect all single-trial data for one neuron.

    Returns
    -------
    trial_rates : (n_trials, n_bins) firing rates in Hz
    trial_preds : (n_trials, n_pred) predictor values
    trial_monkeys : list of str, monkey name for each trial
    """
    neu_df = df_filt[(df_filt['session'] == session)
                     & (df_filt['NeuronID'] == nid)]

    trial_rates = []
    trial_preds = []
    trial_monkeys = []

    for tf, t_group in neu_df.groupby('TaskField'):
        row = t_group.iloc[0]
        monkey_name = row['MonkeyName']
        if monkey_name not in pred_lookup:
            continue

        rel = np.asarray(row['SpikeTimes']) - row['EpochStartStop'][0]
        rel = rel[(rel >= 0) & (rel < analysis_window)]
        counts = np.zeros(n_bins)
        bins_arr = np.floor(rel / bin_width).astype(int)
        bins_arr = bins_arr[bins_arr < n_bins]
        for b in bins_arr:
            counts[b] += 1
        rates = counts / bin_width

        trial_rates.append(rates)
        trial_preds.append(pred_lookup[monkey_name])
        trial_monkeys.append(monkey_name)

    if len(trial_rates) == 0:
        return None, None, None

    return (np.array(trial_rates), np.array(trial_preds), trial_monkeys)


def cross_validated_regression(df, cfg, monkey_list, tbl, predictors, info,
                               n_splits=10, seed=42, shuffle=False,
                               rng=None):
    """
    Split-half cross-validated regression.

    For each split:
        - Randomly divide each neuron's trials into train/test.
        - Fit regression on train -> get betas per neuron per timepoint.
        - Predict test firing rates using the trained betas.
        - Compute R-squared on test data.

    Parameters
    ----------
    shuffle : bool
        If True, permute which monkey gets which predictor values.
    rng : np.random.Generator, optional

    Returns
    -------
    cv_r2_per_time : (n_bins,) mean cross-validated R-squared at each
                     timepoint, averaged across neurons and splits.
    cv_r2_per_neuron : (n_neurons,) mean cross-validated R-squared per
                       neuron, averaged across timepoints and splits.
    cv_r2_overall : float, single summary statistic.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_bins = info['n_bins']
    bin_width = cfg.bin_width
    analysis_window = n_bins * bin_width
    n_pred = len(predictors)
    neuron_id_strings = info['neuron_ids']
    n_neurons = len(neuron_id_strings)

    # Build predictor lookup
    pred_lookup = {}
    for m in monkey_list:
        if m in tbl.index:
            pred_lookup[m] = tbl.loc[m, predictors].values.astype(float)

    # If shuffling, permute predictor values across monkeys
    if shuffle:
        pred_vals = np.array([pred_lookup[m] for m in monkey_list])
        perm = rng.permutation(len(monkey_list))
        pred_lookup_use = {m: pred_vals[perm[i]]
                           for i, m in enumerate(monkey_list)}
    else:
        pred_lookup_use = pred_lookup

    df_filt = df[df['MonkeyName'].isin(monkey_list)].copy()

    # Accumulators across splits
    all_r2 = np.zeros((n_splits, n_neurons, n_bins))
    all_r2[:] = np.nan

    for split in range(n_splits):
        for neu_idx, nid_str in enumerate(neuron_id_strings):
            session, nid = nid_str.split('__', 1)

            rates, preds, monkeys = _collect_neuron_trials(
                df_filt, session, nid, monkey_list, pred_lookup_use,
                n_bins, bin_width, analysis_window)

            if rates is None or len(rates) < n_pred + 4:
                continue

            n_trials = len(rates)

            # Z-score across all trials and times
            mu = rates.mean()
            sigma = rates.std()
            if sigma == 0:
                continue
            rates_z = (rates - mu) / sigma

            # Random train/test split (50/50)
            idx = rng.permutation(n_trials)
            half = n_trials // 2
            train_idx = idx[:half]
            test_idx = idx[half:]

            if len(train_idx) < n_pred + 2 or len(test_idx) < 2:
                continue

            X_train = np.hstack([preds[train_idx],
                                 np.ones((len(train_idx), 1))])
            X_test = np.hstack([preds[test_idx],
                                np.ones((len(test_idx), 1))])

            for t in range(n_bins):
                y_train = rates_z[train_idx, t]
                y_test = rates_z[test_idx, t]

                coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                y_pred = X_test @ coef

                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                if ss_tot > 0:
                    all_r2[split, neu_idx, t] = 1 - ss_res / ss_tot
                else:
                    all_r2[split, neu_idx, t] = 0.0

    # Average across splits
    mean_r2 = np.nanmean(all_r2, axis=0)            # (n_neurons, n_bins)
    cv_r2_per_time = np.nanmean(mean_r2, axis=0)    # (n_bins,)
    cv_r2_per_neuron = np.nanmean(mean_r2, axis=1)  # (n_neurons,)
    cv_r2_overall = np.nanmean(mean_r2)

    return cv_r2_per_time, cv_r2_per_neuron, cv_r2_overall


# ---------------------------------------------------------------------------
# Shuffle test driver
# ---------------------------------------------------------------------------
def run_shuffle_test(df, cfg, monkey_list, tbl, predictors, info,
                     n_shuffles=500, n_splits=10, seed=42):
    """
    Run the cross-validated shuffle test.
    """
    rng = np.random.default_rng(seed)

    # Real data
    print("  Computing real cross-validated R-squared...")
    real_r2_time, real_r2_neuron, real_r2_overall = \
        cross_validated_regression(
            df, cfg, monkey_list, tbl, predictors, info,
            n_splits=n_splits, seed=seed, shuffle=False)

    print(f"  Real CV R-sq (overall): {real_r2_overall:.6f}")
    print(f"  Real CV R-sq (peak over time): {real_r2_time.max():.6f} "
          f"at t = {np.argmax(real_r2_time) * cfg.bin_width:.2f}s")

    # Shuffled data
    n_bins = info['n_bins']
    shuf_cv_r2_overall = np.zeros(n_shuffles)
    shuf_cv_r2_time = np.zeros((n_shuffles, n_bins))

    for i in range(n_shuffles):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  shuffle {i+1}/{n_shuffles}")

        s_r2_time, _, s_r2_overall = cross_validated_regression(
            df, cfg, monkey_list, tbl, predictors, info,
            n_splits=n_splits, rng=rng, shuffle=True)

        shuf_cv_r2_overall[i] = s_r2_overall
        shuf_cv_r2_time[i] = s_r2_time

    # P-values
    p_overall = (shuf_cv_r2_overall >= real_r2_overall).mean()
    p_per_time = np.array([
        (shuf_cv_r2_time[:, t] >= real_r2_time[t]).mean()
        for t in range(n_bins)
    ])

    # Peak CV R-sq across time
    real_peak_cv = real_r2_time.max()
    shuf_peak_cv = shuf_cv_r2_time.max(axis=1)
    p_peak = (shuf_peak_cv >= real_peak_cv).mean()

    return dict(
        real_r2_time=real_r2_time,
        real_r2_neuron=real_r2_neuron,
        real_r2_overall=real_r2_overall,
        shuf_r2_overall=shuf_cv_r2_overall,
        shuf_r2_time=shuf_cv_r2_time,
        p_overall=p_overall,
        p_per_time=p_per_time,
        p_peak=p_peak,
        real_peak_cv=real_peak_cv,
        shuf_peak_cv=shuf_peak_cv,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cv_shuffle_results(results, predictors, time_axis, cfg,
                            save_dir=None, tag='', smooth_sigma=2):
    """
    Three-panel figure:
        1. CV R-sq over time: real vs. shuffle distribution
        2. Histogram of shuffled overall CV R-sq with real marked
        3. Histogram of shuffled peak CV R-sq with real marked
    """
    real_r2_time = results['real_r2_time']
    shuf_r2_time = results['shuf_r2_time']
    real_r2_overall = results['real_r2_overall']
    shuf_r2_overall = results['shuf_r2_overall']
    p_overall = results['p_overall']
    p_peak = results['p_peak']
    real_peak_cv = results['real_peak_cv']
    shuf_peak_cv = results['shuf_peak_cv']
    n_shuffles = len(shuf_r2_overall)

    COLOR_REAL = '#2563eb'
    COLOR_SHUF = '#9ca3af'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: CV R-sq over time ---
    ax = axes[0]
    ci_lo = np.percentile(shuf_r2_time, 2.5, axis=0)
    ci_hi = np.percentile(shuf_r2_time, 97.5, axis=0)
    ci_med = np.median(shuf_r2_time, axis=0)

    ci_lo_s = gaussian_filter1d(ci_lo, sigma=smooth_sigma)
    ci_hi_s = gaussian_filter1d(ci_hi, sigma=smooth_sigma)
    ci_med_s = gaussian_filter1d(ci_med, sigma=smooth_sigma)
    real_s = gaussian_filter1d(real_r2_time, sigma=smooth_sigma)

    ax.fill_between(time_axis, ci_lo_s, ci_hi_s,
                    color=COLOR_SHUF, alpha=0.3, label='shuffle 95% CI')
    ax.plot(time_axis, ci_med_s, color=COLOR_SHUF, lw=1, ls='--',
            label='shuffle median')
    ax.plot(time_axis, real_s, color=COLOR_REAL, lw=2.5, label='real')

    # Mark timepoints where real exceeds shuffle 95th percentile
    p95 = np.percentile(shuf_r2_time, 95, axis=0)
    sig_mask = real_r2_time > p95
    if sig_mask.any():
        ax.scatter(time_axis[sig_mask],
                   real_s[sig_mask],
                   color=COLOR_REAL, s=15, zorder=5, label='p < 0.05')

    ax.axhline(0, color='k', lw=0.3, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-validated R-sq')
    ax.set_title('CV R-sq over time')
    ax.legend(fontsize=8)

    # --- Panel 2: overall CV R-sq histogram ---
    ax2 = axes[1]
    ax2.hist(shuf_r2_overall, bins=40, color=COLOR_SHUF, alpha=0.6,
             edgecolor='white')
    ax2.axvline(real_r2_overall, color=COLOR_REAL, lw=2.5,
                label=f'real (p={p_overall:.4f})')
    ax2.set_xlabel('Mean CV R-sq')
    ax2.set_ylabel('Count')
    ax2.set_title('Overall CV R-sq')
    ax2.legend(fontsize=9)

    # --- Panel 3: peak CV R-sq histogram ---
    ax3 = axes[2]
    ax3.hist(shuf_peak_cv, bins=40, color=COLOR_SHUF, alpha=0.6,
             edgecolor='white')
    ax3.axvline(real_peak_cv, color=COLOR_REAL, lw=2.5,
                label=f'real (p={p_peak:.4f})')
    ax3.set_xlabel('Peak CV R-sq (max over time)')
    ax3.set_ylabel('Count')
    ax3.set_title('Peak CV R-sq')
    ax3.legend(fontsize=9)

    pred_str = ', '.join(predictors)
    fig.suptitle(f'{tag}: Cross-validated shuffle test '
                 f'({n_shuffles} permutations)\n'
                 f'Predictors: {pred_str}',
                 fontsize=11, y=1.03)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir, f'{region}_cv_shuffle_test.png'),
                    dpi=200, bbox_inches='tight')
    return fig


def plot_cv_r2_per_neuron(results, cfg, save_dir=None, tag=''):
    """
    Histogram of per-neuron mean CV R-sq, showing which neurons
    contribute most to the population signal.
    """
    r2_neu = results['real_r2_neuron']
    valid = r2_neu[~np.isnan(r2_neu)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid, bins=50, color='#2563eb', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
    ax.axvline(np.median(valid), color='#dc2626', lw=2,
               label=f'median = {np.median(valid):.4f}')
    ax.set_xlabel('Mean CV R-sq (per neuron)')
    ax.set_ylabel('Count')
    ax.set_title(f'{tag}: Distribution of per-neuron CV R-sq\n'
                 f'({(valid > 0).sum()}/{len(valid)} neurons with CV R-sq > 0)')
    ax.legend()
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir,
                    f'{region}_cv_r2_per_neuron.png'), dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrajectoryConfig(
        region='AMG',
        session=None,
        trial_averaged=True,
        peak_align=False,
        n_components=6,
        bin_width=0.050,
        min_epoch_duration=2.0,
    )
    cfg.validate()

    # --------- user settings ----------
    PREDICTORS = ['familiarity', 'age_z', 'dominance_z', 'sociability_z']
    N_SHUFFLES = 100          # 100 for debugging, 500-1000 for paper
    N_SPLITS = 10             # number of random train/test splits to average
    SEED = 42
    TAG = 'shuffle_cv'
    # ----------------------------------

    print("Loading data...")
    df = load_and_filter(cfg)
    tbl = build_regressor_table(groups=('Zombies', 'Best Frans'),
                                exclude_subject=True)
    check_collinearity(tbl, PREDICTORS)

    monkey_list = tbl.index.tolist()
    R_rates, conditions, info, row_meta = build_rank_tensor(df, cfg, monkey_list)
    tbl_ord = tbl.loc[conditions]

    print(f"\n{'='*60}")
    print(f"Cross-validated shuffle test")
    print(f"  Predictors: {PREDICTORS}")
    print(f"  Shuffles: {N_SHUFFLES}, Splits: {N_SPLITS}")
    print(f"{'='*60}\n")

    results = run_shuffle_test(
        df, cfg, conditions, tbl_ord, PREDICTORS, info,
        n_shuffles=N_SHUFFLES, n_splits=N_SPLITS, seed=SEED)

    # Print results
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATED SHUFFLE TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Overall mean CV R-sq:  {results['real_r2_overall']:.6f}  "
          f"(p = {results['p_overall']:.4f})")

    sig = '***' if results['p_overall'] < 0.001 else \
          '**' if results['p_overall'] < 0.01 else \
          '*' if results['p_overall'] < 0.05 else 'n.s.'
    print(f"  Significance: {sig}")

    print(f"\n  Peak CV R-sq over time: {results['real_peak_cv']:.6f}  "
          f"(p = {results['p_peak']:.4f})")
    sig_peak = '***' if results['p_peak'] < 0.001 else \
               '**' if results['p_peak'] < 0.01 else \
               '*' if results['p_peak'] < 0.05 else 'n.s.'
    print(f"  Significance: {sig_peak}")

    # Timepoints with significant CV R-sq
    sig_times = results['p_per_time'] < 0.05
    if sig_times.any():
        time_axis = np.arange(info['n_bins']) * cfg.bin_width
        sig_intervals = time_axis[sig_times]
        print(f"\n  Significant timepoints (p < 0.05): "
              f"{sig_intervals.min():.2f}s - {sig_intervals.max():.2f}s "
              f"({sig_times.sum()} / {len(sig_times)} bins)")

    # Plots
    time_axis = np.arange(info['n_bins']) * cfg.bin_width
    save_dir = os.path.join(PLOT_SAVE_DIR, TAG)

    plot_cv_shuffle_results(results, PREDICTORS, time_axis, cfg,
                            save_dir=save_dir, tag=TAG)

    plot_cv_r2_per_neuron(results, cfg, save_dir=save_dir, tag=TAG)

    # Save results
    save_path = os.path.join(save_dir, f'{cfg.region}_cv_shuffle_results.npz')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(save_path,
             real_r2_time=results['real_r2_time'],
             real_r2_neuron=results['real_r2_neuron'],
             real_r2_overall=results['real_r2_overall'],
             shuf_r2_overall=results['shuf_r2_overall'],
             shuf_r2_time=results['shuf_r2_time'],
             p_overall=results['p_overall'],
             p_per_time=results['p_per_time'],
             p_peak=results['p_peak'],
             real_peak_cv=results['real_peak_cv'],
             shuf_peak_cv=results['shuf_peak_cv'],
             predictors=PREDICTORS,
             n_shuffles=N_SHUFFLES,
             n_splits=N_SPLITS)
    print(f"\nResults saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    main()
