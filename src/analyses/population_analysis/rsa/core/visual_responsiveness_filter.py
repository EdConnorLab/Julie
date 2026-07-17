# visual_responsiveness_filter.py
"""
Visual responsiveness filter for MTL neurons.

Identifies neurons with significant stimulus-driven responses by comparing
firing rates in a post-stimulus response window to an early baseline window
(0–100 ms post-stimulus onset, before visual responses reach MTL).

Usage:
    from analyses.population_analysis.rsa.core.visual_responsiveness_filter import filter_visually_responsive

    # After load_and_filter(cfg), before compute_firing_rates:
    df = filter_visually_responsive(df, cfg)
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ranksums


def compute_trial_spike_counts(df, window):
    """
    Compute per-trial spike count in a time window for each neuron.

    Parameters
    ----------
    df : DataFrame
        Must contain: NeuronID, SpikeTimes, EpochStartStop, TaskField
    window : tuple (start_s, end_s)
        Time window relative to epoch start.

    Returns
    -------
    counts_df : DataFrame
        Columns: NeuronID, TaskField, SpikeCount
    """
    win_start, win_end = window
    rows = []
    for _, row in df.iterrows():
        epoch_start = row['EpochStartStop'][0]
        spk = np.asarray(row['SpikeTimes']) - epoch_start
        count = int(np.sum((spk >= win_start) & (spk < win_end)))
        rows.append({
            'NeuronID': row['NeuronID'],
            'TaskField': row['TaskField'],
            'SpikeCount': count,
        })
    return pd.DataFrame(rows)


def identify_responsive_neurons(df,
                                 baseline_window=(0.0, 0.100),
                                 response_window=(0.150, 0.600),
                                 alpha=0.05,
                                 min_trials=10,
                                 require_increase=True,
                                 verbose=True):
    """
    Identify visually responsive neurons.

    For each neuron, compares spike counts in the response window to
    the baseline window (0–100 ms, before visual responses reach MTL)
    using a Wilcoxon signed-rank test across trials.

    Parameters
    ----------
    df : DataFrame
        Raw trial data with: NeuronID, SpikeTimes, EpochStartStop, TaskField
    baseline_window : tuple
        (start, end) in seconds relative to epoch start.
        Default (0.0, 0.1) = first 100 ms, before MTL visual response onset.
    response_window : tuple
        (start, end) in seconds relative to epoch start.
        Default (0.15, 0.6) = 150–600 ms post-stimulus.
    alpha : float
        Significance threshold for the Wilcoxon test.
    min_trials : int
        Minimum trials needed to run the test.
    require_increase : bool
        If True, only keep neurons where response > baseline (excitatory).
        If False, keep any neuron with a significant difference (incl. suppressed).
    verbose : bool
        Print summary.

    Returns
    -------
    responsive_ids : list of str
        NeuronIDs that pass the visual responsiveness criterion.
    stats_df : DataFrame
        Per-neuron statistics: NeuronID, baseline_mean, response_mean,
        ratio, p_value, responsive.
    """
    # Scale spike counts to per-second rates for comparability across
    # windows of different durations
    baseline_dur = baseline_window[1] - baseline_window[0]
    response_dur = response_window[1] - response_window[0]

    if verbose:
        print(f"\n  Visual responsiveness filter:")
        print(f"    Baseline:  {baseline_window[0]*1000:.0f}–{baseline_window[1]*1000:.0f} ms")
        print(f"    Response:  {response_window[0]*1000:.0f}–{response_window[1]*1000:.0f} ms")

    # Compute trial-level spike counts in both windows
    baseline_counts = compute_trial_spike_counts(df, baseline_window)
    response_counts = compute_trial_spike_counts(df, response_window)

    # Merge on (NeuronID, TaskField) to get paired observations
    merged = baseline_counts.merge(
        response_counts, on=['NeuronID', 'TaskField'],
        suffixes=('_baseline', '_response'))

    neuron_ids = sorted(merged['NeuronID'].unique())
    stats_rows = []

    for nid in neuron_ids:
        ndf = merged[merged['NeuronID'] == nid]

        if len(ndf) < min_trials:
            stats_rows.append({
                'NeuronID': nid,
                'n_trials': len(ndf),
                'baseline_rate': np.nan,
                'response_rate': np.nan,
                'ratio': np.nan,
                'p_value': np.nan,
                'responsive': False,
                'reason': 'too_few_trials',
            })
            continue

        bl = ndf['SpikeCount_baseline'].values / baseline_dur  # convert to Hz
        rsp = ndf['SpikeCount_response'].values / response_dur

        bl_mean = bl.mean()
        rsp_mean = rsp.mean()
        ratio = rsp_mean / bl_mean if bl_mean > 0 else np.inf

        # Wilcoxon signed-rank test (paired, nonparametric)
        diff = rsp - bl
        if np.all(diff == 0):
            p_val = 1.0
        else:
            try:
                _, p_val = wilcoxon(diff, alternative='two-sided')
            except ValueError:
                # All differences identical (e.g., all zero baseline)
                p_val = 1.0

        is_sig = p_val < alpha
        is_increase = rsp_mean > bl_mean

        if require_increase:
            responsive = is_sig and is_increase
        else:
            responsive = is_sig

        stats_rows.append({
            'NeuronID': nid,
            'n_trials': len(ndf),
            'baseline_rate': round(bl_mean, 2),
            'response_rate': round(rsp_mean, 2),
            'ratio': round(ratio, 3),
            'p_value': round(p_val, 5),
            'responsive': responsive,
            'reason': 'pass' if responsive else (
                'not_sig' if not is_sig else 'decrease_only'),
        })

    stats_df = pd.DataFrame(stats_rows)
    responsive_ids = stats_df[stats_df['responsive']]['NeuronID'].tolist()

    if verbose:
        n_total = len(neuron_ids)
        n_resp = len(responsive_ids)
        n_few = (stats_df['reason'] == 'too_few_trials').sum()

        # Region breakdown
        amg_total = stats_df['NeuronID'].str.contains('AMG').sum()
        amg_resp = stats_df[stats_df['responsive']]['NeuronID'].str.contains('AMG').sum()
        er_total = stats_df['NeuronID'].str.contains('ER').sum()
        er_resp = stats_df[stats_df['responsive']]['NeuronID'].str.contains('ER').sum()

        print(f"    Total neurons:      {n_total}")
        print(f"    Too few trials:     {n_few}")
        print(f"    Responsive:         {n_resp}/{n_total} ({100*n_resp/max(n_total,1):.1f}%)")
        print(f"      AMG:              {amg_resp}/{amg_total}")
        print(f"      ER:               {er_resp}/{er_total}")

    return responsive_ids, stats_df


def filter_visually_responsive(df, cfg, return_stats=False):
    """
    Filter a trial DataFrame to keep only visually responsive neurons.

    Convenience wrapper: call identify_responsive_neurons, then filter df.

    Parameters
    ----------
    df : DataFrame
        Trial-level data (all neurons).
    cfg : RSAConfig or SocialRSAConfig or SocialEncodingConfig
        Must have `visual_responsiveness_filter` (bool) and optionally
        `vr_baseline_window`, `vr_response_window`, `vr_alpha`.
    return_stats : bool
        If True, also return the per-neuron stats DataFrame.

    Returns
    -------
    df_filtered : DataFrame
        Only trials from responsive neurons.
    stats_df : DataFrame (only if return_stats=True)
    """
    if not getattr(cfg, 'visual_responsiveness_filter', False):
        if return_stats:
            return df, None
        return df

    baseline_window = getattr(cfg, 'vr_baseline_window', (0.0, 0.100))
    response_window = getattr(cfg, 'vr_response_window', (0.150, 0.600))
    alpha = getattr(cfg, 'vr_alpha', 0.05)
    require_increase = getattr(cfg, 'vr_require_increase', True)

    responsive_ids, stats_df = identify_responsive_neurons(
        df,
        baseline_window=baseline_window,
        response_window=response_window,
        alpha=alpha,
        require_increase=require_increase,
        verbose=True,
    )

    n_before = df['NeuronID'].nunique()
    df_filtered = df[df['NeuronID'].isin(responsive_ids)].reset_index(drop=True)
    n_after = df_filtered['NeuronID'].nunique()
    print(f"    Filtered: {n_before} → {n_after} neurons")

    if return_stats:
        return df_filtered, stats_df
    return df_filtered
