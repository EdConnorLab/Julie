# data_diagnostics.py
"""
Data diagnostics for RSA analysis pipeline.

Generates summary statistics and visualizations:
  1. Trial counts per identity (overall + per session)
  2. Trial counts per group (overall + per session)
  3. Neuron counts per session per region
  4. Firing rate distributions per identity / per group
  5. Session-level balance check (identity × session heatmap)
  6. Session firing-rate audit (baseline / selectivity / hot-cell provenance)
  7. Pseudopop session-dropping preview (which sessions run_rsa_pseudopop keeps
     at a given per-session min_reps floor)

Run from the same directory as run_rsa.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as sps
import os

from analyses.population_analysis.state_space.data_loading import load_and_filter
from analyses.population_analysis.rsa.core.rsa_core import compute_firing_rates
from  analyses.population_analysis.rsa.core.rsa_config import RSAConfig

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"
SAVE_DIR = "data_diagnostics"

GROUP_COLORS = {
    'Zombies':         '#9467bd',
    'Best Frans':      '#d62728',
    'Instigators':     '#2ca02c',
    'Stranger Things': '#1f77b4',
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _name_to_group(info_df):
    return dict(zip(info_df['Name'].astype(str), info_df['Group Name']))


def get_trial_counts(df):
    """
    Compute trial counts per identity per session.

    All neurons in a session see the same stimulus presentations, so
    the count per (session, identity, neuron) should be constant across
    neurons.  We take the median across neurons as a safeguard.
    """
    per_neuron = (df.groupby(['session', 'MonkeyName', 'NeuronID'])
                  .size()
                  .reset_index(name='n_trials'))
    trial_counts = (per_neuron
                    .groupby(['session', 'MonkeyName'])['n_trials']
                    .median()
                    .round()
                    .astype(int)
                    .reset_index())
    return trial_counts


# ═══════════════════════════════════════════════════════════════════════
# 1. Trial counts per identity
# ═══════════════════════════════════════════════════════════════════════

def report_trial_counts_per_identity(trial_counts, info_df, region, save_dir):
    ntg = _name_to_group(info_df)
    tc = trial_counts.copy()
    tc['Group'] = tc['MonkeyName'].map(ntg)

    summary = (tc.groupby(['MonkeyName', 'Group'])['n_trials']
               .agg(['sum', 'mean', 'std', 'min', 'max', 'count'])
               .round(1)
               .reset_index()
               .sort_values(['Group', 'MonkeyName']))
    summary.columns = ['Identity', 'Group', 'total_trials',
                        'mean_per_sess', 'std_per_sess',
                        'min_per_sess', 'max_per_sess', 'n_sessions']

    print(f"\n{'='*80}")
    print(f"TRIAL COUNTS PER IDENTITY | {region}")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    # Flag low counts
    low = summary[summary['min_per_sess'] < 3]
    if len(low):
        print(f"\n⚠  Identities with < 3 trials in at least one session:")
        for _, r in low.iterrows():
            print(f"   {r['Identity']} ({r['Group']}): min = {r['min_per_sess']:.0f}")

    # ── Bar plot ──
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(summary))
    colors = [GROUP_COLORS.get(g, 'gray') for g in summary['Group']]
    ax.bar(x, summary['mean_per_sess'], yerr=summary['std_per_sess'],
           color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Identity'], rotation=45, ha='right')
    ax.set_ylabel('Mean trials per session (± SD)')
    ax.set_title(f'Trial counts per identity | {region}')
    handles = [Patch(color=GROUP_COLORS.get(g, 'gray'), label=g)
               for g in summary['Group'].unique()]
    ax.legend(handles=handles, loc='upper right')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'trial_counts_per_identity_{region}.png'),
                dpi=150, bbox_inches='tight')

    return summary


# ═══════════════════════════════════════════════════════════════════════
# 2. Trial counts per group
# ═══════════════════════════════════════════════════════════════════════

def report_trial_counts_per_group(trial_counts, info_df, region, save_dir):
    ntg = _name_to_group(info_df)
    tc = trial_counts.copy()
    tc['Group'] = tc['MonkeyName'].map(ntg)

    # Total trials per group per session
    grp_sess = (tc.groupby(['session', 'Group'])['n_trials']
                .sum()
                .reset_index())
    summary = (grp_sess.groupby('Group')['n_trials']
               .agg(['mean', 'std', 'min', 'max', 'count'])
               .round(1)
               .reset_index())
    summary.columns = ['Group', 'mean_total', 'std_total',
                        'min_total', 'max_total', 'n_sessions']

    print(f"\n{'='*80}")
    print(f"TRIAL COUNTS PER GROUP (summed over identities) | {region}")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    # Also: mean trials per identity per group (to check balance)
    per_id = (tc.groupby(['MonkeyName', 'Group'])['n_trials']
              .mean()
              .reset_index())
    grp_balance = (per_id.groupby('Group')['n_trials']
                   .agg(['mean', 'std', 'min', 'max', 'count'])
                   .round(1)
                   .reset_index())
    grp_balance.columns = ['Group', 'mean_per_id', 'std_per_id',
                            'min_per_id', 'max_per_id', 'n_identities']
    print(f"\nMean trials per identity (averaged over sessions):")
    print(grp_balance.to_string(index=False))

    # ── Box plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(grp_sess['Group'].unique())
    data = [grp_sess[grp_sess['Group'] == g]['n_trials'].values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(GROUP_COLORS.get(g, 'gray'))
        patch.set_alpha(0.7)
    ax.set_ylabel('Total trials per session (summed over identities)')
    ax.set_title(f'Trial counts per group | {region}')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'trial_counts_per_group_{region}.png'),
                dpi=150, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════
# 3. Neuron counts per session
# ═══════════════════════════════════════════════════════════════════════

def report_neuron_counts(df, region, save_dir):
    nc = (df.groupby('session')['NeuronID']
          .nunique()
          .reset_index()
          .rename(columns={'NeuronID': 'n_neurons'})
          .sort_values('session'))

    print(f"\n{'='*80}")
    print(f"NEURON COUNTS PER SESSION | {region}")
    print(f"{'='*80}")
    print(nc.to_string(index=False))
    print(f"\nTotal unique neurons (pseudo-pop): {df['NeuronID'].nunique()}")
    print(f"Sessions: {nc['session'].nunique()}")
    print(f"Neurons/session: mean={nc['n_neurons'].mean():.1f}, "
          f"min={nc['n_neurons'].min()}, max={nc['n_neurons'].max()}")

    # ── Bar plot ──
    fig, ax = plt.subplots(figsize=(max(10, len(nc) * 0.5), 5))
    ax.bar(range(len(nc)), nc['n_neurons'], alpha=0.8, color='steelblue')
    ax.set_xticks(range(len(nc)))
    ax.set_xticklabels(nc['session'], rotation=90, fontsize=6)
    ax.set_ylabel('# neurons')
    ax.set_title(f'Neurons per session | {region}')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'neuron_counts_{region}.png'),
                dpi=150, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════
# 4. Firing rate distributions
# ═══════════════════════════════════════════════════════════════════════

def report_firing_rate_distributions(df, info_df, cfg, region, save_dir):
    """
    Compute mean firing rate per neuron per identity (same as RSA pipeline)
    and show distributions per identity and per group.
    """
    ntg = _name_to_group(info_df)
    known = set(info_df['Name'].astype(str))
    identities = sorted(set(df['MonkeyName'].unique()) & known)
    sessions = df['session'].unique()

    # Collect per-neuron mean rates across all sessions (pseudo-pop)
    all_rates = {mid: [] for mid in identities}
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        rate_mat, valid_ids, _ = compute_firing_rates(
            sess_df, identities, cfg.window, min_reps=1)
        for i, mid in enumerate(valid_ids):
            all_rates[mid].extend(rate_mat[i, :].tolist())

    # Sort by group
    id_grp = [(mid, ntg.get(mid, 'Unknown')) for mid in identities
              if len(all_rates[mid]) > 0]
    id_grp.sort(key=lambda x: (x[1], x[0]))
    sorted_ids = [x[0] for x in id_grp]
    sorted_groups = [x[1] for x in id_grp]

    # ── Per-identity box plot ──
    fig, ax = plt.subplots(figsize=(16, 6))
    data = [all_rates[mid] for mid in sorted_ids]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, g in zip(bp['boxes'], sorted_groups):
        patch.set_facecolor(GROUP_COLORS.get(g, 'gray'))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(sorted_ids) + 1))
    ax.set_xticklabels(sorted_ids, rotation=45, ha='right', fontsize=7)
    for tick, g in zip(ax.get_xticklabels(), sorted_groups):
        tick.set_color(GROUP_COLORS.get(g, 'gray'))
    ax.set_ylabel('Firing rate (spk/s)')
    ax.set_title(f'Firing rate distributions per identity | {region}')
    handles = [Patch(color=GROUP_COLORS.get(g, 'gray'), label=g)
               for g in dict.fromkeys(sorted_groups)]
    ax.legend(handles=handles, loc='upper right')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'firing_rates_per_identity_{region}.png'),
                dpi=150, bbox_inches='tight')

    # ── Per-group box plot ──
    group_data = {}
    for mid, g in zip(sorted_ids, sorted_groups):
        group_data.setdefault(g, []).extend(all_rates[mid])

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(group_data.keys())
    data = [group_data[g] for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True,
                    showfliers=False, whis=[5, 95])
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(GROUP_COLORS.get(g, 'gray'))
        patch.set_alpha(0.7)
    ax.set_ylabel('Firing rate (spk/s)')
    ax.set_title(f'Firing rate distributions per group | {region}')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'firing_rates_per_group_{region}.png'),
                dpi=150, bbox_inches='tight')

    # ── Print summary stats ──
    print(f"\n{'='*80}")
    print(f"FIRING RATE SUMMARY | {region}")
    print(f"{'='*80}")
    rows = []
    for mid, g in zip(sorted_ids, sorted_groups):
        r = all_rates[mid]
        rows.append(dict(Identity=mid, Group=g, n_neurons=len(r),
                         mean=np.mean(r), median=np.median(r),
                         std=np.std(r), q5=np.percentile(r, 5),
                         q95=np.percentile(r, 95)))
    fr_df = pd.DataFrame(rows).round(2)
    print(fr_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# 5. Session-level balance check
# ═══════════════════════════════════════════════════════════════════════

def report_session_balance(trial_counts, info_df, region, save_dir):
    """Heatmap: identity × session trial counts."""
    ntg = _name_to_group(info_df)

    pivot = trial_counts.pivot_table(
        index='MonkeyName', columns='session',
        values='n_trials', fill_value=0)

    # Sort by group
    pivot['_group'] = pivot.index.map(ntg)
    pivot = pivot.sort_values(['_group', pivot.index.name])
    groups_sorted = pivot['_group'].tolist()
    pivot = pivot.drop(columns='_group')

    # ── Heatmap ──
    fig, ax = plt.subplots(
        figsize=(max(12, len(pivot.columns) * 0.4),
                 max(6, len(pivot) * 0.35)))

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.max() * 0.6 else 'black'
            if val == 0:
                color = 'red'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=5, color=color, fontweight='bold' if val == 0 else 'normal')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    for tick, g in zip(ax.get_yticklabels(), groups_sorted):
        tick.set_color(GROUP_COLORS.get(g, 'gray'))

    # Group boundaries
    prev = groups_sorted[0]
    for i, g in enumerate(groups_sorted):
        if g != prev:
            ax.axhline(i - 0.5, color='black', lw=1.5)
            prev = g

    fig.colorbar(im, ax=ax, shrink=0.7, label='# trials')
    ax.set_title(f'Session × Identity trial counts | {region}')
    ax.set_xlabel('Session')
    ax.set_ylabel('Identity')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'session_balance_{region}.png'),
                dpi=150, bbox_inches='tight')

    # ── Print missing-data summary ──
    print(f"\n{'='*80}")
    print(f"SESSION BALANCE | {region}")
    print(f"{'='*80}")
    print(f"Matrix: {pivot.shape[0]} identities × {pivot.shape[1]} sessions")
    zero_cells = (pivot == 0).sum().sum()
    total_cells = pivot.shape[0] * pivot.shape[1]
    print(f"Missing cells (0 trials): {zero_cells}/{total_cells} "
          f"({zero_cells / total_cells * 100:.1f}%)")

    missing = (pivot == 0).sum(axis=1)
    ids_with_gaps = missing[missing > 0]
    if len(ids_with_gaps):
        print(f"\nIdentities missing from some sessions:")
        for mid, n_miss in ids_with_gaps.items():
            g = ntg.get(mid, '?')
            print(f"  {mid} ({g}): missing from {n_miss}/{pivot.shape[1]} sessions")

    # Sessions where some identity has very few trials
    low_mask = (pivot > 0) & (pivot < 3)
    low_counts = low_mask.sum(axis=1)
    ids_low = low_counts[low_counts > 0]
    if len(ids_low):
        print(f"\nIdentities with 1-2 trials in some sessions:")
        for mid, n_low in ids_low.items():
            g = ntg.get(mid, '?')
            print(f"  {mid} ({g}): {n_low} sessions with < 3 trials")


# ═══════════════════════════════════════════════════════════════════════
# 6. Session firing-rate audit
# ═══════════════════════════════════════════════════════════════════════
# Question: when population-level firing rates differ across identities, is
# it (a) a uniform population effect, (b) an artifact of one or two outlier
# sessions whose cells happen to fire a lot, or (c) a real but sparse effect
# carried by a subset of "identity cells"?  This block addresses all three:
#
#   6a. Session × identity mean-FR heatmap  (raw + within-session z-scored)
#       → look for hot rows (a session that's hot overall) vs. hot columns
#         (a real population-wide identity preference) vs. isolated hot
#         cells (an effect specific to one session × identity).
#
#   6b. Per-session firing rate distribution, boxplot sorted by median, with
#       outlier sessions flagged (Tukey > Q3 + 1.5·IQR on session medians).
#
#   6c. Per-neuron Kruskal-Wallis test across identities (trial-level rates;
#       non-parametric — appropriate for skewed / zero-heavy spike-count data
#       where ANOVA's normality / homoscedasticity assumptions fail) +
#       modulation index (max − mean_others) / (max + mean_others).
#       → fraction of "tuned" cells per session, distribution of preferred
#         identities per session.
#
#   6d. Top-K "hot cells" per identity (ranked by within-neuron z-score, so
#       baseline firing rate is controlled).  Where do they come from?
#       Concentration in 1–2 sessions = red flag.
# ═══════════════════════════════════════════════════════════════════════

def _compute_trial_rates(sess_df, window):
    """
    Trial-level firing rate (spk/s) per row of sess_df.
    Window (w0, w1) is in seconds relative to EpochStart (= EpochStartStop[0]).
    """
    w0, w1 = window
    duration = w1 - w0
    rates = np.empty(len(sess_df), dtype=float)
    spike_lists = sess_df['SpikeTimes'].values
    epochs = sess_df['EpochStartStop'].values
    for i in range(len(sess_df)):
        t0 = epochs[i][0] + w0
        t1 = epochs[i][0] + w1
        s = np.asarray(spike_lists[i])
        rates[i] = np.sum((s >= t0) & (s < t1)) / duration
    return rates


def _build_trial_rate_df(df, identities, window):
    """
    Trial-level rates with metadata.
    Returns DataFrame: session, NeuronID, MonkeyName, rate.
    """
    chunks = []
    for sess, sess_df in df.groupby('session', sort=True):
        rates = _compute_trial_rates(sess_df, window)
        chunk = sess_df[['NeuronID', 'MonkeyName']].copy()
        chunk['session'] = sess
        chunk['rate'] = rates
        chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    out = out[out['MonkeyName'].isin(identities)]
    return out


# -----------------------------------------------------------------------
# 6a + 6b: baseline session-level FR check
# -----------------------------------------------------------------------

def report_session_fr_baseline(trial_rate_df, info_df, region, save_dir):
    """6a + 6b: are some sessions just overall hotter/colder?"""
    ntg = _name_to_group(info_df)
    identities = sorted(trial_rate_df['MonkeyName'].unique(),
                        key=lambda m: (ntg.get(m, 'zzz'), m))
    sessions = sorted(trial_rate_df['session'].unique())

    # (session, neuron, identity) -> mean rate
    cell_id_mean = (trial_rate_df
                    .groupby(['session', 'NeuronID', 'MonkeyName'])['rate']
                    .mean()
                    .reset_index(name='rate'))

    # ── 6a heatmaps: session × identity mean FR ──
    sess_id_mean = (cell_id_mean
                    .groupby(['session', 'MonkeyName'])['rate']
                    .mean()
                    .unstack('MonkeyName'))
    # reorder identity columns by group
    sess_id_mean = sess_id_mean.reindex(columns=[m for m in identities
                                                 if m in sess_id_mean.columns])
    sess_id_mean = sess_id_mean.reindex(index=sessions)

    # z-score within session (row-wise) — removes session baseline
    row_mu = sess_id_mean.mean(axis=1)
    row_sd = sess_id_mean.std(axis=1).replace(0, np.nan)
    sess_id_z = sess_id_mean.sub(row_mu, axis=0).div(row_sd, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(identities) * 0.7),
                                            max(5, len(sessions) * 0.3)))
    im0 = axes[0].imshow(sess_id_mean.values, aspect='auto', cmap='YlOrRd')
    axes[0].set_title(f'Session × Identity: mean FR (spk/s) | {region}')
    im1 = axes[1].imshow(sess_id_z.values, aspect='auto',
                         cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title(f'Session × Identity: z-scored within session | {region}')

    for ax, im in zip(axes, [im0, im1]):
        ax.set_xticks(range(len(sess_id_mean.columns)))
        ax.set_xticklabels(sess_id_mean.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(sess_id_mean.index)))
        ax.set_yticklabels(sess_id_mean.index, fontsize=6)
        for tick, mid in zip(ax.get_xticklabels(), sess_id_mean.columns):
            tick.set_color(GROUP_COLORS.get(ntg.get(mid, '?'), 'gray'))
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0].set_ylabel('Session')

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'session_identity_fr_heatmap_{region}.png'),
                dpi=150, bbox_inches='tight')

    # ── 6b boxplot of (neuron, identity) mean rates per session ──
    sess_meds = (cell_id_mean.groupby('session')['rate']
                 .median().sort_values())
    sorted_sess = sess_meds.index.tolist()

    # Tukey outlier flagging on session medians
    q1, q3 = sess_meds.quantile(0.25), sess_meds.quantile(0.75)
    iqr = q3 - q1
    hi_thr, lo_thr = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    hot_sess = sess_meds[sess_meds > hi_thr].index.tolist()
    cold_sess = sess_meds[sess_meds < lo_thr].index.tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_sess) * 0.35), 5))
    data = [cell_id_mean[cell_id_mean['session'] == s]['rate'].values
            for s in sorted_sess]
    bp = ax.boxplot(data, labels=sorted_sess, patch_artist=True,
                    showfliers=False, whis=[5, 95])
    for patch, s in zip(bp['boxes'], sorted_sess):
        if s in hot_sess:
            patch.set_facecolor('#d62728'); patch.set_alpha(0.7)
        elif s in cold_sess:
            patch.set_facecolor('#1f77b4'); patch.set_alpha(0.7)
        else:
            patch.set_facecolor('lightgray'); patch.set_alpha(0.7)
    ax.axhline(sess_meds.median(), ls=':', color='k', lw=1,
               label=f'grand median = {sess_meds.median():.2f}')
    ax.set_xticklabels(sorted_sess, rotation=90, fontsize=6)
    ax.set_ylabel('Mean FR per (neuron, identity) [spk/s]')
    ax.set_title(f'Per-session firing rate distribution (sorted) | {region}')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'session_fr_distribution_{region}.png'),
                dpi=150, bbox_inches='tight')

    print(f"\n{'='*80}")
    print(f"6a/b. SESSION FIRING-RATE BASELINE | {region}")
    print(f"{'='*80}")
    print(f"Sessions: {len(sorted_sess)} | grand median FR = "
          f"{sess_meds.median():.2f} spk/s")
    print(f"Session-median range: {sess_meds.min():.2f} → {sess_meds.max():.2f} "
          f"({sess_meds.max() / max(sess_meds.min(), 0.01):.1f}× fold)")
    if hot_sess:
        print(f"\n⚠  HOT outlier sessions (median > Q3 + 1.5·IQR = {hi_thr:.2f}):")
        for s in hot_sess:
            print(f"   {s}: median = {sess_meds[s]:.2f} spk/s")
    if cold_sess:
        print(f"\n⚠  COLD outlier sessions (median < Q1 − 1.5·IQR = {lo_thr:.2f}):")
        for s in cold_sess:
            print(f"   {s}: median = {sess_meds[s]:.2f} spk/s")
    if not hot_sess and not cold_sess:
        print("\nNo Tukey-outlier sessions on session-median firing rate.")

    return cell_id_mean, sess_id_mean


# -----------------------------------------------------------------------
# 6c + 6d: per-neuron selectivity and hot-cell provenance
# -----------------------------------------------------------------------

def report_neuron_selectivity(trial_rate_df, info_df, region, save_dir,
                              alpha=0.05, top_frac=0.10, min_trials_per_id=3):
    """
    6c. Kruskal-Wallis test across identities per (session, neuron) using
        trial-level rates (non-parametric: spike counts are skewed / zero-heavy
        / heteroscedastic, which violates ANOVA assumptions).  Returns a table
        of H-stats, p-values, preferred identity, and modulation index.
    6d. For each identity, the top fraction of neurons ranked by within-neuron
        z-score for that identity → where do they come from (which sessions)?
    """
    ntg = _name_to_group(info_df)

    # ─── 6c. Per-neuron Kruskal-Wallis ───
    rows = []
    for (sess, nid), g in trial_rate_df.groupby(['session', 'NeuronID']):
        groups, labels = [], []
        for k, v in g.groupby('MonkeyName')['rate']:
            if len(v) >= min_trials_per_id:
                groups.append(v.values)
                labels.append(k)
        if len(groups) < 2:
            continue
        try:
            H, p = sps.kruskal(*groups)
        except Exception:
            H, p = np.nan, np.nan
        means = np.array([gr.mean() for gr in groups])
        i_best = int(np.argmax(means))
        pref = labels[i_best]
        mx = means[i_best]
        others_mean = np.mean(np.delete(means, i_best)) if len(means) > 1 else np.nan
        denom = mx + others_mean
        mod_idx = (mx - others_mean) / denom if denom > 0 else np.nan
        rows.append(dict(session=sess, NeuronID=nid, H=H, p=p,
                         preferred=pref, n_identities=len(groups),
                         mean_rate=np.mean(means),
                         modulation=mod_idx))
    sel = pd.DataFrame(rows)
    if len(sel) == 0:
        print(f"\n[6c] No neurons with enough trials/identity in {region}.")
        return sel

    sel['tuned'] = sel['p'] < alpha

    # ── Plot: p-value histogram + per-session tuned fraction ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(sel['p'].dropna(), bins=20, color='steelblue',
                 edgecolor='black', alpha=0.85)
    axes[0].axvline(alpha, color='red', ls='--',
                    label=f'α = {alpha}')
    axes[0].axhline(len(sel) / 20, color='gray', ls=':',
                    label='uniform-null')
    axes[0].set_xlabel('Kruskal-Wallis p-value (identity)')
    axes[0].set_ylabel('# neurons')
    axes[0].set_title(f'Per-neuron identity-tuning p-values | {region}')
    axes[0].legend(fontsize=8)

    by_sess = (sel.groupby('session')
               .agg(n_neurons=('NeuronID', 'count'),
                    n_tuned=('tuned', 'sum'))
               .reset_index())
    by_sess['frac_tuned'] = by_sess['n_tuned'] / by_sess['n_neurons']
    by_sess = by_sess.sort_values('frac_tuned')

    axes[1].bar(range(len(by_sess)), by_sess['frac_tuned'],
                color='steelblue', alpha=0.85)
    axes[1].axhline(alpha, color='red', ls='--',
                    label=f'chance = {alpha}')
    axes[1].set_xticks(range(len(by_sess)))
    axes[1].set_xticklabels(by_sess['session'], rotation=90, fontsize=6)
    axes[1].set_ylabel(f'Fraction tuned (p < {alpha})')
    axes[1].set_title(f'Per-session fraction of identity-tuned cells | {region}')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'neuron_selectivity_kw_{region}.png'),
                dpi=150, bbox_inches='tight')

    # ── Print summary ──
    print(f"\n{'='*80}")
    print(f"6c. PER-NEURON IDENTITY TUNING (Kruskal-Wallis) | {region}")
    print(f"{'='*80}")
    print(f"Neurons tested: {len(sel)}")
    print(f"Tuned (p < {alpha}): {sel['tuned'].sum()} "
          f"({sel['tuned'].mean()*100:.1f}%)  "
          f"[chance ≈ {alpha*100:.0f}%]")
    print(f"Modulation index: median = {sel['modulation'].median():.3f}, "
          f"95th pct = {sel['modulation'].quantile(0.95):.3f}")

    med = by_sess['frac_tuned'].median()
    mad = sps.median_abs_deviation(by_sess['frac_tuned'])
    hot_thr = med + 3 * mad
    hot_tuned = by_sess[by_sess['frac_tuned'] > hot_thr]
    if len(hot_tuned):
        print(f"\n⚠  Sessions with unusually high tuned fraction "
              f"(> median + 3·MAD = {hot_thr:.2f}):")
        for _, r in hot_tuned.iterrows():
            print(f"   {r['session']}: {r['n_tuned']:.0f}/"
                  f"{r['n_neurons']:.0f} ({r['frac_tuned']*100:.1f}%)")

    # ─── 6d. Hot-cell provenance per identity ───
    # MultiIndex (session, NeuronID) — avoids the string-key bug where
    # pandas .str.split('||') interpreted '||' as a regex pattern.
    cell_id = (trial_rate_df
               .groupby(['session', 'NeuronID', 'MonkeyName'])['rate']
               .mean()
               .reset_index())
    pv = cell_id.pivot_table(index=['session', 'NeuronID'],
                             columns='MonkeyName', values='rate')
    # within-row z-score (preference, not baseline rate)
    z = pv.sub(pv.mean(axis=1), axis=0).div(
        pv.std(axis=1).replace(0, np.nan), axis=0)

    sessions_list = sorted(trial_rate_df['session'].unique())
    sess_of_row = pv.index.get_level_values('session')

    K = max(1, int(round(len(z) * top_frac)))
    rows = []
    for mid in z.columns:
        col = z[mid].dropna().sort_values(ascending=False)
        if len(col) == 0:
            rows.append(dict(identity=mid, group=ntg.get(mid, '?'),
                             top1_session_share=np.nan, entropy=np.nan,
                             max_entropy=np.log2(len(sessions_list)),
                             concentration=np.nan, counts={}))
            continue
        top_cells = col.head(K).index
        top_sess = pd.Index([s for (s, _) in top_cells])
        counts = top_sess.value_counts().reindex(sessions_list, fill_value=0)
        total = counts.sum()
        frac = counts / total if total > 0 else counts * 0.0
        if total > 0:
            p_ = frac.values[frac.values > 0]
            ent = float(-np.sum(p_ * np.log2(p_)))
            max_ent = float(np.log2(len(sessions_list)))
            top1_share = float(frac.max())
            conc = 1 - (ent / max_ent) if max_ent > 0 else np.nan
        else:
            ent, max_ent, top1_share, conc = np.nan, np.log2(len(sessions_list)), np.nan, np.nan
        rows.append(dict(identity=mid, group=ntg.get(mid, '?'),
                         top1_session_share=top1_share,
                         entropy=ent, max_entropy=max_ent,
                         concentration=conc, counts=counts.to_dict()))
    prov = pd.DataFrame(rows)

    # Heatmap of identity × session top-cell share
    mat = pd.DataFrame({r['identity']: pd.Series(r['counts'])
                        for _, r in prov.iterrows()}).T
    mat = mat.reindex(columns=sessions_list, fill_value=0)
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    mat_frac = mat.div(row_sums, axis=0)
    ordered = sorted(mat_frac.index, key=lambda m: (ntg.get(m, 'zzz'), m))
    mat_frac = mat_frac.reindex(index=ordered)

    fig, ax = plt.subplots(figsize=(max(12, len(sessions_list) * 0.35),
                                    max(5, len(ordered) * 0.3)))
    max_val = (np.nanmax(mat_frac.values)
               if np.any(np.isfinite(mat_frac.values)) else 1.0)
    im = ax.imshow(mat_frac.values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=min(1.0, max_val * 1.2 + 0.01))
    ax.set_xticks(range(len(sessions_list)))
    ax.set_xticklabels(sessions_list, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)
    for tick, mid in zip(ax.get_yticklabels(), ordered):
        tick.set_color(GROUP_COLORS.get(ntg.get(mid, '?'), 'gray'))
    fig.colorbar(im, ax=ax, shrink=0.7,
                 label=f'Fraction of top-{int(top_frac*100)}% selective cells')
    ax.set_title(f'Top-{int(top_frac*100)}% selective cells per identity: '
                 f'session provenance | {region}\n'
                 f'(uniform spread → ~{1/len(sessions_list):.2f} per session)')
    ax.set_xlabel('Session')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'hot_cell_provenance_{region}.png'),
                dpi=150, bbox_inches='tight')

    print(f"\n{'='*80}")
    print(f"6d. HOT-CELL PROVENANCE | {region}  (top {int(top_frac*100)}% "
          f"selective cells per identity, K = {K})")
    print(f"{'='*80}")
    print(f"Uniform null: each session contributes ~{1/len(sessions_list):.2%}.")
    print(prov[['identity', 'group', 'top1_session_share', 'concentration']]
          .sort_values('top1_session_share', ascending=False)
          .round(3).to_string(index=False))
    flagged = prov[prov['top1_session_share'] > 0.40]
    if len(flagged):
        print(f"\n⚠  Identities whose 'preference' is >40% concentrated in 1 session:")
        for _, r in flagged.iterrows():
            top_sess = max(r['counts'], key=r['counts'].get)
            print(f"   {r['identity']} ({r['group']}): "
                  f"{r['top1_session_share']*100:.0f}% from {top_sess}")

    return sel, prov


# ═══════════════════════════════════════════════════════════════════════
# 7. Pseudo-population session-dropping preview
# ═══════════════════════════════════════════════════════════════════════

def report_pseudopop_session_drop(trial_counts, info_df, region, save_dir,
                                  floors=(3, 5, 7, 10), highlight=7):
    """
    Preview run_rsa_pseudopop's session-dropping.

    run_rsa_pseudopop keeps only sessions in which EVERY candidate identity
    (present in all sessions) has >= min_reps_per_monkey trials, so every cell
    of the pseudo-population is backed by >= min_reps trials and all identities
    share the same neurons within a session.  This reports, per group, how many
    sessions survive at each floor and which sessions the `highlight` floor
    drops (and why).
    """
    ntg = _name_to_group(info_df)
    pivot = trial_counts.pivot_table(index='MonkeyName', columns='session',
                                     values='n_trials', fill_value=0)
    sessions = list(pivot.columns)
    n_sess = len(sessions)

    print(f"\n{'='*80}")
    print(f"7. PSEUDOPOP SESSION-DROPPING PREVIEW | {region}  "
          f"(min_reps highlighted = {highlight})")
    print(f"{'='*80}")
    print("Rule: keep sessions where every candidate identity (present in all "
          "sessions)\n      has >= min_reps trials.")

    groups = sorted({ntg.get(m, 'Unknown') for m in pivot.index})
    kept_curve = {}   # group -> {floor: n_kept}
    for group in groups:
        ids = [m for m in pivot.index if ntg.get(m) == group]
        present_all = [m for m in ids if (pivot.loc[m] > 0).all()]
        if len(present_all) < 3:
            print(f"\n  {group}: only {len(present_all)} identities present in "
                  f"all sessions — skipped.")
            continue
        sub = pivot.loc[present_all]
        print(f"\n  {group}: {len(present_all)} identities present in all "
              f"{n_sess} sessions")
        kept_curve[group] = {}
        for f in floors:
            keep_mask = (sub >= f).all(axis=0)
            kept_curve[group][f] = int(keep_mask.sum())
            marker = '   <-- min_reps' if f == highlight else ''
            print(f"      floor {f:>2}/session -> keep "
                  f"{int(keep_mask.sum()):>2}/{n_sess} sessions{marker}")
            if f == highlight:
                for s in sessions:
                    if not keep_mask[s]:
                        thin = {m: int(sub.loc[m, s]) for m in present_all
                                if sub.loc[m, s] < f}
                        print(f"          drop {s}: " +
                              ", ".join(f"{m}={c}" for m, c in sorted(thin.items())))

    # sessions-kept-vs-floor curve per group
    if kept_curve:
        fig, ax = plt.subplots(figsize=(7, 5))
        for group, curve in kept_curve.items():
            xs = sorted(curve)
            ax.plot(xs, [curve[f] for f in xs], 'o-',
                    color=GROUP_COLORS.get(group, 'gray'), label=group)
        ax.axvline(highlight, ls=':', color='k', alpha=0.6)
        ax.set_xlabel('per-session trial floor (min_reps_per_monkey)')
        ax.set_ylabel('# sessions kept')
        ax.set_title(f'Pseudopop sessions kept vs floor | {region}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'pseudopop_session_drop_{region}.png'),
                    dpi=150, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════
# Monkey info summary
# ═══════════════════════════════════════════════════════════════════════

def print_monkey_info_summary(info_df):
    print(f"\n{'='*80}")
    print(f"MONKEY INFO SUMMARY")
    print(f"{'='*80}")

    for group in sorted(info_df['Group Name'].unique()):
        gdf = info_df[info_df['Group Name'] == group].sort_values('Rank')
        cat = gdf['Group Category'].iloc[0] if 'Group Category' in gdf.columns else ''
        print(f"\n{group} ({cat}):")
        print(f"  {'Name':>6s}  Sex  Age  {'Rank':>6s}  Note")
        print(f"  {'-'*50}")
        for _, row in gdf.iterrows():
            rank_str = f"{row['Rank']:.1f}" if pd.notna(row['Rank']) else 'N/A'
            note = row.get('Note', '')
            note_str = note if pd.notna(note) else ''
            print(f"  {row['Name']:>6s}   {row['Sex']}   {row['Age']:>3d}  "
                  f"{rank_str:>6s}  {note_str}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    print_monkey_info_summary(info_df)

    for region in ['AMG', 'ER']:
        cfg = RSAConfig(
            region=region,
            session=None,
            window=(0.300, 0.500),
            min_epoch_duration=1.0,
            min_reps_per_monkey=7,   # per-session floor previewed in section 7
            normalization=None,
            visual_responsiveness_filter=False,
            peak_latency_filter=False,
        )

        print(f"\n\n{'#'*80}")
        print(f"#  REGION: {region}")
        print(f"{'#'*80}")

        df = load_and_filter(cfg)

        save_dir = os.path.join(SAVE_DIR, region)

        # 1. Trial counts per identity
        trial_counts = get_trial_counts(df)
        report_trial_counts_per_identity(trial_counts, info_df, region, save_dir)

        # 2. Trial counts per group
        report_trial_counts_per_group(trial_counts, info_df, region, save_dir)

        # 3. Neuron counts per session
        report_neuron_counts(df, region, save_dir)

        # 4. Firing rate distributions
        report_firing_rate_distributions(df, info_df, cfg, region, save_dir)

        # 5. Session-level balance check
        report_session_balance(trial_counts, info_df, region, save_dir)

        # 6. Session firing-rate audit (baseline + selectivity + provenance)
        known = set(info_df['Name'].astype(str))
        identities = sorted(set(df['MonkeyName'].unique()) & known)
        trial_rate_df = _build_trial_rate_df(df, identities, cfg.window)
        report_session_fr_baseline(trial_rate_df, info_df, region, save_dir)
        report_neuron_selectivity(trial_rate_df, info_df, region, save_dir)

        # 7. Pseudopop session-dropping preview (matches run_rsa_pseudopop)
        report_pseudopop_session_drop(trial_counts, info_df, region, save_dir,
                                      highlight=cfg.min_reps_per_monkey)

    plt.show()
    print(f"\nAll figures saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
