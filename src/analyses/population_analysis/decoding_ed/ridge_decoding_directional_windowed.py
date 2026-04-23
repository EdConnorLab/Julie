"""
ridge_decoding_directional_windowed.py
======================================

Ridge decoding of directional social behavior using neuron-specific
time windows from a Kruskal-Wallis–tested pkl file.

Each neuron gets its OWN time window (the window where it showed the
strongest stimulus selectivity), rather than a single global window
applied to all neurons.

Two modes:
    1. ALL neurons (all windows from the pkl, best window per neuron)
    2. SIGNIFICANT neurons only (p < 0.05, best window per neuron)

For each neuron, mean firing rate within its window is computed per
trial, then trial-averaged per stimulus monkey → (n_stim, n_neurons).
Ridge LOO decoding + permutation test, same as ridge_decoding_directional.

Usage
-----
    python ridge_decoding_directional_windowed.py

Adjust settings in the USER SETTINGS block.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

sys.path.insert(0, '/home/connorlab/Documents/GitHub/Julie/src')
sys.path.insert(0, '/home/connorlab/Documents/GitHub/Julie/src/analyses')
sys.path.insert(0, '/home/connorlab/Documents/GitHub/Julie/src/analyses/population_analysis')

from state_space.config import TrajectoryConfig
from state_space.data_loading import load_and_filter
from social_rank_analysis import load_group_matrices

MONKEY_INFO_PATH = (
    "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"
)


# ---------------------------------------------------------------------------
# Parse the KW pkl and select neurons + windows
# ---------------------------------------------------------------------------
def load_neuron_windows(pkl_path, sig_only=False, p_thresh=0.05,
                        max_window_end_ms=2000):
    """
    Load neuron windows from the KW-tested pkl file.

    Parameters
    ----------
    pkl_path : str
        Path to si_sorted_*_all_windows_pKW_tested.pkl
    sig_only : bool
        If True, keep only neurons with p-value < p_thresh.
    p_thresh : float
        Raw p-value threshold (default 0.05).
    max_window_end_ms : int
        Drop windows whose end exceeds this value.

    Returns
    -------
    selected : DataFrame
        One row per selected neuron with columns:
        NeuronID (from pkl), session, raw_neuron_id,
        WindowStart_ms, WindowEnd_ms, p-value
    """
    df = pd.read_pickle(pkl_path)
    n_orig = len(df)

    # Filter: window end ≤ max
    df = df[df['WindowEnd_ms'] <= max_window_end_ms].copy()
    print(f"Windows after EndMs ≤ {max_window_end_ms}: "
          f"{len(df)} / {n_orig}")

    # Filter: significance
    if sig_only:
        df = df[df['p-value'] < p_thresh].copy()
        print(f"Windows after p < {p_thresh}: {len(df)}  "
              f"({df['NeuronID'].nunique()} neurons)")

    if len(df) == 0:
        raise ValueError("No neurons survived filtering.")

    # For neurons with multiple windows, keep lowest p-value
    df = df.sort_values('p-value').drop_duplicates('NeuronID', keep='first')

    # Parse NeuronID → session + raw neuron id
    def parse_nid(nid):
        parts = nid.split('_')
        session = '_'.join(parts[:3])
        raw_nid = '_'.join(parts[3:])
        return session, raw_nid

    parsed = df['NeuronID'].apply(parse_nid)
    df['session'] = [p[0] for p in parsed]
    df['raw_neuron_id'] = [p[1] for p in parsed]

    print(f"Selected {len(df)} neurons across "
          f"{df['session'].nunique()} sessions")
    print(f"  Regions: {df['session'].apply(lambda s: s.split('_')[0]).value_counts().to_dict()}")
    print(f"  Window range: {df['WindowStart_ms'].min()}–"
          f"{df['WindowEnd_ms'].max()} ms")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build population matrix with per-neuron windows
# ---------------------------------------------------------------------------
def build_windowed_population_matrix(raw_df, neuron_table, monkey_list):
    """
    Build a trial-averaged (n_monkeys, n_neurons) population matrix
    where each neuron uses its OWN time window.

    Optimised version: iterates trials once per session, looks up
    each neuron's window from a dict.

    Parameters
    ----------
    raw_df : DataFrame
        Output of load_and_filter.
    neuron_table : DataFrame
        Output of load_neuron_windows (reset_index'd).
    monkey_list : list of str

    Returns
    -------
    X : (n_monkeys, n_neurons)
    neuron_ids : list of str
    """
    raw_df = raw_df[raw_df['MonkeyName'].isin(monkey_list)].copy()

    n_neurons = len(neuron_table)
    n_monkeys = len(monkey_list)
    monkey_to_idx = {m: i for i, m in enumerate(monkey_list)}

    rate_sum = np.zeros((n_monkeys, n_neurons))
    rate_count = np.zeros((n_monkeys, n_neurons))

    # Build lookup: (session, raw_neuron_id) → (col_idx, win_start_s, win_end_s)
    neuron_lookup = {}
    for col_idx, (_, nrow) in enumerate(neuron_table.iterrows()):
        key = (nrow['session'], nrow['raw_neuron_id'])
        neuron_lookup[key] = (
            col_idx,
            nrow['WindowStart_ms'] / 1000.0,
            nrow['WindowEnd_ms'] / 1000.0,
        )

    sessions_needed = neuron_table['session'].unique()

    for session in sessions_needed:
        sess_df = raw_df[raw_df['session'] == session]
        if len(sess_df) == 0:
            continue

        # Which neurons from this session are in our table?
        sess_neurons = {k: v for k, v in neuron_lookup.items()
                        if k[0] == session}
        if not sess_neurons:
            continue

        # Iterate each row (one row = one neuron × one trial)
        for _, row in sess_df.iterrows():
            raw_nid = row['NeuronID']
            monkey = row['MonkeyName']
            if monkey not in monkey_to_idx:
                continue

            key = (session, raw_nid)
            if key not in sess_neurons:
                continue

            col_idx, win_start_s, win_end_s = sess_neurons[key]
            m_idx = monkey_to_idx[monkey]

            epoch_start = row['EpochStartStop'][0]
            spikes = row['SpikeTimes']
            rel = spikes - epoch_start
            count = np.sum((rel >= win_start_s) & (rel < win_end_s))
            duration = win_end_s - win_start_s
            rate = count / duration if duration > 0 else 0.0

            rate_sum[m_idx, col_idx] += rate
            rate_count[m_idx, col_idx] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.where(rate_count > 0, rate_sum / rate_count, 0.0)

    # Drop neurons with zero data for all monkeys
    valid_cols = rate_count.sum(axis=0) > 0
    n_dropped = (~valid_cols).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} neurons with no matching data")
        X = X[:, valid_cols]
        neuron_table_out = neuron_table[valid_cols].reset_index(drop=True)
    else:
        neuron_table_out = neuron_table

    print(f"Population matrix: {X.shape}  "
          f"({n_monkeys} monkeys × {X.shape[1]} neurons)")

    neuron_ids = neuron_table_out['NeuronID'].tolist()
    return X, neuron_ids


# ---------------------------------------------------------------------------
# Soft-normalize
# ---------------------------------------------------------------------------
def soft_normalize(X, const=5.0):
    """Soft-normalize each neuron column: x / (range + const)."""
    rng = X.max(axis=0) - X.min(axis=0)
    denom = rng + const
    denom[denom == 0] = 1.0
    return X / denom


# ---------------------------------------------------------------------------
# Find subject monkey
# ---------------------------------------------------------------------------
def find_subject_monkey():
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()
    if 'Note' not in info_df.columns:
        return None
    mask = info_df['Note'].fillna('').str.contains('subject', case=False)
    matches = info_df.loc[mask, 'Name'].tolist()
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# LOO Ridge
# ---------------------------------------------------------------------------
def _loo_ridge(X, y):
    n = len(y)
    if n < 3:
        return np.nan, np.full(n, np.nan)
    loo = LeaveOneOut()
    y_pred = np.full(n, np.nan)
    alphas = np.logspace(-3, 5, 50)
    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler().fit(X[train_ix])
        X_tr = scaler.transform(X[train_ix])
        X_te = scaler.transform(X[test_ix])
        model = RidgeCV(alphas=alphas, scoring='r2')
        model.fit(X_tr, y[train_ix])
        y_pred[test_ix] = model.predict(X_te)
    r2 = r2_score(y, y_pred)
    return r2, y_pred


# ---------------------------------------------------------------------------
# Core decoding loop (same as ridge_decoding_directional, per-source perms)
# ---------------------------------------------------------------------------
def decode_directional(X_pop, monkey_list, behavior_matrix, behavior_name,
                       subject_name=None, n_perm=1000, rng_seed=42):
    bmat = behavior_matrix.loc[monkey_list, monkey_list].values.astype(float)
    n = len(monkey_list)

    subj_idx = None
    if subject_name and subject_name in monkey_list:
        subj_idx = monkey_list.index(subject_name)

    source_results = []

    for src_idx in range(n):
        src_name = monkey_list[src_idx]
        if src_idx == subj_idx:
            continue

        stim_mask = np.ones(n, dtype=bool)
        stim_mask[src_idx] = False
        if subj_idx is not None:
            stim_mask[subj_idx] = False
        stim_indices = np.where(stim_mask)[0]
        stim_names = [monkey_list[i] for i in stim_indices]

        if 'received' in behavior_name.lower() or 'from' in behavior_name.lower():
            y_raw = bmat[stim_indices, src_idx]
        else:
            y_raw = bmat[src_idx, stim_indices]

        X_stim = X_pop[stim_indices]

        if np.std(y_raw) < 1e-8:
            source_results.append({
                'source': src_name, 'stimuli': stim_names,
                'stim_indices': stim_indices,
                'r2': np.nan, 'n_stim': len(stim_indices),
                'skipped': True, 'p_value': np.nan,
                'null_dist': np.full(n_perm, np.nan),
            })
            continue

        y_z = (y_raw - y_raw.mean()) / y_raw.std()
        r2, y_pred = _loo_ridge(X_stim, y_z)

        source_results.append({
            'source': src_name, 'stimuli': stim_names,
            'stim_indices': stim_indices,
            'r2': r2, 'n_stim': len(stim_indices),
            'y_z': y_z,
            'skipped': False, 'p_value': np.nan,
            'null_dist': np.zeros(n_perm),
        })

    # Permutation tests
    valid_indices = [i for i, r in enumerate(source_results)
                     if not r.get('skipped')]
    valid_r2s = [source_results[i]['r2'] for i in valid_indices]
    mean_r2 = np.mean(valid_r2s) if valid_r2s else np.nan

    rng = np.random.default_rng(rng_seed)
    mean_null_dist = np.zeros(n_perm)

    for pi in range(n_perm):
        perm_r2s = []
        for vi in valid_indices:
            res = source_results[vi]
            X_stim = X_pop[res['stim_indices']]
            y_shuf = rng.permutation(res['y_z'])
            r2_shuf, _ = _loo_ridge(X_stim, y_shuf)
            res['null_dist'][pi] = r2_shuf
            if not np.isnan(r2_shuf):
                perm_r2s.append(r2_shuf)
        mean_null_dist[pi] = np.mean(perm_r2s) if perm_r2s else np.nan
        if (pi + 1) % 200 == 0:
            print(f"      perm {pi + 1}/{n_perm}")

    for vi in valid_indices:
        res = source_results[vi]
        null = res['null_dist']
        res['p_value'] = (np.nansum(null >= res['r2']) + 1) / (n_perm + 1)

    mean_p_value = (np.nansum(mean_null_dist >= mean_r2) + 1) / (n_perm + 1)

    return source_results, mean_r2, mean_p_value, mean_null_dist


# ---------------------------------------------------------------------------
# Plotting (same as ridge_decoding_directional)
# ---------------------------------------------------------------------------
def plot_heatmap(all_results, title, save_path=None):
    rows = []
    for res in all_results:
        for sr in res['source_results']:
            rows.append({
                'behavior': res['behavior_name'],
                'source': sr['source'],
                'R²': sr['r2'] if not sr.get('skipped') else np.nan,
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='behavior', columns='source', values='R²')
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8),
                                    max(4, len(pivot.index) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_summary_bar(all_results, title, save_path=None):
    names = [r['behavior_name'] for r in all_results]
    means = [r['mean_r2'] for r in all_results]
    pvals = [r['p_value'] for r in all_results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4))
    colors = ['#2ca02c' if p < 0.05 else '#888888' for p in pvals]
    ax.bar(range(len(names)), means, color=colors, edgecolor='k', lw=0.8)

    y_top = max(abs(min(means)), max(means)) if means else 1
    for i, (m, p) in enumerate(zip(means, pvals)):
        label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        offset = 0.03 * y_top if m >= 0 else -0.06 * y_top
        ax.text(i, m + offset, label, ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Mean LOO R²')
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_null_distributions(all_results, save_dir=None):
    n = len(all_results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    for idx, res in enumerate(all_results):
        ax = axes.flat[idx]
        null = res['null_dist']
        obs = res['mean_r2']
        ax.hist(null[~np.isnan(null)], bins=30, color='#aaaaaa',
                edgecolor='k', lw=0.3)
        ax.axvline(obs, color='red', lw=2, label=f"obs = {obs:.3f}")
        ax.set_title(res['behavior_name'], fontsize=9)
        ax.set_xlabel('Mean R²')
        ax.legend(fontsize=7)
    for idx in range(len(all_results), nrows * ncols):
        axes.flat[idx].axis('off')
    fig.suptitle('Permutation Null Distributions', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_dir:
        path = os.path.join(save_dir, 'null_distributions.png')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"  saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Run one decoding pass (all or significant neurons)
# ---------------------------------------------------------------------------
def run_decoding(raw_df, neuron_table, mats, monkey_list, subject_name,
                 tag, n_perm, save_dir):
    """Run full Ridge directional decoding for one neuron set."""

    print(f"\n{'='*60}")
    print(f"  {tag}: {len(neuron_table)} neurons")
    print(f"{'='*60}")

    X_pop, neuron_ids = build_windowed_population_matrix(
        raw_df, neuron_table, monkey_list)
    X_pop = soft_normalize(X_pop, const=5.0)
    print(f"  Final matrix: {X_pop.shape}")

    behavior_targets = []
    for beh_key in ('affiliation', 'agonism', 'submission'):
        bmat = mats[beh_key].loc[monkey_list, monkey_list]
        behavior_targets.append((f'{beh_key}_given',    bmat))
        behavior_targets.append((f'{beh_key}_received', bmat))

    all_results = []
    for beh_name, bmat in behavior_targets:
        print(f"\n  Decoding: {beh_name}")
        src_results, mean_r2, p_val, null = decode_directional(
            X_pop, monkey_list, bmat, beh_name,
            subject_name=subject_name, n_perm=n_perm)

        for sr in src_results:
            s = " [skip]" if sr.get('skipped') else ""
            p_str = (f"p={sr['p_value']:.4f}"
                     if not np.isnan(sr['p_value']) else "p=N/A")
            print(f"    {sr['source']:>12s}: R²={sr['r2']:+.4f}  "
                  f"{p_str}{s}")
        print(f"    Mean R² = {mean_r2:+.4f}   p = {p_val:.4f}")

        all_results.append({
            'behavior_name': beh_name,
            'source_results': src_results,
            'mean_r2': mean_r2,
            'p_value': p_val,
            'null_dist': null,
        })

    # Plots
    plot_heatmap(all_results,
                 f'Ridge Decoding R² — {tag}',
                 save_path=os.path.join(save_dir, 'heatmap_r2.png'))
    plot_summary_bar(all_results,
                     f'Ridge Directional Decoding — {tag}',
                     save_path=os.path.join(save_dir, 'summary_bar.png'))
    plot_null_distributions(all_results, save_dir=save_dir)

    # CSVs
    os.makedirs(save_dir, exist_ok=True)

    detail_rows = []
    for res in all_results:
        for sr in res['source_results']:
            detail_rows.append({
                'tag': tag,
                'behavior': res['behavior_name'],
                'source_monkey': sr['source'],
                'n_stim': sr['n_stim'],
                'r2': sr['r2'],
                'p_value': sr['p_value'],
                'skipped': sr.get('skipped', False),
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(save_dir, 'decoding_per_source.csv'),
                     index=False)

    summary_rows = []
    for res in all_results:
        summary_rows.append({
            'tag': tag,
            'behavior': res['behavior_name'],
            'n_sources': sum(1 for sr in res['source_results']
                             if not sr.get('skipped')),
            'n_neurons': X_pop.shape[1],
            'mean_r2': res['mean_r2'],
            'p_value': res['p_value'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, 'decoding_summary.csv'),
                      index=False)
    print(f"\n  Results saved → {save_dir}")
    print(summary_df.to_string(index=False))

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==================== USER SETTINGS ====================
    GROUP = 'Zombies'
    PKL_PATH = ("/home/connorlab/Documents/GitHub/Julie/Cortana/"
                "analysis_cache/si_sorted_Zombies_all_windows_pKW_tested.pkl")
    N_PERM = 1000
    MAX_WINDOW_END_MS = 2000
    P_THRESH = 0.05                   # for "significant" subset
    SUBJECT_NAME = None               # None = auto-detect
    SAVE_BASE = (
        "/home/connorlab/Documents/GitHub/Julie/Cortana/"
        "analysis_results/ridge_directional_windowed"
    )
    # =======================================================

    # 1. Load behavior matrices
    print("=" * 60)
    print(f"Group: {GROUP}")
    print("=" * 60)
    mats = load_group_matrices(GROUP)
    monkey_names_in_matrix = list(mats['agonism'].index)
    print(f"Monkeys in behavior matrices: {monkey_names_in_matrix}")

    subject_name = SUBJECT_NAME or find_subject_monkey()
    if subject_name:
        print(f"Subject monkey: {subject_name}")

    # 2. Load ALL neuron windows (before filtering by significance)
    print("\n--- Loading neuron windows ---")
    all_neurons = load_neuron_windows(
        PKL_PATH, sig_only=False,
        max_window_end_ms=MAX_WINDOW_END_MS)
    sig_neurons = load_neuron_windows(
        PKL_PATH, sig_only=True, p_thresh=P_THRESH,
        max_window_end_ms=MAX_WINDOW_END_MS)

    # 3. Load neural data (ALL regions, since neurons span AMG + ER)
    print("\nLoading neural data (ALL regions) ...")
    cfg = TrajectoryConfig(
        region='ALL',
        session=None,
        trial_averaged=True,
        peak_align=False,
        bin_width=0.050,
        min_epoch_duration=2.0,
    )
    cfg.validate()
    raw_df = load_and_filter(cfg)

    # Intersect monkeys
    monkeys_in_data = set(raw_df['MonkeyName'].unique())
    common = [m for m in monkey_names_in_matrix if m in monkeys_in_data]
    missing = [m for m in monkey_names_in_matrix if m not in monkeys_in_data]
    if missing:
        print(f"Monkeys in matrix but NOT in data: {missing}")
    print(f"Usable monkeys: {common}  (n={len(common)})")

    if len(common) < 4:
        raise RuntimeError(f"Only {len(common)} monkeys — too few.")

    # 4. Run both passes
    print("\n\n" + "#" * 60)
    print("# PASS 1: ALL NEURONS (best window per neuron)")
    print("#" * 60)
    run_decoding(
        raw_df, all_neurons, mats, common, subject_name,
        tag=f"{GROUP}_all_neurons",
        n_perm=N_PERM,
        save_dir=os.path.join(SAVE_BASE, GROUP, 'all_neurons'),
    )

    print("\n\n" + "#" * 60)
    print(f"# PASS 2: SIGNIFICANT NEURONS ONLY (p < {P_THRESH})")
    print("#" * 60)
    run_decoding(
        raw_df, sig_neurons, mats, common, subject_name,
        tag=f"{GROUP}_sig_p{P_THRESH}",
        n_perm=N_PERM,
        save_dir=os.path.join(SAVE_BASE, GROUP, 'sig_neurons'),
    )

    plt.show()


if __name__ == '__main__':
    main()
