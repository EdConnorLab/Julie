"""
ridge_decoding_directional.py
=============================

Population-level Ridge decoding of directional social behavior.

This is the population counterpart of behavior_vector_linear_regression.py.
Instead of per-neuron OLS, we use the ENTIRE neural population (all neurons
in a region) with Ridge regression + LOO-CV to decode each source monkey's
directional behavior vector from population responses to stimulus monkeys.

For each group × behavior (affiliation, agonism, submission) × direction
(given / received):
    For each source monkey:
        y = behavior vector toward/from stimulus monkeys  (length n_stim)
        X = population response to each stimulus monkey   (n_stim, n_neurons)
        Ridge LOO → R²
    Report mean R² across source monkeys + permutation test.

Usage
-----
    python ridge_decoding_directional.py

Adjust settings in the USER SETTINGS block at the bottom.
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

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition
from preprocessing import preprocess

sys.path.insert(0,
                '/population/neural_trajectory')
from social_rank_analysis import load_group_matrices

MONKEY_INFO_PATH = (
    "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"
)


# ---------------------------------------------------------------------------
# Neural data → windowed population response per stimulus monkey
# ---------------------------------------------------------------------------
def build_population_matrix(df, cfg, monkey_list, time_window_s, min_reps=1):
    """
    Build (n_monkeys, n_neurons) population response matrix for a set of
    stimulus monkeys, averaged over a specific time window.

    Parameters
    ----------
    monkey_list : list of str
        Stimulus monkeys to include as conditions.
    time_window_s : (float, float)
        (start, end) in seconds. Bins whose start falls in [start, end)
        are averaged.  Use (0, cfg.min_epoch_duration) for the full trial.

    Returns
    -------
    X : (n_monkeys, n_neurons) — rows aligned with monkey_list order
    neuron_ids : list of str
    """
    condition_map = {m: m for m in monkey_list}

    pca_matrix, row_meta, info = build_matrix_by_condition(
        df, cfg, condition_map, min_reps=min_reps)

    # Soft-normalize, no mean-centering (condition differences = signal)
    pca_matrix = preprocess(pca_matrix, info, cfg,
                            soft_normalize=True, mean_center=False)

    n_cond = info['n_conditions']
    n_bins = info['n_bins']
    n_neu = pca_matrix.shape[1]
    R = pca_matrix.reshape(n_cond, n_bins, n_neu)

    # Slice time window
    t_start, t_end = time_window_s
    bin_starts = np.arange(n_bins) * cfg.bin_width
    mask = (bin_starts >= t_start) & (bin_starts < t_end)
    sel_bins = np.where(mask)[0]
    if len(sel_bins) == 0:
        raise ValueError(
            f"No bins in [{t_start}, {t_end}) s.  "
            f"Available: [0, {n_bins * cfg.bin_width:.3f}) s")

    X = R[:, sel_bins, :].mean(axis=1)   # (n_cond, n_neurons)
    conditions = info['conditions']       # sorted monkey names

    # Reorder rows to match monkey_list exactly
    cond_to_row = {c: i for i, c in enumerate(conditions)}
    order = [cond_to_row[m] for m in monkey_list]
    X = X[order]

    print(f"Population matrix: {X.shape}  "
          f"(window {t_start}–{t_end} s, {len(sel_bins)} bins)")
    return X, info['neuron_ids']


# ---------------------------------------------------------------------------
# Identify subject monkey for a group
# ---------------------------------------------------------------------------
def find_subject_monkey(group_name=None):
    """
    Return the subject (recording) monkey name from monkeyinfo.csv.

    The subject monkey appears in every group's behavior matrix but is
    the one being recorded from, so it should be excluded as a source.
    Searches for 'subject' in the Note column (not group-specific,
    since the subject monkey is shared across groups).
    """
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()
    if 'Note' not in info_df.columns:
        return None
    mask = info_df['Note'].fillna('').str.contains('subject', case=False)
    matches = info_df.loc[mask, 'Name'].tolist()
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"[warn] Multiple subject monkeys found: {matches}; using first")
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# LOO Ridge for a single source monkey
# ---------------------------------------------------------------------------
def _loo_ridge(X, y):
    """
    LOO Ridge regression.

    Parameters
    ----------
    X : (n_stim, n_neurons)
    y : (n_stim,) — z-scored behavior values

    Returns
    -------
    r2 : float  (can be negative)
    y_pred : (n_stim,)
    """
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
# Core decoding loop
# ---------------------------------------------------------------------------
def decode_directional(X_pop, monkey_list, behavior_matrix, behavior_name,
                       subject_name=None, n_perm=1000, rng_seed=42):
    """
    For each source monkey, decode their behavior vector from population
    responses to stimulus monkeys using Ridge LOO.

    Each source monkey gets its OWN permutation test (shuffle that
    source's behavior labels independently).  An overall mean R² and
    its permutation p-value are also computed.

    Parameters
    ----------
    X_pop : (n_monkeys, n_neurons)
        Population response for each monkey in monkey_list.
    monkey_list : list of str
        Ordered monkey names matching rows of X_pop and rows/cols of
        behavior_matrix.
    behavior_matrix : DataFrame
        Square matrix, rows = actors, cols = recipients.
    behavior_name : str
        E.g. 'affiliation_given', 'agonism_received'.
    subject_name : str or None
        Subject monkey to exclude entirely.
    n_perm : int
        Number of label shuffles for permutation test.

    Returns
    -------
    source_results : list of dicts (one per source monkey, now with
                     per-source null_dist and p_value)
    mean_r2 : float
    mean_p_value : float (permutation p-value on mean R²)
    mean_null_dist : (n_perm,)
    """
    # Align behavior matrix to monkey_list order
    bmat = behavior_matrix.loc[monkey_list, monkey_list].values.astype(float)
    n = len(monkey_list)

    # Determine subject index (if any)
    subj_idx = None
    if subject_name and subject_name in monkey_list:
        subj_idx = monkey_list.index(subject_name)

    # ------------------------------------------------------------------
    # Pass 1: observed R² for each source monkey
    # ------------------------------------------------------------------
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
                'y_raw': y_raw, 'y_pred': np.full_like(y_raw, np.nan),
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
            'y_raw': y_raw, 'y_z': y_z, 'y_pred': y_pred,
            'skipped': False, 'p_value': np.nan,
            'null_dist': np.zeros(n_perm),
        })

    # ------------------------------------------------------------------
    # Pass 2: permutation tests (per-source AND mean)
    # ------------------------------------------------------------------
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

    # Compute per-source p-values
    for vi in valid_indices:
        res = source_results[vi]
        null = res['null_dist']
        res['p_value'] = (np.nansum(null >= res['r2']) + 1) / (n_perm + 1)

    mean_p_value = (np.nansum(mean_null_dist >= mean_r2) + 1) / (n_perm + 1)

    return source_results, mean_r2, mean_p_value, mean_null_dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(all_results, group_name, save_path=None):
    """
    Heatmap: rows = behavior types, cols = source monkeys, cells = R².
    """
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
    ax.set_title(f'Ridge Decoding R²  —  {group_name}')
    ax.set_ylabel('')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_summary_bar(all_results, group_name, region, time_window_s,
                     save_path=None):
    """Bar chart: mean R² per behavior, colored by significance."""
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
    ax.set_ylabel('Mean LOO R² across source monkeys')
    ax.set_title(f'Ridge Directional Decoding — {group_name} — {region}  '
                 f'[{time_window_s[0]*1000:.0f}–{time_window_s[1]*1000:.0f} ms]')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_null_distributions(all_results, save_path=None):
    """One histogram per behavior: null + observed."""
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
        ax.axvline(obs, color='red', lw=2,
                   label=f"obs = {obs:.3f}")
        ax.set_title(res['behavior_name'], fontsize=9)
        ax.set_xlabel('Mean R²')
        ax.legend(fontsize=7)
    for idx in range(len(all_results), nrows * ncols):
        axes.flat[idx].axis('off')
    fig.suptitle('Permutation Null Distributions', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==================== USER SETTINGS ====================
    GROUP = 'Zombies'                 # 'Zombies' | 'Best Frans' | 'Instigators'
    REGION = 'AMG'                    # 'AMG' | 'ER' | 'ALL'
    TIME_WINDOW_S = (0.00, 2.000)    # (start, end) in seconds; (0, 2.0) = full trial
    N_PERM = 1000
    MIN_REPS = 1
    SUBJECT_NAME = None               # None = auto-detect from monkeyinfo.csv
                                      # or set explicitly, e.g. 'Zap'
    SAVE_DIR = (
        "/home/connorlab/Documents/GitHub/Julie/Cortana/"
        f"analysis_results/ridge_directional/{REGION}/{GROUP}"
    )
    # =======================================================

    cfg = TrajectoryConfig(
        region=REGION,
        session=None,
        trial_averaged=True,
        peak_align=False,
        bin_width=0.050,
        min_epoch_duration=max(TIME_WINDOW_S[1], 2.0),
    )
    cfg.validate()

    # 1. Load behavior matrices
    print("=" * 60)
    print(f"Group: {GROUP}   Region: {REGION}   "
          f"Window: {TIME_WINDOW_S[0]*1000:.0f}–{TIME_WINDOW_S[1]*1000:.0f} ms")
    print("=" * 60)
    mats = load_group_matrices(GROUP)
    monkey_names_in_matrix = list(mats['agonism'].index)
    print(f"Monkeys in behavior matrices: {monkey_names_in_matrix}")

    subject_name = SUBJECT_NAME or find_subject_monkey()
    if subject_name:
        print(f"Subject monkey: {subject_name}")
    else:
        print("[warn] No subject monkey found; proceeding without exclusion")

    # 2. Load neural data
    print("\nLoading neural data ...")
    df = load_and_filter(cfg)

    # Monkeys that appear in both neural data and behavior matrices
    monkeys_in_data = set(df['MonkeyName'].unique())
    common = [m for m in monkey_names_in_matrix if m in monkeys_in_data]
    missing = [m for m in monkey_names_in_matrix if m not in monkeys_in_data]
    if missing:
        print(f"Monkeys in matrix but NOT in neural data: {missing}")
    print(f"Usable monkeys: {common}  (n={len(common)})")

    if len(common) < 4:
        raise RuntimeError(f"Only {len(common)} monkeys — too few.")

    # 3. Build population response matrix
    X_pop, neuron_ids = build_population_matrix(
        df, cfg, common, time_window_s=TIME_WINDOW_S, min_reps=MIN_REPS)
    print(f"Neurons: {X_pop.shape[1]}")

    # 4. Define behavior targets
    #    "Given" = source's row (source → stimulus)
    #    "Received" = source's column (stimulus → source)
    behavior_targets = []
    for beh_key in ('affiliation', 'agonism', 'submission'):
        bmat = mats[beh_key]
        # Restrict to common monkeys
        bmat = bmat.loc[common, common]
        behavior_targets.append((f'{beh_key}_given',    bmat))
        # For "received", we label the direction but the decode_directional
        # function reads the column when it sees "Received" in the name.
        behavior_targets.append((f'{beh_key}_received', bmat))

    # 5. Decode each behavior
    all_results = []
    for beh_name, bmat in behavior_targets:
        print("\n" + "=" * 60)
        print(f"Decoding: {beh_name}")
        print("=" * 60)

        src_results, mean_r2, p_val, null = decode_directional(
            X_pop, common, bmat, beh_name,
            subject_name=subject_name, n_perm=N_PERM)

        print(f"  Per-source R²:")
        for sr in src_results:
            tag = " [skipped]" if sr.get('skipped') else ""
            p_str = f"p={sr['p_value']:.4f}" if not np.isnan(sr['p_value']) else "p=N/A"
            print(f"    {sr['source']:>12s}:  R² = {sr['r2']:+.4f}  "
                  f"{p_str}  (n_stim = {sr['n_stim']}){tag}")
        print(f"  Mean R² = {mean_r2:+.4f}   p = {p_val:.4f}")

        all_results.append({
            'behavior_name': beh_name,
            'source_results': src_results,
            'mean_r2': mean_r2,
            'p_value': p_val,
            'null_dist': null,
        })

    # 6. Plots
    print("\n" + "=" * 60)
    print("Plotting ...")
    print("=" * 60)
    plot_heatmap(all_results, GROUP,
                 save_path=os.path.join(SAVE_DIR, f'{REGION}_{GROUP}_{TIME_WINDOW_S}_heatmap_r2.png'))
    plot_summary_bar(all_results, GROUP, REGION, TIME_WINDOW_S,
                     save_path=os.path.join(SAVE_DIR, f'{REGION}_{GROUP}_{TIME_WINDOW_S}_summary_bar.png'))
    plot_null_distributions(all_results, f'{REGION}_{GROUP}_{TIME_WINDOW_S}_null_distribution.png')

    # 7. Save CSVs — two tables
    os.makedirs(SAVE_DIR, exist_ok=True)

    # (a) Per-source-monkey detail table
    detail_rows = []
    for res in all_results:
        for sr in res['source_results']:
            detail_rows.append({
                'group': GROUP,
                'region': REGION,
                'window_ms': (f"{TIME_WINDOW_S[0]*1000:.0f}-"
                              f"{TIME_WINDOW_S[1]*1000:.0f}"),
                'behavior': res['behavior_name'],
                'source_monkey': sr['source'],
                'n_stim': sr['n_stim'],
                'r2': sr['r2'],
                'p_value': sr['p_value'],
                'skipped': sr.get('skipped', False),
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(SAVE_DIR, 'decoding_per_source.csv')
    detail_df.to_csv(detail_path, index=False)
    print(f"\nPer-source results → {detail_path}")
    print(detail_df.to_string(index=False))

    # (b) Behavior-level summary table
    summary_rows = []
    for res in all_results:
        summary_rows.append({
            'group': GROUP,
            'region': REGION,
            'window_ms': (f"{TIME_WINDOW_S[0]*1000:.0f}-"
                          f"{TIME_WINDOW_S[1]*1000:.0f}"),
            'behavior': res['behavior_name'],
            'n_sources': sum(1 for sr in res['source_results']
                             if not sr.get('skipped')),
            'mean_r2': res['mean_r2'],
            'p_value': res['p_value'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SAVE_DIR, 'decoding_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    print(summary_df.to_string(index=False))

    plt.show()


if __name__ == '__main__':
    main()