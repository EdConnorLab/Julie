"""
ridge_decoding_directional_single_trial.py
==========================================

Single-trial population-level Ridge decoding of directional social
behavior.  Uses individual trials (not trial-averaged) to increase
the number of observations available to Ridge regression.

For each group × behavior × direction × source monkey:
    Each trial to a stimulus monkey is one observation.
    X = (n_trials, n_neurons)  population response on that trial
    y = (n_trials,)            behavior value from source → stimulus
                               (same value for every trial to the
                               same stimulus monkey)

Cross-validation: leave-one-stimulus-monkey-out (LOSMO).
    Train on all trials to N-1 stimulus monkeys,
    test on all trials to the held-out monkey.
    This tests genuine generalisation to unseen monkeys.

Permutation test: shuffle the stimulus-monkey → behavior mapping
(not individual trials), then recompute LOSMO R².

Usage
-----
    python ridge_decoding_directional_single_trial.py

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
from sklearn.metrics import r2_score

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning import _bin_trial

sys.path.insert(0,
                '/population/neural_trajectory')
from social_rank_analysis import load_group_matrices

MONKEY_INFO_PATH = (
    "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"
)


# ---------------------------------------------------------------------------
# Build single-trial population matrix
# ---------------------------------------------------------------------------
def build_single_trial_population(df, cfg, monkey_list, time_window_s):
    """
    Build a single-trial population response matrix from raw data.

    For each trial (identified by session × TaskField), compute the
    mean firing rate of each neuron within the time window, then
    concatenate neurons across sessions (pseudo-population).

    Parameters
    ----------
    df : DataFrame
        Output of load_and_filter.
    monkey_list : list of str
        Stimulus monkeys to include.
    time_window_s : (float, float)

    Returns
    -------
    X : (n_total_trials, n_neurons_total)
    trial_labels : list of str
        Stimulus monkey name for each row of X.
    neuron_ids : list of str
    n_trials_per_monkey : dict {monkey: int}
    """
    n_bins_full = int(np.floor(cfg.min_epoch_duration / cfg.bin_width))
    analysis_window = n_bins_full * cfg.bin_width

    t_start, t_end = time_window_s
    bin_starts = np.arange(n_bins_full) * cfg.bin_width
    win_mask = (bin_starts >= t_start) & (bin_starts < t_end)
    sel_bins = np.where(win_mask)[0]
    if len(sel_bins) == 0:
        raise ValueError(
            f"No bins in [{t_start}, {t_end}) s.  "
            f"Range: [0, {n_bins_full * cfg.bin_width:.3f}) s")

    # Filter to requested stimulus monkeys
    df = df[df['MonkeyName'].isin(monkey_list)].copy()

    # Identify trials: unique (session, TaskField) pairs
    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName']]
                    .reset_index())
    trial_meta = trial_meta[trial_meta['MonkeyName'].isin(monkey_list)]

    sessions = sorted(df['session'].unique())

    # Pass 1: figure out neurons per session
    session_neurons = {}
    all_neuron_ids = []
    for session in sessions:
        sess_df = df[df['session'] == session]
        nids = sorted(sess_df['NeuronID'].unique())
        session_neurons[session] = nids
        all_neuron_ids.extend(f"{session}__{nid}" for nid in nids)

    n_neurons_total = len(all_neuron_ids)

    # Pass 2: build trial × neuron matrix
    # Each trial gets one row: mean rate across selected bins,
    # concatenated across sessions (pseudo-population).
    trial_list = []       # list of (monkey_name, session, task_field)
    trial_vectors = []    # list of (n_neurons_total,) vectors

    for session in sessions:
        sess_df = df[df['session'] == session]
        nids = session_neurons[session]
        neuron_to_idx = {nid: i for i, nid in enumerate(nids)}
        n_neu = len(nids)

        # Offset into the concatenated neuron vector
        offset = sum(len(session_neurons[s]) for s in sessions
                     if s < session)

        sess_trials = trial_meta[trial_meta['session'] == session]

        for _, trow in sess_trials.iterrows():
            monkey = trow['MonkeyName']
            task_field = trow['TaskField']

            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            counts = _bin_trial(trial_rows, neuron_to_idx,
                                n_bins_full, cfg.bin_width, analysis_window)
            # counts: (n_bins_full, n_neu)

            # Convert to rate, slice time window, average
            rates = counts[sel_bins] / cfg.bin_width   # (n_win_bins, n_neu)
            mean_rate = rates.mean(axis=0)             # (n_neu,)

            vec = np.zeros(n_neurons_total)
            vec[offset:offset + n_neu] = mean_rate

            trial_list.append(monkey)
            trial_vectors.append(vec)

    X = np.vstack(trial_vectors)         # (n_total_trials, n_neurons_total)
    trial_labels = trial_list

    # Count trials per monkey
    n_trials_per_monkey = {}
    for m in monkey_list:
        n_trials_per_monkey[m] = sum(1 for t in trial_labels if t == m)

    print(f"Single-trial population matrix: {X.shape}  "
          f"(window {t_start}–{t_end} s, {len(sel_bins)} bins)")
    print(f"  Trials per monkey: "
          + ", ".join(f"{m}={n_trials_per_monkey[m]}" for m in monkey_list))

    return X, trial_labels, all_neuron_ids, n_trials_per_monkey


# ---------------------------------------------------------------------------
# Soft-normalize (neuron-wise, matching Churchland convention)
# ---------------------------------------------------------------------------
def soft_normalize(X, const=5.0):
    """
    Soft-normalize each neuron (column) by its range + const.

    Parameters
    ----------
    X : (n_trials, n_neurons)   in Hz
    const : float               Churchland uses 5 Hz

    Returns
    -------
    X_norm : same shape
    """
    rng = X.max(axis=0) - X.min(axis=0)
    denom = rng + const
    denom[denom == 0] = 1.0  # safety for dead neurons
    return X / denom


# ---------------------------------------------------------------------------
# Identify subject monkey
# ---------------------------------------------------------------------------
def find_subject_monkey():
    """Return the subject (recording) monkey name from monkeyinfo.csv."""
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()
    if 'Note' not in info_df.columns:
        return None
    mask = info_df['Note'].fillna('').str.contains('subject', case=False)
    matches = info_df.loc[mask, 'Name'].tolist()
    if len(matches) >= 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Leave-one-stimulus-monkey-out Ridge
# ---------------------------------------------------------------------------
def _losmo_ridge(X, y, trial_labels, stim_monkeys):
    """
    Leave-one-stimulus-monkey-out Ridge regression.

    For each held-out monkey, train on all other monkeys' trials,
    predict the held-out monkey's trials, compute per-fold R² and
    then an overall R² on the pooled predictions.

    Parameters
    ----------
    X : (n_trials, n_neurons)
    y : (n_trials,)       — behavior value (same for all trials to
                             the same stimulus monkey)
    trial_labels : list   — stimulus monkey name per trial
    stim_monkeys : list   — unique stimulus monkey names

    Returns
    -------
    r2_overall : float     R² on pooled held-out predictions
    y_pred : (n_trials,)   predictions for every trial
    """
    n = len(y)
    y_pred = np.full(n, np.nan)
    alphas = np.logspace(-3, 5, 50)
    trial_labels = np.array(trial_labels)

    for held_out in stim_monkeys:
        test_mask = trial_labels == held_out
        train_mask = ~test_mask

        if train_mask.sum() < 3:
            continue

        scaler = StandardScaler().fit(X[train_mask])
        X_tr = scaler.transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        model = RidgeCV(alphas=alphas, scoring='r2')
        model.fit(X_tr, y[train_mask])
        y_pred[test_mask] = model.predict(X_te)

    # Overall R² on all held-out predictions
    valid = ~np.isnan(y_pred)
    if valid.sum() < 3 or np.std(y[valid]) < 1e-8:
        return np.nan, y_pred

    r2 = r2_score(y[valid], y_pred[valid])
    return r2, y_pred


# ---------------------------------------------------------------------------
# Core decoding loop (single-trial version)
# ---------------------------------------------------------------------------
def decode_directional_single_trial(
        X_pop, trial_labels, monkey_list, behavior_matrix,
        behavior_name, subject_name=None, n_perm=1000, rng_seed=42):
    """
    For each source monkey, decode their behavior vector from
    single-trial population responses using LOSMO Ridge.

    Parameters
    ----------
    X_pop : (n_trials, n_neurons)
    trial_labels : list of str (stimulus monkey per trial)
    monkey_list : list of str (all monkeys in behavior matrix order)
    behavior_matrix : DataFrame (square, indexed by monkey name)
    behavior_name : str
    subject_name : str or None
    n_perm : int

    Returns
    -------
    source_results : list of dicts
    mean_r2 : float
    mean_p_value : float
    mean_null_dist : (n_perm,)
    """
    bmat = behavior_matrix.loc[monkey_list, monkey_list].values.astype(float)
    n = len(monkey_list)
    trial_labels_arr = np.array(trial_labels)

    subj_idx = None
    if subject_name and subject_name in monkey_list:
        subj_idx = monkey_list.index(subject_name)

    source_results = []

    for src_idx in range(n):
        src_name = monkey_list[src_idx]
        if src_idx == subj_idx:
            continue

        # Stimulus monkeys for this source
        stim_mask = np.ones(n, dtype=bool)
        stim_mask[src_idx] = False
        if subj_idx is not None:
            stim_mask[subj_idx] = False
        stim_indices = np.where(stim_mask)[0]
        stim_names = [monkey_list[i] for i in stim_indices]

        # Behavior values
        if 'received' in behavior_name.lower() or 'from' in behavior_name.lower():
            beh_per_stim = {monkey_list[i]: bmat[i, src_idx]
                            for i in stim_indices}
        else:
            beh_per_stim = {monkey_list[i]: bmat[src_idx, i]
                            for i in stim_indices}

        # Select trials for stimulus monkeys
        trial_mask = np.isin(trial_labels_arr, stim_names)
        X_sel = X_pop[trial_mask]
        labels_sel = trial_labels_arr[trial_mask]

        # Build y: each trial gets its stimulus monkey's behavior value
        y_raw = np.array([beh_per_stim[m] for m in labels_sel])

        if np.std(y_raw) < 1e-8 or len(X_sel) < 5:
            source_results.append({
                'source': src_name, 'stimuli': stim_names,
                'r2': np.nan, 'n_trials': len(X_sel),
                'n_stim': len(stim_names),
                'skipped': True, 'p_value': np.nan,
                'null_dist': np.full(n_perm, np.nan),
            })
            continue

        # Z-score behavior values (computed on unique stimulus means,
        # then applied to all trials)
        unique_vals = np.array(list(beh_per_stim.values()))
        beh_mean, beh_std = unique_vals.mean(), unique_vals.std()
        if beh_std < 1e-8:
            beh_std = 1.0
        y_z = (y_raw - beh_mean) / beh_std

        r2, y_pred = _losmo_ridge(X_sel, y_z, labels_sel, stim_names)

        source_results.append({
            'source': src_name, 'stimuli': stim_names,
            'r2': r2, 'n_trials': len(X_sel),
            'n_stim': len(stim_names),
            'y_z': y_z, 'labels_sel': labels_sel,
            'X_sel': X_sel, 'stim_names': stim_names,
            'beh_per_stim': beh_per_stim,
            'beh_mean': beh_mean, 'beh_std': beh_std,
            'skipped': False, 'p_value': np.nan,
            'null_dist': np.zeros(n_perm),
        })

    # ------------------------------------------------------------------
    # Permutation tests
    # ------------------------------------------------------------------
    valid_indices = [i for i, r in enumerate(source_results)
                     if not r.get('skipped')]
    valid_r2s = [source_results[i]['r2'] for i in valid_indices]
    mean_r2 = np.nanmean(valid_r2s) if valid_r2s else np.nan

    rng = np.random.default_rng(rng_seed)
    mean_null_dist = np.zeros(n_perm)

    for pi in range(n_perm):
        perm_r2s = []
        for vi in valid_indices:
            res = source_results[vi]
            stim_names = res['stim_names']
            beh_per_stim = res['beh_per_stim']

            # Shuffle: randomly reassign behavior values to stimulus
            # monkeys (permute the mapping, not individual trials)
            vals = list(beh_per_stim.values())
            rng.shuffle(vals)
            shuf_map = dict(zip(stim_names, vals))

            y_shuf_raw = np.array([shuf_map[m] for m in res['labels_sel']])
            y_shuf = (y_shuf_raw - res['beh_mean']) / res['beh_std']

            r2_shuf, _ = _losmo_ridge(
                res['X_sel'], y_shuf, res['labels_sel'], stim_names)

            res['null_dist'][pi] = r2_shuf
            if not np.isnan(r2_shuf):
                perm_r2s.append(r2_shuf)

        mean_null_dist[pi] = np.nanmean(perm_r2s) if perm_r2s else np.nan
        if (pi + 1) % 100 == 0:
            print(f"      perm {pi + 1}/{n_perm}")

    # Per-source p-values
    for vi in valid_indices:
        res = source_results[vi]
        null = res['null_dist']
        res['p_value'] = (np.nansum(null >= res['r2']) + 1) / (n_perm + 1)

    mean_p_value = (np.nansum(mean_null_dist >= mean_r2) + 1) / (n_perm + 1)

    # Clean up large arrays from results to save memory
    for res in source_results:
        for key in ('X_sel', 'y_z', 'labels_sel'):
            res.pop(key, None)

    return source_results, mean_r2, mean_p_value, mean_null_dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(all_results, group_name, save_path=None):
    """Heatmap: rows = behavior types, cols = source monkeys, cells = R²."""
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
    ax.set_title(f'Ridge Decoding R² (single-trial LOSMO) — {group_name}')
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
    ax.set_ylabel('Mean LOO R² (LOSMO)')
    ax.set_title(f'Ridge Single-Trial Decoding — {group_name} — {region}  '
                 f'[{time_window_s[0]*1000:.0f}–{time_window_s[1]*1000:.0f} ms]')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  saved → {save_path}")
    return fig


def plot_null_distributions(all_results, save_dir=None):
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
    fig.suptitle('Permutation Null Distributions (single-trial)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_dir:
        path = os.path.join(save_dir, 'null_distributions.png')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"  saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==================== USER SETTINGS ====================
    GROUP = 'Zombies'                 # 'Zombies' | 'Best Frans' | 'Instigators'
    REGION = 'AMG'                    # 'AMG' | 'ER' | 'ALL'
    TIME_WINDOW_S = (0.200, 0.500)    # (start, end) in seconds
    N_PERM = 100
    SUBJECT_NAME = None               # None = auto-detect
    SAVE_DIR = (
        "/home/connorlab/Documents/GitHub/Julie/Cortana/"
        f"analysis_results/ridge_directional_single_trial/{REGION}/{GROUP}"
    )
    # =======================================================

    cfg = TrajectoryConfig(
        region=REGION,
        session=None,
        trial_averaged=True,   # not used for matrix building here,
                               # but needed for cfg.validate()
        peak_align=False,
        bin_width=0.050,
        min_epoch_duration=max(TIME_WINDOW_S[1], 2.0),
    )
    cfg.validate()

    # 1. Load behavior matrices
    print("=" * 60)
    print(f"Group: {GROUP}   Region: {REGION}   "
          f"Window: {TIME_WINDOW_S[0]*1000:.0f}–{TIME_WINDOW_S[1]*1000:.0f} ms")
    print("  Mode: SINGLE-TRIAL  |  CV: leave-one-stimulus-monkey-out")
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

    monkeys_in_data = set(df['MonkeyName'].unique())
    common = [m for m in monkey_names_in_matrix if m in monkeys_in_data]
    missing = [m for m in monkey_names_in_matrix if m not in monkeys_in_data]
    if missing:
        print(f"Monkeys in matrix but NOT in neural data: {missing}")
    print(f"Usable monkeys: {common}  (n={len(common)})")

    if len(common) < 4:
        raise RuntimeError(f"Only {len(common)} monkeys — too few.")

    # 3. Build single-trial population matrix
    X_pop, trial_labels, neuron_ids, n_per_monkey = \
        build_single_trial_population(df, cfg, common,
                                       time_window_s=TIME_WINDOW_S)

    # Soft-normalize
    X_pop = soft_normalize(X_pop, const=5.0)
    print(f"Neurons: {X_pop.shape[1]}   Total trials: {X_pop.shape[0]}")

    # 4. Behavior targets
    behavior_targets = []
    for beh_key in ('affiliation', 'agonism', 'submission'):
        bmat = mats[beh_key].loc[common, common]
        behavior_targets.append((f'{beh_key}_given',    bmat))
        behavior_targets.append((f'{beh_key}_received', bmat))

    # 5. Decode
    all_results = []
    for beh_name, bmat in behavior_targets:
        print("\n" + "=" * 60)
        print(f"Decoding: {beh_name}  (single-trial, LOSMO)")
        print("=" * 60)

        src_results, mean_r2, p_val, null = \
            decode_directional_single_trial(
                X_pop, trial_labels, common, bmat, beh_name,
                subject_name=subject_name, n_perm=N_PERM)

        print(f"  Per-source R²:")
        for sr in src_results:
            tag = " [skipped]" if sr.get('skipped') else ""
            p_str = (f"p={sr['p_value']:.4f}"
                     if not np.isnan(sr['p_value']) else "p=N/A")
            n_info = f"n_trials={sr['n_trials']}, n_stim={sr['n_stim']}"
            print(f"    {sr['source']:>12s}:  R² = {sr['r2']:+.4f}  "
                  f"{p_str}  ({n_info}){tag}")
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
                 save_path=os.path.join(SAVE_DIR, 'heatmap_r2.png'))
    plot_summary_bar(all_results, GROUP, REGION, TIME_WINDOW_S,
                     save_path=os.path.join(SAVE_DIR, 'summary_bar.png'))
    plot_null_distributions(all_results, save_dir=SAVE_DIR)

    # 7. Save CSVs
    os.makedirs(SAVE_DIR, exist_ok=True)

    # (a) Per-source detail
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
                'n_trials': sr['n_trials'],
                'r2': sr['r2'],
                'p_value': sr['p_value'],
                'skipped': sr.get('skipped', False),
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(SAVE_DIR, 'decoding_per_source.csv')
    detail_df.to_csv(detail_path, index=False)
    print(f"\nPer-source results → {detail_path}")
    print(detail_df.to_string(index=False))

    # (b) Behavior-level summary
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
            'total_trials': sum(sr['n_trials']
                                for sr in res['source_results']),
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
