# run_pc_dots_by_window.py
"""
Option 1: For each neuron, use its own response window (lowest-p KW window
from the pkl file). Compute ONE average firing rate per (neuron, condition).
No time dimension -> one dot per condition in PC space.

condition = identity (stimulus-monkey name).
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TrajectoryConfig
from data_loading import load_and_filter
from pca_runner import run_pca


def preprocess_dots(pca_matrix, softnorm_const=5.0,
                    soft_normalize=True, mean_center=False):
    """
    Preprocess a (n_conditions, n_neurons) firing-rate matrix. Already in Hz,
    so no counts->rate conversion. Soft-norm range is over conditions only
    (no time axis).
    """
    R = pca_matrix.astype(float).copy()
    if soft_normalize:
        rng = R.max(axis=0) - R.min(axis=0)          # (n_neurons,)
        R = R / (rng + softnorm_const)
    if mean_center:
        R = R - R.mean(axis=0, keepdims=True)
    print(f"Preprocess (dots): Hz | softnorm={'on' if soft_normalize else 'off'} "
          f"(c={softnorm_const}) | mean_center={'on' if mean_center else 'off'}")
    print(f"  matrix range after preprocess: [{R.min():.3f}, {R.max():.3f}]")
    return R


MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"
WINDOWS_PKL = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/si_sorted_Zombies_all_windows_pKW_tested.pkl"


# ---------------------------------------------------------------------------
# 1. Pick one window per neuron (lowest p-value)
# ---------------------------------------------------------------------------
def load_neuron_windows(pkl_path):
    """Returns dict {NeuronID: (start_s, end_s, duration_s)}."""
    with open(pkl_path, 'rb') as f:
        wdf = pickle.load(f)
    best = wdf.loc[wdf.groupby('NeuronID')['p-value'].idxmin()].reset_index(drop=True)
    return {
        row['NeuronID']: (row['WindowStart_ms'] / 1000.0,
                          row['WindowEnd_ms'] / 1000.0,
                          (row['WindowEnd_ms'] - row['WindowStart_ms']) / 1000.0)
        for _, row in best.iterrows()
    }


# ---------------------------------------------------------------------------
# 2. Build matrix: one firing rate per (condition, neuron).   No time axis.
# ---------------------------------------------------------------------------
def _rate_in_window(trial_rows, neuron_to_idx, neuron_windows, n_neurons):
    """Firing rate (Hz) per neuron for this trial, using each neuron's own window."""
    rates = np.full(n_neurons, np.nan)
    for _, row in trial_rows.iterrows():
        nid = row['NeuronID']
        if nid not in neuron_windows:
            continue
        start_s, end_s, dur_s = neuron_windows[nid]
        rel = row['SpikeTimes'] - row['EpochStartStop'][0]
        count = np.sum((rel >= start_s) & (rel < end_s))
        rates[neuron_to_idx[nid]] = count / dur_s
    return rates


def build_matrix_dots(df, cfg, condition_map, neuron_windows, min_reps=1):
    """
    Returns
    -------
    pca_matrix : (n_conditions, n_neurons_total)    -- one row per condition
    row_meta_df: DataFrame with column 'condition'
    info       : dict
    """
    df = df[df['MonkeyName'].isin(condition_map)].copy()
    df['condition'] = df['MonkeyName'].map(condition_map)

    conditions = sorted(set(condition_map.values()))
    n_conditions = len(conditions)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}

    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName', 'condition']]
                    .reset_index())

    # Drop sessions missing conditions or with too few reps
    sessions = sorted(df['session'].unique())
    valid = []
    for s in sessions:
        counts = (trial_meta[trial_meta['session'] == s]
                  .groupby('condition').size())
        if len(counts) < n_conditions or counts.min() < min_reps:
            continue
        valid.append(s)
    dropped = set(sessions) - set(valid)
    if dropped:
        print(f"Dropped {len(dropped)} sessions (missing conditions or < {min_reps} reps)")
    sessions = valid
    df = df[df['session'].isin(sessions)].reset_index(drop=True)
    trial_meta = trial_meta[trial_meta['session'].isin(sessions)].reset_index(drop=True)
    if not sessions:
        raise ValueError("No sessions survived filtering.")

    session_matrices, all_neuron_ids = [], []
    for session in sessions:
        sess_df = df[df['session'] == session]
        # Only keep neurons that have a window in the pkl
        neuron_ids = sorted([n for n in sess_df['NeuronID'].unique()
                             if n in neuron_windows])
        if not neuron_ids:
            print(f"  skipping {session}: no neurons with windows")
            continue
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)

        sess_trials = trial_meta[trial_meta['session'] == session]

        sum_rates = np.zeros((n_conditions, n_neurons), dtype=float)
        reps = np.zeros(n_conditions, dtype=int)

        for task_field, tgroup in sess_trials.groupby('TaskField'):
            c_idx = cond_to_idx[tgroup['condition'].iloc[0]]
            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            rates = _rate_in_window(trial_rows, neuron_to_idx,
                                    neuron_windows, n_neurons)
            # Accumulate only non-NaN (neurons present on this trial)
            mask = ~np.isnan(rates)
            sum_rates[c_idx, mask] += rates[mask]
            reps[c_idx] += 1

        avg = sum_rates / reps[:, None]            # (n_conditions, n_neurons)
        session_matrices.append(avg)
        all_neuron_ids.extend(f"{session}__{nid}" for nid in neuron_ids)

    pca_matrix = np.hstack(session_matrices)       # (n_conditions, n_neurons_total)

    row_meta_df = pd.DataFrame({'condition': conditions})
    info = dict(n_bins=1, bin_width=None,
                conditions=conditions, neuron_ids=all_neuron_ids,
                trial_averaged=True, n_conditions=n_conditions)
    print(f"PCA matrix (dots): {pca_matrix.shape} | {n_conditions} conditions")
    return pca_matrix, row_meta_df, info


# ---------------------------------------------------------------------------
# 3. Plot dots in PC space
# ---------------------------------------------------------------------------
def plot_pc_dots_2d(pca_result, info, group_map=None, title='', save_path=None):
    scores = pca_result['scores_3d'][:, 0, :]   # (n_conditions, n_components)
    var = pca_result['var']
    conds = info['conditions']

    fig, ax = plt.subplots(figsize=(7, 6))
    groups = [group_map.get(c, c) if group_map else c for c in conds]
    uniq_groups = sorted(set(groups))
    cmap = plt.cm.tab10
    g2color = {g: cmap(i % 10) for i, g in enumerate(uniq_groups)}

    for i, c in enumerate(conds):
        color = g2color[groups[i]]
        ax.scatter(scores[i, 0], scores[i, 1], s=80, color=color,
                   edgecolor='k', linewidth=0.5)
        ax.annotate(str(c), (scores[i, 0], scores[i, 1]),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_xlabel(f'PC1 ({var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrajectoryConfig(
        region='AMG', session=None, trial_averaged=True, peak_align=False,
        n_components=6, bin_width=0.050, min_epoch_duration=2.0,
        analysis='identity',
    )
    cfg.validate()

    MEAN_CENTER = True
    SOFT_NORMALIZE = True
    MIN_REPS_PER_COND = 5
    SAVE = True
    PLOT_SAVE_DIR = f'/Cortana/analysis_results/population_trajectory/{cfg.analysis}_dots'

    # --- load windows ---
    neuron_windows = load_neuron_windows(WINDOWS_PKL)
    print(f"Loaded windows for {len(neuron_windows)} neurons")

    # --- load data, set up identity condition map ---
    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)

    monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
    common = sorted(set.intersection(*monkeys_per_session))
    name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
    known = [m for m in common if m in name_to_group]
    condition_map = {m: m for m in known}
    group_map = {m: name_to_group[m] for m in known}
    print(f"identity analysis: {len(known)} monkeys")

    # --- build dot matrix, preprocess, PCA ---
    pca_matrix, row_meta_df, info = build_matrix_dots(
        df, cfg, condition_map, neuron_windows, min_reps=MIN_REPS_PER_COND)

    pca_matrix = preprocess_dots(pca_matrix,
                                 soft_normalize=SOFT_NORMALIZE,
                                 mean_center=MEAN_CENTER)
    pca_result = run_pca(pca_matrix, info, cfg)

    # --- plot ---
    suffix = f"[{cfg.region}] {cfg.analysis} dots MC {MEAN_CENTER} SN {SOFT_NORMALIZE}"
    save_path = f"{PLOT_SAVE_DIR}/pc_dots_2d.png" if SAVE else None
    plot_pc_dots_2d(pca_result, info, group_map=group_map,
                    title=suffix, save_path=save_path)
    plt.show()


if __name__ == '__main__':
    main()