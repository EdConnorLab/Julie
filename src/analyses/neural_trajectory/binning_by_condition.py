# binning_by_condition.py
"""
Build a PCA matrix where the condition axis is user-defined via a
{MonkeyName: label} mapping. Used for Option B analyses where you want
the PCs optimized for a specific comparison (group, familiarity, sex,
rank, etc.) rather than per-stimulus-monkey identity.

Trial-averaged only. Pooled across sessions.
"""
import numpy as np
import pandas as pd
from binning import _bin_trial


def build_matrix_by_condition(df, cfg, condition_map, min_reps=1, group_map=None):
    """
    Parameters
    ----------
    df : DataFrame
        Output of load_and_filter.
    cfg : TrajectoryConfig
    condition_map : dict {MonkeyName: condition_label}
        Trials whose MonkeyName is not in this dict are excluded.
    min_reps : int
        Minimum trials per condition per session. Sessions where any
        condition has fewer than this are dropped.

    Returns
    -------
    pca_matrix : ndarray, shape (n_conditions * n_bins, n_neurons_total)
    row_meta_df : DataFrame with columns (condition, time_bin, time_bin_start_s)
    info : dict
    """
    n_bins = int(np.floor(cfg.min_epoch_duration / cfg.bin_width))
    analysis_window = n_bins * cfg.bin_width

    df = df[df['MonkeyName'].isin(condition_map)].copy()
    df['condition'] = df['MonkeyName'].map(condition_map)

    conditions = sorted(set(condition_map.values()))
    n_conditions = len(conditions)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}

    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName', 'condition']]
                    .reset_index())

    # Drop sessions missing any condition or with too few reps
    sessions = sorted(df['session'].unique())
    valid = []
    for s in sessions:
        counts = (trial_meta[trial_meta['session'] == s]
                  .groupby('condition').size())
        # --- diagnostic (comment out if noisy) ---
        # missing = [c for c in conditions if c not in counts.index]
        # short = {c: int(counts[c]) for c in counts.index if counts[c] < min_reps}
        # if missing or short:
        #     # print(f"  [diagnostics] session {s}: missing={missing} | "
        #     #       f"below_min_reps({min_reps})={short}")
        # -----------------------------------------
        if len(counts) < n_conditions:
            continue
        if counts.min() < min_reps:
            continue
        valid.append(s)
    dropped = set(sessions) - set(valid)
    if dropped:
        print(f"Dropped {len(dropped)} sessions "
              f"(missing conditions or < {min_reps} reps/cond)")
    sessions = valid
    df = df[df['session'].isin(sessions)].reset_index(drop=True)
    trial_meta = trial_meta[trial_meta['session'].isin(sessions)].reset_index(drop=True)

    if not sessions:
        raise ValueError("No sessions survived filtering.")
    # For every session, compute mean firing count in each bin for each neuron across all trials of each condition
    session_matrices, all_neuron_ids = [], []
    for session in sessions:
        sess_df = df[df['session'] == session]
        neuron_ids = sorted(sess_df['NeuronID'].unique())
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)

        sess_trials = trial_meta[trial_meta['session'] == session]

        sum_counts = np.zeros((n_conditions, n_bins, n_neurons), dtype=float)
        reps = np.zeros(n_conditions, dtype=int)

        for task_field, tgroup in sess_trials.groupby('TaskField'):
            c_idx = cond_to_idx[tgroup['condition'].iloc[0]] # find the index of (number associated with) the condition
            reps[c_idx] += 1
            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            sum_counts[c_idx] += _bin_trial(
                trial_rows, neuron_to_idx, n_bins, cfg.bin_width, analysis_window)

        avg = sum_counts / reps[:, None, None] # reshapes reps from (n_conditions,) to (n_conditions, 1, 1) so NumPy can broadcast it against the 3D sum_counts array
        session_matrices.append(avg.reshape(n_conditions * n_bins, n_neurons))
        all_neuron_ids.extend(f"{session}__{nid}" for nid in neuron_ids)

    pca_matrix = np.hstack(session_matrices)

    row_keys = [(c, b) for c in conditions for b in range(n_bins)]
    row_meta_df = pd.DataFrame(row_keys, columns=['condition', 'time_bin'])
    row_meta_df['time_bin_start_s'] = row_meta_df['time_bin'] * cfg.bin_width

    if group_map is not None:
        row_meta_df['group'] = row_meta_df['condition'].map(group_map)

    info = dict(n_bins=n_bins, bin_width=cfg.bin_width,
                conditions=conditions, neuron_ids=all_neuron_ids,
                trial_averaged=True,
                n_conditions=n_conditions)
    print(f"PCA matrix by condition: {pca_matrix.shape} | "
          f"{n_conditions} conditions: {conditions}")
    return pca_matrix, row_meta_df, info
