# binning.py
import numpy as np
import pandas as pd


def _bin_trial(trial_rows, neuron_to_idx, n_bins, bin_width, analysis_window):
    counts = np.zeros((n_bins, len(neuron_to_idx)), dtype=float)
    for _, row in trial_rows.iterrows():
        n_idx = neuron_to_idx[row['NeuronID']]
        rel = row['SpikeTimes'] - row['EpochStartStop'][0]
        rel = rel[(rel >= 0) & (rel < analysis_window)]
        bins = np.floor(rel / bin_width).astype(int)
        bins = bins[bins < n_bins]
        for b in bins:
            counts[b, n_idx] += 1
    return counts


def build_matrix(df, cfg):
    """
    Returns (pca_matrix, row_meta_df, info).

    Row order (averaged):     monkey -> time_bin
    Row order (non-averaged): monkey -> rep -> time_bin
    """
    n_bins = int(np.floor(cfg.min_epoch_duration / cfg.bin_width))
    analysis_window = n_bins * cfg.bin_width

    sessions = sorted(df['session'].unique())
    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName', 'MonkeyGroup']]
                    .reset_index())

    # Common monkeys across sessions
    monkey_per_session = trial_meta.groupby('session')['MonkeyName'].apply(set)
    common_monkeys = sorted(set.intersection(*monkey_per_session))

    # Drop sessions with too few reps for any common monkey
    rep_counts = (trial_meta[trial_meta['MonkeyName'].isin(common_monkeys)]
                  .groupby(['session', 'MonkeyName']).size()
                  .groupby('session').min())
    valid = rep_counts[rep_counts >= cfg.min_reps_per_monkey].index.tolist()
    dropped = set(sessions) - set(valid)
    if dropped:
        print(f"Dropped sessions (< {cfg.min_reps_per_monkey} reps): {sorted(dropped)}")
    sessions = sorted(valid)
    df = df[df['session'].isin(sessions)].reset_index(drop=True)
    trial_meta = trial_meta[trial_meta['session'].isin(sessions)].reset_index(drop=True)

    if cfg.trial_averaged:
        return _build_averaged(df, trial_meta, sessions, common_monkeys,
                               n_bins, analysis_window, cfg)
    return _build_non_averaged(df, trial_meta, sessions, common_monkeys,
                                n_bins, analysis_window, cfg)


def _build_averaged(df, trial_meta, sessions, common_monkeys,
                    n_bins, analysis_window, cfg):
    monkey_to_idx = {m: i for i, m in enumerate(common_monkeys)}
    n_monkeys = len(common_monkeys)

    session_matrices, all_neuron_ids = [], []
    for session in sessions:
        sess_df = df[df['session'] == session]
        neuron_ids = sorted(sess_df['NeuronID'].unique())
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)

        sess_trials = trial_meta[(trial_meta['session'] == session)
                                 & (trial_meta['MonkeyName'].isin(common_monkeys))]

        sum_counts = np.zeros((n_monkeys, n_bins, n_neurons), dtype=float)
        reps = np.zeros(n_monkeys, dtype=int)

        for task_field, tgroup in sess_trials.groupby('TaskField'):
            m_idx = monkey_to_idx[tgroup['MonkeyName'].iloc[0]]
            reps[m_idx] += 1
            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            sum_counts[m_idx] += _bin_trial(trial_rows, neuron_to_idx,
                                            n_bins, cfg.bin_width, analysis_window)

        avg = sum_counts / reps[:, None, None]
        session_matrices.append(avg.reshape(n_monkeys * n_bins, n_neurons))
        all_neuron_ids.extend(f"{session}__{nid}" for nid in neuron_ids)

    pca_matrix = np.hstack(session_matrices)

    m2g = (trial_meta.drop_duplicates('MonkeyName')
           .set_index('MonkeyName')['MonkeyGroup'].to_dict())
    row_keys = [(m, b) for m in common_monkeys for b in range(n_bins)]
    row_meta_df = pd.DataFrame(row_keys, columns=['monkey_name', 'time_bin'])
    row_meta_df['time_bin_start_s'] = row_meta_df['time_bin'] * cfg.bin_width
    row_meta_df['monkey_group'] = row_meta_df['monkey_name'].map(m2g)

    info = dict(n_bins=n_bins, bin_width=cfg.bin_width,
                common_monkeys=common_monkeys, neuron_ids=all_neuron_ids,
                trial_averaged=True, n_monkeys=n_monkeys)
    print(f"PCA matrix (averaged): {pca_matrix.shape}")
    return pca_matrix, row_meta_df, info


def _build_non_averaged(df, trial_meta, sessions, common_monkeys,
                        n_bins, analysis_window, cfg):
    trial_meta = trial_meta.sort_values(['session', 'MonkeyName', 'TaskField']).copy()
    trial_meta['rep'] = trial_meta.groupby(['session', 'MonkeyName']).cumcount()

    rep_counts = (trial_meta[trial_meta['MonkeyName'].isin(common_monkeys)]
                  .groupby(['session', 'MonkeyName']).size()
                  .groupby('session').min())
    global_min_reps = int(rep_counts.min())
    print(f"Using {global_min_reps} reps/monkey/session (non-averaged)")

    trial_meta = trial_meta[trial_meta['rep'] < global_min_reps]
    n_monkeys = len(common_monkeys)

    row_keys = [(m, r, b) for m in common_monkeys
                           for r in range(global_min_reps)
                           for b in range(n_bins)]

    session_matrices, all_neuron_ids = [], []
    for session in sessions:
        sess_df = df[df['session'] == session]
        neuron_ids = sorted(sess_df['NeuronID'].unique())
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)

        sess_trials = trial_meta[(trial_meta['session'] == session)
                                 & (trial_meta['MonkeyName'].isin(common_monkeys))]

        trial_bins = {}
        for _, trow in sess_trials.iterrows():
            trial_rows = sess_df[sess_df['TaskField'] == trow['TaskField']]
            trial_bins[(trow['MonkeyName'], trow['rep'])] = _bin_trial(
                trial_rows, neuron_to_idx, n_bins, cfg.bin_width, analysis_window)

        mat = np.zeros((len(row_keys), n_neurons), dtype=float)
        for i, (m, r, b) in enumerate(row_keys):
            counts = trial_bins.get((m, r))
            if counts is not None:
                mat[i] = counts[b]

        session_matrices.append(mat)
        all_neuron_ids.extend(f"{session}__{nid}" for nid in neuron_ids)

    pca_matrix = np.hstack(session_matrices)

    m2g = (trial_meta.drop_duplicates('MonkeyName')
           .set_index('MonkeyName')['MonkeyGroup'].to_dict())
    row_meta_df = pd.DataFrame(row_keys, columns=['monkey_name', 'rep', 'time_bin'])
    row_meta_df['time_bin_start_s'] = row_meta_df['time_bin'] * cfg.bin_width
    row_meta_df['monkey_group'] = row_meta_df['monkey_name'].map(m2g)

    info = dict(n_bins=n_bins, bin_width=cfg.bin_width,
                common_monkeys=common_monkeys, neuron_ids=all_neuron_ids,
                trial_averaged=False, n_monkeys=n_monkeys,
                global_min_reps=global_min_reps)
    print(f"PCA matrix (non-averaged): {pca_matrix.shape}")
    return pca_matrix, row_meta_df, info