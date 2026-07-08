# run_pc_padded_window_trajectory.py
"""
Option 2: Per-neuron response window, padded to a uniform length.

- For each neuron, use its lowest-p window from the pkl.
- Pad every window to the length of the longest window (split front/back;
  if odd, extra bin goes to the front).
- If the back pad would exceed the trial end (2000 ms), shift the overflow
  to the front. Same for front overflow (shift to back).
- Bin within the padded window at cfg.bin_width and build the standard
  (conditions * time_bins, neurons) PCA matrix. Plot short trajectories.

condition = identity.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TrajectoryConfig
from data_loading import load_and_filter
from preprocessing import preprocess
from pca_runner import run_pca
from plotting import (compute_trajectories,
                      plot_group_mean_2d_mpl, plot_pc_vs_time_group_mean,
                      plot_per_group_3d_mpl, plot_per_group_2d_mpl,
                      plot_all_shaded_3d_mpl, plot_all_shaded_3d_plotly)


MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"
WINDOWS_PKL = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/si_sorted_Zombies_all_windows_pKW_tested.pkl"
TRIAL_DURATION_S = 2.0


# ---------------------------------------------------------------------------
# 1. Pick one window per neuron + compute padded windows
# ---------------------------------------------------------------------------
def load_and_pad_windows(pkl_path, bin_width, trial_dur=TRIAL_DURATION_S):
    """
    Returns
    -------
    padded : dict {NeuronID: (pad_start_s, pad_end_s)}  -- all same length
    target_bins : int  -- number of bins in the padded window
    """
    with open(pkl_path, 'rb') as f:
        wdf = pickle.load(f)

    best = wdf.loc[wdf.groupby('NeuronID')['p-value'].idxmin()].reset_index(drop=True)
    best['size_ms'] = best['WindowEnd_ms'] - best['WindowStart_ms']

    # Drop neurons whose window extends past the trial end (computer-lag trials)
    trial_ms = int(round(trial_dur * 1000))
    over = best['WindowEnd_ms'] > trial_ms
    if over.any():
        print(f"Dropping {over.sum()} neurons with windows past {trial_ms} ms")
        best = best[~over].reset_index(drop=True)

    target_ms = int(best['size_ms'].max())

    # Enforce bin_width divisibility
    bw_ms = int(round(bin_width * 1000))
    if target_ms % bw_ms != 0:
        target_ms = int(np.ceil(target_ms / bw_ms)) * bw_ms
    target_bins = target_ms // bw_ms

    padded = {}
    for _, row in best.iterrows():
        s, e = int(row['WindowStart_ms']), int(row['WindowEnd_ms'])
        need = target_ms - (e - s)
        front = (need + 1) // 2   # extra goes to front if odd
        back = need - front
        ns, ne = s - front, e + back

        # Handle back overflow -> shift to front
        if ne > trial_ms:
            ns -= (ne - trial_ms)
            ne = trial_ms
        # Handle front overflow -> shift to back
        if ns < 0:
            ne += (-ns)
            ns = 0

        # Sanity check
        if ne > trial_ms or ns < 0 or (ne - ns) != target_ms:
            raise ValueError(
                f"Cannot pad {row['NeuronID']} window [{s},{e}] to {target_ms}ms "
                f"within [0,{trial_ms}] — trial too short.")
        padded[row['NeuronID']] = (ns / 1000.0, ne / 1000.0)

    print(f"Target window: {target_ms} ms = {target_bins} bins @ {bw_ms} ms")
    return padded, target_bins


# ---------------------------------------------------------------------------
# 2. Bin spikes within each neuron's padded window
# ---------------------------------------------------------------------------
def _bin_trial_padded(trial_rows, neuron_to_idx, padded_windows,
                      n_bins, bin_width):
    """
    Returns (n_bins, n_neurons) spike counts, where each neuron's bins cover
    its own padded window. Neurons not in padded_windows are skipped (zeros).
    """
    counts = np.zeros((n_bins, len(neuron_to_idx)), dtype=float)
    for _, row in trial_rows.iterrows():
        nid = row['NeuronID']
        if nid not in padded_windows:
            continue
        n_idx = neuron_to_idx[nid]
        start_s, end_s = padded_windows[nid]
        rel = row['SpikeTimes'] - row['EpochStartStop'][0]
        rel = rel[(rel >= start_s) & (rel < end_s)]
        bins = np.floor((rel - start_s) / bin_width).astype(int)
        bins = bins[(bins >= 0) & (bins < n_bins)]
        for b in bins:
            counts[b, n_idx] += 1
    return counts


# ---------------------------------------------------------------------------
# 3. Build matrix (same shape contract as binning_by_condition)
# ---------------------------------------------------------------------------
def build_matrix_padded(df, cfg, condition_map, padded_windows, n_bins,
                        min_reps=1, group_map=None):
    df = df[df['MonkeyName'].isin(condition_map)].copy()
    df['condition'] = df['MonkeyName'].map(condition_map)

    conditions = sorted(set(condition_map.values()))
    n_conditions = len(conditions)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}

    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName', 'condition']]
                    .reset_index())

    sessions = sorted(df['session'].unique())
    valid = []
    for s in sessions:
        counts = trial_meta[trial_meta['session'] == s].groupby('condition').size()
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
        neuron_ids = sorted([n for n in sess_df['NeuronID'].unique()
                             if n in padded_windows])
        if not neuron_ids:
            print(f"  skipping {session}: no neurons with windows")
            continue
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)

        sess_trials = trial_meta[trial_meta['session'] == session]
        sum_counts = np.zeros((n_conditions, n_bins, n_neurons), dtype=float)
        reps = np.zeros(n_conditions, dtype=int)

        for task_field, tgroup in sess_trials.groupby('TaskField'):
            c_idx = cond_to_idx[tgroup['condition'].iloc[0]]
            reps[c_idx] += 1
            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            sum_counts[c_idx] += _bin_trial_padded(
                trial_rows, neuron_to_idx, padded_windows, n_bins, cfg.bin_width)

        avg = sum_counts / reps[:, None, None]
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
                trial_averaged=True, n_conditions=n_conditions)
    print(f"PCA matrix (padded): {pca_matrix.shape} | {n_conditions} conditions, {n_bins} bins")
    return pca_matrix, row_meta_df, info


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrajectoryConfig(
        region='ER', session=None, trial_averaged=True, peak_align=False,
        n_components=6, bin_width=0.100, min_epoch_duration=2.0,
        analysis='identity',
    )
    cfg.validate()

    MEAN_CENTER = True
    SOFT_NORMALIZE = True
    MIN_REPS_PER_COND = 5
    SAVE = True
    PLOT_SAVE_DIR = f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/population_trajectory/{cfg.analysis}_padded_window'

    # --- load + pad windows ---
    padded_windows, n_bins = load_and_pad_windows(WINDOWS_PKL, cfg.bin_width)
    print(f"Padded {len(padded_windows)} neurons to uniform window")

    # --- load data + identity condition map ---
    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
    common = sorted(set.intersection(*monkeys_per_session))
    name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
    known = [m for m in common if m in name_to_group]
    condition_map = {m: m for m in known}
    group_map = {m: name_to_group[m] for m in known}
    print(f"identity analysis: {len(known)} monkeys")

    # --- build matrix, preprocess, PCA ---
    pca_matrix, row_meta_df, info = build_matrix_padded(
        df, cfg, condition_map, padded_windows, n_bins,
        min_reps=MIN_REPS_PER_COND, group_map=group_map)

    pca_matrix = preprocess(pca_matrix, info, cfg,
                            soft_normalize=SOFT_NORMALIZE, mean_center=MEAN_CENTER)
    pca_result = run_pca(pca_matrix, info, cfg)

    cond_trajs, group_trajs, c2g = compute_trajectories(
        pca_result, row_meta_df, info, cfg)

    var = pca_result['var']
    suffix = f"[{cfg.region}] {cfg.analysis} padded-win MC {MEAN_CENTER} SN {SOFT_NORMALIZE}"

    plot_group_mean_2d_mpl(group_trajs, var, cfg, suffix=suffix,
                           save=SAVE, save_dir=PLOT_SAVE_DIR)
    plot_pc_vs_time_group_mean(group_trajs, var, cfg, row_meta_df, info,
                               pcs=range(1, cfg.n_components + 1),
                               suffix=suffix, save=SAVE)

    plot_all_shaded_3d_plotly(cond_trajs, c2g, var, cfg, suffix,
                              save=SAVE, save_dir=PLOT_SAVE_DIR)
    plot_per_group_3d_mpl(cond_trajs, c2g, var, cfg, suffix,
                          save=SAVE, save_dir=PLOT_SAVE_DIR)
    plot_all_shaded_3d_mpl(cond_trajs, c2g, var, cfg, suffix,
                           save=SAVE, save_dir=PLOT_SAVE_DIR)
    plot_per_group_2d_mpl(cond_trajs, c2g, var, cfg, suffix=suffix,
                          save=SAVE, save_dir=PLOT_SAVE_DIR)
    plt.show()


if __name__ == '__main__':
    main()
