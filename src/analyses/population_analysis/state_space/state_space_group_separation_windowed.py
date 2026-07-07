# state_space_group_separation_windowed.py
"""
Sibling of state_space_group_separation.py.  Same categorical state-space test,
but the per-identity feature vector is built from a UNIFORM TIME WINDOW
(default 300–500 ms post-stimulus) rather than averaging across the full epoch.
This matches the RSA analysis window so the state-space and RSA tests ask
comparable questions on the same temporal slice.

WHY a sibling.  The original script bins the entire 0–2 s epoch, PCAs the time-
binned matrix, and then averages PCs across bins to get one point per identity.
If a familiarity signal is transient (e.g. a brief modulation 300–500 ms after
face onset), averaging it across the whole 2 s dilutes it with non-selective
onset transients and baseline, and the population geometry test sees noise.
Restricting the integration to the literature-motivated window removes that
dilution.

Together with state_space_group_separation.py, this gives a 2×2 defensive matrix
per region:
                       full epoch (0-2s)    windowed (300-500 ms)
  4-box original     state_space_group_separation.py   |  THIS SCRIPT
  3-box pooled       (also from above)                 |  (also from this)

The choice of (window × pooling) is committed up front — both scripts each run
both pooling schemes — so nothing is selected post-hoc.

The window default (300–500 ms) was chosen a priori from the face/familiarity
response literature and matches the RSA configuration.  Do NOT sweep windows
in this script: that is the cherry-pick we are explicitly avoiding.
"""

import os
import pickle
from math import factorial

import numpy as np
import pandas as pd

from config import TrajectoryConfig
from data_loading import load_and_filter
from pca_runner import run_pca

# Reuse all statistical machinery and plotting from the sibling script so the
# two analyses are guaranteed identical apart from the feature-matrix step.
from state_space_group_separation import (
    run_centroid_test, run_decoder_test, run_dim_sweep,
    plot_time_averaged, plot_dim_sweep,
    SOCIAL_GROUP, MONKEY_INFO_PATH,
)

# ── window (relative to EpochStartStop[0], i.e. stimulus onset) ──
WINDOW = (0.300, 0.500)


# ═══════════════════════════════════════════════════════════════════════
# Windowed feature-matrix builder (single uniform window per neuron)
# ═══════════════════════════════════════════════════════════════════════

def _rate_in_uniform_window(trial_rows, neuron_to_idx, n_neurons, start_s, end_s):
    """Per-neuron firing rate (Hz) for this trial in [start_s, end_s) post-stim."""
    rates = np.full(n_neurons, np.nan)
    dur_s = end_s - start_s
    for _, row in trial_rows.iterrows():
        nid = row['NeuronID']
        if nid not in neuron_to_idx:
            continue
        rel = np.asarray(row['SpikeTimes']) - row['EpochStartStop'][0]
        count = int(np.sum((rel >= start_s) & (rel < end_s)))
        rates[neuron_to_idx[nid]] = count / dur_s
    return rates


def build_matrix_windowed(df, condition_map, window, min_reps=5):
    """
    Build (n_identities, n_neurons_total) firing-rate matrix in the requested
    window. Mirrors run_pc_dots_by_window.build_matrix_dots but with a single
    uniform window applied to every neuron.
    """
    start_s, end_s = window
    df = df[df['MonkeyName'].isin(condition_map)].copy()
    df['condition'] = df['MonkeyName'].map(condition_map)

    conditions = sorted(set(condition_map.values()))
    n_conditions = len(conditions)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}

    trial_meta = (df.groupby(['session', 'TaskField'])
                    .first()[['MonkeyName', 'condition']].reset_index())

    # keep only sessions that have every identity at >= min_reps trials
    sessions = sorted(df['session'].unique())
    valid = []
    for s in sessions:
        counts = trial_meta[trial_meta['session'] == s].groupby('condition').size()
        if len(counts) < n_conditions or counts.min() < min_reps:
            continue
        valid.append(s)
    dropped = set(sessions) - set(valid)
    if dropped:
        print(f"  Dropped {len(dropped)} sessions (missing identities or <{min_reps} reps)")
    sessions = valid
    df = df[df['session'].isin(sessions)].reset_index(drop=True)
    trial_meta = trial_meta[trial_meta['session'].isin(sessions)].reset_index(drop=True)
    if not sessions:
        raise ValueError("No sessions survived filtering.")

    session_matrices, all_neuron_ids = [], []
    for session in sessions:
        sess_df = df[df['session'] == session]
        neuron_ids = sorted(sess_df['NeuronID'].unique())
        neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        n_neurons = len(neuron_ids)
        sess_trials = trial_meta[trial_meta['session'] == session]

        sum_rates = np.zeros((n_conditions, n_neurons), dtype=float)
        reps = np.zeros(n_conditions, dtype=int)
        for task_field, tgroup in sess_trials.groupby('TaskField'):
            c_idx = cond_to_idx[tgroup['condition'].iloc[0]]
            trial_rows = sess_df[sess_df['TaskField'] == task_field]
            rates = _rate_in_uniform_window(trial_rows, neuron_to_idx, n_neurons,
                                            start_s, end_s)
            mask = ~np.isnan(rates)
            sum_rates[c_idx, mask] += rates[mask]
            reps[c_idx] += 1
        avg = sum_rates / reps[:, None]            # (n_conditions, n_neurons)
        session_matrices.append(avg)
        all_neuron_ids.extend(f"{session}__{nid}" for nid in neuron_ids)

    pca_matrix = np.hstack(session_matrices)       # (n_conditions, n_neurons_total)
    info = dict(n_bins=1, bin_width=None,
                conditions=conditions, neuron_ids=all_neuron_ids,
                trial_averaged=True, n_conditions=n_conditions)
    print(f"  Windowed matrix [{int(start_s*1000)}–{int(end_s*1000)} ms]: "
          f"{pca_matrix.shape}  |  {n_conditions} identities, "
          f"{len(sessions)} sessions, {pca_matrix.shape[1]} neurons total")
    return pca_matrix, info


def preprocess_windowed(R, soft_normalize=True, mean_center=True, soft_const=5.0):
    """Soft-norm range over identities (no time axis), then optional mean-center."""
    R = R.astype(float).copy()
    if soft_normalize:
        rng = R.max(axis=0) - R.min(axis=0)
        R = R / (rng + soft_const)
    if mean_center:
        R = R - R.mean(axis=0, keepdims=True)
    return R


# ═══════════════════════════════════════════════════════════════════════
# Per-region pipeline
# ═══════════════════════════════════════════════════════════════════════

def analyze_region(region, window, dims_sweep,
                   n_pc_centroid_report, n_pc_decoder_report,
                   n_perms=1000, n_perms_sweep=500,
                   mean_center=True, soft_normalize=True,
                   min_reps=5, seed=42, output_dir=None, pool_strangers=False):
    print(f"\n{'═' * 76}")
    print(f"  STATE-SPACE GROUP SEPARATION (WINDOWED) — region = {region}")
    print(f"{'═' * 76}")

    n_components = max(max(dims_sweep), n_pc_centroid_report, n_pc_decoder_report)
    cfg = TrajectoryConfig(
        region=region, session=None, trial_averaged=True, peak_align=False,
        n_components=n_components, bin_width=0.050, min_epoch_duration=2.0,
        analysis='identity',
    )
    cfg.validate()

    if output_dir is None:
        output_dir = f'./state_space_group_separation_windowed_{region}'
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_filter(cfg)
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    monkeys_per_session = df.groupby('session')['MonkeyName'].apply(set)
    common = sorted(set.intersection(*monkeys_per_session))
    name_to_group = dict(zip(info_df['Name'].astype(str), info_df['Group Name']))
    known = [m for m in common if m in name_to_group]
    condition_map = {m: m for m in known}
    group_map = {m: name_to_group[m] for m in known}

    # KEY DIFFERENCE: windowed (n_identities, n_neurons) matrix, no time bins.
    pca_matrix, info = build_matrix_windowed(df, condition_map,
                                             window=window, min_reps=min_reps)
    pca_matrix_pp = preprocess_windowed(pca_matrix,
                                        soft_normalize=soft_normalize,
                                        mean_center=mean_center)
    pca_result = run_pca(pca_matrix_pp, info, cfg)

    identities = list(info['conditions'])
    n_neurons = pca_matrix_pp.shape[1]
    n_ids = len(identities)
    pc_mean_full = pca_result['scores_3d'][:, 0, :]   # squeeze singleton bin axis
    raw_group_labels = np.array([group_map[i] for i in identities])

    if pool_strangers:
        missing = sorted(set(raw_group_labels) - set(SOCIAL_GROUP))
        if missing:
            raise KeyError(f"SOCIAL_GROUP missing: {missing} — edit map.")
        group_labels = np.array([SOCIAL_GROUP[g] for g in raw_group_labels])
        label_scheme = '3box_pooled_strangers'
    else:
        group_labels = raw_group_labels
        label_scheme = '4box_original'
    var = pca_result['var']

    win_tag = f'{int(window[0]*1000)}-{int(window[1]*1000)}ms'
    centroid_features = pc_mean_full[:, :n_pc_centroid_report]
    decoder_features = pc_mean_full[:, :n_pc_decoder_report]

    print(f"\n  Window: {win_tag}  |  Label scheme: {label_scheme}  "
          f"({'4 groups, original' if not pool_strangers else '3 groups; Instigators+Stranger Things → Strangers'})")
    print(f"  Identities ({n_ids}):")
    for g in sorted(np.unique(group_labels)):
        members = [i for i, l in zip(identities, group_labels) if l == g]
        print(f"    {g:<18s} ({len(members)}): {', '.join(members)}")
    print(f"\n  soft_normalize={soft_normalize}  mean_center={mean_center}  PCA n_comp={n_components}")
    print(f"  report panel: centroid=top {n_pc_centroid_report} PCs, decoder=top {n_pc_decoder_report} PCs (fixed)")
    print(f"  dim sweep: {dims_sweep}")
    print(f"  PCA var: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))
    _, inv = np.unique(group_labels, return_inverse=True)
    grp_sizes = np.bincount(inv)
    n_distinct = factorial(n_ids) // int(np.prod([factorial(int(s)) for s in grp_sizes]))
    print(f"  distinct label permutations: {n_distinct}")

    print(f"\n  [1] Centroid test (top {n_pc_centroid_report} PCs)")
    centroid_res = run_centroid_test(centroid_features, group_labels,
                                     n_perms=n_perms, seed=seed)
    om = centroid_res['omnibus']
    print(f"      omnibus obs={om['obs']:.4f}  z={om['z']:.2f}  p={om['p']:.4f}  "
          f"CI[{om['ci'][0]:.3f},{om['ci'][1]:.3f}]")

    print(f"  [2] LOO decoder (top {n_pc_decoder_report} PCs)")
    decoder_res = run_decoder_test(decoder_features, group_labels,
                                   n_perms=n_perms, seed=seed)
    om = decoder_res['omnibus']
    print(f"      omnibus obs={om['obs']:.4f}  z={om['z']:.2f}  p={om['p']:.4f}  "
          f"CI[{om['ci'][0]:.3f},{om['ci'][1]:.3f}]")

    print(f"\n  [3] Dimensionality sweep (n_perms_sweep={n_perms_sweep})")
    centroid_sweep = run_dim_sweep(pc_mean_full, group_labels, dims_sweep,
                                   kind='centroid', n_perms=n_perms_sweep, seed=seed)
    decoder_sweep = run_dim_sweep(pc_mean_full, group_labels, dims_sweep,
                                  kind='decoder', n_perms=n_perms_sweep, seed=seed)
    print(f"      centroid: best dim={centroid_sweep['best_dim']}, "
          f"sweep-corrected p={centroid_sweep['p_sweep_corrected']:.4f}")
    print(f"      decoder : best dim={decoder_sweep['best_dim']}, "
          f"sweep-corrected p={decoder_sweep['p_sweep_corrected']:.4f}")
    for nm, sw in [('centroid', centroid_sweep), ('decoder', decoder_sweep)]:
        print(f"        {nm}: " + "  ".join(
            f"d{d}:z={z:+.2f}" for d, z in zip(sw['dims'], sw['z'])))

    plot_time_averaged(centroid_res, decoder_res, region,
                       os.path.join(output_dir,
                                    f'time_averaged_{region}_{win_tag}_{label_scheme}.png'),
                       centroid_label=f'top {n_pc_centroid_report} PCs',
                       decoder_label=f'top {n_pc_decoder_report} PCs')
    plot_dim_sweep(centroid_sweep, decoder_sweep, region,
                   os.path.join(output_dir,
                                f'dim_sweep_{region}_{win_tag}_{label_scheme}.png'))
    print(f"\n  Figures saved to: {output_dir}  (tagged {win_tag}, {label_scheme})")

    results = dict(
        region=region, identities=identities,
        window=window, window_tag=win_tag,
        label_scheme=label_scheme, pool_strangers=pool_strangers,
        group_labels=group_labels.tolist(),
        raw_group_labels=raw_group_labels.tolist(), group_map=group_map,
        n_neurons=n_neurons, n_components=n_components,
        n_pc_centroid_report=n_pc_centroid_report, n_pc_decoder_report=n_pc_decoder_report,
        mean_center=mean_center, soft_normalize=soft_normalize,
        pca_variance=var.tolist(),
        time_averaged=dict(centroid=centroid_res, decoder=decoder_res),
        dim_sweep=dict(centroid=centroid_sweep, decoder=decoder_sweep),
    )
    pkl_path = os.path.join(
        output_dir,
        f'state_space_group_separation_windowed_{region}_{win_tag}_{label_scheme}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Pickle saved ({win_tag}, {label_scheme}).")
    return results


def main():
    N_PERMS = 1000
    N_PERMS_SWEEP = 500
    DIMS_SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    N_PC_CENTROID_REPORT = 3
    N_PC_DECODER_REPORT = 6
    MEAN_CENTER = True
    SOFT_NORMALIZE = True
    MIN_REPS = 5
    SEED = 42
    OUTPUT_ROOT = './state_space_group_separation_windowed'
    REGIONS = ['AMG', 'ER']

    # Run BOTH pooling schemes per region — committed up front, no cherry-pick.
    POOL_OPTIONS = [(False, '4box_original'),
                    (True,  '3box_pooled_strangers')]

    all_results = {}
    for region in REGIONS:
        all_results[region] = {}
        for pool_strangers, tag in POOL_OPTIONS:
            all_results[region][tag] = analyze_region(
                region=region, window=WINDOW, dims_sweep=DIMS_SWEEP,
                n_pc_centroid_report=N_PC_CENTROID_REPORT,
                n_pc_decoder_report=N_PC_DECODER_REPORT,
                n_perms=N_PERMS, n_perms_sweep=N_PERMS_SWEEP,
                mean_center=MEAN_CENTER, soft_normalize=SOFT_NORMALIZE,
                min_reps=MIN_REPS, seed=SEED,
                output_dir=os.path.join(OUTPUT_ROOT, region),
                pool_strangers=pool_strangers)

    win_tag = f'{int(WINDOW[0]*1000)}-{int(WINDOW[1]*1000)}ms'
    print(f"\n\n{'═'*100}")
    print(f"  CROSS-REGION SUMMARY (state space, WINDOWED {win_tag}) — both label schemes")
    print(f"{'═'*100}")
    print(f"  {'region':<6} {'scheme':<22} {'test':<28} {'obs':>8} {'z':>7} {'p':>9}")
    print('-' * 100)
    for region in REGIONS:
        for _, tag in POOL_OPTIONS:
            r = all_results[region][tag]
            c = r['time_averaged']['centroid']['omnibus']
            d = r['time_averaged']['decoder']['omnibus']
            cs = r['dim_sweep']['centroid']
            ds = r['dim_sweep']['decoder']
            print(f"  {region:<6} {tag:<22} {'centroid (report dim)':<28} "
                  f"{c['obs']:>8.4f} {c['z']:>7.2f} {c['p']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'decoder (report dim)':<28} "
                  f"{d['obs']:>8.4f} {d['z']:>7.2f} {d['p']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'centroid (sweep-corr.)':<28} "
                  f"{'—':>8} {'—':>7} {cs['p_sweep_corrected']:>9.4f}")
            print(f"  {region:<6} {tag:<22} {'decoder (sweep-corr.)':<28} "
                  f"{'—':>8} {'—':>7} {ds['p_sweep_corrected']:>9.4f}")
        print()


if __name__ == '__main__':
    main()
