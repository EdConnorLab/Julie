# rsa_pkl_core.py
"""
Core RSA functions for the "pkl-driven" workflow:
  - load_pkl_windows:                pkl file → {NeuronID: (win_start_s, win_end_s)}
  - compute_firing_rates_pkl:        per-neuron windows (or a common window) → rate matrix
  - run_rsa_pkl_pseudopop:           pool selected neurons across sessions and run one RSA
  - run_rsa_pkl_pseudopop_social:    same, but returns the dict shape expected by run_rsa_social.py

The "result" dict returned by run_rsa_pkl_pseudopop has the exact same
structure as rsa_core.run_rsa_pseudopop, so downstream plotting and social
pipelines work unchanged.
"""
import numpy as np
import pandas as pd

from rsa_core import (
    normalize_rates,
    build_neural_rdm,
    build_model_rdms,
    compare_rdms,
    compare_rdms_partial,
)


# ──────────────────────────────────────────────────────────
# 1. Pkl loading + filtering + deduplication
# ──────────────────────────────────────────────────────────

def load_pkl_windows(cfg):
    """
    Load (NeuronID, window) pairs from cfg.pkl_path, optionally filter by
    p-value, dedupe by keeping the most significant window per NeuronID.

    Parameters
    ----------
    cfg : PklRSAConfig or PklSocialRSAConfig

    Returns
    -------
    neuron_windows : dict {NeuronID (str) : (win_start_s, win_end_s) tuple}
        Window edges converted to seconds.
    pkl_df : DataFrame
        The filtered, deduplicated pkl rows (for logging / sanity checks).
    """
    pkl_df = pd.read_pickle(cfg.pkl_path)

    required = [cfg.pkl_neuron_id_col, cfg.pkl_window_start_col,
                cfg.pkl_window_end_col, cfg.pkl_p_value_col]
    missing = [c for c in required if c not in pkl_df.columns]
    if missing:
        raise ValueError(
            f"pkl is missing required column(s): {missing}.  "
            f"Available columns: {list(pkl_df.columns)}")

    n_total = len(pkl_df)
    print(f"  Loaded {n_total} rows from {cfg.pkl_path}")

    # ── Optional p-value filter ──
    if cfg.pkl_filter_significant:
        pkl_df = pkl_df[pkl_df[cfg.pkl_p_value_col] < cfg.pkl_p_threshold].copy()
        print(f"  After p < {cfg.pkl_p_threshold} filter: {len(pkl_df)} rows "
              f"({n_total - len(pkl_df)} dropped)")

    # ── Deduplicate: keep min p-value per NeuronID ──
    n_pre = len(pkl_df)
    pkl_df = (pkl_df.sort_values(cfg.pkl_p_value_col, ascending=True)
                    .drop_duplicates(subset=[cfg.pkl_neuron_id_col], keep='first')
                    .reset_index(drop=True))
    n_dup_removed = n_pre - len(pkl_df)
    if n_dup_removed > 0:
        print(f"  Deduplication (min p-value per NeuronID): "
              f"removed {n_dup_removed} duplicate rows → {len(pkl_df)} unique neurons")
    else:
        print(f"  No duplicate NeuronIDs to deduplicate ({len(pkl_df)} unique neurons)")

    # ── Build the lookup dict, converting ms → s ──
    neuron_windows = {}
    for _, row in pkl_df.iterrows():
        nid = str(row[cfg.pkl_neuron_id_col])
        start_s = float(row[cfg.pkl_window_start_col]) / 1000.0
        end_s   = float(row[cfg.pkl_window_end_col]) / 1000.0
        if start_s >= end_s:
            raise ValueError(
                f"Bad window for {nid}: start {start_s*1000:.0f} ms >= "
                f"end {end_s*1000:.0f} ms")
        neuron_windows[nid] = (start_s, end_s)

    # ── Sanity check against min_epoch_duration ──
    max_end = max(w[1] for w in neuron_windows.values())
    if max_end > cfg.min_epoch_duration:
        print(f"  ⚠  Largest pkl window end ({max_end*1000:.0f} ms) exceeds "
              f"min_epoch_duration ({cfg.min_epoch_duration*1000:.0f} ms). "
              f"Trials shorter than {max_end*1000:.0f} ms will be filtered out "
              f"upstream — consider lowering min_epoch_duration if too few trials survive.")

    # Window-length distribution
    durs_ms = np.array([(w[1] - w[0]) * 1000 for w in neuron_windows.values()])
    print(f"  Window durations: min={durs_ms.min():.0f} ms, "
          f"median={np.median(durs_ms):.0f} ms, max={durs_ms.max():.0f} ms")

    return neuron_windows, pkl_df


# ──────────────────────────────────────────────────────────
# 2. Firing-rate computation with heterogeneous (per-neuron) windows
# ──────────────────────────────────────────────────────────

def compute_firing_rates_pkl(df, identities, neuron_windows, min_reps,
                              window_mode='per_neuron', common_window=None):
    """
    Compute trial-averaged firing rate per neuron per identity, using a
    neuron-specific window if window_mode='per_neuron', else common_window.

    Parameters
    ----------
    df : DataFrame
        Filtered to a single session.  Must contain MonkeyName, NeuronID,
        SpikeTimes, EpochStartStop, TaskField.  Should already have been
        filtered to include only NeuronIDs of interest.
    identities : list of str
        Ordered stimulus identities to include.
    neuron_windows : dict {NeuronID: (start_s, end_s)}
        Used when window_mode='per_neuron'.
    min_reps : int
        Minimum repetitions per identity; identities with fewer are dropped.
    window_mode : str
        'per_neuron' or 'common'.
    common_window : tuple (start_s, end_s) or None
        Required when window_mode='common'.

    Returns
    -------
    rate_matrix : ndarray, shape (n_identities, n_neurons)
    valid_identities : list of str
    neuron_ids : list of str
    """
    neuron_ids = sorted(df['NeuronID'].unique())
    neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    # Per-neuron window (and duration) lookup, indexed by column
    if window_mode == 'per_neuron':
        win_lookup = np.empty((n_neurons, 2), dtype=float)
        for nid, idx in neuron_to_idx.items():
            if nid not in neuron_windows:
                raise KeyError(
                    f"NeuronID {nid!r} not found in neuron_windows — "
                    f"df was not properly filtered before calling "
                    f"compute_firing_rates_pkl")
            win_lookup[idx, 0], win_lookup[idx, 1] = neuron_windows[nid]
    elif window_mode == 'common':
        if common_window is None:
            raise ValueError("window_mode='common' requires common_window")
        win_lookup = np.tile(np.array(common_window, dtype=float), (n_neurons, 1))
    else:
        raise ValueError(f"Unknown window_mode: {window_mode!r}")

    win_durs = win_lookup[:, 1] - win_lookup[:, 0]   # seconds, per column

    # Vectorized per-row spike count using each row's neuron-specific window.
    nid_arr = df['NeuronID'].to_numpy()
    row_neuron_idx = np.array([neuron_to_idx[n] for n in nid_arr])
    epoch_starts = df['EpochStartStop'].map(lambda x: x[0]).to_numpy()
    spike_arrays = df['SpikeTimes'].to_numpy(dtype=object)
    counts = np.empty(len(df), dtype=np.int64)
    for i, (spk, t0, j) in enumerate(zip(spike_arrays, epoch_starts, row_neuron_idx)):
        s = np.asarray(spk) - t0
        counts[i] = np.count_nonzero((s >= win_lookup[j, 0]) & (s < win_lookup[j, 1]))

    counts_df = pd.DataFrame({
        'TaskField': df['TaskField'].to_numpy(),
        'NeuronID':  nid_arr,
        'Count':     counts,
    })

    trial_meta = (df.groupby('TaskField')['MonkeyName'].first().reset_index())

    valid_identities = []
    rate_rows = []

    for monkey in identities:
        trials = trial_meta.loc[trial_meta['MonkeyName'] == monkey,
                                'TaskField'].to_numpy()
        if len(trials) < min_reps:
            continue

        sub = counts_df[counts_df['TaskField'].isin(trials)]
        sum_counts   = np.zeros(n_neurons, dtype=float)
        trial_counts = np.zeros(n_neurons, dtype=float)
        if len(sub):
            agg = sub.groupby('NeuronID')['Count'].agg(['sum', 'count'])
            for nid, row in agg.iterrows():
                j = neuron_to_idx[nid]
                sum_counts[j]   = row['sum']
                trial_counts[j] = row['count']

        with np.errstate(invalid='ignore', divide='ignore'):
            avg_rate = np.where(trial_counts > 0,
                                sum_counts / trial_counts / win_durs,
                                np.nan)
        rate_rows.append(avg_rate)
        valid_identities.append(monkey)

    rate_matrix = np.array(rate_rows)   # (n_identities, n_neurons)
    if rate_matrix.size and np.isnan(rate_matrix).any():
        n_nan = int(np.isnan(rate_matrix).sum())
        print(f"  compute_firing_rates_pkl: {n_nan} (identity, neuron) cells "
              f"had no recorded trials — filling with 0")
        rate_matrix = np.nan_to_num(rate_matrix, nan=0.0)
    return rate_matrix, valid_identities, neuron_ids


# ──────────────────────────────────────────────────────────
# 3. Pooled (pseudo-population) RSA from a pkl whitelist
# ──────────────────────────────────────────────────────────

def run_rsa_pkl_pseudopop(df, info_df, cfg, neuron_windows):
    """
    Pool the selected neurons across sessions into one pseudo-population,
    compute one neural RDM, and run RSA against cfg.model_factors.

    The selection of neurons and their windows is provided in
    ``neuron_windows`` (from load_pkl_windows).

    Parameters
    ----------
    df : DataFrame  (already filtered by region, min_epoch_duration, etc.)
    info_df : DataFrame from monkeyinfo.csv
    cfg : PklRSAConfig or PklSocialRSAConfig
    neuron_windows : dict {NeuronID: (start_s, end_s)}

    Returns
    -------
    result : dict — same shape as rsa_core.run_rsa_pseudopop, with the
        added key ``pkl_window_mode`` for traceability.
    """
    # ── Restrict df to the pkl neurons ──
    requested_ids = set(neuron_windows.keys())
    df_ids = set(df['NeuronID'].unique())
    keep_ids = requested_ids & df_ids
    missing = requested_ids - df_ids

    if missing:
        print(f"  ⚠  {len(missing)} pkl NeuronIDs not found in the loaded data "
              f"(after region/visual/peak filters).  Example missing: "
              f"{sorted(missing)[:3]}")
    if not keep_ids:
        raise ValueError("No pkl NeuronIDs are present in the loaded data.")

    df = df[df['NeuronID'].isin(keep_ids)].reset_index(drop=True)
    print(f"  Pool size after intersecting pkl with df: {len(keep_ids)} neurons")

    # ── Group neurons by session ──
    sessions = sorted(df['session'].unique())
    known = set(info_df['Name'].astype(str))

    # Common identities across sessions (same logic as run_rsa_pseudopop)
    per_session_ids = []
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        monkeys = set(sess_df['MonkeyName'].unique()) & known
        per_session_ids.append(monkeys)
    common_ids = sorted(set.intersection(*per_session_ids)) if per_session_ids else []

    if len(common_ids) < 3:
        raise ValueError(
            f"Only {len(common_ids)} common identities across the "
            f"{len(sessions)} sessions containing pkl neurons.")

    # ── Build rate matrix per session, hstack ──
    session_matrices = []
    all_neuron_ids = []

    for sess in sessions:
        sess_df = df[df['session'] == sess]
        rate_mat, valid_ids, nids = compute_firing_rates_pkl(
            sess_df, common_ids, neuron_windows,
            min_reps=1,
            window_mode=cfg.pkl_window_mode,
            common_window=cfg.window)
        if valid_ids != common_ids:
            print(f"  Warning: session {sess} dropped some common identities")
            continue
        session_matrices.append(rate_mat)
        all_neuron_ids.extend(f"{sess}__{nid}" for nid in nids)

    combined_matrix = np.hstack(session_matrices)  # (n_identities, n_neurons_total)
    print(f"  Pseudo-population matrix: {combined_matrix.shape}")

    # ── Normalize, build RDM, run model comparisons ──
    if cfg.normalization is not None:
        combined_matrix = normalize_rates(
            combined_matrix, method=cfg.normalization,
            soft_const=cfg.soft_normalize_const)

    neural_rdm = build_neural_rdm(combined_matrix, metric=cfg.neural_metric)

    all_factors = list(cfg.model_factors)
    for pf in getattr(cfg, 'partial_out', []):
        if pf not in all_factors:
            all_factors.append(pf)
    model_rdms, model_labels = build_model_rdms(common_ids, info_df, all_factors)

    partial_out = getattr(cfg, 'partial_out', [])
    comparisons = {}
    for factor in cfg.model_factors:
        confound_factors = [p for p in partial_out if p != factor]
        if confound_factors:
            confound_rdms = [model_rdms[p] for p in confound_factors]
            comparisons[factor] = compare_rdms_partial(
                neural_rdm, model_rdms[factor], confound_rdms,
                n_permutations=cfg.n_permutations,
                rng_seed=cfg.rng_seed)
        else:
            comparisons[factor] = compare_rdms(
                neural_rdm, model_rdms[factor],
                n_permutations=cfg.n_permutations,
                rng_seed=cfg.rng_seed)

    return dict(
        session='pseudo_pop_pkl',
        n_neurons=combined_matrix.shape[1],
        n_identities=len(common_ids),
        identities=common_ids,
        neuron_ids=all_neuron_ids,
        rate_matrix=combined_matrix,
        neural_rdm=neural_rdm,
        model_rdms=model_rdms,
        model_labels=model_labels,
        comparisons=comparisons,
        pkl_window_mode=cfg.pkl_window_mode,
        pkl_n_requested=len(requested_ids),
        pkl_n_used=len(keep_ids),
    )
