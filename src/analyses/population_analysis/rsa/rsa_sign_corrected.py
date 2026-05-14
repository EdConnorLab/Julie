# rsa_sign_corrected.py
"""
Sign-corrected RSA for opponent neuron populations (e.g. amygdala).

Motivation
----------
If a region contains two subpopulations that code the same axis in
OPPOSITE directions (e.g. some AMG neurons fire MORE for high-submission
focal monkeys, others fire LESS), pooling them into a single population
RDM washes out the signal: opposing neurons contribute opposite-direction
vectors that partially cancel.

Approach: split-half within neuron (avoid circularity)
------------------------------------------------------
For each neuron:
  - Trials per identity are randomly split into two halves (A, B).
  - On half A: regress (firing rate per identity) ~ (submission score per
    identity); record the SIGN of the slope.
  - On half B: compute firing rate per identity; flip sign of neurons
    whose half-A slope was negative.
  - Pool flipped half-B vectors across sessions → pseudo-population matrix
    → neural RDM → compare to social RDM.

Two output modes:
  - 'flip'         : single combined RDM with negative-slope neurons flipped
  - 'subpop'       : two RDMs, one per subpopulation (pos slope, neg slope)
  - 'both'         : returns both for direct comparison

The submission axis is computed from the submission interaction matrix
(column sum per focal monkey within its group → "total submission
received"). Ranked within group so groups are comparable.
"""

import numpy as np
import pandas as pd

from rsa_core import build_neural_rdm, normalize_rates


# ──────────────────────────────────────────────────────────
# Submission axis
# ──────────────────────────────────────────────────────────

def compute_submission_scores(identities, info_df, interactions,
                               behavior_type='submission',
                               aggregate='column_sum', rank_within_group=True):
    """
    Compute a scalar submission score per identity.

    Parameters
    ----------
    identities : list of str
    info_df : DataFrame  (monkeyinfo.csv)
    interactions : dict  (load_all_interaction_matrices output)
    behavior_type : str
        Which interaction matrix to use ('submission' by default).
    aggregate : str
        'column_sum' = total received from groupmates (high = dominant)
        'row_sum'    = total given to groupmates (high = subordinate)
    rank_within_group : bool
        If True, replace raw scores with within-group ranks. Makes
        scores comparable across groups.

    Returns
    -------
    scores : ndarray (n_identities,)
        NaN where identity not found in any group's matrix.
    """
    info = info_df.set_index(info_df['Name'].astype(str))

    scores = np.full(len(identities), np.nan)
    by_group = {}  # group → {'ids': [...], 'idx_in_identities': [...], 'vals': [...]}

    for k, m in enumerate(identities):
        if m not in info.index:
            continue
        group = info.loc[m, 'Group Name']
        if group not in interactions or behavior_type not in interactions[group]:
            continue
        matrix, monkey_ids = interactions[group][behavior_type]
        if m not in monkey_ids:
            continue
        idx = monkey_ids.index(m)

        if aggregate == 'column_sum':
            val = float(np.nansum(matrix[:, idx])) - float(matrix[idx, idx])
        elif aggregate == 'row_sum':
            val = float(np.nansum(matrix[idx, :])) - float(matrix[idx, idx])
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

        bucket = by_group.setdefault(group, {'k': [], 'val': []})
        bucket['k'].append(k)
        bucket['val'].append(val)

    if rank_within_group:
        for group, bucket in by_group.items():
            vals = np.asarray(bucket['val'], dtype=float)
            ranks = np.argsort(np.argsort(vals)).astype(float)
            for k, r in zip(bucket['k'], ranks):
                scores[k] = r
    else:
        for group, bucket in by_group.items():
            for k, v in zip(bucket['k'], bucket['val']):
                scores[k] = v

    return scores


# ──────────────────────────────────────────────────────────
# Split-half firing rates
# ──────────────────────────────────────────────────────────

def compute_firing_rates_split_half(df, identities, window, min_reps_total, rng):
    """
    Compute firing rates per neuron per identity for two random halves
    of the trials. Trials for each identity are split randomly into A/B.

    Parameters
    ----------
    df : DataFrame (already filtered to one session)
    identities : list of str
    window : (start_s, end_s)
    min_reps_total : int
        Identity must have at least this many trials TOTAL to be kept.
        Each half then gets ⌈n/2⌉ and ⌊n/2⌋ trials.
    rng : np.random.Generator

    Returns
    -------
    rate_A : ndarray (n_identities_valid, n_neurons)
    rate_B : ndarray (n_identities_valid, n_neurons)
    valid_identities : list of str
    neuron_ids : list
    """
    win_start, win_end = window
    win_dur = win_end - win_start

    neuron_ids = sorted(df['NeuronID'].unique())
    neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    trial_meta = (df.groupby('TaskField')
                    .first()[['MonkeyName']]
                    .reset_index())

    valid = []
    rows_A = []
    rows_B = []

    for monkey in identities:
        trials = trial_meta[trial_meta['MonkeyName'] == monkey]['TaskField'].values
        if len(trials) < min_reps_total:
            continue

        shuffled = trials.copy()
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        if half < 1:
            continue
        trials_A = shuffled[:half]
        trials_B = shuffled[half:2 * half]  # symmetric halves; drop the odd extra

        def _avg_rate(trial_list):
            sum_counts = np.zeros(n_neurons, dtype=float)
            for tf in trial_list:
                trial_rows = df[df['TaskField'] == tf]
                for _, row in trial_rows.iterrows():
                    n_idx = neuron_to_idx[row['NeuronID']]
                    epoch_start = row['EpochStartStop'][0]
                    spk = row['SpikeTimes'] - epoch_start
                    count = np.sum((spk >= win_start) & (spk < win_end))
                    sum_counts[n_idx] += count
            return (sum_counts / len(trial_list)) / win_dur

        rows_A.append(_avg_rate(trials_A))
        rows_B.append(_avg_rate(trials_B))
        valid.append(monkey)

    return np.array(rows_A), np.array(rows_B), valid, neuron_ids


# ──────────────────────────────────────────────────────────
# Sign estimation
# ──────────────────────────────────────────────────────────

def estimate_neuron_signs(rate_matrix, scores):
    """
    OLS slope sign for each neuron's (firing rate per identity)
    against the scalar `scores` (per identity).

    Parameters
    ----------
    rate_matrix : ndarray (n_identities, n_neurons)
    scores : ndarray (n_identities,)
        May contain NaN; identities with NaN are dropped from the regression.

    Returns
    -------
    signs : ndarray (n_neurons,)  ∈ {+1, -1}
        Neurons with zero (or undefined) slope get +1 (no flip).
    slopes : ndarray (n_neurons,)
        Raw OLS slopes (for diagnostics).
    """
    mask = ~np.isnan(scores)
    if mask.sum() < 3:
        n_neurons = rate_matrix.shape[1]
        return np.ones(n_neurons), np.full(n_neurons, np.nan)

    x = scores[mask]
    R = rate_matrix[mask, :]  # (n_valid, n_neurons)
    x_centered = x - x.mean()
    var_x = float(np.sum(x_centered ** 2))
    if var_x < 1e-12:
        n_neurons = rate_matrix.shape[1]
        return np.ones(n_neurons), np.zeros(n_neurons)

    R_centered = R - R.mean(axis=0, keepdims=True)
    slopes = (x_centered[:, None] * R_centered).sum(axis=0) / var_x
    signs = np.where(slopes < 0, -1.0, 1.0)
    return signs, slopes


# ──────────────────────────────────────────────────────────
# Per-session split-half pipeline
# ──────────────────────────────────────────────────────────

def _session_split_half(sess_df, common_ids, scores_for_common, cfg, rng):
    """
    For one session:
      - split trials A/B for each identity
      - estimate per-neuron sign on A
      - return half-B rate matrix, signs, slopes, neuron_ids
        (rows aligned to common_ids; identities lacking enough trials → NaN row)

    Returns
    -------
    rate_B_aligned : ndarray (len(common_ids), n_neurons)
        NaN row where identity missing.
    signs : ndarray (n_neurons,)
    slopes : ndarray (n_neurons,)
    neuron_ids : list
    """
    rate_A, rate_B, valid_ids, neuron_ids = compute_firing_rates_split_half(
        sess_df, common_ids, cfg.window, cfg.min_reps_per_monkey, rng)

    n_neurons = len(neuron_ids)

    if len(valid_ids) < 3:
        rate_B_aligned = np.full((len(common_ids), n_neurons), np.nan)
        return rate_B_aligned, np.ones(n_neurons), np.full(n_neurons, np.nan), neuron_ids

    # Subset submission scores to valid_ids order
    valid_idx_in_common = [common_ids.index(m) for m in valid_ids]
    scores_valid = scores_for_common[valid_idx_in_common]

    signs, slopes = estimate_neuron_signs(rate_A, scores_valid)

    # Align rate_B back to common_ids (NaN for missing identities)
    rate_B_aligned = np.full((len(common_ids), n_neurons), np.nan)
    for k, m in enumerate(valid_ids):
        rate_B_aligned[common_ids.index(m), :] = rate_B[k, :]

    return rate_B_aligned, signs, slopes, neuron_ids


# ──────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────

def run_sign_corrected_pseudopop(df, info_df, interactions, cfg,
                                  mode='flip', sign_axis='submission',
                                  axis_aggregate='column_sum',
                                  rng_seed=42):
    """
    Pseudo-population RSA with sign correction.

    Parameters
    ----------
    df : DataFrame  (already filtered by region, min epoch, etc.)
    info_df : DataFrame
    interactions : dict
    cfg : RSAConfig
    mode : str
        'flip'   = single combined RDM, negative-slope neurons flipped
        'subpop' = two separate RDMs (pos slope, neg slope)
        'both'   = returns flip-combined AND subpop RDMs
    sign_axis : str
        Behavior type to define the preference axis ('submission' default).
    axis_aggregate : str
        'column_sum' (received, default) or 'row_sum' (given).
    rng_seed : int

    Returns
    -------
    result : dict with keys
        'identities'   : list of str
        'scores'       : ndarray (n_identities,) — axis values used
        'signs'        : ndarray (n_neurons_total,) — +1/-1 per pooled neuron
        'slopes'       : ndarray (n_neurons_total,)
        'rate_matrix_flipped' : ndarray (n_identities, n_neurons) or None
        'rate_matrix_pos'     : ndarray or None
        'rate_matrix_neg'     : ndarray or None
        'neural_rdm_flipped'  : ndarray (n,n) or None
        'neural_rdm_pos'      : ndarray or None
        'neural_rdm_neg'      : ndarray or None
        'n_neurons_pos', 'n_neurons_neg' : int
        'session_neurons'     : list of (session, neuron_id) for traceability
    """
    rng = np.random.default_rng(rng_seed)

    sessions = sorted(df['session'].unique())
    known = set(info_df['Name'].astype(str))

    # Find identities present in all sessions (pseudopop convention)
    per_session_ids = []
    for sess in sessions:
        sess_df = df[df['session'] == sess]
        per_session_ids.append(set(sess_df['MonkeyName'].unique()) & known)
    common_ids = sorted(set.intersection(*per_session_ids))

    # Identity-coverage diagnostic (helps explain pair-count shortfalls)
    union_ids = sorted(set().union(*per_session_ids))
    dropped = sorted(set(union_ids) - set(common_ids))
    info_lookup = info_df.set_index(info_df['Name'].astype(str))
    print(f"\n  Identity coverage across {len(sessions)} sessions:")
    print(f"    union of identities  : {len(union_ids)}")
    print(f"    common to all sessions: {len(common_ids)}")
    if dropped:
        print(f"    dropped (missing in ≥1 session):")
        for m in dropped:
            grp = info_lookup.loc[m, 'Group Name'] if m in info_lookup.index else '?'
            missing_in = [s for s, ids in zip(sessions, per_session_ids) if m not in ids]
            print(f"      {m} ({grp}) — missing in {len(missing_in)} session(s): "
                  f"{missing_in[:5]}{'…' if len(missing_in) > 5 else ''}")
    # Per-group surviving counts
    from collections import Counter
    surviving_groups = Counter(
        info_lookup.loc[m, 'Group Name'] for m in common_ids if m in info_lookup.index)
    print(f"    per-group surviving counts: {dict(surviving_groups)}")

    if len(common_ids) < 3:
        raise ValueError(f"Only {len(common_ids)} common identities; cannot run RSA")

    # Submission scores aligned to common_ids
    scores = compute_submission_scores(
        common_ids, info_df, interactions,
        behavior_type=sign_axis, aggregate=axis_aggregate,
        rank_within_group=True)

    # Per-session split-half
    all_rate_B = []      # list of (n_common, n_neurons_session)
    all_signs = []
    all_slopes = []
    session_neurons = []

    for sess in sessions:
        sess_df = df[df['session'] == sess]
        rate_B_aligned, signs, slopes, neuron_ids = _session_split_half(
            sess_df, common_ids, scores, cfg, rng)
        all_rate_B.append(rate_B_aligned)
        all_signs.append(signs)
        all_slopes.append(slopes)
        session_neurons.extend([(sess, nid) for nid in neuron_ids])

    # Drop sessions/neurons where rate_B is fully NaN (no valid identities)
    pooled_B = np.hstack(all_rate_B)           # (n_common, n_neurons_total)
    pooled_signs = np.concatenate(all_signs)
    pooled_slopes = np.concatenate(all_slopes)

    # Drop neurons that have NaN for any identity (incomplete coverage)
    keep_neuron = ~np.any(np.isnan(pooled_B), axis=0)
    pooled_B = pooled_B[:, keep_neuron]
    pooled_signs = pooled_signs[keep_neuron]
    pooled_slopes = pooled_slopes[keep_neuron]
    session_neurons = [sn for sn, k in zip(session_neurons, keep_neuron) if k]

    print(f"  Sign-corrected pseudopop: {pooled_B.shape[1]} neurons "
          f"({int((pooled_signs > 0).sum())} pos, "
          f"{int((pooled_signs < 0).sum())} neg)")

    # Optional normalization
    if cfg.normalization is not None:
        pooled_B = normalize_rates(pooled_B, method=cfg.normalization,
                                    soft_const=getattr(cfg, 'soft_normalize_const', 5.0))

    result = {
        'identities': common_ids,
        'scores': scores,
        'signs': pooled_signs,
        'slopes': pooled_slopes,
        'session_neurons': session_neurons,
        'n_neurons_pos': int((pooled_signs > 0).sum()),
        'n_neurons_neg': int((pooled_signs < 0).sum()),
        'rate_matrix_flipped': None,
        'rate_matrix_pos': None,
        'rate_matrix_neg': None,
        'neural_rdm_flipped': None,
        'neural_rdm_pos': None,
        'neural_rdm_neg': None,
    }

    if mode in ('flip', 'both'):
        flipped = pooled_B * pooled_signs[None, :]
        result['rate_matrix_flipped'] = flipped
        result['neural_rdm_flipped'] = build_neural_rdm(
            flipped, metric=cfg.neural_metric)

    if mode in ('subpop', 'both'):
        pos_mask = pooled_signs > 0
        neg_mask = pooled_signs < 0
        if pos_mask.sum() >= 2:
            result['rate_matrix_pos'] = pooled_B[:, pos_mask]
            result['neural_rdm_pos'] = build_neural_rdm(
                result['rate_matrix_pos'], metric=cfg.neural_metric)
        if neg_mask.sum() >= 2:
            result['rate_matrix_neg'] = pooled_B[:, neg_mask]
            result['neural_rdm_neg'] = build_neural_rdm(
                result['rate_matrix_neg'], metric=cfg.neural_metric)

    # Match keys used by run_rsa_pseudopop callers
    result['session'] = 'pseudo_pop_sign_corrected'
    result['n_neurons'] = pooled_B.shape[1]
    result['n_identities'] = len(common_ids)
    result['rate_matrix'] = result['rate_matrix_flipped'] if result['rate_matrix_flipped'] is not None else pooled_B
    result['neural_rdm'] = result['neural_rdm_flipped']

    return result


# ──────────────────────────────────────────────────────────
# Convenience: run multiple split-half draws and average
# ──────────────────────────────────────────────────────────

def run_sign_corrected_multi_draw(df, info_df, interactions, cfg,
                                   mode='flip', sign_axis='submission',
                                   axis_aggregate='column_sum',
                                   n_draws=10, rng_seed=42):
    """
    Run `run_sign_corrected_pseudopop` n_draws times with different
    trial-split seeds and average the resulting RDM(s). Reduces
    variance from the random A/B split.

    Returns the same dict structure as run_sign_corrected_pseudopop,
    but rate_matrix_* and neural_rdm_* are averaged across draws.
    """
    rng = np.random.default_rng(rng_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_draws)

    accum = None
    for d, s in enumerate(seeds):
        res = run_sign_corrected_pseudopop(
            df, info_df, interactions, cfg,
            mode=mode, sign_axis=sign_axis,
            axis_aggregate=axis_aggregate, rng_seed=int(s))
        if accum is None:
            accum = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in res.items()}
            accum['_count_flipped'] = 0
            accum['_count_pos'] = 0
            accum['_count_neg'] = 0

        for key, cnt_key in [('neural_rdm_flipped', '_count_flipped'),
                              ('neural_rdm_pos',     '_count_pos'),
                              ('neural_rdm_neg',     '_count_neg')]:
            if res.get(key) is not None:
                if accum.get(key) is None or accum[cnt_key] == 0:
                    accum[key] = res[key].copy()
                else:
                    accum[key] = accum[key] + res[key]
                accum[cnt_key] += 1

    for key, cnt_key in [('neural_rdm_flipped', '_count_flipped'),
                          ('neural_rdm_pos',     '_count_pos'),
                          ('neural_rdm_neg',     '_count_neg')]:
        if accum.get(cnt_key, 0) > 0:
            accum[key] = accum[key] / accum[cnt_key]

    for k in ('_count_flipped', '_count_pos', '_count_neg'):
        accum.pop(k, None)

    accum['neural_rdm'] = accum['neural_rdm_flipped']
    return accum
