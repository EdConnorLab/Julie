# rsa_subset.py
"""
Group-size control for social RSA with significance testing.

Best Frans has n=5 monkeys (10 pairs). Zombies and Instigators have n=9
(36 pairs). Pair-count imbalance can drive per-group ρ comparisons.

This module:
  1. Caps every group at `target_size` (default 5) and runs the
     dissimilarity-mode social RSA `n_draws` times with different
     random subsamples of the larger groups.
  2. Returns aggregated per-group ρ statistics AND between-group Δρ
     statistics — both with p-values derived from permutation nulls
     pooled across draws.

Significance approach
---------------------
Each draw d performs M neural-RDM-shuffle permutations, yielding
  obs ρ_d        and  null ρ_{d, m}   for m = 1..M
The statistic we report is ρ_mean = mean_d ρ_d. Its null distribution
is built as
  null_mean_m = mean_d null ρ_{d, m}
which is a valid null for the mean statistic because each draw's
permutations are independent.

p-value (two-tailed) = mean( |null_mean| ≥ |ρ_mean| )

Same logic for between-group Δρ.
"""

import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

from rsa_core import build_neural_rdm
from rsa_social import build_social_rdms, _get_group_indices
from rsa_utils import (
    upper_triangle as _upper_triangle,
    partial_spearman as _partial_spearman,
    finite_mask,
    stars as _stars,
)


# ──────────────────────────────────────────────────────────
# Subsampling
# ──────────────────────────────────────────────────────────

def balanced_subset_indices(identities, info_df, target_size, rng):
    """Random index subset that caps every group at `target_size`."""
    group_indices = _get_group_indices(identities, info_df)
    keep = []
    for group, idxs in group_indices.items():
        if len(idxs) <= target_size:
            keep.extend(idxs)
        else:
            chosen = rng.choice(idxs, size=target_size, replace=False)
            keep.extend(chosen.tolist())
    return np.sort(np.asarray(keep, dtype=int))


def _subset_result(result, sub_idx, cfg):
    rate_sub = result['rate_matrix'][sub_idx, :]
    ids_sub = [result['identities'][i] for i in sub_idx]
    rdm_sub = build_neural_rdm(rate_sub, metric=cfg.neural_metric)
    return ids_sub, rate_sub, rdm_sub


# ──────────────────────────────────────────────────────────
# Single-draw stats (observed + null arrays)
# ──────────────────────────────────────────────────────────

def _rho_or_partial(v_neural, v_social, v_confound, mask):
    if mask.sum() < (4 if v_confound is not None else 3):
        return np.nan
    if v_confound is not None:
        return _partial_spearman(v_neural, v_social, v_confound, mask)
    r, _ = spearmanr(v_neural[mask], v_social[mask])
    return r


def _draw_stats(neural_rdm, social_rdms, identities, info_df,
                confound_matrix, n_permutations, rng):
    """
    Compute observed and per-permutation null statistics for ONE draw.

    Returns
    -------
    per_group : dict {mat_name: {group: {'rho': float, 'null': ndarray(M,),
                                          'n_pairs': int}}}
    between : dict {mat_name: {(g_a, g_b): {'delta': float,
                                             'null': ndarray(M,),
                                             'rho_a': float, 'rho_b': float}}}
    """
    group_indices = _get_group_indices(identities, info_df)
    groups = sorted(group_indices.keys())
    n = neural_rdm.shape[0]

    # Pre-permute neural RDMs once per perm index, reused for all
    # (matrix, group) combinations within the draw.
    perm_rdms = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_rdms.append(neural_rdm[np.ix_(perm, perm)])

    per_group = {}
    between = {}

    for mat_name, social in social_rdms.items():
        per_group[mat_name] = {}
        between[mat_name] = {}

        # ── per-group ─────────────────────────
        group_obs = {}        # group → rho (or nan)
        group_null = {}       # group → ndarray(M,) (nan-filled)
        group_pairs = {}      # group → (v_neural_sub, v_social_sub, v_conf_sub, mask)
        for g, idxs in group_indices.items():
            idx = np.asarray(idxs)
            if len(idx) < 3:
                group_obs[g] = np.nan
                group_null[g] = np.full(n_permutations, np.nan)
                per_group[mat_name][g] = {
                    'rho': np.nan, 'null': group_null[g], 'n_pairs': 0}
                continue

            n_sub = neural_rdm[np.ix_(idx, idx)]
            s_sub = social[np.ix_(idx, idx)]
            v_n = _upper_triangle(n_sub)
            v_s = _upper_triangle(s_sub)
            v_c = (_upper_triangle(confound_matrix[np.ix_(idx, idx)])
                   if confound_matrix is not None else None)
            mask = (finite_mask(v_n, v_s, v_c) if v_c is not None
                    else finite_mask(v_n, v_s))

            min_required = 4 if v_c is not None else 3
            if mask.sum() < min_required:
                group_obs[g] = np.nan
                group_null[g] = np.full(n_permutations, np.nan)
                per_group[mat_name][g] = {
                    'rho': np.nan, 'null': group_null[g],
                    'n_pairs': int(mask.sum())}
                continue

            rho = _rho_or_partial(v_n, v_s, v_c, mask)
            null = np.full(n_permutations, np.nan)
            for m_i, p_rdm in enumerate(perm_rdms):
                v_np = _upper_triangle(p_rdm[np.ix_(idx, idx)])
                m_perm = (finite_mask(v_np, v_s, v_c) if v_c is not None
                          else finite_mask(v_np, v_s))
                null[m_i] = _rho_or_partial(v_np, v_s, v_c, m_perm)

            group_obs[g] = rho
            group_null[g] = null
            group_pairs[g] = (idx, mask)
            per_group[mat_name][g] = {
                'rho': float(rho), 'null': null,
                'n_pairs': int(mask.sum())}

        # ── between-group Δρ ───────────────────
        for g_a, g_b in combinations(groups, 2):
            ra = group_obs.get(g_a, np.nan)
            rb = group_obs.get(g_b, np.nan)
            if np.isnan(ra) or np.isnan(rb):
                between[mat_name][(g_a, g_b)] = {
                    'delta': np.nan, 'rho_a': ra, 'rho_b': rb,
                    'null': np.full(n_permutations, np.nan)}
                continue
            null_a = group_null[g_a]
            null_b = group_null[g_b]
            between[mat_name][(g_a, g_b)] = {
                'delta': float(ra - rb),
                'rho_a': float(ra), 'rho_b': float(rb),
                'null': null_a - null_b,
            }

    return per_group, between


# ──────────────────────────────────────────────────────────
# Aggregation across draws
# ──────────────────────────────────────────────────────────
def _collect_nodes(per_draw, key_path):
    """Walk each draw's nested dict via key_path; return list of {obs, null} dicts."""
    out = []
    for d in per_draw:
        node = d
        for k in key_path:
            node = node.get(k, None) if node is not None else None
            if node is None:
                break
        if node is not None:
            out.append(node)
    return out


def _aggregate(per_draw, key_path, n_permutations, ci_level=0.95):
    """
    Aggregate one statistic across draws using the *pooled-null* approach:

        ρ_mean    = mean_d obs_d           (the reported statistic)
        null_mean[m] = mean_d null_{d,m}   (built once, valid null for ρ_mean)
        p_val (two-tailed) = mean( |null_mean| ≥ |ρ_mean| )

    Each draw's permutations are independent (different random subsamples +
    different perms), so the per-perm-index mean is itself a draw from the
    null distribution of ρ_mean.
    """
    nodes = _collect_nodes(per_draw, key_path)
    obs_vals = np.array([n['obs'] for n in nodes if not np.isnan(n['obs'])],
                        dtype=float)
    null_arrays = [n['null'] for n in nodes
                   if not np.isnan(n['obs']) and n['null'] is not None]

    if len(obs_vals) == 0:
        return {'mean': np.nan, 'std': np.nan,
                'ci_low': np.nan, 'ci_high': np.nan,
                'p_val': None, 'n_valid': 0,
                'n_draws_total': len(per_draw)}

    rho_mean = float(np.mean(obs_vals))
    alpha = 1.0 - ci_level

    # Pooled null: average across draws at each permutation index.
    p_val = None
    if null_arrays:
        null_stack = np.vstack(null_arrays)            # (n_draws, n_perms)
        # nanmean down columns: each col is a permutation index across draws
        null_mean = np.nanmean(null_stack, axis=0)
        valid = ~np.isnan(null_mean)
        if valid.any():
            p_val = float(np.mean(np.abs(null_mean[valid]) >= np.abs(rho_mean)))

    return {
        'mean':           rho_mean,
        'std':            float(np.std(obs_vals)),
        'ci_low':         float(np.percentile(obs_vals, 100 * alpha / 2)),
        'ci_high':        float(np.percentile(obs_vals, 100 * (1 - alpha / 2))),
        'p_val':          p_val,
        'n_valid':        int(len(obs_vals)),
        'n_draws_total':  int(len(per_draw)),
    }


# ──────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────

def run_subsampled_dissimilarity_rsa(result, info_df, interactions, cfg,
                                     target_size=5, n_draws=200,
                                     n_permutations_per_draw=200,
                                     rng_seed=42,
                                     symmetrize=False, log_transform=False,
                                     behavior_types=('affiliation', 'agonism', 'submission'),
                                     confound_matrix_fn=None,
                                     ci_level=0.95):
    """
    Run dissimilarity-mode social RSA `n_draws` times on subsampled
    identities (every group capped at `target_size`). Each draw runs
    `n_permutations_per_draw` neural-RDM shuffles to build a null
    distribution. Aggregate across draws for per-group and between-
    group statistics.

    Returns
    -------
    summary : dict
        {
          'per_group':   {mat_name: {group: {mean, std, ci_low, ci_high,
                                              p_val, n_valid, n_draws_total}}},
          'between':     {mat_name: {(g_a, g_b): {mean_delta, std_delta,
                                                   ci_low, ci_high, p_val,
                                                   mean_rho_a, mean_rho_b,
                                                   n_valid, n_draws_total}}},
          'meta': {target_size, n_draws, n_permutations_per_draw, ...},
        }
    """
    rng = np.random.default_rng(rng_seed)
    perm_rng = np.random.default_rng(rng_seed + 17)

    draw_records = []  # list of (per_group_dict, between_dict)

    for draw_i in range(n_draws):
        sub_idx = balanced_subset_indices(
            result['identities'], info_df, target_size, rng)
        ids_sub, _, rdm_sub = _subset_result(result, sub_idx, cfg)

        social_rdms = build_social_rdms(
            ids_sub, interactions,
            behavior_types=list(behavior_types),
            symmetrize=symmetrize, profile_metric='correlation',
            log_transform=log_transform, include_combined=True,
            rank_transform=(getattr(cfg, 'transform_social_behavior', None) == 'rank'))

        confound = None
        if confound_matrix_fn is not None:
            confound = confound_matrix_fn(ids_sub, info_df)

        per_group, between = _draw_stats(
            rdm_sub, social_rdms, ids_sub, info_df,
            confound, n_permutations_per_draw, perm_rng)

        # Reshape into nested 'obs'/'null' form for _aggregate
        pg_packed = {}
        for mat, by_grp in per_group.items():
            pg_packed[mat] = {}
            for grp, entry in by_grp.items():
                pg_packed[mat][grp] = {'obs': entry['rho'], 'null': entry['null']}

        bw_packed = {}
        for mat, by_pair in between.items():
            bw_packed[mat] = {}
            for pair, entry in by_pair.items():
                bw_packed[mat][pair] = {
                    'obs': entry['delta'], 'null': entry['null'],
                    'rho_a': entry['rho_a'], 'rho_b': entry['rho_b']}

        draw_records.append((pg_packed, bw_packed))

    # ── aggregate per-group ──
    per_group_summary = {}
    if draw_records:
        first_pg = draw_records[0][0]
        for mat_name, by_grp in first_pg.items():
            per_group_summary[mat_name] = {}
            for grp in by_grp.keys():
                draws_pg = [r[0] for r in draw_records]
                per_group_summary[mat_name][grp] = _aggregate(
                    draws_pg, [mat_name, grp],
                    n_permutations_per_draw, ci_level)

    # ── aggregate between-group ──
    between_summary = {}
    if draw_records:
        first_bw = draw_records[0][1]
        for mat_name, by_pair in first_bw.items():
            between_summary[mat_name] = {}
            for pair in by_pair.keys():
                # Aggregate Δρ
                draws_bw = [r[1] for r in draw_records]
                base = _aggregate(
                    draws_bw, [mat_name, pair],
                    n_permutations_per_draw, ci_level)
                # Also record mean ρ per side, for context
                rho_a_vals = [r[1][mat_name][pair]['rho_a'] for r in draw_records
                              if mat_name in r[1] and pair in r[1][mat_name]]
                rho_b_vals = [r[1][mat_name][pair]['rho_b'] for r in draw_records
                              if mat_name in r[1] and pair in r[1][mat_name]]
                rho_a_arr = np.asarray(rho_a_vals, dtype=float)
                rho_b_arr = np.asarray(rho_b_vals, dtype=float)
                base['mean_rho_a'] = float(np.nanmean(rho_a_arr))
                base['mean_rho_b'] = float(np.nanmean(rho_b_arr))
                between_summary[mat_name][pair] = base

    return {
        'per_group': per_group_summary,
        'between':   between_summary,
        'meta': {
            'target_size': target_size,
            'n_draws': n_draws,
            'n_permutations_per_draw': n_permutations_per_draw,
            'ci_level': ci_level,
            'symmetrize': symmetrize,
            'log_transform': log_transform,
        },
    }


# ──────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────

def print_subsample_summary(summary, title='Subsampled RSA — per-group'):
    pg = summary['per_group']
    meta = summary['meta']
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"  target_size={meta['target_size']}, n_draws={meta['n_draws']}, "
          f"perms/draw={meta['n_permutations_per_draw']}")
    print(f"{'='*78}")

    # Get groups
    all_groups = set()
    for by_grp in pg.values():
        all_groups.update(by_grp.keys())
    all_groups = sorted(all_groups)

    print(f"\n  {'Matrix':<28s}", end='')
    for g in all_groups:
        print(f"{g:>26s}", end='')
    print()
    print(f"  {'-'*28}{'-'*(26*len(all_groups))}")

    for mat_name in sorted(pg.keys()):
        row = f"  {mat_name:<28s}"
        for g in all_groups:
            e = pg[mat_name].get(g, {})
            if e.get('mean') is None or np.isnan(e.get('mean', np.nan)):
                row += f"{'--':>26s}"
            else:
                p = e.get('p_val')
                star = _stars(p)
                p_str = f"p={p:.3f}" if p is not None else "p=  -  "
                row += f"  {e['mean']:>+.3f}±{e['std']:.3f} {p_str}{star:<3s}"
        print(row)
    print()


def print_subsample_between_groups(summary,
                                    title='Subsampled RSA — between-group Δρ'):
    bw = summary['between']
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")

    all_pairs = set()
    for by_pair in bw.values():
        all_pairs.update(by_pair.keys())
    all_pairs = sorted(all_pairs)

    for pair in all_pairs:
        g_a, g_b = pair
        print(f"\n  {g_a}  vs  {g_b}:")
        print(f"    {'Matrix':<28s}{'mean ρ_a':>12s}{'mean ρ_b':>12s}"
              f"{'Δρ':>10s}{'95% CI':>22s}{'p':>10s}")
        print(f"    {'-'*94}")
        for mat_name in sorted(bw.keys()):
            e = bw[mat_name].get(pair, {})
            if np.isnan(e.get('mean', np.nan)):
                print(f"    {mat_name:<28s}{'--':>56s}")
                continue
            p = e.get('p_val')
            star = _stars(p)
            ci = f"[{e['ci_low']:+.3f},{e['ci_high']:+.3f}]"
            p_str = f"{p:.3f}" if p is not None else "  -  "
            print(f"    {mat_name:<28s}"
                  f"{e['mean_rho_a']:>+12.3f}"
                  f"{e['mean_rho_b']:>+12.3f}"
                  f"{e['mean']:>+10.3f}"
                  f"{ci:>22s}"
                  f"   {p_str}{star}")
    print()


def summary_to_json(summary):
    """Strip ndarrays etc. to make summary JSON-serializable."""
    def _clean(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (np.floating, float)):
                out[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            elif isinstance(v, dict):
                out[k] = _clean(v)
            else:
                out[k] = v
        return out

    out = {'meta': summary['meta'], 'per_group': {}, 'between': {}}
    for mat, by_grp in summary['per_group'].items():
        out['per_group'][mat] = {g: _clean(e) for g, e in by_grp.items()}
    for mat, by_pair in summary['between'].items():
        out['between'][mat] = {
            f"{a}__vs__{b}": _clean(e) for (a, b), e in by_pair.items()}
    return out
