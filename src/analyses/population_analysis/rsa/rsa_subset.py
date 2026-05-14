# rsa_subset.py
"""
Group-size control for social RSA.

Best Frans has n=5 monkeys (10 pairs). Zombies and Instigators have n=9
(36 pairs). Pair-count imbalance can drive per-group ρ comparisons.

This module provides utilities to downsample larger groups to a target
size (default 5), run RSA on the subsampled identities N times with
different random draws, and aggregate per-group ρ (mean, std, CI).

Typical use:
    from rsa_subset import run_subsampled_dissimilarity_rsa
    summary = run_subsampled_dissimilarity_rsa(
        result, info_df, interactions, cfg,
        target_size=5, n_draws=200, rng_seed=42,
        symmetrize=False, log_transform=False)

`result` is the dict returned by run_rsa_pseudopop / run_rsa_session.
`summary` is {group: {matrix_name: {rho_mean, rho_std, rho_ci_low,
rho_ci_high, draw_rhos}}}.
"""

import numpy as np

from rsa_core import build_neural_rdm
from rsa_social import (
    build_social_rdms,
    compare_neural_to_social_by_group,
    _get_group_indices,
)


def _group_of(identity, info_df):
    info = info_df.set_index(info_df['Name'].astype(str))
    if identity in info.index:
        return info.loc[identity, 'Group Name']
    return None


def balanced_subset_indices(identities, info_df, target_size, rng):
    """
    Pick a subset of indices into `identities` such that every group is
    capped at `target_size`. Groups already at or below the cap are kept
    intact. Returns a sorted np.ndarray of indices.
    """
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
    """Apply index subset to a pseudopop/session result dict."""
    rate_sub = result['rate_matrix'][sub_idx, :]
    ids_sub = [result['identities'][i] for i in sub_idx]
    rdm_sub = build_neural_rdm(rate_sub, metric=cfg.neural_metric)
    return ids_sub, rate_sub, rdm_sub


def run_subsampled_dissimilarity_rsa(result, info_df, interactions, cfg,
                                     target_size=5, n_draws=200, rng_seed=42,
                                     symmetrize=False, log_transform=False,
                                     behavior_types=('affiliation', 'agonism', 'submission'),
                                     confound_matrix_fn=None,
                                     ci_level=0.95):
    """
    Run the dissimilarity-mode social RSA N times, each time downsampling
    every group to `target_size`. Aggregate per-group ρ across draws.

    Parameters
    ----------
    result : dict
        Output of run_rsa_pseudopop or run_rsa_session. Must contain
        'rate_matrix', 'identities'.
    info_df : DataFrame  (monkeyinfo.csv)
    interactions : dict  (output of load_all_interaction_matrices)
    cfg : SocialRSAConfig
    target_size : int
    n_draws : int
    rng_seed : int
    symmetrize, log_transform : bool
    behavior_types : tuple of str
    confound_matrix_fn : callable(identities, info_df) → (n,n) ndarray or None
        E.g. lambda ids, info: build_rank_distance_matrix(ids, info).
        Computed once per draw on the subsampled identities.
    ci_level : float

    Returns
    -------
    summary : dict {group: {matrix_name: {rho_mean, rho_std,
                                          rho_ci_low, rho_ci_high,
                                          n_pairs_typical, draw_rhos}}}
    """
    rng = np.random.default_rng(rng_seed)

    # Per (matrix, group) collect a list of ρ values
    collected = {}  # {mat_name: {group: [rhos]}}

    for draw_i in range(n_draws):
        sub_idx = balanced_subset_indices(
            result['identities'], info_df, target_size, rng)
        ids_sub, _, rdm_sub = _subset_result(result, sub_idx, cfg)

        social_rdms = build_social_rdms(
            ids_sub, interactions,
            behavior_types=list(behavior_types),
            symmetrize=symmetrize, profile_metric='correlation',
            log_transform=log_transform, include_combined=True,
            rank_transform=getattr(cfg, 'rank_transform_behavior', False))

        confound = None
        if confound_matrix_fn is not None:
            confound = confound_matrix_fn(ids_sub, info_df)

        group_comp = compare_neural_to_social_by_group(
            rdm_sub, social_rdms, ids_sub, info_df,
            n_permutations=0, n_bootstrap=0,
            rng_seed=cfg.rng_seed,
            confound_matrix=confound)

        for mat_name, groups in group_comp.items():
            mat_bucket = collected.setdefault(mat_name, {})
            for group, entry in groups.items():
                bucket = mat_bucket.setdefault(group, [])
                bucket.append(entry.get('rho', np.nan))

    # Aggregate
    alpha = 1.0 - ci_level
    summary = {}
    for mat_name, by_group in collected.items():
        for group, rhos in by_group.items():
            arr = np.asarray(rhos, dtype=float)
            valid = arr[~np.isnan(arr)]
            entry = {
                'rho_mean': float(np.mean(valid)) if valid.size else np.nan,
                'rho_std':  float(np.std(valid))  if valid.size else np.nan,
                'rho_ci_low':  float(np.percentile(valid, 100 * alpha / 2)) if valid.size else np.nan,
                'rho_ci_high': float(np.percentile(valid, 100 * (1 - alpha / 2))) if valid.size else np.nan,
                'n_draws_valid': int(valid.size),
                'n_draws_total': int(arr.size),
                'draw_rhos': arr,
            }
            summary.setdefault(group, {})[mat_name] = entry

    return summary


def print_subsample_summary(summary, title='Subsampled RSA (per-group)'):
    """Pretty-print the output of run_subsampled_dissimilarity_rsa."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for group in sorted(summary.keys()):
        print(f"\n  {group}:")
        print(f"    {'Matrix':<28s}{'mean ρ':>10s}{'std':>8s}{'95% CI':>22s}{'n_valid':>10s}")
        print(f"    {'-'*78}")
        for mat in sorted(summary[group].keys()):
            e = summary[group][mat]
            ci = f"[{e['rho_ci_low']:+.3f},{e['rho_ci_high']:+.3f}]"
            print(f"    {mat:<28s}{e['rho_mean']:>+10.4f}{e['rho_std']:>8.3f}"
                  f"{ci:>22s}{e['n_draws_valid']:>10d}")
    print()
