# run_rsa_extensions.py
"""
Runner for the two general-purpose RSA extensions:

  Block 1 — Subset (rsa_subset.py)
      Cap every group at n=5 monkeys, run the dissimilarity RSA N times
      with random subsamples of the larger groups (Zombies, Instigators).
      Each draw runs an inner permutation test; per-group and between-
      group statistics are aggregated across draws with p-values
      derived from pooled nulls.

  Block 2 — Raw social (rsa_raw_social.py)
      Build pair-level signed-asymmetry RDMs and per-monkey
      concat[row, col] directed-vector RDMs (no symmetrization).
      Compare to the standard pseudo-pop neural RDM.

The sign-corrected analysis lives in `run_rsa_signcorrect.py` since
it is AMG-specific and needs different defaults (no rank partialing,
different region, different identity-coverage handling).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from rsa_config import SocialRSAConfig
from rsa_core import run_rsa_pseudopop
from rsa_social import (
    load_all_interaction_matrices,
    compare_neural_to_social_by_group,
    build_rank_distance_matrix,
    print_group_comparisons,
)
from rsa_subset import (
    run_subsampled_dissimilarity_rsa,
    print_subsample_summary,
    print_subsample_between_groups,
    summary_to_json,
)
from rsa_raw_social import build_raw_social_rdms


# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"

BEHAVIOR_FILES = {
    'Zombies': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_feature_df_submission.xlsx',
    },
    'Best Frans': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/bestfrans_social_data/bestfrans_feature_df_submission.xlsx',
    },
    'Instigators': {
        'affiliation': '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_affiliation.xlsx',
        'agonism':     '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_agonism.xlsx',
        'submission':  '/home/connorlab/Documents/GitHub/Julie/social_data/instigators_social_data/instigators_feature_df_submission.xlsx',
    },
}


def _serialize_group_comp(group_comp):
    out = {}
    for mat_name, groups in group_comp.items():
        out[mat_name] = {}
        for grp, entry in groups.items():
            clean = {}
            for k, v in entry.items():
                if k == 'null_distribution':
                    continue
                if isinstance(v, (np.floating, float)):
                    clean[k] = None if np.isnan(v) else float(v)
                elif isinstance(v, (np.integer, int)):
                    clean[k] = int(v)
                else:
                    clean[k] = v
            out[mat_name][grp] = clean
    return out


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='RSA extensions runner (subset + raw social)')
    parser.add_argument('--region', type=str, default=None,
                        help='Brain region (AMG, ER, ALL). Overrides cfg.')
    parser.add_argument('--skip-subset', action='store_true')
    parser.add_argument('--skip-raw',    action='store_true')
    parser.add_argument('--n-draws', type=int, default=200,
                        help='Number of subsample draws in Block 1.')
    parser.add_argument('--perms-per-draw', type=int, default=200,
                        help='Inner permutations per draw in Block 1.')
    parser.add_argument('--target-size', type=int, default=5,
                        help='Per-group cap for Block 1.')
    args, _ = parser.parse_known_args()

    RUN_SUBSET = not args.skip_subset
    RUN_RAW    = not args.skip_raw

    cfg = SocialRSAConfig(
        region='ALL',
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=[],
        exclude_groups=['Stranger Things'],
        normalization=None,
        partial_out_rank=True,
        rank_transform_behavior=False,
        n_permutations=1000,
        between_group_permutations=0,
        n_bootstrap=0,
        pseudo_population=True,
        save_plots=True,
        save_dir='rsa_extensions_results',
    )
    if args.region:
        cfg.region = args.region
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    print("\n--- Building standard pseudo-population result ---")
    pp_result = run_rsa_pseudopop(df, info_df, cfg)

    save_root = f"{cfg.save_dir}/{cfg.region}"
    os.makedirs(save_root, exist_ok=True)
    all_json = {
        'region': cfg.region,
        'config': {
            'window': list(cfg.window),
            'neural_metric': cfg.neural_metric,
            'normalization': cfg.normalization,
            'partial_out_rank': cfg.partial_out_rank,
        },
    }

    # ═══════════════════════════════════════════════════════
    # BLOCK 1 — SUBSET (group-size control + significance)
    # ═══════════════════════════════════════════════════════
    if RUN_SUBSET:
        print(f"\n{'█'*70}")
        print(f"█  BLOCK 1: SUBSET  —  cap groups at n={args.target_size}, "
              f"{args.n_draws} draws, {args.perms_per_draw} perms/draw")
        print(f"{'█'*70}")

        confound_fn = None
        if cfg.partial_out_rank:
            confound_fn = lambda ids, info: build_rank_distance_matrix(ids, info)

        all_json['subset'] = {}
        for symmetrize, label in [(False, 'asymmetric'), (True, 'symmetrized')]:
            print(f"\n  --- Condition: {label} ---")
            summary = run_subsampled_dissimilarity_rsa(
                pp_result, info_df, interactions, cfg,
                target_size=args.target_size,
                n_draws=args.n_draws,
                n_permutations_per_draw=args.perms_per_draw,
                rng_seed=cfg.rng_seed,
                symmetrize=symmetrize, log_transform=False,
                behavior_types=('affiliation', 'agonism', 'submission'),
                confound_matrix_fn=confound_fn,
                ci_level=0.95)

            print_subsample_summary(
                summary, title=f"Per-group ρ  (subsampled, {label})")
            print_subsample_between_groups(
                summary, title=f"Between-group Δρ  (subsampled, {label})")

            all_json['subset'][label] = summary_to_json(summary)

    # ═══════════════════════════════════════════════════════
    # BLOCK 2 — RAW SOCIAL (no symmetrization)
    # ═══════════════════════════════════════════════════════
    if RUN_RAW:
        print(f"\n{'█'*70}")
        print(f"█  BLOCK 2: RAW SOCIAL  —  signed asymmetry + concat[row,col]")
        print(f"{'█'*70}")

        identities = pp_result['identities']
        neural_rdm = pp_result['neural_rdm']

        raw_rdms = build_raw_social_rdms(
            identities, interactions,
            behavior_types=('affiliation', 'agonism', 'submission'),
            log_transform=False,
            rank_transform=cfg.rank_transform_behavior,
            concat_metric='correlation',
            include_combined=True)

        rank_confound = None
        if cfg.partial_out_rank:
            rank_confound = build_rank_distance_matrix(identities, info_df)
            print("  Partial out rank: ON")

        print(f"\n  Per-group RSA (neural 1-r vs raw social RDMs):")
        group_comp = compare_neural_to_social_by_group(
            neural_rdm, raw_rdms, identities, info_df,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
            rng_seed=cfg.rng_seed,
            confound_matrix=rank_confound)
        print_group_comparisons(group_comp)

        all_json['raw_social'] = _serialize_group_comp(group_comp)

    json_path = f"{save_root}/extensions_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_json, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    if cfg.save_plots:
        plt.show()


if __name__ == '__main__':
    main()
