# run_rsa_subset.py
"""
Subset / group-size-control RSA runner.

Caps every group at `target_size` monkeys (default 5 — matches Best
Frans) and runs the dissimilarity-mode social RSA `n_draws` times with
random subsamples of the larger groups (Zombies, Instigators). Each
draw runs an inner permutation test; per-group ρ and between-group Δρ
are aggregated across draws with p-values derived from pooled nulls.

Run:
    python run_rsa_subset.py                                 # defaults
    python run_rsa_subset.py --region AMG --no-partial-rank
    python run_rsa_subset.py --n-draws 500 --perms-per-draw 500
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
    build_rank_distance_matrix,
)
from rsa_subset import (
    run_subsampled_dissimilarity_rsa,
    print_subsample_summary,
    print_subsample_between_groups,
    summary_to_json,
)


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


def main():
    parser = argparse.ArgumentParser(description='Subset / group-size RSA')
    parser.add_argument('--region', type=str, default='ALL')
    parser.add_argument('--target-size', type=int, default=5,
                        help='Per-group cap (default: 5, matches Best Frans).')
    parser.add_argument('--n-draws', type=int, default=200)
    parser.add_argument('--perms-per-draw', type=int, default=200)
    parser.add_argument('--symmetrize', action='store_true',
                        help='Run only the symmetrized condition.')
    parser.add_argument('--asymmetric-only', action='store_true',
                        help='Run only the asymmetric condition.')
    parser.add_argument('--no-partial-rank', action='store_true',
                        help='Disable rank-distance partialing.')
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args()

    # Block-specific config: partial_out_rank ON by default (rank is a
    # natural confound to address with the partial Spearman approach).
    cfg = SocialRSAConfig(
        region=args.region,
        session=None,
        window=(0.300, 0.600),
        min_epoch_duration=1.0,
        min_reps_per_monkey=7,
        neural_metric='correlation',
        model_factors=[],
        exclude_groups=['Stranger Things'],
        normalization=None,
        partial_out_rank=not args.no_partial_rank,
        rank_transform_behavior=False,
        n_permutations=0,             # inner perms handled per draw
        between_group_permutations=0,
        n_bootstrap=0,
        pseudo_population=True,
        save_plots=True,
        save_dir='rsa_subset_results',
        rng_seed=args.seed,
    )
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

    print(f"\n{'█'*70}")
    print(f"█  SUBSET RSA  —  region={cfg.region}, target_size={args.target_size}")
    print(f"█  n_draws={args.n_draws}, perms/draw={args.perms_per_draw}")
    print(f"█  partial rank: {cfg.partial_out_rank}")
    print(f"{'█'*70}")

    if args.symmetrize and args.asymmetric_only:
        raise SystemExit("Choose either --symmetrize OR --asymmetric-only, not both.")

    if args.symmetrize:
        conditions = [(True,  'symmetrized')]
    elif args.asymmetric_only:
        conditions = [(False, 'asymmetric')]
    else:
        conditions = [(False, 'asymmetric'), (True, 'symmetrized')]

    confound_fn = None
    if cfg.partial_out_rank:
        confound_fn = lambda ids, info: build_rank_distance_matrix(ids, info)

    out_json = {
        'region': cfg.region,
        'config': {
            'window': list(cfg.window),
            'neural_metric': cfg.neural_metric,
            'partial_out_rank': cfg.partial_out_rank,
            'target_size': args.target_size,
            'n_draws': args.n_draws,
            'perms_per_draw': args.perms_per_draw,
        },
        'results': {},
    }

    for symmetrize, label in conditions:
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

        out_json['results'][label] = summary_to_json(summary)

    save_root = f"{cfg.save_dir}/{cfg.region}"
    os.makedirs(save_root, exist_ok=True)
    json_path = f"{save_root}/subset_results.json"
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    if cfg.save_plots:
        plt.show()


if __name__ == '__main__':
    main()
