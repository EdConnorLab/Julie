# run_rsa_rawsocial.py
"""
Raw-social-matrix RSA runner.

Builds two RDM families that DO NOT symmetrize the social matrix:
  - signed asymmetry (pair-level M[i,j] - M[j,i])
  - concat[row, col] per-monkey vector (correlation distance)

Both preserve directed structure. The signed-asymmetry RDM is
particularly rank-correlated (dominant→subordinate flow), so this
runner reports results both WITH and WITHOUT rank partialing by
default — toggle with --partial-only or --no-partial-only.

Run:
    python run_rsa_rawsocial.py                       # both versions
    python run_rsa_rawsocial.py --region AMG
    python run_rsa_rawsocial.py --partial-only        # only rank-partialed
    python run_rsa_rawsocial.py --no-partial-only     # only raw (no partial)
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


def main():
    parser = argparse.ArgumentParser(description='Raw-social RSA (no symmetrization)')
    parser.add_argument('--region', type=str, default='ALL')
    parser.add_argument('--partial-only',    action='store_true',
                        help='Run only the rank-partialed version.')
    parser.add_argument('--no-partial-only', action='store_true',
                        help='Run only the raw (no partial) version.')
    parser.add_argument('--rank-transform-behavior', action='store_true',
                        help='Rank-transform behavioral profiles before distance.')
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-bootstrap',    type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args()

    if args.partial_only and args.no_partial_only:
        raise SystemExit("Choose at most one of --partial-only / --no-partial-only.")

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
        partial_out_rank=False,            # toggled per-run below
        rank_transform_behavior=args.rank_transform_behavior,
        n_permutations=args.n_permutations,
        between_group_permutations=0,
        n_bootstrap=args.n_bootstrap,
        pseudo_population=True,
        save_plots=True,
        save_dir='rsa_rawsocial_results',
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
    identities = pp_result['identities']
    neural_rdm = pp_result['neural_rdm']

    print(f"\n{'█'*70}")
    print(f"█  RAW SOCIAL RSA  —  region={cfg.region}")
    print(f"█  signed asymmetry  +  concat[row,col] (correlation distance)")
    print(f"█  rank_transform_behavior: {cfg.rank_transform_behavior}")
    print(f"{'█'*70}")

    raw_rdms = build_raw_social_rdms(
        identities, interactions,
        behavior_types=('affiliation', 'agonism', 'submission'),
        log_transform=False,
        rank_transform=cfg.rank_transform_behavior,
        concat_metric='correlation',
        include_combined=True)

    # Which versions to run
    if args.partial_only:
        versions = [('rank_partialed', True)]
    elif args.no_partial_only:
        versions = [('raw_no_partial', False)]
    else:
        versions = [('raw_no_partial', False), ('rank_partialed', True)]

    rank_confound_mat = build_rank_distance_matrix(identities, info_df)

    out_json = {
        'region': cfg.region,
        'config': {
            'window': list(cfg.window),
            'neural_metric': cfg.neural_metric,
            'rank_transform_behavior': cfg.rank_transform_behavior,
            'n_permutations': cfg.n_permutations,
            'n_bootstrap': cfg.n_bootstrap,
        },
        'results': {},
    }

    for tag, do_partial in versions:
        confound = rank_confound_mat if do_partial else None
        partial_str = "(rank partialed)" if do_partial else "(no partial)"
        print(f"\n  --- {tag} {partial_str} ---")
        group_comp = compare_neural_to_social_by_group(
            neural_rdm, raw_rdms, identities, info_df,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
            rng_seed=cfg.rng_seed,
            confound_matrix=confound)
        print_group_comparisons(group_comp)
        out_json['results'][tag] = _serialize_group_comp(group_comp)

    save_root = f"{cfg.save_dir}/{cfg.region}"
    os.makedirs(save_root, exist_ok=True)
    json_path = f"{save_root}/rawsocial_results.json"
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    if cfg.save_plots:
        plt.show()


if __name__ == '__main__':
    main()
