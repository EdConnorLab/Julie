# run_rsa_signcorrect.py
"""
Standalone runner for sign-corrected RSA (AMG-focused).

The sign-correction motivation is specific to amygdala: an opponent
population where roughly half the neurons code submission-axis with
positive slope and half with negative slope cancel out in a naive
population RDM. We use a trial split-half to estimate per-neuron sign
on half A, then build the population RDM on half B with negatives
flipped.

Defaults: region='AMG', partial_out_rank=False (the sign axis
correlates strongly with rank, so partialing rank would partly undo
the very axis used for flipping).

Run:
    python run_rsa_signcorrect.py                    # AMG, defaults
    python run_rsa_signcorrect.py --region ER        # try ER too
    python run_rsa_signcorrect.py --partial-rank     # partial out rank
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from rsa_config import SocialRSAConfig
from rsa_social import (
    load_all_interaction_matrices,
    build_social_rdms,
    compare_neural_to_social_by_group,
    build_rank_distance_matrix,
    print_group_comparisons,
)
from rsa_sign_corrected import run_sign_corrected_pseudopop


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
    parser = argparse.ArgumentParser(description='Sign-corrected RSA runner (AMG)')
    parser.add_argument('--region', type=str, default='AMG',
                        help="Brain region (default 'AMG').")
    parser.add_argument('--mode', type=str, default='both',
                        choices=['flip', 'subpop', 'both'])
    parser.add_argument('--sign-axis', type=str, default='submission',
                        help="Behavior used as preference axis.")
    parser.add_argument('--axis-aggregate', type=str, default='column_sum',
                        choices=['column_sum', 'row_sum'])
    parser.add_argument('--partial-rank', action='store_true',
                        help="Partial out rank distance from neural↔social "
                             "correlation. Default OFF for sign-corrected — "
                             "rank correlates with the sign axis.")
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args()

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
        partial_out_rank=args.partial_rank,
        rank_transform_behavior=False,
        n_permutations=1000,
        between_group_permutations=0,
        n_bootstrap=0,
        pseudo_population=True,
        save_plots=True,
        save_dir='rsa_signcorrect_results',
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

    print(f"\n{'█'*70}")
    print(f"█  SIGN-CORRECTED RSA  —  region={cfg.region}")
    print(f"█  sign axis : {args.sign_axis} ({args.axis_aggregate})")
    print(f"█  mode      : {args.mode}")
    print(f"█  partial rank: {cfg.partial_out_rank}")
    print(f"{'█'*70}")

    sc_result = run_sign_corrected_pseudopop(
        df, info_df, interactions, cfg,
        mode=args.mode,
        sign_axis=args.sign_axis,
        axis_aggregate=args.axis_aggregate,
        rng_seed=cfg.rng_seed)

    identities = sc_result['identities']

    # Social RDMs aligned to surviving identities
    social_rdms = build_social_rdms(
        identities, interactions,
        behavior_types=['affiliation', 'agonism', 'submission'],
        symmetrize=False, profile_metric='correlation',
        log_transform=False, include_combined=True,
        rank_transform=cfg.rank_transform_behavior)

    rank_confound = None
    if cfg.partial_out_rank:
        rank_confound = build_rank_distance_matrix(identities, info_df)

    out_json = {
        'region': cfg.region,
        'mode': args.mode,
        'sign_axis': args.sign_axis,
        'axis_aggregate': args.axis_aggregate,
        'partial_out_rank': cfg.partial_out_rank,
        'n_neurons_total': int(sc_result['n_neurons']),
        'n_neurons_pos':   int(sc_result['n_neurons_pos']),
        'n_neurons_neg':   int(sc_result['n_neurons_neg']),
        'identities':      identities,
    }

    for tag, rdm_key in [('flipped_combined', 'neural_rdm_flipped'),
                          ('subpop_pos_only', 'neural_rdm_pos'),
                          ('subpop_neg_only', 'neural_rdm_neg')]:
        rdm = sc_result.get(rdm_key)
        if rdm is None:
            print(f"\n  [{tag}] no RDM (not enough neurons)")
            continue
        print(f"\n  --- {tag} ---")
        group_comp = compare_neural_to_social_by_group(
            rdm, social_rdms, identities, info_df,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
            rng_seed=cfg.rng_seed,
            confound_matrix=rank_confound)
        print_group_comparisons(group_comp)
        out_json[tag] = _serialize_group_comp(group_comp)

    save_root = f"{cfg.save_dir}/{cfg.region}"
    os.makedirs(save_root, exist_ok=True)
    json_path = f"{save_root}/signcorrect_results.json"
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    if cfg.save_plots:
        plt.show()


if __name__ == '__main__':
    main()
