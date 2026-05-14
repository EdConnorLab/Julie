# run_rsa_extensions.py
"""
Runner script for the three RSA extensions.

Each analysis is in its own clearly-delimited block. Toggle the
RUN_* flags at the top of main() to enable/disable any block.

  Block 1 — Subset (rsa_subset.py)
      Cap each group at n=5 monkeys, run dissimilarity RSA N times
      with random subsampling of larger groups, aggregate per-group ρ.

  Block 2 — Raw social (rsa_raw_social.py)
      Build pair-level signed asymmetry RDMs AND per-monkey
      concat[row, col] directed-vector RDMs. Compare to neural RDM.

  Block 3 — Sign-corrected (rsa_sign_corrected.py)
      AMG-focused. Trial split-half within neuron, sign each neuron
      by its submission-axis slope, flip negatives before building
      the population RDM. Compare to social RDMs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from rsa_config import SocialRSAConfig
from rsa_core import run_rsa_pseudopop, build_neural_rdm
from rsa_social import (
    load_all_interaction_matrices,
    build_social_rdms,
    compare_neural_to_social_by_group,
    build_rank_distance_matrix,
    print_group_comparisons,
)

# The three extension modules
from rsa_subset import (
    run_subsampled_dissimilarity_rsa,
    print_subsample_summary,
)
from rsa_raw_social import build_raw_social_rdms
from rsa_sign_corrected import (
    run_sign_corrected_pseudopop,
    compute_submission_scores,
)


# ──────────────────────────────────────────────────────────
# Paths (mirrors run_rsa_social.py)
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


# ──────────────────────────────────────────────────────────
# Small helper: JSON-serialize a per-group comparison dict
# ──────────────────────────────────────────────────────────

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
    import argparse
    parser = argparse.ArgumentParser(description='RSA extensions runner')
    parser.add_argument('--region', type=str, default=None,
                        help='Brain region (AMG, ER, ALL). Overrides cfg.')
    parser.add_argument('--skip-subset',  action='store_true')
    parser.add_argument('--skip-raw',     action='store_true')
    parser.add_argument('--skip-sign',    action='store_true')
    args, _ = parser.parse_known_args()

    # ── Toggles ──
    RUN_SUBSET        = not args.skip_subset
    RUN_RAW_SOCIAL    = not args.skip_raw
    RUN_SIGN_CORRECT  = not args.skip_sign

    # ── Config ──
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

    # ── Load shared inputs ──
    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
        print(f"Excluded groups {cfg.exclude_groups}: "
              f"{df['MonkeyName'].nunique()} monkeys remaining")

    print("\n--- Loading interaction matrices ---")
    interactions = load_all_interaction_matrices(BEHAVIOR_FILES)

    # Standard pseudo-population result (used by blocks 1 and 2).
    # Block 3 builds its own rate matrix from spike data via split-half.
    if RUN_SUBSET or RUN_RAW_SOCIAL:
        print("\n--- Building standard pseudo-population result ---")
        pp_result = run_rsa_pseudopop(df, info_df, cfg)
    else:
        pp_result = None

    save_root = f"{cfg.save_dir}/{cfg.region}"
    os.makedirs(save_root, exist_ok=True)
    all_json = {'region': cfg.region, 'config': {
        'window': list(cfg.window),
        'neural_metric': cfg.neural_metric,
        'normalization': cfg.normalization,
        'partial_out_rank': cfg.partial_out_rank,
        'n_permutations': cfg.n_permutations,
    }}

    # ═══════════════════════════════════════════════════════
    # BLOCK 1 — SUBSET (group-size control)
    # ═══════════════════════════════════════════════════════
    if RUN_SUBSET:
        print(f"\n{'█'*70}")
        print(f"█  BLOCK 1: SUBSET — cap groups at n=5, n_draws=200")
        print(f"{'█'*70}")

        confound_fn = None
        if cfg.partial_out_rank:
            confound_fn = lambda ids, info: build_rank_distance_matrix(ids, info)

        # Two conditions: asymmetric (raw rows) and symmetrized
        for symmetrize, label in [(False, 'asymmetric'), (True, 'symmetrized')]:
            print(f"\n  --- Subsampling condition: {label} ---")
            summary = run_subsampled_dissimilarity_rsa(
                pp_result, info_df, interactions, cfg,
                target_size=5, n_draws=200, rng_seed=cfg.rng_seed,
                symmetrize=symmetrize, log_transform=False,
                behavior_types=('affiliation', 'agonism', 'submission'),
                confound_matrix_fn=confound_fn,
                ci_level=0.95)
            print_subsample_summary(
                summary,
                title=f"Subsampled (n=5 cap, 200 draws) — {label}")

            # Serialize (drop the raw draw_rhos arrays for JSON)
            clean = {}
            for grp, by_mat in summary.items():
                clean[grp] = {m: {k: v for k, v in e.items() if k != 'draw_rhos'}
                              for m, e in by_mat.items()}
            all_json.setdefault('subset', {})[label] = clean

    # ═══════════════════════════════════════════════════════
    # BLOCK 2 — RAW SOCIAL (no symmetrization)
    # ═══════════════════════════════════════════════════════
    if RUN_RAW_SOCIAL:
        print(f"\n{'█'*70}")
        print(f"█  BLOCK 2: RAW SOCIAL — signed asymmetry + concat[row,col]")
        print(f"{'█'*70}")

        identities = pp_result['identities']
        neural_rdm = pp_result['neural_rdm']

        # Build BOTH raw RDM families in one dict
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

    # ═══════════════════════════════════════════════════════
    # BLOCK 3 — SIGN-CORRECTED (split-half, AMG-focused)
    # ═══════════════════════════════════════════════════════
    if RUN_SIGN_CORRECT:
        print(f"\n{'█'*70}")
        print(f"█  BLOCK 3: SIGN-CORRECTED — split-half AMG opponent population")
        print(f"{'█'*70}")

        if cfg.region != 'AMG':
            print(f"  NOTE: region is {cfg.region!r}, not 'AMG'. The "
                  f"sign-correction motivation is amygdala-specific.")

        sc_result = run_sign_corrected_pseudopop(
            df, info_df, interactions, cfg,
            mode='both',                       # flip + subpop side-by-side
            sign_axis='submission',
            axis_aggregate='column_sum',       # submission RECEIVED
            rng_seed=cfg.rng_seed)

        identities_sc = sc_result['identities']

        # Social RDMs aligned to the sign-corrected identities
        social_rdms_sc = build_social_rdms(
            identities_sc, interactions,
            behavior_types=['affiliation', 'agonism', 'submission'],
            symmetrize=False, profile_metric='correlation',
            log_transform=False, include_combined=True,
            rank_transform=cfg.rank_transform_behavior)

        rank_confound_sc = None
        if cfg.partial_out_rank:
            rank_confound_sc = build_rank_distance_matrix(identities_sc, info_df)

        # Compare each available neural RDM (flipped, pos-only, neg-only)
        sc_json = {
            'n_neurons_total': int(sc_result['n_neurons']),
            'n_neurons_pos':   int(sc_result['n_neurons_pos']),
            'n_neurons_neg':   int(sc_result['n_neurons_neg']),
            'identities':      identities_sc,
        }

        for tag, rdm_key in [('flipped_combined', 'neural_rdm_flipped'),
                              ('subpop_pos_only', 'neural_rdm_pos'),
                              ('subpop_neg_only', 'neural_rdm_neg')]:
            rdm = sc_result.get(rdm_key)
            if rdm is None:
                print(f"\n  [{tag}] no RDM (not enough neurons in subpop)")
                continue
            print(f"\n  --- {tag} ---")
            group_comp_sc = compare_neural_to_social_by_group(
                rdm, social_rdms_sc, identities_sc, info_df,
                n_permutations=cfg.n_permutations,
                n_bootstrap=cfg.n_bootstrap,
                rng_seed=cfg.rng_seed,
                confound_matrix=rank_confound_sc)
            print_group_comparisons(group_comp_sc)
            sc_json[tag] = _serialize_group_comp(group_comp_sc)

        all_json['sign_corrected'] = sc_json

    # ── Write combined JSON ──
    json_path = f"{save_root}/extensions_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_json, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    if cfg.save_plots:
        plt.show()


if __name__ == '__main__':
    main()
