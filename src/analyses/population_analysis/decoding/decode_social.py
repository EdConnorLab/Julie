# decode_social.py
"""
Social variable decoding from pseudo-population neural activity.

Decodes: Familiarity, Group membership, Sex, Age, Dominance rank.
Uses shared DecodeConfig and decode_utils.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decode_config import DecodeConfig
from decode_utils import (
    load_and_prepare, load_metadata, filter_neurons_by_min_trials,
    build_pseudo_population, run_classification, run_regression,
    run_classification_permutation, run_regression_permutation,
    compute_p_value, map_labels, map_continuous, median_split,
)

# Familiarity mapping
FAMILIAR_GROUPS = ['Zombies', 'Best Frans']
UNFAMILIAR_GROUPS = ['Instigators', 'Stranger Things']


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL MAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def map_familiarity_labels(identity_labels, group_lookup):
    labels = []
    for mid in identity_labels:
        group = group_lookup.get(mid, None)
        if group in FAMILIAR_GROUPS:
            labels.append('Familiar')
        elif group in UNFAMILIAR_GROUPS:
            labels.append('Unfamiliar')
        else:
            labels.append(None)
    return np.array(labels)


def map_group_labels(identity_labels, group_lookup):
    return np.array([group_lookup.get(mid, None) for mid in identity_labels])


def map_sex_labels(identity_labels, meta_lookup):
    return np.array([meta_lookup.get(mid, {}).get('Sex', None)
                     for mid in identity_labels])


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC VARIABLE DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def decode_variable(variable_name, X_draws, cfg, *,
                    labels_draws=None, continuous_draws=None,
                    is_regression=False, run_perms=False):
    """
    Run decoding for a single variable across pseudo-population draws.
    Classification: provide labels_draws.
    Regression: provide continuous_draws, set is_regression=True.
    """
    print(f"\n  --- {variable_name} "
          f"{'(regression)' if is_regression else '(classification)'} ---")

    draw_scores = []

    for d in range(len(X_draws)):
        if is_regression:
            y = continuous_draws[d]
            if np.sum(~np.isnan(y)) < 10:
                print(f"    Draw {d + 1}: too few valid samples, skipping")
                continue
            score = run_regression(X_draws[d], y, cfg, seed_offset=d)
            draw_scores.append(score)
        else:
            y = labels_draws[d]
            valid = y != None  # noqa
            if np.sum(valid) < 10:
                print(f"    Draw {d + 1}: too few valid samples, skipping")
                continue
            X_valid, y_valid = X_draws[d][valid], y[valid]
            fold_accs = run_classification(X_valid, y_valid, cfg, do_balance=True)
            draw_scores.append(np.mean(fold_accs))

    if not draw_scores:
        print("    No valid draws!")
        return None

    mean_score = np.mean(draw_scores)
    std_score = np.std(draw_scores)
    metric = 'R²' if is_regression else 'accuracy'
    print(f"    Mean {metric}: {mean_score:.4f} ±{std_score:.4f}")

    # Permutation test (on first valid draw)
    perm_scores = None
    p_value = None
    if run_perms:
        print(f"    Running permutation test ({cfg.n_permutations} permutations)...")
        if is_regression:
            perm_scores = run_regression_permutation(
                X_draws[0], continuous_draws[0], cfg,
                n_perms=cfg.n_permutations, seed_offset=9999)
        else:
            y = labels_draws[0]
            valid = y != None  # noqa
            perm_scores = run_classification_permutation(
                X_draws[0][valid], y[valid], cfg,
                n_perms=cfg.n_permutations, seed_offset=9999)
        p_value = compute_p_value(mean_score, perm_scores)
        print(f"    Permutation p-value: {p_value:.4f}")

    # Chance level
    if is_regression:
        chance = 0.0
    else:
        y_example = labels_draws[0]
        valid = y_example != None  # noqa
        unique = np.unique(y_example[valid])
        chance = 1.0 / len(unique)

    return {
        'variable': variable_name,
        'type': 'regression' if is_regression else 'classification',
        'mean_score': mean_score,
        'std_score': std_score,
        'draw_scores': draw_scores,
        'chance': chance,
        'perm_scores': perm_scores,
        'p_value': p_value,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    clf_results = {k: v for k, v in all_results.items()
                   if v is not None and v['type'] == 'classification'}
    reg_results = {k: v for k, v in all_results.items()
                   if v is not None and v['type'] == 'regression'}

    # 1. Classification bar plot
    if clf_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(clf_results.keys())
        means = [clf_results[n]['mean_score'] for n in names]
        stds = [clf_results[n]['std_score'] for n in names]
        chances = [clf_results[n]['chance'] for n in names]
        bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                      '#937860', '#DA8BC3', '#8C8C8C']
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=5,
               color=bar_colors[:len(names)], edgecolor='black', linewidth=0.8, alpha=0.7)
        for i, ch in enumerate(chances):
            ax.plot([i - 0.3, i + 0.3], [ch, ch], 'k--', linewidth=1)
        for i, name in enumerate(names):
            r = clf_results[name]
            label = f'{means[i]:.3f}'
            if r['p_value'] is not None:
                p = r['p_value']
                p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                label += f'\n{p_str}'
            ax.text(i, means[i] + stds[i] + 0.01, label,
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Social Variable Decoding (Classification)')
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) + 0.1)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'social_decoding_classification.png'), dpi=200)
        plt.close()
        print(f"  Saved: social_decoding_classification.png")

    # 2. Regression bar plot
    if reg_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(reg_results.keys())
        means = [reg_results[n]['mean_score'] for n in names]
        stds = [reg_results[n]['std_score'] for n in names]
        bar_colors = ['#55A868', '#C44E52']
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=5,
               color=bar_colors[:len(names)], edgecolor='black', linewidth=0.8, alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='Chance (R²=0)')
        for i, name in enumerate(names):
            r = reg_results[name]
            label = f'{means[i]:.3f}'
            if r['p_value'] is not None:
                p = r['p_value']
                p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                label += f'\n{p_str}'
            y_pos = max(means[i] + stds[i], 0) + 0.01
            ax.text(i, y_pos, label, ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('R²')
        ax.set_title('Social Variable Decoding (Regression)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'social_decoding_regression.png'), dpi=200)
        plt.close()
        print(f"  Saved: social_decoding_regression.png")

    # 3. Permutation distributions
    for name, r in all_results.items():
        if r is None or r['perm_scores'] is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(r['perm_scores'], bins=50, color='gray', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Permuted')
        ax.axvline(r['mean_score'], color='red', linewidth=2,
                   label=f'Observed = {r["mean_score"]:.3f}')
        metric = 'R²' if r['type'] == 'regression' else 'Accuracy'
        if r['type'] == 'classification':
            ax.axvline(r['chance'], color='blue', linestyle='--', linewidth=1,
                       label=f'Chance = {r["chance"]:.3f}')
        p = r['p_value']
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        ax.set_title(f'{name} ({p_str})')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        fname = f'permutation_{name.replace(" ", "_").replace("/", "_")}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()
        print(f"  Saved: {fname}")

    print(f"\nAll figures saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ─── CONFIG ──────────────────────────────────────────────────────────
    cfg = DecodeConfig(
        data_path='/Cortana/sorted_spike_cache_filtered',
        min_trials=7,
        n_pseudo_draws=10,
        n_pca=50,
        skip_perm=True,
        n_permutations=1000,
        response_window=(0.200, 0.500),       # e.g. (0.0, 0.5)
        output_dir='./social_decoding_output',
    )
    cfg.validate()

    metadata_csv = '/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv'
    ADULT_AGE_CUTOFF = 4
    # ─────────────────────────────────────────────────────────────────────

    # ── Load & prepare ──
    df = load_and_prepare(cfg)

    # ── Load metadata ──
    print("\nLoading metadata...")
    meta = load_metadata(metadata_csv)
    meta_lookup = {row['MonkeyName']: row.to_dict() for _, row in meta.iterrows()}
    group_lookup = dict(zip(df['MonkeyName'], df['MonkeyGroup']))

    # ── All identities ──
    all_identities = sorted(df['MonkeyName'].unique())
    print(f"\nTotal identities: {len(all_identities)}")

    missing = [mid for mid in all_identities if mid not in meta_lookup]
    if missing:
        print(f"  WARNING: {len(missing)} identities missing from metadata: {missing}")

    # ── Filter neurons ──
    if cfg.min_trials > 0:
        df, _ = filter_neurons_by_min_trials(df, all_identities, cfg.min_trials)

    # ── Build pseudo-population ──
    print(f"\n{'=' * 60}")
    print("BUILDING PSEUDO-POPULATION")
    print(f"{'=' * 60}")
    X_draws, id_labels_draws, neuron_ids, valid_ids = build_pseudo_population(
        df, all_identities, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed
    )
    n_neurons = len(neuron_ids)
    print(f"Neurons: {n_neurons}")
    print(f"Feature matrix shape: {X_draws[0].shape}")

    all_results = {}
    run_perms = not cfg.skip_perm

    # ─── 1. Familiarity ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 1: FAMILIARITY")
    print(f"{'=' * 60}")
    fam_labels_draws = [map_familiarity_labels(ids, group_lookup) for ids in id_labels_draws]
    example = fam_labels_draws[0]
    valid = example != None  # noqa
    unique, counts = np.unique(example[valid], return_counts=True)
    print(f"  Classes: {dict(zip(unique, counts))}")

    all_results['Familiarity'] = decode_variable(
        'Familiarity', X_draws, cfg, labels_draws=fam_labels_draws, run_perms=run_perms)

    # ─── 1b. Familiarity (females only) ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 1b: FAMILIARITY (FEMALES ONLY)")
    print(f"{'=' * 60}")
    female_monkeys = [mid for mid in all_identities
                      if meta_lookup.get(mid, {}).get('Sex') == 'F']
    fam_counts = {'Familiar': 0, 'Unfamiliar': 0}
    for mid in female_monkeys:
        grp = group_lookup.get(mid)
        if grp in FAMILIAR_GROUPS:
            fam_counts['Familiar'] += 1
        elif grp in UNFAMILIAR_GROUPS:
            fam_counts['Unfamiliar'] += 1
    print(f"  Female monkeys: {len(female_monkeys)}")
    print(f"  Familiar: {fam_counts['Familiar']}, Unfamiliar: {fam_counts['Unfamiliar']}")

    if min(fam_counts.values()) >= 3:
        df_fem = df[df['MonkeyName'].isin(female_monkeys)]
        if cfg.min_trials > 0:
            df_fem, _ = filter_neurons_by_min_trials(df_fem, female_monkeys, cfg.min_trials)
        X_fem, fem_ids, _, _ = build_pseudo_population(
            df_fem, female_monkeys, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed)
        fem_fam = [map_familiarity_labels(ids, group_lookup) for ids in fem_ids]
        all_results['Familiarity (F only)'] = decode_variable(
            'Familiarity (F only)', X_fem, cfg, labels_draws=fem_fam, run_perms=run_perms)
    else:
        print("  Too few females in one category, skipping")
        all_results['Familiarity (F only)'] = None

    # ─── 2. Group membership ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 2: GROUP MEMBERSHIP")
    print(f"{'=' * 60}")
    group_labels_draws = [map_group_labels(ids, group_lookup) for ids in id_labels_draws]
    all_results['Group'] = decode_variable(
        'Group', X_draws, cfg, labels_draws=group_labels_draws, run_perms=run_perms)

    # ─── 3. Sex ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 3: SEX")
    print(f"{'=' * 60}")
    sex_labels_draws = [map_sex_labels(ids, meta_lookup) for ids in id_labels_draws]
    all_results['Sex'] = decode_variable(
        'Sex', X_draws, cfg, labels_draws=sex_labels_draws, run_perms=run_perms)

    # ─── 4. Age ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 4: AGE")
    print(f"{'=' * 60}")

    # 4a. Adult vs Juvenile
    age_cont_draws = [map_continuous(ids, meta_lookup, 'Age') for ids in id_labels_draws]
    age_class_draws = []
    for ages in age_cont_draws:
        labels = np.full(len(ages), None, dtype=object)
        vm = ~np.isnan(ages)
        labels[vm & (ages >= ADULT_AGE_CUTOFF)] = 'Adult'
        labels[vm & (ages < ADULT_AGE_CUTOFF)] = 'Juvenile'
        age_class_draws.append(labels)

    all_results['Age (class)'] = decode_variable(
        'Age (class)', X_draws, cfg, labels_draws=age_class_draws, run_perms=run_perms)

    # 4b. Age (females only)
    print(f"\n{'=' * 60}")
    print("VARIABLE 4b: AGE (FEMALES ONLY)")
    print(f"{'=' * 60}")
    fem_age_counts = {'Adult': 0, 'Juvenile': 0}
    for mid in female_monkeys:
        age = meta_lookup.get(mid, {}).get('Age')
        if age is not None and not np.isnan(float(age)):
            fem_age_counts['Adult' if float(age) >= ADULT_AGE_CUTOFF else 'Juvenile'] += 1
    print(f"  Adult: {fem_age_counts['Adult']}, Juvenile: {fem_age_counts['Juvenile']}")

    if min(fem_age_counts.values()) >= 3:
        df_fem_age = df[df['MonkeyName'].isin(female_monkeys)]
        if cfg.min_trials > 0:
            df_fem_age, _ = filter_neurons_by_min_trials(df_fem_age, female_monkeys, cfg.min_trials)
        X_fage, fage_ids, _, _ = build_pseudo_population(
            df_fem_age, female_monkeys, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed)
        fem_age_class = []
        for ids in fage_ids:
            ages = map_continuous(ids, meta_lookup, 'Age')
            labels = np.full(len(ages), None, dtype=object)
            vm = ~np.isnan(ages)
            labels[vm & (ages >= ADULT_AGE_CUTOFF)] = 'Adult'
            labels[vm & (ages < ADULT_AGE_CUTOFF)] = 'Juvenile'
            fem_age_class.append(labels)
        all_results['Age (F only)'] = decode_variable(
            'Age (F only)', X_fage, cfg, labels_draws=fem_age_class, run_perms=run_perms)
    else:
        print("  Too few females in one age category, skipping")
        all_results['Age (F only)'] = None

    # 4c. Sex × Age
    print(f"\n{'=' * 60}")
    print("VARIABLE 4c: SEX × AGE")
    print(f"{'=' * 60}")
    sex_age_draws = []
    for ids in id_labels_draws:
        ages = map_continuous(ids, meta_lookup, 'Age')
        sexes = map_sex_labels(ids, meta_lookup)
        labels = np.full(len(ages), None, dtype=object)
        for i in range(len(ages)):
            if sexes[i] is None or np.isnan(ages[i]):
                continue
            if ages[i] < ADULT_AGE_CUTOFF:
                labels[i] = 'Juvenile'
            elif sexes[i] == 'M':
                labels[i] = 'Adult Male'
            else:
                labels[i] = 'Adult Female'
        sex_age_draws.append(labels)
    all_results['Sex x Age'] = decode_variable(
        'Sex x Age', X_draws, cfg, labels_draws=sex_age_draws, run_perms=run_perms)

    # 4d. Age regression
    all_results['Age (regression)'] = decode_variable(
        'Age (regression)', X_draws, cfg,
        continuous_draws=age_cont_draws, is_regression=True, run_perms=run_perms)

    # ─── 5. Dominance rank (familiar groups only) ────────────────────────
    print(f"\n{'=' * 60}")
    print("VARIABLE 5: DOMINANCE RANK (familiar groups only)")
    print(f"{'=' * 60}")
    ranked_monkeys = [mid for mid in all_identities
                      if meta_lookup.get(mid, {}).get('Rank') is not None
                      and not np.isnan(float(meta_lookup[mid]['Rank']))
                      and group_lookup.get(mid) in FAMILIAR_GROUPS]
    print(f"  Monkeys with rank in familiar groups: {len(ranked_monkeys)}")

    if len(ranked_monkeys) >= 4:
        df_ranked = df[df['MonkeyName'].isin(ranked_monkeys)]
        if cfg.min_trials > 0:
            df_ranked, _ = filter_neurons_by_min_trials(df_ranked, ranked_monkeys, cfg.min_trials)
        X_rank, rank_ids, _, _ = build_pseudo_population(
            df_ranked, ranked_monkeys, n_draws=cfg.n_pseudo_draws, seed=cfg.random_seed)

        rank_cont = [map_continuous(ids, meta_lookup, 'Rank') for ids in rank_ids]
        rank_class = [median_split(r, 'Rank') for r in rank_cont]
        all_results['Rank (class)'] = decode_variable(
            'Rank (class)', X_rank, cfg, labels_draws=rank_class, run_perms=run_perms)
        all_results['Rank (regression)'] = decode_variable(
            'Rank (regression)', X_rank, cfg,
            continuous_draws=rank_cont, is_regression=True, run_perms=run_perms)
    else:
        print("  Too few ranked monkeys, skipping")
        all_results['Rank (class)'] = None
        all_results['Rank (regression)'] = None

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neurons: {n_neurons}")
    if cfg.min_trials > 0:
        print(f"  (filtered: min_trials >= {cfg.min_trials})")
    if cfg.response_window:
        print(f"Response window: {cfg.response_window[0]}–{cfg.response_window[1]} s")
    print()

    for name, r in all_results.items():
        if r is None:
            print(f"  {name:25s}: FAILED")
            continue
        metric = 'R²' if r['type'] == 'regression' else 'acc'
        chance_str = (f"chance={r['chance']:.3f}" if r['type'] == 'classification'
                      else 'chance=0')
        p_str = f", p={r['p_value']:.4f}" if r['p_value'] is not None else ""
        print(f"  {name:25s}: {metric}={r['mean_score']:.4f} ±{r['std_score']:.4f} "
              f"({chance_str}{p_str})")

    # ── Save ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    plot_all_results(all_results, cfg.output_dir)
    save_path = os.path.join(cfg.output_dir, 'social_decoding_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'config': cfg, 'results': all_results}, f)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
