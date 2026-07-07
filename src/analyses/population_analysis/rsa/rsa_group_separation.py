# rsa_group_separation.py
"""
RSA-side group-separation / familiarity test.  Belongs in the RSA folder
(imports your existing rsa_core / rsa_config / data-loading stack).

This is the COMPANION to the state-space test, asking a different question:
does the neural REPRESENTATIONAL geometry track a graded familiarity structure,
and which metric ("ruler") is needed to see it?

Why this script exists
----------------------
1. GRADED familiarity. Your rsa_core 'familiarity' factor is binary
   (familiar/unfamiliar). Here we use a graded ordinal
   {Zombies=2, Best Frans=1, Instigators=0, Stranger Things=0} so the model can
   express home > neighbor > stranger. (Instigators and Stranger Things are tied
   at 0 because the subject cannot know either group's social structure — they
   are both just "strangers". See SOCIAL_GROUP for the pooled 3-level version.)

2. METRIC SWEEP (the "ruler"). Correlation distance removes overall firing-rate
   magnitude; Euclidean keeps it. Since ER's familiarity gradient was shown to
   be magnitude-carried (it dies under z-scoring), correlation distance is blind
   to it. We run BOTH metrics so the gradient appearing under Euclidean but not
   correlation is itself the evidence for a gain/magnitude code.

3. GAIN CONTROL. If the familiarity effect is just "novel faces fire more", it
   should vanish when we partial out each identity's mean firing rate. We test
   familiarity | group AND familiarity | (group + mean_rate). Surviving the
   second means it is not merely a gain effect.

4. MONOTONIC CONTAINMENT SLOPE. The gradient hypothesis predicts within-group
   dissimilarity decreases monotonically with familiarity (strangers most
   dispersed, home most consolidated). We regress within-group mean dissimilarity
   on familiarity level and permutation-test the slope.

Everything heavy (rate matrix, normalization, neural RDM, partial Spearman,
permutation null) is reused from rsa_core — this script only adds the graded
model, the metric sweep, the rate control, and the slope test on top.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from rsa_config import RSAConfig
from rsa_core import (run_rsa_pseudopop, build_neural_rdm,
                      compare_rdms, compare_rdms_partial)

MONKEY_INFO_PATH = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"

# ── EDIT HERE ──────────────────────────────────────────────────────────
# Graded familiarity (larger = more familiar). Instigators/Stranger tied at 0.
FAMILIARITY_LEVEL = {
    'Zombies': 2, 'Best Frans': 1, 'Instigators': 0, 'Stranger Things': 0,
}
# Pooled social grouping: the two stranger groups become one "Strangers" class,
# because the subject cannot distinguish them as social groups.
SOCIAL_GROUP = {
    'Zombies': 'Zombies', 'Best Frans': 'Best Frans',
    'Instigators': 'Strangers', 'Stranger Things': 'Strangers',
}
METRICS = ['correlation', 'euclidean']   # the two "rulers"
# ───────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# small local model-RDM builders (so we don't have to edit rsa_core)
# ═══════════════════════════════════════════════════════════════════════

def categorical_rdm(labels):
    labels = np.asarray(labels, dtype=object)
    return (labels[:, None] != labels[None, :]).astype(float)


def absdiff_rdm(values):
    v = np.asarray(values, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def _upper(M):
    return M[np.triu_indices_from(M, k=1)]


def within_group_dissimilarity(neural_rdm, group_labels):
    """Mean within-group dissimilarity for each group."""
    groups = sorted(set(group_labels))
    out = {}
    gl = np.asarray(group_labels, dtype=object)
    for g in groups:
        idx = np.where(gl == g)[0]
        if len(idx) < 2:
            out[g] = np.nan
            continue
        vals = [neural_rdm[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]]
        out[g] = float(np.mean(vals))
    return out


def containment_slope_test(neural_rdm, group_labels, fam_map,
                           n_perms=2000, seed=42):
    """
    Regress within-group mean dissimilarity on familiarity level across groups.
    Gradient hypothesis → NEGATIVE slope (more familiar = more consolidated).
    Permutation null shuffles identity→group assignment.
    Returns (slope, p_one_sided_negative, per_group_dict).
    """
    rng = np.random.default_rng(seed)
    gl = np.asarray(group_labels, dtype=object)
    groups = sorted(set(group_labels))
    fam = np.array([fam_map[g] for g in groups], float)

    def slope_from_labels(labels):
        wd = within_group_dissimilarity(neural_rdm, labels)
        y = np.array([wd[g] for g in groups], float)
        ok = ~np.isnan(y)
        if ok.sum() < 2 or np.std(fam[ok]) == 0:
            return np.nan
        return float(np.polyfit(fam[ok], y[ok], 1)[0])

    obs = slope_from_labels(gl)
    null = np.array([slope_from_labels(rng.permutation(gl)) for _ in range(n_perms)])
    null = null[~np.isnan(null)]
    p_neg = float((null <= obs).mean()) if len(null) else np.nan  # one-sided: slope < 0
    wd = within_group_dissimilarity(neural_rdm, gl)
    return obs, p_neg, wd


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def run_region(region, n_permutations=2000, window=(0.300, 0.500),
               normalization='soft', min_reps=7, output_root='./rsa_group_separation'):
    cfg = RSAConfig(
        region=region,
        session=None,                       # pseudo-population
        window=window,
        min_epoch_duration=1.0,
        min_reps_per_monkey=min_reps,
        neural_metric='correlation',        # base; we sweep metrics below
        model_factors=['group'],            # we add familiarity/social locally
        partial_out=[],
        exclude_groups=[],
        exclude_identities=[],
        normalization=normalization,
        n_permutations=0,                   # comparisons are run explicitly below
        rdm_sort_mode='by_group',
        visual_responsiveness_filter=False,
        peak_latency_filter=False,
        peak_latency_range=(0.200, 0.500),
        peak_latency_search_window=(0.0, 1.00),
        save_plots=False,
        save_dir=output_root,
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)
    if cfg.exclude_groups:
        df = df[~df['MonkeyGroup'].isin(cfg.exclude_groups)].reset_index(drop=True)
    if cfg.exclude_identities:
        df = df[~df['MonkeyName'].isin(cfg.exclude_identities)].reset_index(drop=True)
    df = filter_visually_responsive(df, cfg)
    df = apply_neuron_filters(df, cfg)

    # Build the pseudo-population once (gives identities, rate matrices, group RDM)
    result = run_rsa_pseudopop(df, info_df, cfg)
    ids = result['identities']
    groups = list(result['model_labels']['group'])      # group name per identity
    missing = sorted({g for g in groups if g not in FAMILIARITY_LEVEL or g not in SOCIAL_GROUP})
    if missing:
        raise KeyError(f"Groups not in FAMILIARITY_LEVEL/SOCIAL_GROUP maps: {missing} "
                       f"— edit the maps at the top of the file.")
    rate_norm = result['rate_matrix']                   # normalized (cfg.normalization)
    rate_raw = result['raw_rate_matrix']                # pre-normalization

    # model RDMs (built from the actual identity order)
    fam_levels = [FAMILIARITY_LEVEL[g] for g in groups]
    fam_rdm = absdiff_rdm(fam_levels)
    group_rdm = result['model_rdms']['group']
    social_labels = [SOCIAL_GROUP[g] for g in groups]
    social_rdm = categorical_rdm(social_labels)
    mean_rate = rate_raw.mean(axis=1)                   # per-identity mean firing
    rate_rdm = absdiff_rdm(mean_rate)                   # |mean_rate_i - mean_rate_j|

    print(f"\n{'═'*74}\n  RSA GROUP SEPARATION — region = {region}\n{'═'*74}")
    print(f"  {len(ids)} identities, {result['n_neurons']} neurons, "
          f"window {int(window[0]*1000)}-{int(window[1]*1000)} ms, norm={normalization}")
    for g in sorted(set(groups)):
        print(f"    {g:<16s} fam={FAMILIARITY_LEVEL[g]}  social={SOCIAL_GROUP[g]}  "
              f"(n={groups.count(g)})")

    rows = []
    per_metric = {}
    for metric in METRICS:
        neural = build_neural_rdm(rate_norm, metric=metric)

        fam_marg = compare_rdms(neural, fam_rdm, n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        soc_marg = compare_rdms(neural, social_rdm, n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        fam_g   = compare_rdms_partial(neural, fam_rdm, [group_rdm],
                                       n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        grp_f   = compare_rdms_partial(neural, group_rdm, [fam_rdm],
                                       n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        fam_gr  = compare_rdms_partial(neural, fam_rdm, [group_rdm, rate_rdm],
                                       n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        slope, p_slope, wd = containment_slope_test(
            neural, groups, FAMILIARITY_LEVEL, n_perms=n_permutations, seed=cfg.rng_seed)

        per_metric[metric] = dict(fam_marg=fam_marg, soc_marg=soc_marg, fam_g=fam_g,
                                  grp_f=grp_f, fam_gr=fam_gr,
                                  slope=slope, p_slope=p_slope, within=wd, neural=neural)

        print(f"\n  ── metric = {metric} {'(removes magnitude)' if metric=='correlation' else '(keeps magnitude)'} ──")
        _line('familiarity (marginal)', fam_marg)
        _line('social 3-level (marginal)', soc_marg)
        _line('familiarity | group', fam_g, note='graded code beyond categories')
        _line('group | familiarity', grp_f)
        _line('familiarity | group + mean_rate', fam_gr, note='survives = NOT just gain')
        sl_str = f"slope={slope:+.4f}  p(neg)={p_slope:.4f}"
        print(f"    {'containment slope (within vs fam)':38s} {sl_str}   "
              f"(neg = more familiar → more consolidated)")
        print(f"       within-group dissim: " +
              "  ".join(f"{g[:8]}={wd[g]:.3f}" for g in sorted(wd)))

        rows.append((region, metric, fam_g['rho'], fam_g['p_val'],
                     fam_gr['rho'], fam_gr['p_val'], slope, p_slope))

    _plot_region(region, per_metric, groups,
                 os.path.join(output_root, f'rsa_group_separation_{region}.png'))
    return per_metric, rows


def _line(name, comp, note=''):
    rho = comp['rho']
    p = comp['p_val']
    p_str = f"p={p:.4f}" if p is not None else "p=NA"
    star = _star(p)
    tail = f"   <- {note}" if note else ''
    print(f"    {name:38s} rho={rho:+.4f}  {p_str} {star}{tail}")


def _star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))


def _plot_region(region, per_metric, groups, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: partial rho for each metric
    ax = axes[0]
    labels = ['fam | group', 'fam | group+rate', 'group | fam']
    width = 0.38
    x = np.arange(len(labels))
    for mi, metric in enumerate(METRICS):
        pm = per_metric[metric]
        vals = [pm['fam_g']['rho'], pm['fam_gr']['rho'], pm['grp_f']['rho']]
        ps = [pm['fam_g']['p_val'], pm['fam_gr']['p_val'], pm['grp_f']['p_val']]
        color = '#48c' if metric == 'correlation' else '#c44'
        bars = ax.bar(x + (mi - 0.5) * width, vals, width,
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                      label=f'{metric}')
        for xi, v, p in zip(x + (mi - 0.5) * width, vals, ps):
            ax.text(xi, v + (0.005 if v >= 0 else -0.02), _star(p),
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')
    ax.axhline(0, color='gray', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('partial Spearman rho')
    ax.set_title(f'{region}: familiarity RSA by ruler\n'
                 'correlation removes magnitude, euclidean keeps it')
    ax.legend(fontsize=9)

    # right: within-group dissimilarity vs familiarity level (euclidean = magnitude ruler)
    ax = axes[1]
    for metric in METRICS:
        pm = per_metric[metric]
        wd = pm['within']
        gs = sorted(wd, key=lambda g: FAMILIARITY_LEVEL[g])
        xv = [FAMILIARITY_LEVEL[g] for g in gs]
        yv = [wd[g] for g in gs]
        color = '#48c' if metric == 'correlation' else '#c44'
        ax.plot(xv, yv, 'o-', color=color, lw=2, markersize=8,
                label=f"{metric} (slope p={pm['p_slope']:.3f})")
        for g in gs:
            ax.annotate(g[:8], (FAMILIARITY_LEVEL[g], wd[g]), fontsize=7,
                        textcoords='offset points', xytext=(4, 4))
    ax.set_xlabel('familiarity level (0=stranger, 2=home)')
    ax.set_ylabel('within-group mean dissimilarity')
    ax.set_title(f'{region}: consolidation gradient\n'
                 '(gradient → downward slope: familiar = consolidated)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


def main():
    N_PERMUTATIONS = 2000
    WINDOW = (0.300, 0.500)       # match your RSA pipeline window
    NORMALIZATION = 'soft'        # NOT zscore — that removes the magnitude signal
    MIN_REPS = 7
    OUTPUT_ROOT = './rsa_group_separation'
    REGIONS = ['AMG', 'ER']

    all_rows = []
    for region in REGIONS:
        _, rows = run_region(region, n_permutations=N_PERMUTATIONS, window=WINDOW,
                             normalization=NORMALIZATION, min_reps=MIN_REPS,
                             output_root=OUTPUT_ROOT)
        all_rows.extend(rows)

    print(f"\n\n{'═'*86}\n  CROSS-REGION SUMMARY (RSA familiarity)\n{'═'*86}")
    print(f"  {'region':<6} {'metric':<12} {'fam|grp':>9} {'p':>8} "
          f"{'fam|grp+rate':>13} {'p':>8} {'slope':>9} {'p_neg':>8}")
    print('-' * 86)
    for (region, metric, fg_r, fg_p, fgr_r, fgr_p, slope, p_slope) in all_rows:
        fg_p = np.nan if fg_p is None else fg_p
        fgr_p = np.nan if fgr_p is None else fgr_p
        print(f"  {region:<6} {metric:<12} {fg_r:>9.3f} {fg_p:>8.4f} "
              f"{fgr_r:>13.3f} {fgr_p:>8.4f} {slope:>9.4f} {p_slope:>8.4f}")


if __name__ == '__main__':
    main()
