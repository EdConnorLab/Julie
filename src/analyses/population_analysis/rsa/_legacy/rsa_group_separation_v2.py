# rsa_group_separation.py
"""
RSA-side group-separation / familiarity test.

Question:
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
from analyses.population_analysis.rsa.core.visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters
from analyses.population_analysis.rsa.core.rsa_config import RSAConfig
from analyses.population_analysis.rsa.core.rsa_core import (run_rsa_pseudopop, build_neural_rdm,
                      compare_rdms, compare_rdms_partial, normalize_rates)

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
    Permutation null shuffles identity→group assignment.
    Two-sided test: is the slope more extreme (either direction) than chance?
    (Sign is reported separately; negative = familiar more consolidated,
     positive = familiar more spread apart.)
    Returns (slope, p_two_sided, per_group_dict).
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
    p_two = float((np.abs(null) >= abs(obs)).mean()) if len(null) else np.nan  # two-sided
    wd = within_group_dissimilarity(neural_rdm, gl)
    return obs, p_two, wd


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS: pair-level slope with covariate control, + variance-quartile sweep
# ═══════════════════════════════════════════════════════════════════════

def containment_slope_pairlevel(neural_rdm, group_labels, fam_map,
                                cov=None, n_perms=2000, seed=42):
    """
    Pair-level version of the slope (better powered, supports a covariate).
    Regress each WITHIN-group pair's dissimilarity on its group's familiarity
    level, optionally controlling for a per-identity covariate `cov` (entered per
    pair as the mean of the two identities' values).
    Two-sided permutation test on the familiarity coefficient.
    Returns (fam_coefficient, p_two_sided).

    `cov` examples: per-identity mean firing rate (gain control) or per-identity
    trial count (noise/reps control). If the coefficient survives the covariate,
    the effect is not explained by that nuisance.
    """
    gl0 = np.asarray(group_labels, dtype=object)
    n = len(gl0)
    iu = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cov = None if cov is None else np.asarray(cov, float)

    def fam_coef(labels):
        lab = np.asarray(labels, dtype=object)
        d, f, c = [], [], []
        for (i, j) in iu:
            if lab[i] == lab[j]:
                d.append(neural_rdm[i, j])
                f.append(fam_map[lab[i]])
                if cov is not None:
                    c.append(0.5 * (cov[i] + cov[j]))
        need = 3 + (1 if cov is not None else 0)
        if len(d) < need or np.std(f) == 0:
            return np.nan
        cols = [np.ones(len(d)), np.asarray(f, float)]
        if cov is not None:
            cc = np.asarray(c, float)
            if np.std(cc) == 0:
                return np.nan
            cols.append(cc)
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, np.asarray(d, float), rcond=None)
        return float(beta[1])

    rng = np.random.default_rng(seed)
    obs = fam_coef(gl0)
    null = np.array([fam_coef(rng.permutation(gl0)) for _ in range(n_perms)])
    null = null[~np.isnan(null)]
    p_two = float((np.abs(null) >= abs(obs)).mean()) if len(null) else np.nan  # two-sided
    return obs, p_two


def slope_across_quartiles(raw_matrix, group_labels, fam_map, metric,
                           soft_const, quartiles=(0.25, 0.50, 0.75, 1.0),
                           n_perms=1000, seed=42):
    """
    Recompute the pair-level familiarity slope using only the top-Q% of neurons
    by variance (across identities). If the effect holds at top-25% and persists
    with more neurons it is distributed; if it only appears once sparse
    high-variance cells are included, it is carried by a few neurons
    (e.g. G942/48Z in ER). Two-sided p from containment_slope_pairlevel.
    """
    std = raw_matrix.std(axis=0, ddof=0)
    order = np.argsort(std)[::-1]
    out = []
    for q in quartiles:
        k = max(1, int(np.ceil(len(order) * q)))
        sub = normalize_rates(raw_matrix[:, order[:k]], method='soft', soft_const=soft_const)
        rdm = build_neural_rdm(sub, metric=metric)
        coef, p = containment_slope_pairlevel(rdm, group_labels, fam_map,
                                              cov=None, n_perms=n_perms, seed=seed)
        out.append(dict(q=q, k=k, coef=coef, p=p))
    return out


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

    # per-identity trial counts (reps) for the noise control; balanced design → ~flat
    try:
        reps_series = (df.drop_duplicates(['session', 'TaskField'])
                         .groupby('MonkeyName').size())
        reps = np.array([float(reps_series.get(m, np.nan)) for m in ids])
        if not np.isfinite(reps).all():
            reps = None
    except Exception:
        reps = None

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

        # ── robustness on the slope (the correctly-signed ER headline) ──
        pl_coef, pl_p = containment_slope_pairlevel(
            neural, groups, FAMILIARITY_LEVEL, cov=None,
            n_perms=n_permutations, seed=cfg.rng_seed)
        plr_coef, plr_p = containment_slope_pairlevel(
            neural, groups, FAMILIARITY_LEVEL, cov=mean_rate,
            n_perms=n_permutations, seed=cfg.rng_seed)
        if reps is not None:
            plreps_coef, plreps_p = containment_slope_pairlevel(
                neural, groups, FAMILIARITY_LEVEL, cov=reps,
                n_perms=n_permutations, seed=cfg.rng_seed)
        else:
            plreps_coef, plreps_p = np.nan, np.nan
        # variance-quartile sweep only on the magnitude ruler (where ER lives)
        quart = (slope_across_quartiles(rate_raw, groups, FAMILIARITY_LEVEL,
                                        'euclidean', cfg.soft_normalize_const,
                                        n_perms=max(500, n_permutations // 2),
                                        seed=cfg.rng_seed)
                 if metric == 'euclidean' else None)

        per_metric[metric] = dict(fam_marg=fam_marg, soc_marg=soc_marg, fam_g=fam_g,
                                  grp_f=grp_f, fam_gr=fam_gr,
                                  slope=slope, p_slope=p_slope, within=wd, neural=neural,
                                  pl_coef=pl_coef, pl_p=pl_p,
                                  plr_coef=plr_coef, plr_p=plr_p,
                                  plreps_coef=plreps_coef, plreps_p=plreps_p,
                                  quart=quart)

        print(f"\n  ── metric = {metric} {'(removes magnitude)' if metric=='correlation' else '(keeps magnitude)'} ──")
        _line('familiarity (marginal)', fam_marg)
        _line('social 3-level (marginal)', soc_marg)
        _line('familiarity | group', fam_g, note='graded code beyond categories')
        _line('group | familiarity', grp_f)
        _line('familiarity | group + mean_rate', fam_gr, note='survives = NOT just gain')
        sl_str = f"slope={slope:+.4f}  p(2-sided)={p_slope:.4f}"
        print(f"    {'within-group slope vs familiarity':38s} {sl_str}   "
              f"(sign: + = familiar more spread, - = familiar more similar)")
        print(f"       within-group dissim: " +
              "  ".join(f"{g[:8]}={wd[g]:.3f}" for g in sorted(wd)))
        print(f"    {'slope pair-level (plain)':38s} coef={pl_coef:+.4f}  p(2-sided)={pl_p:.4f}")
        print(f"    {'slope pair-level | mean_rate':38s} coef={plr_coef:+.4f}  p(2-sided)={plr_p:.4f}"
              f"   <- survives = not a gain artifact")
        reps_str = (f"coef={plreps_coef:+.4f}  p(2-sided)={plreps_p:.4f}"
                    if reps is not None else "reps unavailable (no TaskField/session cols)")
        print(f"    {'slope pair-level | reps':38s} {reps_str}"
              f"   <- survives = not a noise/trial-count artifact")
        if quart is not None:
            qs = "  ".join(f"top{int(qd['q']*100)}%(n={qd['k']}):{qd['coef']:+.3f}/p={qd['p']:.3f}"
                           for qd in quart)
            print(f"    {'slope across variance quartiles':38s} {qs}")
            print(f"       (holds at top-25% → distributed; only at 100% → carried by sparse cells)")

        rows.append((region, metric, fam_g['rho'], fam_g['p_val'],
                     fam_gr['rho'], fam_gr['p_val'], slope, p_slope,
                     plr_coef, plr_p))

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    # third: consolidation slope across variance quartiles (euclidean ruler)
    ax = axes[2]
    quart = per_metric.get('euclidean', {}).get('quart')
    if quart:
        xq = [int(q['q'] * 100) for q in quart]
        cq = [q['coef'] for q in quart]
        ax.plot(xq, cq, 'o-', color='#c44', lw=2, markersize=8)
        for q in quart:
            ax.annotate(f"p={q['p']:.3f}", (int(q['q'] * 100), q['coef']),
                        fontsize=7, textcoords='offset points', xytext=(4, 4))
        ax.axhline(0, color='gray', lw=1, ls=':')
        ax.set_xlabel('% of neurons kept (top by variance)')
        ax.set_ylabel('pair-level consolidation coef')
        ax.set_title(f'{region}: is the gradient distributed?\n'
                     '(holds at top-25% = distributed; only at 100% = sparse)')
    else:
        ax.axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


def plot_slide_familiarity(region_results, out_path):
    """
    Slide-D figure: familiarity|group partial correlation, per region, both metrics.

    For each region (AMG, ER) two bars: correlation distance and euclidean distance.
    Message:
      - AMG significant under BOTH metrics  -> robust, not magnitude-dependent
      - ER significant under euclidean only -> familiarity effect carried by firing-rate magnitude
    Sign also shows the opposite-direction story (AMG positive, ER negative).
    No +rate bars and no slope statistic (slope test is one-sided for consolidation
    and is misleading for AMG's positive-direction effect).

    region_results : dict {region: per_metric_dict}  (per_metric from run_region)
    """
    regions = ['AMG', 'ER']
    metric_color = {'correlation': '#6aa9e0', 'euclidean': '#d85a3b'}
    width = 0.34

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(regions))

    for mi, metric in enumerate(METRICS):           # ['correlation', 'euclidean']
        vals, ps = [], []
        for r in regions:
            comp = region_results[r][metric]['fam_g']
            vals.append(comp['rho'])
            ps.append(comp['p_val'])
        xpos = x + (mi - 0.5) * width
        ax.bar(xpos, vals, width, color=metric_color[metric],
               edgecolor='black', linewidth=0.8,
               label=f'{metric} distance')
        for xi, v, p in zip(xpos, vals, ps):
            off = 0.012 if v >= 0 else -0.012
            ax.text(xi, v + off, _star(p), ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=13, fontweight='bold')

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=13)
    ax.set_ylabel('familiarity\u2013geometry correlation\n'
                  '(partial Spearman \u03c1, group partialled out)', fontsize=11)
    ax.set_title('Familiarity is encoded beyond group identity', fontsize=13, fontweight='bold')

    # direction annotations (sign meaning, in plain language)
    allv = []
    for metric in METRICS:
        for r in regions:
            allv.append(region_results[r][metric]['fam_g']['rho'])
    ymax, ymin = max(allv), min(allv)
    span = (ymax - ymin) if ymax != ymin else 1.0
    pad = 0.05 * span
    ax.text(0, ymax + 3 * pad, 'familiar = spread apart',
            ha='center', fontsize=9.5, style='italic', color='#444')
    ax.text(1, ymin - 3 * pad, 'familiar = more similar',
            ha='center', fontsize=9.5, style='italic', color='#444')
    ax.set_ylim(ymin - 6 * pad, ymax + 6 * pad)

    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Slide figure saved: {out_path}")


def main():
    N_PERMUTATIONS = 2000
    WINDOW = (0.300, 0.500)       # match your RSA pipeline window
    NORMALIZATION = 'soft'        # NOT zscore — that removes the magnitude signal
    MIN_REPS = 7
    OUTPUT_ROOT = './rsa_group_separation'
    REGIONS = ['AMG', 'ER']

    all_rows = []
    region_results = {}
    for region in REGIONS:
        per_metric, rows = run_region(region, n_permutations=N_PERMUTATIONS, window=WINDOW,
                                      normalization=NORMALIZATION, min_reps=MIN_REPS,
                                      output_root=OUTPUT_ROOT)
        region_results[region] = per_metric
        all_rows.extend(rows)

    # Slide-D summary figure: familiarity|group per region, both metrics
    if all(r in region_results for r in ('AMG', 'ER')):
        plot_slide_familiarity(
            region_results,
            os.path.join(OUTPUT_ROOT, 'slide_familiarity_both_regions.png'),
        )

    print(f"\n\n{'═'*100}\n  CROSS-REGION SUMMARY (RSA familiarity)\n{'═'*100}")
    print(f"  {'region':<6} {'metric':<12} {'fam|grp':>9} {'p':>8} "
          f"{'fam|grp+rate':>13} {'p':>8} {'slope':>9} {'p_2sided':>8} "
          f"{'slope|rate':>11} {'p_2sided':>8}")
    print('-' * 100)
    for (region, metric, fg_r, fg_p, fgr_r, fgr_p, slope, p_slope, plr_c, plr_p) in all_rows:
        fg_p = np.nan if fg_p is None else fg_p
        fgr_p = np.nan if fgr_p is None else fgr_p
        print(f"  {region:<6} {metric:<12} {fg_r:>9.3f} {fg_p:>8.4f} "
              f"{fgr_r:>13.3f} {fgr_p:>8.4f} {slope:>9.4f} {p_slope:>8.4f} "
              f"{plr_c:>11.4f} {plr_p:>8.4f}")


if __name__ == '__main__':
    main()
