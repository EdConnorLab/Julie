# familiarity_models_compare.py
"""
Marginal familiarity RSA under TWO familiarity models, side by side, so you can
decide which framing to present. No group partialling — this is the plain
"does neural geometry track familiarity?" question.

Two models
----------
GRADED  (3 levels):  home=2, neighbor=1, unfamiliar=0
    pairwise model dissimilarity = |level_i - level_j|
    (two home -> 0 ; home+unfamiliar -> 2 ; neighbor+unfamiliar -> 1 ; ...)

BINARY  (2 levels):  familiar = {home, neighbor} ; unfamiliar = {Instigators, Stranger Things}
    pairwise model dissimilarity = 0 if same class, 1 if different class

For each model we sweep BOTH neural rulers (correlation- and euclidean-distance
neural RDMs) and compare each to the model RDM with a marginal (not partial)
Spearman RSA + permutation test.

Outputs 4 PNGs into OUTPUT_ROOT:
    familiarity_bars_graded.png   - AMG/ER x {correlation,euclidean}, GRADED model
    familiarity_bars_binary.png   - AMG/ER x {correlation,euclidean}, BINARY model
    model_rdm_graded.png          - the graded model RDM (identities ordered by group)
    model_rdm_binary.png          - the binary model RDM (identities ordered by group)

Reuses all heavy machinery from rsa_group_separation_v2 / rsa_core; this script
only defines the two models and the plotting.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reuse the exact pipeline + helpers already in your v2 script
from rsa_group_separation_v2 import (
    MONKEY_INFO_PATH, FAMILIARITY_LEVEL, METRICS,
    absdiff_rdm, categorical_rdm, _star,
)
from rsa_config import RSAConfig
from rsa_core import run_rsa_pseudopop, build_neural_rdm, compare_rdms
from analyses.population_analysis.state_space.data_loading import load_and_filter
from visual_responsiveness_filter import filter_visually_responsive
from analyses.population_analysis.neuron_filters import apply_neuron_filters

# ── models ──────────────────────────────────────────────────────────────
# Graded = the FAMILIARITY_LEVEL map already in the v2 script (home2/neighbor1/unfam0)
# Binary = familiar {home, neighbor} vs unfamiliar {both stranger groups}
BINARY_CLASS = {
    'Zombies': 'familiar', 'Best Frans': 'familiar',
    'Instigators': 'unfamiliar', 'Stranger Things': 'unfamiliar',
}
# fixed group order for the RDM heatmaps (block structure visible)
GROUP_ORDER = ['Zombies', 'Best Frans', 'Instigators', 'Stranger Things']
GROUP_SHORT = {'Zombies': 'Home', 'Best Frans': 'Neighbor',
               'Instigators': 'Unfam-1', 'Stranger Things': 'Unfam-2'}
REGION_COLORS = {'correlation': '#6aa9e0', 'euclidean': '#d85a3b'}


def compute_marginal(region, n_permutations=2000, window=(0.300, 0.500),
                     normalization='soft', min_reps=7, output_root='./familiarity_models'):
    """
    Build the pseudo-population for `region`, then compute MARGINAL familiarity RSA
    (no partialling) for BOTH models x BOTH metrics.

    Returns dict:
      { 'graded': {metric: {'rho':..,'p':..}}, 'binary': {metric: {...}},
        'groups': [group per identity], 'ids': [...] }
    """
    cfg = RSAConfig(
        region=region, session=None, window=window, min_epoch_duration=1.0,
        min_reps_per_monkey=min_reps, neural_metric='correlation',
        model_factors=['group'], partial_out=[], exclude_groups=[],
        exclude_identities=[], normalization=normalization, n_permutations=0,
        rdm_sort_mode='by_group', visual_responsiveness_filter=False,
        peak_latency_filter=False, peak_latency_range=(0.200, 0.500),
        peak_latency_search_window=(0.0, 1.00), save_plots=False, save_dir=output_root,
    )
    cfg.validate()

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    df = load_and_filter(cfg)
    df = filter_visually_responsive(df, cfg)
    df = apply_neuron_filters(df, cfg)

    result = run_rsa_pseudopop(df, info_df, cfg)
    groups = list(result['model_labels']['group'])   # group name per identity
    ids = result['identities']
    rate_norm = result['rate_matrix']

    # ---- model RDMs (in the identity order returned by the pipeline) ----
    graded_levels = [FAMILIARITY_LEVEL[g] for g in groups]
    graded_rdm = absdiff_rdm(graded_levels)                       # 0/1/2 differences
    binary_labels = [BINARY_CLASS[g] for g in groups]
    binary_rdm = categorical_rdm(binary_labels)                  # 0/1

    out = {'graded': {}, 'binary': {}, 'groups': groups, 'ids': ids}
    print(f"\n=== {region}: {len(ids)} identities, {result['n_neurons']} neurons ===")
    for metric in METRICS:
        neural = build_neural_rdm(rate_norm, metric=metric)
        g = compare_rdms(neural, graded_rdm, n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        b = compare_rdms(neural, binary_rdm, n_permutations=n_permutations, rng_seed=cfg.rng_seed)
        out['graded'][metric] = {'rho': g['rho'], 'p': g['p_val']}
        out['binary'][metric] = {'rho': b['rho'], 'p': b['p_val']}
        print(f"  {metric:<12s}  graded rho={g['rho']:+.3f} p={g['p_val']:.4f}   "
              f"binary rho={b['rho']:+.3f} p={b['p_val']:.4f}")
    return out


def _bar_panel(region_results, model_key, title, out_path):
    """Two regions x two metrics bar chart for one model (marginal rho)."""
    regions = ['AMG', 'ER']
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(regions))

    allv = []
    for mi, metric in enumerate(METRICS):
        vals, ps = [], []
        for r in regions:
            d = region_results[r][model_key][metric]
            vals.append(d['rho']); ps.append(d['p']); allv.append(d['rho'])
        xpos = x + (mi - 0.5) * width
        ax.bar(xpos, vals, width, color=REGION_COLORS[metric],
               edgecolor='black', linewidth=0.8, label=f'{metric} distance')
        for xi, v, p in zip(xpos, vals, ps):
            off = 0.010 if v >= 0 else -0.010
            ax.text(xi, v + off, _star(p), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=13, fontweight='bold')

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=13)
    ax.set_ylabel('familiarity\u2013geometry correlation\n(Spearman \u03c1, marginal)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    ymax, ymin = (max(allv), min(allv)) if allv else (0.1, -0.1)
    span = (ymax - ymin) or 1.0
    pad = 0.06 * span
    if ymax > 0:
        ax.text(0, ymax + 2.2 * pad, 'familiar = spread apart',
                ha='center', fontsize=9.5, style='italic', color='#444')
    if ymin < 0:
        ax.text(1, ymin - 2.2 * pad, 'familiar = more similar',
                ha='center', fontsize=9.5, style='italic', color='#444')
    ax.set_ylim(min(ymin - 5 * pad, -0.02), max(ymax + 5 * pad, 0.02))
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}")


def _model_rdm_panel(kind, out_path):
    """
    Draw the model RDM itself (identities ordered by group). Uses a canonical
    small design: 2 home, 2 neighbor, 2 unfam-1, 2 unfam-2 = 8 slots, purely to
    SHOW the block structure of the model (values are model-defined, not data).
    """
    # one representative block layout so the picture is legible
    blocks = [('Zombies', 2), ('Best Frans', 2), ('Instigators', 2), ('Stranger Things', 2)]
    order_groups = []
    for g, n in blocks:
        order_groups += [g] * n

    if kind == 'graded':
        levels = [FAMILIARITY_LEVEL[g] for g in order_groups]
        M = absdiff_rdm(levels)
        cbar_label = 'model dissimilarity (|level difference|)'
        title = 'Graded familiarity model\n(home=2, neighbor=1, unfamiliar=0)'
    else:
        labels = [BINARY_CLASS[g] for g in order_groups]
        M = categorical_rdm(labels)
        cbar_label = 'model dissimilarity (0 = same class, 1 = different)'
        title = 'Binary familiarity model\n(familiar = home+neighbor  vs  unfamiliar)'

    fig, ax = plt.subplots(figsize=(6, 5.4))
    im = ax.imshow(M, cmap='viridis', aspect='equal')
    n = len(order_groups)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ticklabels = [GROUP_SHORT[g] for g in order_groups]
    ax.set_xticklabels(ticklabels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ticklabels, fontsize=9)

    # group boundary lines
    bnds = np.cumsum([n for _, n in blocks])[:-1] - 0.5
    for b in bnds:
        ax.axhline(b, color='white', lw=2); ax.axvline(b, color='white', lw=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}")


def main():
    OUTPUT_ROOT = './familiarity_models'
    N_PERMUTATIONS = 2000

    region_results = {}
    for region in ['AMG', 'ER']:
        region_results[region] = compute_marginal(
            region, n_permutations=N_PERMUTATIONS, output_root=OUTPUT_ROOT)

    # bar charts (one per model, side-by-side use in slides)
    _bar_panel(region_results, 'graded',
               'Graded familiarity model (home>neighbor>unfamiliar)',
               os.path.join(OUTPUT_ROOT, 'familiarity_bars_graded.png'))
    _bar_panel(region_results, 'binary',
               'Binary familiarity model (familiar vs unfamiliar)',
               os.path.join(OUTPUT_ROOT, 'familiarity_bars_binary.png'))

    # model RDM pictures
    _model_rdm_panel('graded', os.path.join(OUTPUT_ROOT, 'model_rdm_graded.png'))
    _model_rdm_panel('binary', os.path.join(OUTPUT_ROOT, 'model_rdm_binary.png'))


if __name__ == '__main__':
    main()
