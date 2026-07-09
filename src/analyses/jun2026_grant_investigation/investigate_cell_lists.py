"""
Cell-list investigation for the Aim 3c allocentric linear-regression result.

Question
--------
The PI's grant reports a significant allocentric effect (neurons encoding focal
monkeys' social-relationship patterns): total permutation p ~ 0.008 on his
74-cell list. Re-running the *identical* pipeline on the SI-sorted, KW-passed
38-cell list does NOT reproduce it (p ~ 0.2). Why?

The two cell lists differ on several axes at once:
  1. cell count            (74 vs 38)
  2. unit type             (his = 37 sorted + 37 multiunit; KW = all sorted)
  3. sorting pipeline      (his manual/legacy sort vs SpikeInterface sort)
  4. QC / window selection (KW = Kruskal-Wallis-significant, stimulus-locked
                            windows; his = wider/later fixed windows)

This script isolates factors one at a time by running the same permutation test
across a set of named conditions and printing a comparison table. The comparable
quantity across conditions is the permutation p-value, NOT the raw summed R^2
(the summed statistic scales trivially with cell count; the permutation null is
summed over the same cells, so p controls for count).

A methodological knob is also swept: SUB_TO[110E -> 94B] = 5 (grant's hardcoded
transcription typo, reproduces his exact numbers) vs 15 (corrected per the
submission feature df). See common.build_beh_matrices.

Run:  python investigate_cell_lists.py            # full, NPERM=10000
      NPERM=500 python investigate_cell_lists.py  # fast smoke test
"""

import os
import time
import numpy as np
import pandas as pd

from common import (
    load_data, load_data_kw, build_valid_k_per_source, build_y_per,
    vec_r2, scalar_r2, build_beh_matrices, SUB_TO_GRANT_TYPO, SUB_TO_CORRECTED,
    NMONKEYS, SUBJECT, MONKEY_NAME, BEH_NAMES, AFF_TO, AFF_FROM,
)

# ------------------------------------------------------------------ config
THRESH = 0.5
NPERM = int(os.environ.get('NPERM', 10000))
SEED = 20251121
OUT_CSV = os.path.join(os.path.dirname(__file__), 'investigation_results.csv')


# ------------------------------------------------------------------ helpers
def subset_mask(df, subset):
    """Boolean mask over rows: 'all' | 'sorted' (Cell name has 'Unit') | 'multiunit'."""
    if subset == 'all':
        return np.ones(len(df), bool)
    has_unit = df['Cell'].astype(str).str.contains('Unit', case=False, na=False).to_numpy()
    return has_unit if subset == 'sorted' else ~has_unit


def run_condition(X_mean, all_beh, thresh=THRESH, nperm=NPERM, seed=SEED):
    """Observed pass + permutation test on one (cells x matrices) condition.

    Uses the exact pipeline from replicate_analysis.py so numbers stay
    comparable to the grant. Each condition gets a fresh RNG seeded identically,
    so the null draws are matched across conditions.
    """
    ncells = X_mean.shape[0]
    valid_k = build_valid_k_per_source()
    y_per = build_y_per(valid_k, all_beh)
    X_per_source = {s: X_mean[:, valid_k[s]].copy() for s in range(NMONKEYS)}

    # secondary regression target: subject's affiliation (to+from) per source monkey
    y_sec = np.array([AFF_TO[SUBJECT, sm] + AFF_FROM[SUBJECT, sm] for sm in range(NMONKEYS)])

    # ---- observed ----
    obs_src = np.zeros(NMONKEYS)
    obs_beh = np.zeros(6)
    obs_grid = np.zeros((6, NMONKEYS))
    obs_nsig = 0
    for ib in range(6):
        for s in range(NMONKEYS):
            r2 = vec_r2(X_per_source[s], y_per[(ib, s)])
            mask = r2 > thresh
            if mask.any():
                c = np.abs(r2[mask]).sum()
                obs_src[s] += c
                obs_beh[ib] += c
                obs_grid[ib, s] += c
                obs_nsig += int(mask.sum())
    obs_total = obs_src.sum()
    obs_r2beh = scalar_r2(obs_src, y_sec)

    # ---- permutation ----
    rng = np.random.default_rng(seed)
    nless_src = np.zeros(NMONKEYS, int)
    nless_total = nless_nsig = nless_r2beh = 0
    nless_beh = np.zeros(6, int)
    nless_grid = np.zeros((6, NMONKEYS), int)
    for _ in range(nperm):
        r_src = np.zeros(NMONKEYS)
        r_beh = np.zeros(6)
        r_grid = np.zeros((6, NMONKEYS))
        nrs = 0
        for ib in range(6):
            for s in range(NMONKEYS):
                Xs = X_per_source[s]
                y = y_per[(ib, s)]
                perm = np.argsort(rng.random((ncells, Xs.shape[1])), axis=1)
                r2 = vec_r2(np.take_along_axis(Xs, perm, axis=1), y)
                mask = r2 > thresh
                if mask.any():
                    c = np.abs(r2[mask]).sum()
                    r_src[s] += c
                    r_beh[ib] += c
                    r_grid[ib, s] += c
                    nrs += int(mask.sum())
        rt = r_src.sum()
        nless_src += (r_src < obs_src)
        nless_beh += (r_beh < obs_beh)
        nless_grid += (r_grid < obs_grid)
        nless_total += int(rt < obs_total)
        nless_nsig += int(nrs < obs_nsig)
        nless_r2beh += int(scalar_r2(r_src, y_sec) < obs_r2beh)

    p = lambda nless: 1.0 - nless / nperm
    return dict(
        ncells=ncells, obs_total=obs_total, p_total=p(nless_total),
        obs_nsig=obs_nsig, p_nsig=p(nless_nsig),
        obs_r2beh=obs_r2beh, p_r2beh=p(nless_r2beh),
        obs_src=obs_src, p_src=p(nless_src),
        obs_beh=obs_beh, p_beh=p(nless_beh),
        obs_grid=obs_grid, p_grid=p(nless_grid),
    )


# ------------------------------------------------------------------ conditions
def build_conditions():
    """Return list of (key, description, X_mean, sub_to_value)."""
    Xh74, dfh74 = load_data(ncells=74)            # his exact baseline slice
    m_sorted = subset_mask(dfh74, 'sorted')
    m_multi = subset_mask(dfh74, 'multiunit')
    Xkw, dfkw = load_data_kw()                    # 38 SI-sorted KW combos

    return [
        ('A_his74_grant',  'his 74 cells (37 sort + 37 multi), GRANT matrix SUB=5   [exact baseline]',
         Xh74, SUB_TO_GRANT_TYPO),
        ('B_his74_corr',   'his 74 cells, corrected matrix SUB=15                    [matrix-fix only]',
         Xh74, SUB_TO_CORRECTED),
        ('C_his_sorted37', 'his 37 SORTED single units, corrected matrix',
         Xh74[m_sorted], SUB_TO_CORRECTED),
        ('D_his_multi37',  'his 37 MULTIUNITS, corrected matrix',
         Xh74[m_multi], SUB_TO_CORRECTED),
        ('E_kw38_corr',    'KW 38 SI-sorted units, corrected matrix                  [the null result]',
         Xkw, SUB_TO_CORRECTED),
        ('F_kw38_grant',   'KW 38 SI-sorted units, GRANT matrix SUB=5',
         Xkw, SUB_TO_GRANT_TYPO),
    ]


# ------------------------------------------------------------------ main
def main():
    print(f"NPERM={NPERM}  THRESH={THRESH}  SEED={SEED}\n")
    conditions = build_conditions()
    results = {}
    t0 = time.time()
    for key, desc, X, sub_val in conditions:
        all_beh = build_beh_matrices(sub_val)
        t1 = time.time()
        r = run_condition(X, all_beh)
        r['desc'] = desc
        results[key] = r
        print(f"[{time.time()-t1:5.1f}s] {key:16s} n={r['ncells']:2d}  "
              f"sumR2={r['obs_total']:7.3f}  p_total={r['p_total']:.4f}  "
              f"nsig={r['obs_nsig']:3d} p={r['p_nsig']:.4f}")
    print(f"\nTotal {time.time()-t0:.1f}s\n")

    # ---- summary table ----
    print("=" * 100)
    print("SUMMARY  (p_total is the comparable quantity across conditions; sumR2 scales with n)")
    print("=" * 100)
    hdr = f"{'condition':16s} {'n':>3s} {'sumR2':>8s} {'sumR2/n':>8s} {'p_total':>8s} " \
          f"{'nsig':>5s} {'p_nsig':>7s} {'r2_beh':>7s} {'p_r2beh':>8s}   description"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for key, desc, _, _ in conditions:
        r = results[key]
        star = '*' if r['p_total'] < 0.05 else ' '
        print(f"{key:16s} {r['ncells']:3d} {r['obs_total']:8.3f} "
              f"{r['obs_total']/r['ncells']:8.3f} {r['p_total']:7.4f}{star} "
              f"{r['obs_nsig']:5d} {r['p_nsig']:7.4f} {r['obs_r2beh']:7.4f} {r['p_r2beh']:8.4f}   {desc}")
        rows.append(dict(condition=key, n=r['ncells'], sumR2=r['obs_total'],
                         sumR2_per_cell=r['obs_total']/r['ncells'], p_total=r['p_total'],
                         nsig=r['obs_nsig'], p_nsig=r['p_nsig'],
                         r2_beh=r['obs_r2beh'], p_r2beh=r['p_r2beh'], description=desc))

    # ---- per-source p-values (which focal monkeys drive it) ----
    print("\n" + "=" * 100)
    print("PER-SOURCE permutation p (uncorrected). Bonferroni threshold for 10 sources = 0.005")
    print("=" * 100)
    print(f"{'condition':16s} " + "  ".join(f"{m:>6s}" for m in MONKEY_NAME))
    for key, _, _, _ in conditions:
        r = results[key]
        cells = []
        for s in range(NMONKEYS):
            ps = r['p_src'][s]
            tag = '*' if ps < 0.05 else ' '
            cells.append(f"{ps:5.3f}{tag}")
        print(f"{key:16s} " + " ".join(f"{c:>6s}" for c in cells))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved summary table -> {OUT_CSV}")

    make_figure(conditions, results)


UNIT_TYPE = {  # for figure coloring
    'A_his74_grant': 'mixed', 'B_his74_corr': 'mixed',
    'C_his_sorted37': 'sorted', 'D_his_multi37': 'multiunit',
    'E_kw38_corr': 'sorted', 'F_kw38_grant': 'sorted',
}
UNIT_COLOR = {'sorted': '#4C72B0', 'multiunit': '#C44E52', 'mixed': '#8172B3'}


def make_figure(conditions, results):
    """One-panel figure: -log10(total permutation p) per condition, colored by
    unit type, with the p=0.05 line. Shows the effect lives in multiunit / mixed
    lists and vanishes in every all-sorted list."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from common import set_plot_style
    set_plot_style()

    keys = [k for k, *_ in conditions]
    floor = 1.0 / NPERM
    neglogp = [-np.log10(max(results[k]['p_total'], floor)) for k in keys]
    colors = [UNIT_COLOR[UNIT_TYPE[k]] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(keys)), neglogp, color=colors, edgecolor='#222', linewidth=0.6)
    ax.axhline(-np.log10(0.05), ls='--', color='#444', lw=1)
    ax.text(len(keys) - 0.4, -np.log10(0.05) + 0.05, 'p = 0.05', ha='right', va='bottom', fontsize=8, color='#444')
    for i, k in enumerate(keys):
        p = results[k]['p_total']
        lab = f'p<{floor:g}' if p < floor else f'p={p:.4f}'
        ax.text(i, neglogp[i] + 0.05, f"{lab}\nn={results[k]['ncells']}", ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('-log10( total permutation p )')
    ax.set_ylim(0, max(neglogp) * 1.25 + 0.3)
    ax.set_title('Allocentric effect is carried by multiunit activity, not sorted single units')
    handles = [plt.Rectangle((0, 0), 1, 1, color=UNIT_COLOR[t]) for t in ['sorted', 'multiunit', 'mixed']]
    ax.legend(handles, ['sorted single units', 'multiunit', 'mixed (sort+multi)'],
              loc='upper right', frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'figures', 'fig5_unit_type.png')
    fig.savefig(out, dpi=150)
    print(f"Saved figure -> {out}")


if __name__ == '__main__':
    main()
