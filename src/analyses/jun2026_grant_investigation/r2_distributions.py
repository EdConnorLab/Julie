"""
R^2 distributions for the three cell lists (updated pipeline).

Lists compared
--------------
  1. GRANT - Ed's 74-cell list (first 74 of the 75-cell xlsx), mean spike COUNTS
             -> common.load_data(HIS_XLSX, ncells=74)
  2. KW     - SI-sorted, Kruskal-Wallis-passed list, rebuilt from analysis_cache
             -> spike_count_connector.load_data_from_cache('KW')
  3. ANOVA  - SI-sorted, ANOVA-passed list, rebuilt from analysis_cache
             -> spike_count_connector.load_data_from_cache('ANOVA')

All three go through the exact regression the grant pipeline thresholds: for every
(cell x source-monkey x behavior) we compute R^2 between the cell's firing profile
across the valid sink monkeys and the source monkey's behavioral profile. This
script just shows how those R^2 are distributed and how many cross a threshold --
it does NOT run the permutation test.

R^2 is scale-invariant, so mean COUNT (GRANT) vs mean count (KW/ANOVA via the
connector, value='count') are directly comparable.

Run it (PyCharm: hit Run). Outputs -> ./investigation_outputs/
  r2_distributions.png   two panels: all pooled fits, and best fit per cell
  r2_all_fits.csv        one row per (list, cell, source, behavior) fit
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# --- stable pieces from common.py ---
from common import (
    MONKEY_NAME, SUBJECT, NMONKEYS, K_TO_FULL,
    AFF_TO, SUB_TO, AGN_TO, vec_r2, build_valid_k_per_source, set_plot_style,
    load_data, HIS_XLSX,
)

# =====================================================================
# CONFIG - edit these
# =====================================================================
THRESH_PRIMARY = 0.05        # your threshold: R^2 > 0.05
THRESH_GRANT = 0.50          # the grant pipeline's threshold, drawn for reference
SOURCES = 'allocentric'      # 'allocentric' (drop 81G as a source) | 'all' (10 sources)
BEHAVIORS = 'all'            # 'all' (6) or a list of indices into BEH_NAMES
GRANT_NCELLS = 74            # GRANT list: first 74 rows (None/0 for all 75)
DENSITY = True               # normalize histograms (lists have different n)
NBINS = 40
# =====================================================================

BEH_NAMES = ['aff_to', 'aff_from', 'sub_to', 'sub_from', 'agn_to', 'agn_from']
ALL_BEH = [AFF_TO, AFF_TO.T, SUB_TO, SUB_TO.T, AGN_TO, AGN_TO.T]

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'investigation_outputs')
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------- loaders
def load_lists():
    """Return [(label, X_mean, meta), ...] for GRANT, KW, ANOVA.

    GRANT comes from the xlsx (mean counts); KW/ANOVA are rebuilt from the
    analysis_cache significance pkls + the sorted spike cache via the connector.
    All X use the same 9-column order (common.MONKEY_NAME minus subject 81G).
    """
    from spike_count_connector import load_data_from_cache  # heavy import (clat); lazy

    Xg, dfg = load_data(HIS_XLSX, ncells=GRANT_NCELLS)
    mg = dfg.reset_index(drop=True)[['Cell']].copy()

    Xkw, mkw = load_data_from_cache('KW', value='count')
    Xan, man = load_data_from_cache('ANOVA', value='count')
    return [('GRANT', Xg, mg), ('KW', Xkw, mkw), ('ANOVA', Xan, man)]


# ---------------------------------------------------------------- R^2 table
def r2_long_table(X, meta, label):
    """One row per (cell, source, behavior) fit, with its R^2."""
    valid_k = build_valid_k_per_source()
    sources = range(NMONKEYS) if SOURCES == 'all' else [s for s in range(NMONKEYS) if s != SUBJECT]
    behs = range(6) if BEHAVIORS == 'all' else BEHAVIORS
    rows = []
    for s in sources:
        ks = valid_k[s]
        fulls = [K_TO_FULL[k] for k in ks]
        Xs = X[:, ks]
        for ib in behs:
            y = np.array([ALL_BEH[ib][s, f] for f in fulls], float)
            r2 = vec_r2(Xs, y)
            for ci in range(X.shape[0]):
                rows.append((label, ci, str(meta.iloc[ci]['Cell']), MONKEY_NAME[s], BEH_NAMES[ib], r2[ci]))
    return pd.DataFrame(rows, columns=['list', 'cell_idx', 'cell', 'source', 'behavior', 'r2'])


# ---------------------------------------------------------------- summary + plot
def summarize(tab):
    print(f"\n{'list':7s} {'cells':>6s} {'fits':>6s} "
          f"{'>'+str(THRESH_PRIMARY):>10s} {'%':>6s}   "
          f"{'>'+str(THRESH_GRANT):>10s} {'%':>6s}   "
          f"{'cells best>'+str(THRESH_PRIMARY):>14s}  {'best>'+str(THRESH_GRANT):>10s}")
    for lab, g in tab.groupby('list', sort=False):
        ncells = g['cell_idx'].nunique()
        nfits = len(g)
        n05 = int((g.r2 > THRESH_PRIMARY).sum())
        n50 = int((g.r2 > THRESH_GRANT).sum())
        best = g.groupby('cell_idx').r2.max()
        c05 = int((best > THRESH_PRIMARY).sum())
        c50 = int((best > THRESH_GRANT).sum())
        print(f"{lab:7s} {ncells:6d} {nfits:6d} "
              f"{n05:10d} {100*n05/nfits:5.1f}   "
              f"{n50:10d} {100*n50/nfits:5.1f}   "
              f"{c05:14d}  {c50:10d}")


COLORS = {'GRANT': '#8172B3', 'KW': '#4C72B0', 'ANOVA': '#C44E52'}


def make_figure(tab):
    set_plot_style()
    labels = list(dict.fromkeys(tab['list']))
    bins = np.linspace(0, 1, NBINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lab in labels:
        r = tab.loc[tab['list'] == lab, 'r2'].to_numpy()
        axes[0].hist(r, bins=bins, density=DENSITY, histtype='step', lw=1.8,
                     color=COLORS.get(lab, None), label=f"{lab} (n_fits={len(r)})")
    axes[0].set_title('R² of every (cell × source × behavior) fit')

    for lab in labels:
        best = tab[tab['list'] == lab].groupby('cell_idx').r2.max().to_numpy()
        axes[1].hist(best, bins=bins, density=DENSITY, histtype='step', lw=1.8,
                     color=COLORS.get(lab, None), label=f"{lab} (n_cells={len(best)})")
    axes[1].set_title('Best R² per cell (max over sources × behaviors)')

    for ax in axes:
        ax.axvline(THRESH_PRIMARY, ls='--', color='#333', lw=1)
        ax.axvline(THRESH_GRANT, ls=':', color='#333', lw=1)
        ax.text(THRESH_PRIMARY, ax.get_ylim()[1], f' {THRESH_PRIMARY}', va='top', fontsize=8)
        ax.text(THRESH_GRANT, ax.get_ylim()[1], f' {THRESH_GRANT}', va='top', fontsize=8)
        ax.set_xlabel('R²')
        ax.set_ylabel('density' if DENSITY else 'count')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"R² distributions  (sources={SOURCES}, behaviors={BEHAVIORS})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'r2_distributions.png')
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")
    return fig


def main():
    lists = load_lists()
    print("loaded:  " + "   ".join(f"{lab} {X.shape}" for lab, X, _ in lists))

    tab = pd.concat([r2_long_table(X, meta, lab) for lab, X, meta in lists], ignore_index=True)
    summarize(tab)

    csv = os.path.join(OUTDIR, 'r2_all_fits.csv')
    tab.to_csv(csv, index=False)
    print(f"Saved per-fit R² table -> {csv}  ({len(tab)} rows)")

    make_figure(tab)
    plt.show()


if __name__ == '__main__':
    main()
