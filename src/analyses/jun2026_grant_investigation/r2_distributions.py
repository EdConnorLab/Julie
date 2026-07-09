"""
STEP 2 of the cell-list investigation: R^2 distributions for three cell lists.

Lists compared
--------------
  1. GRANT  - his 74-cell list (first 74 of the 75-cell xlsx), mean spike COUNTS
  2. KW     - SI-sorted, Kruskal-Wallis-passed list, mean spike RATES (38 combos)
  3. ANOVA  - SI-sorted, ANOVA-passed list, mean spike RATES (34 combos)

For each list we compute R^2 for every (cell x source-monkey x behavior) linear
fit -- exactly the fits the grant pipeline thresholds -- and look at how the R^2
values are distributed, and how many cross a threshold.

The ANOVA list file is only a significance table (NeuronID/window/F/p) with no
firing rates, so its rates (and, for a faithful cross-check, KW's too) are pulled
from the full windowed-rates file and averaged over trials per
(neuron, monkey, window).

This script does NOT run the permutation test. It just builds the three X
matrices and shows the R^2 distributions so you can eyeball them.

Run it (PyCharm: just hit Run), inspect r2_distributions.png and the printed
table. Outputs go to ./investigation_outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# --- stable pieces from common.py (all predate any recent edits) ---
from common import (
    MONKEY_NAME, SUBJECT, NMONKEYS, K_TO_FULL,
    AFF_TO, SUB_TO, AGN_TO, vec_r2, build_valid_k_per_source, set_plot_style,
)

# =====================================================================
# CONFIG - edit these
# =====================================================================
THRESH_PRIMARY = 0.05        # your threshold: R^2 > 0.05
THRESH_GRANT = 0.50          # the grant pipeline's threshold, drawn for reference
SOURCES = 'allocentric'      # 'allocentric' (drop 81G as a source) | 'all' (10 sources)
BEHAVIORS = 'all'            # 'all' (6) or a list of indices into BEH_NAMES
DENSITY = True               # normalize histograms (lists have different n)
NBINS = 40
# Absolute paths work on the connorlab machine; relative fallback for other checkouts.
_ABS = '/home/connorlab/Documents/GitHub/Julie'
# =====================================================================

BEH_NAMES = ['aff_to', 'aff_from', 'sub_to', 'sub_from', 'agn_to', 'agn_from']
ALL_BEH = [AFF_TO, AFF_TO.T, SUB_TO, SUB_TO.T, AGN_TO, AGN_TO.T]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUTDIR = os.path.join(HERE, 'investigation_outputs')
os.makedirs(OUTDIR, exist_ok=True)


def _resolve(rel):
    """Prefer the connorlab absolute path; fall back to repo-relative."""
    a = os.path.join(_ABS, rel)
    return a if os.path.exists(a) else os.path.join(REPO, rel)


HIS_XLSX = _resolve('Cortana/old/Ed and ANOVA/used_for_R01/'
                    'zombies_spike_counts_for_all_anova_passed_time_windowed_cells_old--usedforgrant.xlsx')
KW_MEAN_PKL = _resolve('Cortana/old/generated_for_ed_si_sorted/'
                       'si_sorted_Zombies_significant_windows_pKW_passed_mean_spike_rates.pkl')
ANOVA_LIST_PKL = _resolve('Cortana/analysis_cache/'
                          'si_sorted_Zombies_significant_windows_pANOVA_passed.pkl')
FULL_RATES_PKL = _resolve('Cortana/old/generated_for_ed_si_sorted/'
                          'si_sorted_Zombies_windowed_cells_spike_rates.pkl')

ORDER9 = [m for i, m in enumerate(MONKEY_NAME) if i != SUBJECT]  # 9 non-subject Zombies


# ---------------------------------------------------------------- loaders
def load_grant74():
    """His 74-cell list -> (X_mean counts (74,9), meta). Parses the per-trial
    count-list strings and averages them, columns ordered as ORDER9."""
    import re
    df = pd.read_excel(HIS_XLSX).iloc[:74]
    cols = [df.columns.get_loc(m) for m in ORDER9]
    arr = df.values
    X = np.zeros((74, 9))
    for i in range(74):
        for k in range(9):
            nums = [int(t) for t in re.findall(r'\b\d+\b', str(arr[i, cols[k]]))]
            X[i, k] = sum(nums) / len(nums) if nums else 0.0
    meta = pd.DataFrame({'Cell': df['Cell'].astype(str).values, 'list': 'GRANT'})
    return X, meta


def _combos_from_kw():
    d = pd.read_pickle(KW_MEAN_PKL)
    d = d[d.MonkeyGroup == 'Zombies']
    return d.groupby(['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']).ngroups, \
        d[['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']].drop_duplicates(), d


def _combos_from_anova():
    d = pd.read_pickle(ANOVA_LIST_PKL)
    return d[['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']].drop_duplicates()


def _rates_for_combos(combos, full):
    """Given a (NeuronID, ws, we) combo table, average SpikeRate over trials per
    (combo, MonkeyName) and pivot to (n_combo, 9) with columns = ORDER9. Keeps
    only combos that have all 9 non-subject Zombies present."""
    fz = full[full.MonkeyGroup == 'Zombies']
    key = ['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']
    keep = fz.merge(combos, on=key, how='inner')
    means = keep.groupby(key + ['MonkeyName'])['SpikeRate'].mean().reset_index()
    piv = means.pivot_table(index=key, columns='MonkeyName', values='SpikeRate')
    piv = piv.reindex(columns=ORDER9).dropna()           # exactly the 9; drop combo if any missing
    X = piv.to_numpy(float)
    meta = piv.reset_index()[key].copy()
    meta['Cell'] = meta['NeuronID']
    return X, meta


def load_kw_and_anova():
    """Build KW and ANOVA X from the full rates file, and cross-check KW against
    the KW mean-pkl to prove the rate path is faithful."""
    full = pd.read_pickle(FULL_RATES_PKL)

    _, kw_combos, kw_meanpkl = _combos_from_kw()
    Xkw, meta_kw = _rates_for_combos(kw_combos, full)
    meta_kw['list'] = 'KW'

    Xan, meta_an = _rates_for_combos(_combos_from_anova(), full)
    meta_an['list'] = 'ANOVA'

    # cross-check: KW rates-from-full vs MeanSpikeRate in the KW pkl
    z = kw_meanpkl.copy()
    ref = z.pivot_table(index=['NeuronID', 'WindowStart_ms', 'WindowEnd_ms'],
                        columns='MonkeyName', values='MeanSpikeRate')[ORDER9]
    got = meta_kw.set_index(['NeuronID', 'WindowStart_ms', 'WindowEnd_ms'])
    common_idx = ref.index.intersection(got.index)
    ref_arr = ref.loc[common_idx].to_numpy(float)
    got_arr = Xkw[[list(got.index).index(i) for i in common_idx]]
    maxdiff = np.nanmax(np.abs(ref_arr - got_arr)) if len(common_idx) else np.nan
    print(f"[cross-check] KW rates-from-full vs KW mean-pkl: "
          f"{len(common_idx)} combos compared, max abs diff = {maxdiff:.3g}")

    return (Xkw, meta_kw), (Xan, meta_an)


# ---------------------------------------------------------------- R^2 table
def r2_long_table(X, meta, label):
    """Return a long DataFrame with one row per (cell, source, behavior) fit."""
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
                rows.append((label, ci, meta.iloc[ci]['Cell'], MONKEY_NAME[s], BEH_NAMES[ib], r2[ci]))
    return pd.DataFrame(rows, columns=['list', 'cell_idx', 'cell', 'source', 'behavior', 'r2'])


# ---------------------------------------------------------------- summary + plot
def summarize(tab):
    print(f"\n{'list':7s} {'cells':>6s} {'fits':>6s} "
          f"{'>'+str(THRESH_PRIMARY):>10s} {'%':>6s}   "
          f"{'>'+str(THRESH_GRANT):>10s} {'%':>6s}   "
          f"{'cells w/ best>'+str(THRESH_PRIMARY):>16s}  {'best>'+str(THRESH_GRANT):>10s}")
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
              f"{c05:16d}  {c50:10d}")


COLORS = {'GRANT': '#8172B3', 'KW': '#4C72B0', 'ANOVA': '#C44E52'}


def make_figure(tab):
    set_plot_style()
    labels = list(dict.fromkeys(tab['list']))
    bins = np.linspace(0, 1, NBINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel A: all pooled fits
    for lab in labels:
        r = tab.loc[tab['list'] == lab, 'r2'].to_numpy()
        axes[0].hist(r, bins=bins, density=DENSITY, histtype='step', lw=1.8,
                     color=COLORS[lab], label=f"{lab} (n_fits={len(r)})")
    axes[0].set_title('R² of every (cell × source × behavior) fit')

    # panel B: per-cell best fit
    for lab in labels:
        best = tab[tab['list'] == lab].groupby('cell_idx').r2.max().to_numpy()
        axes[1].hist(best, bins=bins, density=DENSITY, histtype='step', lw=1.8,
                     color=COLORS[lab], label=f"{lab} (n_cells={len(best)})")
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
    Xg, mg = load_grant74()
    (Xkw, mkw), (Xan, man) = load_kw_and_anova()
    print(f"\nloaded:  GRANT {Xg.shape}   KW {Xkw.shape}   ANOVA {Xan.shape}")

    tab = pd.concat([
        r2_long_table(Xg, mg, 'GRANT'),
        r2_long_table(Xkw, mkw, 'KW'),
        r2_long_table(Xan, man, 'ANOVA'),
    ], ignore_index=True)

    summarize(tab)

    csv = os.path.join(OUTDIR, 'r2_all_fits.csv')
    tab.to_csv(csv, index=False)
    print(f"Saved per-fit R² table -> {csv}  ({len(tab)} rows)")

    make_figure(tab)
    plt.show()


if __name__ == '__main__':
    main()
