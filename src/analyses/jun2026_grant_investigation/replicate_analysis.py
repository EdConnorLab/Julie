"""
Replication of allbeh_rand_var_exp_2_5_2025.py (Ed's grant script).

Reproduces his Table 1 to 4 decimal places when run on the
'_old--usedforgrant.xlsx' file with DATA_SOURCE='grant_xlsx', NCELLS=74.
Every statistic and the permutation scheme below is faithful to his script.

In PyCharm: edit the CONFIG block below, then just hit Run.


ONE REGRESSION = one (cell, source monkey s, behavior b)
--------------------------------------------------------
  x = that cell's mean firing to each *valid sink* monkey
      (sinks exclude the source s and the subject 81G  ->  n = 8;
       when s == 81G nothing extra is excluded          ->  n = 9)
  y = source s's behavior-b value toward those same sink monkeys
  R^2 = how well the cell's firing profile across monkeys linearly
        matches s's behavioral profile across monkeys (OLS w/ intercept;
        vec_r2 == sklearn r2_score on training data, verified).
A "hit" is R^2 > THRESH (0.5). The summaries aggregate |R^2| over hits.

6 behaviors b = {affiliation, submission, agonism} x {to, from};
'from' matrices are transposes of 'to'. 10 source monkeys; 81G is subject.


OBSERVED SUMMARIES (all are aggregations of the same hit set)
------------------------------------------------------------
  sumRsquared_total : sum |R^2| over ALL hits (cells x 6 beh x 10 sources)
  nsig              : COUNT of hits (not a sum)
  per source s      : sum |R^2| over the 6 behaviors and all cells, for source s
  per behavior b    : sum |R^2| over the 10 sources and all cells, for behavior b
  per grid (b,s)    : sum |R^2| for that single (behavior, source) cell (6x10 table)

  r2_beh (secondary, "correlation of affiliation to sumRsquared"):
    A second, ACROSS-MONKEY regression with 10 points (one per monkey):
      x = obs_sumR2_per_src[m]  (encoding strength attributed to monkey m as source)
      y = AFF_TO[81G,m] + AFF_FROM[81G,m]  (subject 81G's mutual affiliation w/ m)
    r2_beh = R^2 of that fit. Asks: are the monkeys whose social patterns are
    most strongly encoded the ones 81G is most affiliated with? (affiliation only;
    81G's own point sits at y=0.)


PERMUTATION NULL  -  what is shuffled
-------------------------------------
For every (behavior b, source s), and INDEPENDENTLY for every cell, the cell's
firing values are permuted across the sink-monkey axis (lines using
argsort(random) + take_along_axis). y is held FIXED. So the shuffle randomizes
the MONKEY LABELS linking neural firing to behavior, per cell -- it breaks any
real firing<->behavior correspondence while preserving each cell's set of firing
values and the behavior vector. (Permuting x is equivalent to permuting y for R^2.)

Key properties (all faithful to Ed's random.shuffle(x) inside his b/cell/source loop):
  - per-cell independent (each neuron's labels scrambled separately);
  - a FRESH permutation is drawn for every (behavior, source) and every iteration,
    so within one iteration the same neuron is scrambled differently across the
    6 behaviors and 10 sources -- it is NOT one global monkey relabeling;
  - NOT shuffled: the behavior matrices, the cell identities, the valid-sink sets,
    and (for r2_beh) the affiliation vector y_sec. Only x's monkey assignment.
Each iteration recomputes ALL summaries from this one shuffled dataset.

p-value convention (line: p = 1 - nless/NPERM)
  nless counts shuffles STRICTLY LESS than observed, so
    p = fraction of shuffles that matched-or-exceeded observed  (right-tailed).
  Ties count toward p; the observed is not added to the null. At NPERM=10000 the
  resolution is 1e-4, so a printed p=0.0000 means "no shuffle reached observed"
  (report as < 1e-4). Lower p => more significant.

Caveats: total / nsig / r2_beh are single omnibus tests; per-source (10),
per-behavior (6) and per-grid (60) p-values are each UNCORRECTED. The exchangeable
unit under this null is the monkey axis within a regression, not the cell.
"""

import time
import pandas as pd
import numpy as np
from common import (
    load_data, build_valid_k_per_source, build_y_per, vec_r2, scalar_r2,
    NMONKEYS, SUBJECT, MONKEY_NAME, BEH_NAMES, AFF_TO, AFF_FROM,
    HIS_XLSX,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
# Where the neural data comes from:
#   'grant_xlsx'      - Ed's original 74/75-cell xlsx (mean spike counts)  -> load_data
#   'cache_kw'        - SI-sorted KW list, rebuilt from analysis_cache      -> connector
#   'cache_anova'     - SI-sorted ANOVA list, rebuilt from analysis_cache   -> connector
#   'cache_mua_kw'    - threshold-MUA KW list (MAD/RMS offline detection)   -> connector
#   'cache_mua_anova' - threshold-MUA ANOVA list                           -> connector
#   'cache_mua_grantcells' - the grant's UNSORTED (MUA) cells only, recomputed from the
#                     threshold-MUA cache over each cell's own grant window (matched to
#                     NeuronIDs by date/round/channel). Use CELL_SUBSET='all'; compare
#                     against DATA_SOURCE='grant_xlsx' + CELL_SUBSET='multiunit'.
#   'cache_mua_grant_detected' - same grant MUA cells, but each cell's window(s) come from
#                     the in-house response-window DETECTOR (threshold_window_detection) run
#                     on the threshold-MUA source, not the grant xlsx. Row count differs from
#                     37 (detector yields 0/1/several windows per neuron). Use CELL_SUBSET='all'.
#   'cache_mua_grant_detected_sig_anova' - the detected windows that PASSED the permutation
#                     ANOVA test (uncorrected p<alpha). Run grant_mua_window_significance.py
#                     first to build output/grant_mua_window_significance.csv. CELL_SUBSET='all'.
#   'cache_mua_grant_detected_sig_kw' - same, but windows that passed the permutation KW test
#                     (uncorrected p<alpha). CELL_SUBSET='all'.
DATA_SOURCE = 'cache_kw'
NCELLS      = 74             # 'grant_xlsx' only: his hardcoded value; None/0 for all rows
CELL_SUBSET = 'all'          # 'all' | 'sorted' (Cell name has 'Unit') | 'multiunit' (no 'Unit')
DROP_OVERLAPPING_WINDOWS = False  # 'grant_xlsx' & 'cache_mua_grantcells' only: when a cell has
#                            overlapping time windows, keep only the WIDEST (drops narrower dups)
THRESH      = 0.5             # R^2 cutoff
NPERM       = 10000           # drop to 1000 for fast smoke-tests
RANDOM_SEED = 20251121
# =====================================================================


def _drop_overlapping_grant_windows(X_mean, df):
    """grant_xlsx post-filter for DROP_OVERLAPPING_WINDOWS: when a cell (Date, Round No.,
    Cell) has overlapping time windows, keep only the widest. Masks X_mean and df together.
    Reuses the pure overlap_keep_mask; unparseable windows are always kept."""
    import ast
    from grant_mua_matching import overlap_keep_mask
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            lo, hi = ast.literal_eval(str(r['Time Window']))
            key = (str(r['Date']), int(r['Round No.']), str(r['Cell']))
            lo, hi = float(lo), float(hi)
        except Exception:
            key, lo, hi = ('__row__', i), 0.0, 0.0   # unparseable -> singleton, always kept
        rows.append((key, lo, hi))
    mask = overlap_keep_mask(rows)
    n_drop = mask.count(False)
    if n_drop:
        drop_df = df[[not m for m in mask]][['Date', 'Round No.', 'Cell', 'Time Window']]
        print(f"  DROP_OVERLAPPING_WINDOWS: removed {n_drop} narrower overlapping window(s):")
        print(drop_df.to_string(index=False))
    keep = np.array(mask, dtype=bool)
    return X_mean[keep], df[keep].reset_index(drop=True)


def load_neural_data():
    """Dispatch on DATA_SOURCE. Returns (X_mean (ncells,9), df with 'Cell')."""
    if DATA_SOURCE == 'grant_xlsx':
        X_mean, df = load_data(HIS_XLSX, ncells=NCELLS)
        if DROP_OVERLAPPING_WINDOWS:
            X_mean, df = _drop_overlapping_grant_windows(X_mean, df)
        return X_mean, df
    if DATA_SOURCE in ('cache_kw', 'cache_anova', 'cache_mua_kw', 'cache_mua_anova'):
        from spike_count_connector import load_data_from_cache
        list_name = {'cache_kw': 'KW', 'cache_anova': 'ANOVA',
                     'cache_mua_kw': 'MUA_KW', 'cache_mua_anova': 'MUA_ANOVA'}[DATA_SOURCE]
        return load_data_from_cache(list_name)
    if DATA_SOURCE == 'cache_mua_grantcells':
        from spike_count_connector import load_grant_mua_from_cache
        return load_grant_mua_from_cache(dedup_overlapping=DROP_OVERLAPPING_WINDOWS)
    if DATA_SOURCE == 'cache_mua_grant_detected':
        from spike_count_connector import load_grant_mua_detected_windows
        return load_grant_mua_detected_windows()
    if DATA_SOURCE == 'cache_mua_grant_detected_sig_anova':
        from spike_count_connector import load_grant_mua_significant_windows
        return load_grant_mua_significant_windows(test='ANOVA', corrected=False)
    if DATA_SOURCE == 'cache_mua_grant_detected_sig_kw':
        from spike_count_connector import load_grant_mua_significant_windows
        return load_grant_mua_significant_windows(test='KW', corrected=False)
    raise ValueError(f"unknown DATA_SOURCE={DATA_SOURCE!r}")


def main():
    print(f"Loading data (DATA_SOURCE={DATA_SOURCE})")
    X_mean, df = load_neural_data()
    # ---- CELL SUBSET FILTER (sorted vs multiunit) ----
    # 'Unit' in the Cell name = manually sorted single unit; absence = multiunit.
    # Mask X_mean and df together so neural rows stay aligned with metadata.
    if CELL_SUBSET != 'all':
        cell_names = df['Cell'].astype(str)
        has_unit = cell_names.str.contains('Unit', case=False, na=False).to_numpy()
        keep = has_unit if CELL_SUBSET == 'sorted' else ~has_unit
        X_mean = X_mean[keep]
        df = df.iloc[keep].reset_index(drop=True)
        print(f"  CELL_SUBSET='{CELL_SUBSET}': kept {int(keep.sum())} of {len(keep)} cells")
    else:
        print(f"  CELL_SUBSET='all': using all {X_mean.shape[0]} cells")

    ncells = X_mean.shape[0]
    print(f"  Using ncells={ncells}")

    valid_k = build_valid_k_per_source()
    y_per = build_y_per(valid_k)
    X_per_source = {s: X_mean[:, valid_k[s]].copy() for s in range(NMONKEYS)}

    # ---- DEGENERACY SCAN ----
    # A regression's R^2 is forced to 0 (not a real fit) whenever the neural
    # response is constant across the valid sinks (n*Sxx - Sx^2 == 0), or the
    # behavior vector is constant. vec_r2() returns 0 in both cases, which is
    # indistinguishable from a genuine ~0 fit and never crosses THRESH, so these
    # regressions silently drop out of nsig / sumR^2. This scan lists them so a
    # structural zero can be told apart from a real one. (The original script
    # would instead raise ZeroDivisionError on these same cases.)
    print("\n--- DEGENERACY SCAN (regressions forced to R^2=0) ---")
    EPS = 1e-12
    n_forced = 0
    for s in range(NMONKEYS):
        xvar = X_per_source[s].var(axis=1)              # (ncells,)
        for ic in np.where(xvar < EPS)[0]:
            n_forced += 6                                # all 6 behaviors hit for this (cell, source)
            print(f"  [const neural]   source={MONKEY_NAME[s]:>5s}  cell#{ic}:  "
                  f"resp={X_per_source[s][ic]}  -> R^2=0 for all 6 behaviors")
    for ib in range(6):
        for s in range(NMONKEYS):
            if np.var(y_per[(ib, s)]) < EPS:
                n_forced += ncells
                print(f"  [const behavior] {BEH_NAMES[ib]:>8s} source={MONKEY_NAME[s]:>5s}:  "
                      f"behavior vector constant -> R^2=0 for all {ncells} cells")
    if n_forced == 0:
        print("  none - every regression has non-degenerate neural and behavior variance.")
    print(f"  total regressions forced to 0: {n_forced} / {ncells*6*NMONKEYS}")

    # ---- OBSERVED PASS ----
    print("\n--- OBSERVED PASS ---")
    obs_sumR2_per_src = np.zeros(NMONKEYS)
    obs_sumR2_per_beh = np.zeros(6)
    obs_sumR2_grid = np.zeros((6, NMONKEYS))
    obs_R2_per_cell = np.zeros((6, NMONKEYS, ncells))
    obs_nsig = 0

    for ib in range(6):
        for s in range(NMONKEYS):
            r2 = vec_r2(X_per_source[s], y_per[(ib, s)])
            obs_R2_per_cell[ib, s] = r2
            mask = r2 > THRESH
            if mask.any():
                contrib = np.abs(r2[mask]).sum()
                obs_sumR2_per_src[s] += contrib
                obs_sumR2_per_beh[ib] += contrib
                obs_sumR2_grid[ib, s] += contrib
                obs_nsig += int(mask.sum())

    obs_total = obs_sumR2_per_src.sum()
    print(f"nsig (regressions with R^2>{THRESH}): {obs_nsig} / {ncells*6*NMONKEYS}")
    print(f"sumRsquared_total: {obs_total:.4f}\n")

    print("Per source monkey:")
    for s in range(NMONKEYS):
        print(f"  {MONKEY_NAME[s]:>6s} (Z{s}): {obs_sumR2_per_src[s]:.4f}")
    print("\nPer behavior:")
    for ib in range(6):
        print(f"  {BEH_NAMES[ib]:>10s}: {obs_sumR2_per_beh[ib]:.4f}")

    # Secondary regression: affiliation with subject vs sumR^2 (10 points)
    y_sec = np.array([AFF_TO[SUBJECT, sm] + AFF_FROM[SUBJECT, sm] for sm in range(NMONKEYS)])
    obs_r2_beh = scalar_r2(obs_sumR2_per_src, y_sec)
    print(f"\nSecondary regression r2_beh: {obs_r2_beh:.4f}")

    # ---- PERMUTATION TEST ----
    print(f"\n--- PERMUTATION TEST ({NPERM:,} iterations) ---")
    rng = np.random.default_rng(RANDOM_SEED)
    nless_per_src = np.zeros(NMONKEYS, dtype=int)
    nless_total = 0
    nless_nsig = 0
    nless_per_beh = np.zeros(6, dtype=int)
    nless_grid = np.zeros((6, NMONKEYS), dtype=int)
    nless_r2beh = 0

    t0 = time.time()
    for ir in range(NPERM):
        rnd_sumR2_src = np.zeros(NMONKEYS)
        rnd_sumR2_beh = np.zeros(6)
        rnd_sumR2_grid = np.zeros((6, NMONKEYS))
        nrndsig = 0
        for ib in range(6):
            for s in range(NMONKEYS):
                X_sub = X_per_source[s]; y = y_per[(ib, s)]
                n_v = X_sub.shape[1]
                keys = rng.random((ncells, n_v))
                perm_idx = np.argsort(keys, axis=1)
                X_perm = np.take_along_axis(X_sub, perm_idx, axis=1)
                r2 = vec_r2(X_perm, y)
                mask = r2 > THRESH
                if mask.any():
                    contrib = np.abs(r2[mask]).sum()
                    rnd_sumR2_src[s] += contrib
                    rnd_sumR2_beh[ib] += contrib
                    rnd_sumR2_grid[ib, s] += contrib
                    nrndsig += int(mask.sum())
        rnd_total = rnd_sumR2_src.sum()
        nless_per_src += (rnd_sumR2_src < obs_sumR2_per_src).astype(int)
        if rnd_total < obs_total: nless_total += 1
        if nrndsig < obs_nsig: nless_nsig += 1
        nless_per_beh += (rnd_sumR2_beh < obs_sumR2_per_beh).astype(int)
        nless_grid += (rnd_sumR2_grid < obs_sumR2_grid).astype(int)
        rnd_r2beh = scalar_r2(rnd_sumR2_src, y_sec)
        if rnd_r2beh < obs_r2_beh: nless_r2beh += 1

        if (ir+1) % 2000 == 0:
            print(f"  iter {ir+1}/{NPERM} elapsed={time.time()-t0:.1f}s")
    print(f"\nDone in {time.time()-t0:.1f}s.")

    # ---- REPORT ----
    print("\n" + "="*60)
    print("PERMUTATION RESULTS  (p = 1 - nless/NPERM; lower p => more significant)")
    print("="*60)
    print(f"sumRsquared_total:  obs={obs_total:.4f}  p={1-nless_total/NPERM:.4f}")
    print(f"nsig (count):       obs={obs_nsig}      p={1-nless_nsig/NPERM:.4f}")
    print(f"r2_beh (secondary): obs={obs_r2_beh:.4f}  p={1-nless_r2beh/NPERM:.4f}")

    print("\nPer source monkey (sumRsquared across all behaviors):")
    for s in range(NMONKEYS):
        p = 1 - nless_per_src[s] / NPERM
        print(f"  {MONKEY_NAME[s]:>6s} (Z{s}):  obs={obs_sumR2_per_src[s]:7.4f}  p={p:.4f}")

    print("\nPer behavior (sumRsquared across all source monkeys):")
    for ib in range(6):
        p = 1 - nless_per_beh[ib] / NPERM
        print(f"  {BEH_NAMES[ib]:>10s}:  obs={obs_sumR2_per_beh[ib]:7.4f}  p={p:.4f}")

    print("\nPer (behavior x source) grid:")
    print(f"  {'':>10s}  " + "  ".join(f"{m:>13s}" for m in MONKEY_NAME))
    for ib in range(6):
        parts = [f"  {BEH_NAMES[ib]:>10s} "]
        for s in range(NMONKEYS):
            p = 1 - nless_grid[ib, s] / NPERM
            parts.append(f" {obs_sumR2_grid[ib,s]:5.2f}(p={p:.3f})")
        print("".join(parts))


if __name__ == '__main__':
    main()
