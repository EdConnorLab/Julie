"""
Replication of allbeh_rand_var_exp_2_5_2025.py.

Reproduces his Table 1 to 4 decimal places when run on the
'_old--usedforgrant.xlsx' file with NCELLS=74.

In PyCharm: edit the CONFIG block below, then just hit Run.

Pipeline (per cell, per behavior, per source monkey):
  - For each sink monkey (excluding source and subject 81G):
      x = mean spike count of cell when sink monkey was visualized
      y = behavior_matrix[source][sink]
  - n = 8 data points (or 9 if source == subject 81G)
  - Fit OLS, compute R^2
  - Threshold at R^2 > 0.5, accumulate per-source / per-behavior / per-(behavior,source) sums

Permutation test:
  - NPERM iterations, x shuffled per-cell per-(behavior,source)
  - Compares shuffled to observed for each summary statistic
"""

import time
import numpy as np
from common import (
    load_data, build_valid_k_per_source, build_y_per, vec_r2, scalar_r2,
    NMONKEYS, SUBJECT, MONKEY_NAME, BEH_NAMES, AFF_TO, AFF_FROM,
    DEFAULT_DATA_FILE,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
DATA_FILE   = DEFAULT_DATA_FILE
NCELLS      = 74             # his hardcoded value; use None or 0 for all rows
CELL_SUBSET = 'multiunit'          # 'all' | 'sorted' (Cell name has 'Unit') | 'multiunit' (no 'Unit')
THRESH      = 0.5             # R^2 cutoff
NPERM       = 10000           # drop to 1000 for fast smoke-tests
RANDOM_SEED = 20251121
# =====================================================================


def main():
    print(f"Loading data from {DATA_FILE}")
    X_mean, df = load_data(DATA_FILE, ncells=NCELLS)

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
