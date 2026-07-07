"""
Figure 3: Multiple-comparisons survival across the 60 (behavior x source) tests.

Panel (a): heatmap of -log10(p) for each (behavior, source) cell.
Panel (b): how many tests survive raw / FDR / Bonferroni correction.

In PyCharm: edit the CONFIG block below, then just hit Run.
Default 10,000 perms takes ~30s. Drop NPERM to 1000 for faster iteration.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import (
    load_data, build_valid_k_per_source, build_y_per, vec_r2,
    BEH_NAMES, MONKEY_NAME, NMONKEYS, set_plot_style, COLORS, DEFAULT_DATA_FILE,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
DATA_FILE   = DEFAULT_DATA_FILE
NCELLS      = 74
OUTPUT      = 'fig3_multcomp.png'

NPERM       = 10000           # 10000 default; drop to 1000 for ~3s runs
RANDOM_SEED = 20251121

THRESH      = 0.5             # R^2 cutoff matching his pipeline
# =====================================================================


def run_permutation(X_mean, y_per, X_per_source, ncells, nperm, seed):
    """Compute observed (behavior, source) sums and p-values via permutation."""
    obs_sum = np.zeros((6, NMONKEYS))
    for ib in range(6):
        for s in range(NMONKEYS):
            r2 = vec_r2(X_per_source[s], y_per[(ib, s)])
            mask = r2 > THRESH
            if mask.any():
                obs_sum[ib, s] = np.abs(r2[mask]).sum()

    rng = np.random.default_rng(seed)
    nless = np.zeros((6, NMONKEYS), dtype=int)
    t0 = time.time()
    for ir in range(nperm):
        rnd_sum = np.zeros((6, NMONKEYS))
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
                    rnd_sum[ib, s] = np.abs(r2[mask]).sum()
        nless += (rnd_sum < obs_sum).astype(int)
        if (ir+1) % 2000 == 0:
            print(f"  iter {ir+1}/{nperm} elapsed={time.time()-t0:.1f}s")
    p_grid = 1 - nless/nperm
    return obs_sum, p_grid


def main():
    set_plot_style()
    X_mean, _ = load_data(DATA_FILE, ncells=NCELLS)
    valid_k = build_valid_k_per_source()
    y_per = build_y_per(valid_k)
    X_per_source = {s: X_mean[:, valid_k[s]].copy() for s in range(NMONKEYS)}

    print(f"Running {NPERM} permutations...")
    obs_sum, p_grid = run_permutation(X_mean, y_per, X_per_source, NCELLS, NPERM, RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'wspace':0.32})

    # Panel (a): heatmap
    ax = axes[0]
    neg_log = -np.log10(p_grid + 1e-4)
    im = ax.imshow(neg_log, aspect='auto', cmap='Reds', vmin=0, vmax=4)
    ax.set_xticks(range(10)); ax.set_xticklabels(MONKEY_NAME, rotation=45)
    ax.set_yticks(range(6)); ax.set_yticklabels(BEH_NAMES)
    ax.set_xlabel("Source monkey"); ax.set_ylabel("Behavior")
    ax.set_title("(a) -log10(p) for all 60 (behavior x source) tests")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85); cbar.set_label('-log10(p)')
    for ib in range(6):
        for sm in range(10):
            marker = ''
            if p_grid[ib, sm] < 0.05/60:
                marker = '**'
            elif p_grid[ib, sm] < 0.05:
                marker = '*'
            if marker:
                ax.text(sm, ib, marker, ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='black' if neg_log[ib, sm] < 2 else 'white')
    ax.text(0.02, -0.18, '* = raw p<0.05  |  ** = Bonferroni p<0.05',
            transform=ax.transAxes, fontsize=8.5)

    # Panel (b): survivorship counts
    ax = axes[1]
    ps = p_grid.flatten()
    n_raw05 = int((ps < 0.05).sum())
    n_raw01 = int((ps < 0.01).sum())
    n_bonf  = int((ps * 60 < 0.05).sum())
    # BH-FDR
    n_fdr = 0
    p_sorted = np.sort(ps); m = len(p_sorted)
    for i, p in enumerate(p_sorted):
        if p <= (i+1)/m * 0.05:
            n_fdr = i+1

    levels = ['Raw p<0.05', 'Raw p<0.01', 'BH-FDR\nq=0.05', 'Bonferroni\np<0.05']
    counts = [n_raw05, n_raw01, n_fdr, n_bonf]
    colors_bar = [COLORS['blue'], COLORS['blue'], COLORS['gray'], COLORS['gray']]
    ax.bar(levels, counts, color=colors_bar, edgecolor='black', linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.15, str(c), ha='center', fontweight='bold', fontsize=11)

    binom_p = 1 - stats.binom.cdf(n_raw05 - 1, 60, 0.05)
    ax.axhline(3.0, color='red', linestyle='--', linewidth=1.5, label='Expected by chance (p<0.05): 3.0')
    ax.axhline(0.6, color='red', linestyle=':', linewidth=1.5, label='Expected by chance (p<0.01): 0.6')
    ax.set_ylabel("# of 60 (behavior x monkey) tests significant")
    ax.set_title(f"(b) Aggregate (binom p={binom_p:.3f})\n  ")
    ax.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax.set_ylim(0, max(8, n_raw05 + 2))

    fig.suptitle(
        f"Figure 3. Aggregate count of nominally significant tests ({n_raw05} vs 3 expected, binomial p~{binom_p:.3f});\n"
        f"NO individual (behavior x monkey) effect survives multiple-comparison correction.",
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.show()
    print(f"Saved {OUTPUT}")
    print(f"  Raw p<0.05: {n_raw05}/60; Bonferroni: {n_bonf}/60; BH-FDR: {n_fdr}/60; aggregate binom p={binom_p:.4f}")


if __name__ == '__main__':
    main()
