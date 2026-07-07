"""
Figure 4: Mean R^2 across all 4,440 regressions, observed vs null distribution.

This is the 'cleaner' alternative to his sum-of-thresholded R^2 statistic:
  - takes the mean R^2 across ALL regressions (no threshold)
  - compares to a permutation null
  - reports effect size (observed minus null) alongside the p-value

In PyCharm: edit the CONFIG block below, then just hit Run.
Default 10,000 perms takes ~30s. Drop NPERM to 1000 for faster iteration.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from common import (
    load_data, build_valid_k_per_source, build_y_per, vec_r2,
    compute_observed_R2, NMONKEYS, set_plot_style, COLORS, DEFAULT_DATA_FILE,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
DATA_FILE   = DEFAULT_DATA_FILE
NCELLS      = 74
OUTPUT      = 'fig4_effect_size.png'

NPERM       = 10000           # 10000 default; drop to 1000 for ~3s runs
RANDOM_SEED = 42
# =====================================================================


def run_null_distribution(X_per_source, y_per, ncells, nperm, seed):
    """Compute the null distribution of mean R^2 across all 4,440 regressions."""
    rng = np.random.default_rng(seed)
    null_means = np.zeros(nperm)
    t0 = time.time()
    for ir in range(nperm):
        all_rs = []
        for ib in range(6):
            for s in range(NMONKEYS):
                X_sub = X_per_source[s]; y = y_per[(ib, s)]
                n_v = X_sub.shape[1]
                keys = rng.random((ncells, n_v))
                perm_idx = np.argsort(keys, axis=1)
                X_perm = np.take_along_axis(X_sub, perm_idx, axis=1)
                all_rs.extend(vec_r2(X_perm, y).tolist())
        null_means[ir] = np.mean(all_rs)
        if (ir+1) % 2000 == 0:
            print(f"  iter {ir+1}/{nperm} elapsed={time.time()-t0:.1f}s")
    return null_means


def main():
    set_plot_style()
    X_mean, _ = load_data(DATA_FILE, ncells=NCELLS)
    obs_R2, valid_k, y_per, X_per_source = compute_observed_R2(X_mean, NCELLS)

    obs_mean = float(obs_R2.mean())
    print(f"Observed mean R^2 across all {obs_R2.size} regressions: {obs_mean:.4f}")

    print(f"Running {NPERM} permutations for null distribution...")
    null_means = run_null_distribution(X_per_source, y_per, NCELLS, NPERM, RANDOM_SEED)

    null_mean = null_means.mean()
    p_value = (null_means >= obs_mean).sum() / len(null_means)
    diff = obs_mean - null_mean
    pct = diff / null_mean * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'wspace': 0.3})

    # Panel (a)
    ax = axes[0]
    ax.hist(null_means, bins=60, color=COLORS['gray'], edgecolor='black',
            linewidth=0.3, alpha=0.7, label=f'Null distribution ({NPERM:,} shuffles)')
    ax.axvline(obs_mean, color=COLORS['red'], linewidth=2.5, label=f'Observed = {obs_mean:.4f}')
    ax.axvline(null_mean, color='black', linewidth=1.5, linestyle='--',
               label=f'Null mean = {null_mean:.4f}')
    ax.set_xlabel(f"Mean R^2 across all {obs_R2.size:,} regressions")
    ax.set_ylabel("# of permutations")
    ax.set_title(f"(a) Population-level effect (p ~ {p_value:.4f})")
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)

    # Panel (b)
    ax = axes[1]
    bars = [null_mean, obs_mean]
    ax.bar(['Null (chance)', 'Observed'], bars,
           color=[COLORS['gray'], COLORS['blue']], edgecolor='black', linewidth=0.5)
    for i, v in enumerate(bars):
        ax.text(i, v + 0.001, f"{v:.4f}", ha='center', fontweight='bold', fontsize=11)
    ax.annotate(
        f'difference =\n{diff:.4f}\n({pct:.1f}% over null)',
        xy=(1, obs_mean), xytext=(1.4, obs_mean - 0.05),
        fontsize=10, fontweight='bold', color=COLORS['red'], ha='center'
    )
    ax.set_ylim(0, max(bars) * 1.4)
    ax.set_ylabel("Mean R^2")
    ax.set_title("(b) Observed vs chance mean R^2")

    fig.suptitle(
        f"Figure 4. Real population-level effect: observed mean R^2 ({obs_mean:.4f}) "
        f"exceeds the null ({null_mean:.4f}) at p ~ {p_value:.4f}.\n"
        f"Statistically robust, but small in absolute size (~{diff:.4f} above chance).",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.show()
    print(f"Saved {OUTPUT}")
    print(f"  obs={obs_mean:.4f}, null={null_mean:.4f}, diff={diff:.4f}, p={p_value:.4f}")


if __name__ == '__main__':
    main()
