"""
Figure 2: Three panels showing
  (top row)     the most outlier-dominated y-vectors in the behavior matrices
  (bottom-left) the observed R^2 distribution overlaid with the theoretical null at n=8
  (bottom-right) excess-over-chance ratios at different R^2 thresholds

In PyCharm: edit the CONFIG block below, then just hit Run.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import (
    load_data, compute_observed_R2, BEH_NAMES, MONKEY_NAME,
    set_plot_style, COLORS, DEFAULT_DATA_FILE,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
DATA_FILE  = DEFAULT_DATA_FILE       # or an absolute path string
NCELLS     = 74
OUTPUT     = 'fig2_thresholds_and_outliers.png'

# R^2 thresholds to evaluate in the right-hand panel
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Number of worst y-vectors to show in the top row (max 3 because of layout)
N_WORST_Y  = 3
# =====================================================================


def find_outlier_y_vectors(y_per):
    """Return list of (behavior_name, source_name, y_vector, max_fraction) sorted by
    most-outlier-dominated first."""
    out = []
    for (ib, source), y in y_per.items():
        if y.sum() == 0:
            continue
        max_frac = y.max() / y.sum()
        out.append((BEH_NAMES[ib], MONKEY_NAME[source], y, max_frac))
    out.sort(key=lambda r: -r[3])
    return out


def main():
    set_plot_style()
    X_mean, _ = load_data(DATA_FILE, ncells=NCELLS)
    obs_R2, valid_k, y_per, X_per_source = compute_observed_R2(X_mean, NCELLS)
    all_r2 = obs_R2.flatten()
    ntotal = len(all_r2)

    # Theoretical null R^2 at n=8: Beta(0.5, 3)
    r2_grid = np.linspace(0, 1, 500)
    null_pdf = stats.beta.pdf(r2_grid, 0.5, 3.0)
    null_pdf_clip = np.where(np.isfinite(null_pdf), null_pdf, 0)

    # Outlier y-vectors
    worst_y = find_outlier_y_vectors(y_per)[:N_WORST_Y]

    # Excess ratios per threshold
    expected = [(1 - stats.f.cdf(t*6/(1-t), 1, 6)) * ntotal for t in THRESHOLDS]
    observed = [(all_r2 > t).sum() for t in THRESHOLDS]
    ratios = [o/e if e > 0 else 0 for o, e in zip(observed, expected)]

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4)

    # Top row: outlier y-vectors
    for i, (beh, src, y, frac) in enumerate(worst_y):
        ax = fig.add_subplot(gs[0, i])
        colors = [COLORS['red'] if (v == y.max() and y.max() > 0) else COLORS['blue'] for v in y]
        ax.bar(range(len(y)), y, color=colors, edgecolor='black', linewidth=0.4)
        ax.set_title(f"{beh} from {src}\n(one point = {frac:.0%} of total)", fontsize=9.5)
        ax.set_xticks(range(len(y))); ax.set_xticklabels([f"M{j}" for j in range(len(y))], fontsize=8)
        ax.set_ylabel("Behavior count"); ax.set_xlabel("Stimulus monkey")
        ax.set_ylim(0, max(y.max()*1.15, 1))

    # Bottom-left: null PDF + observed R^2 histogram
    ax = fig.add_subplot(gs[1, :2])
    mask = (r2_grid > 0.02) & (r2_grid < 0.99)
    ax.fill_between(r2_grid[mask], 0, null_pdf_clip[mask], color=COLORS['gray'],
                    alpha=0.35, label='Null density (n=8, no real tuning)')
    ax.plot(r2_grid[mask], null_pdf_clip[mask], color=COLORS['dark'], linewidth=1.5)
    counts, bins = np.histogram(all_r2, bins=40, range=(0, 1), density=True)
    ax.bar((bins[:-1]+bins[1:])/2, counts, width=np.diff(bins)[0]*0.9, color=COLORS['blue'],
           alpha=0.7, edgecolor='black', linewidth=0.3,
           label=f'Observed R^2 (n={ntotal} regressions)')
    ax.axvline(0.5, color=COLORS['red'], linewidth=2, linestyle='--', label='His threshold (R^2>0.5)')
    ax.set_xlabel("R^2"); ax.set_ylabel("Density")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(counts.max(), null_pdf_clip[mask].max())*1.15)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title("Tail of the observed R^2 distribution sits above the null - consistent with real (small) signal",
                 fontsize=10.5)

    # Bottom-right: bar chart of expected vs observed
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(THRESHOLDS)); w = 0.38
    ax.bar(x-w/2, expected, w, label='Expected by chance', color=COLORS['gray'],
           edgecolor='black', linewidth=0.4)
    ax.bar(x+w/2, observed, w, label='Observed', color=COLORS['blue'],
           edgecolor='black', linewidth=0.4)
    for i, r in enumerate(ratios):
        ax.text(i+w/2, observed[i] + ntotal*0.01, f"{r:.1f}x",
                ha='center', fontsize=8, fontweight='bold', color=COLORS['red'])
    ax.set_xticks(x); ax.set_xticklabels([f"R^2>{t}" for t in THRESHOLDS])
    ax.set_ylabel("# regressions")
    ax.set_title("Excess over chance grows with R^2\n(suggests genuine high-R^2 tail)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Figure 2. R^2 distribution shows a small excess over chance at high R^2 "
        f"({ratios[2]:.1f}x at R^2>0.5 -> {ratios[-1]:.1f}x at R^2>{THRESHOLDS[-1]}). The signal is in the tail;\n"
        f"the R^2>0.5 threshold itself is loose (= uncorrected p<0.05 per regression at n=8).",
        fontsize=11.5, fontweight='bold', y=1.02
    )
    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.show()
    print(f"Saved {OUTPUT}")
    print(f"  Excess ratios @ R^2>{THRESHOLDS}: {[f'{r:.2f}' for r in ratios]}")


if __name__ == '__main__':
    main()
