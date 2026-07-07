"""
Figure 1: Leave-one-out (LOO) fragility of his 'best-tuned' cells.

In PyCharm: edit the CONFIG block below, then just hit Run.

For 4 cells with R^2 > 0.5 from his analysis, show:
    Top row:    the full fit with all data points (one point colored red = outlier)
    Bottom row: the same fit after removing the red point - R^2 typically collapses
"""

import numpy as np
import matplotlib.pyplot as plt
from common import (
    load_data, compute_observed_R2, MONKEY_NAME, BEH_NAMES, set_plot_style, COLORS,
    DEFAULT_DATA_FILE,
)


# =====================================================================
# CONFIG - EDIT THESE
# =====================================================================
DATA_FILE = DEFAULT_DATA_FILE      # or set to an absolute path string
NCELLS    = 74                     # his hardcoded value; use None for all rows
OUTPUT    = 'fig1_LOO_fragility.png'

# (behavior_index, source_index, cell_index) - pick 4 cells to show
#   behavior_index: 0=aff_to, 1=aff_from, 2=sub_to, 3=sub_from, 4=agn_to, 5=agn_from
#   source_index:   0=7124, 1=69X, 2=72X, 3=94B, 4=110E, 5=67G, 6=81G(subj), 7=143H, 8=87J, 9=151J
TARGET_CELLS = [
    (0, 4, 44),   # aff_to,  110E, cell 44
    (2, 1, 60),   # sub_to,  69X,  cell 60
    (3, 2, 20),   # sub_from, 72X, cell 20
    (4, 2, 40),   # agn_to,  72X,  cell 40
]
# =====================================================================


def leave_one_out_r2(x, y):
    """Recompute R^2 with each point removed in turn. Returns array of length n."""
    n = len(x)
    out = []
    for d in range(n):
        keep = [j for j in range(n) if j != d]
        xl, yl = x[keep], y[keep]
        if xl.var() == 0 or yl.var() == 0:
            out.append(0.0)
        else:
            r = np.corrcoef(xl, yl)[0, 1]
            out.append(r*r)
    return np.array(out)


def main():
    set_plot_style()
    X_mean, _ = load_data(DATA_FILE, ncells=NCELLS)
    obs_R2, valid_k, y_per, X_per_source = compute_observed_R2(X_mean, NCELLS)

    # Gather records for the selected cells
    selected = []
    for ib, s, icell in TARGET_CELLS:
        x = X_per_source[s][icell]
        y = y_per[(ib, s)]
        r2 = float(obs_R2[ib, s, icell])
        loo = leave_one_out_r2(x, y)
        selected.append({
            'ib': ib, 's': s, 'cell': icell, 'r2': r2,
            'r2_min': float(loo.min()), 'worst': int(loo.argmin()),
            'x': x, 'y': y,
        })

    # Aggregate LOO statistics over all R^2 > 0.5 cells
    n_above_05 = 0; n_drops_03 = 0; n_drops_01 = 0
    for ib in range(6):
        for s in range(10):
            for icell in range(NCELLS):
                if obs_R2[ib, s, icell] > 0.5:
                    n_above_05 += 1
                    loo = leave_one_out_r2(X_per_source[s][icell], y_per[(ib, s)])
                    if loo.min() < 0.3: n_drops_03 += 1
                    if loo.min() < 0.1: n_drops_01 += 1
    frac_03 = n_drops_03 / n_above_05 * 100
    frac_01 = n_drops_01 / n_above_05 * 100

    # Build figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), gridspec_kw={'hspace':0.45, 'wspace':0.35})
    for i, rec in enumerate(selected):
        cell = rec['cell']; beh = BEH_NAMES[rec['ib']]; src = MONKEY_NAME[rec['s']]
        r2 = rec['r2']; r2_min = rec['r2_min']
        xs = rec['x']; ys = rec['y']; worst = rec['worst']
        n = len(xs)
        keep = [j for j in range(n) if j != worst]
        xs_k, ys_k = xs[keep], ys[keep]

        # Top: full fit with outlier highlighted
        ax = axes[0, i]
        ax.scatter(xs, ys, s=70, c=COLORS['blue'], edgecolors='black', linewidth=0.5, zorder=3)
        ax.scatter([xs[worst]], [ys[worst]], s=120, facecolor=COLORS['red'],
                   edgecolors='black', linewidth=1.2, zorder=4, label='outlier point')
        p = np.polyfit(xs, ys, 1); xf = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xf, p[0]*xf + p[1], color=COLORS['blue'], linewidth=2, alpha=0.85)
        ax.set_title(f"Cell {cell}: {beh} -> {src}\nR^2 = {r2:.3f}", fontweight='bold')
        ax.set_xlabel("Mean firing rate"); ax.set_ylabel("Behavior count")
        ax.legend(loc='best', frameon=False, fontsize=8)

        # Bottom: fit after dropping the outlier
        ax = axes[1, i]
        ax.scatter(xs_k, ys_k, s=70, c=COLORS['blue'], edgecolors='black', linewidth=0.5, zorder=3)
        ax.scatter([xs[worst]], [ys[worst]], s=120, facecolor='none', edgecolors=COLORS['red'],
                   linewidth=1.5, linestyle='--', zorder=4, label='removed')
        if xs_k.var() > 0 and ys_k.var() > 0:
            p = np.polyfit(xs_k, ys_k, 1); xf = np.linspace(xs_k.min(), xs_k.max(), 50)
            ax.plot(xf, p[0]*xf + p[1], color=COLORS['green'], linewidth=2, alpha=0.85)
        ax.set_title(f"After dropping ONE point\nR^2 = {r2_min:.3f}", color=COLORS['red'], fontweight='bold')
        ax.set_xlabel("Mean firing rate"); ax.set_ylabel("Behavior count")
        ax.legend(loc='best', frameon=False, fontsize=8)
        ax.set_xlim(axes[0, i].get_xlim())

    fig.suptitle(
        f"Figure 1. 'Best-tuned' neurons collapse when ONE data point is removed.\n"
        f"Across all R^2>0.5 fits (ncells={NCELLS}, {n_above_05} cells): "
        f"{frac_03:.1f}% drop below R^2=0.3, {frac_01:.1f}% drop below R^2=0.1.",
        fontsize=12, fontweight='bold', y=1.0
    )
    plt.savefig(OUTPUT, bbox_inches='tight')
    plt.show()
    print(f"Saved {OUTPUT}")
    print(f"  {n_above_05} cells above R^2>0.5; {frac_03:.1f}% drop <0.3, {frac_01:.1f}% drop <0.1.")


if __name__ == '__main__':
    main()
