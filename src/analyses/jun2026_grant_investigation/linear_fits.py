"""
STEP 3: linear-fit plots for the cells whose regressions clear R^2 > THRESH.

For each (cell x source-monkey x behavior) fit above threshold, draw the actual
scatter that produced the R^2:
    x = the cell's mean firing to each valid sink monkey
    y = the source monkey's behavior toward those same sink monkeys
    points labeled by sink monkey; OLS line overlaid; R^2 in the title.

This lets you eyeball WHAT a "significant" linear fit looks like (is it a real
graded relationship, or one leverage point dragging the line?). Fits are ranked
by R^2 and the top TOP_N per list are shown.

Reuses the same loaders as r2_distributions.py:
  GRANT -> xlsx (common.load_data);  KW/ANOVA -> connector (analysis_cache).

Run (PyCharm: hit Run). Outputs -> ./investigation_outputs/
  linear_fits_<LIST>.png     grid of the top TOP_N above-threshold fits (session-
                             qualified titles, e.g. 2023-09-29_1_C_025_U3)
  linear_fits_<LIST>.csv     top CSV_TOP_N fits: list, full_name, cell, location,
                             date, round, source, behavior, R^2
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    MONKEY_NAME, SUBJECT, NMONKEYS, K_TO_FULL, vec_r2,
    build_valid_k_per_source, set_plot_style,
)
from r2_distributions import ALL_BEH, BEH_NAMES, COLORS, OUTDIR, load_lists

# =====================================================================
# CONFIG - edit these
# =====================================================================
LIST = 'all'                 # 'GRANT' | 'KW' | 'ANOVA' | 'all'
THRESH = 0.50                # only plot fits with R^2 > THRESH
TOP_N = 16                   # how many highest-R^2 fits to PLOT per list
CSV_TOP_N = 30               # how many highest-R^2 fits to write to the CSV per list
SOURCES = 'allocentric'      # 'allocentric' (drop 81G as a source) | 'all'
BEHAVIORS = 'all'            # 'all' (6) or a list of indices into BEH_NAMES
NCOLS = 4                    # grid columns
# =====================================================================


def _short_channel(cell):
    """Channel.C_003_Unit 3 -> C_003_U3."""
    s = str(cell)
    if 'Channel.' in s:
        s = s.split('Channel.', 1)[1]
    return s.replace('Unit ', 'U').replace(' ', '')


def _fmt_date(v):
    try:
        return pd.to_datetime(v).strftime('%Y-%m-%d')
    except Exception:
        return str(v)


def _session_info(lab, row):
    """Return (location, date, round, full_name) for one cell.

    GRANT: from the xlsx 'Cell'/'Date'/'Round No.' columns (no location available).
    KW/ANOVA: parsed from the NeuronID 'LOC_DATE_ROUND_Channel.C_xxx_Unit y'.
    full_name is a session-qualified label, e.g. 2023-09-29_1_C_025_U3 (GRANT) or
    AMG_2023-09-29_1_C_025_U3 (KW/ANOVA)."""
    cell = str(row['Cell'])
    if lab == 'GRANT':
        date = _fmt_date(row['Date']) if 'Date' in row else ''
        rnd = row['Round No.'] if 'Round No.' in row else ''
        try:
            rnd = str(int(rnd))
        except (ValueError, TypeError):
            rnd = str(rnd)
        full = '_'.join(p for p in (date, rnd, _short_channel(cell)) if p)
        return '', date, rnd, full
    parts = cell.split('_', 3)                       # LOC, DATE, ROUND, Channel.C_xxx_Unit y
    if len(parts) == 4:
        loc, date, rnd, chan = parts
        return loc, date, rnd, f"{loc}_{date}_{rnd}_{_short_channel(chan)}"
    return '', '', '', _short_channel(cell)


def collect_fits(lab, X, meta):
    """All (cell, source, behavior) fits for one list, each with its x/y/R^2."""
    valid_k = build_valid_k_per_source()
    sources = range(NMONKEYS) if SOURCES == 'all' else [s for s in range(NMONKEYS) if s != SUBJECT]
    behs = range(6) if BEHAVIORS == 'all' else BEHAVIORS
    fits = []
    for s in sources:
        ks = valid_k[s]
        fulls = [K_TO_FULL[k] for k in ks]
        sinks = [MONKEY_NAME[f] for f in fulls]
        Xs = X[:, ks]
        for ib in behs:
            y = np.array([ALL_BEH[ib][s, f] for f in fulls], float)
            r2 = vec_r2(Xs, y)
            for ci in range(X.shape[0]):
                row = meta.iloc[ci]
                loc, date, rnd, full = _session_info(lab, row)
                fits.append(dict(
                    list=lab, cell=str(row['Cell']), location=loc, date=date, round=rnd,
                    full_name=full, source=MONKEY_NAME[s],
                    behavior=BEH_NAMES[ib], r2=float(r2[ci]),
                    x=Xs[ci].astype(float).copy(), y=y.copy(), sinks=sinks,
                ))
    return fits


def plot_grid(fits, lab):
    """Grid of scatter+OLS panels for the given (already-selected) fits."""
    set_plot_style()
    n = len(fits)
    if n == 0:
        print(f"  {lab}: no fits above R^2 > {THRESH}")
        return None
    ncol = min(NCOLS, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    color = COLORS.get(lab, '#4C72B0')
    for ax, f in zip(axes, fits):
        x, y = f['x'], f['y']
        ax.scatter(x, y, s=22, color=color, zorder=3, edgecolor='white', linewidth=0.5)
        if x.max() > x.min():                                    # OLS line (same fit as R^2)
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, b0 + b1 * xs, color='#333', lw=1.2, zorder=2)
        for xi, yi, nm in zip(x, y, f['sinks']):
            ax.annotate(nm, (xi, yi), fontsize=6, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')
        ax.set_title(f"{f['full_name']}\n{f['source']}·{f['behavior']}  R²={f['r2']:.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis('off')
    fig.supxlabel('mean firing to sink monkey (x)', fontsize=9)
    fig.supylabel('source monkey behavior toward sink (y)', fontsize=9)
    fig.suptitle(f"{lab}: top {n} fits with R² > {THRESH}  "
                 f"(sources={SOURCES}, behaviors={BEHAVIORS})", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'linear_fits_{lab}.png')
    fig.savefig(out, dpi=150)
    print(f"  saved -> {out}")
    return fig


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    lists = load_lists()
    want = [lab for lab, *_ in lists] if LIST == 'all' else [LIST]
    for lab, X, meta in lists:
        if lab not in want:
            continue
        fits = [f for f in collect_fits(lab, X, meta) if f['r2'] > THRESH]
        fits.sort(key=lambda f: f['r2'], reverse=True)
        print(f"{lab}: {len(fits)} fits > {THRESH}; plotting top {min(TOP_N, len(fits))}, "
              f"CSV top {min(CSV_TOP_N, len(fits))}")
        # CSV: top CSV_TOP_N fits with full session-qualified identity (arrays dropped)
        csv_rows = fits[:CSV_TOP_N]
        if csv_rows:
            cols = ('list', 'full_name', 'cell', 'location', 'date', 'round',
                    'source', 'behavior', 'r2')
            pd.DataFrame([{k: f[k] for k in cols} for f in csv_rows]).to_csv(
                os.path.join(OUTDIR, f'linear_fits_{lab}.csv'), index=False)
        plot_grid(fits[:TOP_N], lab)
    plt.show()


if __name__ == '__main__':
    main()
