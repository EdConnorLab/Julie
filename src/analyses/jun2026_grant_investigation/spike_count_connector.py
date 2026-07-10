"""
Connector: build replicate_analysis's input (mean spike COUNTS per cell x
stimulus monkey) directly from the analysis_cache significance pkls + the
SI-sorted raw spike cache. No dependence on the deleted *_mean_spike_rates files.

Pipeline (mirrors notebooks/generating_files_for_ed.ipynb, "anova passed windows"):

    windows pkl (analysis_cache/..._significant_windows_p{KW,ANOVA}_passed.pkl)
        has NeuronID / WindowStart_ms / WindowEnd_ms / Date / Round No.
      -> extract_spike_counts_from_windows(windows, SISortedSpikeSource(...))
        -> per-trial SpikeCount for every stimulus monkey shown, in each window
      -> keep MonkeyGroup == 'Zombies'
      -> mean SpikeCount over trials, per (NeuronID, window, MonkeyName)
      -> pivot to (n_combos, 9) with columns = the 9 non-subject Zombies

This is what replicate_analysis needs. R^2 is scale-invariant, so mean COUNT and
mean RATE give identical fits; MeanSpikeRate is emitted too for convenience.

Drop-in use inside replicate_analysis (replaces the old load_data_kw call):

    from spike_count_connector import load_data_from_cache
    X_mean, df = load_data_from_cache('KW')      # or 'ANOVA'

Or regenerate long-format pkls (same schema the old load_data_kw read):

    python spike_count_connector.py              # builds KW + ANOVA, saves + prints

Requirements: run with the repo `src/` on the path (PyCharm sources root, or the
bootstrap below). Relies on project_util.PROJECT_BASE_PATH pointing at the repo
(the raw cache lives at <PROJECT_BASE_PATH>/Cortana/sorted_spike_cache_filtered).
"""

import os
import sys
import numpy as np
import pandas as pd

# --- make the repo's src/ importable when run as a plain script ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..'))            # .../Julie/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analyses.spike_count import extract_spike_counts_from_windows   # noqa: E402
from data_access.spike_source import SISortedSpikeSource             # noqa: E402
from project_util import PROJECT_BASE_PATH                           # noqa: E402

# =====================================================================
# CONFIG
# =====================================================================
CACHE_SUBDIR = 'sorted_spike_cache_filtered'   # raw SI-sorted spikes (pre-QC-filtered)
GROUP = 'Zombies'                               # stimulus group used in the regression
MONKEY_NAME = ['7124', '69X', '72X', '94B', '110E', '67G', '81G', '143H', '87J', '151J']
SUBJECT = 6                                     # 81G, excluded as a stimulus column
ORDER9 = [m for i, m in enumerate(MONKEY_NAME) if i != SUBJECT]  # matches common.MONKEY_NAME order

# name -> significance-windows pkl (relative to the repo root)
LISTS = {
    'KW':    'Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pKW_passed.pkl',
    'ANOVA': 'Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pANOVA_passed.pkl',
}
OUTDIR = os.path.join(_HERE, 'regenerated_inputs')
# =====================================================================

KEY = ['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']


def _windows_for(list_name):
    """Load a significance-windows pkl and normalize Date/Round No. dtypes so the
    spike-cache label (f'{Date}_round_{Round No.}') matches the cache filenames."""
    path = os.path.join(PROJECT_BASE_PATH, LISTS[list_name])
    if not os.path.exists(path):                       # fall back to this checkout
        path = os.path.abspath(os.path.join(_HERE, '..', '..', '..', LISTS[list_name]))
    w = pd.read_pickle(path).copy()
    missing = {'NeuronID', 'WindowStart_ms', 'WindowEnd_ms', 'Date', 'Round No.'} - set(w.columns)
    if missing:
        raise ValueError(f"{list_name} windows pkl missing columns: {missing}")
    w['Date'] = pd.to_datetime(w['Date']).dt.strftime('%Y-%m-%d')   # -> '2023-09-26'
    w['Round No.'] = w['Round No.'].astype(int)
    return w[KEY + ['Date', 'Round No.']].drop_duplicates()


def _keep_loadable_sessions(w, source):
    """Drop windows whose session has no sorted cache (pre_filtered -> load None),
    so extract_spike_counts_from_windows never dereferences a missing session."""
    ok = []
    for (d, r), _ in w.groupby(['Date', 'Round No.']):
        df = source.load(d, int(r))
        ok.append((d, int(r), df is not None and not getattr(df, 'empty', True)))
    good = {(d, r) for d, r, k in ok if k}
    dropped = [(d, r) for d, r, k in ok if not k]
    if dropped:
        print(f"  [warn] no sorted cache for sessions {dropped} -- their windows are skipped")
    return w[[(d, int(r)) in good for d, r in zip(w['Date'], w['Round No.'])]]


def build_long_counts(list_name):
    """Return a long DataFrame with mean spike count/rate per (neuron, window,
    monkey) for the given list, restricted to the stimulus GROUP.

    Columns: NeuronID, MonkeyName, MonkeyGroup, WindowStart_ms, WindowEnd_ms,
             n_trials, MeanSpikeCount, MeanSpikeRate
    """
    source = SISortedSpikeSource(cache_subdir=CACHE_SUBDIR, pre_filtered=True)
    windows = _keep_loadable_sessions(_windows_for(list_name), source)

    per_trial = extract_spike_counts_from_windows(windows, source)   # per-trial SpikeCount
    per_trial = per_trial[per_trial['MonkeyGroup'] == GROUP]

    g = per_trial.groupby(KEY + ['MonkeyName', 'MonkeyGroup'], as_index=False).agg(
        n_trials=('SpikeCount', 'size'),
        MeanSpikeCount=('SpikeCount', 'mean'),
    )
    dur_s = (g['WindowEnd_ms'] - g['WindowStart_ms']) / 1000.0
    g['MeanSpikeRate'] = g['MeanSpikeCount'] / dur_s
    return g


def load_data_from_cache(list_name, value='count'):
    """Drop-in replacement for the old load_data_kw. Returns (X_mean, meta).

    X_mean: (n_combos, 9) mean spike count (value='count') or rate (value='rate'),
            columns ordered as ORDER9 (== common.MONKEY_NAME minus subject 81G).
    meta:   DataFrame with NeuronID/window and a 'Cell' + 'Time Window' column.
    Only (neuron, window) combos with all 9 non-subject Zombies present are kept.
    """
    col = 'MeanSpikeCount' if value == 'count' else 'MeanSpikeRate'
    long = build_long_counts(list_name)
    piv = long.pivot_table(index=KEY, columns='MonkeyName', values=col)
    piv = piv.reindex(columns=ORDER9).dropna()          # require all 9 stimulus monkeys
    X_mean = piv.to_numpy(float)
    meta = piv.reset_index()[KEY].copy()
    meta['Cell'] = meta['NeuronID']
    meta['Time Window'] = list(zip(meta['WindowStart_ms'], meta['WindowEnd_ms']))
    return X_mean, meta


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for name in LISTS:
        print(f"\n=== {name} ===")
        long = build_long_counts(name)
        X, meta = load_data_from_cache(name)
        n_neurons = meta['NeuronID'].nunique()
        print(f"  built X_mean {X.shape}  ({len(meta)} neuron x window combos, "
              f"{n_neurons} unique neurons, all {len(ORDER9)} stimulus monkeys present)")
        # save long table (superset schema; load_data_kw reads 'MeanSpikeRate')
        out = os.path.join(OUTDIR, f'{name}_zombies_windowed_spike_counts.pkl')
        long.to_pickle(out)
        long.to_csv(out.replace('.pkl', '.csv'), index=False)
        print(f"  saved -> {out}")
    print("\nUse in replicate_analysis:")
    print("    from spike_count_connector import load_data_from_cache")
    print("    X_mean, df = load_data_from_cache('KW')   # or 'ANOVA'")


if __name__ == '__main__':
    main()
