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
if _HERE not in sys.path:                     # sibling modules (grant_mua_matching, common)
    sys.path.insert(0, _HERE)

from analyses.spike_count import extract_spike_counts_from_windows   # noqa: E402
from data_access.spike_source import SISortedSpikeSource, ThresholdMUASpikeSource  # noqa: E402
from project_util import PROJECT_BASE_PATH                           # noqa: E402
from grant_mua_matching import (                                     # noqa: E402
    match_grant_cells_to_mua_neuronids, is_mua_name, summarize_problems,
    match_rows, MATCH_TABLE_COLUMNS,
)

# =====================================================================
# CONFIG
# =====================================================================
CACHE_SUBDIR = 'sorted_spike_cache_filtered'   # raw SI-sorted spikes (pre-QC-filtered)
GROUP = 'Zombies'                               # stimulus group used in the regression
MONKEY_NAME = ['7124', '69X', '72X', '94B', '110E', '67G', '81G', '143H', '87J', '151J']
SUBJECT = 6                                     # 81G, excluded as a stimulus column
ORDER9 = [m for i, m in enumerate(MONKEY_NAME) if i != SUBJECT]  # matches common.MONKEY_NAME order
GRANT_NCELLS = 74                               # grant xlsx: first 74 rows (== replicate_analysis.NCELLS)

# per-list spike sources (factories -> a fresh instance per use)
def _si_sorted_source():
    return SISortedSpikeSource(cache_subdir=CACHE_SUBDIR, pre_filtered=True)


def _mua_source():
    # offline MAD/RMS negative-crossing MUA; tune params to match the offline analysis
    return ThresholdMUASpikeSource(noise_method='mad', threshold_multiplier=4.0, refractory_ms=1.0)


# name -> {significance-windows pkl (repo-relative), spike source factory}
LISTS = {
    'KW':        {'pkl': 'Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pKW_passed.pkl',
                  'source': _si_sorted_source},
    'ANOVA':     {'pkl': 'Cortana/analysis_cache/si_sorted_Zombies_significant_windows_pANOVA_passed.pkl',
                  'source': _si_sorted_source},
    'MUA_KW':    {'pkl': 'Cortana/analysis_cache/threshold_mua_Zombies_significant_windows_pKW_passed.pkl',
                  'source': _mua_source},
    'MUA_ANOVA': {'pkl': 'Cortana/analysis_cache/threshold_mua_Zombies_significant_windows_pANOVA_passed.pkl',
                  'source': _mua_source},
}
OUTDIR = os.path.join(_HERE, 'regenerated_inputs')
# =====================================================================

KEY = ['NeuronID', 'WindowStart_ms', 'WindowEnd_ms']


def _windows_for(list_name):
    """Load a significance-windows pkl and normalize Date/Round No. dtypes so the
    spike-cache label (f'{Date}_round_{Round No.}') matches the cache filenames."""
    rel = LISTS[list_name]['pkl']
    path = os.path.join(PROJECT_BASE_PATH, rel)
    if not os.path.exists(path):                       # fall back to this checkout
        path = os.path.abspath(os.path.join(_HERE, '..', '..', '..', rel))
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


def extract_per_trial(list_name):
    """Per-trial SpikeCount for every (neuron, window, stimulus monkey) in the
    list, restricted to GROUP. This is the one expensive step; both the long
    (mean) and wide (per-trial lists) builders reuse its output."""
    source = LISTS[list_name]['source']()
    windows = _keep_loadable_sessions(_windows_for(list_name), source)
    per_trial = extract_spike_counts_from_windows(windows, source)
    return per_trial[per_trial['MonkeyGroup'] == GROUP].copy()


def build_long_counts(list_name, per_trial=None):
    """Long DataFrame with mean spike count/rate per (neuron, window, monkey).

    Columns: NeuronID, MonkeyName, MonkeyGroup, WindowStart_ms, WindowEnd_ms,
             n_trials, MeanSpikeCount, MeanSpikeRate
    """
    pt = extract_per_trial(list_name) if per_trial is None else per_trial
    g = pt.groupby(KEY + ['MonkeyName', 'MonkeyGroup'], as_index=False).agg(
        n_trials=('SpikeCount', 'size'),
        MeanSpikeCount=('SpikeCount', 'mean'),
    )
    dur_s = (g['WindowEnd_ms'] - g['WindowStart_ms']) / 1000.0
    g['MeanSpikeRate'] = g['MeanSpikeCount'] / dur_s
    return g


def build_wide_counts(list_name, per_trial=None):
    """Wide DataFrame in Ed's grant xlsx format: one row per (neuron, window),
    with a column per stimulus monkey holding the list of per-trial spike counts.
    Directly consumable by common.load_data. Mirrors generating_files_for_ed."""
    pt = extract_per_trial(list_name) if per_trial is None else per_trial
    pt = pt.copy()
    parts = pt['NeuronID'].str.split('_', n=3, expand=True)   # Location, Date, Round, Cell
    pt['Date'], pt['Round No.'], pt['Cell'] = parts[1], parts[2], parts[3]
    pt['Time Window'] = list(zip(pt['WindowStart_ms'], pt['WindowEnd_ms']))
    idx = ['Date', 'Round No.', 'Cell', 'Time Window']
    grp = pt.groupby(idx + ['MonkeyName'])['SpikeCount'].apply(list).reset_index()
    wide = grp.pivot(index=idx, columns='MonkeyName', values='SpikeCount').reset_index()
    # order the 9 stimulus-monkey columns as ORDER9 (extras, if any, kept after)
    present = [m for m in ORDER9 if m in wide.columns]
    extras = [c for c in wide.columns if c not in idx + present]
    return wide[idx + present + extras]


def load_data_from_cache(list_name, value='count', per_trial=None):
    """Drop-in replacement for the old load_data_kw. Returns (X_mean, meta).

    X_mean: (n_combos, 9) mean spike count (value='count') or rate (value='rate'),
            columns ordered as ORDER9 (== common.MONKEY_NAME minus subject 81G).
    meta:   DataFrame with NeuronID/window and a 'Cell' + 'Time Window' column.
    Only (neuron, window) combos with all 9 non-subject Zombies present are kept.
    """
    col = 'MeanSpikeCount' if value == 'count' else 'MeanSpikeRate'
    long = build_long_counts(list_name, per_trial=per_trial)
    piv = long.pivot_table(index=KEY, columns='MonkeyName', values=col)
    piv = piv.reindex(columns=ORDER9).dropna()          # require all 9 stimulus monkeys
    X_mean = piv.to_numpy(float)
    meta = piv.reset_index()[KEY].copy()
    meta['Cell'] = meta['NeuronID']
    meta['Time Window'] = list(zip(meta['WindowStart_ms'], meta['WindowEnd_ms']))
    return X_mean, meta


# =====================================================================
# GRANT MUA cells, recomputed from the threshold-MUA cache (matched by channel)
# =====================================================================
def _mua_session_ids_fn(source):
    """Return session_ids_fn(date, round) -> [NeuronID] | None, backed by `source` and
    caching each session so its cache is read once here (extract_spike_counts_from_windows
    re-reads it, but the threshold-MUA cache is on disk so that second read is cheap)."""
    sessions = {}

    def fn(date, round_no):
        key = (date, int(round_no))
        if key not in sessions:
            df = source.load(date, int(round_no))
            if df is None or getattr(df, 'empty', True):
                sessions[key] = None
            else:
                sessions[key] = df['NeuronID'].astype(str).unique().tolist()
        return sessions[key]

    return fn


def _report_problems(problems):
    """Print grouped, per-cell warnings for anything the matcher could not resolve."""
    if not problems:
        return
    print(f"[grant-mua][warn] {len(problems)} cell(s) with issues: {summarize_problems(problems)}")
    for p in problems:
        c = p['cell']
        print(f"    [{p['kind']}] {c.date} round{c.round_no} {c.match_value}: {p['detail']}")


def _default_match_csv():
    return os.path.join(_HERE, 'output', 'grant_mua_match_table.csv')   # 'output/' is gitignored


def _write_match_rows_csv(rows, out_csv=None):
    out_csv = out_csv or _default_match_csv()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows, columns=list(MATCH_TABLE_COLUMNS)).to_csv(out_csv, index=False)
    return out_csv


def export_grant_mua_match_table(out_csv=None):
    """Match the grant's UNSORTED (MUA) cells to their threshold-MUA NeuronIDs and write a
    CSV you can eyeball: one row per grant MUA cell with its matched NeuronID and status
    (matched / no_match / ambiguous / missing_session / no_window). Matching ONLY -- no
    count extraction -- so it is fast. Same match logic Run B uses, so the CSV is an exact
    audit of the mapping. Returns (rows, path)."""
    from analyses.zombies_raster_review.unit_lists import load_mixed_manual_requests
    import common

    reqs = load_mixed_manual_requests(common.HIS_XLSX)
    reqs = reqs[:GRANT_NCELLS] if GRANT_NCELLS and GRANT_NCELLS > 0 else reqs
    mua_reqs = [r for r in reqs if is_mua_name(r.match_value)]
    print(f"[grant-mua] {len(reqs)} grant rows -> {len(mua_reqs)} unsorted (MUA) cells")
    source = _mua_source()
    matched, problems = match_grant_cells_to_mua_neuronids(mua_reqs, _mua_session_ids_fn(source))
    _report_problems(problems)
    rows = match_rows(matched, problems)
    path = _write_match_rows_csv(rows, out_csv)
    print(f"[grant-mua] {len(rows)} grant MUA cells ({len(matched)} matched, "
          f"{len(problems)} unmatched) -> {path}")
    return rows, path


def load_grant_mua_from_cache(value='count'):
    """Rebuild replicate_analysis's input for the grant's UNSORTED (MUA) cells, with spike
    counts recomputed from the threshold-MUA cache over each cell's own grant time window.

    Returns (X_mean (n,9), meta) in the same shape as load_data_from_cache, so it is a
    drop-in for DATA_SOURCE='cache_mua_grantcells'. Matching never guesses Location -- the
    NeuronID is taken verbatim from the threshold-MUA source (see grant_mua_matching). Any
    cell that cannot be matched/completed is reported; the final line reconciles the used
    count against the grant MUA list so Run A vs Run B cell counts can be compared.
    """
    from analyses.zombies_raster_review.unit_lists import load_mixed_manual_requests
    import common

    reqs = load_mixed_manual_requests(common.HIS_XLSX)
    reqs = reqs[:GRANT_NCELLS] if GRANT_NCELLS and GRANT_NCELLS > 0 else reqs
    mua_reqs = [r for r in reqs if is_mua_name(r.match_value)]
    print(f"[grant-mua] {len(reqs)} grant rows -> {len(mua_reqs)} unsorted (MUA) cells "
          f"({len(reqs) - len(mua_reqs)} sorted units skipped)")

    source = _mua_source()
    matched, problems = match_grant_cells_to_mua_neuronids(mua_reqs, _mua_session_ids_fn(source))
    _report_problems(problems)
    csv_path = _write_match_rows_csv(match_rows(matched, problems))   # audit table, even on failure
    print(f"[grant-mua] match table -> {csv_path}")
    if not matched:
        raise RuntimeError("[grant-mua] no grant MUA cell matched the threshold-MUA cache -- "
                           "check that the cache/recordings exist for these sessions")

    windows = pd.DataFrame([{
        'NeuronID': m['neuron_id'],
        'WindowStart_ms': m['window_ms'][0],
        'WindowEnd_ms': m['window_ms'][1],
        'Date': m['cell'].date,
        'Round No.': int(m['cell'].round_no),
    } for m in matched])
    dup = windows.duplicated(subset=KEY, keep=False)
    if dup.any():
        print(f"[grant-mua][warn] {int(dup.sum())} duplicate (NeuronID, window) row(s) collapsed:")
        print(windows.loc[dup, KEY].to_string(index=False))
        windows = windows.drop_duplicates(subset=KEY)

    per_trial = extract_spike_counts_from_windows(windows, source)
    per_trial = per_trial[per_trial['MonkeyGroup'] == GROUP].copy()

    col = 'MeanSpikeCount' if value == 'count' else 'MeanSpikeRate'
    long = build_long_counts(None, per_trial=per_trial)      # reuse the mean count/rate builder
    piv = long.pivot_table(index=KEY, columns='MonkeyName', values=col)

    full = piv.reindex(columns=ORDER9)                       # require all 9 stimulus monkeys
    incomplete = full[full.isna().any(axis=1)]
    if len(incomplete):
        print(f"[grant-mua][warn] {len(incomplete)} matched cell(s) dropped for missing "
              f"stimulus monkeys:")
        for idx, row in incomplete.iterrows():
            miss = [m for m in ORDER9 if pd.isna(row[m])]
            print(f"        {idx[0]}  win={idx[1]}-{idx[2]}ms  missing={miss}")
    full = full.dropna()

    X_mean = full.to_numpy(float)
    meta = full.reset_index()[KEY].copy()
    meta['Cell'] = meta['NeuronID']
    meta['Time Window'] = list(zip(meta['WindowStart_ms'], meta['WindowEnd_ms']))
    print(f"[grant-mua] matched {len(matched)}/{len(mua_reqs)} MUA cells; used {len(meta)} "
          f"after requiring all {len(ORDER9)} stimulus monkeys "
          f"(Run A 'grant_xlsx'+multiunit uses {len(mua_reqs)} cells).")
    return X_mean, meta


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for name in LISTS:
        print(f"\n=== {name} ===")
        per_trial = extract_per_trial(name)                 # expensive step, run once
        long = build_long_counts(name, per_trial=per_trial)
        wide = build_wide_counts(name, per_trial=per_trial)
        X, meta = load_data_from_cache(name, per_trial=per_trial)
        print(f"  X_mean {X.shape}  ({len(meta)} neuron x window combos, "
              f"{meta['NeuronID'].nunique()} unique neurons, all {len(ORDER9)} stimulus monkeys present)")

        long_pkl = os.path.join(OUTDIR, f'{name}_zombies_windowed_spike_counts.pkl')
        long.to_pickle(long_pkl)
        long.to_csv(long_pkl.replace('.pkl', '.csv'), index=False)
        wide_xlsx = os.path.join(OUTDIR, f'{name}_zombies_spike_counts_wide.xlsx')
        wide.to_excel(wide_xlsx, index=False)
        print(f"  saved long  -> {long_pkl}")
        print(f"  saved wide  -> {wide_xlsx}  (Ed's grant format; load via common.load_data)")
    print("\nUse in replicate_analysis:  set DATA_SOURCE = 'cache_kw' (or 'cache_anova')")


if __name__ == '__main__':
    main()
