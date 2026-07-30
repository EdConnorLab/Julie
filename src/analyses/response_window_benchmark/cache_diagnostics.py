"""
cache_diagnostics.py — is ``NeuronID`` the SAME neuron in two different caches?

Why this exists
---------------
The same label can mean different neurons in different SI caches. Unit numbering
in ``analyze_sorted_spikes.find_consensus_units`` is assigned by *order of
appearance* within a sorting run::

    chan_count[chan_idx] = chan_count.get(chan_idx, 0) + 1
    'Channel': f"{channel}_Unit {chan_count[chan_idx]}"

so a fresh consensus sort (TDC/MS5/KS4 agreement) can hand ``C_014_Unit 1`` to a
different physical unit than a previous run did. On top of that, the older caches
carried a channel-labelling bug that was repaired *after the fact* by
``zombies_raster_review.relabel_si_channels``, while a newly built cache gets the
fixed labelling natively. Two caches can therefore disagree about which neuron a
NeuronID names — silently.

That matters here because the answer key was annotated from rasters drawn out of
``sorted_spike_cache_filtered``, but the benchmark reads
``sorted_spike_cache_pre1000ms``. If the labels disagree, the benchmark scores a
detector on a *different neuron* than the one the window was drawn for, and the
scoreboard is meaningless.

What it does
------------
For one session, load both caches and:

* report which NeuronIDs are in A only / B only / both,
* per unit, its trial count, spike count and mean rate in each cache,
* flag same-named units whose firing rates disagree (the smoking gun),
* **coincidence-match** units across the caches by spike time (reusing
  ``zombies_raster_review.coincidence_match``), which identifies the same
  physical neuron regardless of what it is called,
* emit a remap table ``A NeuronID -> B NeuronID`` you can apply to the answer key.

Spike times in both caches come from the same recording clock, so coincidence is
a valid identity test. By default only post-onset spikes are compared, so the
pre-stim cache's extra baseline spikes don't skew the ratio.

Run::

    cd src
    python -m analyses.response_window_benchmark.cache_diagnostics \
        --date 2023-09-26 --round 2
    # or every session in the answer key:
    python -m analyses.response_window_benchmark.cache_diagnostics \
        --answer-key ../response_window_test/window_test.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analyses.zombies_raster_review.coincidence_match import (          # noqa: E402
    session_units, match_units_across_sources, DEFAULT_WINDOW_MS,
)

REFERENCE_CACHE = "sorted_spike_cache_filtered"     # what the answer key was drawn from
PRESTIM_CACHE = "sorted_spike_cache_pre1000ms"      # what the benchmark reads


def _load(cache_subdir: str, date: str, round_no: int):
    from data_access.spike_source import SISortedSpikeSource
    try:
        return SISortedSpikeSource(cache_subdir=cache_subdir, pre_filtered=True).load(date, round_no)
    except Exception as e:
        print(f"[cache-diag] {cache_subdir} {date} r{round_no}: load failed ({e})")
        return None


def _post_only(df: pd.DataFrame) -> pd.DataFrame:
    """Trim each trial's spikes to >= stimulus onset, so a pre-stim cache and a
    strict cache are compared over the same interval."""
    out = df.copy(deep=True)
    trimmed = []
    for _, r in out.iterrows():
        s = np.asarray(list(r["SpikeTimes"]), dtype=float)
        onset = r["EpochStartStop"][0]
        trimmed.append(s[s >= onset])
    out["SpikeTimes"] = trimmed
    return out


def unit_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-NeuronID trial count, spike count and mean rate over the epoch."""
    rows = []
    for nid, g in df.groupby(df["NeuronID"].astype(str)):
        n_spikes = int(sum(len(s) for s in g["SpikeTimes"]))
        dur = float(sum(stop - start for start, stop in g["EpochStartStop"]))
        rows.append({"NeuronID": nid, "n_trials": len(g), "n_spikes": n_spikes,
                     "rate_hz": (n_spikes / dur) if dur > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("NeuronID").reset_index(drop=True)


def compare_session(
    date: str, round_no: int, *,
    cache_a: str = REFERENCE_CACHE, cache_b: str = PRESTIM_CACHE,
    rate_tol: float = 2.0, window_ms: float = DEFAULT_WINDOW_MS,
    coincidence_threshold: float = 0.2, ratio_threshold: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare one session across two caches. Returns (name_table, match_table)."""
    da, db = _load(cache_a, date, round_no), _load(cache_b, date, round_no)
    if da is None or getattr(da, "empty", True) or db is None or getattr(db, "empty", True):
        print(f"[cache-diag] {date} r{round_no}: need BOTH caches; skipping")
        return pd.DataFrame(), pd.DataFrame()

    da_p, db_p = _post_only(da), _post_only(db)
    sa, sb = unit_stats(da_p), unit_stats(db_p)

    # --- by NAME -----------------------------------------------------------
    name = sa.merge(sb, on="NeuronID", how="outer", suffixes=("_A", "_B"), indicator=True)
    name["Date"], name["Round No."] = date, round_no
    name["where"] = name["_merge"].map({"left_only": f"{cache_a} only",
                                        "right_only": f"{cache_b} only",
                                        "both": "both"})
    name = name.drop(columns=["_merge"])
    both = name[name["where"] == "both"].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = both["rate_hz_A"] / both["rate_hz_B"].replace(0, np.nan)
    both["rate_ratio"] = ratio
    both["RATE_MISMATCH"] = (ratio > rate_tol) | (ratio < 1.0 / rate_tol) | ratio.isna()
    name = name.merge(both[["NeuronID", "rate_ratio", "RATE_MISMATCH"]], on="NeuronID", how="left")

    # --- by SPIKE TRAIN (identity regardless of label) ----------------------
    ua, ub = session_units(da_p, "NeuronID"), session_units(db_p, "NeuronID")
    # Both caches come from the SAME recording, so a true re-labelling of one
    # neuron should be near-perfectly coincident. The chance-ratio guard (tuned
    # for matching across different sorters) is relaxed here, since a high-rate
    # unit has high chance coincidence and would otherwise be rejected.
    matches = match_units_across_sources(
        ua, ub, window_ms=window_ms,
        coincidence_threshold=coincidence_threshold, ratio_threshold=ratio_threshold)
    best: Dict[str, tuple] = {}
    for m in matches:
        a_id, b_id = getattr(m, "mixed_id", None), getattr(m, "si_id", None)
        score = float(getattr(m, "coincidence", 0.0) or 0.0)
        if a_id is None or b_id is None:
            continue
        if a_id not in best or score > best[a_id][1]:
            best[a_id] = (b_id, score)
    mrows = []
    for a_id in sorted(ua):
        b_id, score = best.get(a_id, (None, np.nan))
        mrows.append({"Date": date, "Round No.": round_no,
                      f"{cache_a}_NeuronID": a_id,
                      f"best_match_in_{cache_b}": b_id,
                      "coincidence": score,
                      "NAME_AGREES": (b_id == a_id) if b_id else False})
    match = pd.DataFrame(mrows)

    n_mismatch = int(name["RATE_MISMATCH"].fillna(False).sum())
    n_disagree = int((~match["NAME_AGREES"]).sum())
    print(f"[cache-diag] {date} r{round_no}: {len(sa)} units in {cache_a}, "
          f"{len(sb)} in {cache_b}; {n_mismatch} same-named unit(s) with >{rate_tol}x "
          f"rate mismatch; {n_disagree}/{len(match)} unit(s) whose best spike-train "
          f"match has a DIFFERENT name")
    return name, match


def run(sessions: List[Tuple[str, int]], out_dir: str, **kw) -> None:
    names, matches = [], []
    for date, rnd in sessions:
        n, m = compare_session(date, int(rnd), **kw)
        if not n.empty:
            names.append(n)
        if not m.empty:
            matches.append(m)
    os.makedirs(out_dir, exist_ok=True)
    if names:
        p = os.path.join(out_dir, "cache_compare_by_name.csv")
        pd.concat(names, ignore_index=True).to_csv(p, index=False)
        print(f"[cache-diag] by-name comparison -> {p}")
    if matches:
        p = os.path.join(out_dir, "cache_unit_remap.csv")
        allm = pd.concat(matches, ignore_index=True)
        allm.to_csv(p, index=False)
        print(f"[cache-diag] spike-train remap  -> {p}")
        bad = allm[~allm["NAME_AGREES"]]
        if len(bad):
            print(f"\n[cache-diag] {len(bad)}/{len(allm)} unit(s) are named differently "
                  f"across caches — the answer key CANNOT be applied by NeuronID alone:")
            print(bad.head(25).to_string(index=False))


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date")
    p.add_argument("--round", type=int, dest="round_no")
    p.add_argument("--answer-key", help="compare every session named in this xlsx")
    p.add_argument("--cache-a", default=REFERENCE_CACHE)
    p.add_argument("--cache-b", default=PRESTIM_CACHE)
    p.add_argument("--rate-tol", type=float, default=2.0)
    p.add_argument("--coincidence", type=float, default=0.2)
    p.add_argument("--ratio", type=float, default=1.5,
                   help="chance-ratio floor; lower it if high-rate units go unmatched")
    p.add_argument("--out", default=os.path.join(_HERE, "output"))
    a = p.parse_args(argv)

    if a.answer_key:
        from analyses.response_window_benchmark.answer_key import load_answer_key
        cands, _, _ = load_answer_key(a.answer_key)
        sessions = sorted({(r["Date"], int(r["Round No."])) for _, r in cands.iterrows()})
    elif a.date and a.round_no is not None:
        sessions = [(a.date, a.round_no)]
    else:
        p.error("give --date/--round or --answer-key")
    run(sessions, a.out, cache_a=a.cache_a, cache_b=a.cache_b, rate_tol=a.rate_tol,
        coincidence_threshold=a.coincidence, ratio_threshold=a.ratio)


if __name__ == "__main__":
    _cli()
