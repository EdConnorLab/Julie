"""
build_filtered_cache.py -- rebuild sorted_spike_cache_filtered from the SI-sorted cache.

WHY THIS IS A MODULE AND NOT A NOTEBOOK CELL
--------------------------------------------
``sorted_spike_cache_filtered`` is the cache almost every analysis actually reads,
and most of them read its pickles STRAIGHT OFF DISK -- plot_psth, plot_psth_heatmap,
plot_temporal_rasters, population_analysis, the decoding and mixed-effects configs,
state_space, eye_pipeline_db, spike_count_connector. None of them go through a
SpikeSource, so none of them get the load-time window clip.

That makes this the one place where the window has to be baked in. The source cache
holds a 1000 ms pre-stimulus baseline; this writes the STRICT [onset, offset] view,
so every direct reader keeps seeing exactly what it saw before. If you need a
baseline, read the source cache through ``SISortedSpikeSource.with_pre_stimulus(...)``
rather than widening this one.

Rerun this whenever the sorting is rerun. The QC gate below is computed from spike
counts and firing rates, so a re-sort changes which neurons pass, and a filtered
cache left over from a previous sort will quietly disagree with its parent about
which units exist at all.

HOW TO RUN
----------
    cd src
    python -m analyses.build_filtered_cache            # rebuild everything
    python -m analyses.build_filtered_cache --dry-run  # report, write nothing

Or open it in PyCharm and hit Run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyses.spike_count import filter_good_neurons
from data_access.data_loader import normalize_monkey_names
from data_access.spike_window import (
    FILTERED_CACHE_SUBDIR, SORTED_CACHE_SUBDIR, STRICT, clip_spike_window,
)
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

# Per-condition coverage instead of a total-trial floor: require every one of the 37
# photos to have >= 6 reps. This targets the balanced coverage that correlation-distance
# RDM/RSA and state-space trajectories need directly, rather than a session total that
# can hide a starved photo. min_trial_count is left at 1; the per-condition rule is the
# active gate.
QC = dict(
    min_total_spikes=300,
    min_firing_rate_hz=1,
    min_trial_count=1,
    n_time_blocks=4,
    min_active_blocks=3,
    max_isi_violation_rate=0.02,
    refractory_ms=2.0,
    min_reps_per_condition=6,
    n_conditions=37,
)


def build_filtered_cache(monkey=SUBJECT_MONKEY, *,
                         source_subdir=SORTED_CACHE_SUBDIR,
                         output_subdir=FILTERED_CACHE_SUBDIR,
                         dry_run=False):
    root = Path(DATA_BASE_PATH) / monkey
    src, dst = root / source_subdir, root / output_subdir
    if not src.exists():
        raise FileNotFoundError(
            f"source cache not found: {src}. It is the single source of truth for the "
            f"SI-sorted spikes; generate it with analyze_sorted_spikes before filtering.")
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    print(f"source : {src}")
    print(f"output : {dst}{'   (DRY RUN, nothing written)' if dry_run else ''}")
    print(f"window : strict [onset, offset]  (source holds a pre-stimulus baseline)")

    kept_total = seen_total = 0
    empty = []
    for path in sorted(src.glob("*.pkl")):
        df = pd.read_pickle(path)
        # Strict window BEFORE the QC gate: filter_good_neurons derives firing rates
        # by dividing raw spike counts by the epoch duration, so a retained baseline
        # would inflate every rate by ~40% and let neurons through that should not be.
        df = clip_spike_window(normalize_monkey_names(df), STRICT,
                               cache_subdir=source_subdir)
        seen = df["NeuronID"].nunique()
        keep = filter_good_neurons(df, **QC)
        seen_total += seen
        kept_total += len(keep)
        if not keep:
            empty.append(path.name)
            print(f"  {path.name}: 0/{seen} neurons passed -- not written")
            continue
        print(f"  {path.name}: {len(keep)}/{seen} neurons passed")
        if not dry_run:
            df[df["NeuronID"].isin(keep)].to_pickle(dst / path.name)

    print(f"\n{kept_total}/{seen_total} neurons passed QC across "
          f"{len(list(src.glob('*.pkl')))} session(s)")
    if empty:
        # Say it out loud: a session with no survivors leaves no file, and a stale file
        # from a previous sort would otherwise sit there looking current.
        print(f"{len(empty)} session(s) produced no output: {', '.join(empty)}")
        if not dry_run:
            stale = [n for n in empty if (dst / n).exists()]
            if stale:
                print(f"WARNING: {len(stale)} of those still have an OLD file in "
                      f"{dst} from a previous run: {', '.join(stale)}. "
                      f"Delete them -- they no longer correspond to anything.")
    return kept_total


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--monkey", default=SUBJECT_MONKEY)
    p.add_argument("--source-subdir", default=SORTED_CACHE_SUBDIR)
    p.add_argument("--output-subdir", default=FILTERED_CACHE_SUBDIR)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be written, write nothing")
    a = p.parse_args()
    build_filtered_cache(a.monkey, source_subdir=a.source_subdir,
                         output_subdir=a.output_subdir, dry_run=a.dry_run)
