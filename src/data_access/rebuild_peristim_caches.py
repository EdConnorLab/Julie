"""
rebuild_peristim_caches.py -- build the pre-stimulus caches for every session.

The exploded and MUA pre-stimulus builders each do one session at a time, which
is fine for a raster but not for a rebuild: the caches they write are the single
source of truth every analysis reads, so they have to be built across the whole
session list in one pass. This is that pass.

    cd src
    python -m data_access.rebuild_peristim_caches                # exploded + MUA
    python -m data_access.rebuild_peristim_caches --what exploded
    python -m data_access.rebuild_peristim_caches --dry-run      # list the plan

Or open it in PyCharm and hit Run.

WHAT IT WRITES
    exploded_spike_cache_pre1000ms/     manual-sorted + unsorted, 1000 ms baseline
    threshold_mua_spike_cache_pre1000ms/  offline MAD/RMS MUA, 1000 ms baseline

The SI-sorted cache is NOT built here -- it comes from analyze_sorted_spikes on
the machine holding the sorter output, via run_analyze_in_batches.sh, which now
passes --pre-stimulus-time 1.0 and writes sorted_spike_cache_pre1000ms.

ONE BAD SESSION NEVER STOPS THE RUN. Each session is attempted independently and
the failures are listed at the end, because the failures are the point: the merge
in combine_unsorted_with_sorted now REFUSES a session whose two compiled.pkl
sources describe different rounds rather than silently concatenating them, so a
session that used to produce a quietly corrupt cache now shows up here by name.

A note on scope: the exploded cache is scoped to the InitialRegression sheet of
the recording metadata workbook, the same list the analyses iterate. A session
missing from that sheet gets Location='Unknown' and NeuronIDs to match, so add it
there first rather than passing it in by hand.
"""
from __future__ import annotations

import argparse
import traceback
from datetime import date as _date

import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from data_access.exploded_peristim_builder import build_exploded_peristim_cache
from data_access.mua_peristim_builder import build_mua_peristim_cache
from data_access.spike_window import CACHE_PRE_STIMULUS_TIME
from project_util import SUBJECT_MONKEY


def sessions_from_workbook():
    """(date, round) for every round in the InitialRegression sheet."""
    meta = RecordingMetadataReader().get_metadata_for_preliminary_analysis()
    out = []
    for _, r in meta.iterrows():
        try:
            out.append((pd.to_datetime(r["Date"]).strftime("%Y-%m-%d"), int(r["Round No."])))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def ensure_compiled(date, round_no, monkey=SUBJECT_MONKEY):
    """Compile the round folder's compiled.pkl if it is manually sorted and missing.

    Only manually-sorted sessions need it: it is what read_sorted_data windows the
    hand-sorted spike indices against. Sessions with no sorted_spikes.pkl take the
    unsorted path and never open it.
    """
    from pathlib import Path

    from compile.compile_common import INTAN_BASE_PATH
    from compile.compile_posthoc_manually_sorted import compile_data

    reader = RecordingMetadataReader()
    folder = reader.get_intan_folder_name_for_specific_round(date, round_no)
    round_dir = Path(INTAN_BASE_PATH) / monkey / date / folder
    if not (round_dir / "sorted_spikes.pkl").exists():
        return "not manually sorted"
    if (round_dir / "compiled.pkl").exists():
        return "compiled.pkl present"
    compile_data(experiment_name=folder, day=_date.fromisoformat(date))
    return "compiled.pkl built"


def rebuild(what="both", *, pre_stimulus_time=CACHE_PRE_STIMULUS_TIME,
            monkey=SUBJECT_MONKEY, sessions=None, force=False, dry_run=False,
            compile_missing=True, noise_method="mad", threshold_multiplier=4.0,
            refractory_ms=1.0):
    sessions = sessions or sessions_from_workbook()
    ms = int(round(pre_stimulus_time * 1000))
    print(f"{len(sessions)} session(s); pre-stimulus {ms} ms; building: {what}"
          + ("   (DRY RUN, nothing written)" if dry_run else ""))

    ok, skipped, failed = [], [], []
    for i, (date, round_no) in enumerate(sessions, 1):
        tag = f"{date} round {round_no}"
        print(f"\n[{i}/{len(sessions)}] {tag}")
        if dry_run:
            continue
        try:
            if what in ("exploded", "both"):
                if compile_missing:
                    print(f"  {ensure_compiled(date, round_no, monkey)}")
                df, path = build_exploded_peristim_cache(
                    date, round_no, pre_stimulus_time, monkey=monkey, force=force)
                if df is None:
                    skipped.append((tag, "exploded", "no trials"))
                else:
                    ok.append((tag, "exploded", df["NeuronID"].nunique()))
            if what in ("mua", "both"):
                df, path = build_mua_peristim_cache(
                    date, round_no, pre_stimulus_time, monkey=monkey, force=force,
                    noise_method=noise_method,
                    threshold_multiplier=threshold_multiplier,
                    refractory_ms=refractory_ms)
                if df is None:
                    skipped.append((tag, "mua", "no trials"))
                else:
                    ok.append((tag, "mua", df["NeuronID"].nunique()))
        except Exception as exc:
            failed.append((tag, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED - {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f" REBUILD SUMMARY  ({len(sessions)} session(s), pre-stimulus {ms} ms)")
    print("=" * 70)
    print(f"  built    : {len(ok)}")
    print(f"  skipped  : {len(skipped)}")
    print(f"  FAILED   : {len(failed)}")
    for tag, kind, why in skipped:
        print(f"    skip  {tag} [{kind}]: {why}")
    for tag, err in failed:
        print(f"    FAIL  {tag}: {err}")
    if failed:
        print("\n  A refusal mentioning 'share no task ids' means that session's two "
              "compiled.pkl sources describe different rounds -- fix the file, do not "
              "rerun past it.")
    return ok, skipped, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the pre-stimulus caches for every session.")
    p.add_argument("--what", default="both", choices=("exploded", "mua", "both"))
    p.add_argument("--pre-stimulus-time", type=float, default=CACHE_PRE_STIMULUS_TIME,
                   dest="pre_stimulus_time",
                   help=f"seconds of baseline (default {CACHE_PRE_STIMULUS_TIME})")
    p.add_argument("--monkey", default=SUBJECT_MONKEY)
    p.add_argument("--force", action="store_true", help="rebuild sessions already cached")
    p.add_argument("--dry-run", action="store_true", help="list the sessions, build nothing")
    p.add_argument("--no-compile-missing", action="store_false", dest="compile_missing",
                   help="do not compile a missing compiled.pkl for manually-sorted rounds")
    a = p.parse_args()
    rebuild(a.what, pre_stimulus_time=a.pre_stimulus_time, monkey=a.monkey,
            force=a.force, dry_run=a.dry_run, compile_missing=a.compile_missing)
