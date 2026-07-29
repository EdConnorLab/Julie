"""
regenerate_exploded_cache.py — rebuild exploded_spike_cache pkls for many sessions.

Run this after manually sorting units: it refreshes each session's canonical
exploded cache so the new units appear downstream.

Per session it decides what to do from what is on disk:

  sorted_spikes.pkl  compiled.pkl   action
  -----------------  ------------   ------------------------------------------
  absent             -              skip compile; compiled.pkl is never read
                                    (data_loader.load_raw_data gates on
                                    sorted_spikes.pkl existing)
  present            present        reuse the existing compiled.pkl
  present            absent         run compile_data, then rebuild

compile_data needs the lab DB (172.30.6.59) plus info.rhd / digitalin.dat /
notes.txt in the round directory, so it is only invoked when genuinely needed.

Sessions are taken from SESSIONS if set, otherwise discovered from a cache
directory (defaults to the canonical one; point it at a recovered cache to
rebuild exactly the sessions that used to be tracked).

Failures are collected per session so one bad round does not abort the run.

HOW TO RUN: open in PyCharm and click Run — edit the CONFIG block at the bottom.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from compile.compile_common import INTAN_BASE_PATH
from data_access.cache_utils import ExplodedSpikeCacheManager
from project_util import PROJECT_BASE_PATH, SUBJECT_MONKEY

SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_round_(\d+)\.pkl$")


def discover_sessions(cache_dir):
    """List (date, round_no) for every ``{date}_round_{n}.pkl`` in a cache dir."""
    out = []
    for path in sorted(Path(cache_dir).glob("*.pkl")):
        m = SESSION_RE.search(path.name)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def round_dir_for(date, round_no, reader, monkey=SUBJECT_MONKEY):
    """Intan round directory, using the metadata's Folder Name (never a guess)."""
    folder = reader.get_intan_folder_name_for_specific_round(date, round_no)
    return Path(INTAN_BASE_PATH) / monkey / str(date) / folder


def regenerate(sessions, *, monkey=SUBJECT_MONKEY, dry_run=False):
    """Rebuild the exploded cache for each (date, round_no). Returns a report dict."""
    reader = RecordingMetadataReader()
    mgr = ExplodedSpikeCacheManager(monkey=monkey)

    compiled, reused, unsorted, failed = [], [], [], []

    for date, round_no in sessions:
        tag = f"{date} round {round_no}"
        try:
            round_dir = round_dir_for(date, round_no, reader, monkey)
            has_sorted = (round_dir / "sorted_spikes.pkl").exists()
            has_compiled = (round_dir / "compiled.pkl").exists()

            if not has_sorted:
                unsorted.append(tag)
                action = "no sorted_spikes.pkl - compiled.pkl not needed"
            elif has_compiled:
                reused.append(tag)
                action = "reusing existing compiled.pkl"
            else:
                action = "compiling compiled.pkl"

            print(f"[{tag}] {action}")
            if dry_run:
                continue

            if has_sorted and not has_compiled:
                # imported lazily: needs the lab DB, so keep dry runs importable
                from compile.compile_posthoc_manually_sorted import compile_data
                compile_data(
                    experiment_name=round_dir.name,
                    day=dt.date.fromisoformat(date),
                )
                compiled.append(tag)

            df = mgr.load_or_compute(date, round_no, force_recompute=True)
            units = sorted({c for c in df["Channel"].astype(str).unique() if "_Unit" in c})
            print(f"    rebuilt: {df['NeuronID'].nunique()} neurons, "
                  f"{df['TaskField'].nunique()} trials, {len(units)} sorted unit(s)")

        except Exception as exc:  # keep going; report at the end
            failed.append((tag, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED - {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print(f"regenerated      : {len(sessions) - len(failed)}/{len(sessions)}")
    print(f"  compiled first : {len(compiled)}")
    print(f"  reused compiled: {len(reused)}")
    print(f"  never sorted   : {len(unsorted)}")
    if failed:
        print(f"failed           : {len(failed)}")
        for tag, err in failed:
            print(f"  {tag}: {err}")
    return {"compiled": compiled, "reused": reused,
            "unsorted": unsorted, "failed": failed}


# ===== Run directly in PyCharm — edit this block and hit Run ==================
if __name__ == "__main__":
    # Explicit list, or None to discover from CACHE_DIR.
    SESSIONS = None

    # Discovery source. The canonical cache rebuilds what you already have;
    # a recovered cache rebuilds exactly the sessions that used to be tracked.
    CACHE_DIR = Path(PROJECT_BASE_PATH) / SUBJECT_MONKEY / "exploded_spike_cache_gitrecovered"

    DRY_RUN = True   # True = print the plan without touching anything

    sessions = SESSIONS or discover_sessions(CACHE_DIR)
    print(f"{len(sessions)} session(s) from {CACHE_DIR if SESSIONS is None else 'SESSIONS'}\n")
    regenerate(sessions, dry_run=DRY_RUN)
