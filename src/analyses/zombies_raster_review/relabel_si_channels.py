"""
relabel_si_channels.py — repair the mis-assigned channel names in an SI sorted
cache (the bug diagnosed by ``validate_si_channel_labels.py``).

The SI cache's ``SpikeTimes`` are correct, but each unit's channel NAME was
computed by ``analyze_sorted_spikes.py`` from a channel index used as if it were
the Intan native number — wrong, because the recording is not in native order
(only 24 of 32 channels; read index != native). This rewrites ``Channel`` /
``BaseChannel`` / ``NeuronID`` to the channel where each unit's spikes ACTUALLY
peak — measured from the recording, the same proven method
``validate_si_channel_labels`` uses — and leaves ``SpikeTimes`` untouched. So no
analysis result changes; only the labels are corrected.

Safety
------
DRY-RUN by default: it prints the old→new mapping and writes nothing. Pass
``--apply`` to rewrite the cache pickles (each is copied to ``<name>.bak`` first).
``--mapping-csv PATH`` writes the old→new ``NeuronID`` map so you can update
``zombies_si_sorted_anova_passed.csv`` and anything else keyed by ``NeuronID``.

Works on any cache subdir — run it on ``sorted_spike_cache`` AND, once you point
it there, ``sorted_spike_cache_filtered``::

    cd src
    # preview one session
    python -m analyses.zombies_raster_review.relabel_si_channels --date 2023-09-26 --round 2
    # preview the whole cache
    python -m analyses.zombies_raster_review.relabel_si_channels
    # actually write it, and dump the id map
    python -m analyses.zombies_raster_review.relabel_si_channels --apply --mapping-csv remap.csv
    # the filtered cache used by the analyses
    python -m analyses.zombies_raster_review.relabel_si_channels \
        --cache-subdir sorted_spike_cache_filtered --apply --mapping-csv remap_filtered.csv
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REQUIRED_COLS = {"NeuronID", "Channel", "BaseChannel", "SpikeTimes",
                 "Date", "Round No.", "Location"}


def _norm_channel(text) -> str:
    """Any channel spelling → ``C_020`` (underscore, 3-digit)."""
    tok = str(text).split("Channel.")[-1].split("_Unit")[0].replace("-", "_").strip()
    if tok.startswith("C_") and tok[2:].isdigit():
        return f"C_{int(tok[2:]):03d}"
    return tok


def _true_base_channel(unit_df, voltages, sample_rate) -> Optional[str]:
    """The channel where this unit's spikes actually peak, e.g. ``"C_020"``."""
    from analyses.zombies_raster_review.waveform_footprint import unit_footprint
    fp = unit_footprint(unit_df, voltages, sample_rate, "u")
    return _norm_channel(fp.peak_channel) if fp is not None else None


def relabel_dataframe(df: pd.DataFrame, voltages, sample_rate
                      ) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Return ``(new_df, {old_neuron_id: new_neuron_id}, unresolved_ids)``.

    Each unit's true peak channel is measured; units are then renumbered
    ``_Unit 1, 2, …`` per true channel in a deterministic (sorted) order.
    """
    from clat.intan.channels import Channel  # local: needs clat

    true_base: Dict[str, Optional[str]] = {}
    for nid, udf in df.groupby(df["NeuronID"].astype(str)):
        true_base[nid] = _true_base_channel(udf, voltages, sample_rate)

    counter: Dict[str, int] = {}
    remap: Dict[str, dict] = {}
    unresolved: List[str] = []
    for nid in sorted(true_base):
        base = true_base[nid]
        if base is None:
            unresolved.append(nid)
            continue
        counter[base] = counter.get(base, 0) + 1
        row = df[df["NeuronID"].astype(str) == nid].iloc[0]
        chan_unit = f"Channel.{base}_Unit {counter[base]}"
        new_nid = f"{row['Location']}_{row['Date']}_{row['Round No.']}_{chan_unit}"
        remap[nid] = {"base_enum": Channel[base], "channel": chan_unit, "neuron_id": new_nid}

    new_df = df.copy()
    ids = new_df["NeuronID"].astype(str)
    new_df["BaseChannel"] = [remap[i]["base_enum"] if i in remap else b
                             for i, b in zip(ids, new_df["BaseChannel"])]
    new_df["Channel"] = [remap[i]["channel"] if i in remap else c
                         for i, c in zip(ids, new_df["Channel"])]
    new_df["NeuronID"] = [remap[i]["neuron_id"] if i in remap else i for i in ids]

    id_map = {old: v["neuron_id"] for old, v in remap.items()}
    return new_df, id_map, unresolved


def _cache_dir(cache_subdir: str, monkey: Optional[str]) -> Path:
    from data_access.cache_utils import SortedSpikeCacheManager
    mgr = (SortedSpikeCacheManager(monkey, cache_subdir=cache_subdir) if monkey
           else SortedSpikeCacheManager(cache_subdir=cache_subdir))
    return Path(mgr.cache_dir)


def relabel_cache(cache_subdir: str, *, only_date: Optional[str] = None,
                  only_round: Optional[int] = None, apply: bool = False,
                  backup: bool = True, mapping_csv: Optional[str] = None,
                  monkey: Optional[str] = None) -> None:
    from analyses.zombies_raster_review.waveform_footprint import load_session_voltages

    cache_dir = _cache_dir(cache_subdir, monkey)
    print(f"Cache dir: {cache_dir}")
    pkls = sorted(cache_dir.glob("*_round_*.pkl"))
    if not pkls:
        print("No cache pickles found.")
        return

    all_map: Dict[str, str] = {}
    for pkl in pkls:
        try:
            df = pd.read_pickle(pkl)
        except Exception as e:
            print(f"[{pkl.name}] load failed: {e}")
            continue
        if df is None or df.empty or not REQUIRED_COLS.issubset(df.columns):
            print(f"[{pkl.name}] skip (missing columns or empty)")
            continue
        date = str(df["Date"].iloc[0])
        round_no = int(df["Round No."].iloc[0])
        if only_date and (date != only_date or (only_round is not None and round_no != only_round)):
            continue

        try:
            voltages, sr = load_session_voltages(date, round_no)
        except Exception as e:
            print(f"[{pkl.name}] recording unavailable, cannot relabel: {e}")
            continue

        new_df, id_map, unresolved = relabel_dataframe(df, voltages, sr)
        print(f"\n[{pkl.name}] {len(id_map)} unit(s) relabelled"
              f"{f', {len(unresolved)} unresolved' if unresolved else ''}:")
        for old, new in sorted(id_map.items()):
            tag = "" if _norm_channel(old) == _norm_channel(new) else "  (channel changed)"
            print(f"    {old}\n      -> {new}{tag}")
        all_map.update(id_map)

        if apply:
            if backup:
                backup_path = pkl.with_suffix(".pkl.bak")
                if not backup_path.exists():
                    shutil.copy2(pkl, backup_path)
                new_df.to_pickle(pkl)
                print(f"    written (backup: {backup_path.name})")
            else:
                new_df.to_pickle(pkl)
                print("    written (no backup)")

    if mapping_csv and all_map:
        pd.DataFrame({"old_NeuronID": list(all_map), "new_NeuronID": list(all_map.values())}) \
            .to_csv(mapping_csv, index=False)
        print(f"\nWrote id map ({len(all_map)} rows) → {mapping_csv}")

    mode = "APPLIED" if apply else "DRY-RUN (nothing written; pass --apply to write)"
    print(f"\n{mode}. {len(all_map)} unit(s) across {len(pkls)} session file(s).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-subdir", default="sorted_spike_cache",
                   help="cache folder under <monkey>/ (e.g. sorted_spike_cache_filtered)")
    p.add_argument("--date", default=None, help="restrict to one date, e.g. 2023-09-26")
    p.add_argument("--round", type=int, default=None, dest="round_no")
    p.add_argument("--monkey", default=None, help="subject folder (default: pipeline default)")
    p.add_argument("--apply", action="store_true", help="write changes (else dry-run)")
    p.add_argument("--no-backup", action="store_true",
                   help="don't write .pkl.bak copies (saves disk; original is unrecoverable)")
    p.add_argument("--mapping-csv", default=None, help="write old→new NeuronID map here")
    args = p.parse_args(argv)
    relabel_cache(args.cache_subdir, only_date=args.date, only_round=args.round_no,
                  apply=args.apply, backup=not args.no_backup,
                  mapping_csv=args.mapping_csv, monkey=args.monkey)


if __name__ == "__main__":
    main()
