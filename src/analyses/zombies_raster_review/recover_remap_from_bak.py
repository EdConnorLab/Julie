"""
recover_remap_from_bak.py — rebuild the real old→new NeuronID map from backups.

If the SI cache was relabelled (``relabel_si_channels.py``) *without*
``--mapping-csv``, the genuine old→new map was never saved — and re-running the
relabeler can't recover it (on an already-relabelled cache it just maps the new
ids to themselves). But relabelling preserves row order, so each ``.pkl.bak``
backup (original ids) lines up row-for-row with its relabelled ``.pkl`` (new
ids). This zips their ``NeuronID`` columns back into the map, which you then feed
to ``apply_remap_to_list.py`` to fix ``zombies_si_sorted_anova_passed.csv``.

Needs the ``.pkl.bak`` files (i.e. you ran ``--apply`` with backups on and
haven't deleted them). No recording required.

    cd src
    python -m analyses.zombies_raster_review.recover_remap_from_bak --out remap.csv
    # then:
    python -m analyses.zombies_raster_review.apply_remap_to_list \
        --list analyses/zombies_raster_review/unit_lists/zombies_si_sorted_anova_passed.csv \
        --remap remap.csv --out <same path>
"""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd


def recover_map(cache_subdir: str, out_csv: str, monkey: Optional[str] = None) -> dict:
    from analyses.zombies_raster_review.relabel_si_channels import _cache_dir

    cache_dir = _cache_dir(cache_subdir, monkey)
    print(f"Cache dir: {cache_dir}")
    baks = sorted(cache_dir.glob("*_round_*.pkl.bak"))
    if not baks:
        print("No .pkl.bak backups found — cannot recover the map from here.")
        return {}

    mapping: dict = {}
    skipped: list = []
    for bak in baks:
        cur = bak.with_suffix("")  # 'x.pkl.bak' -> 'x.pkl'
        if not cur.exists():
            print(f"[{bak.name}] no relabelled counterpart {cur.name}; skip")
            skipped.append(bak.name)
            continue
        old_df, new_df = pd.read_pickle(bak), pd.read_pickle(cur)
        if len(old_df) != len(new_df):
            print(f"[{bak.name}] row count differs ({len(old_df)} vs {len(new_df)}); skip")
            skipped.append(bak.name)
            continue
        # sanity: rows must line up (same trial sequence) for the zip to be valid
        if "TaskField" in old_df and "TaskField" in new_df and not (
                old_df["TaskField"].reset_index(drop=True)
                .equals(new_df["TaskField"].reset_index(drop=True))):
            print(f"[{bak.name}] row order differs from backup; skip (recover this one manually)")
            skipped.append(bak.name)
            continue
        pairs = 0
        for o, n in zip(old_df["NeuronID"].astype(str), new_df["NeuronID"].astype(str)):
            if o not in mapping:
                mapping[o] = n
                pairs += 1
        print(f"[{bak.name}] {pairs} unit(s)")

    if mapping:
        pd.DataFrame({"old_NeuronID": list(mapping), "new_NeuronID": list(mapping.values())}) \
            .to_csv(out_csv, index=False)
        changed = sum(1 for o, n in mapping.items() if o != n)
        print(f"\nWrote {len(mapping)} unit(s) ({changed} with a changed channel) → {out_csv}")

    processed = len(baks) - len(skipped)
    print(f"\nComplete? processed {processed}/{len(baks)} backup(s), {len(skipped)} skipped.")
    if skipped:
        print("  SKIPPED (NOT in remap — keep these .bak / investigate before deleting):")
        for name in skipped:
            print(f"    {name}")
    else:
        print("  → every .bak is captured; safe to delete them after this.")
    return mapping


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-subdir", default="sorted_spike_cache",
                   help="cache folder whose .pkl.bak backups to read")
    p.add_argument("--out", default="remap.csv", help="output old→new NeuronID map csv")
    p.add_argument("--monkey", default=None)
    args = p.parse_args(argv)
    recover_map(args.cache_subdir, args.out, args.monkey)


if __name__ == "__main__":
    main()
