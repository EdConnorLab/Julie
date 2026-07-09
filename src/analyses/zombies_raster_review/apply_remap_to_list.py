"""
apply_remap_to_list.py — update a unit-list's NeuronIDs after relabelling.

``relabel_si_channels.py --mapping-csv remap.csv`` writes an ``old_NeuronID →
new_NeuronID`` map. After relabelling the SI cache, any list that references SI
units by ``NeuronID`` (e.g. ``zombies_si_sorted_anova_passed.csv``) still holds
the OLD ids, so the overlay's anchor lookup finds nothing and no figures are
written. This rewrites the list's ``NeuronID`` column through that map, leaving
every other column (windows, stats) untouched.

    cd src
    python -m analyses.zombies_raster_review.apply_remap_to_list \
        --list analyses/zombies_raster_review/unit_lists/zombies_si_sorted_anova_passed.csv \
        --remap remap.csv --out zombies_si_sorted_anova_passed.csv

Dry-run by default (prints how many rows would change and any ids missing from
the map); pass ``--out PATH`` to write (``--out`` same as ``--list`` overwrites
in place). Only the SI (csv) list needs this — the mixed (xlsx) list matches by
channel and its cache was not relabelled.
"""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd


def apply_remap(list_path: str, remap_path: str, out_path: Optional[str] = None,
                id_col: str = "NeuronID") -> pd.DataFrame:
    lst = pd.read_csv(list_path)
    remap = pd.read_csv(remap_path)
    if id_col not in lst.columns:
        raise ValueError(f"'{id_col}' not in {list_path} (columns: {list(lst.columns)})")
    mapping = dict(zip(remap["old_NeuronID"].astype(str), remap["new_NeuronID"].astype(str)))

    old_ids = lst[id_col].astype(str)
    changed = missing = 0
    new_ids = []
    for nid in old_ids:
        if nid in mapping:
            new_ids.append(mapping[nid])
            changed += int(mapping[nid] != nid)
        else:
            new_ids.append(nid)
            missing += 1
    lst[id_col] = new_ids

    print(f"{changed}/{len(lst)} row(s) remapped · {missing} not found in the map")
    if missing:
        miss = [n for n in old_ids if n not in mapping]
        print(f"  not in map (left unchanged): {miss[:10]}{' …' if len(miss) > 10 else ''}")

    if out_path:
        lst.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
    else:
        print("dry-run (pass --out to write)")
    return lst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", dest="list_path", required=True, help="unit-list csv to update")
    p.add_argument("--remap", dest="remap_path", required=True,
                   help="remap.csv from relabel_si_channels (old_NeuronID,new_NeuronID)")
    p.add_argument("--out", dest="out_path", default=None,
                   help="output csv (omit for dry-run; same path as --list to overwrite)")
    p.add_argument("--id-col", default="NeuronID")
    args = p.parse_args(argv)
    apply_remap(args.list_path, args.remap_path, args.out_path, id_col=args.id_col)


if __name__ == "__main__":
    main()
