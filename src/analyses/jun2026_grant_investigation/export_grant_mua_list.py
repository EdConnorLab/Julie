"""
Write a CSV of the grant's UNSORTED (MUA) cells matched to their threshold-MUA NeuronIDs,
so the mapping can be eyeballed before running the full comparison.

This does MATCHING ONLY (no permutation analysis, no count extraction), so it is quick.
It must run on the machine that has `threshold_mua_spike_cache` / the raw recordings,
since the NeuronIDs (incl. Location) are read verbatim from the threshold-MUA source.

    python export_grant_mua_list.py [optional/output/path.csv]

Columns: grant_cell, date, round_no, time_window, neuron_id, status, detail
  status: matched | no_match | ambiguous | missing_session | no_window
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_count_connector import export_grant_mua_match_table  # noqa: E402


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else None
    rows, path = export_grant_mua_match_table(out_csv)
    print("status counts:", dict(Counter(r["status"] for r in rows)))
    print("saved:", path)


if __name__ == "__main__":
    main()
