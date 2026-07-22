"""
Detect in-house response windows for the grant's UNSORTED (MUA) neurons and write them to
CSV -- DETECTION ONLY (no count extraction, no permutation analysis), so it is relatively
quick and lets you eyeball the windows the algorithm found before running replicate_analysis.

Windows come from analyses.response_window_finder.threshold_window_detection run on the
ThresholdMUASpikeSource (MAD x4, ref 1.0 ms), same detector as run_mua_preprocessing.py.
Must run on the machine with threshold_mua_spike_cache / the raw recordings.

    python export_grant_mua_detected_windows.py [optional/output/path.csv]

Columns: Date, Round No., NeuronID, WindowStart_ms, WindowEnd_ms  (one row per detected window)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_count_connector import export_grant_mua_detected_windows  # noqa: E402


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else None
    detected, path = export_grant_mua_detected_windows(out_csv)
    n_neurons = detected["NeuronID"].nunique() if not detected.empty else 0
    print(f"{len(detected)} detected window(s) across {n_neurons} neurons")
    print("saved:", path)


if __name__ == "__main__":
    main()
