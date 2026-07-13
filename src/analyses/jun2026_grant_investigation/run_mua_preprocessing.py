"""
Generate the threshold-MUA significance lists by running the EXISTING preprocessing
pipeline with the new ThresholdMUASpikeSource. Purely additive: it imports and
reuses preprocess_and_select_significant_neurons.py without modifying it.

Outputs (named `{source.name}_{group}_...`, i.e. prefix 'threshold_mua') into the
analysis_cache dir:
  threshold_mua_Zombies_response_windows.pkl
  threshold_mua_Zombies_significant_windows_pKW_passed.pkl
  threshold_mua_Zombies_significant_windows_pANOVA_passed.pkl
which spike_count_connector.py then consumes for the 'MUA_KW' / 'MUA_ANOVA' lists.

MUST run on the machine with the raw recordings (needs amplifier.dat /
preprocessed_data.dat + compiled.pkl). Detection params are set below.

PyCharm: hit Run. CLI: python run_mua_preprocessing.py
"""

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..'))          # .../Julie/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader  # noqa: E402
from analyses.preprocessing.preprocess_and_select_significant_neurons import (        # noqa: E402
    PreprocessConfig,
    select_all_significant_neurons_using_pANOVA,
    detect_all_response_windows_for_all_neurons,
    detect_significant_windows_using_pKW,
    detect_significant_windows_using_pANOVA,
)
from data_access.spike_source import ThresholdMUASpikeSource                          # noqa: E402

# ---- edit these ----
ANALYSIS_CACHE_DIR = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/"
NOISE_METHOD = "mad"         # 'mad' = median(|v|)/0.6745, or 'rms'
THRESHOLD_MULTIPLIER = 4.0
REFRACTORY_MS = 1.0
# --------------------


def main():
    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()
    cfg = PreprocessConfig(group_name="Zombies", bin_size=0.05,
                           analysis_cache_dir=Path(ANALYSIS_CACHE_DIR), save=True)

    source = ThresholdMUASpikeSource(
        noise_method=NOISE_METHOD,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
        refractory_ms=REFRACTORY_MS,
    )  # pre_filtered=True -> single-unit ISI QC skipped (multiunit)

    # same 3 stages as preprocess_and_select_significant_neurons.__main__, MUA source
    select_all_significant_neurons_using_pANOVA(metadata, cfg, source=source)
    windows = detect_all_response_windows_for_all_neurons(metadata, cfg, source=source)
    detect_significant_windows_using_pKW(windows, cfg, source=source)
    detect_significant_windows_using_pANOVA(windows, cfg, source=source)
    print(f"Done. Wrote threshold_mua_Zombies_* to {ANALYSIS_CACHE_DIR}")


if __name__ == "__main__":
    main()
