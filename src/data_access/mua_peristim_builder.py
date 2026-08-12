"""
mua_peristim_builder.py — pre-stimulus multi-unit-activity (MUA) cache for the
UNSORTED channels, built offline from the session's amplifier.dat.

Why MUA (not spike.dat): unsorted channels have no pre-stimulus spikes recoverable
from the caches (compiled.pkl is clipped to [onset, offset], and the online
spike.dat is hand-thresholded and not used here). Offline MAD/RMS threshold-crossing
detection on the full amplifier.dat gives channel-level MUA across the WHOLE
recording, so a pre-stimulus baseline is trivial to window. Same detector as
ThresholdMUASpikeSource / the grant MUA lists (detect_mad_spikes_for_recording).

Correctness note (epoch clock — the reason this is a separate builder):
ThresholdMUASpikeCacheManager windows each trial against compiled.pkl's
EpochStartStop. For STITCHED (merged) sessions that is WRONG: compiled.pkl carries
per-sub-folder *reset* clocks, while the amplifier.dat in the session folder is the
*continuous* stitched recording — so 2nd+ sub-folder trials get windowed at the
wrong times. This builder instead derives epochs from the SAME folder's
digitalin.dat / notes.txt that the amplifier came from (fnc=2, matching the
SI-sorted path), so spikes and epochs always share one clock. Trial metadata
(MonkeyId/Name/Group) is joined from compiled.pkl by task id.

Output: a separate variant cache (threshold_mua_spike_cache_pre{ms}ms/), leaving the
canonical threshold_mua_spike_cache/ (grant pipeline) untouched.

Runs where the Intan session lives (needs amplifier.dat / digitalin.dat / notes.txt
/ info.rhd). PyCharm-runnable via the CONFIG block at the bottom (no CLI).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels
from clat.intan.rhd import load_intan_rhd_format

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from data_access.data_loader import explode_spike_data
from data_access.threshold_detection import detect_mad_spikes_for_recording, read_amplifier_data_robust
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

PROJECT_ROOT = Path(DATA_BASE_PATH)


def mua_peristim_cache_subdir(pre_stimulus_time):
    """Cache subdir for a pre-stimulus window (strict window keeps the canonical name)."""
    if pre_stimulus_time and pre_stimulus_time > 0:
        return f"threshold_mua_spike_cache_pre{int(round(pre_stimulus_time * 1000))}ms"
    return "threshold_mua_spike_cache"


def _detect_mua(round_dir, noise_method, threshold_multiplier, refractory_ms):
    """(spike_times_by_channel {Channel: [sec]}, sample_rate) from the full continuous
    amplifier.dat — spike times share this folder's clock."""
    info_path = os.path.join(round_dir, "info.rhd")
    amp_path = os.path.join(round_dir, "amplifier.dat")
    if not os.path.exists(amp_path):
        raise FileNotFoundError(f"amplifier.dat missing in {round_dir}; needed for MUA.")
    rhd = load_intan_rhd_format.read_data(info_path)
    sample_rate = rhd["frequency_parameters"]["amplifier_sample_rate"]
    amp_channels = rhd["amplifier_channels"]
    voltages = read_amplifier_data_robust(amp_path, amp_channels, round_dir)
    spike_times_by_channel, det_info = detect_mad_spikes_for_recording(
        voltages, sample_rate, noise_method=noise_method,
        threshold_multiplier=threshold_multiplier, refractory_ms=refractory_ms, apply_filter=True)
    for ch, ci in det_info.items():
        print(f"  {ch}: {ci['n_spikes']} spikes (thr={ci['threshold']:.1f})")
    return spike_times_by_channel, sample_rate


def continuous_epochs(round_dir, sample_rate, fnc=2):
    """{task_id: (onset_s, offset_s)} from this folder's own digitalin/notes (continuous
    clock, matching the amplifier the MUA was detected on).

    Public because anything comparing amplifier-derived spikes against compiled.pkl
    needs it: on a STITCHED session compiled.pkl's epochs are on per-sub-folder reset
    clocks, and these are the epochs that put both on one clock (see the module
    docstring, and analyses.mua_threshold_tuning, which aligns the two by task id)."""
    stim_epochs = epoch_using_marker_channels(
        os.path.join(round_dir, "digitalin.dat"), false_negative_correction_duration=fnc)
    epochs_for_task = map_task_id_to_epochs_with_livenotes(
        os.path.join(round_dir, "notes.txt"), stim_epochs)
    return {tid: (on / sample_rate, off / sample_rate) for tid, (on, off) in epochs_for_task.items()}


def _windowed_mua_data(compiled, spike_times_by_channel, epochs_by_task, pre_stimulus_time):
    """Per-trial combined MUA SpikeTimes with a widened lower bound; metadata joined
    from compiled.pkl by task id. Returns a wide-format DataFrame ready for explode."""
    meta_by_task = {int(r["TaskField"]): (r["MonkeyId"], r["MonkeyName"], r["MonkeyGroup"])
                    for _, r in compiled.iterrows()}
    rows, skipped = [], 0
    for task_id, (on_s, off_s) in epochs_by_task.items():
        meta = meta_by_task.get(int(task_id))
        if meta is None:
            skipped += 1
            continue
        lo_s = max(on_s - pre_stimulus_time, 0.0)  # clamp to recording start
        spikes = {ch: [t for t in times if lo_s <= t < off_s]
                  for ch, times in spike_times_by_channel.items()}
        mid, mname, mgroup = meta
        rows.append({"TaskField": task_id, "MonkeyId": mid, "MonkeyName": mname,
                     "MonkeyGroup": mgroup, "SpikeTimes": spikes, "EpochStartStop": (on_s, off_s)})
    if skipped:
        print(f"  [warn] {skipped} marker epoch(s) had no matching TaskField in compiled.pkl; skipped")
    return pd.DataFrame(rows)


def build_mua_peristim_cache(date, round_no, pre_stimulus_time, *,
                             noise_method="mad", threshold_multiplier=4.0, refractory_ms=1.0,
                             fnc=2, monkey=SUBJECT_MONKEY, cache_subdir=None, force=False):
    """Build + write the pre-stimulus MUA cache for one session. Returns (exploded_df, path).
    Skipped (returns the cached df) if the variant pkl already exists and force is False."""
    date = str(date)
    if cache_subdir is None:
        cache_subdir = mua_peristim_cache_subdir(pre_stimulus_time)
    cache_dir = PROJECT_ROOT / monkey / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = f"{date}_round_{round_no}_{noise_method}{threshold_multiplier}_ref{refractory_ms}"
    path = cache_dir / f"{label}.pkl"
    if path.exists() and not force:
        return pd.read_pickle(path), path

    reader = RecordingMetadataReader()
    pickle_filepath, _, round_dir = reader.get_metadata_for_spike_analysis(date, round_no, monkey)
    round_dir = str(round_dir)
    compiled = pd.read_pickle(pickle_filepath)
    if compiled is None or compiled.empty:
        print(f"  [skip] empty compiled.pkl for {date} round {round_no}")
        return None, path

    spikes_by_ch, sample_rate = _detect_mua(round_dir, noise_method, threshold_multiplier, refractory_ms)
    epochs_by_task = continuous_epochs(round_dir, sample_rate, fnc=fnc)
    combined = _windowed_mua_data(compiled, spikes_by_ch, epochs_by_task, pre_stimulus_time)
    if combined.empty:
        print(f"  [skip] no trials for {date} round {round_no}")
        return None, path

    exploded = explode_spike_data(combined, date, round_no, curated_channels_only=False)
    exploded.to_pickle(path)
    print(f"  saved {path}  ({exploded['NeuronID'].nunique()} channels, {len(combined)} trials)")
    return exploded, path


# ===== Run directly in PyCharm — edit this block and hit Run (no CLI) =========
if __name__ == "__main__":
    DATE = "2023-09-26"
    ROUND_NO = 2
    PRE_STIMULUS_TIME = 1.0        # seconds of pre-stimulus baseline
    NOISE_METHOD = "mad"           # 'mad' = median(|v|)/0.6745, or 'rms'
    THRESHOLD_MULTIPLIER = 4.0
    REFRACTORY_MS = 1.0
    FORCE = False                  # True = rebuild even if the variant already exists

    build_mua_peristim_cache(DATE, ROUND_NO, PRE_STIMULUS_TIME,
                             noise_method=NOISE_METHOD, threshold_multiplier=THRESHOLD_MULTIPLIER,
                             refractory_ms=REFRACTORY_MS, force=FORCE)
