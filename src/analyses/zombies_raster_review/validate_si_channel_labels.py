"""
validate_si_channel_labels.py — is each SI unit's assigned channel where its
spikes ACTUALLY peak?

Motivation
----------
In ``spikesorting/sort_spikes/analyze_sorted_spikes.py::find_consensus_units`` the
SI unit's channel is assigned like this::

    ks4_ch = si.get_template_extremum_channel(analyzer_KS4, outputs='index')[ks4_uid]
    channel = Channel[f"C_{ks4_ch:03}"]
    'SpikeIdx': sorting_KS4.get_unit_spike_train(unit_id=sorting_KS4.get_unit_ids()[ks4_uid])

Two things there can silently decouple the *label* from the *spikes*:

1. ``get_template_extremum_channel(outputs='index')`` returns the channel's
   **position in the recording** (``amplifier.dat`` order), but ``Channel[f"C_{i:03}"]``
   uses that position as the Intan **native number**. Those differ whenever
   ``amplifier.dat`` is not stored in native order 0..31 — the very reason
   ``sort_spikes.compute_device_channel_index`` maps ``native_order -> rec index``.

2. ``get_unit_ids()[ks4_uid]`` fetches the spike train of the unit at *position*
   ``ks4_uid`` while the channel came from the unit with *id* ``ks4_uid``. If KS4's
   ids are not a contiguous ``0..N-1``, the label and the spikes are from
   different units.

Either makes an SI ``NeuronID`` whose channel doesn't match where its own spikes
peak. This script checks that directly, on real data, and does NOT modify the
sorting pipeline. Run it from ``src/`` for one session::

    python -m analyses.zombies_raster_review.validate_si_channel_labels --date 2023-09-26 --round 2

For every SI unit it prints the assigned channel next to the channel where the
unit's *stored spike times* actually peak — measured on BOTH windowsort's
recording (300 Hz high-pass) and SI's own recording (band-pass + CMR), so you can
also see whether the two preprocessings disagree. It then reports the two
structural checks (KS4 id contiguity; amplifier.dat native ordering).
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np

RADIUS = 25
MAX_SPIKES = 300


def _norm_channel(text) -> str:
    """Any channel spelling → ``C_020`` (underscore, 3-digit)."""
    tok = str(text).split("Channel.")[-1].split("_Unit")[0].replace("-", "_").strip()
    if tok.startswith("C_"):
        num = tok[2:]
        if num.isdigit():
            return f"C_{int(num):03d}"
    return tok


def _spike_indices(unit_df, sample_rate: float, spike_col: str = "SpikeTimes") -> np.ndarray:
    """Concatenated spike sample indices for one unit (from the cache times)."""
    from analyses.zombies_raster_review.coincidence_match import unit_spike_train
    times = unit_spike_train(unit_df, spike_col=spike_col)
    if times.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.round(times * sample_rate).astype(np.int64)


def _windowsort_peak(unit_df, voltages, sample_rate) -> Optional[str]:
    """Channel where the unit's spikes peak on windowsort's recording."""
    from spikesorting.cross_channel_analysis.waveforms import compute_footprint
    idx = _spike_indices(unit_df, sample_rate)
    if idx.size == 0:
        return None
    fp = compute_footprint("u", idx, voltages, radius=RADIUS, max_spikes=MAX_SPIKES)
    return _norm_channel(fp.peak_channel)


def _si_recording_peak(rec, spike_idx: np.ndarray, native_by_recidx: Dict[int, int]) -> Optional[str]:
    """Channel where the unit's spikes peak on SI's (band-pass + CMR) recording.

    Cuts ±RADIUS windows straight from the SpikeInterface recording, averages, and
    takes the largest peak-to-peak channel; maps that recording index → Intan
    native number → ``C_0xx`` (the correct index→name conversion).
    """
    if spike_idx.size == 0:
        return None
    n_frames = rec.get_num_frames()
    idx = spike_idx[(spike_idx - RADIUS >= 0) & (spike_idx + RADIUS < n_frames)]
    if idx.size == 0:
        return None
    if idx.size > MAX_SPIKES:
        idx = idx[np.linspace(0, idx.size - 1, MAX_SPIKES).astype(np.int64)]
    acc = None
    for i in idx:
        snip = rec.get_traces(start_frame=int(i) - RADIUS, end_frame=int(i) + RADIUS,
                              return_scaled=False)
        acc = snip.astype(np.float64) if acc is None else acc + snip
    mean = acc / idx.size                       # (2*RADIUS, n_channels)
    p2p = mean.max(axis=0) - mean.min(axis=0)
    peak_recidx = int(np.argmax(p2p))
    native = native_by_recidx.get(peak_recidx)
    return f"C_{int(native):03d}" if native is not None else f"recidx{peak_recidx}"


def diagnose(date: str, round_no: int, monkey: Optional[str] = None) -> None:
    from data_access.spike_source import SISortedSpikeSource
    from analyses.zombies_raster_review.coincidence_match import session_units
    from analyses.zombies_raster_review.waveform_footprint import load_session_voltages

    print(f"\n=== SI channel-label validation: {date} round {round_no} ===")

    si_df = SISortedSpikeSource().load(date, round_no)
    if si_df is None or si_df.empty:
        print("No SI units for this session.")
        return
    units = session_units(si_df, "NeuronID")
    print(f"{len(units)} SI unit(s) in the cache.\n")

    # windowsort recording (300 Hz high-pass), Intan-channel keyed
    try:
        voltages, sr = load_session_voltages(date, round_no)
    except Exception as e:
        print(f"[windowsort recording unavailable: {e}]")
        voltages, sr = None, None

    # SI's own recording (band-pass + CMR) + recording-index → native map
    rec_si, native_by_recidx = None, {}
    try:
        from spikesorting.sort_spikes.sort_spikes import (
            build_intan_session_path, get_recording_session_info, load_and_preprocess_recording,
        )
        kw = {"monkey": monkey} if monkey else {}
        intan_dir = build_intan_session_path(date, round_no, **kw)
        _, enabled = get_recording_session_info(intan_dir)
        native_by_recidx = {i: ch["native_order"] for i, ch in enumerate(enabled)}
        rec_si = load_and_preprocess_recording(intan_dir)
        if sr is None:
            sr = rec_si.get_sampling_frequency()
    except Exception as e:
        print(f"[SI recording unavailable: {e}]")

    if sr is None:
        print("No recording available on either side — cannot measure peaks.")
        return

    # per-unit: assigned channel vs where the spikes actually peak
    header = f"{'NeuronID':<44} {'assigned':>9} {'ws-peak':>9} {'si-peak':>9}  flag"
    print(header)
    print("-" * len(header))
    mismatches = 0
    for uid, udf in units.items():
        assigned = _norm_channel(uid)
        idx = _spike_indices(udf, sr)
        ws_peak = _windowsort_peak(udf, voltages, sr) if voltages is not None else None
        si_peak = _si_recording_peak(rec_si, idx, native_by_recidx) if rec_si is not None else None
        ref = ws_peak or si_peak
        flag = ""
        if ref is not None and ref != assigned:
            flag = "<<< MISMATCH"
            mismatches += 1
        print(f"{uid[:44]:<44} {assigned:>9} {str(ws_peak):>9} {str(si_peak):>9}  {flag}")
    print(f"\n{mismatches}/{len(units)} unit(s) have an assigned channel that "
          f"differs from where their spikes peak.")

    _structural_checks(date, round_no, monkey)


def _structural_checks(date: str, round_no: int, monkey: Optional[str]) -> None:
    """The two upstream conditions that would cause a mismatch."""
    print("\n--- structural checks (root cause) ---")
    try:
        import spikeinterface.sorters as ss
        from spikesorting.sort_spikes.sort_spikes import (
            build_intan_session_path, get_recording_session_info,
        )
        kw = {"monkey": monkey} if monkey else {}
        intan_dir = build_intan_session_path(date, round_no, **kw)

        import os
        sorting_KS4 = ss.read_sorter_folder(os.path.join(intan_dir, "kilosort4_output"))
        ids = list(sorting_KS4.get_unit_ids())
        contiguous = list(ids) == list(range(len(ids)))
        print(f"KS4 unit ids (n={len(ids)}): {ids[:12]}{' …' if len(ids) > 12 else ''}")
        print(f"  contiguous 0..N-1? {contiguous}  "
              f"→ if False, get_unit_ids()[ks4_uid] fetches the WRONG unit's spikes")

        _, enabled = get_recording_session_info(intan_dir)
        native = [ch["native_order"] for ch in enabled]
        native_ordered = native == list(range(len(native)))
        print(f"amplifier.dat native_order (n={len(native)}): {native[:12]}"
              f"{' …' if len(native) > 12 else ''}")
        print(f"  equals recording index 0..N-1? {native_ordered}  "
              f"→ if False, Channel[f'C_{{index:03}}'] names the WRONG channel")
    except Exception as e:
        print(f"[structural checks unavailable: {e}]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="e.g. 2023-09-26")
    p.add_argument("--round", type=int, required=True, dest="round_no")
    p.add_argument("--monkey", default=None, help="subject folder (default: pipeline default)")
    args = p.parse_args(argv)
    diagnose(args.date, args.round_no, args.monkey)


if __name__ == "__main__":
    main()
