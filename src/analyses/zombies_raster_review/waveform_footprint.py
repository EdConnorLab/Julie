"""
waveform_footprint.py — average cross-channel spike waveforms for the overlay.

For each unit in an overlay we want its **spatial footprint**: the average spike
waveform on every contact of the linear probe. Two units that are really one
neuron share the same footprint (peak on the same contact, same decay); two
distinct neurons that merely fire synchronously do not.

Why we cut *both* sorters' waveforms from the *same* voltages
------------------------------------------------------------
The SI (SpikeInterface) sort stores its own average templates in
``analyzer_*_binary`` folders, but (a) those templates are computed on
band-pass(300–6000 Hz) + common-median-referenced data, while the manual sort's
waveforms come off windowsort's 300 Hz high-pass ``preprocessed_data.dat`` — so
their **amplitudes are not directly comparable** — and (b) the csv ``NeuronID``
cannot be reliably mapped back to a SpikeInterface analyzer unit id (that id is
not stored, and the pipeline resolves it positionally; see
``sort_spikes/analyze_sorted_spikes.py``).

Both problems vanish if we ignore the stored templates and instead cut snippets
from **one** voltage source (windowsort's preprocessed recording) at **each
unit's own spike times** — which we already have for both sorts. Same data, same
method → the manual and SI footprints are apples-to-apples, and no analyzer /
unit-id mapping is needed. The spike *time* of a spike is the same physical
moment whichever sorter detected it, so indexing either sort's times into the
shared voltages is correctly aligned.

This needs the raw recording on disk (``amplifier.dat`` / ``preprocessed_data.dat``
+ ``info.rhd``), exactly like the real ``mixed``/``si`` runs. When that data is
absent every function degrades to ``None`` / ``{}`` and the overlay simply omits
the footprint panel.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from spikesorting.cross_channel_analysis.waveforms import Footprint, compute_footprint

DEFAULT_RADIUS = 25       # ±25 samples, matching windowsort's spike view
DEFAULT_MAX_SPIKES = 500  # subsample per unit for a fast, stable average


def load_session_voltages(date: str, round_no: int):
    """``(voltages_by_channel, sample_rate)`` for one session, or raise.

    ``voltages_by_channel`` is ``{channel_name: 1-D np.ndarray}`` from windowsort's
    (300 Hz high-pass) preprocessed recording — the same substrate the manual
    sort used. Heavy: reads/caches the whole recording. Raises if the recording,
    ``windowsort``, or ``clat`` is unavailable (the caller catches this).
    """
    from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
    from clat.intan.rhd import load_intan_rhd_format
    from spikesorting.cross_channel_analysis.waveforms import load_voltages

    _, _, round_dir = RecordingMetadataReader().get_metadata_for_spike_analysis(date, round_no)
    round_dir = str(round_dir)
    rhd = load_intan_rhd_format.read_data(os.path.join(round_dir, "info.rhd"))
    sample_rate = float(rhd["frequency_parameters"]["amplifier_sample_rate"])
    return load_voltages(round_dir), sample_rate


def unit_footprint(
    unit_df,
    voltages_by_channel: Dict[str, np.ndarray],
    sample_rate: float,
    uid: str,
    *,
    spike_col: str = "SpikeTimes",
    radius: int = DEFAULT_RADIUS,
    max_spikes: int = DEFAULT_MAX_SPIKES,
) -> Optional[Footprint]:
    """Cross-channel mean-waveform footprint for one unit, or ``None``.

    The unit's per-trial spike times (seconds) are concatenated, converted to
    sample indices, and averaged across all channels of ``voltages_by_channel``
    via the vetted :func:`compute_footprint`.
    """
    from analyses.zombies_raster_review.coincidence_match import unit_spike_train

    times = unit_spike_train(unit_df, spike_col=spike_col)
    if times.size == 0:
        return None
    idx = np.round(times * sample_rate).astype(np.int64)
    try:
        return compute_footprint(uid, idx, voltages_by_channel,
                                 radius=radius, max_spikes=max_spikes)
    except Exception as e:  # bad channel keys, empty voltages, …
        print(f"[waveforms] footprint failed for {uid}: {e}")
        return None


def extract_footprints(
    units_by_id: Dict[str, "object"],
    date: str,
    round_no: int,
    *,
    spike_col: str = "SpikeTimes",
) -> Dict[str, Footprint]:
    """Footprint for every unit in a session, keyed by unit id.

    Loads the session's voltages once, then cuts each unit's footprint from them.
    Returns ``{}`` if the recording can't be loaded (no data / no windowsort), so
    the overlay just skips the footprint panel.
    """
    try:
        voltages, sample_rate = load_session_voltages(date, round_no)
    except Exception as e:
        print(f"[waveforms] voltage load failed for {date} r{round_no}: {e}")
        return {}

    out: Dict[str, Footprint] = {}
    for uid, df in units_by_id.items():
        fp = unit_footprint(df, voltages, sample_rate, uid, spike_col=spike_col)
        if fp is not None:
            out[uid] = fp
    print(f"[waveforms] {date} r{round_no}: footprints for {len(out)}/{len(units_by_id)} units")
    return out
