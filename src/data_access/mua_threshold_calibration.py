"""
mua_threshold_calibration.py — pick the MUA detection parameters that make the offline
threshold-MUA cache look like the ONLINE unsorted channels of the exploded cache.

WHICH TOOL: this one reads ``spike.dat`` and measures EVERY channel, which is only
meaningful where the online threshold on that channel is one you stand behind. If the
hand-set thresholds were not tracked across all 32 channels, or the session was
interrupted and its spike.dat has to be stitched first, use
``analyses.mua_threshold_tuning`` instead: it scores against a curated answer key of
unsorted cells read from the exploded cache, opens no raw online file, and
cross-validates the choice. This module remains the way to see, per channel, what
threshold the online detector was actually using.

THE TWO STREAMS, AND WHY THEY DON'T LOOK ALIKE
----------------------------------------------
An unsorted channel appears twice in this project, detected two different ways:

  * ``exploded_spike_cache_pre1000ms`` (the "unsorted" rows) -- ONLINE detection.
    Intan's spike scope crossed a threshold the experimenter dialled in BY HAND, per
    channel, during the recording; ``spike.dat`` holds those timestamps and
    ``compile.julie_one_file_spike_parsing`` windows them into ``compiled.pkl``.
    The threshold is a fixed voltage in µV, and nothing records what it was.
  * ``threshold_mua_spike_cache_pre1000ms`` -- OFFLINE detection, this repo's
    ``threshold_detection.detect_mad_spikes_for_recording``: re-derive a threshold from
    the data as ``-multiplier * sigma_noise`` per channel and re-detect on the whole
    ``amplifier.dat``.

Same electrode, same trials, but the offline default (4 x MAD) is usually a much
SHALLOWER threshold than the hand-set one, so its rasters are denser and its mean
waveform is smaller and blunter -- it is averaging in small spikes and noise crossings
that the online detector never accepted. That is the appearance gap this module
measures and closes.

WHAT IT MEASURES
----------------
The online threshold is not written down anywhere, but it is recoverable: every spike
``spike.dat`` accepted on a channel had to reach it, so the SHALLOWEST online spikes
sit just past it. So, per channel, on our own filtered trace:

  1. snap each online timestamp to its local trough (the online stamp is the crossing,
     not the trough) and measure the trough depths in µV;
  2. take a low percentile of those depths as the effective online threshold
     (``ONLINE_THRESHOLD_PERCENTILE``; the percentile rather than the min so one
     mis-snapped spike can't set it -- which biases it slightly DEEP);
  3. divide by that channel's ``sigma_noise`` to get the multiplier the online
     detector was effectively using -- directly comparable to
     ``threshold_multiplier``.

Because it is measured on the same filtered trace the offline detector runs on, the
comparison is in one set of units, and Intan's own filter settings drop out.

Then it sweeps candidate parameters and, per channel and candidate, reports what a
reviewer actually sees: spike RATE relative to online (raster density), mean-waveform
PEAK-TO-PEAK relative to online (the waveform panel), and how much the two spike
trains agree (recall/precision at ``MATCH_TOL_MS``). ``recommend()`` then names the
grid point that best matches, by rate and by amplitude.

A WORD BEFORE YOU RETUNE
------------------------
``ThresholdMUASpikeSource`` exists precisely BECAUSE the online thresholds were hand
set and drift across 32 channels and across sessions (see its docstring). Matching
them therefore re-imports that inconsistency. Two different jobs, so keep them apart:

  * making the two look alike for a side-by-side figure, or showing that a result does
    not hinge on which detector produced it -- that is what this module is for;
  * the analysis cache -- prefer ONE principled multiplier for every channel and
    session, chosen once (this module's median implied multiplier is a good, data-
    derived choice) rather than per channel.

The per-channel spread is printed for exactly that reason: if the implied multipliers
scatter widely, no single global multiplier can match every channel, and that spread
IS the reason the offline detector was introduced.

HOW TO RUN
----------
Needs the raw Intan session (``amplifier.dat``, ``info.rhd``, ``spike.dat``), so run it
on the rig. PyCharm: open this file, edit the CONFIG block at the bottom, hit Run. Or::

    cd src
    python -m data_access.mua_threshold_calibration --date 2023-09-26 --round 2
    python -m data_access.mua_threshold_calibration --date 2023-09-26 --round 2 --plots
    python -m data_access.mua_threshold_calibration --sessions 2023-09-26:2,2023-10-27:3

It only ever READS the recording; the CSVs (and PNGs, with --plots) go to
``<data root>/<monkey>/mua_calibration/``. It writes no spike cache -- once you have a
multiplier, rebuild with ``data_access.rebuild_peristim_caches`` (see NEXT STEPS at the
bottom of this file).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from data_access.threshold_detection import detect_mad_spikes, estimate_noise, highpass_filter
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY
from spikesorting.cross_channel_analysis.waveforms import DEFAULT_RADIUS, mean_waveform

# ---------------------------------------------------------------------------
# Local copies, so this module ADDS to the pipeline and changes nothing in it.
#
# Each mirrors a definition that already lives in a module the analyses import.
# They are copies rather than shared helpers on purpose: the caches, rasters and
# regressions that read those modules are in use, and a calibration tool is not a
# reason to edit them. The cost is that a change over there does not reach here —
# so each copy names its original, and the pairing is what to check if a number
# printed here ever disagrees with the same number on a raster figure.
# ---------------------------------------------------------------------------
MICROVOLTS_PER_BIT = 0.195     # Intan int16 -> microvolts (threshold_detection)


def _open_amplifier_memmap(file_path, amplifier_channels):
    """(data, channels) for an Intan amplifier file, WITHOUT reading it.

    ``data`` is a read-only int16 memmap shaped ``(n_samples, n_channels)`` (ADC units;
    multiply by MICROVOLTS_PER_BIT for microvolts); ``channels`` is the Channel enum per
    column, in file order.

    Same reshape rule as ``threshold_detection.read_amplifier_data_robust``: info.rhd's
    amplifier_channels is the authoritative channel count, and a trailing partial sample
    is dropped. Memmapped rather than read, because the sweep wants one channel at a time
    and a whole 32-channel round as float64 is tens of GB. (No digitalin.dat channel-count
    warning here -- that check belongs to the cache builders, which is where it fires.)
    """
    from clat.intan.channels import Channel

    nch = len(amplifier_channels)
    n_samples = (os.path.getsize(file_path) // 2) // nch
    mm = np.memmap(file_path, dtype=np.int16, mode='r')
    data = mm[:n_samples * nch].reshape(n_samples, nch)
    channels = [Channel(ch.get("native_channel_name", f"Channel_{i}"))
                for i, ch in enumerate(amplifier_channels)]
    return data, channels


def peak_to_peak(wave) -> float:
    """Peak-to-peak amplitude of one waveform, in the units of ``wave`` (µV)."""
    wave = np.asarray(wave, dtype=float)
    return float(wave.max() - wave.min()) if wave.size else 0.0


def trough_to_peak_ms(wave, sample_rate):
    """Trough-to-peak width (ms) for a clean biphasic negative spike, else ``None``.

    Copy of ``zombies_raster._trough_to_peak_ms`` (which cannot be imported: that module
    forces the Agg matplotlib backend at import), so the width printed here is the width
    printed in the raster review's info panel. The gates are the point: a monophasic or
    positive-going waveform, or one still rising at the window edge, returns ``None``
    rather than an edge-pinned, meaningless number.
    """
    wave = np.asarray(wave, dtype=float)
    n = wave.size
    if n < 3:
        return None
    trough = int(np.argmin(wave))
    if not (0 < trough < n - 1):                  # a real, interior trough
        return None
    if abs(wave[trough]) < wave.max():            # the negative deflection must dominate
        return None
    seg = wave[trough:]
    pk_rel = int(np.argmax(seg))
    if pk_rel == 0 or pk_rel == seg.size - 1:     # no interior rebound peak captured
        return None
    if wave[trough + pk_rel] <= 0:                # rebound must rise above baseline
        return None
    return pk_rel / sample_rate * 1000.0

# Candidate multipliers to sweep. 4.0 is the current default; the online thresholds
# usually land well above it, hence the long upper tail.
DEFAULT_MULTIPLIERS = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0)

# Which percentile of online trough depths counts as "the online threshold". Low, but
# not the minimum: one mis-snapped spike would drag the minimum to noise level.
ONLINE_THRESHOLD_PERCENTILE = 5.0

# How far to look either side of an online timestamp for its trough. The Intan stamp is
# the threshold crossing, so the trough follows within a fraction of a spike width.
SNAP_MS = 1.0

# Two spikes count as the same event if their troughs are this close.
MATCH_TOL_MS = 0.5

# A channel needs at least this many online spikes before its implied threshold means
# anything; below it, the low percentile is just the shallowest of a handful.
MIN_ONLINE_SPIKES = 50

# Waveform cutting, matching the raster review's footprint panel so the amplitudes and
# widths printed here are the ones you see on those figures.
WAVEFORM_RADIUS = DEFAULT_RADIUS         # ±25 samples, windowsort's spike view
MAX_WAVEFORM_SPIKES = 500


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def session_round_dir(date, round_no, monkey=SUBJECT_MONKEY) -> str:
    """The Intan round folder for a session (where amplifier.dat / spike.dat live)."""
    _, _, round_dir = RecordingMetadataReader().get_metadata_for_spike_analysis(
        str(date), round_no, monkey)
    return str(round_dir)


def load_online_spike_times(round_dir) -> dict:
    """``{channel string: np.ndarray of seconds}`` from the session's ``spike.dat``.

    These are the online, hand-thresholded detections -- the same timestamps that
    become the unsorted rows of the exploded cache. On stitched sessions
    ``compile.intan_file_stitcher.stitch_spike_dat`` rewrites them onto the continuous
    clock, so they share ``amplifier.dat``'s time base.
    """
    from clat.intan.spike_file import fetch_spike_tstamps_from_file

    path = os.path.join(round_dir, "spike.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"spike.dat missing in {round_dir}. It is the online detection this "
            f"calibration measures against; without it there is nothing to match.")
    tstamps_by_channel, _sample_rate = fetch_spike_tstamps_from_file(path)
    return {str(ch): np.sort(np.asarray(list(ts), dtype=float))
            for ch, ts in tstamps_by_channel.items()}


def _wanted(channel: str, only_channels) -> bool:
    """Channel filter accepting either full ('Channel.C_005') or bare ('C_005') names."""
    if not only_channels:
        return True
    return any(channel == str(c) or channel.endswith(str(c)) for c in only_channels)


def iter_filtered_channels(round_dir, *, only_channels=None, max_seconds=None,
                           highpass_cutoff=300.0):
    """Yield ``(channel string, filtered voltage in µV, sample_rate)``, one channel at a time.

    Same substrate as the detector: raw ``amplifier.dat`` with OUR high-pass applied
    (``threshold_detection.highpass_filter``), never ``preprocessed_data.dat`` (which is
    re-derived and can be shorter than the recording). Channel-at-a-time off a memmap,
    because a whole 32-channel round as float64 is tens of GB.

    ``max_seconds`` truncates to the first N seconds -- calibration converges long
    before a full round, and it makes the sweep interactive.
    """
    from clat.intan.rhd import load_intan_rhd_format

    rhd = load_intan_rhd_format.read_data(os.path.join(round_dir, "info.rhd"))
    sample_rate = float(rhd["frequency_parameters"]["amplifier_sample_rate"])
    amp_path = os.path.join(round_dir, "amplifier.dat")
    if not os.path.exists(amp_path):
        raise FileNotFoundError(f"amplifier.dat missing in {round_dir}; needed for MUA.")

    data, channels = _open_amplifier_memmap(amp_path, rhd["amplifier_channels"])
    n_samples = data.shape[0] if max_seconds is None else min(
        data.shape[0], int(round(max_seconds * sample_rate)))

    for i, ch in enumerate(channels):
        name = str(ch)
        if not _wanted(name, only_channels):
            continue
        raw = np.asarray(data[:n_samples, i], dtype=np.float64) * MICROVOLTS_PER_BIT
        yield name, highpass_filter(raw, sample_rate, cutoff=highpass_cutoff), sample_rate


# --------------------------------------------------------------------------- #
# Spike-train helpers
# --------------------------------------------------------------------------- #
def snap_to_trough(voltage, indices, search_samples):
    """Move each index to the local minimum within ±``search_samples``.

    Online timestamps mark the threshold CROSSING; ``detect_mad_spikes`` returns
    TROUGHS. Snapping puts both on the trough so depths, waveforms and match
    tolerances all mean the same thing.
    """
    v = np.asarray(voltage)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return idx
    lo = np.clip(idx - search_samples, 0, v.size - 1)
    hi = np.clip(idx + search_samples + 1, 1, v.size)
    return np.array([l + int(np.argmin(v[l:h])) for l, h in zip(lo, hi)], dtype=np.int64)


def match_fraction(a_indices, b_indices, tol_samples):
    """Fraction of ``a`` events with a ``b`` event within ``tol_samples``.

    Nearest-neighbour, not a bijection: one ``b`` may account for several ``a``. Good
    enough for "do these two trains describe the same events", which is all it is used
    for, and it never silently fails on trains of very different length.
    """
    a = np.asarray(a_indices, dtype=np.int64)
    b = np.sort(np.asarray(b_indices, dtype=np.int64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    pos = np.searchsorted(b, a)
    left = b[np.clip(pos - 1, 0, b.size - 1)]
    right = b[np.clip(pos, 0, b.size - 1)]
    nearest = np.minimum(np.abs(left - a), np.abs(right - a))
    return float(np.mean(nearest <= tol_samples))


def _wave_stats(voltage, indices, sample_rate):
    """``(peak-to-peak µV, trough-to-peak ms | None, mean waveform)`` for a spike train."""
    if len(indices) == 0:
        return float("nan"), None, np.zeros(2 * WAVEFORM_RADIUS)
    wave = mean_waveform(voltage, indices, radius=WAVEFORM_RADIUS,
                         max_spikes=MAX_WAVEFORM_SPIKES)
    return peak_to_peak(wave), trough_to_peak_ms(wave, sample_rate), wave


# --------------------------------------------------------------------------- #
# Per-channel calibration
# --------------------------------------------------------------------------- #
def calibrate_channel(channel, voltage, sample_rate, online_times, *,
                      multipliers=DEFAULT_MULTIPLIERS, noise_methods=("mad",),
                      refractory_ms=(1.0,),
                      online_percentile=ONLINE_THRESHOLD_PERCENTILE):
    """Calibrate one channel. Returns ``(summary dict, sweep rows, diagnostics dict)``.

    ``summary`` describes the ONLINE detection and the multiplier it implies;
    ``sweep rows`` are one per (noise method, multiplier, refractory) candidate;
    ``diagnostics`` carries what the figure needs and a table cannot hold -- the online
    trough depths and the mean waveform of the online train and of every candidate.
    """
    duration_s = voltage.size / sample_rate
    sigma = {m: float(estimate_noise(voltage, m)) for m in ("mad", "rms")}

    online_idx_raw = np.round(np.asarray(online_times, dtype=float) * sample_rate).astype(np.int64)
    online_idx_raw = online_idx_raw[(online_idx_raw >= 0) & (online_idx_raw < voltage.size)]
    online_idx = snap_to_trough(voltage, online_idx_raw, int(round(SNAP_MS / 1000 * sample_rate)))

    depths = -voltage[online_idx] if online_idx.size else np.array([])   # positive µV
    n_online = int(online_idx.size)
    enough = n_online >= MIN_ONLINE_SPIKES

    online_p2p, online_width, online_wave = _wave_stats(voltage, online_idx, sample_rate)
    # Positive-going means the online detector was NOT thresholding negatively on this
    # channel (or the stamps are misaligned); its implied threshold would be nonsense.
    polarity = "negative" if (online_wave.size and abs(online_wave.min()) >= online_wave.max()) \
        else "positive"

    thr_uv = float(np.percentile(depths, online_percentile)) if enough else float("nan")
    summary = {
        "Channel": channel,
        "duration_s": duration_s,
        "sigma_mad_uv": sigma["mad"],
        "sigma_rms_uv": sigma["rms"],
        "n_online": n_online,
        "online_rate_hz": n_online / duration_s if duration_s else float("nan"),
        # how far the online stamp sat from the trough -- a large value means the stamps
        # are not crossings and everything downstream needs a second look
        "snap_offset_ms": float(np.median(np.abs(online_idx - online_idx_raw)))
        / sample_rate * 1000.0 if n_online else float("nan"),
        "online_trough_min_uv": float(depths.min()) if n_online else float("nan"),
        f"online_thr_p{online_percentile:g}_uv": thr_uv,
        "online_trough_median_uv": float(np.median(depths)) if n_online else float("nan"),
        "implied_mult_mad": thr_uv / sigma["mad"] if enough and sigma["mad"] else float("nan"),
        "implied_mult_rms": thr_uv / sigma["rms"] if enough and sigma["rms"] else float("nan"),
        "online_p2p_uv": online_p2p,
        "online_width_ms": online_width if online_width is not None else float("nan"),
        "online_polarity": polarity,
        "trustworthy": bool(enough and polarity == "negative"),
    }

    rows = []
    diag = {"online_wave": online_wave, "online_depths": depths, "candidate_waves": {}}
    tol_samples = MATCH_TOL_MS / 1000.0 * sample_rate
    for method in noise_methods:
        for refr in refractory_ms:
            refr_samples = max(1, int(refr / 1000.0 * sample_rate))
            for mult in multipliers:
                threshold = -mult * sigma[method]
                idx = detect_mad_spikes(voltage, threshold, refr_samples)
                p2p, width, wave = _wave_stats(voltage, idx, sample_rate)
                rows.append({
                    "Channel": channel,
                    "noise_method": method,
                    "threshold_multiplier": mult,
                    "refractory_ms": refr,
                    "threshold_uv": threshold,
                    "n_detected": int(idx.size),
                    "rate_hz": idx.size / duration_s if duration_s else float("nan"),
                    "n_online": n_online,
                    # >1 = denser raster than the unsorted lane, <1 = sparser
                    "rate_ratio": idx.size / n_online if n_online else float("nan"),
                    "recall_of_online": match_fraction(online_idx, idx, tol_samples),
                    "precision_vs_online": match_fraction(idx, online_idx, tol_samples),
                    "p2p_uv": p2p,
                    # >1 = taller mean waveform than the unsorted lane, <1 = smaller
                    "p2p_ratio": p2p / online_p2p if online_p2p else float("nan"),
                    "width_ms": width if width is not None else float("nan"),
                    "trustworthy": summary["trustworthy"],
                })
                diag["candidate_waves"][(method, mult, refr)] = wave

    return summary, rows, diag


# --------------------------------------------------------------------------- #
# Session / multi-session calibration
# --------------------------------------------------------------------------- #
def calibrate_session(date, round_no, *, monkey=SUBJECT_MONKEY,
                      multipliers=DEFAULT_MULTIPLIERS, noise_methods=("mad",),
                      refractory_ms=(1.0,), only_channels=None, max_seconds=None,
                      online_percentile=ONLINE_THRESHOLD_PERCENTILE,
                      plot_dir=None, verbose=True):
    """Calibrate every channel of one session. Returns ``(channels_df, sweep_df)``.

    Reads only; with ``plot_dir`` set, also writes one diagnostic PNG per channel.
    """
    date = str(date)
    round_dir = session_round_dir(date, round_no, monkey)
    online_by_channel = load_online_spike_times(round_dir)
    if verbose:
        print(f"[calib] {date} round {round_no}: {round_dir}")
        print(f"[calib] spike.dat carries {len(online_by_channel)} channel(s) of "
              f"online detections")

    summaries, sweep_rows = [], []
    for channel, voltage, sample_rate in iter_filtered_channels(
            round_dir, only_channels=only_channels, max_seconds=max_seconds):
        online = online_by_channel.get(channel)
        if online is None:
            if verbose:
                print(f"  {channel}: no online detections in spike.dat — skipped "
                      f"(nothing to match against)")
            continue
        if max_seconds is not None:
            online = online[online < voltage.size / sample_rate]

        summary, rows, diag = calibrate_channel(
            channel, voltage, sample_rate, online, multipliers=multipliers,
            noise_methods=noise_methods, refractory_ms=refractory_ms,
            online_percentile=online_percentile)
        summary.update({"Date": date, "Round No.": int(round_no)})
        for r in rows:
            r.update({"Date": date, "Round No.": int(round_no)})
        summaries.append(summary)
        sweep_rows.extend(rows)

        if verbose:
            mult = summary["implied_mult_mad"]
            if summary["trustworthy"]:
                note = ""
            elif summary["n_online"] < MIN_ONLINE_SPIKES:
                note = f"   [not trustworthy: only {summary['n_online']} online spikes]"
            else:
                note = "   [not trustworthy: positive-going online waveform]"
            print(f"  {channel}: {summary['n_online']} online spikes, "
                  f"sigma_mad={summary['sigma_mad_uv']:.1f} µV, "
                  f"online thr≈{summary[f'online_thr_p{online_percentile:g}_uv']:.1f} µV "
                  f"→ implied multiplier {mult:.2f}{note}")

        if plot_dir:
            plot_channel_calibration(summary, rows, diag, sample_rate,
                                     out_dir=plot_dir, date=date, round_no=round_no)

    channels_df = pd.DataFrame(summaries)
    sweep_df = pd.DataFrame(sweep_rows)
    return channels_df, sweep_df


def calibrate_sessions(sessions, **kwargs):
    """Calibrate several ``(date, round_no)`` sessions and concatenate the tables.

    One multiplier serves the whole cache, so the choice should be made across
    sessions, not from whichever one happened to be open.
    """
    channel_frames, sweep_frames = [], []
    for date, round_no in sessions:
        try:
            ch, sw = calibrate_session(date, round_no, **kwargs)
        except Exception as exc:                       # a missing spike.dat must not stop the pass
            print(f"[calib] FAILED {date} round {round_no}: {type(exc).__name__}: {exc}")
            continue
        channel_frames.append(ch)
        sweep_frames.append(sw)
    empty = pd.DataFrame()
    return (pd.concat(channel_frames, ignore_index=True) if channel_frames else empty,
            pd.concat(sweep_frames, ignore_index=True) if sweep_frames else empty)


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def grid_summary(sweep_df, *, trustworthy_only=True):
    """Collapse the sweep to one row per (noise method, multiplier, refractory).

    ``rate_mismatch`` is the median ``|log2(rate_ratio)|`` across channels -- 0 means
    the offline detector fires as often as the online one on the typical channel, and
    log-space so "twice as many" and "half as many" are penalised equally.
    ``p2p_mismatch`` is the same measure on mean-waveform amplitude.
    """
    df = sweep_df
    if trustworthy_only and "trustworthy" in df.columns:
        df = df[df["trustworthy"]]
    if df.empty:
        return pd.DataFrame()

    def _mismatch(ratios):
        r = np.asarray(ratios, dtype=float)
        r = r[np.isfinite(r) & (r > 0)]
        return float(np.median(np.abs(np.log2(r)))) if r.size else float("nan")

    out = []
    keys = ["noise_method", "threshold_multiplier", "refractory_ms"]
    for key, g in df.groupby(keys, dropna=False):
        out.append(dict(zip(keys, key), **{
            "n_channels": int(len(g)),
            "median_rate_ratio": float(np.nanmedian(g["rate_ratio"])),
            "rate_mismatch": _mismatch(g["rate_ratio"]),
            "median_p2p_ratio": float(np.nanmedian(g["p2p_ratio"])),
            "p2p_mismatch": _mismatch(g["p2p_ratio"]),
            "median_recall": float(np.nanmedian(g["recall_of_online"])),
            "median_precision": float(np.nanmedian(g["precision_vs_online"])),
        }))
    return pd.DataFrame(out).sort_values(["noise_method", "refractory_ms",
                                          "threshold_multiplier"], ignore_index=True)


def recommend(channels_df, sweep_df, *, trustworthy_only=True):
    """Name the parameters that best reproduce the online look. Returns a dict.

    Three independent answers, so a bad one is visible rather than authoritative:

      ``implied``     the median multiplier the online thresholds themselves imply
                      (from trough depths — no sweep involved);
      ``rate_match``  the swept grid point whose spike rate best matches online —
                      this is what raster density follows;
      ``p2p_match``   the grid point whose mean-waveform amplitude best matches —
                      this is what the waveform panel follows.

    They should land close together. If they don't, believe the one that matches the
    figure you are trying to reproduce, and read the per-channel spread before
    treating any of them as a global setting.
    """
    grid = grid_summary(sweep_df, trustworthy_only=trustworthy_only)
    ch = channels_df
    if trustworthy_only and "trustworthy" in ch.columns:
        ch = ch[ch["trustworthy"]]

    out = {"n_channels": int(len(ch)), "grid": grid}
    for method in ("mad", "rms"):
        col = f"implied_mult_{method}"
        vals = pd.to_numeric(ch.get(col), errors="coerce").dropna() if col in ch else pd.Series(dtype=float)
        out[f"implied_{method}"] = {
            "median": float(vals.median()) if len(vals) else float("nan"),
            "q25": float(vals.quantile(0.25)) if len(vals) else float("nan"),
            "q75": float(vals.quantile(0.75)) if len(vals) else float("nan"),
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "n": int(len(vals)),
        }
    if not grid.empty:
        for name, col in (("rate_match", "rate_mismatch"), ("p2p_match", "p2p_mismatch")):
            usable = grid.dropna(subset=[col])
            out[name] = (usable.loc[usable[col].idxmin()].to_dict()
                         if not usable.empty else None)
    return out


def print_recommendation(rec):
    """Human-readable version of :func:`recommend`, in the order you should read it."""
    print("\n" + "=" * 78)
    print(f" MUA CALIBRATION — {rec['n_channels']} trustworthy channel(s)")
    print("=" * 78)
    for method in ("mad", "rms"):
        imp = rec[f"implied_{method}"]
        if imp["n"]:
            print(f"  online thresholds imply, in units of sigma_{method}: "
                  f"median {imp['median']:.2f}  (IQR {imp['q25']:.2f}–{imp['q75']:.2f}, "
                  f"range {imp['min']:.2f}–{imp['max']:.2f}, n={imp['n']})")
    for name, label in (("rate_match", "best RATE match (raster density)"),
                        ("p2p_match", "best AMPLITUDE match (waveform panel)")):
        g = rec.get(name)
        if g:
            print(f"  {label}: {g['noise_method']} x{g['threshold_multiplier']} "
                  f"ref{g['refractory_ms']}  "
                  f"(median rate ratio {g['median_rate_ratio']:.2f}, "
                  f"p2p ratio {g['median_p2p_ratio']:.2f}, "
                  f"recall {g['median_recall']:.2f}, precision {g['median_precision']:.2f})")
    imp = rec["implied_mad"]
    if imp["n"] and np.isfinite(imp["q25"]) and np.isfinite(imp["q75"]):
        spread = imp["q75"] - imp["q25"]
        if spread > 1.0:
            print(f"\n  NOTE: the implied multipliers spread over {spread:.1f} sigma "
                  f"(IQR) across channels — that is the hand-set drift the offline\n"
                  f"  detector was introduced to remove, and no single global multiplier "
                  f"will match every channel. Match per figure, not per cache.")
    if not rec["grid"].empty:
        print("\n  full grid (median across channels):")
        cols = ["noise_method", "threshold_multiplier", "refractory_ms", "n_channels",
                "median_rate_ratio", "median_p2p_ratio", "median_recall", "median_precision"]
        print(rec["grid"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# Diagnostic figure
# --------------------------------------------------------------------------- #
def plot_channel_calibration(summary, sweep_rows, diag, sample_rate, *, out_dir,
                             date, round_no, max_candidates=4):
    """One PNG per channel: waveform overlay, trough-depth histogram, rate/amplitude curves.

    The left panel is the one to look at — the online mean waveform against each
    candidate multiplier's. When a candidate's waveform sits on top of the online one,
    that multiplier reproduces the waveform panel of the raster review. The middle panel
    is the evidence behind the implied threshold: the online spikes' trough depths, with
    the estimated online threshold at the shallow edge and each candidate's threshold
    drawn over them.

    ``matplotlib`` is imported here, not at module scope, so importing this module never
    pins a backend (``zombies_raster`` forces Agg, which is why its width helper moved
    into ``waveforms``).
    """
    import matplotlib.pyplot as plt

    channel = summary["Channel"]
    os.makedirs(out_dir, exist_ok=True)
    rows = sorted(sweep_rows, key=lambda r: r["threshold_multiplier"])
    if not rows:
        return None
    waves = diag.get("candidate_waves", {})

    # Candidates for the two left panels: a spread around the best rate match, but only
    # within ONE (noise method, refractory) family -- mixing families in one overlay
    # makes two different thresholds share a label and reads as noise.
    finite = [r for r in rows if np.isfinite(r.get("rate_ratio", np.nan)) and r["rate_ratio"] > 0]
    best = min(finite, key=lambda r: abs(np.log2(r["rate_ratio"]))) if finite else rows[0]
    family = [r for r in rows if r["noise_method"] == best["noise_method"]
              and r["refractory_ms"] == best["refractory_ms"]]
    order = sorted(family, key=lambda r: abs(r["threshold_multiplier"] - best["threshold_multiplier"]))
    chosen = sorted(order[:max_candidates], key=lambda r: r["threshold_multiplier"])
    cmap = plt.get_cmap("viridis")
    colors = {id(r): cmap(i / max(len(chosen) - 1, 1)) for i, r in enumerate(chosen)}

    fig, (ax_w, ax_h, ax_r) = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- mean waveforms -----------------------------------------------------
    online_wave = np.asarray(diag.get("online_wave", []), dtype=float)
    if online_wave.size:
        t_ms = (np.arange(online_wave.size) - online_wave.size / 2.0) / sample_rate * 1000.0
        ax_w.plot(t_ms, online_wave, color="k", lw=2.0,
                  label=f"online (unsorted)  {summary['online_p2p_uv']:.0f} µV")
    for r in chosen:
        w = np.asarray(waves.get((r["noise_method"], r["threshold_multiplier"],
                                  r["refractory_ms"]), []), dtype=float)
        if not w.size:
            continue
        t_ms = (np.arange(w.size) - w.size / 2.0) / sample_rate * 1000.0
        ax_w.plot(t_ms, w, lw=1.2, color=colors[id(r)],
                  label=f"{r['noise_method']} x{r['threshold_multiplier']:g}  "
                        f"{r['p2p_uv']:.0f} µV")
    ax_w.axvline(0, color="0.7", lw=0.6, ls=":")
    ax_w.set_xlabel("time (ms)")
    ax_w.set_ylabel("amplitude (µV)")
    ax_w.set_title(f"mean waveform — {channel}", fontsize=10)
    ax_w.legend(fontsize=7, framealpha=0.85)

    # --- trough-depth histogram + candidate thresholds ----------------------
    depths = np.asarray(diag.get("online_depths", []), dtype=float)
    thr_key = next((k for k in summary if k.startswith("online_thr_p")), None)
    if depths.size:
        # clip the long amplitude tail so the shallow edge -- the part that carries the
        # threshold -- is not squashed into the first bin
        hi = float(np.percentile(depths, 99))
        ax_h.hist(depths[depths <= hi], bins=60, color="0.75", edgecolor="none")
        if thr_key and np.isfinite(summary[thr_key]):
            ax_h.axvline(summary[thr_key], color="crimson", lw=1.8,
                         label=f"online threshold ≈ {summary[thr_key]:.0f} µV")
        ax_h.axvline(summary["online_trough_min_uv"], color="0.4", ls=":", lw=1,
                     label=f"shallowest online {summary['online_trough_min_uv']:.0f} µV")
    for r in chosen:
        ax_h.axvline(-r["threshold_uv"], lw=1.2, ls="--", color=colors[id(r)],
                     label=f"{r['noise_method']} x{r['threshold_multiplier']:g} → "
                           f"{-r['threshold_uv']:.0f} µV")
    ax_h.set_xlabel("trough depth (µV, positive)")
    ax_h.set_ylabel("online spikes")
    ax_h.set_title("online spike depths vs candidate thresholds", fontsize=10)
    ax_h.legend(fontsize=7, framealpha=0.85)

    # --- rate and amplitude vs multiplier -----------------------------------
    families = sorted({(r["noise_method"], r["refractory_ms"]) for r in rows})
    for method, refr in families:
        g = [r for r in rows if r["noise_method"] == method and r["refractory_ms"] == refr]
        mults = [r["threshold_multiplier"] for r in g]
        tag = f"{method} ref{refr:g}"
        ax_r.plot(mults, [r["rate_ratio"] for r in g], "o-", ms=4, label=f"{tag} rate")
        ax_r.plot(mults, [r["p2p_ratio"] for r in g], "s--", ms=4, label=f"{tag} p2p")
    ax_r.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("threshold multiplier")
    ax_r.set_ylabel("offline / online")
    ax_r.set_title("match vs multiplier (1.0 = identical)", fontsize=10)
    ax_r.legend(fontsize=7, framealpha=0.85)

    imp = summary.get("implied_mult_mad", float("nan"))
    fig.suptitle(f"{date} round {round_no} · {channel} · "
                 f"sigma_mad {summary['sigma_mad_uv']:.1f} µV · "
                 f"implied multiplier {imp:.2f}"
                 + ("" if summary["trustworthy"] else "  [NOT TRUSTWORTHY]"),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    safe = channel.replace(".", "_").replace(" ", "")
    path = os.path.join(out_dir, f"{date}_r{round_no}_{safe}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def output_dir(monkey=SUBJECT_MONKEY) -> Path:
    return Path(DATA_BASE_PATH) / monkey / "mua_calibration"


def write_tables(channels_df, sweep_df, rec, *, out_dir, label):
    """Write the three tables and return their paths (nothing else is written)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("channels", channels_df), ("sweep", sweep_df),
                     ("grid", rec.get("grid"))):
        if df is None or getattr(df, "empty", True):
            continue
        p = out_dir / f"{label}_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        print(f"  wrote {p}")
    return paths


def _parse_sessions(spec):
    """``'2023-09-26:2,2023-10-27:3'`` -> ``[('2023-09-26', 2), ('2023-10-27', 3)]``."""
    out = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        date, _, rnd = token.partition(":")
        out.append((date.strip(), int(rnd)))
    return out


def run(sessions, *, monkey=SUBJECT_MONKEY, multipliers=DEFAULT_MULTIPLIERS,
        noise_methods=("mad",), refractory_ms=(1.0,), only_channels=None,
        max_seconds=None, plots=False, out_dir=None, label=None):
    """Calibrate, recommend, write. The one entry point the CLI and CONFIG both call."""
    out_dir = Path(out_dir) if out_dir else output_dir(monkey)
    label = label or ("_".join(f"{d}r{r}" for d, r in sessions)
                      if len(sessions) <= 3 else f"{len(sessions)}sessions")
    plot_dir = str(out_dir / f"{label}_plots") if plots else None

    channels_df, sweep_df = calibrate_sessions(
        sessions, monkey=monkey, multipliers=multipliers, noise_methods=noise_methods,
        refractory_ms=refractory_ms, only_channels=only_channels,
        max_seconds=max_seconds, plot_dir=plot_dir)
    if channels_df.empty:
        print("[calib] nothing calibrated — no channel had both a filtered trace and "
              "online detections.")
        return channels_df, sweep_df, {}

    rec = recommend(channels_df, sweep_df)
    print_recommendation(rec)
    write_tables(channels_df, sweep_df, rec, out_dir=out_dir, label=label)
    if plot_dir:
        print(f"  wrote per-channel figures to {plot_dir}")
    return channels_df, sweep_df, rec


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="single session date, YYYY-MM-DD")
    p.add_argument("--round", type=int, dest="round_no", help="single session round number")
    p.add_argument("--sessions", help="'2023-09-26:2,2023-10-27:3' for several at once")
    p.add_argument("--monkey", default=SUBJECT_MONKEY)
    p.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS),
                   help="comma-separated multipliers to sweep")
    p.add_argument("--noise-methods", default="mad", help="'mad', 'rms', or 'mad,rms'")
    p.add_argument("--refractory-ms", default="1.0", help="comma-separated refractory periods")
    p.add_argument("--channels", default=None,
                   help="restrict to these channels, e.g. 'C_005,C_026'")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="calibrate on the first N seconds only (faster)")
    p.add_argument("--plots", action="store_true", help="also write per-channel figures")
    p.add_argument("--out", default=None, help="output directory (default <data>/<monkey>/mua_calibration)")
    a = p.parse_args(argv)

    if a.sessions:
        sessions = _parse_sessions(a.sessions)
    elif a.date and a.round_no is not None:
        sessions = [(a.date, a.round_no)]
    else:
        p.error("give --date and --round, or --sessions")
    return run(sessions, monkey=a.monkey,
               multipliers=tuple(float(m) for m in a.multipliers.split(",")),
               noise_methods=tuple(m.strip() for m in a.noise_methods.split(",")),
               refractory_ms=tuple(float(r) for r in a.refractory_ms.split(",")),
               only_channels=[c.strip() for c in a.channels.split(",")] if a.channels else None,
               max_seconds=a.max_seconds, plots=a.plots, out_dir=a.out)


# ===== Run directly in PyCharm — edit this block and hit Run (no CLI) =========
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:                     # called with flags -> behave as a CLI
        _cli()
        raise SystemExit(0)

    # Sessions to calibrate. One is enough to look at; use several before changing a
    # cache-wide default, since one multiplier has to serve them all.
    SESSIONS = [("2023-09-26", 2)]

    MULTIPLIERS = DEFAULT_MULTIPLIERS         # the sweep grid
    NOISE_METHODS = ("mad",)                  # ("mad", "rms") to compare noise estimators
    REFRACTORY_MS = (1.0,)                    # e.g. (1.0, 1.5, 2.0)
    ONLY_CHANNELS = None                      # e.g. ["C_005", "C_026"]; None = all
    MAX_SECONDS = None                        # e.g. 300.0 for a fast first pass
    PLOTS = True                              # per-channel diagnostic PNGs

    run(SESSIONS, multipliers=MULTIPLIERS, noise_methods=NOISE_METHODS,
        refractory_ms=REFRACTORY_MS, only_channels=ONLY_CHANNELS,
        max_seconds=MAX_SECONDS, plots=PLOTS)

    # NEXT STEPS once you have a multiplier M (nothing here writes a spike cache):
    #
    #   1. rebuild the MUA cache with it — this writes a SEPARATE pkl per parameter set
    #      (the label carries them: '{date}_round_{n}_mad{M}_ref{r}.pkl'), so the
    #      existing mad4.0 cache is left alone:
    #
    #        from data_access.rebuild_peristim_caches import rebuild
    #        rebuild("mua", threshold_multiplier=M)
    #
    #   2. read it back by asking for the same parameters — a source built with other
    #      parameters looks for a filename that does not exist:
    #
    #        ThresholdMUASpikeSource(threshold_multiplier=M)
    #
    #   3. compare against the unsorted lane on real figures. The side-by-side already
    #      exists: analyses.jun2026_grant_investigation.raster_review_by_source with
    #      MODE="pairs" draws lane A = unsorted (exploded cache) and lane B = threshold
    #      MUA for the SAME channel, with the waveform panel. Its threshold-MUA lane comes
    #      from spike_count_connector._mua_source(), which hardcodes threshold_multiplier
    #      =4.0 -- edit that to M first or it keeps loading the mad4.0 cache, and put it
    #      back afterwards unless you mean to move the MUA_KW / MUA_ANOVA lists too.
