"""
plots.py — figures for judging whether cross-channel units are the same neuron.

Session-level overview
    * :func:`plot_coincidence_matrix` — units × units synchrony heatmap; duplicate
      pairs light up as off-diagonal hot cells.
    * :func:`plot_distance_vs_coincidence` — every unit pair as a point; genuine
      duplicates cluster at (small channel distance, high coincidence).

Per candidate pair
    * :func:`plot_pair_report` — one multi-panel figure per suspected duplicate:
      cross-correlogram, the two auto-correlograms + the merged-train ACG
      (refractory check), ISI histograms, and — if voltages are supplied — the
      two spatial footprints side by side.

All functions return a Matplotlib ``Figure`` (Agg-safe, no GUI needed).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from . import spiketrain_metrics as stm
from . import waveforms as wf


# --------------------------------------------------------------------------- #
# Session overview
# --------------------------------------------------------------------------- #
def plot_coincidence_matrix(session, *, window_ms: float = 0.4, ax=None):
    """Heatmap of pairwise coincidence fraction across all units."""
    mat, uids = stm.coincidence_matrix(session, window_ms=window_ms)
    if ax is None:
        size = max(6, 0.5 * len(uids))
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(uids)))
    ax.set_yticks(range(len(uids)))
    ax.set_xticklabels(uids, rotation=90, fontsize=7)
    ax.set_yticklabels(uids, fontsize=7)
    ax.set_title(f"Coincidence fraction (±{window_ms} ms window)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coincidence")
    fig.tight_layout()
    return fig


def plot_distance_vs_coincidence(pairs: List[stm.PairMetrics], *, ax=None):
    """Scatter of channel distance vs coincidence for every unit pair.

    Duplicates live in the top-left (close channels, high synchrony). Points
    with unknown distance (off-probe channels) are drawn at the right edge.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    known = [p for p in pairs if p.distance_um is not None]
    unknown = [p for p in pairs if p.distance_um is None]
    if known:
        ax.scatter([p.distance_um for p in known], [p.coincidence for p in known],
                   c="#1f77b4", s=30, alpha=0.7, label="on-probe pair")
    if unknown:
        xmax = max([p.distance_um for p in known], default=100.0)
        ax.scatter([xmax * 1.15] * len(unknown), [p.coincidence for p in unknown],
                   c="0.6", s=30, alpha=0.7, marker="x", label="unknown distance")
    ax.axhline(0.3, color="crimson", ls="--", lw=1, label="candidate threshold")
    ax.set_xlabel("channel distance (µm)")
    ax.set_ylabel("coincidence fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Channel distance vs spike-train synchrony")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Correlogram helpers
# --------------------------------------------------------------------------- #
def _bar_correlogram(ax, corr: stm.Correlogram, color, title, refractory_ms=1.5):
    width = corr.bin_size_ms
    ax.bar(corr.bin_centers_ms, corr.counts, width=width, color=color, align="center")
    ax.axvspan(-refractory_ms, refractory_ms, color="crimson", alpha=0.12, zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lag (ms)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)


def _isi_hist(ax, spikes, sample_rate, color, title, max_ms=50.0, refractory_ms=1.5):
    if np.asarray(spikes).size >= 2:
        isi_ms = np.diff(np.asarray(spikes)) / sample_rate * 1e3
        isi_ms = isi_ms[isi_ms <= max_ms]
        ax.hist(isi_ms, bins=np.linspace(0, max_ms, 60), color=color)
    ax.axvspan(0, refractory_ms, color="crimson", alpha=0.12, zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ISI (ms)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# Per-pair report
# --------------------------------------------------------------------------- #
def plot_pair_report(
    session,
    pair: stm.PairMetrics,
    *,
    ccg_window_ms: float = 25.0,
    ccg_bin_ms: float = 0.5,
    refractory_ms: float = 1.5,
    voltages_by_channel: Optional[Dict[str, np.ndarray]] = None,
):
    """Full diagnostic figure for one suspected duplicate pair."""
    ua = session.unit(pair.uid_a)
    ub = session.unit(pair.uid_b)
    sr = session.sample_rate

    ca = "#1f77b4"
    cb = "#ff7f0e"
    cmerge = "#2ca02c"

    have_wf = voltages_by_channel is not None
    n_rows = 3 if have_wf else 2
    fig = plt.figure(figsize=(12, 3.4 * n_rows))
    gs = fig.add_gridspec(n_rows, 3)

    # Row 0: CCG (spanning) + coincidence text
    ax_ccg = fig.add_subplot(gs[0, :2])
    ccg = stm.cross_correlogram(ua.spike_indices, ub.spike_indices,
                                sample_rate=sr, window_ms=ccg_window_ms, bin_size_ms=ccg_bin_ms)
    _bar_correlogram(ax_ccg, ccg, "0.3",
                     f"Cross-correlogram  {ua.uid}  →  {ub.uid}", refractory_ms)

    ax_txt = fig.add_subplot(gs[0, 2]); ax_txt.axis("off")
    dist = "n/a" if pair.distance_um is None else f"{pair.distance_um:.0f} µm"
    gap = "n/a" if pair.contacts_apart is None else str(pair.contacts_apart)
    verdict = _verdict(pair)
    ax_txt.text(0.0, 1.0, "\n".join([
        f"coincidence:      {pair.coincidence:.3f}",
        f"× over chance:    {pair.ratio_over_chance:.1f}",
        f"channel distance: {dist}  ({gap} contacts)",
        f"spikes:  A={pair.n_a}   B={pair.n_b}",
        f"refractory  A={pair.refractory_a:.3f}",
        f"            B={pair.refractory_b:.3f}",
        f"     merged   ={pair.refractory_merged:.3f}",
        "",
        f"→ {verdict}",
    ]), va="top", ha="left", family="monospace", fontsize=10)

    # Row 1: ACG A, ACG B, ACG merged
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_m = fig.add_subplot(gs[1, 2])
    acg_a = stm.auto_correlogram(ua.spike_indices, sample_rate=sr,
                                 window_ms=ccg_window_ms, bin_size_ms=ccg_bin_ms)
    acg_b = stm.auto_correlogram(ub.spike_indices, sample_rate=sr,
                                 window_ms=ccg_window_ms, bin_size_ms=ccg_bin_ms)
    merged = stm.merge_trains(ua.spike_indices, ub.spike_indices, sample_rate=sr)
    acg_m = stm.auto_correlogram(merged, sample_rate=sr,
                                 window_ms=ccg_window_ms, bin_size_ms=ccg_bin_ms)
    _bar_correlogram(ax_a, acg_a, ca, f"ACG {ua.uid}", refractory_ms)
    _bar_correlogram(ax_b, acg_b, cb, f"ACG {ub.uid}", refractory_ms)
    _bar_correlogram(ax_m, acg_m, cmerge, "ACG merged train", refractory_ms)

    # Row 2 (optional): spatial footprints
    if have_wf:
        fa = wf.compute_footprint(ua.uid, ua.spike_indices, voltages_by_channel)
        fb = wf.compute_footprint(ub.uid, ub.spike_indices, voltages_by_channel)
        sim = wf.footprint_similarity(fa, fb)
        ax_fa = fig.add_subplot(gs[2, 0])
        ax_fb = fig.add_subplot(gs[2, 1])
        ax_ov = fig.add_subplot(gs[2, 2])
        _plot_footprint(ax_fa, fa, ca, f"Footprint {ua.uid}")
        _plot_footprint(ax_fb, fb, cb, f"Footprint {ub.uid}")
        # overlay of the two peak-channel waveforms
        _plot_peak_overlay(ax_ov, fa, ca, fb, cb, sim)

    fig.suptitle(f"Duplicate check:  {ua.uid}   vs   {ub.uid}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _verdict(pair: stm.PairMetrics) -> str:
    if pair.coincidence >= 0.5 and pair.ratio_over_chance >= 5:
        base = "LIKELY SAME UNIT"
    elif pair.coincidence >= 0.3 and pair.ratio_over_chance >= 5:
        base = "POSSIBLE DUPLICATE"
    else:
        base = "likely distinct"
    if not pair.merged_refractory_ok:
        base += "\n  (merged train breaks\n   refractoriness — caution)"
    return base


def _plot_footprint(ax, fp: wf.Footprint, color, title):
    """Stacked mean waveforms down the probe, peak channel highlighted."""
    n = len(fp.channels)
    offsets = fp.waveforms.max() - fp.waveforms.min()
    step = (offsets if offsets > 0 else 1.0) * 0.6
    for i, ch in enumerate(fp.channels):
        y = fp.waveforms[i] + (n - i) * step
        lw = 2.0 if ch == fp.peak_channel else 0.7
        c = color if ch == fp.peak_channel else "0.6"
        ax.plot(y, color=c, lw=lw)
        ax.text(-1, (n - i) * step, ch, fontsize=5, ha="right", va="center")
    ax.set_title(f"{title}\npeak {fp.peak_channel}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_peak_overlay(ax, fa, ca, fb, cb, sim):
    ia = fa.channels.index(fa.peak_channel)
    ib = fb.channels.index(fb.peak_channel)
    ax.plot(fa.waveforms[ia], color=ca, lw=1.6, label=f"{fa.uid} @ {fa.peak_channel}")
    ax.plot(fb.waveforms[ib], color=cb, lw=1.6, label=f"{fb.uid} @ {fb.peak_channel}")
    ax.set_title(f"peak-channel waveforms\nfootprint cosine sim = {sim:.3f}", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
