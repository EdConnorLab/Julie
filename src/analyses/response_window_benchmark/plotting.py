"""
plotting.py — figures for the bake-off.

* :func:`plot_cell_raster`        raster + PSTH only (annotation phase — no
                                  algorithm windows shown, so you annotate blind)
* :func:`plot_cell_with_windows`  raster + PSTH + one window-lane per method plus
                                  a TRUTH lane (scoring phase)

The pre-stimulus window [-pre, 0) is shaded grey (the baseline the vs-baseline
detectors use); the dashed line at 0 is stimulus onset; window lanes live on the
post-stim window [0, t_stop].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from analyses.zombies_raster_review.zombies_raster import zombies_monkey_order
from .psth import extract_trials, trial_averaged_rate, TrialData
from .detectors import WindowDetector, DetectorResult

Window = Tuple[float, float]
TRUTH_COLOR = "#111111"
BASELINE_SHADE = "#EAEAEA"


def _grouped_trials(td: TrialData):
    """[(monkey, [trial spike arrays])] in rank order, subject excluded."""
    ordered, _ = zombies_monkey_order(pd.unique(td.labels)) if td.n_trials else ([], {})
    groups = []
    for m in ordered:
        idx = [i for i in range(td.n_trials) if td.labels[i] == m]
        if idx:
            groups.append((m, [td.trials[i] for i in idx]))
    return groups


def _draw_raster_psth(ax_r, ax_p, td, *, truth_windows, pre, xlim, bin_s):
    left = -pre if pre > 0 else 0.0
    groups = _grouped_trials(td)
    total = sum(len(t) for _, t in groups)

    # raster
    y, tick_pos, tick_lab = 0, [], []
    for i, (m, trials) in enumerate(groups):
        n = len(trials)
        if i % 2 == 0:
            ax_r.axhspan(y - 0.5, y + n - 0.5, color="0.96", zorder=0)
        ax_r.eventplot(trials, lineoffsets=np.arange(y, y + n),
                       colors="black", linewidths=0.8, linelengths=0.9)
        tick_pos.append(y + n / 2 - 0.5)
        tick_lab.append(f"{m}  {n}tr")
        y += n
    if pre > 0:
        ax_r.axvspan(left, 0, color=BASELINE_SHADE, zorder=0)
    ax_r.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    for lo, hi in (truth_windows or []):
        ax_r.axvspan(lo, hi, color=TRUTH_COLOR, alpha=0.08, zorder=0)
    ax_r.set_ylim(-0.5, max(total - 0.5, 0.5))
    ax_r.invert_yaxis()
    ax_r.set_yticks(tick_pos)
    ax_r.set_yticklabels(tick_lab, fontsize=8)
    ax_r.tick_params(axis="y", length=0)
    ax_r.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax_r.spines[s].set_visible(False)

    # PSTH over the full shown range
    centers, mean, sem = trial_averaged_rate(td, bin_s, lo=left, hi=xlim)
    ax_p.fill_between(centers, mean - sem, mean + sem, color="#4E79A7", alpha=0.25)
    ax_p.plot(centers, mean, color="#2F5D8A", lw=1.5)
    if pre > 0:
        ax_p.axvspan(left, 0, color=BASELINE_SHADE, zorder=0)
    ax_p.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    for lo, hi in (truth_windows or []):
        ax_p.axvspan(lo, hi, color=TRUTH_COLOR, alpha=0.10)
    ax_p.set_ylim(bottom=0)
    ax_p.set_ylabel("rate (Hz)", fontsize=9)
    ax_p.set_xlim(left, xlim)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)
    return total


def _save_or_keep(fig, save_path):
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)


def plot_cell_raster(neuron_df, *, title="", bin_s=0.05, xlim=2.4,
                     pre_stimulus_time=0.0, spike_col="SpikeTimes", save_path=None):
    """Raster + PSTH only (no algorithm windows) — for unbiased annotation."""
    td = extract_trials(neuron_df, spike_col=spike_col, t_stop=xlim,
                        pre_stimulus_time=pre_stimulus_time)
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.1], hspace=0.1, figure=fig)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_r)
    total = _draw_raster_psth(ax_r, ax_p, td, truth_windows=None,
                              pre=pre_stimulus_time, xlim=xlim, bin_s=bin_s)
    if total == 0:
        print(f"[skip] {title}: no trials")
        plt.close(fig)
        return None
    ax_r.set_title(title, fontsize=11, loc="left")
    ax_p.set_xlabel("time from stimulus onset (s)", fontsize=10)
    _save_or_keep(fig, save_path)
    return fig


def _draw_windows_lane(ax, y, windows, color, *, height=0.72):
    for lo, hi in windows or []:
        ax.add_patch(Rectangle((lo, y - height / 2), max(hi - lo, 1e-3), height,
                               facecolor=color, edgecolor="none", alpha=0.85))


def plot_cell_with_windows(
    neuron_df,
    detectors: Sequence[WindowDetector],
    results: Dict[str, DetectorResult],
    *,
    truth_windows: Optional[List[Window]] = None,
    title: str = "",
    bin_s: float = 0.05,
    xlim: float = 2.4,
    pre_stimulus_time: float = 0.0,
    spike_col: str = "SpikeTimes",
    save_path: Optional[str] = None,
):
    """Raster + PSTH + per-method window lanes for one unit.

    ``results`` maps ``detector.name`` -> :class:`DetectorResult`. ``detectors``
    fixes lane order/colours. ``truth_windows`` (if given) is a bold top lane and
    is faintly shaded across raster/PSTH.
    """
    td = extract_trials(neuron_df, spike_col=spike_col, t_stop=xlim,
                        pre_stimulus_time=pre_stimulus_time)
    left = -pre_stimulus_time if pre_stimulus_time > 0 else 0.0
    n_lanes = len(detectors) + (1 if truth_windows is not None else 0)
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 1, height_ratios=[3.2, 1.1, max(1.0, 0.32 * n_lanes)],
                  hspace=0.12, figure=fig)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_r)
    ax_w = fig.add_subplot(gs[2], sharex=ax_r)

    total = _draw_raster_psth(ax_r, ax_p, td, truth_windows=truth_windows,
                              pre=pre_stimulus_time, xlim=xlim, bin_s=bin_s)
    if total == 0:
        print(f"[skip] {title}: no trials")
        plt.close(fig)
        return None
    ax_r.set_title(title, fontsize=11, loc="left")
    ax_p.tick_params(labelbottom=False)

    # window lanes
    lanes: List[Tuple[str, List[Window], str]] = []
    if truth_windows is not None:
        lanes.append(("TRUTH", truth_windows, TRUTH_COLOR))
    for det in detectors:
        res = results.get(det.name)
        lanes.append((det.name, res.windows if res else [], det.color))

    lane_labels = []
    for idx, (label, windows, color) in enumerate(lanes):
        yy = len(lanes) - 1 - idx  # truth on top
        _draw_windows_lane(ax_w, yy, windows, color)
        lane_labels.append((yy, label))
    if pre_stimulus_time > 0:
        ax_w.axvspan(left, 0, color=BASELINE_SHADE, zorder=0)
    ax_w.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax_w.set_ylim(-0.6, len(lanes) - 0.4)
    ax_w.set_yticks([yy for yy, _ in lane_labels])
    ax_w.set_yticklabels([lab for _, lab in lane_labels], fontsize=8)
    ax_w.tick_params(axis="y", length=0)
    ax_w.set_xlabel("time from stimulus onset (s)", fontsize=10)
    ax_w.set_xlim(left, xlim)
    for s in ("top", "right", "left"):
        ax_w.spines[s].set_visible(False)

    _save_or_keep(fig, save_path)
    return fig
