"""
plotting.py — one comparison figure per cell.

Three stacked panels sharing a time axis:

    ┌───────────────────────────────────────────────┐
    │ RASTER  — trials grouped by stimulus monkey     │
    │           (reuses zombies_raster grouping)       │
    ├───────────────────────────────────────────────┤
    │ PSTH    — trial-averaged rate (Hz) ± SEM         │
    ├───────────────────────────────────────────────┤
    │ WINDOWS — one lane per method (+ a TRUTH lane):  │
    │           each detector's detected windows drawn │
    │           as coloured bars, so you can compare   │
    │           at a glance which method matches truth  │
    └───────────────────────────────────────────────┘

The window-lane design (rather than stacking six translucent bands on the
raster) keeps many methods legible at once.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from analyses.zombies_raster_review.zombies_raster import zombies_trials_by_monkey
from .psth import TrialData, trial_averaged_rate, extract_trials
from .detectors import WindowDetector, DetectorResult

Window = Tuple[float, float]
TRUTH_COLOR = "#111111"


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
    spike_col: str = "SpikeTimes",
    save_path: Optional[str] = None,
):
    """Render the raster + PSTH + per-method window lanes for one unit.

    ``results`` maps ``detector.name`` -> :class:`DetectorResult`. ``detectors``
    fixes lane order and colours. ``truth_windows`` (if given) is drawn as a bold
    top lane and faintly shaded across the raster/PSTH for reference.
    """
    by_monkey = [m for m in zombies_trials_by_monkey(neuron_df, spike_col=spike_col)
                 if len(m.trials) > 0]
    total = sum(len(m.trials) for m in by_monkey)
    if total == 0:
        print(f"[skip] {title}: no trials")
        return None

    n_lanes = len(detectors) + (1 if truth_windows is not None else 0)
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 1, height_ratios=[3.2, 1.1, max(1.0, 0.32 * n_lanes)],
                  hspace=0.12, figure=fig)
    ax_r = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_r)
    ax_w = fig.add_subplot(gs[2], sharex=ax_r)

    # --- raster ---
    y = 0
    tick_pos, tick_lab = [], []
    for i, m in enumerate(by_monkey):
        n = len(m.trials)
        if i % 2 == 0:
            ax_r.axhspan(y - 0.5, y + n - 0.5, color="0.96", zorder=0)
        ax_r.eventplot(m.trials, lineoffsets=np.arange(y, y + n),
                       colors="black", linewidths=0.8, linelengths=0.9)
        tick_pos.append(y + n / 2 - 0.5)
        tick_lab.append(f"{m.monkey}  {n}tr")
        y += n
    ax_r.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    for lo, hi in (truth_windows or []):
        ax_r.axvspan(lo, hi, color=TRUTH_COLOR, alpha=0.08, zorder=0)
    ax_r.set_ylim(-0.5, total - 0.5)
    ax_r.invert_yaxis()
    ax_r.set_yticks(tick_pos)
    ax_r.set_yticklabels(tick_lab, fontsize=8)
    ax_r.tick_params(axis="y", length=0)
    ax_r.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax_r.spines[s].set_visible(False)
    ax_r.set_title(title, fontsize=11, loc="left")

    # --- PSTH (own extraction so it matches the detectors' view) ---
    td = extract_trials(neuron_df, spike_col=spike_col, t_stop=xlim)
    centers, mean, sem = trial_averaged_rate(td, bin_s)
    ax_p.fill_between(centers, mean - sem, mean + sem, color="#4E79A7", alpha=0.25)
    ax_p.plot(centers, mean, color="#2F5D8A", lw=1.5)
    ax_p.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    for lo, hi in (truth_windows or []):
        ax_p.axvspan(lo, hi, color=TRUTH_COLOR, alpha=0.10)
    ax_p.set_ylim(bottom=0)
    ax_p.set_ylabel("rate (Hz)", fontsize=9)
    ax_p.tick_params(labelbottom=False)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)

    # --- window lanes ---
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
    ax_w.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax_w.set_ylim(-0.6, len(lanes) - 0.4)
    ax_w.set_yticks([yy for yy, _ in lane_labels])
    ax_w.set_yticklabels([lab for _, lab in lane_labels], fontsize=8)
    ax_w.tick_params(axis="y", length=0)
    ax_w.set_xlabel("time from stimulus onset (s)", fontsize=10)
    ax_w.set_xlim(0, xlim)
    for s in ("top", "right", "left"):
        ax_w.spines[s].set_visible(False)

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)
    return fig
