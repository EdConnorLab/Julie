"""
zombies_raster.py — a readable, single-group raster for the Zombies stimuli.

The existing ``analyses/raster_plotting.py`` draws a grid of tiny per-monkey
eventplots — hard to scan and mostly whitespace. This is a purpose-built
redesign for reviewing *one unit against the Zombies group* (≈9–10 monkeys),
focused on legibility:

    ┌───────────────────────────────────────────────┐
    │  RASTER  — one row per trial, trials stacked    │
    │  and grouped by stimulus monkey in RANK order   │
    │  (dominant → subordinate). Each monkey block:   │
    │    · alternating background shade                │
    │    · spikes coloured by rank (viridis ramp)      │
    │    · monkey id + rank label on the left          │
    │  stimulus onset at t=0 (dashed); the ANOVA       │
    │  significant window shaded in gold.              │
    ├───────────────────────────────────────────────┤
    │  PSTH — trial-averaged firing rate (Hz) across   │
    │  all Zombies trials, ±SEM, same x-axis, window   │
    │  shaded. Makes the response obvious at a glance.  │
    └───────────────────────────────────────────────┘

Input is a per-trial DataFrame for a *single* unit (as returned by a
SpikeSource, filtered to one NeuronID/Channel) with at least the columns
``MonkeyGroup``, ``MonkeyName``, ``EpochStartStop`` and a spike-times column
(default ``"SpikeTimes"``, per-trial list/array of absolute spike times in s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analyses.enums.monkey_names import get_monkeys_by_rank

GROUP = "Zombies"


# --------------------------------------------------------------------------- #
# Data shaping
# --------------------------------------------------------------------------- #
@dataclass
class MonkeyTrials:
    monkey: str
    rank: Optional[int]          # 1 = most dominant; None if unranked
    trials: List[np.ndarray]     # per-trial spike times aligned to stimulus onset


def _align(spikes, start, stop) -> np.ndarray:
    """Spikes within [start, stop] re-zeroed to stimulus onset."""
    s = np.asarray(list(spikes), dtype=float)
    if s.size == 0:
        return s
    s = s[(s >= start) & (s <= stop)]
    return s - start


def zombies_trials_by_monkey(
    neuron_df,
    *,
    spike_col: str = "SpikeTimes",
    group: str = GROUP,
) -> List[MonkeyTrials]:
    """Group a single unit's trials by stimulus monkey, ranked dominant→subordinate.

    Monkeys present but absent from the rank list (e.g. 81G) are appended after
    the ranked ones so nothing is silently dropped.
    """
    df = neuron_df[neuron_df["MonkeyGroup"] == group]
    ranked = get_monkeys_by_rank(group)
    present = list(df["MonkeyName"].dropna().unique())
    ordered = [m for m in ranked if m in present]
    ordered += [m for m in present if m not in ranked]  # unranked → end

    out: List[MonkeyTrials] = []
    for m in ordered:
        mdf = df[df["MonkeyName"] == m]
        trials = [_align(r[spike_col], *r["EpochStartStop"]) for _, r in mdf.iterrows()]
        rank = (ranked.index(m) + 1) if m in ranked else None
        out.append(MonkeyTrials(monkey=str(m), rank=rank, trials=trials))
    return out


# --------------------------------------------------------------------------- #
# PSTH
# --------------------------------------------------------------------------- #
def _psth(all_trials: List[np.ndarray], *, xlim: float, bin_ms: float
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-averaged firing rate (Hz) and SEM across trials."""
    bin_s = bin_ms / 1000.0
    edges = np.arange(0, xlim + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    if not all_trials:
        z = np.zeros(centers.size)
        return centers, z, z
    per_trial = np.stack([np.histogram(t, bins=edges)[0] / bin_s for t in all_trials])
    mean = per_trial.mean(axis=0)
    sem = per_trial.std(axis=0) / np.sqrt(per_trial.shape[0]) if per_trial.shape[0] > 1 else np.zeros_like(mean)
    return centers, mean, sem


# --------------------------------------------------------------------------- #
# Main plot
# --------------------------------------------------------------------------- #
def plot_zombies_raster(
    neuron_df,
    *,
    neuron_label: str,
    spike_col: str = "SpikeTimes",
    window_s: Optional[Tuple[float, float]] = None,
    p_value: Optional[float] = None,
    xlim: float = 2.0,
    psth_bin_ms: float = 50.0,
    save_path: Optional[str] = None,
):
    """Render the raster + PSTH for one unit against the Zombies group.

    Returns the Matplotlib ``Figure`` (or ``None`` if the unit has no Zombies
    trials).
    """
    by_monkey = zombies_trials_by_monkey(neuron_df, spike_col=spike_col)
    by_monkey = [m for m in by_monkey if len(m.trials) > 0]
    total_trials = sum(len(m.trials) for m in by_monkey)
    if total_trials == 0:
        print(f"[skip] {neuron_label}: no Zombies trials")
        return None

    n_ranks = max((m.rank for m in by_monkey if m.rank), default=1)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(8.5, 9))
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_psth = fig.add_subplot(gs[1], sharex=ax)

    # --- raster ---
    y = 0
    all_trials: List[np.ndarray] = []
    tick_pos, tick_labels, tick_colors = [], [], []
    for i, m in enumerate(by_monkey):
        n = len(m.trials)
        color = cmap((m.rank - 1) / max(1, n_ranks - 1)) if m.rank else "0.5"
        # alternating background band spanning this monkey's trials
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + n - 0.5, color="0.96", zorder=0)
        # spikes
        ax.eventplot(m.trials, lineoffsets=np.arange(y, y + n),
                     colors=[color], linewidths=0.9, linelengths=0.9)
        center = y + n / 2 - 0.5
        tick_pos.append(center)
        tick_labels.append(str(m.monkey))
        tick_colors.append(color)
        # rank number on the right edge (axes-fraction x, data-coord y)
        rank_txt = f"#{m.rank}" if m.rank else "unranked"
        ax.text(1.01, center, rank_txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=color, clip_on=False)
        all_trials.extend(m.trials)
        y += n

    # stimulus onset + significant window
    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    if window_s is not None:
        ax.axvspan(window_s[0], window_s[1], color="#F2C94C", alpha=0.30, zorder=1,
                   label="ANOVA sig. window")
    ax.set_ylim(-0.5, total_trials - 0.5)
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()  # most dominant monkey (rank 1) at the top
    ax.set_ylabel("stimulus monkey  (dominant → subordinate)", fontsize=10)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=9)
    for tick_label, c in zip(ax.get_yticklabels(), tick_colors):
        tick_label.set_color(c)
        tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    sub = f"{total_trials} trials · {len(by_monkey)} monkeys"
    if p_value is not None:
        sub += f" · p={p_value:.3g}"
    ax.set_title(f"{neuron_label}\n{sub}", fontsize=11, loc="left")

    # --- PSTH ---
    centers, mean, sem = _psth(all_trials, xlim=xlim, bin_ms=psth_bin_ms)
    ax_psth.fill_between(centers, mean - sem, mean + sem, color="#4E79A7", alpha=0.25)
    ax_psth.plot(centers, mean, color="#2F5D8A", lw=1.5)
    ax_psth.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    if window_s is not None:
        ax_psth.axvspan(window_s[0], window_s[1], color="#F2C94C", alpha=0.30)
    ax_psth.set_xlim(0, xlim)
    ax_psth.set_ylim(bottom=0)
    ax_psth.set_xlabel("time from stimulus onset (s)", fontsize=10)
    ax_psth.set_ylabel("rate (Hz)", fontsize=10)
    for s in ("top", "right"):
        ax_psth.spines[s].set_visible(False)

    # rank colourbar legend
    if n_ranks > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=1, vmax=n_ranks))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.08)
        cbar.set_label("dominance rank", fontsize=8)
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=7)

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)
    return fig
