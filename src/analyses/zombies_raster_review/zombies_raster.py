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
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analyses.enums.monkey_names import get_monkeys_by_rank

GROUP = "Zombies"

# The recording subject. She is never a stimulus, so she must never appear as a
# row in these rasters even if she shows up in a session's trial table.
SUBJECT_MONKEY_ID = "81G"

# palette for overlay mode (manual vs SI-sorted, etc.)
_OVERLAY_COLORS = ["#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#F28E2B"]


def zombies_monkey_order(present) -> Tuple[list, dict]:
    """Return Zombies monkeys present, ranked dominant→subordinate.

    The recording subject (:data:`SUBJECT_MONKEY_ID`) is dropped. Monkeys present
    but absent from the rank list are appended after the ranked ones (rank
    ``None``) rather than silently dropped. Returns ``(ordered_monkeys,
    rank_by_monkey)``.
    """
    ranked = get_monkeys_by_rank(GROUP)
    present = [m for m in present if m != SUBJECT_MONKEY_ID]
    ordered = [m for m in ranked if m in present]
    ordered += [m for m in present if m not in ranked]
    rank_by_monkey = {m: (ranked.index(m) + 1 if m in ranked else None) for m in ordered}
    return ordered, rank_by_monkey


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

    The recording subject (81G) is excluded — she is never a stimulus. Monkeys
    present but absent from the rank list are appended after the ranked ones.
    """
    df = neuron_df[neuron_df["MonkeyGroup"] == group]
    ordered, rank_by_monkey = zombies_monkey_order(df["MonkeyName"].dropna().unique())

    out: List[MonkeyTrials] = []
    for m in ordered:
        mdf = df[df["MonkeyName"] == m]
        trials = [_align(r[spike_col], *r["EpochStartStop"]) for _, r in mdf.iterrows()]
        out.append(MonkeyTrials(monkey=str(m), rank=rank_by_monkey[m], trials=trials))
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

    _save_or_keep(fig, save_path)
    return fig


def _save_or_keep(fig, save_path):
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Overlay mode — compare two sorts of the same channel on one raster
# --------------------------------------------------------------------------- #
def _aligned_by_key(df, monkey, spike_col, key_col):
    """For one monkey, return ``(keys, aligned_list)`` in row order.

    ``keys`` are the trial keys (from ``key_col``, e.g. TaskField) or ``None`` if
    that column is missing/empty — the caller then falls back to positional
    matching so overlays still line up trial-for-trial.
    """
    mdf = df[df["MonkeyName"] == monkey]
    keys, aligned = [], []
    for _, r in mdf.iterrows():
        aligned.append(_align(r[spike_col], *r["EpochStartStop"]))
        keys.append(r[key_col] if (key_col in mdf.columns and pd.notna(r[key_col])) else None)
    return keys, aligned


def plot_overlay_raster(
    unit_dfs: "dict",
    *,
    title: str,
    spike_col: str = "SpikeTimes",
    key_col: str = "TaskField",
    window_s: Optional[Tuple[float, float]] = None,
    xlim: float = 2.0,
    psth_bin_ms: float = 50.0,
    save_path: Optional[str] = None,
):
    """Overlay several sorts of the *same channel* on one raster for comparison.

    ``unit_dfs`` maps a label (e.g. ``"manual"``, ``"SI"``) to that unit's
    per-trial DataFrame. Trials are grouped by stimulus monkey (rank order,
    subject excluded) and, within each monkey, matched across sorts by
    ``key_col`` (falling back to trial order). Each trial row is split into one
    thin lane per sort, coloured consistently, so you can see spike-for-spike
    where the sorts agree or differ. The PSTH panel overlays each sort's rate.

    Returns the ``Figure`` (or ``None`` if there are no Zombies trials).
    """
    labels = list(unit_dfs.keys())
    colors = {lab: _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)] for i, lab in enumerate(labels)}

    # union of monkeys present across all sorts, ranked, subject excluded
    present = set()
    for df in unit_dfs.values():
        z = df[df["MonkeyGroup"] == GROUP]
        present.update(z["MonkeyName"].dropna().unique())
    ordered, rank_by_monkey = zombies_monkey_order(present)

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_psth = fig.add_subplot(gs[1], sharex=ax)

    lane_h = 0.8 / len(labels)
    y = 0
    tick_pos, tick_labels = [], []
    psth_trials = {lab: [] for lab in labels}

    for i, monkey in enumerate(ordered):
        # matched trials for this monkey, per sort
        per_sort = {}
        for lab in labels:
            zdf = unit_dfs[lab][unit_dfs[lab]["MonkeyGroup"] == GROUP]
            per_sort[lab] = _aligned_by_key(zdf, monkey, spike_col, key_col)

        # decide the row order: keyed union if every sort has usable keys, else positional
        keyed = all(all(k is not None for k in per_sort[lab][0]) and
                    len(set(per_sort[lab][0])) == len(per_sort[lab][0])
                    for lab in labels if per_sort[lab][0])
        if keyed:
            row_keys = []
            for lab in labels:
                for k in per_sort[lab][0]:
                    if k not in row_keys:
                        row_keys.append(k)
            lookup = {lab: dict(zip(per_sort[lab][0], per_sort[lab][1])) for lab in labels}
            n_rows = len(row_keys)
            def spikes_for(lab, r):
                return lookup[lab].get(row_keys[r], np.empty(0))
        else:
            n_rows = max((len(per_sort[lab][1]) for lab in labels), default=0)
            def spikes_for(lab, r):
                lst = per_sort[lab][1]
                return lst[r] if r < len(lst) else np.empty(0)

        if n_rows == 0:
            continue
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + n_rows - 0.5, color="0.96", zorder=0)

        for r in range(n_rows):
            for li, lab in enumerate(labels):
                sp = spikes_for(lab, r)
                if len(sp) == 0:
                    continue
                offset = y + r - 0.4 + lane_h * (li + 0.5)
                ax.eventplot([sp], lineoffsets=offset, colors=[colors[lab]],
                             linewidths=0.9, linelengths=lane_h * 0.9)
                psth_trials[lab].append(sp)

        tick_pos.append(y + n_rows / 2 - 0.5)
        rank = rank_by_monkey[monkey]
        tick_labels.append(f"{monkey}" + (f"  #{rank}" if rank else ""))
        y += n_rows

    total_rows = y
    if total_rows == 0:
        print(f"[skip] {title}: no Zombies trials")
        plt.close(fig)
        return None

    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    if window_s is not None:
        ax.axvspan(window_s[0], window_s[1], color="#F2C94C", alpha=0.25, zorder=1)
    ax.set_ylim(-0.5, total_rows - 0.5)
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()
    ax.set_ylabel("stimulus monkey  (dominant → subordinate)", fontsize=10)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left")

    # legend for the sorts
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[lab], lw=2.2, label=lab) for lab in labels]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    # --- overlaid PSTHs ---
    for lab in labels:
        centers, mean, sem = _psth(psth_trials[lab], xlim=xlim, bin_ms=psth_bin_ms)
        ax_psth.fill_between(centers, mean - sem, mean + sem, color=colors[lab], alpha=0.18)
        ax_psth.plot(centers, mean, color=colors[lab], lw=1.5, label=lab)
    ax_psth.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    if window_s is not None:
        ax_psth.axvspan(window_s[0], window_s[1], color="#F2C94C", alpha=0.25)
    ax_psth.set_xlim(0, xlim)
    ax_psth.set_ylim(bottom=0)
    ax_psth.set_xlabel("time from stimulus onset (s)", fontsize=10)
    ax_psth.set_ylabel("rate (Hz)", fontsize=10)
    for s in ("top", "right"):
        ax_psth.spines[s].set_visible(False)

    _save_or_keep(fig, save_path)
    return fig
