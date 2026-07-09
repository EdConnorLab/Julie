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

# palette for overlay mode (manual vs SI-sorted, etc.). A 10-colour qualitative
# set (Tableau 10) so groups with many units stay distinguishable in both the
# raster lanes and the probe map; colours still cycle beyond 10.
_OVERLAY_COLORS = [
    "#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#F28E2B",
    "#76B7B2", "#EDC948", "#AF7AA1", "#FF9DA7", "#9C755F",
]


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
def _window_label_ms(window_s: Tuple[float, float]) -> str:
    """``(0.3, 0.4)`` s → ``"300–400 ms"`` for on-plot annotation."""
    return f"{window_s[0] * 1000:.0f}–{window_s[1] * 1000:.0f} ms"


def _annotate_window_ms(ax, window_s: Tuple[float, float]):
    """Label the window's extent in ms at the top of the shaded band on ``ax``.

    Drawn just inside the top of the panel (x in data coords, y in axes
    fraction, so the y-axis inversion doesn't matter) with a light background so
    it stays readable over spikes and never collides with the title.
    """
    xmid = (window_s[0] + window_s[1]) / 2.0
    ax.text(xmid, 0.99, _window_label_ms(window_s),
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, fontweight="bold", color="#8A6D00", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E0C060", alpha=0.85))


# Which list a window came from → its highlight colour (amber = mixed/xlsx,
# violet = SI/csv). A ``None`` source (a single legacy window) stays amber.
_WINDOW_SRC_COLORS = {"mixed": "#F2C94C", "SI": "#B39DDB", None: "#F2C94C"}


def _shade_windows(ax, windows, *, alpha: float = 0.22):
    """Shade every ``(lo_s, hi_s, source, label)`` window band on ``ax``."""
    for lo, hi, src, _ in windows:
        ax.axvspan(lo, hi, color=_WINDOW_SRC_COLORS.get(src, "#F2C94C"),
                   alpha=alpha, zorder=1)


def _annotate_windows(ax, windows):
    """Callout each window at the top of ``ax``, coloured by its source list.

    Callouts are staggered by left-edge order so several windows (e.g. one from
    the mixed list and one from the SI list, or a cell with multiple windows) do
    not stack on the same line. Each says which list it came from and its ms
    extent; the outline colour also encodes the source.
    """
    y_levels = [0.985, 0.90, 0.815, 0.73]
    for rank, idx in enumerate(sorted(range(len(windows)), key=lambda k: windows[k][0])):
        lo, hi, src, label = windows[idx]
        color = _WINDOW_SRC_COLORS.get(src, "#F2C94C")
        head = f"{src} {label}\n" if src else ""
        ax.text((lo + hi) / 2.0, y_levels[rank % len(y_levels)],
                f"{head}{lo * 1000:.0f}–{hi * 1000:.0f} ms",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color="0.1", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.1, alpha=0.92))


def plot_zombies_raster(
    neuron_df,
    *,
    neuron_label: str,
    spike_col: str = "SpikeTimes",
    window_s: Optional[Tuple[float, float]] = None,
    p_value: Optional[float] = None,
    xlim: float = 2.4,
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

    fig = plt.figure(figsize=(8.5, 9))
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_psth = fig.add_subplot(gs[1], sharex=ax)

    # --- raster ---
    y = 0
    all_trials: List[np.ndarray] = []
    tick_pos, tick_labels = [], []
    for i, m in enumerate(by_monkey):
        n = len(m.trials)
        # alternating background band spanning this monkey's trials
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + n - 0.5, color="0.96", zorder=0)
        # spikes (black)
        ax.eventplot(m.trials, lineoffsets=np.arange(y, y + n),
                     colors="black", linewidths=0.9, linelengths=0.9)
        center = y + n / 2 - 0.5
        tick_pos.append(center)
        tick_labels.append(str(m.monkey))
        # rank number on the right edge (axes-fraction x, data-coord y)
        rank_txt = f"#{m.rank}" if m.rank else "unranked"
        ax.text(1.01, center, rank_txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color="0.4", clip_on=False)
        all_trials.extend(m.trials)
        y += n

    # stimulus onset + significant window
    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    if window_s is not None:
        ax.axvspan(window_s[0], window_s[1], color="#F2C94C", alpha=0.30, zorder=1,
                   label="ANOVA sig. window")
        _annotate_window_ms(ax, window_s)
    ax.set_ylim(-0.5, total_trials - 0.5)
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()  # most dominant monkey (rank 1) at the top
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    sub = f"{total_trials} trials · {len(by_monkey)} monkeys"
    if window_s is not None:
        sub += f" · window {_window_label_ms(window_s)}"
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
# Probe map — where each overlaid unit sits on the 32-channel linear probe
# --------------------------------------------------------------------------- #
def _unit_channel_token(df) -> Optional[str]:
    """Best-effort ``C_020``-style channel token for a unit's per-trial frame."""
    for col in ("Channel", "NeuronID"):
        if col in df.columns and len(df) > 0:
            raw = str(df[col].iloc[0])
            tok = raw.split("Channel.")[-1].split("_Unit")[0].strip()
            if tok:
                return tok
    return None


def _draw_probe_map(ax, unit_dfs, colors, pair_coincidences=None):
    """Draw the 32-channel linear probe, marking each overlaid unit's contact.

    Marker colour matches the raster lane; units sharing a contact (e.g. manual
    ``C_025`` and SI ``C_025_Unit 1``) are offset horizontally so they don't
    overlap. Vertical position is physical depth (contacts are ``Y_PITCH_UM``
    apart), so units clustered on nearby contacts are plausibly one neuron, while
    high-coincidence units far apart are distinct neurons firing synchronously —
    which is exactly what this panel is here to reveal.

    ``pair_coincidences`` is a list of ``(label_a, label_b, coincidence)``: each
    matched cross-sort pair is drawn as a line between the two units' markers,
    labelled with its coincidence — so you read each pair's value *and* how far
    apart the two units are in one glance.
    """
    from collections import defaultdict
    from spikesorting.cross_channel_analysis import probe_geometry as geom

    n = len(geom.PROBE_CHANNEL_ORDER)
    pitch = geom.Y_PITCH_UM

    # backbone shank: a thin line down the probe with a short tick per contact
    ax.plot([0, 0], [0, (n - 1) * pitch], color="0.75", lw=1.2, zorder=1)
    for c in range(n):
        ax.plot([-0.12, 0.12], [c * pitch, c * pitch], color="0.82", lw=0.8, zorder=1)

    per_contact = defaultdict(list)   # contact index -> [(label, token)]
    off_probe = []
    for label, df in unit_dfs.items():
        tok = _unit_channel_token(df)
        ci = geom.contact_index_of(tok) if tok else None
        (off_probe if ci is None else per_contact[ci]).append((label, tok))

    marked_depths = []
    pos = {}          # lane label -> (x, depth) of its marker, for pair lines
    x0, x_step = 0.22, 0.28
    for ci, units in sorted(per_contact.items()):
        depth = ci * pitch
        marked_depths.append(depth)
        for j, (label, tok) in enumerate(units):
            x = x0 + x_step * j
            pos[label] = (x, depth)
            ax.plot([0, x], [depth, depth], color="0.6", lw=0.7, zorder=2)  # connector
            ax.scatter([x], [depth], s=58, color=colors.get(label, "0.3"),
                       edgecolor="black", linewidth=0.4, zorder=3)
        # channel id once, just past the last marker on this contact
        ax.text(x0 + x_step * (len(units) - 1) + 0.22, depth, units[0][1],
                va="center", ha="left", fontsize=7, color="0.25", clip_on=False)

    # per-pair coincidence: a line between the two units' markers, labelled
    for la, lb, coinc in (pair_coincidences or []):
        if la in pos and lb in pos:
            (xa, da), (xb, db) = pos[la], pos[lb]
            ax.plot([xa, xb], [da, db], color="0.45", lw=0.9, zorder=2)
            ax.text((xa + xb) / 2 + 0.06, (da + db) / 2, f"{coinc:.2f}",
                    fontsize=6.5, ha="left", va="center", color="0.1", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.55",
                              lw=0.3, alpha=0.92))

    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(-pitch, (n - 1) * pitch + pitch)
    ax.invert_yaxis()  # contact 0 at the top, deeper contacts downward
    ax.set_ylabel("depth on probe (µm)", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    # headline: how far apart the marked units are (span in µm) — goes in the
    # title so it never overlaps the markers
    if len(marked_depths) >= 2:
        span = max(marked_depths) - min(marked_depths)
        ax.set_title(f"probe (32 ch)\nunit span {span:.0f} µm", fontsize=9)
    else:
        ax.set_title("probe (32 ch)", fontsize=9)
    if off_probe:
        ax.text(0.5, -0.02, f"{len(off_probe)} off-probe", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="0.6")


# How many contacts on either side of a unit's peak to draw its footprint over.
# A unit's spike only bleeds onto a handful of nearby contacts, so drawing the
# whole 32-contact column is mostly flat clutter; ±this keeps it legible.
FOOTPRINT_CONTACT_RADIUS = 5


def _draw_footprints(ax, footprints_by_label, colors, n_contacts=FOOTPRINT_CONTACT_RADIUS):
    """Draw each unit's normalised cross-channel waveform along the probe.

    Shares the probe's depth axis: a unit's average waveform is drawn as a small
    trace at each contact's depth, so its spatial footprint reads as a depth
    profile (biggest at the peak contact, decaying away). Only contacts within
    ``n_contacts`` of that unit's own peak are drawn — a spike bleeds onto just a
    few neighbours, so the rest would be flat clutter. Each unit is normalised to
    its own peak (so shapes compare regardless of amplitude) and coloured to its
    raster lane. Two units that are one neuron peak at the same contact with the
    same shape; synchronous-but-distinct neurons peak at different contacts.
    """
    from spikesorting.cross_channel_analysis import probe_geometry as geom

    n = len(geom.PROBE_CHANNEL_ORDER)
    pitch = geom.Y_PITCH_UM
    amp = 0.6 * pitch  # a full-scale (normalised = 1) deflection spans ~0.6 contacts

    drew = False
    for label, fp in (footprints_by_label or {}).items():
        if fp is None:
            continue
        w = np.asarray(fp.waveforms, dtype=float)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        if peak <= 0:
            continue
        w = w / peak
        color = colors.get(label, "0.3")
        xs = np.linspace(0.1, 0.9, w.shape[1])
        contacts = [geom.contact_index_of(c) for c in fp.channels]
        # this unit's peak contact (largest peak-to-peak among mapped contacts)
        p2p = w.max(axis=1) - w.min(axis=1)
        mapped = [r for r, ci in enumerate(contacts) if ci is not None]
        if not mapped:
            continue
        peak_ci = contacts[max(mapped, key=lambda r: p2p[r])]
        for row, ci in enumerate(contacts):
            if ci is None or abs(ci - peak_ci) > n_contacts:
                continue
            depth = ci * pitch
            ax.plot(xs, depth - w[row] * amp, color=color, lw=0.7, alpha=0.85, zorder=3)
            drew = True

    ax.set_xlim(0, 1)
    ax.set_ylim(-pitch, (n - 1) * pitch + pitch)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"footprint\n(norm., ±{n_contacts} ch)", fontsize=9)
    if not drew:
        ax.text(0.5, 0.5, "waveforms\nunavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.55")


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
    windows: Optional[list] = None,
    pair_coincidences: Optional[list] = None,
    footprints: Optional[dict] = None,
    xlim: float = 2.4,
    psth_bin_ms: float = 50.0,
    show_probe: bool = True,
    save_path: Optional[str] = None,
):
    """Overlay several sorts of the *same channel* on one raster for comparison.

    ``unit_dfs`` maps a label (e.g. ``"manual"``, ``"SI"``) to that unit's
    per-trial DataFrame. Trials are grouped by stimulus monkey (rank order,
    subject excluded) and, within each monkey, matched across sorts by
    ``key_col`` (falling back to trial order). Each trial row is split into one
    thin lane per sort, coloured consistently, so you can see spike-for-spike
    where the sorts agree or differ. The PSTH panel overlays each sort's rate.

    ``windows`` is a list of ``(lo_s, hi_s, source, label)`` tuples — every
    ANOVA-significant window in the group, from either list, each shaded and
    called out with which list (``"mixed"``/``"SI"``) and cell it came from. For
    a single window with no source, pass the legacy ``window_s`` instead.

    ``pair_coincidences`` is ``[(label_a, label_b, coincidence), …]`` drawn as
    labelled links on the probe map. ``footprints`` maps a lane label to its
    :class:`~spikesorting.cross_channel_analysis.waveforms.Footprint`; when given
    (and ``show_probe``), a waveform-footprint panel is added on the right.

    Returns the ``Figure`` (or ``None`` if there are no Zombies trials).
    """
    labels = list(unit_dfs.keys())
    colors = {lab: _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)] for i, lab in enumerate(labels)}

    # windows to shade: an explicit multi-source list wins; else fall back to the
    # single legacy ``window_s`` (source-less). Each entry is (lo, hi, src, label).
    if windows is None:
        windows = [(window_s[0], window_s[1], None, None)] if window_s is not None else []

    # union of monkeys present across all sorts, ranked, subject excluded
    present = set()
    for df in unit_dfs.values():
        z = df[df["MonkeyGroup"] == GROUP]
        present.update(z["MonkeyName"].dropna().unique())
    ordered, rank_by_monkey = zombies_monkey_order(present)

    ax_probe = ax_wave = None
    if show_probe and footprints is not None:
        fig = plt.figure(figsize=(12.5, 9))
        gs = GridSpec(2, 3, width_ratios=[5.2, 1.05, 1.5], height_ratios=[3.2, 1.0],
                      hspace=0.08, wspace=0.10, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_psth = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_probe = fig.add_subplot(gs[:, 1])
        ax_wave = fig.add_subplot(gs[:, 2])
    elif show_probe:
        fig = plt.figure(figsize=(11, 9))
        gs = GridSpec(2, 2, width_ratios=[5.5, 1.15], height_ratios=[3.2, 1.0],
                      hspace=0.08, wspace=0.14, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_psth = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_probe = fig.add_subplot(gs[:, 1])
    else:
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
    _shade_windows(ax, windows, alpha=0.25)
    _annotate_windows(ax, windows)
    ax.set_ylim(-0.5, total_rows - 0.5)
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()
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
    _shade_windows(ax_psth, windows, alpha=0.25)
    ax_psth.set_xlim(0, xlim)
    ax_psth.set_ylim(bottom=0)
    ax_psth.set_xlabel("time from stimulus onset (s)", fontsize=10)
    ax_psth.set_ylabel("rate (Hz)", fontsize=10)
    for s in ("top", "right"):
        ax_psth.spines[s].set_visible(False)

    # --- probe map: where these units sit on the linear probe ---
    if ax_probe is not None:
        _draw_probe_map(ax_probe, unit_dfs, colors, pair_coincidences)
    # --- footprint: each unit's normalised waveform along the probe ---
    if ax_wave is not None:
        _draw_footprints(ax_wave, footprints, colors)

    _save_or_keep(fig, save_path)
    return fig
