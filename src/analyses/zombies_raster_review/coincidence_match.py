"""
coincidence_match.py — pair a unit from one spike source with the *same* neuron
in the other source, so the overlay raster compares like-for-like.

Why this exists
---------------
SpikeInterface (the automated sorter behind ``SISortedSpikeSource``) does not
tie a neuron to a single channel — one neuron is seen on several contacts, and
we force SI to name the neuron after its strongest channel. The manual/mixed
sort (``MixedManualSpikeSource``) may have assigned *that same neuron* to a
different channel. So the SI ``NeuronID`` ``...Channel.C_020_Unit 1`` and the
mixed ``Cell`` ``Channel.C_011_Unit 1`` can be one and the same neuron.

Matching the two sorts by channel name is therefore wrong. Instead we match by
**spike-time coincidence**: the physics from
``spikesorting.cross_channel_analysis`` — one action potential crosses threshold
on both records within a fraction of a millisecond, so two records of one neuron
have near-synchronous spike trains. We reuse that module's vetted
:func:`~spikesorting.cross_channel_analysis.spiketrain_metrics.coincidence`
(nothing in that package is modified here).

Because a single neuron can be split across channels *within* the mixed sort
too (two mixed units that together equal one SI unit), matched pairs are grouped
into connected components, so all copies of one neuron end up in a single
overlay.

Time base
---------
Both sources store per-trial ``SpikeTimes`` as absolute recording-clock seconds
(``spike_index / sample_rate``, see ``data_access/data_loader.py``), so the two
sorts of one ``(date, round)`` share one clock and their spike trains are
directly comparable. We concatenate a unit's trials into one train and convert
seconds → integer sample indices at a nominal rate purely to feed the
integer-indexed ``coincidence`` routine; the rate cancels out of the millisecond
window, so its exact value does not matter as long as it is used consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spikesorting.cross_channel_analysis.spiketrain_metrics import coincidence

# Nominal sample rate used only to discretise second-valued spike times into the
# integer indices ``coincidence`` expects. Intan default; see module docstring —
# it cancels out of the ms window, so the precise value is immaterial.
NOMINAL_SAMPLE_RATE = 30_000.0

# Cross-sort defaults. A little more permissive than cross_channel_analysis's
# 0.3, because a manual sort and an automated sort of the same neuron agree less
# than two channels of one manual sort do (different thresholds, drift, missed
# spikes). Still required to be well above chance.
DEFAULT_COINCIDENCE_THRESHOLD = 0.2
DEFAULT_RATIO_THRESHOLD = 5.0
DEFAULT_WINDOW_MS = 0.4


# --------------------------------------------------------------------------- #
# Spike-train assembly
# --------------------------------------------------------------------------- #
def unit_spike_train(unit_df: pd.DataFrame, *, spike_col: str = "SpikeTimes") -> np.ndarray:
    """Concatenate one unit's per-trial spike times into a single sorted train.

    Uses the *raw* absolute times (no per-trial re-zeroing) so the train stays on
    the recording clock and is comparable across sorts. Returns seconds.
    """
    trains = []
    for spikes in unit_df[spike_col]:
        arr = np.asarray(list(spikes), dtype=float) if spikes is not None else np.empty(0)
        if arr.size:
            trains.append(arr)
    if not trains:
        return np.empty(0, dtype=float)
    return np.sort(np.concatenate(trains))


def _to_samples(times_s: np.ndarray, *, sample_rate: float = NOMINAL_SAMPLE_RATE) -> np.ndarray:
    """Seconds → sorted int64 sample indices for ``coincidence``."""
    if times_s.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.asarray(times_s, dtype=float) * sample_rate).astype(np.int64)


def session_units(session_df: pd.DataFrame, id_col: str) -> Dict[str, pd.DataFrame]:
    """Split a loaded session DataFrame into ``{unit_id: per-trial rows}``.

    ``id_col`` is ``"Channel"`` for the mixed source (keys become the
    ``str(Channel)``, e.g. ``"Channel.C_011_Unit 1"``) or ``"NeuronID"`` for SI.
    """
    out: Dict[str, pd.DataFrame] = {}
    ids = session_df[id_col].astype(str)
    for uid, udf in session_df.groupby(ids):
        out[str(uid)] = udf
    return out


# --------------------------------------------------------------------------- #
# Cross-source matching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrossSortMatch:
    """One mixed unit and one SI unit that look like the same neuron."""
    mixed_id: str
    si_id: str
    n_mixed: int
    n_si: int
    coincidence: float
    ratio_over_chance: float


def match_units_across_sources(
    units_mixed: Dict[str, pd.DataFrame],
    units_si: Dict[str, pd.DataFrame],
    *,
    spike_col: str = "SpikeTimes",
    window_ms: float = DEFAULT_WINDOW_MS,
    coincidence_threshold: float = DEFAULT_COINCIDENCE_THRESHOLD,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    sample_rate: float = NOMINAL_SAMPLE_RATE,
) -> List[CrossSortMatch]:
    """Every mixed×SI unit pair whose spike trains are coincident above chance.

    Returns matches sorted by coincidence, most duplicate-like first. Compares
    *across* sources only (finding duplicates *within* a source is the job of
    ``cross_channel_analysis``).
    """
    trains_mixed = {uid: _to_samples(unit_spike_train(df, spike_col=spike_col), sample_rate=sample_rate)
                    for uid, df in units_mixed.items()}
    trains_si = {uid: _to_samples(unit_spike_train(df, spike_col=spike_col), sample_rate=sample_rate)
                 for uid, df in units_si.items()}

    matches: List[CrossSortMatch] = []
    for mid, tm in trains_mixed.items():
        if tm.size == 0:
            continue
        for sid, ts in trains_si.items():
            if ts.size == 0:
                continue
            cr = coincidence(tm, ts, sample_rate=sample_rate, window_ms=window_ms)
            if cr.coincidence >= coincidence_threshold and cr.ratio_over_chance >= ratio_threshold:
                matches.append(CrossSortMatch(
                    mixed_id=mid, si_id=sid,
                    n_mixed=cr.n_a, n_si=cr.n_b,
                    coincidence=cr.coincidence, ratio_over_chance=cr.ratio_over_chance,
                ))
    matches.sort(key=lambda m: m.coincidence, reverse=True)
    return matches


# --------------------------------------------------------------------------- #
# Grouping matched pairs into overlay groups
# --------------------------------------------------------------------------- #
# A node is a ("mixed"|"si", unit_id) tuple.
Node = Tuple[str, str]


@dataclass
class MatchGroup:
    """A connected set of units (across both sorts) that are mutually coincident.

    ``mixed_ids`` / ``si_ids`` list the members from each source.
    ``best_coincidence`` is the strongest pairwise coincidence inside the group.
    ``pairs`` keeps every matched ``(mixed_id, si_id, coincidence)`` so callers
    can report each pair, not just the best — one number hides weak links in a
    group with 3+ units.
    """
    mixed_ids: List[str]
    si_ids: List[str]
    best_coincidence: float
    pairs: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def n_units(self) -> int:
        return len(self.mixed_ids) + len(self.si_ids)


def group_matches(matches: List[CrossSortMatch]) -> List[MatchGroup]:
    """Collapse pairwise matches into connected components (union-find).

    Handles the many-to-one cases the manual sort creates — e.g. one SI unit that
    equals two mixed units on a channel puts all three in one group, hence one
    overlay.
    """
    parent: Dict[Node, Node] = {}

    def find(x: Node) -> Node:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: Node, b: Node):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in matches:
        union(("mixed", m.mixed_id), ("si", m.si_id))

    # bucket nodes by component root
    members: Dict[Node, List[Node]] = {}
    for node in list(parent.keys()):
        members.setdefault(find(node), []).append(node)
    # collect the pairs (and the strongest coincidence) per component
    best_in_root: Dict[Node, float] = {}
    pairs_in_root: Dict[Node, List[Tuple[str, str, float]]] = {}
    for m in matches:
        root = find(("mixed", m.mixed_id))
        best_in_root[root] = max(best_in_root.get(root, 0.0), m.coincidence)
        pairs_in_root.setdefault(root, []).append((m.mixed_id, m.si_id, m.coincidence))

    groups: List[MatchGroup] = []
    for root, nodes in members.items():
        mixed_ids = sorted(uid for src, uid in nodes if src == "mixed")
        si_ids = sorted(uid for src, uid in nodes if src == "si")
        pairs = sorted(pairs_in_root.get(root, []), key=lambda p: p[2], reverse=True)
        groups.append(MatchGroup(mixed_ids, si_ids, best_in_root.get(root, 0.0), pairs))
    groups.sort(key=lambda g: g.best_coincidence, reverse=True)
    return groups
