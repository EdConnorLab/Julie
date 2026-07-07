"""
loader.py — read ``sorted_spikes.pkl`` into tidy, analysis-ready structures.

``sorted_spikes.pkl`` (written by ``SortedSpikeExporter.save_sorted_spikes`` in
``spikesorting/datahandler.py``) is a nested dict::

    {
        Channel.C_021: {"Unit 1": np.ndarray([spike_idx, ...]),
                        "Unit 2": np.ndarray([...])},
        Channel.C_010: {"Unit 1": np.ndarray([...])},
        ...
    }

Keys are ``clat.intan.channels.Channel`` enum members; each value maps a
per-channel unit name to an array of spike **sample indices** into
``amplifier.dat`` / ``preprocessed_data.dat``. The sample rate lives in the
session's ``info.rhd``.

This module flattens that into a flat list of :class:`SortedUnit` records with a
globally-unique id (``"C-021 / Unit 1"``), which is what every downstream
metric and plot consumes.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def channel_to_name(channel_key) -> str:
    """Normalise a pickle channel key to a plain name like ``'C-021'``.

    Works whether the key is a ``Channel`` enum (has ``.value``) or a string,
    so this module does not need ``clat`` importable to read a pickle.
    """
    if hasattr(channel_key, "value"):
        return str(channel_key.value)
    return str(channel_key)


@dataclass
class SortedUnit:
    """One manually-sorted unit on one channel."""
    channel: str                 # e.g. "C-021"
    unit_name: str               # e.g. "Unit 1"
    spike_indices: np.ndarray    # sorted int64 sample indices into the recording

    @property
    def uid(self) -> str:
        """Globally-unique unit id across the session."""
        return f"{self.channel} / {self.unit_name}"

    @property
    def n_spikes(self) -> int:
        return int(self.spike_indices.size)

    def firing_rate_hz(self, sample_rate: float, n_samples: Optional[int] = None) -> float:
        """Mean firing rate. Duration is inferred from the last spike unless
        ``n_samples`` (the recording length) is given."""
        if self.n_spikes == 0:
            return 0.0
        span = n_samples if n_samples is not None else (int(self.spike_indices[-1]) + 1)
        duration_s = span / sample_rate
        return self.n_spikes / duration_s if duration_s > 0 else 0.0


@dataclass
class Session:
    """All sorted units for one recording session, plus its sample rate."""
    units: List[SortedUnit]
    sample_rate: float
    session_dir: Optional[str] = None
    label: Optional[str] = None
    _by_uid: Dict[str, SortedUnit] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._by_uid = {u.uid: u for u in self.units}

    def unit(self, uid: str) -> SortedUnit:
        return self._by_uid[uid]

    @property
    def uids(self) -> List[str]:
        return [u.uid for u in self.units]

    def channels(self) -> List[str]:
        seen = []
        for u in self.units:
            if u.channel not in seen:
                seen.append(u.channel)
        return seen


def read_sample_rate(session_dir: str) -> float:
    """Read the amplifier sample rate from the session's ``info.rhd``."""
    from clat.intan.rhd import load_intan_rhd_format  # local import: heavy dep
    rhd = load_intan_rhd_format.read_data(os.path.join(session_dir, "info.rhd"))
    return float(rhd["frequency_parameters"]["amplifier_sample_rate"])


def load_sorted_spikes(
    pkl_path: str,
    *,
    sample_rate: Optional[float] = None,
    min_spikes: int = 1,
) -> Session:
    """Load a ``sorted_spikes.pkl`` into a :class:`Session`.

    Parameters
    ----------
    pkl_path
        Path to ``sorted_spikes.pkl`` (or a labelled variant).
    sample_rate
        Amplifier sample rate in Hz. If ``None``, it is read from ``info.rhd``
        sitting next to the pickle.
    min_spikes
        Drop units with fewer than this many spikes (empty/degenerate units).
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict in {pkl_path}, got {type(raw).__name__}")

    session_dir = os.path.dirname(os.path.abspath(pkl_path))
    if sample_rate is None:
        sample_rate = read_sample_rate(session_dir)

    units: List[SortedUnit] = []
    for channel_key, units_by_name in raw.items():
        channel = channel_to_name(channel_key)
        for unit_name, spike_indices in units_by_name.items():
            idx = np.asarray(spike_indices, dtype=np.int64).ravel()
            idx.sort(kind="stable")
            if idx.size < min_spikes:
                continue
            units.append(SortedUnit(channel=channel, unit_name=str(unit_name), spike_indices=idx))

    label = os.path.splitext(os.path.basename(pkl_path))[0]
    return Session(units=units, sample_rate=sample_rate, session_dir=session_dir, label=label)
