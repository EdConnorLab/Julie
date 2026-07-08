"""
cross_channel_analysis — detect manually-sorted units that are the *same*
neuron recorded on different (usually adjacent) channels.

See ``README.md`` for the scientific rationale and usage. Quick start::

    from spikesorting.cross_channel_analysis.loader import load_sorted_spikes
    from spikesorting.cross_channel_analysis import spiketrain_metrics as stm

    session = load_sorted_spikes("/path/to/session/sorted_spikes.pkl")
    pairs = stm.all_pairs(session)
    duplicates = stm.candidate_duplicates(pairs)
"""
from . import probe_geometry, loader, spiketrain_metrics, waveforms  # noqa: F401

__all__ = ["probe_geometry", "loader", "spiketrain_metrics", "waveforms"]
