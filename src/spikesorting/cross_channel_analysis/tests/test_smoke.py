"""
End-to-end smoke test on a fabricated session with a *known* planted duplicate.

Run with either::

    pytest src/spikesorting/cross_channel_analysis/tests/test_smoke.py
    python  src/spikesorting/cross_channel_analysis/tests/test_smoke.py   # no pytest

The synthetic session (see ``make_synthetic_session.py``) plants neuron N1 on
adjacent channels C-021 and C-010, so ``C-021 / Unit 1`` and ``C-010 / Unit 1``
must come out as a high-coincidence candidate, while the independent unit on
C-026 must not.
"""
import os
import sys
import tempfile

import numpy as np

# allow running as a plain script (add repo 'src' to path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spikesorting.cross_channel_analysis.loader import Session, SortedUnit
from spikesorting.cross_channel_analysis import spiketrain_metrics as stm
from spikesorting.cross_channel_analysis import probe_geometry as geom
from spikesorting.cross_channel_analysis.make_synthetic_session import make_synthetic_session
from spikesorting.cross_channel_analysis.run_analysis import analyze_session


def _build_session():
    sorted_spikes, voltages, sr = make_synthetic_session(with_voltages=True, seed=0)
    units = []
    for ch, by_unit in sorted_spikes.items():
        for name, idx in by_unit.items():
            units.append(SortedUnit(ch, name, np.sort(np.asarray(idx, dtype=np.int64))))
    return Session(units=units, sample_rate=sr, label="synthetic"), voltages


def test_geometry_adjacency():
    # C-021 and C-010 are neighbouring contacts on the probe map.
    assert geom.contacts_apart("C-021", "C-010") == 1
    assert geom.channel_distance_um("C-021", "C-010") == geom.Y_PITCH_UM


def test_planted_duplicate_is_detected():
    session, _ = _build_session()
    pairs = stm.all_pairs(session)
    candidates = stm.candidate_duplicates(pairs)
    keys = {frozenset((p.uid_a, p.uid_b)) for p in candidates}

    dup = frozenset(("C-021 / Unit 1", "C-010 / Unit 1"))
    assert dup in keys, f"planted duplicate not detected; candidates={keys}"

    # the independent unit must NOT be flagged against the duplicate
    for p in candidates:
        assert "C-026 / Unit 1" not in (p.uid_a, p.uid_b), \
            "independent unit wrongly flagged as duplicate"


def test_coincidence_ordering():
    session, _ = _build_session()
    pairs = stm.all_pairs(session)
    top = pairs[0]
    assert frozenset((top.uid_a, top.uid_b)) == frozenset(
        ("C-021 / Unit 1", "C-010 / Unit 1"))
    assert top.coincidence > 0.5
    assert top.ratio_over_chance > 5


def test_footprint_similarity_high_for_duplicate():
    from spikesorting.cross_channel_analysis import waveforms as wf
    session, voltages = _build_session()
    a = session.unit("C-021 / Unit 1")
    b = session.unit("C-010 / Unit 1")
    c = session.unit("C-026 / Unit 1")
    fa = wf.compute_footprint(a.uid, a.spike_indices, voltages)
    fb = wf.compute_footprint(b.uid, b.spike_indices, voltages)
    fc = wf.compute_footprint(c.uid, c.spike_indices, voltages)
    sim_dup = wf.footprint_similarity(fa, fb)
    sim_diff = wf.footprint_similarity(fa, fc)
    assert sim_dup > sim_diff, (sim_dup, sim_diff)
    assert sim_dup > 0.8


def test_full_pipeline_writes_outputs():
    session, voltages = _build_session()
    with tempfile.TemporaryDirectory() as tmp:
        pairs, candidates = analyze_session(session, tmp, voltages_by_channel=voltages)
        assert len(candidates) >= 1
        for name in ["overview_coincidence_matrix.png",
                     "overview_distance_vs_coincidence.png",
                     "candidate_duplicates.csv", "summary.txt"]:
            assert os.path.exists(os.path.join(tmp, name)), f"missing {name}"
        # at least one per-pair report figure
        assert any(f.startswith("pair_") for f in os.listdir(tmp))


if __name__ == "__main__":
    test_geometry_adjacency()
    test_planted_duplicate_is_detected()
    test_coincidence_ordering()
    test_footprint_similarity_high_for_duplicate()
    test_full_pipeline_writes_outputs()
    print("All smoke tests passed.")
