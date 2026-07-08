"""
Tests for the exploded-cache (sorted + unsorted) coincidence pipeline.

Self-contained: builds a synthetic exploded-format DataFrame with a planted
cross-channel duplicate — no real cache or DB required. Run with::

    pytest src/spikesorting/cross_channel_analysis/tests/test_exploded.py
    python  src/spikesorting/cross_channel_analysis/tests/test_exploded.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spikesorting.cross_channel_analysis.exploded_loader import (
    split_channel, session_from_exploded_df, find_sessions, pair_type,
    is_sorted_unit, TICK_RATE_HZ,
)
from spikesorting.cross_channel_analysis.exploded_analysis import analyze_exploded_session


def _synthetic_exploded_df(seed=0):
    """One epoch; C-021 & C-010 (adjacent) share a neuron; C-026 independent;
    C-020 is near-silent (should be dropped by min_spikes)."""
    rng = np.random.default_rng(seed)
    neuron = np.sort(rng.uniform(0, 100, size=2000))          # shared neuron, seconds
    c021 = neuron + rng.normal(0, 2 / TICK_RATE_HZ, size=neuron.size)   # ~2-tick jitter
    keep = rng.random(neuron.size) < 0.85
    c010 = neuron[keep] + rng.normal(0, 2 / TICK_RATE_HZ, size=keep.sum())
    c026 = np.sort(rng.uniform(0, 100, size=1500))            # independent
    c020 = np.sort(rng.uniform(0, 100, size=3))               # near-silent

    rows = [
        {"Channel": "Channel.C_021_Unit 1", "SpikeTimes": list(np.sort(c021)), "EpochStartStop": (0.0, 100.0)},
        {"Channel": "Channel.C_010", "SpikeTimes": list(np.sort(c010)), "EpochStartStop": (0.0, 100.0)},
        {"Channel": "Channel.C_026", "SpikeTimes": list(c026), "EpochStartStop": (0.0, 100.0)},
        {"Channel": "Channel.C_020", "SpikeTimes": list(c020), "EpochStartStop": (0.0, 100.0)},
    ]
    return pd.DataFrame(rows)


def test_split_channel():
    assert split_channel("Channel.C_010_Unit 1") == ("C-010", "Unit 1", True)
    assert split_channel("Channel.C_006") == ("C-006", "unsorted", False)


def test_session_build_ticks_and_naming():
    df = _synthetic_exploded_df()
    session = session_from_exploded_df(df, min_spikes=50)
    # near-silent C-020 (3 spikes) dropped; 3 units remain
    uids = session.uids
    assert "C-020 / unsorted" not in uids
    assert "C-021 / Unit 1" in uids and "C-010 / unsorted" in uids
    # spikes stored as integer ticks
    u = session.unit("C-021 / Unit 1")
    assert np.issubdtype(u.spike_indices.dtype, np.integer)
    assert session.sample_rate == TICK_RATE_HZ


def test_pair_type_classification():
    df = _synthetic_exploded_df()
    session = session_from_exploded_df(df, min_spikes=50)
    assert pair_type(session, "C-021 / Unit 1", "C-010 / unsorted") == "sorted-unsorted"
    assert pair_type(session, "C-010 / unsorted", "C-026 / unsorted") == "unsorted-unsorted"


def test_coincidence_not_truncated_and_duplicate_detected():
    # regression guard: feeding second-timestamps must NOT collapse to 1-second
    # bins (the old int64-cast bug). The planted adjacent duplicate should be the
    # top pair; the independent channel should not be a candidate.
    df = _synthetic_exploded_df()
    with tempfile.TemporaryDirectory() as tmp:
        # save so analyze_exploded_session can read it like a real cache file
        pkl = os.path.join(tmp, "2023-01-01_round_1.pkl")
        df.to_pickle(pkl)
        cand = analyze_exploded_session(pkl, os.path.join(tmp, "out"), min_spikes=50)
        pairs = {frozenset((r["uid_a"], r["uid_b"])): r for r in cand}
        dup = frozenset(("C-021 / Unit 1", "C-010 / unsorted"))
        assert dup in pairs, f"planted duplicate missing; candidates={list(pairs)}"
        assert pairs[dup]["coincidence"] > 0.6
        # independent channel must not pair with the duplicate
        for r in cand:
            assert "C-026 / unsorted" not in (r["uid_a"], r["uid_b"])
        for name in ["all_pairs.csv", "candidate_duplicates.csv",
                     "coincidence_matrix.png", "distance_vs_coincidence.png"]:
            assert os.path.exists(os.path.join(tmp, "out", name))


def test_find_sessions_parses_names():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["2023-09-26_round_1.pkl", "2023-10-03_round_4.pkl", "junk.pkl"]:
            open(os.path.join(tmp, name), "w").close()
        found = find_sessions(tmp)
        assert ("2023-09-26", 1) == (found[0][1], found[0][2])
        assert len(found) == 2  # junk.pkl ignored


if __name__ == "__main__":
    test_split_channel()
    test_session_build_ticks_and_naming()
    test_pair_type_classification()
    test_coincidence_not_truncated_and_duplicate_detected()
    test_find_sessions_parses_names()
    print("All exploded-path tests passed.")
