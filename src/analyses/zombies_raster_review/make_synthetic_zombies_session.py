"""
make_synthetic_zombies_session.py — fabricate a per-trial unit DataFrame.

The real spike-source caches live next to the Intan recordings and need DB
access, so they aren't in this repo. This builds a DataFrame in the exact shape
a SpikeSource returns (columns ``MonkeyGroup``, ``MonkeyName``,
``EpochStartStop``, ``SpikeTimes``, ``NeuronID``, ``Channel``, ``Date``,
``Round No.``) for one synthetic unit shown to the Zombies group — so the raster
and PSTH can be exercised and eyeballed.

The unit is given a rank-graded response inside a chosen window (more dominant
monkeys drive a stronger response) so the raster visibly "does something".
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from analyses.enums.monkey_names import get_monkeys_by_rank

GROUP = "Zombies"


def _trial_spikes(rng, epoch, baseline_hz, resp_hz, window_s):
    """One trial: baseline Poisson + an elevated burst inside the window."""
    start, stop = epoch
    dur = stop - start
    spikes = []
    # baseline over whole epoch
    n_base = rng.poisson(baseline_hz * dur)
    spikes.extend(rng.uniform(start, stop, size=n_base))
    # response burst inside window
    if window_s is not None and resp_hz > 0:
        w0, w1 = start + window_s[0], start + window_s[1]
        n_resp = rng.poisson(resp_hz * (w1 - w0))
        spikes.extend(rng.uniform(w0, w1, size=n_resp))
    return np.sort(np.asarray(spikes, dtype=float))


def make_synthetic_zombies_unit(
    *,
    neuron_id: str = "AMG_2023-09-26_2_Channel.C_018_Unit 1",
    channel: str = "Channel.C_018_Unit 1",
    date: str = "2023-09-26",
    round_no: int = 2,
    window_s: Optional[Tuple[float, float]] = (0.1, 0.45),
    trials_per_monkey: int = 10,
    epoch_len: float = 2.5,
    baseline_hz: float = 4.0,
    peak_resp_hz: float = 40.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a per-trial DataFrame for one synthetic Zombies-driven unit."""
    rng = np.random.default_rng(seed)
    ranked = get_monkeys_by_rank(GROUP)
    epoch = (0.0, epoch_len)

    rows = []
    for rank_idx, monkey in enumerate(ranked):
        # dominant monkeys (low rank_idx) drive a stronger response
        resp_hz = peak_resp_hz * (1.0 - rank_idx / len(ranked))
        n_trials = trials_per_monkey + int(rng.integers(-2, 3))  # jitter trial count
        for _ in range(max(3, n_trials)):
            spikes = _trial_spikes(rng, epoch, baseline_hz, resp_hz, window_s)
            rows.append({
                "TaskField": None,
                "MonkeyGroup": GROUP,
                "MonkeyName": monkey,
                "EpochStartStop": epoch,
                "SpikeTimes": spikes,
                "NeuronID": neuron_id,
                "Channel": channel,
                "Date": date,
                "Round No.": round_no,
            })
    return pd.DataFrame(rows)
