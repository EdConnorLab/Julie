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


SUBJECT_MONKEY_ID = "81G"


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
    include_subject: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a per-trial DataFrame for one synthetic Zombies-driven unit.

    If ``include_subject`` is True, a few 81G (subject) trials are added so the
    subject-exclusion logic can be tested; the raster must never show them.
    """
    rng = np.random.default_rng(seed)
    monkeys = list(get_monkeys_by_rank(GROUP))
    if include_subject:
        monkeys = monkeys + [SUBJECT_MONKEY_ID]
    epoch = (0.0, epoch_len)

    rows = []
    task_id = 0
    for rank_idx, monkey in enumerate(monkeys):
        # dominant monkeys (low rank_idx) drive a stronger response
        resp_hz = peak_resp_hz * (1.0 - rank_idx / len(monkeys))
        n_trials = trials_per_monkey + int(rng.integers(-2, 3))  # jitter trial count
        for _ in range(max(3, n_trials)):
            task_id += 1
            spikes = _trial_spikes(rng, epoch, baseline_hz, resp_hz, window_s)
            rows.append({
                "TaskField": task_id,
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


def make_synthetic_overlay_pair(
    *,
    window_s: Optional[Tuple[float, float]] = (0.1, 0.45),
    seed: int = 1,
) -> dict:
    """Two sorts of the same channel sharing trials — for overlay mode.

    Returns ``{"manual": df, "SI": df}``. The SI sort keeps ~80% of the manual
    unit's spikes and adds a little jitter/noise, so the overlay shows where two
    sorts of the same channel agree and differ (matched trial-for-trial by
    ``TaskField``).
    """
    manual = make_synthetic_zombies_unit(
        neuron_id="AMG_2023-09-26_2_Channel.C_018_Unit 1",
        channel="Channel.C_018_Unit 1", window_s=window_s, seed=seed)

    rng = np.random.default_rng(seed + 100)
    si_rows = []
    for _, r in manual.iterrows():
        sp = np.asarray(r["SpikeTimes"], dtype=float)
        keep = rng.random(sp.size) < 0.8              # SI misses ~20%
        sp = np.sort(sp[keep] + rng.normal(0, 0.001, size=keep.sum()))  # ~1 ms jitter
        row = r.to_dict()
        row["SpikeTimes"] = sp
        row["NeuronID"] = "AMG_2023-09-26_2_Channel.C_018_Unit 1_SI"
        si_rows.append(row)
    si = pd.DataFrame(si_rows)
    return {"manual": manual, "SI": si}


def _spread_over_recording(df: pd.DataFrame, *, spacing_s: float = 6.0) -> pd.DataFrame:
    """Place each trial at a distinct absolute time on a shared recording clock.

    ``make_synthetic_zombies_unit`` reuses one ``(0.0, epoch_len)`` epoch for
    every trial, which is fine for a per-trial raster (it re-zeros each trial)
    but wrong for a *concatenated* spike train: piling all trials into 2.5 s
    makes chance coincidence enormous. Offsetting trial ``i`` by ``i*spacing_s``
    spreads them over a realistic recording so coincidence-over-chance behaves
    like real data. Rendering is unaffected (it re-zeros by ``EpochStartStop``).
    """
    out = df.copy(deep=True)
    spike_col, offsets = [], []
    for i, (_, r) in enumerate(out.iterrows()):
        off = i * spacing_s
        offsets.append((r["EpochStartStop"][0] + off, r["EpochStartStop"][1] + off))
        spike_col.append(np.asarray(r["SpikeTimes"], dtype=float) + off)
    out["SpikeTimes"] = spike_col
    out["EpochStartStop"] = offsets
    return out


def _copy_unit_as(df: pd.DataFrame, *, channel: str, neuron_id: str,
                  keep_frac: float, jitter_s: float, seed: int) -> pd.DataFrame:
    """A near-duplicate of ``df``'s trains: keep a fraction of spikes + jitter.

    Simulates the same neuron seen by the *other* sorter on a *different*
    channel — high spike-time coincidence but a new channel/NeuronID.
    """
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)
    new_spikes = []
    for sp in out["SpikeTimes"]:
        sp = np.asarray(sp, dtype=float)
        keep = rng.random(sp.size) < keep_frac
        kept = np.sort(sp[keep] + rng.normal(0, jitter_s, size=int(keep.sum())))
        new_spikes.append(kept)
    out["SpikeTimes"] = new_spikes
    out["Channel"] = channel
    out["NeuronID"] = neuron_id
    return out


def make_synthetic_cross_source_session(
    *,
    date: str = "2023-09-26",
    round_no: int = 2,
    window_s: Optional[Tuple[float, float]] = (0.1, 0.45),
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fabricate a mixed-source and an SI-source session for one recording.

    A single neuron is deliberately placed on **different channels** in the two
    sorts — ``Channel.C_011_Unit 1`` (mixed) vs ``...Channel.C_020_Unit 1`` (SI)
    — with coincident spike trains, so a channel-name match would fail but the
    coincidence matcher pairs them. Each source also gets an *independent* decoy
    unit that should match nothing across sources.

    Returns ``(mixed_df, si_df)`` in the exploded per-trial shape both
    SpikeSources return. No DB / recordings needed.
    """
    # the shared "true" neuron, generated once then spread over the recording
    truth = _spread_over_recording(make_synthetic_zombies_unit(
        neuron_id="_truth_", channel="_truth_", date=date, round_no=round_no,
        window_s=window_s, seed=seed))

    # mixed sort calls it C_011; SI sort (strongest channel) calls it C_020
    mixed_true = _copy_unit_as(
        truth, channel="Channel.C_011_Unit 1",
        neuron_id=f"AMG_{date}_{round_no}_Channel.C_011_Unit 1",
        keep_frac=0.92, jitter_s=0.0003, seed=seed + 1)
    si_true = _copy_unit_as(
        truth, channel="Channel.C_020_Unit 1",
        neuron_id=f"AMG_{date}_{round_no}_Channel.C_020_Unit 1",
        keep_frac=0.85, jitter_s=0.0003, seed=seed + 2)

    # independent decoys (their own spike trains, uncorrelated with truth)
    mixed_decoy = _spread_over_recording(make_synthetic_zombies_unit(
        neuron_id=f"AMG_{date}_{round_no}_Channel.C_005_Unit 1",
        channel="Channel.C_005_Unit 1", date=date, round_no=round_no,
        window_s=window_s, seed=seed + 50))
    si_decoy = _spread_over_recording(make_synthetic_zombies_unit(
        neuron_id=f"AMG_{date}_{round_no}_Channel.C_007_Unit 1",
        channel="Channel.C_007_Unit 1", date=date, round_no=round_no,
        window_s=window_s, seed=seed + 60))

    mixed_df = pd.concat([mixed_true, mixed_decoy], ignore_index=True)
    si_df = pd.concat([si_true, si_decoy], ignore_index=True)
    return mixed_df, si_df
