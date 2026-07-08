# Cross-channel unit analysis

Tools to decide whether units that were **manually sorted per-channel** in
[`windowsort`](https://pypi.org/project/windowsort/) (via `juliewindowsort.py`)
are actually the **same neuron** picked up on more than one electrode.

## Why this matters

The manual sorter (`windowsort`) is driven one channel at a time. Each channel's
sort is saved into `sorted_spikes.pkl`:

```python
{
    Channel.C_021: {"Unit 1": np.ndarray([spike_idx, ...]), "Unit 2": ...},
    Channel.C_010: {"Unit 1": np.ndarray([...])},
    ...
}
```

(keys are `clat.intan.channels.Channel` enums; values map a per-channel unit
name to spike **sample indices** into `amplifier.dat`.)

Because contacts on the 32-channel linear probe are only 65 µm apart, one
neuron's extracellular spike routinely crosses threshold on **adjacent
channels**. Sorted independently, that single neuron becomes two "units" — a
duplicate — which double-counts it in every downstream analysis. This package
finds those duplicates.

## The evidence it computes

For every pair of units on *different* channels:

1. **Coincidence fraction** — fraction of one train's spikes with a partner in
   the other within ±0.4 ms. The single most decisive number: real duplicates
   sit near 1.0, independent neurons near chance (~0). Also reported as a
   multiple of the chance level.
2. **Cross-correlogram (CCG)** — a duplicate produces a needle-thin peak at zero
   lag; genuinely distinct neurons produce a broad or offset bump.
3. **Refractory check on the merged train** — a single neuron can't fire twice
   within its refractory period, so pooling two copies of it keeps the
   auto-correlogram's zero-lag hole. If merging *fills* that hole, the two units
   are probably different neurons.
4. **Physical channel distance** — from the probe map (`probe_geometry.py`).
   Duplicates should be on nearby contacts.
5. **Spatial footprint** *(sorted_spikes path only, needs the raw recording)* —
   the mean waveform across all contacts triggered on each unit's spikes. Same
   neuron ⇒ footprints peak at the same contact with near-identical shape.

## Two input paths

| path | source | units compared | evidence |
|------|--------|----------------|----------|
| **sorted_spikes** | one session's `sorted_spikes.pkl` (+ optional raw recording) | manually sorted only | all five (incl. waveform footprints) |
| **exploded** | `Cortana/exploded_spike_cache/{date}_round_{round}.pkl` | **sorted *and* unsorted** whole-channels | coincidence, distance, refractory (no footprints — no raw voltages in the cache) |

The **exploded** path is what lets you compare sorted-vs-unsorted and
unsorted-vs-unsorted pairs. The exploded cache stores per-trial spike **times in
seconds** for every channel; a channel that was manually sorted appears only as
its sorted unit(s) (its raw trace is dropped), so sorted-vs-unsorted pairs are
always cross-channel — exactly the duplicate question. Spikes are converted to
integer sample ticks (30 kHz) so the same metrics apply.

## Layout

| file | purpose |
|------|---------|
| `probe_geometry.py` | probe channel order → physical position, channel distances |
| `loader.py` | read `sorted_spikes.pkl` → tidy `Session` of `SortedUnit`s |
| `spiketrain_metrics.py` | coincidence, CCG/ACG, refractory, pairwise sweep |
| `waveforms.py` | cross-channel mean-waveform footprints (needs voltages) |
| `plots.py` | coincidence matrix, distance-vs-coincidence, per-pair report |
| `exploded_loader.py` | read an exploded-cache pkl → `Session` of sorted+unsorted units (integer ticks) |
| `exploded_analysis.py` | all-units coincidence per session + batch; CSVs, matrix, pair-type scatter |
| `run_analysis.py` | **PyCharm entry point** (edit the config block, press ▶ Run) |
| `make_synthetic_session.py` | fabricate a session with a *known* duplicate |
| `tests/` | `test_smoke.py` (sorted path), `test_exploded.py` (exploded path) |

## Usage — just press ▶ Run in PyCharm

Open `run_analysis.py`, edit the **RUN CONFIG** block at the bottom, then run the
file directly (no terminal, no arguments):

```python
MODE = "demo"     # "demo" | "sorted_pkl" | "exploded_session" | "exploded_batch"
OUT_DIR = ...     # defaults to results/ next to the file (gitignored)
```

| `MODE` | what it does | needs |
|--------|--------------|-------|
| `"demo"` | fabricated session with a planted duplicate (sorted path, incl. footprints) | nothing |
| `"sorted_pkl"` | one `sorted_spikes.pkl`; footprints if `WITH_RAW_VOLTAGES` | a session dir |
| `"exploded_session"` | all sorted+unsorted units for ONE exploded pkl | `EXPLODED_PKL` |
| `"exploded_batch"` | every session in the exploded cache dir | `EXPLODED_CACHE_DIR` |

### Outputs

*sorted_pkl / demo:* `coincidence_matrix.png`, `distance_vs_coincidence.png`,
per-candidate `pair_*.png` reports, `candidate_duplicates.csv`, `all_pairs.csv`,
`summary.txt`.

*exploded_session:* per-session folder with `all_pairs.csv` (every cross-channel
pair, with a `pair_type` column: sorted-sorted / sorted-unsorted /
unsorted-unsorted), `candidate_duplicates.csv`, `coincidence_matrix.png`, and
`distance_vs_coincidence.png` (coloured by pair type).

*exploded_batch:* one folder **per session** as above, **plus** a top-level
`all_candidates.csv` aggregating candidate pairs across every session (with
`session` and `pair_type` columns) — the dataset-wide table for deciding which
units to merge/drop.

### As a library

```python
from spikesorting.cross_channel_analysis.exploded_loader import load_exploded_session
from spikesorting.cross_channel_analysis import spiketrain_metrics as stm

session = load_exploded_session(".../Cortana/exploded_spike_cache/2023-10-04_round_4.pkl")
pairs = stm.all_pairs(session)              # every cross-channel pair, ranked
dups  = stm.candidate_duplicates(pairs)     # high coincidence AND well above chance
```

## Tuning

Config-block / function defaults:

- `window_ms = 0.4` — coincidence half-window (≈ propagation + a couple samples
  of jitter at 30 kHz).
- `coincidence_threshold = 0.3`, `ratio_threshold = 5.0` — a candidate needs
  coincidence ≥ 0.3 **and** ≥ 5× chance.
- `min_spikes = 50` *(exploded only)* — drop near-silent channels. A unit with a
  few spikes coincides 1.0 with any dense partner and would masquerade as a
  duplicate of everything, so it is excluded before pairing.
- `refractory_ms = 1.5` — refractory period for the merged-train check.

## Tests

```bash
python src/spikesorting/cross_channel_analysis/tests/test_smoke.py
python src/spikesorting/cross_channel_analysis/tests/test_exploded.py   # or: pytest
```

## Notes / limitations

- The probe map (`PROBE_CHANNEL_ORDER`, 65 µm pitch) is copied from
  `spikesorting/sort_spikes/sort_spikes.py`. Off-map channels get `distance = n/a`
  but spike-train metrics still work.
- Coincidence assumes both channels share the same recording clock (they do —
  same Intan file).
- Exploded-cache spikes are only the within-trial-epoch spikes, so CCG/ACG on
  those trains have mild trial-boundary artifacts; the central coincidence peak
  (what duplicate detection uses) is unaffected.
- **A pair that is high-coincidence with *many* channels at once, including
  distant ones, is usually a shared artifact (movement/stim transient), not a
  duplicate.** Real duplicates are high-coincidence with their *near* neighbours
  only — sort `all_pairs.csv` by `contacts_apart` to separate the two.
- Footprint panels need `windowsort` + `clat` and the raw Intan files; the
  exploded path needs neither.
