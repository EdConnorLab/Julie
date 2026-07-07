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
5. **Spatial footprint** *(optional, needs the raw recording)* — the mean
   waveform across all contacts triggered on each unit's spikes. Same neuron ⇒
   footprints peak at the same contact with near-identical shape (cosine
   similarity ≈ 1).

## Layout

| file | purpose |
|------|---------|
| `probe_geometry.py` | probe channel order → physical position, channel distances |
| `loader.py` | read `sorted_spikes.pkl` → tidy `Session` of `SortedUnit`s |
| `spiketrain_metrics.py` | coincidence, CCG/ACG, refractory, pairwise sweep |
| `waveforms.py` | cross-channel mean-waveform footprints (needs voltages) |
| `plots.py` | coincidence matrix, distance-vs-coincidence, per-pair report |
| `run_analysis.py` | CLI: session → figures + ranked CSV + summary |
| `make_synthetic_session.py` | fabricate a session with a *known* duplicate |
| `tests/test_smoke.py` | end-to-end test on the synthetic session |

## Usage

Run from the repo's `src/` directory (so `spikesorting` is importable).

```bash
# spike-train analysis only (no raw recording needed)
python -m spikesorting.cross_channel_analysis.run_analysis /path/to/231030_round2

# add the raw-recording spatial footprints (heavy: loads/filters the recording)
python -m spikesorting.cross_channel_analysis.run_analysis /path/to/session --with-voltages

# try it with no data — fabricated session with a planted duplicate
python -m spikesorting.cross_channel_analysis.run_analysis --demo
```

Outputs (written to `<session>/cross_channel_analysis/` by default):

- `overview_coincidence_matrix.png` — units × units synchrony heatmap
- `overview_distance_vs_coincidence.png` — duplicates cluster at close+synchronous
- `pair_NN_<a>__<b>.png` — one diagnostic report per candidate duplicate
- `candidate_duplicates.csv`, `all_pairs.csv`, `summary.txt`

### As a library

```python
from spikesorting.cross_channel_analysis.loader import load_sorted_spikes
from spikesorting.cross_channel_analysis import spiketrain_metrics as stm

session = load_sorted_spikes("/path/to/session/sorted_spikes.pkl")
pairs = stm.all_pairs(session)                 # every cross-channel pair, ranked
dups  = stm.candidate_duplicates(pairs)        # the ones that look like duplicates
```

## Tuning

Defaults in `run_analysis.py` / `spiketrain_metrics.py`:

- `window_ms = 0.4` — coincidence half-window (≈ propagation + a couple samples
  of jitter at 30 kHz).
- `coincidence_threshold = 0.3`, `ratio_threshold = 5.0` — a pair is a candidate
  only if coincidence ≥ 0.3 **and** ≥ 5× chance.
- `refractory_ms = 1.5` — refractory period for the merged-train check.

## Tests

```bash
python src/spikesorting/cross_channel_analysis/tests/test_smoke.py   # or: pytest
```

## Notes / limitations

- The probe map (`PROBE_CHANNEL_ORDER`, 65 µm pitch) is copied from
  `spikesorting/sort_spikes/sort_spikes.py`. If a session used a different probe,
  update `probe_geometry.py`; channel distances become `n/a` for off-map
  channels but the spike-train metrics still work.
- Coincidence assumes both channels share the same recording clock (they do —
  same Intan file), so spike indices are directly comparable.
- Footprint panels need `windowsort` + `clat` and the raw Intan files; the
  spike-train analysis needs neither.
