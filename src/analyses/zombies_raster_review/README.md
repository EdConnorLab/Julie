# Zombies raster review

A side investigation for **validating analysis by eye**: readable rasters of
specific ANOVA-passed units, plotted against the **Zombies** stimulus group
(≈9–10 monkeys), from two spike sources.

This does **not** modify `analyses/plot_rasters.py`. It reuses the same data
formats (`SpikeSource.load` DataFrames, `MonkeyGroup`/`MonkeyName`/
`EpochStartStop`/`SpikeTimes`) and the same rank logic
(`analyses/enums/monkey_names.py`), but ships its own, cleaner rendering.

## What it plots

One figure per unit (see `zombies_raster.py`):

- **Raster (top)** — one row per trial, trials stacked and grouped by stimulus
  monkey in **dominance-rank order** (most dominant at the top). Each monkey
  block gets an alternating background shade, its spikes coloured along a
  viridis rank ramp, its id on the left and rank `#n` on the right.
- **PSTH (bottom)** — trial-averaged firing rate (Hz) ± SEM across all Zombies
  trials, sharing the raster's time axis.
- Stimulus onset marked at `t = 0`; the **ANOVA-significant window** (from the
  unit list) shaded in gold on both panels, so you see the response and *why the
  unit was selected* at a glance.

Compared to the grid-of-tiny-eventplots in `plot_rasters.py`, everything for one
unit is in a single scannable figure with the response window and rate called
out.

## Inputs — two curated unit lists (bundled in `unit_lists/`)

| file | spike source | key column | window |
|------|--------------|-----------|--------|
| `zombies_mixed_manual_anova_passed.xlsx` | `MixedManualSpikeSource` (manually sorted **+** unsorted whole-channels) | `Cell` = `str(Channel)`, e.g. `Channel.C_027_Unit 1` or `Channel.C_020` | `Time Window` `"(start_ms, end_ms)"` |
| `zombies_si_sorted_anova_passed.csv` | `SISortedSpikeSource` (SpikeInterface consensus) | `NeuronID`, e.g. `AMG_2023-09-26_2_Channel.C_018_Unit 1` | `WindowStart_ms` / `WindowEnd_ms` |

`unit_lists.py` parses both into a common `RasterRequest` (source, date, round,
how to match the unit, and the significant window).

## Usage

Run from the repo's `src/` directory (so `analyses` / `data_access` import).

```bash
cd src

# manually-sorted + unsorted units
python -m analyses.zombies_raster_review.run_zombies_rasters --source mixed

# SpikeInterface-sorted units
python -m analyses.zombies_raster_review.run_zombies_rasters --source si

# preview the format on a fabricated unit (no data / DB needed)
python -m analyses.zombies_raster_review.run_zombies_rasters --demo
```

`--list <path>` overrides the bundled list; `--out <dir>` sets the figure output
directory; `--xlim` and `--psth-bin-ms` tune the axes. Sessions are loaded once
each and reused across all requested units in that session.

### As a library

```python
from analyses.zombies_raster_review.zombies_raster import plot_zombies_raster
# neuron_df: one unit's per-trial rows from a SpikeSource
plot_zombies_raster(neuron_df, neuron_label="C-018 Unit 1",
                    window_s=(0.10, 0.45), p_value=0.004, save_path="out.png")
```

## Layout

| file | purpose |
|------|---------|
| `unit_lists.py` | parse the Excel/CSV lists → `RasterRequest`s |
| `zombies_raster.py` | the raster + PSTH renderer (Zombies-focused) |
| `run_zombies_rasters.py` | CLI: list → load sessions → render + save |
| `make_synthetic_zombies_session.py` | fabricate a session for demo/tests |
| `tests/test_smoke.py` | end-to-end tests (no DB / recordings) |
| `unit_lists/` | the two curated unit lists |

## Tests

```bash
python src/analyses/zombies_raster_review/tests/test_smoke.py   # or: pytest
```

## Notes

- The `--source mixed`/`--source si` real runs need the project's spike caches
  (and, on cache miss, DB + Intan files), exactly like `plot_rasters.py`. The
  `--demo` path and the tests need none of that.
- Rank order comes from `Zombies.RANK` in `analyses/enums/monkey_names.py`
  (9 ranked monkeys; 81G is unranked and, if present, is drawn after the ranked
  ones rather than dropped).
- Windows in the lists are in **milliseconds**; the plot axis is in seconds.
