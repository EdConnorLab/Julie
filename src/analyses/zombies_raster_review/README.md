# Zombies raster review

A side investigation for **validating analysis by eye**: readable rasters of
specific ANOVA-passed units, plotted against the **Zombies** stimulus group
(≈9–10 monkeys), from two spike sources.

This does **not** modify `analyses/plot_rasters.py`. It reuses the same data
formats (`SpikeSource.load` DataFrames, `MonkeyGroup`/`MonkeyName`/
`EpochStartStop`/`SpikeTimes`) and the same rank logic
(`analyses/enums/monkey_names.py`), but ships its own, cleaner rendering.

## What it plots

**Single-unit** figure (`plot_zombies_raster`):

- **Raster (top)** — one row per trial, trials stacked and grouped by stimulus
  monkey in **dominance-rank order** (most dominant at the top). Each monkey
  block gets an alternating background shade, its spikes coloured along a
  viridis rank ramp, its id on the left and rank `#n` on the right.
- **PSTH (bottom)** — trial-averaged firing rate (Hz) ± SEM across all Zombies
  trials, sharing the raster's time axis.
- Stimulus onset marked at `t = 0`; the **ANOVA-significant window** (from the
  unit list) shaded in gold on both panels, so you see the response and *why the
  unit was selected* at a glance.

**Overlay** figure (`plot_overlay_raster`, `MODE = "overlay"`) — compares two
sorts of the **same channel** (manual vs SI) on one raster. Trials are matched
across sorts by `TaskField`; each trial row is split into a thin lane per sort,
coloured consistently, and the PSTHs are overlaid — so you can see spike-for-spike
where the two sorts agree or differ.

The recording **subject (81G) is always excluded** — she is never a stimulus, so
she never appears as a row even if she shows up in a session's trial table.

Compared to the grid-of-tiny-eventplots in `plot_rasters.py`, everything is in a
single scannable figure with the response window and rate called out.

## Inputs — two curated unit lists (bundled in `unit_lists/`)

| file | spike source | key column | window |
|------|--------------|-----------|--------|
| `zombies_mixed_manual_anova_passed.xlsx` | `MixedManualSpikeSource` (manually sorted **+** unsorted whole-channels) | `Cell` = `str(Channel)`, e.g. `Channel.C_027_Unit 1` or `Channel.C_020` | `Time Window` `"(start_ms, end_ms)"` |
| `zombies_si_sorted_anova_passed.csv` | `SISortedSpikeSource` (SpikeInterface consensus) | `NeuronID`, e.g. `AMG_2023-09-26_2_Channel.C_018_Unit 1` | `WindowStart_ms` / `WindowEnd_ms` |

`unit_lists.py` parses both into a common `RasterRequest` (source, date, round,
how to match the unit, and the significant window).

## Usage — just press ▶ Run in PyCharm

Open `run_zombies_rasters.py` and edit the **RUN CONFIG** block at the bottom,
then run the file directly (no terminal, no command-line arguments):

```python
MODE = "demo"     # "demo" | "mixed" | "si" | "overlay"
OUT_DIR = ...     # defaults to output/ next to this file (gitignored)

# for MODE == "overlay":
OVERLAY_DATE    = "2023-09-26"
OVERLAY_ROUND   = 2
OVERLAY_CHANNEL = "C_018"        # base channel; both sorts overlaid
OVERLAY_WINDOW_MS = None         # e.g. (100, 450) to shade a window
```

| `MODE` | what it does |
|--------|--------------|
| `"demo"` | fabricated data (no DB/recordings) — writes a single-unit raster and an overlay raster so you can see both formats immediately |
| `"mixed"` | one raster per unit in the mixed-manual Excel list (`MixedManualSpikeSource`) |
| `"si"` | one raster per unit in the SI-sorted CSV list (`SISortedSpikeSource`) |
| `"overlay"` | overlay the manual and SI sorts of one channel in one session |

Figures land in `OUT_DIR` (default `output/` beside the file, so they show up in
the PyCharm project tree). Sessions are loaded once each and reused across all
requested units.

> A terminal entry point (`main()` with `--source`/`--demo` flags) is still there
> if you ever want it, but it is not needed for the PyCharm workflow.

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
  (9 ranked monkeys). The subject, 81G, is excluded outright
  (`SUBJECT_MONKEY_ID` in `zombies_raster.py`). Any other unranked monkey that
  appeared would be drawn after the ranked ones rather than dropped.
- Windows in the lists are in **milliseconds**; the plot axis is in seconds.
