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
  block gets an alternating background shade, black spikes, its id on the left
  and rank `#n` on the right.
- **PSTH (bottom)** — trial-averaged firing rate (Hz) ± SEM across all Zombies
  trials, sharing the raster's time axis.
- Stimulus onset marked at `t = 0`; the **ANOVA-significant window** (from the
  unit list) shaded in gold on both panels **and labelled in ms** (e.g.
  `300–400 ms`) at the top of the band, so you see the response and *why the
  unit was selected* at a glance.
- The time axis runs to **2400 ms** by default (`xlim = 2.4`), so windows late
  in the trial are visible.

**Overlay** figure (`plot_overlay_raster`, `MODE = "overlay"`) — compares the
manual (mixed) and SI sorts of the **same neuron** on one raster. The two sorts
are paired by **spike-time coincidence**, *not* by channel name: SpikeInterface
names a neuron after its strongest channel, but the manual sort may have put that
same neuron on a different channel, so `NeuronID … C_020` can be `Cell C_011` in
the mixed list. Trials are matched across sorts by `TaskField`; each trial row is
split into a thin lane per sort, coloured consistently, and the PSTHs are
overlaid — so you can see spike-for-spike where the two sorts agree or differ.
See **Overlay matching** below.

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

# for MODE == "overlay" (sweeps every (date, round) in the two lists by default):
OVERLAY_DATE  = None     # e.g. "2023-09-26" to restrict to one date
OVERLAY_ROUND = None     # e.g. 2 (only used when OVERLAY_DATE is set)
COINCIDENCE_THRESHOLD = 0.2   # min coincidence fraction to call a match
RATIO_THRESHOLD       = 5.0   # …and it must be this many × chance
```

| `MODE` | what it does |
|--------|--------------|
| `"demo"` | fabricated data (no DB/recordings) — writes a single-unit raster, a same-channel overlay, and a **coincidence-matched cross-channel overlay** so you can see every format immediately |
| `"mixed"` | one raster per unit in the mixed-manual Excel list (`MixedManualSpikeSource`) |
| `"si"` | one raster per unit in the SI-sorted CSV list (`SISortedSpikeSource`) |
| `"overlay"` | for each `(date, round)` in the lists, coincidence-match the manual and SI sorts and overlay each matched neuron |

Figures land in `OUT_DIR` (default `output/` beside the file, so they show up in
the PyCharm project tree). Sessions are loaded once each and reused across all
requested units.

> A terminal entry point (`main()` with `--source`/`--demo`/`--overlay` flags) is
> still there if you ever want it, but it is not needed for the PyCharm workflow.

## Overlay matching — by coincidence, not by channel name

SpikeInterface (the automated sorter behind `SISortedSpikeSource`) does not tie a
neuron to a single channel: one neuron is picked up on several contacts, and we
force SI to name it after its strongest channel. The manual/mixed sort may have
assigned *that same neuron* to a **different** channel. So the SI `NeuronID`
`…Channel.C_020_Unit 1` and the mixed `Cell` `Channel.C_011_Unit 1` can be one and
the same neuron — matching the two sorts by channel name would miss it.

Instead, for **each unique `(date, round)`**, both sorts are loaded, each unit's
per-trial spikes are concatenated into one train (they share the recording clock),
and every mixed×SI unit pair is scored by **spike-time coincidence** — the same
metric used in `spikesorting.cross_channel_analysis` (reused here, not modified).
A pair is a match when its coincidence is ≥ `COINCIDENCE_THRESHOLD` **and**
≥ `RATIO_THRESHOLD` × chance. Matched pairs are collapsed into connected groups
(so an SI unit that equals *two* mixed units on a channel lands in one overlay),
and each group is drawn as one overlay. The ANOVA-passed cells in the two lists
are the *anchors*: only groups containing an anchor are drawn, and the anchor's
significant window is shaded.

Matching happens in `coincidence_match.py`; thresholds default to the values in
`coincidence_match.DEFAULT_*` and are a little more permissive than the
within-manual-sort defaults (a manual and an automated sort of one neuron agree
less than two channels of a single manual sort do).

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
| `coincidence_match.py` | pair the two sorts' units by spike-time coincidence |
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
- Windows in the lists are in **milliseconds**; the plot axis is in seconds and
  runs to 2.4 s (2400 ms) by default.
- Coincidence uses the fact that both sorts share one recording clock (same Intan
  file), so their spike trains are directly comparable. Second-valued spike times
  are discretised to integer sample indices only to feed the shared
  `coincidence` routine; the nominal rate cancels out of the millisecond window.
