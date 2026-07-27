# response_window_benchmark

A standalone bake-off for **response-window / high-firing-rate detectors**. It runs
several candidate algorithms on a curated set of real example cells, draws each
method's detected windows next to the raster/PSTH, and scores them against windows
you annotate by eye.

It is deliberately separate from `jun2026_grant_investigation` and reuses (never
modifies) the spike sources, the cell lists, and the raster plotting engine.

## Why this exists

The current detector (`response_window_finder/threshold_window_detection.py`)
z-scores the **summed PSTH over the whole trial** and keeps bins with `z > 0.5`.
Two problems make it miss responses that are obvious in the raster:

1. **Self-baseline.** The "baseline" is the whole-trial mean, so a strong response
   inflates its own baseline and suppresses its own detection.
2. **Arbitrary threshold.** `z > 0.5` is tied to no false-positive rate, and
   collapsing across trials discards the trial-to-trial reliability that separates
   signal from noise.

## The methods compared

| name | idea | reference |
|------|------|-----------|
| `current` | whole-trial z-score, `z>0.5` (the baseline to beat) | existing code |
| `cusum` | one-sided CUSUM change detection on the z-scored PSTH | — |
| `baseline_z` | z-score vs the **real pre-stimulus baseline**, N-sigma, k-consecutive | classic PSTH practice |
| `poisson` | per-bin Poisson surprise that firing exceeds the pre-stim baseline | Hanes et al. 1995; Legéndy & Salcman 1985 |
| `cluster_perm` | per-bin stat vs each trial's pre-stim baseline, temporal clustering, permutation null (FWE over time) | Maris & Oostenveld 2007 |
| `zeta` | binning-free deviation of the cumulative spike distribution from stationarity | Montijn et al. 2021 (eLife 71969) |

The baseline comes **directly from the pre-stimulus window** (the SI-sorted
pre-stim caches, e.g. `sorted_spike_cache_pre1000ms`, store spikes from
`onset - pre` onward). There is no within-trial baseline estimation — the
`baseline_z` / `poisson` / `cluster_perm` detectors require a pre-stim period and
refuse without one. All detectors are pure numpy/scipy — nothing new to install
on the lab machine beyond the pinned scientific stack.

Scope: this first pass targets **SI-sorted units** from the pre-stim cache. Other
sources (MUA, mixed) will slot in once their pre-stim caches exist.

## Main workflow — score against a hand-picked answer key

Give it a spreadsheet of cells with their true windows; it runs every detector on
each cell and compares.

```bash
cd src
python -m analyses.response_window_benchmark.run_benchmark --mode answerkey \
    --answer-key /path/to/window_test.xlsx --out output_set1
```

Outputs in `output_set1/`:
- `comparison/<cell>.png` — raster + PSTH + one window-lane per method + a **TRUTH** lane.
- `scoreboard.csv` + `scoreboard.png` — per method: hits / misses / `false_alarms` /
  `off_target_fa` / precision / recall / **F1** / `mean_coverage` / `mean_iou` / onset error.
- `diagnostic_table.csv` — **one row per cell**: the truth window(s) next to EACH
  method's raw detected window(s) + per-method hit / off-target / coverage. This is
  where you SEE over-detection (a response split into several windows).
- `scores_per_cell.csv`, `answerkey_cells.csv`, `answerkey_skipped.csv`.

Two answer-key layouts are auto-detected:
- **NeuronID layout** (`window_test.xlsx`): `NeuronID | Source | WindowStart | WindowStop`
  (ms). `Source` picks the cache (`SI sorted` → `sorted_spike_cache_pre1000ms`,
  `mixed`/`manual` → `exploded_spike_cache_pre1000ms`, `mua` →
  `threshold_mua_spike_cache_pre1000ms`). A NeuronID on several rows = several windows.
- **Cell layout** (`Ed_handpicked_..._ANOVA_passed_Zombies.xlsx`):
  `Date | Round No. | Time Window | Cell | [P Value]`; loaded from the pre-stim
  exploded cache. Online-thresholded channels (no `_Unit`) have no pre-stim and are skipped.

Region is never matched: cells resolve by `{Date}_{Round}_Channel.C_XXX[_Unit N]`, so a
sheet `Unknown_…` still matches the cache's real `AMG_…`. Partial caches are fine —
sessions you haven't built yet print "session not in cache — skipped". Use a separate
`--out` per answer key so you can keep a **dev set** and a **held-out set** distinct
(the fair comparison is on the set a method was *not* tuned on).

## Optional workflow — auto-select SI cells for blind annotation

```bash
cd src

# 1) pick ~20 SI-sorted cells from the pre-stim cache (stratified across
#    region / session), render UNMARKED rasters (baseline shown) to inspect,
#    and write a blank template
python -m analyses.response_window_benchmark.run_benchmark --mode select --n 20 \
    --cache-subdir sorted_spike_cache_pre1000ms --pre-stim 1.0

#    -> output/candidate_cells.csv
#    -> output/ground_truth_template.xlsx      (fill in the windows you SEE)
#    -> output/rasters_for_annotation/*.png

# 2) after filling the template, run every detector, draw comparisons, and score
python -m analyses.response_window_benchmark.run_benchmark --mode score

#    -> output/comparison/*.png                (raster + PSTH + one lane per method + TRUTH)
#    -> output/scores_per_cell.csv
#    -> output/scoreboard.csv  +  scoreboard.png   (winner = best precision/recall/F1)
```

`--mode score` also works before you annotate — it just draws the overlays and
skips scoring, so you can eyeball which method tracks the response first.

The `ListWindow_ms` column in the template is the **old detector's** window, shown
for reference only. It is *not* the answer — annotate what you see.

## Ground-truth template

Fill, per cell: `TrueWindow{1,2,3}_start_ms` / `_end_ms` for each window you see
(blank if unused), optional `ResponseType` (transient / sustained / multi-peak /
suppression), and `NoResponse` = `y` for cells with no response. `cell_key` is the
identity column — don't edit it.

## Verify the code without the caches

```bash
cd src
python -m analyses.response_window_benchmark.synthetic_smoke_test
```

Builds a synthetic cell with a known window, checks every detector runs and the
principled ones recover it, and renders a comparison figure — no lab data needed.

## Files

- `psth.py` — trial extraction (keeps pre-stim), PSTH matrix, pre-stim baseline, optimal bin width
- `detectors.py` — the six detectors behind one `WindowDetector` interface
- `plotting.py` — the per-cell raster + PSTH + window-lane comparison figure
- `cell_selection.py` — stratified candidate sampling from the five cell lists
- `ground_truth.py` — annotation template writer + loader
- `scoring.py` — IoU matching + precision/recall/F1 scoreboard
- `run_benchmark.py` — the `select` / `score` entrypoint
- `synthetic_smoke_test.py` — cache-free end-to-end check

## Tuning

Detector defaults (`detectors.default_detectors`) are first guesses. Once you have
annotations, adjust per-method knobs — `n_sigma` (`baseline_z`), `surprise_thr`
(`poisson`), `t_thresh`/`alpha`/`n_perm` (`cluster_perm`), `alpha`/`rate_frac`
(`zeta`) — and re-run `--mode score` to move the scoreboard.
