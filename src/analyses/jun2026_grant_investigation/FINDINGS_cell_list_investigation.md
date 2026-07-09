# Aim 3c allocentric result — why the cell lists disagree

**Question.** The grant reports a significant allocentric effect (single neurons
encoding focal monkeys' social-relationship patterns): total permutation
**p ≈ 0.008** on the PI's 74-cell list. Re-running the *identical* pipeline on the
SI-sorted, KW-passed 38-cell list does **not** reproduce it (**p ≈ 0.21**). Why?

**Answer.** The effect is carried almost entirely by the **unsorted multiunit**
half of the PI's list. His own *sorted single units* show no effect (p ≈ 0.87);
his *multiunit* channels are extremely significant (p < 0.0001). Because the
SI-sorted KW list contains only sorted single units, it agrees with the PI's
own sorted units — both are null. The discrepancy is a **unit-type** effect, not
a cell-count, QC, time-window, or matrix-typo effect.

## How to reproduce

```bash
cd src/analyses/jun2026_grant_investigation
python investigate_cell_lists.py            # full, NPERM=10000 (~3 min)
NPERM=500 python investigate_cell_lists.py  # fast smoke test
```

Outputs `investigation_results.csv` and `figures/fig5_unit_type.png`. Paths
resolve on both the PI's machine and a fresh checkout (see `common.resolve_path`).

## Results (NPERM=10000, THRESH=0.5, seed=20251121)

Each condition runs the exact grant pipeline; **the comparable quantity is the
permutation p**, not the summed R² (which scales trivially with cell count).

| condition | cells | unit type | sumR² | **p_total** | interpretation |
|---|---:|---|---:|---:|---|
| A_his74_grant | 74 | mixed | 192.68 | **0.0078** ✱ | exact grant baseline (reproduces his ~0.0078) |
| B_his74_corr | 74 | mixed | 192.56 | **0.0067** ✱ | matrix typo fixed (SUB=15) — barely moves |
| C_his_sorted37 | 37 | sorted | 78.51 | **0.8684** | PI's own sorted single units: **no effect** |
| D_his_multi37 | 37 | multiunit | 114.05 | **0.0000** ✱ | PI's multiunit channels: **very strong** |
| E_kw38_corr | 38 | sorted | 82.77 | **0.2060** | SI-sorted KW list: no effect (the "null result") |
| F_kw38_grant | 38 | sorted | 82.35 | **0.2335** | KW list under the grant matrix — still null |

✱ = p < 0.05. See `figures/fig5_unit_type.png`.

## What each comparison isolates

- **Baseline reproduces (A).** p = 0.0078, and the per-source significant set is
  72X, 94B, 67G, 151J — matching the grant. The edited/portable pipeline is
  faithful, so every difference below is attributable to *cells*, not code.
- **The matrix typo is a non-factor (A vs B).** Correcting SUB_TO[110E→94B] from
  5 to 15 moves p from 0.0078 to 0.0067. It does not explain anything; it is set
  aside for the cell-list question.
- **Unit type is the whole story (C vs D).** Same file, same sorting pipeline,
  same time windows, same QC, **same cell count (37)** — split only by whether
  the channel was sorted into a single unit. Sorted → p = 0.87; multiunit →
  p = 0.0000. Equal n with opposite results **rules out cell-count / statistical
  power** as the explanation.
- **The KW list agrees with the PI's sorted units (C vs E).** Both all-sorted,
  both null (0.87 and 0.21). The KW-vs-his-74 gap is fully explained by the fact
  that his 74 is half multiunit while the KW 38 is all single units.
- **Time windows are not needed to explain the gap.** His sorted units use the
  *same wide/late windows* as his multiunit units yet are already null, so the
  narrow stimulus-locked KW windows are not required to kill the effect. Window
  and SI-vs-manual-sort differences only shuffle results *within* the already-null
  sorted regime (0.87 → 0.21); they never produce significance.

## Not an artifact of firing rate or degeneracy

A reviewer's first objection — "sorted single units are just quieter, so their
regressions are degenerate/underpowered" — does not hold:

- **Zero degenerate (constant-response) regressions** in any group (0/370 sorted,
  0/370 multiunit, 0/380 KW). Nothing is being silently forced to R² = 0.
- **Multiunit fires *less*, not more** (median response 1.63 vs 2.93 sorted, 4.76
  KW). Since R² is scale-invariant, the multiunit effect is a genuine difference
  in the *pattern* of relative firing across the 8 stimulus monkeys, not a rate
  or SNR artifact.

## Interpretation

Multiunit activity pools spikes from several neurons on a channel. The leading
explanation for why pooled activity tracks the social matrices while isolated
single units do not: pooled/population activity captures a **global stimulus
dimension** (e.g. salience, familiarity, or dominance rank of each stimulus
monkey) that is itself correlated with the affiliation/submission/agonism
matrices, whereas individual sorted neurons are selective and idiosyncratic and
rarely match a specific social matrix. This is exactly the confound that
single-unit isolation is meant to remove.

**Bottom line for the claim.** The grant's allocentric single-neuron encoding
result rests on multiunit (multi-neuron hash) activity. When the analysis is
restricted to isolated single units — the appropriate unit for a single-neuron
encoding claim, whether the PI's own sorted units or the SI-sorted QC'd units —
the effect is not present.

## Caveats

- All permutation p-values are **uncorrected**. Under Bonferroni over 10 sources
  (α = 0.005), only a couple of multiunit/mixed sources survive (94B, 151J); no
  sorted-unit source survives in any condition.
- The KW "38 cells" are 38 (neuron × significant-window) combinations from **34
  neurons** — 4 neurons contribute two windows — so those fits are not fully
  independent. This does not affect the conclusion (the KW list is null either
  way), but the independent unit is the neuron, not the fit.
- C vs E were not further decomposed into (SI-vs-manual sort) × (QC) × (windows)
  because both land in the same non-significant regime; decomposition is
  unnecessary to answer the question and is left as optional follow-up.
