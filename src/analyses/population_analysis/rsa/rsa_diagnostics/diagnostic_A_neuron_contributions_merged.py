"""
Diagnostic A: Per-neuron contribution to a target identity's neural RDM row.

Tests whether a target identity's row in the neural RDM is driven by:
  - A few neurons with extreme target responses (sparse identity coding), or
  - Many neurons each contributing modestly (distributed coding).

Two metrics per neuron:
  Option 1 — Leave-one-neuron-out (LOO) Δ on the target's mean row dissim.
             Δ_k = mean_row_target(full) − mean_row_target(without neuron k)
             +Δ → cell distinguishes target from others (sparse-coding signature)
             −Δ → cell co-fires with target and others (correlation supporter)
             Goes through the actual correlation computation.

  Option 2 — Per-neuron target deviance score:
             dev_k = (r_k(target) − mean(r_k(others))) / std(r_k(all identities))
             Signed: +dev → target above others; −dev → below.
             Direct from rate matrix; no correlation step.

Cumulative removal curve: rank neurons by SIGNED Δ.  The "distinguishers first"
curve (Δ descending) is the cleaner sparse-vs-distributed test — it isolates
the cells mechanistically making the target distinct. Sparse → drops fast.
The "supporters first" curve (Δ ascending) is shown for symmetry and should
RISE (removing supporters destabilizes the target's correlation with others).
The |Δ| curve is kept as a secondary reference panel since it mixes both
directions.

Specificity check: top-5 neurons by |target deviance| are scored for deviance
to every other identity. Target-specific cells show one dark column; general
identity-selective cells show several.

Inputs
------
{REGION}_pseudopop_raw_rate_matrix.csv : saved from result['raw_rate_matrix'].
    Shape: (n_identities=9, n_neurons). Columns are neuron IDs.

Outputs (saved to ./{REGION}_diagnostic_A_{TARGET_IDENTITY}/)
-------------------------------------------------------------
- per_neuron_contributions.csv  : full per-neuron table
- option1_loo_delta_sorted.png  : signed LOO Δ, sorted
- option2_deviance_sorted.png   : signed deviance scores, sorted
- option1_vs_option2_scatter.png: agreement between the two metrics
- cumulative_removal_curve.png  : signed and |Δ| orderings vs random
- top5_tuning_profiles.png      : top-5 neurons' tuning across identities
- specificity_heatmap.png       : top-5 deviance to every identity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ──────────────────────────────────────────────────────────
# Config — edit paths if needed
# ──────────────────────────────────────────────────────────
REGION = 'AMG'
RATE_MATRIX_CSV = f'{REGION}_pseudopop_raw_rate_matrix.csv'
TARGET_IDENTITY = '7124'
EXPECTED_IDENTITIES = ['110E', '143H', '151J', '67G', '69X',
                       '7124', '72X',  '87J',  '94B']
TARGET_INDEX = EXPECTED_IDENTITIES.index(TARGET_IDENTITY)
SAVE_DIR = f'{REGION}_diagnostic_A_{TARGET_IDENTITY}'
N_RANDOM_SHUFFLES = 50
RNG_SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Load and validate
# ──────────────────────────────────────────────────────────
df = pd.read_csv(RATE_MATRIX_CSV)
print(f"Loaded {RATE_MATRIX_CSV}")
print(f"  Shape: {df.shape}")
print(f"  Columns (first 5): {list(df.columns[:5])}")

if df.shape[0] != 9:
    raise ValueError(
        f"Expected 9 rows (identities). Got {df.shape[0]}. "
        f"If CSV is (neurons × identities), transpose before saving.")

rate_matrix = df.values.astype(float)           # (9, n_neurons)
neuron_ids = [str(c) for c in df.columns]
n_identities, n_neurons = rate_matrix.shape
print(f"  Identities = {n_identities}, neurons = {n_neurons}")
print(f"  Target identity '{TARGET_IDENTITY}' at index {TARGET_INDEX}\n")


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def compute_rdm(rate_mat):
    """1 − Pearson r between all pairs of identity rate vectors."""
    return squareform(pdist(rate_mat, metric='correlation'))


def mean_row_offdiag(rdm, idx):
    """Mean of row `idx` excluding the diagonal entry."""
    row = np.delete(rdm[idx], idx)
    return float(np.mean(row))


# ──────────────────────────────────────────────────────────
# Baseline: target identity's mean row dissim from full population
# ──────────────────────────────────────────────────────────
rdm_full = compute_rdm(rate_matrix)
mean_row_full = mean_row_offdiag(rdm_full, TARGET_INDEX)
print(f"Baseline {TARGET_IDENTITY} mean row dissim "
      f"(full {n_neurons} neurons): {mean_row_full:.4f}\n")


# ──────────────────────────────────────────────────────────
# Option 1: Leave-one-neuron-out Δ
# ──────────────────────────────────────────────────────────
print("Option 1: computing LOO Δ ...")
loo_deltas = np.zeros(n_neurons)
for k in range(n_neurons):
    keep = np.ones(n_neurons, dtype=bool)
    keep[k] = False
    rdm_k = compute_rdm(rate_matrix[:, keep])
    loo_deltas[k] = mean_row_full - mean_row_offdiag(rdm_k, TARGET_INDEX)
print(f"  Δ range: [{loo_deltas.min():+.5f}, {loo_deltas.max():+.5f}]")
print(f"  Mean |Δ|: {np.mean(np.abs(loo_deltas)):.5f}\n")


# ──────────────────────────────────────────────────────────
# Option 2: per-neuron target deviance score
# ──────────────────────────────────────────────────────────
print("Option 2: computing per-neuron deviance score ...")
r_target = rate_matrix[TARGET_INDEX, :]                              # (n_neurons,)
others_mask = np.ones(n_identities, dtype=bool); others_mask[TARGET_INDEX] = False
r_others_mean = np.mean(rate_matrix[others_mask, :], axis=0)
r_all_std = np.std(rate_matrix, axis=0, ddof=0)
with np.errstate(divide='ignore', invalid='ignore'):
    deviance = np.where(r_all_std > 1e-10,
                        (r_target - r_others_mean) / r_all_std, 0.0)
print(f"  Deviance range: [{deviance.min():+.3f}, {deviance.max():+.3f}]")
print(f"  Mean |deviance|: {np.mean(np.abs(deviance)):.3f}\n")


# ──────────────────────────────────────────────────────────
# Summary tables
# ──────────────────────────────────────────────────────────
results = pd.DataFrame({
    'neuron_idx': np.arange(n_neurons),
    'neuron_id': neuron_ids,
    f'rate_to_{TARGET_IDENTITY}': r_target,
    'rate_to_others_mean': r_others_mean,
    'loo_delta': loo_deltas,
    'deviance': deviance,
    'abs_loo_delta': np.abs(loo_deltas),
    'abs_deviance': np.abs(deviance),
})
results.to_csv(os.path.join(SAVE_DIR, 'per_neuron_contributions.csv'),
               index=False)

cols_show = ['neuron_id', f'rate_to_{TARGET_IDENTITY}', 'rate_to_others_mean',
             'loo_delta', 'deviance']

print("=" * 70)
print(f"TOP 10 by +Δ (cells distinguishing {TARGET_IDENTITY})")
print("=" * 70)
top_pos = results.sort_values('loo_delta', ascending=False).head(10)
print(top_pos[cols_show].to_string(index=False))

print("\n" + "=" * 70)
print(f"TOP 10 by −Δ (cells supporting {TARGET_IDENTITY}↔others correlation)")
print("=" * 70)
top_neg = results.sort_values('loo_delta', ascending=True).head(10)
print(top_neg[cols_show].to_string(index=False))

print("\n" + "=" * 70)
print(f"TOP 10 by |deviance| (selectivity for {TARGET_IDENTITY})")
print("=" * 70)
top_dev = results.sort_values('abs_deviance', ascending=False).head(10)
print(top_dev[cols_show].to_string(index=False))

rho_agreement, _ = spearmanr(results['abs_loo_delta'], results['abs_deviance'])
print(f"\nAgreement: Spearman ρ(|LOO Δ|, |deviance|) = {rho_agreement:+.3f}")
print("(Note: |Δ| mixes distinguishers and supporters; deviance does not.)\n")


# ──────────────────────────────────────────────────────────
# Cumulative removal curves (signed orderings + |Δ| reference)
# Loop stops at 2 remaining neurons → no NaN tail, no warnings.
# ──────────────────────────────────────────────────────────
print("Computing cumulative removal curves ...")
MAX_REMOVE = n_neurons - 2

def cumulative_remove_curve(rate_mat, idx, removal_order, n_steps):
    curve = [mean_row_offdiag(compute_rdm(rate_mat), idx)]
    keep = np.ones(rate_mat.shape[1], dtype=bool)
    for k_drop in removal_order[:n_steps]:
        keep[k_drop] = False
        if keep.sum() < 2:
            break
        curve.append(mean_row_offdiag(compute_rdm(rate_mat[:, keep]), idx))
    return np.array(curve)

order_pos_signed = np.argsort(-loo_deltas)        # most positive first → distinguishers
order_neg_signed = np.argsort(loo_deltas)          # most negative first → supporters
order_abs        = np.argsort(-np.abs(loo_deltas)) # largest magnitude either way

curve_pos = cumulative_remove_curve(rate_matrix, TARGET_INDEX,
                                     order_pos_signed, MAX_REMOVE)
curve_neg = cumulative_remove_curve(rate_matrix, TARGET_INDEX,
                                     order_neg_signed, MAX_REMOVE)
curve_abs = cumulative_remove_curve(rate_matrix, TARGET_INDEX,
                                     order_abs, MAX_REMOVE)

# Random baseline
rng = np.random.default_rng(RNG_SEED)
n_pts = MAX_REMOVE + 1
random_curves = np.zeros((N_RANDOM_SHUFFLES, n_pts))
for s in range(N_RANDOM_SHUFFLES):
    random_curves[s] = cumulative_remove_curve(
        rate_matrix, TARGET_INDEX, rng.permutation(n_neurons), MAX_REMOVE)
random_mean = np.mean(random_curves, axis=0)
random_lo = np.percentile(random_curves, 2.5, axis=0)
random_hi = np.percentile(random_curves, 97.5, axis=0)


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────
print("Generating plots ...\n")

# 1. Sorted bar of LOO Δ
fig, ax = plt.subplots(figsize=(12, 4))
sorted_deltas = np.sort(loo_deltas)[::-1]
ax.bar(range(n_neurons), sorted_deltas,
       color=['#d62728' if d > 0 else '#1f77b4' for d in sorted_deltas],
       width=1.0, edgecolor='none')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Neurons (sorted by LOO Δ)')
ax.set_ylabel(f'LOO Δ on {TARGET_IDENTITY} row dissim')
ax.set_title(f'Option 1: per-neuron LOO Δ for {TARGET_IDENTITY}\n'
             f'(red = distinguishes target, blue = supports correlation)')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'option1_loo_delta_sorted.png'), dpi=150)
plt.close(fig)

# 2. Sorted bar of deviance
fig, ax = plt.subplots(figsize=(12, 4))
sorted_dev = np.sort(deviance)[::-1]
ax.bar(range(n_neurons), sorted_dev,
       color=['#d62728' if d > 0 else '#1f77b4' for d in sorted_dev],
       width=1.0, edgecolor='none')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Neurons (sorted by deviance)')
ax.set_ylabel(f'{TARGET_IDENTITY} deviance (z-units)')
ax.set_title(f'Option 2: per-neuron {TARGET_IDENTITY} deviance\n'
             f'(red = {TARGET_IDENTITY} above others, blue = below)')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'option2_deviance_sorted.png'), dpi=150)
plt.close(fig)

# 3. Agreement scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(np.abs(loo_deltas), np.abs(deviance), alpha=0.5, s=15)
ax.set_xlabel('|LOO Δ| (Option 1)')
ax.set_ylabel('|Deviance| (Option 2)')
ax.set_title(f'Option 1 vs Option 2\nSpearman ρ = {rho_agreement:+.3f}')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'option1_vs_option2_scatter.png'), dpi=150)
plt.close(fig)

# 4. Cumulative removal — signed orderings (primary) + |Δ| (reference)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
x = np.arange(n_pts)

ax = axes[0]
ax.fill_between(x, random_lo, random_hi, color='gray', alpha=0.3,
                label='Random (95% band)')
ax.plot(x, random_mean, color='gray', lw=1.5, label='Random (mean)')
ax.plot(x[:len(curve_pos)], curve_pos, color='#d62728', lw=2.0,
        label='Distinguishers first (Δ desc)')
ax.plot(x[:len(curve_neg)], curve_neg, color='#1f77b4', lw=2.0,
        label='Supporters first (Δ asc)')
ax.axhline(mean_row_full, color='k', ls='--', lw=0.8,
           label=f'Baseline ({mean_row_full:.3f})')
ax.set_xlabel('Number of neurons removed')
ax.set_ylabel(f'{TARGET_IDENTITY} mean row dissim (1−r)')
ax.set_title('Signed-Δ ordering (primary test)\n'
             '(sparse → red drops fast below random)')
ax.legend(loc='best', fontsize=8)

ax = axes[1]
ax.fill_between(x, random_lo, random_hi, color='gray', alpha=0.3,
                label='Random (95% band)')
ax.plot(x, random_mean, color='gray', lw=1.5, label='Random (mean)')
ax.plot(x[:len(curve_abs)], curve_abs, color='purple', lw=2.0,
        label='|Δ| desc (both directions)')
ax.axhline(mean_row_full, color='k', ls='--', lw=0.8,
           label=f'Baseline ({mean_row_full:.3f})')
ax.set_xlabel('Number of neurons removed')
ax.set_title('|Δ| ordering (mixed signs, reference)\n'
             '(harder to interpret — destabilizes structure)')
ax.legend(loc='best', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'cumulative_removal_curve.png'), dpi=150)
plt.close(fig)

# 5. Top-5 tuning profiles
# Use +Δ (distinguishers) for the top row instead of |Δ|, which mixes signs.
top5_pos_idx = order_pos_signed[:5]
top5_dev_idx = top_dev['neuron_idx'].values[:5]

fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey='row')
for col, k in enumerate(top5_pos_idx):
    ax = axes[0, col]
    rates = rate_matrix[:, k]
    bar_colors = ['#d62728' if i == TARGET_INDEX else 'steelblue'
                  for i in range(n_identities)]
    ax.bar(range(n_identities), rates, color=bar_colors)
    ax.set_xticks(range(n_identities))
    ax.set_xticklabels(EXPECTED_IDENTITIES, rotation=45, fontsize=7)
    ax.set_title(f'#{col+1} by +Δ\n{neuron_ids[k][:18]}\n'
                 f'Δ={loo_deltas[k]:+.4f}, dev={deviance[k]:+.2f}', fontsize=8)
    if col == 0:
        ax.set_ylabel('Firing rate')

for col, k in enumerate(top5_dev_idx):
    ax = axes[1, col]
    rates = rate_matrix[:, k]
    bar_colors = ['#d62728' if i == TARGET_INDEX else 'steelblue'
                  for i in range(n_identities)]
    ax.bar(range(n_identities), rates, color=bar_colors)
    ax.set_xticks(range(n_identities))
    ax.set_xticklabels(EXPECTED_IDENTITIES, rotation=45, fontsize=7)
    ax.set_title(f'#{col+1} by |dev|\n{neuron_ids[k][:18]}\n'
                 f'Δ={loo_deltas[k]:+.4f}, dev={deviance[k]:+.2f}', fontsize=8)
    if col == 0:
        ax.set_ylabel('Firing rate')

fig.suptitle(f'Top-5 neurons by each metric: tuning across identities\n'
             f'(red bar = {TARGET_IDENTITY})', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'top5_tuning_profiles.png'), dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────────────────
# Specificity check
# ──────────────────────────────────────────────────────────
print(f"Specificity check: top-5 (by {TARGET_IDENTITY} deviance) — "
      f"deviance to each identity\n")
deviance_per_identity = np.zeros((n_neurons, n_identities))
for i in range(n_identities):
    others = np.ones(n_identities, dtype=bool); others[i] = False
    r_i = rate_matrix[i, :]
    r_o = np.mean(rate_matrix[others, :], axis=0)
    r_s = np.std(rate_matrix, axis=0, ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        deviance_per_identity[:, i] = np.where(
            r_s > 1e-10, (r_i - r_o) / r_s, 0.0)

print("-" * 70)
header = f"  {'Neuron':<20s}" + "".join(f"{n:>8s}"
                                         for n in EXPECTED_IDENTITIES)
print(header)
print("-" * 70)
for k in top5_dev_idx:
    row = f"  {neuron_ids[k][:18]:<20s}"
    for i in range(n_identities):
        v = deviance_per_identity[k, i]
        marker = "*" if i == TARGET_INDEX else " "
        row += f"{v:+6.2f}{marker} "
    print(row)
print(f"\n(* = {TARGET_IDENTITY})\n")

fig, ax = plt.subplots(figsize=(8, 4))
top5_data = deviance_per_identity[top5_dev_idx, :]
vlim = float(np.max(np.abs(top5_data))) if top5_data.size else 1.0
im = ax.imshow(top5_data, aspect='auto', cmap='RdBu_r',
               vmin=-vlim, vmax=vlim)
ax.set_xticks(range(n_identities))
ax.set_xticklabels(EXPECTED_IDENTITIES, rotation=45, fontsize=8)
ax.set_yticks(range(5))
ax.set_yticklabels([neuron_ids[k][:18] for k in top5_dev_idx], fontsize=8)
ax.axvline(TARGET_INDEX - 0.5, color='k', lw=2)
ax.axvline(TARGET_INDEX + 0.5, color='k', lw=2)
fig.colorbar(im, ax=ax, label='Deviance (z)')
ax.set_title(f'Specificity of top-5 |{TARGET_IDENTITY} deviance| neurons\n'
             f'(boxed column = {TARGET_IDENTITY})')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'specificity_heatmap.png'), dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────────────────
# Read-out: concentration of contribution
# Primary metric: top-N share of +Δ signal (sparse-coding signature)
# Secondary metric: top-N share of |Δ| (mixed signs, for reference)
# ──────────────────────────────────────────────────────────
print("=" * 70)
print(f"CONCENTRATION OF DISTINGUISHING CONTRIBUTION (Δ > 0 only)")
print("=" * 70)

pos_deltas_sorted = np.sort(loo_deltas[loo_deltas > 0])[::-1]
n_pos = len(pos_deltas_sorted)
print(f"  Number of distinguishing cells (Δ > 0): {n_pos}/{n_neurons}")

if n_pos > 0:
    cum_pos = np.cumsum(pos_deltas_sorted) / pos_deltas_sorted.sum()
    for n in [1, 3, 5, 10, 20]:
        if n <= n_pos:
            print(f"  Top {n:>2d} distinguishers → "
                  f"{cum_pos[n-1]*100:5.1f}% of total +Δ")
    top5_pos_frac = cum_pos[4] if n_pos >= 5 else cum_pos[-1]
    if top5_pos_frac > 0.5:
        verdict = "STRONGLY SPARSE (top 5 distinguishers carry >50% of +Δ)"
    elif top5_pos_frac > 0.3:
        verdict = "MODERATELY SPARSE (top 5 distinguishers carry 30–50%)"
    else:
        verdict = "DISTRIBUTED (top 5 distinguishers carry <30%)"
    print(f"\n  Verdict from +Δ concentration: {verdict}")
else:
    print("  No cells with Δ > 0 found.")
print(f"  Cross-check by inspecting the cumulative removal curve plot.\n")

print("=" * 70)
print("CONCENTRATION OF |Δ| (mixed signs, for reference)")
print("=" * 70)
abs_deltas_sorted = np.sort(np.abs(loo_deltas))[::-1]
cum_abs = np.cumsum(abs_deltas_sorted) / abs_deltas_sorted.sum()
for n in [1, 3, 5, 10, 20]:
    if n <= n_neurons:
        print(f"  Top {n:>2d} by |Δ| → {cum_abs[n-1]*100:5.1f}% of total |Δ|")
print()

print(f"All outputs in: {SAVE_DIR}/")
