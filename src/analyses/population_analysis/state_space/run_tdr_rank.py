"""
run_tdr_rank.py
===============

First-pass Targeted Dimensionality Reduction (TDR) for the social-rank
hypothesis. Implements the linear-regression axis approach from
Mante, Sussillo, Shenoy & Newsome (2013, Nature 503:78-84).

Pipeline
--------
1. Load neural data and pick a region (default AMG).
2. Pick the analysis group(s) — within-group runs only for now.
3. Build a per-stimulus-monkey trial-averaged firing-rate tensor via
   binning_by_condition.build_matrix_by_condition with each stimulus
   monkey as its own condition.
4. Convert counts to Hz and z-score every neuron across (condition x time).
5. For each timepoint t, regress the population response on the rank
   variable across conditions, yielding a per-neuron beta vector beta[t].
6. Define the rank axis as the time-averaged beta across a window
   of strong encoding (or just the L2-largest single timepoint), then
   normalize to unit length.
7. Project the population trajectory of every condition onto the rank
   axis to get a 1-D "rank projection" time course per stimulus monkey.
8. Plot, colored by rank.

Sensitivity check
-----------------
Re-runs the full_dominance_first analysis after dropping the rank-1 monkey (the alpha
male confound) so you can see whether the rank signal survives without
the only adult male in each group.

What this script intentionally does NOT do
------------------------------------------
- No permutation test. Eyeball first; we'll add stats once we know
  whether there's anything to test.
- No cross-group pooling. Within-group only.
- No QR-orthogonalization against other axes (familiarity, sex, etc).
  Single-axis TDR.
- No Instigators (matrices incomplete) or Stranger Things (no rank).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TrajectoryConfig
from data_loading import load_and_filter
from binning_by_condition import build_matrix_by_condition

# Reuse the script we already wrote for rank computation
sys.path.insert(0, '/population/neural_trajectory')
from social_rank_analysis import load_group_matrices, davids_score


MONKEY_INFO_PATH = "/social_data/monkeyinfo.csv"
PLOT_SAVE_DIR = ("/home/connorlab/Documents/JulieData/Cortana/"
                 "analysis_results/tdr_rank")


# ---------------------------------------------------------------------------
# Rank variable
# ---------------------------------------------------------------------------
def compute_group_rank(group_name):
    """
    Returns a Series indexed by monkey name (str) of within-group
    z-scored DS_combined. Monkeys with all-zero rows AND all-zero
    columns in both agonism and submission are dropped (no rank
    information available).
    """
    mats = load_group_matrices(group_name)
    ago = mats['agonism']
    sub = mats['submission']
    combined = ago + sub.T   # wins matrix: i dominated j

    ds = davids_score(combined)

    # Drop monkeys with no rank information at all.
    # A monkey has no info iff her row and column are both entirely zero
    # in BOTH agonism and submission. David's score collapses these to
    # exactly the neutral midpoint, which would otherwise pollute the
    # z-scoring.
    has_info = []
    for m in ds.index:
        zero_in_ago = (ago.loc[m].sum() == 0) and (ago[m].sum() == 0)
        zero_in_sub = (sub.loc[m].sum() == 0) and (sub[m].sum() == 0)
        has_info.append(not (zero_in_ago and zero_in_sub))
    ds = ds[has_info]

    if ds.std(ddof=0) == 0:
        raise ValueError(f"David's scores in {group_name} have zero variance")

    z = (ds - ds.mean()) / ds.std(ddof=0)
    z.name = 'rank_z'
    return z


# ---------------------------------------------------------------------------
# Neural tensor
# ---------------------------------------------------------------------------
def build_rank_tensor(df, cfg, monkey_list):
    """
    Trial-average per stimulus monkey, return a (n_monkeys, n_bins, n_neurons)
    tensor in Hz. The condition axis order follows sorted(monkey_list)
    because build_matrix_by_condition sorts internally.
    """
    cond_map = {m: m for m in monkey_list}
    pca_matrix, row_meta_df, info = build_matrix_by_condition(
        df, cfg, cond_map, min_reps=1)

    n_bins = info['n_bins']
    n_conditions = info['n_conditions']
    n_neurons = pca_matrix.shape[1]

    # counts -> Hz, then reshape
    rates = pca_matrix / cfg.bin_width
    R = rates.reshape(n_conditions, n_bins, n_neurons)

    return R, info['conditions'], info, row_meta_df


def zscore_neurons(R):
    """
    Z-score each neuron across (conditions x time). Returns same shape.
    Drops any neuron with zero variance (constant firing) by setting its
    z-scored values to zero rather than dividing by zero.
    """
    n_cond, n_bins, n_neu = R.shape
    flat = R.reshape(-1, n_neu)
    mu = flat.mean(axis=0, keepdims=True) # mean fr of each neuron across all conditions and all time bins
    sd = flat.std(axis=0, keepdims=True, ddof=0)
    out = np.zeros_like(flat) # initialize as zeros so that zero-variance neurons become 0
    nz = sd[0] > 0
    out[:, nz] = (flat[:, nz] - mu[:, nz]) / sd[:, nz] # if zero variance then don't use it for z-scoring (only nz True elements were z-scored)
    return out.reshape(n_cond, n_bins, n_neu)


# ---------------------------------------------------------------------------
# TDR core
# ---------------------------------------------------------------------------
def fit_rank_axis(R_z, rank_vec):
    """
    For every timepoint t, regress R_z[:, t, n] on rank_vec across
    conditions. Returns:
        beta : (n_bins, n_neurons)  per-timepoint regression slopes
        norms : (n_bins,)           L2 norm of beta at each timepoint
    """
    n_cond, n_bins, n_neu = R_z.shape
    x = rank_vec.astype(float)
    x = x - x.mean() # rank_vec is already z-scored so mean should be 0 but just making sure
    denom = (x ** 2).sum() # Σ(x_i − x̄)² but since x.mean is 0, we could just do x**2
    if denom == 0:
        raise ValueError("rank_vec has zero variance")

    beta = np.zeros((n_bins, n_neu))
    for t in range(n_bins):
        Y = R_z[:, t, :]                    # (n_cond, n_neu)
        Yc = Y - Y.mean(axis=0, keepdims=True) # y_i − ȳ
        beta[t] = (x[:, None] * Yc).sum(axis=0) / denom # x[:, None] makes x a column vector with (n_cond, 1)

    norms = np.linalg.norm(beta, axis=1)
    return beta, norms


def define_axis(beta, norms, mode='peak', window=None):
    """
    Collapse the per-timepoint beta into a single fixed direction.

    mode='peak'   : pick the single timepoint with the largest ||beta||.
    mode='window' : average beta across a contiguous bin range
                    (provided as window=(start, stop) inclusive).
    Returns a unit vector (n_neurons,).
    """
    if mode == 'peak':
        t_star = int(np.argmax(norms))
        v = beta[t_star]
        chosen = f"peak t={t_star}"
    elif mode == 'window':
        s, e = window
        v = beta[s:e + 1].mean(axis=0)
        chosen = f"window bins {s}-{e}"
    else:
        raise ValueError(mode)

    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Axis vector is zero length")
    return v / n, chosen


def project(R_z, axis):
    """Project (n_cond, n_bins, n_neu) onto unit vector axis -> (n_cond, n_bins)."""
    return R_z @ axis


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rank_projection(proj, conditions, ranks, time_axis,
                         title='', save_path=None):
    """
    proj       : (n_cond, n_bins) projection onto rank axis
    conditions : list of monkey names in the same order as proj's first axis
    ranks      : Series indexed by monkey name -> rank z-score
    time_axis  : (n_bins,) seconds
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    rank_vals = np.array([ranks[c] for c in conditions])
    norm = plt.Normalize(rank_vals.min(), rank_vals.max())
    cmap = plt.get_cmap('coolwarm')

    for i, c in enumerate(conditions):
        color = cmap(norm(rank_vals[i]))
        ax.plot(time_axis, proj[i], color=color, lw=2,
                label=f'{c} (z={rank_vals[i]:+.2f})')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Projection on rank axis (a.u.)')
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Rank (z-scored DS_combined)\n← subordinate    dominant →')

    ax.legend(fontsize=7, loc='best', ncol=2)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_beta_norm(norms, time_axis, title='', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(time_axis, norms, lw=2)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('||beta_rank(t)||')
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# One full_dominance_first run for one group
# ---------------------------------------------------------------------------
def run_one_group(df, cfg, group_name, drop_rank1=False):
    print(f"\n{'='*60}")
    print(f"Group: {group_name}   drop_rank1={drop_rank1}")
    print('='*60)

    rank_z = compute_group_rank(group_name)
    print(f"  monkeys with rank info: {len(rank_z)}")

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()

    # Exclude the subject monkey
    subject = info_df.loc[info_df['Note'].fillna('').str.contains('subject',
                          case=False), 'Name'].tolist()
    rank_z = rank_z.drop(index=[s for s in subject if s in rank_z.index],
                         errors='ignore')

    if drop_rank1:
        # The single highest z-score in this group = the alpha
        alpha = rank_z.idxmax()
        print(f"  dropping alpha: {alpha}  (z={rank_z[alpha]:+.2f})")
        rank_z = rank_z.drop(index=alpha)

    monkey_list = rank_z.index.tolist()
    print(f"  using {len(monkey_list)} stimulus monkeys: {monkey_list}")

    R, conditions, info, row_meta = build_rank_tensor(df, cfg, monkey_list)
    print(f"  neural tensor: {R.shape}  (n_monkeys, n_bins, n_neurons)")

    # Reorder rank_z to match the (sorted) condition order from binning
    rank_vec = np.array([rank_z[c] for c in conditions])

    R_z = zscore_neurons(R)
    beta, norms = fit_rank_axis(R_z, rank_vec)
    axis, chosen = define_axis(beta, norms, mode='peak')
    print(f"  axis chosen at: {chosen}, peak ||beta|| = {norms.max():.3f}")

    proj = project(R_z, axis)
    time_axis = np.arange(info['n_bins']) * cfg.bin_width

    tag = f"{group_name.replace(' ', '_')}{'_no_alpha' if drop_rank1 else ''}"
    plot_rank_projection(
        proj, conditions, rank_z, time_axis,
        title=f"{cfg.region} — {group_name} {'(no alpha)' if drop_rank1 else ''}: "
              f"projection on rank axis",
        save_path=os.path.join(PLOT_SAVE_DIR, f'{cfg.region}_projection_{tag}.png'),
    )
    plot_beta_norm(
        norms, time_axis,
        title=f"{cfg.region} — {group_name} {'(no alpha)' if drop_rank1 else ''}: "
              f"||beta_rank(t)||",
        save_path=os.path.join(PLOT_SAVE_DIR, f'{cfg.region}_beta_norm_{tag}.png'),
    )

    return dict(beta=beta, norms=norms, axis=axis, proj=proj,
                conditions=conditions, rank_z=rank_z, info=info)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrajectoryConfig(
        region='AMG',
        session=None,
        trial_averaged=True,
        peak_align=False,
        n_components=6,        # not actually used by TDR, kept for compatibility
        bin_width=0.050,
        min_epoch_duration=2.0,
    )
    cfg.validate()

    df = load_and_filter(cfg)

    groups_to_run = ['Zombies', 'Best Frans']
    results = {}
    for g in groups_to_run:
        results[(g, False)] = run_one_group(df, cfg, g, drop_rank1=False)
        results[(g, True)]  = run_one_group(df, cfg, g, drop_rank1=True)

    plt.show()
    return results


if __name__ == '__main__':
    main()
