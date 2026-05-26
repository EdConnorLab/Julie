"""
run_tdr_multi.py
================

Full Mante-style Targeted Dimensionality Reduction with multiple
predictors and PCA denoising.

Pipeline (Mante, Sussillo, Shenoy & Newsome 2013, Nature 503:78-84;
supplementary section 6.5-6.7):

    1. Build regressor table for each stimulus monkey, with predictors:
          dominance_z   : within-group z-scored David's score
          sociability_z : within-group z-scored behavioral PC2
          age_z         : within-group z-scored age
       All predictors are pooled across Zombies + Best Frans (N=14).

    2. Build a trial-averaged (n_cond, n_bins, n_neurons) neural tensor
       via binning_by_condition.build_matrix_by_condition, with each
       stimulus monkey as a condition.

    3. Z-score every neuron across (conditions x time).

    4. At each timepoint, fit MULTIPLE linear regression of z-scored
       firing rate on all predictors jointly. Each neuron gets one
       coefficient per predictor per timepoint.

    5. PCA-denoise. Fit PCA on the condition-averaged (n_cond*n_bins,
       n_neurons) matrix, keep top N_pca components, build the
       denoising projector D = V V^T (V is n_neurons x N_pca). Project
       each beta vector into the PC subspace: beta_denoised = D @ beta.

    6. For each predictor, find t_max where ||beta_denoised(t)|| is
       maximal, and use beta_denoised(t_max) as the fixed axis for
       that predictor.

    7. QR-orthogonalize the stack of fixed axes in a user-specified
       order. The first predictor keeps its original axis; subsequent
       predictors get only the component orthogonal to all earlier
       ones. Changing the order changes which predictor is "protected"
       and therefore which question the analysis answers.

    8. Project population activity onto each orthogonalized axis and
       plot over time, colored by that axis's predictor.

Sensitivity check: rerun the full pipeline after dropping the rank-1
alpha from each group, so we can see whether each axis survives the
alpha-male confound.

Notes
-----
- Only Zombies and Best Frans are used. Instigators matrices are still
  in progress and Stranger Things has no behavioral data at all.
- 70G, 79G, 144H (Instigators) would be excluded anyway because their
  David's scores are zero from missing data.
- 19J's official rank in monkeyinfo.csv is probably wrong, but we use
  David's score directly so the issue resolves itself.
- Imports core utilities from run_tdr_rank.py to avoid duplication.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter1d

from config import TrajectoryConfig
from data_loading import load_and_filter
from population_analysis.state_space.preprocessing import preprocess

# Reuse infrastructure from the univariate rank script.
# NOTE: zscore_neurons is intentionally NOT imported — TDR uses the
# Churchland soft-norm + mean-center pipeline via preprocess() instead.
from run_tdr_rank import (
    build_rank_tensor,
    plot_beta_norm,
    MONKEY_INFO_PATH,
)

# social_rank_analysis also needed for David's score + behavioral PCA
sys.path.insert(0,
                '/population/neural_trajectory')
from social_rank_analysis import (
    load_group_matrices,
    davids_score,
    behavioral_pca,
)


PLOT_SAVE_DIR = ("/home/connorlab/Documents/GitHub/Julie/Cortana/"
                 "analysis_results/tdr_multi")


# ---------------------------------------------------------------------------
# Regressor table
# ---------------------------------------------------------------------------
def build_regressor_table(groups=('Zombies', 'Best Frans'),
                          exclude_subject=True,
                          familiarity_map=None):
    """
    Returns a DataFrame indexed by monkey name with columns:
        group, dominance_z, sociability_z, age_z, sex, familiarity
    All z-scores are within-group. Sex is coded +1/-1. Familiarity
    is coded +1 (in-group) / -1 (out-group).

    Parameters
    ----------
    familiarity_map : dict, optional
        {group_name: float} mapping each group to a familiarity value.
        Default: {'Zombies': +1, 'Best Frans': -1} (in-group / out-group).

    Drops monkeys with no rank information (zero rows + zero columns in
    both agonism and submission matrices) and, by default, the subject
    monkey.
    """
    if familiarity_map is None:
        familiarity_map = {'Zombies': 1.0, 'Best Frans': -1.0}

    info_df = pd.read_csv(MONKEY_INFO_PATH)
    info_df['Name'] = info_df['Name'].astype(str).str.strip()
    info_idx = info_df.set_index('Name')

    # Behavioral PCA across all groups for PC2 (sociability).
    # We pass all three groups into behavioral_pca for a shared PC basis,
    # then subset to the groups we're actually analyzing.
    all_mats = {g: load_group_matrices(g)
                for g in ('Zombies', 'Best Frans', 'Instigators')}
    pca_scores, pca_obj, loadings = behavioral_pca(all_mats,
                                                   z_score_within_group=True)

    rows = []
    for g in groups:
        mats = all_mats[g]
        ago = mats['agonism']
        sub = mats['submission']
        combined = ago + sub.T                     # wins matrix: i dominated j
        ds = davids_score(combined)

        for m in ds.index:
            # Drop monkeys with no dominance data at all
            zero_ago = ((ago.loc[m].sum() == 0)
                        and (ago[m].sum() == 0))
            zero_sub = ((sub.loc[m].sum() == 0)
                        and (sub[m].sum() == 0))
            if zero_ago and zero_sub:
                continue

            if m not in info_idx.index:
                print(f"[warn] {m} in matrices but not in monkeyinfo.csv; skipping")
                continue

            age = info_idx.loc[m, 'Age']
            if pd.isna(age):
                print(f"[warn] {m} has no age; skipping")
                continue

            # Sex: look for 'Sex' column in monkeyinfo.csv
            sex_val = np.nan
            if 'Sex' in info_idx.columns:
                s = info_idx.loc[m, 'Sex']
                if isinstance(s, str):
                    s = s.strip().upper()
                    if s in ('M', 'MALE'):
                        sex_val = 1.0
                    elif s in ('F', 'FEMALE'):
                        sex_val = -1.0
            if pd.isna(sex_val):
                print(f"[warn] {m} has no valid sex; skipping")
                continue

            # Sociability = PC2 from the joint behavioral PCA
            if m in pca_scores.index:
                soc = pca_scores.loc[m, 'PC2']
            else:
                print(f"[warn] {m} not in PCA scores; skipping")
                continue

            rows.append({
                'name': m,
                'group': g,
                'dominance_raw': ds[m],
                'sociability_raw': soc,
                'age_raw': float(age),
                'sex': sex_val,
                'familiarity': familiarity_map.get(g, 0.0),
                'note': info_idx.loc[m, 'Note'] if 'Note' in info_idx.columns else '',
            })

    tbl = pd.DataFrame(rows).set_index('name')

    if exclude_subject:
        subject_mask = tbl['note'].fillna('').str.contains('subject', case=False)
        if subject_mask.any():
            dropped = tbl.index[subject_mask].tolist()
            print(f"Excluding subject monkey: {dropped}")
            tbl = tbl[~subject_mask]

    # Within-group z-scoring
    for col_raw, col_z in [('dominance_raw', 'dominance_z'),
                           ('sociability_raw', 'sociability_z'),
                           ('age_raw', 'age_z')]:
        parts = []
        for g, sub in tbl.groupby('group'):
            x = sub[col_raw]
            if x.std(ddof=0) == 0:
                z = x * 0.0
            else:
                z = (x - x.mean()) / x.std(ddof=0)
            parts.append(z)
        tbl[col_z] = pd.concat(parts)

    return tbl


def check_collinearity(tbl, predictors):
    """
    Print pairwise correlations AND Variance Inflation Factors (VIF).

    Pairwise correlation catches when two predictors are linearly related.
    VIF catches when one predictor is explained by ALL the others together
    (multiway collinearity), which pairwise correlation can miss.

    VIF for predictor j = 1 / (1 - R²_j), where R²_j is the R-squared
    from regressing predictor j on all other predictors.

    Rules of thumb:
        VIF < 5  : acceptable
        5 ≤ VIF < 10 : concerning — betas for this predictor are inflated
        VIF ≥ 10 : serious — predictor is nearly redundant with others

    Pairwise correlation thresholds:
        |r| < 0.5  : no concern
        0.5–0.7    : moderate — betas somewhat unstable but interpretable
        0.7–0.8    : concerning
        |r| > 0.8  : serious
    """
    X = tbl[predictors].values
    n_pred = len(predictors)

    # --- Pairwise correlations ---
    corr = tbl[predictors].corr()
    print("\nPredictor correlation matrix:")
    print(corr.round(2).to_string())
    off_diag = corr.where(~np.eye(n_pred, dtype=bool))
    max_abs = off_diag.abs().max().max()

    if max_abs > 0.8:
        print(f"  [WARN] Strong pairwise correlation (max |r| = {max_abs:.2f})")
    elif max_abs > 0.5:
        print(f"  [NOTE] Moderate pairwise correlation (max |r| = {max_abs:.2f})")
    else:
        print(f"  Pairwise correlations OK (max |r| = {max_abs:.2f})")

    # --- Variance Inflation Factors ---
    print("\nVariance Inflation Factors:")
    vif_warn = False
    for j in range(n_pred):
        # Regress predictor j on all others
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        others_aug = np.hstack([others, np.ones((X.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(others_aug, y, rcond=None)
        y_hat = others_aug @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else float('inf')

        flag = ''
        if vif >= 10:
            flag = '  *** SERIOUS'
            vif_warn = True
        elif vif >= 5:
            flag = '  ** CONCERNING'
            vif_warn = True
        print(f"  {predictors[j]:20s}  VIF = {vif:6.2f}  (R² = {r_sq:.2f}){flag}")

    if vif_warn:
        print("  [WARN] High VIF detected. Consider dropping or combining "
              "correlated predictors.")
    else:
        print("  All VIFs < 5: multicollinearity OK")


# ---------------------------------------------------------------------------
# Preprocessing (Churchland-style + temporal smoothing)
# ---------------------------------------------------------------------------
def churchland_preprocess_tensor(R_counts, info, cfg,
                                 soft_normalize=True, mean_center=True):
    """
    Full TDR-appropriate preprocessing of a (n_cond, n_bins, n_neurons)
    SPIKE COUNT tensor.

    Steps, in order:
      1. Gaussian temporal smoothing along the time axis (sigma = cfg.smoothing_sigma
         bins), applied to spike counts. Linear, so equivalent to smoothing rates.
      2. Flatten to (n_cond * n_bins, n_neurons) and hand off to preprocessing.preprocess,
         which does counts -> Hz, soft-normalize by (range + 5 Hz), and mean-center
         across conditions at each timepoint.
      3. Reshape back to (n_cond, n_bins, n_neurons).

    Parameters
    ----------
    R_counts : ndarray
        (n_cond, n_bins, n_neurons) spike count tensor from build_rank_tensor.
        NOTE: build_rank_tensor currently returns rates (counts / bin_width).
        We un-do that division here because preprocess() expects counts. See
        run_tdr() for the call site.
    info : dict from build_matrix_by_condition
    cfg  : TrajectoryConfig
    """
    n_cond, n_bins, n_neu = R_counts.shape

    # 1. Smooth along time. gaussian_filter1d on axis=1 treats each
    # (condition, neuron) cell as an independent 1-D time series.
    R_smooth = gaussian_filter1d(R_counts, sigma=cfg.smoothing_sigma,
                                 axis=1, mode='nearest')

    # 2. Flatten and call preprocess(). It expects an info dict with
    # n_bins, bin_width, n_conditions, trial_averaged. build_matrix_by_condition
    # already sets these; we pass them through.
    flat = R_smooth.reshape(n_cond * n_bins, n_neu)
    flat_pre = preprocess(flat, info, cfg,
                          soft_normalize=soft_normalize,
                          mean_center=mean_center)

    # 3. Reshape back
    return flat_pre.reshape(n_cond, n_bins, n_neu)



def fit_multi_axes(R_z, X):
    """
    At each timepoint, fit ordinary multiple linear regression of
    R_z[:, t, :] on predictors X, jointly across all predictors.

    Parameters
    ----------
    R_z : (n_cond, n_bins, n_neurons)
        Neuron-z-scored firing rates.
    X : (n_cond, n_predictors)
        Predictor matrix. Will be augmented with an intercept column
        internally.

    Returns
    -------
    betas : (n_predictors, n_bins, n_neurons)
        Coefficient for each predictor at each timepoint for each neuron.
        The intercept is computed internally but not returned.
    norms : (n_predictors, n_bins)
        L2 norm of each predictor's beta vector across neurons, per
        timepoint.
    """
    n_cond, n_bins, n_neu = R_z.shape
    n_pred = X.shape[1]

    # Augment with intercept column
    X_aug = np.hstack([X, np.ones((n_cond, 1))])          # (n_cond, n_pred + 1)

    # Solve for all timepoints and neurons at once.
    # Flatten the (n_bins, n_neurons) axes into a single column axis
    # so we can do one lstsq call per timepoint.
    betas = np.zeros((n_pred, n_bins, n_neu))
    for t in range(n_bins):
        Y = R_z[:, t, :]                                   # (n_cond, n_neu)
        # lstsq solves X_aug @ coef = Y in least-squares sense.
        # coef has shape (n_pred + 1, n_neu).
        coef, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
        betas[:, t, :] = coef[:n_pred]                     # drop intercept

    norms = np.linalg.norm(betas, axis=2)                  # (n_pred, n_bins)
    return betas, norms


def fit_multi_axes_single_trial(df, cfg, monkey_list, tbl, predictors,
                                info):
    """
    Per-neuron, single-trial regression matching Mante Eq. 1.

    Instead of regressing on the 14 condition-averaged population
    vectors (N=14 data-points), this function loops over neurons
    individually and regresses each neuron's firing rate at each
    timepoint against the predictor values, using ALL single trials
    (typically ~70-100 per neuron).  This is what Mante et al. did.

    The resulting beta vectors are arranged into population vectors
    of length N_neurons so they can be denoised and projected exactly
    as in the condition-averaged pipeline.

    Parameters
    ----------
    df : DataFrame
        Raw data from load_and_filter (one row per neuron × trial).
    cfg : TrajectoryConfig
    monkey_list : list of str
        Stimulus monkey names that define the conditions (must match
        the monkeys in tbl).
    tbl : DataFrame
        Regressor table indexed by monkey name, with columns for
        each predictor.
    predictors : list of str
        Column names in tbl to use as predictors.
    info : dict
        Output from build_rank_tensor.  Must contain 'neuron_ids'
        (list of '{session}__{NeuronID}' strings) and 'n_bins'.

    Returns
    -------
    betas : ndarray, shape (n_pred, n_bins, n_neurons)
    norms : ndarray, shape (n_pred, n_bins)
    n_trials_per_neuron : list of int
        Number of usable trials for each neuron (for diagnostics).
    r2 : ndarray, shape (n_neurons, n_bins)
        R² of the full model (all predictors) at each timepoint for
        each neuron.  Useful for diagnosing fit quality.
    """
    n_bins = info['n_bins']
    bin_width = cfg.bin_width
    analysis_window = n_bins * bin_width
    n_pred = len(predictors)
    if 'neuron_ids' not in info:
        raise KeyError("info dict must contain 'neuron_ids' — a list of "
                        "'{session}__{NeuronID}' strings matching the "
                        "neuron ordering in the condition-averaged tensor.")
    neuron_id_strings = info['neuron_ids']
    n_neurons = len(neuron_id_strings)

    # Predictor lookup:  monkey_name -> (n_pred,) array
    pred_lookup = {}
    for m in monkey_list:
        if m in tbl.index:
            pred_lookup[m] = tbl.loc[m, predictors].values.astype(float)

    # Pre-filter df to relevant monkeys for speed
    df_filt = df[df['MonkeyName'].isin(monkey_list)].copy()

    betas = np.zeros((n_pred, n_bins, n_neurons))
    r2 = np.full((n_neurons, n_bins), np.nan)
    n_trials_per_neuron = []

    for neu_idx, nid_str in enumerate(neuron_id_strings):
        session, nid = nid_str.split('__', 1)

        # All rows for this neuron that show a relevant monkey
        neu_df = df_filt[(df_filt['session'] == session)
                         & (df_filt['NeuronID'] == nid)]

        # Collect single-trial firing rates and matching predictors
        trial_rates = []          # each entry: (n_bins,) count array
        trial_preds = []          # each entry: (n_pred,) predictor array

        for tf, t_group in neu_df.groupby('TaskField'):
            row = t_group.iloc[0]
            monkey_name = row['MonkeyName']
            if monkey_name not in pred_lookup:
                continue

            # Bin spikes for this single trial
            rel = np.asarray(row['SpikeTimes']) - row['EpochStartStop'][0]
            rel = rel[(rel >= 0) & (rel < analysis_window)]
            counts = np.zeros(n_bins)
            bins = np.floor(rel / bin_width).astype(int)
            bins = bins[bins < n_bins]
            for b in bins:
                counts[b] += 1
            # Convert to rates (Hz)
            rates = counts / bin_width

            trial_rates.append(rates)
            trial_preds.append(pred_lookup[monkey_name])

        n_trials = len(trial_rates)
        n_trials_per_neuron.append(n_trials)

        # Need more trials than parameters (predictors + intercept)
        if n_trials < n_pred + 2:
            continue

        trial_rates = np.array(trial_rates)   # (n_trials, n_bins)
        trial_preds = np.array(trial_preds)   # (n_trials, n_pred)

        # Z-score per neuron across ALL trials and times (Mante §6.3)
        mu = trial_rates.mean()
        sigma = trial_rates.std()
        if sigma == 0:
            continue
        trial_rates = (trial_rates - mu) / sigma

        # Augment predictor matrix with intercept column
        X_aug = np.hstack([trial_preds,
                           np.ones((n_trials, 1))])      # (n_trials, n_pred+1)

        # Regression at each timepoint
        for t in range(n_bins):
            y = trial_rates[:, t]                         # (n_trials,)
            coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            betas[:, t, neu_idx] = coef[:n_pred]

            # R² = 1 - SS_res / SS_tot
            y_hat = X_aug @ coef
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[neu_idx, t] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    norms = np.linalg.norm(betas, axis=2)                 # (n_pred, n_bins)

    # Diagnostics
    arr = np.array(n_trials_per_neuron)
    print(f"  Single-trial regression: {n_neurons} neurons")
    print(f"    trials/neuron: min={arr.min()}, median={np.median(arr):.0f}, "
          f"max={arr.max()}, mean={arr.mean():.1f}")
    print(f"    neurons with 0 usable trials: {(arr == 0).sum()}")

    # R² diagnostics
    valid_r2 = r2[~np.isnan(r2)]
    mean_r2_per_neuron = np.nanmean(r2, axis=1)
    mean_r2_per_time = np.nanmean(r2, axis=0)
    print(f"  R² diagnostics (full model, all predictors):")
    print(f"    overall: mean={valid_r2.mean():.4f}, "
          f"median={np.median(valid_r2):.4f}, "
          f"max={valid_r2.max():.4f}")
    print(f"    per-neuron mean R²: "
          f"min={mean_r2_per_neuron[~np.isnan(mean_r2_per_neuron)].min():.4f}, "
          f"median={np.nanmedian(mean_r2_per_neuron):.4f}, "
          f"max={np.nanmax(mean_r2_per_neuron):.4f}")
    print(f"    neurons with mean R² > 0.05: "
          f"{(mean_r2_per_neuron > 0.05).sum()} / {n_neurons}")
    print(f"    neurons with mean R² > 0.10: "
          f"{(mean_r2_per_neuron > 0.10).sum()} / {n_neurons}")
    print(f"    peak mean R² across time at t={time_axis_from_bins(mean_r2_per_time, cfg)}")

    return betas, norms, n_trials_per_neuron, r2


def time_axis_from_bins(arr, cfg):
    """Helper: find the peak of an array and return its time in seconds."""
    pk = np.argmax(arr)
    return f"{pk * cfg.bin_width:.2f}s (R²={arr[pk]:.4f})"


# ---------------------------------------------------------------------------
# PCA denoising (Mante supplement 6.7)
# ---------------------------------------------------------------------------
def pca_denoising_projector(R_z, n_pca):
    """
    Fit PCA on the condition-averaged data matrix and return
    D = V V^T, the projector onto the top-n_pca PC subspace.

    Input R_z has shape (n_cond, n_bins, n_neurons); we flatten to
    (n_cond*n_bins, n_neurons) for PCA, same way binning_by_condition's
    output feeds into run_pca.

    Returns
    -------
    D : (n_neurons, n_neurons) projector matrix
    pca : the fitted sklearn PCA object (for inspection)
    """
    n_cond, n_bins, n_neu = R_z.shape
    flat = R_z.reshape(-1, n_neu)

    # Full-rank PCA, then we slice top-k components
    pca = PCA(n_components=min(n_pca, flat.shape[0], flat.shape[1])).fit(flat)
    V = pca.components_.T                                  # (n_neurons, n_pca)
    D = V @ V.T                                            # (n_neurons, n_neurons)
    return D, pca


def denoise_betas(betas, D):
    """
    Project each beta vector onto the PC subspace.

    betas : (n_pred, n_bins, n_neurons)
    D     : (n_neurons, n_neurons)
    Returns (n_pred, n_bins, n_neurons)
    """
    # For each (pred, t), out[:] = D @ betas[pred, t, :]
    # Contract last axis of betas with D
    return betas @ D.T                                     # D symmetric but explicit


# ---------------------------------------------------------------------------
# Fixed axes + QR orthogonalization (Mante supplement 6.7)
# ---------------------------------------------------------------------------
def fixed_axes_from_peak(betas_denoised):
    """
    For each predictor, pick the timepoint of maximum ||beta_denoised|| and
    return the beta vector at that timepoint as the fixed axis.

    betas_denoised : (n_pred, n_bins, n_neurons)
    Returns
    -------
    axes : (n_pred, n_neurons)
    peak_bins : list of int, one per predictor
    """
    norms = np.linalg.norm(betas_denoised, axis=2)          # (n_pred, n_bins)
    peak_bins = norms.argmax(axis=1).tolist()
    axes = np.stack([betas_denoised[p, peak_bins[p], :]
                     for p in range(betas_denoised.shape[0])], axis=0)
    return axes, peak_bins


def qr_orthogonalize(axes, order):
    """
    QR-orthogonalize a stack of axes in a specified order. The first
    axis in `order` is kept unchanged; each subsequent axis is replaced
    by its component orthogonal to all earlier axes.

    axes : (n_pred, n_neurons)
    order : list of int indices into axes, specifying the orthogonalization
            priority. The predictor at order[0] is "protected."

    Returns
    -------
    orth_axes : (n_pred, n_neurons)
        Unit-length vectors, returned in the ORIGINAL predictor order
        (not in `order` order), so indexing by predictor index still
        works downstream.
    """
    n_pred, n_neu = axes.shape
    # Gather axes in the requested order, as column vectors
    A = axes[order].T                                       # (n_neurons, n_pred)
    Q, R = np.linalg.qr(A)                                  # Q: (n_neurons, n_pred)

    # Q's columns are orthonormal; column i corresponds to order[i].
    # Put back into original predictor order.
    orth = np.zeros_like(axes)
    for i, p in enumerate(order):
        # Preserve sign: QR can flip; use sign of the diagonal of R
        sign = np.sign(R[i, i]) if R[i, i] != 0 else 1.0
        orth[p] = sign * Q[:, i]
    return orth


# ---------------------------------------------------------------------------
# Projection + plotting
# ---------------------------------------------------------------------------
def project_population(R_z, axes):
    """
    Project population activity onto each axis.

    R_z : (n_cond, n_bins, n_neurons)
    axes : (n_pred, n_neurons), unit length

    Returns (n_pred, n_cond, n_bins)
    """
    # R_z @ axes.T -> (n_cond, n_bins, n_pred), then move pred axis to front
    projected = R_z @ axes.T
    return np.moveaxis(projected, -1, 0)


def plot_projection(proj_for_this_axis, conditions, predictor_values,
                    time_axis, predictor_name, title='', save_path=None):
    """
    proj_for_this_axis : (n_cond, n_bins)
    conditions         : list of monkey names
    predictor_values   : array of the predictor values in condition order
                         (used to color the traces)
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = np.asarray(predictor_values, dtype=float)
    norm = plt.Normalize(vals.min(), vals.max())
    cmap = plt.get_cmap('coolwarm')
    for i, c in enumerate(conditions):
        ax.plot(time_axis, proj_for_this_axis[i],
                color=cmap(norm(vals[i])),
                lw=2, label=f'{c} ({vals[i]:+.2f})')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel(f'Projection on {predictor_name} axis (a.u.)')
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{predictor_name} (z-scored)')

    ax.legend(fontsize=7, loc='best', ncol=2)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_projection_grouped(proj_for_this_axis, conditions, predictor_values,
                            time_axis, predictor_name, title='',
                            save_path=None, n_groups=3):
    """
    Plot MEAN trajectories grouped by predictor value.

    For binary predictors (only 2 unique values): plots one mean
    trajectory per level with SEM shading.

    For continuous predictors: divides conditions into n_groups
    quantile bins, plots the mean trajectory per bin with SEM shading.

    Parameters
    ----------
    proj_for_this_axis : (n_cond, n_bins)
    conditions         : list of monkey names
    predictor_values   : array of the predictor values in condition order
    time_axis          : (n_bins,) time in seconds
    predictor_name     : str
    title              : str
    save_path          : str or None
    n_groups           : int, number of bins for continuous predictors
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = np.asarray(predictor_values, dtype=float)
    proj = np.asarray(proj_for_this_axis)

    unique_vals = np.unique(vals)
    is_binary = len(unique_vals) == 2

    if is_binary:
        # Binary: one group per level
        group_labels = []
        for uv in sorted(unique_vals):
            mask = vals == uv
            group_labels.append((mask, f'{predictor_name} = {uv:+.1f}'))
    else:
        # Continuous: split into quantile groups
        # Use percentile boundaries so groups are as balanced as possible
        boundaries = np.linspace(0, 100, n_groups + 1)
        percentiles = np.percentile(vals, boundaries)
        # Make sure edges include min/max
        percentiles[0] = vals.min() - 1e-10
        percentiles[-1] = vals.max() + 1e-10

        group_labels = []
        for g in range(n_groups):
            lo, hi = percentiles[g], percentiles[g + 1]
            if g == n_groups - 1:
                mask = (vals >= lo) & (vals <= hi)
            else:
                mask = (vals >= lo) & (vals < hi)

            if mask.sum() == 0:
                continue

            member_vals = vals[mask]
            label = (f'{predictor_name} [{member_vals.min():+.1f}, '
                     f'{member_vals.max():+.1f}] (n={mask.sum()})')
            group_labels.append((mask, label))

    # Colors: always-visible palette for grouped trajectories.
    # coolwarm has white at center which is invisible on white background,
    # so we use handpicked colors: blue → gray → red for low → mid → high.
    _GROUP_PALETTES = {
        1: ['#6b7280'],
        2: ['#2563eb', '#dc2626'],
        3: ['#2563eb', '#6b7280', '#dc2626'],
        4: ['#2563eb', '#60a5fa', '#f87171', '#dc2626'],
        5: ['#1e40af', '#60a5fa', '#6b7280', '#f87171', '#b91c1c'],
    }
    n_actual = len(group_labels)
    if n_actual in _GROUP_PALETTES:
        colors = _GROUP_PALETTES[n_actual]
    else:
        # Fallback: use tab10 for larger group counts
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i / max(n_actual - 1, 1)) for i in range(n_actual)]

    for (mask, label), color in zip(group_labels, colors):
        traces = proj[mask]                        # (n_in_group, n_bins)
        mean_trace = traces.mean(axis=0)
        sem_trace = traces.std(axis=0) / np.sqrt(traces.shape[0])

        ax.plot(time_axis, mean_trace, color=color, lw=2.5, label=label)
        ax.fill_between(time_axis,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=color, alpha=0.2)

    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel(f'Projection on {predictor_name} axis (a.u.)')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Summary / paper-quality plots
# ---------------------------------------------------------------------------
def plot_state_space_2d(result, axis_pairs=None, color_by=None,
                        save_dir=None, tag='', cfg=None):
    """
    2-D state-space trajectory plots (Mante Fig. 2 equivalent).

    Each condition (stimulus monkey) traces a path through the space
    defined by two TDR axes. Colored by a chosen predictor.

    Parameters
    ----------
    result : dict returned by run_tdr
    axis_pairs : list of (str, str) tuples, or None for all pairs
    color_by : str or None. If None, colors by the first predictor.
    """
    proj = result['projection']           # (n_pred, n_cond, n_bins)
    predictors = result['predictors']
    conditions = result['conditions']
    tbl = result['tbl']
    time_axis = result['time_axis']
    pred_idx = {n: i for i, n in enumerate(predictors)}

    if color_by is None:
        color_by = predictors[0]
    color_vals = tbl[color_by].values.astype(float)
    norm = plt.Normalize(color_vals.min(), color_vals.max())
    cmap = plt.get_cmap('coolwarm')

    if axis_pairs is None:
        # All unique pairs
        axis_pairs = [(predictors[i], predictors[j])
                      for i in range(len(predictors))
                      for j in range(i + 1, len(predictors))]

    figs = []
    for ax_x_name, ax_y_name in axis_pairs:
        ix, iy = pred_idx[ax_x_name], pred_idx[ax_y_name]
        fig, ax = plt.subplots(figsize=(7, 6))

        for c_idx, c_name in enumerate(conditions):
            color = cmap(norm(color_vals[c_idx]))
            traj_x = proj[ix, c_idx, :]
            traj_y = proj[iy, c_idx, :]

            # Trajectory line
            ax.plot(traj_x, traj_y, color=color, lw=1.5, alpha=0.7)
            # Start point (circle)
            ax.plot(traj_x[0], traj_y[0], 'o', color=color,
                    ms=8, zorder=5)
            # End point (triangle)
            ax.plot(traj_x[-1], traj_y[-1], '^', color=color,
                    ms=7, zorder=5)
            # Label
            ax.annotate(c_name, (traj_x[0], traj_y[0]),
                        fontsize=6, fontweight='bold', color=color,
                        xytext=(4, 4), textcoords='offset points')
            # Time markers every 500ms
            dt = time_axis[1] - time_axis[0]
            step = max(1, int(0.5 / dt))
            for t_idx in range(step, len(time_axis) - 1, step):
                ax.plot(traj_x[t_idx], traj_y[t_idx], '.',
                        color=color, ms=4, zorder=4)

        ax.axhline(0, color='k', lw=0.3, alpha=0.3)
        ax.axvline(0, color='k', lw=0.3, alpha=0.3)
        ax.set_xlabel(f'{ax_x_name} axis')
        ax.set_ylabel(f'{ax_y_name} axis')
        ax.set_title(f'{tag}: state space (colored by {color_by})\n'
                     f'○ = start, △ = end, dots = 500 ms')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(color_by)

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            region = cfg.region if cfg else ''
            fname = f'{region}_statespace_{ax_x_name}_vs_{ax_y_name}_by_{color_by}.png'
            fig.savefig(os.path.join(save_dir, fname), dpi=200)
        figs.append(fig)
    return figs


def plot_state_space_3d(result, axes_names=None, color_by=None,
                        save_dir=None, tag='', cfg=None):
    """
    3-D state-space trajectory plot.

    axes_names : list of 3 predictor names. Defaults to first 3.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    proj = result['projection']
    predictors = result['predictors']
    conditions = result['conditions']
    tbl = result['tbl']
    pred_idx = {n: i for i, n in enumerate(predictors)}

    if axes_names is None:
        axes_names = predictors[:3]
    if len(axes_names) < 3:
        return []
    ix, iy, iz = [pred_idx[n] for n in axes_names]

    if color_by is None:
        color_by = predictors[0]
    color_vals = tbl[color_by].values.astype(float)
    norm = plt.Normalize(color_vals.min(), color_vals.max())
    cmap = plt.get_cmap('coolwarm')

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for c_idx, c_name in enumerate(conditions):
        color = cmap(norm(color_vals[c_idx]))
        tx = proj[ix, c_idx, :]
        ty = proj[iy, c_idx, :]
        tz = proj[iz, c_idx, :]

        ax.plot(tx, ty, tz, color=color, lw=1.5, alpha=0.7)
        ax.scatter(tx[0], ty[0], tz[0], color=color, s=50,
                   marker='o', zorder=5)
        ax.scatter(tx[-1], ty[-1], tz[-1], color=color, s=40,
                   marker='^', zorder=5)
        # Label each trajectory with monkey name
        ax.text(tx[0], ty[0], tz[0], f' {c_name}', color=color,
                fontsize=6, fontweight='bold', zorder=6)

    ax.set_xlabel(axes_names[0])
    ax.set_ylabel(axes_names[1])
    ax.set_zlabel(axes_names[2])
    ax.set_title(f'{tag}: 3D state space (colored by {color_by})')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label=color_by)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fname = f'{region}_statespace_3d_by_{color_by}.png'
        fig.savefig(os.path.join(save_dir, fname), dpi=200)
    return [fig]


def plot_demixing_matrix(result, save_dir=None, tag='', cfg=None,
                         n_groups=3):
    """
    Cross-projection / demixing matrix (rows=axes, cols=color-by).

    Diagonal: projection onto axis V colored by predictor V.
              Should show clean separation.
    Off-diagonal: projection onto axis V colored by predictor W.
                  Should show overlapping traces if demixing is good.
    """
    proj = result['projection']          # (n_pred, n_cond, n_bins)
    predictors = result['predictors']
    tbl = result['tbl']
    time_axis = result['time_axis']
    n_pred = len(predictors)

    fig, axes = plt.subplots(n_pred, n_pred,
                             figsize=(3.5 * n_pred, 2.8 * n_pred),
                             sharex=True)

    _GROUP_PALETTES = {
        1: ['#6b7280'],
        2: ['#2563eb', '#dc2626'],
        3: ['#2563eb', '#6b7280', '#dc2626'],
        4: ['#2563eb', '#60a5fa', '#f87171', '#dc2626'],
        5: ['#1e40af', '#60a5fa', '#6b7280', '#f87171', '#b91c1c'],
    }

    for row in range(n_pred):       # axis projected onto
        for col in range(n_pred):   # colored by this predictor
            ax = axes[row, col]
            vals = tbl[predictors[col]].values.astype(float)
            traces = proj[row]       # (n_cond, n_bins)

            unique_vals = np.unique(vals)
            is_binary = len(unique_vals) == 2

            if is_binary:
                groups = []
                for uv in sorted(unique_vals):
                    mask = vals == uv
                    groups.append((mask, f'{uv:+.1f}'))
            else:
                boundaries = np.linspace(0, 100, n_groups + 1)
                percentiles = np.percentile(vals, boundaries)
                percentiles[0] = vals.min() - 1e-10
                percentiles[-1] = vals.max() + 1e-10
                groups = []
                for g in range(n_groups):
                    lo, hi = percentiles[g], percentiles[g + 1]
                    if g == n_groups - 1:
                        mask = (vals >= lo) & (vals <= hi)
                    else:
                        mask = (vals >= lo) & (vals < hi)
                    if mask.sum() > 0:
                        groups.append((mask, f'{g+1}'))

            n_g = len(groups)
            colors = _GROUP_PALETTES.get(n_g,
                [plt.get_cmap('tab10')(i / max(n_g - 1, 1))
                 for i in range(n_g)])

            for (mask, lbl), color in zip(groups, colors):
                grp_traces = traces[mask]
                mean_t = grp_traces.mean(axis=0)
                sem_t = grp_traces.std(axis=0) / np.sqrt(grp_traces.shape[0])
                ax.plot(time_axis, mean_t, color=color, lw=1.8)
                ax.fill_between(time_axis, mean_t - sem_t, mean_t + sem_t,
                                color=color, alpha=0.15)

            ax.axhline(0, color='k', lw=0.3, alpha=0.3)

            # Highlight diagonal
            if row == col:
                ax.set_facecolor('#f0f7ff')

            if row == 0:
                ax.set_title(f'color: {predictors[col]}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{predictors[row]}\naxis', fontsize=9)
            if row == n_pred - 1:
                ax.set_xlabel('Time (s)', fontsize=8)

            ax.tick_params(labelsize=7)

    fig.suptitle(f'{tag}: Demixing matrix\n'
                 f'(diagonal = matched axis & color → should separate)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir, f'{region}_demixing_matrix.png'),
                    dpi=200, bbox_inches='tight')
    return fig


def plot_beta_norms_combined(result, save_dir=None, tag='', cfg=None,
                             smooth_sigma=2):
    """
    All predictors' beta norms on one plot, with peak times marked.
    """
    betas_den = result['betas_denoised']   # (n_pred, n_bins, n_neurons)
    predictors = result['predictors']
    time_axis = result['time_axis']
    peak_bins = result['peak_bins']
    n_pred = len(predictors)

    norms = np.linalg.norm(betas_den, axis=2)   # (n_pred, n_bins)

    _COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
               '#0891b2', '#be185d', '#4f46e5']

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in range(n_pred):
        y = gaussian_filter1d(norms[p], sigma=smooth_sigma)
        color = _COLORS[p % len(_COLORS)]
        ax.plot(time_axis, y, color=color, lw=2.5, label=predictors[p])
        # Mark peak
        pk = peak_bins[p]
        ax.plot(time_axis[pk], y[pk], 'v', color=color, ms=10, zorder=5)

    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('||β(t)|| (denoised)')
    ax.set_title(f'{tag}: Beta norms — all predictors')
    ax.legend(fontsize=9)
    ax.axhline(0, color='k', lw=0.3, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir, f'{region}_beta_norms_combined.png'),
                    dpi=200)
    return fig


def plot_beta_norms_subplots(result, save_dir=None, tag='', cfg=None,
                             smooth_sigma=2):
    """
    Each predictor's beta norm as a separate subplot in one figure.
    Shares x-axis, shows peak time as a vertical dashed line.
    """
    betas_den = result['betas_denoised']
    predictors = result['predictors']
    time_axis = result['time_axis']
    peak_bins = result['peak_bins']
    n_pred = len(predictors)

    norms = np.linalg.norm(betas_den, axis=2)   # (n_pred, n_bins)

    COLOR = '#2563eb'

    fig, axes = plt.subplots(n_pred, 1, figsize=(9, 2.5 * n_pred),
                             sharex=True)
    if n_pred == 1:
        axes = [axes]

    for p, ax in enumerate(axes):
        y = gaussian_filter1d(norms[p], sigma=smooth_sigma)
        pk = peak_bins[p]

        ax.plot(time_axis, y, color=COLOR, lw=2.5)
        ax.fill_between(time_axis, 0, y, color=COLOR, alpha=0.12)

        # Peak dashed line
        ax.axvline(time_axis[pk], color=COLOR, ls='--', lw=1, alpha=0.6)

        # Text annotation (no marker)
        ax.text(0.98, 0.92,
                f'peak: {time_axis[pk]:.2f}s,  ||β|| = {norms[p, pk]:.3f}',
                transform=ax.transAxes, fontsize=9, color=COLOR,
                ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor=COLOR,
                          alpha=0.8, boxstyle='round,pad=0.3'))

        # Y-axis padding: 20% above max
        y_max = y.max()
        ax.set_ylim(0, y_max * 1.25)

        ax.set_ylabel(f'||β|| {predictors[p]}', fontsize=9)
        ax.axhline(0, color='k', lw=0.3, alpha=0.3)

    axes[-1].set_xlabel('Time from trial start (s)')
    fig.suptitle(f'{tag}: Beta norms (denoised)', fontsize=11, y=1.01)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir, f'{region}_beta_norms_subplots.png'),
                    dpi=200, bbox_inches='tight')
    return fig


def plot_axis_composition(result, save_dir=None, tag='', cfg=None):
    """
    How each task-related axis projects onto the PCs
    (Mante Extended Data Fig. 4c equivalent).

    Shows which PCs contribute to each TDR axis. If axes span
    multiple PCs, it means the social variables are represented
    by distributed patterns (mixed selectivity), not single PCs.
    """
    orth_axes = result['orth_axes']       # (n_pred, n_neurons)
    pca = result['pca']
    predictors = result['predictors']
    n_pred = len(predictors)
    n_pcs = pca.n_components_

    # Project each axis onto each PC
    # pca.components_ is (n_pcs, n_neurons)
    V = pca.components_                   # (n_pcs, n_neurons)
    projections = orth_axes @ V.T         # (n_pred, n_pcs)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, n_pcs + 1)
    width = 0.8 / n_pred

    _COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
               '#0891b2', '#be185d', '#4f46e5']

    for p in range(n_pred):
        offset = (p - n_pred / 2 + 0.5) * width
        color = _COLORS[p % len(_COLORS)]
        ax.bar(x + offset, projections[p] ** 2, width=width,
               color=color, alpha=0.8, label=predictors[p])

    ax.set_xlabel('Principal component')
    ax.set_ylabel('Projection² (axis onto PC)')
    ax.set_title(f'{tag}: Axis composition — which PCs form each axis')
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        region = cfg.region if cfg else ''
        fig.savefig(os.path.join(save_dir, f'{region}_axis_composition.png'),
                    dpi=200)
    return fig


# ---------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------
def run_tdr(df, cfg, tbl, predictors, qr_order_names, n_pca_denoise,
            tag=''):
    """
    df : neural DataFrame from load_and_filter
    cfg : TrajectoryConfig
    tbl : regressor table (DataFrame indexed by monkey name)
    predictors : list of column names in tbl to use as predictors
    qr_order_names : list of predictor names specifying QR priority order.
                     The first name is "protected" (axis stays unchanged).
    n_pca_denoise : number of PCs to keep for Mante-style denoising
    tag : string used in plot filenames
    """
    print(f"\n{'='*60}\nTDR run: {tag}\n{'='*60}")
    print(f"  predictors: {predictors}")
    print(f"  QR order  : {qr_order_names}")
    print(f"  N_pca     : {n_pca_denoise}")
    print(f"  n_monkeys : {len(tbl)}")
    print(tbl[predictors].round(2).to_string())

    check_collinearity(tbl, predictors)

    monkey_list = tbl.index.tolist()
    R_rates, conditions, info, row_meta = build_rank_tensor(df, cfg, monkey_list)
    print(f"  neural tensor: {R_rates.shape}  (n_monkeys, n_bins, n_neurons)")

    # Reorder tbl to match the (sorted) condition order from binning
    tbl_ord = tbl.loc[conditions]
    X = tbl_ord[predictors].values                          # (n_cond, n_pred)

    # build_rank_tensor returns rates (Hz), but preprocess() expects counts.
    # Undo the counts -> Hz conversion so churchland_preprocess_tensor can
    # smooth, then re-do it inside preprocess().
    R_counts = R_rates * cfg.bin_width
    R_z = churchland_preprocess_tensor(R_counts, info, cfg,
                                       soft_normalize=True, mean_center=True)

    # Scree diagnostic — full PCA on the preprocessed tensor
    full_pca = PCA().fit(R_z.reshape(-1, R_z.shape[-1]))
    cum = np.cumsum(full_pca.explained_variance_ratio_)
    print(f"  scree: PC1={full_pca.explained_variance_ratio_[0]:.1%}, "
          f"PC8={cum[7]:.1%}, PC20={cum[min(19, len(cum)-1)]:.1%}, "
          f"80% at PC{np.argmax(cum >= 0.80) + 1}")

    # Step 1: per-neuron, SINGLE-TRIAL regression (Mante Eq. 1)
    # This is the key difference from the old code, which regressed on
    # condition-averaged data (N=14 data points). Here each neuron uses
    # all of its individual trials (~70-100 data points).
    betas_raw, norms_raw, trial_counts, r2 = fit_multi_axes_single_trial(
        df, cfg, conditions, tbl_ord, predictors, info)
    print(f"  raw beta shape: {betas_raw.shape}")

    # Step 2: PCA denoising
    D, pca = pca_denoising_projector(R_z, n_pca=n_pca_denoise)
    print(f"  PCA: kept {n_pca_denoise} PCs, "
          f"cum var = {pca.explained_variance_ratio_[:n_pca_denoise].sum():.1%}")
    betas_den = denoise_betas(betas_raw, D)
    norms_den = np.linalg.norm(betas_den, axis=2)           # (n_pred, n_bins)

    # Step 3: fixed axes at peak time
    fixed_axes, peak_bins = fixed_axes_from_peak(betas_den)
    for p, pname in enumerate(predictors):
        print(f"  peak for {pname}: t_bin = {peak_bins[p]} "
              f"(t = {peak_bins[p] * cfg.bin_width:.2f} s), "
              f"||beta|| = {norms_den[p, peak_bins[p]]:.3f}")

    # Step 4: QR orthogonalization in user-specified order
    name_to_idx = {n: i for i, n in enumerate(predictors)}
    order = [name_to_idx[n] for n in qr_order_names]
    orth_axes = qr_orthogonalize(fixed_axes, order)

    # Step 5: project population activity onto orthogonalized axes
    proj = project_population(R_z, orth_axes)               # (n_pred, n_cond, n_bins)

    # Step 6: plots
    time_axis = np.arange(info['n_bins']) * cfg.bin_width
    save_dir = os.path.join(PLOT_SAVE_DIR, tag)

    for p, pname in enumerate(predictors):
        plot_projection(
            proj[p], conditions, X[:, p], time_axis,
            predictor_name=pname,
            title=f"{tag}: projection on {pname} axis",
            save_path=os.path.join(save_dir, f'{cfg.region}_projection_{pname}.png'),
        )
        plot_projection_grouped(
            proj[p], conditions, X[:, p], time_axis,
            predictor_name=pname,
            title=f"{tag}: grouped projection on {pname} axis",
            save_path=os.path.join(save_dir, f'{cfg.region}_projection_grouped_{pname}.png'),
            n_groups=3,
        )
        plot_beta_norm(
            gaussian_filter1d(norms_den[p], sigma=2), time_axis,
            title=f"{tag}: ||beta_{pname}(t)|| (denoised)",
            save_path=os.path.join(save_dir, f'{cfg.region}_beta_norm_{pname}.png'),
        )

    # Step 7: summary plots
    result = dict(
        tbl=tbl_ord, predictors=predictors, conditions=conditions,
        R_z=R_z, betas_raw=betas_raw, betas_denoised=betas_den,
        fixed_axes=fixed_axes, orth_axes=orth_axes, peak_bins=peak_bins,
        projection=proj, time_axis=time_axis, pca=pca, info=info,
        trial_counts=trial_counts, r2=r2,
    )

    plot_beta_norms_combined(result, save_dir=save_dir, tag=tag, cfg=cfg)

    plot_beta_norms_subplots(result, save_dir=save_dir, tag=tag, cfg=cfg)

    plot_demixing_matrix(result, save_dir=save_dir, tag=tag, cfg=cfg,
                         n_groups=3)

    plot_axis_composition(result, save_dir=save_dir, tag=tag, cfg=cfg)

    # State-space: 2D for all pairs, colored by each predictor
    for color_pred in predictors:
        plot_state_space_2d(result, color_by=color_pred,
                            save_dir=save_dir, tag=tag, cfg=cfg)

    # State-space: 3D for first 3 predictors
    if len(predictors) >= 3:
        plot_state_space_3d(result, color_by=predictors[0],
                            save_dir=save_dir, tag=tag, cfg=cfg)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrajectoryConfig(
        region='AMG',
        session=None,
        trial_averaged=True,
        peak_align=False,
        n_components=6,          # not used for TDR itself
        bin_width=0.050,
        min_epoch_duration=2.0,
    )
    cfg.validate()

    # --------- user settings ----------
    # PREDICTORS = ['sex', 'age_z', 'sociability_z', 'dominance_z', 'familiarity']
    PREDICTORS = ['familiarity', 'age_z', 'sociability_z', 'dominance_z']
    # QR ORDER — change this to answer different questions.
    # The first name is "protected": its axis stays unchanged, and
    # subsequent axes are orthogonalized against everything earlier.
    #
    #   ['dominance_z', 'sociability_z', 'age_z', 'sex', 'familiarity']
    #       -> dominance axis is primary; everything else is represented
    #          only by its component orthogonal to all earlier axes.
    #
    # Try different orderings to test robustness. For example, put
    # familiarity first to ask "is there rank/sociability encoding
    # beyond what group membership already explains?"
    QR_ORDER = ['familiarity', 'age_z', 'sociability_z', 'dominance_z']

    N_PCA_DENOISE = 12          # Mante uses 12; with 14 conditions the
                                 # condition-related subspace is at most 13-D,
                                 # so values above ~12 add noise, not signal.
                                 # Try 6/8/10/12 for sensitivity.
    # ----------------------------------

    df = load_and_filter(cfg)
    tbl_full = build_regressor_table(groups=('Zombies', 'Best Frans'),
                                     exclude_subject=True)

    # Main run
    run_tdr(df, cfg, tbl_full, PREDICTORS, QR_ORDER, N_PCA_DENOISE,
            tag='full')

    # # Sensitivity: drop the alpha (top dominance_z) in each group
    # alphas = (tbl_full.groupby('group')['dominance_z']
    #           .idxmax().tolist())
    # print(f"\nDropping alphas for sensitivity rerun: {alphas}")
    # tbl_no_alpha = tbl_full.drop(index=alphas)
    # run_tdr(df, cfg, tbl_no_alpha, PREDICTORS, QR_ORDER, N_PCA_DENOISE,
    #         tag='no_alpha')

    plt.show()


if __name__ == '__main__':
    main()
