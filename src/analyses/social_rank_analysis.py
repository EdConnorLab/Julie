"""
social_rank_analysis.py
=======================

Compute dominance rank scores from dyadic interaction matrices.

Provides:
    load_group_matrices(group_name, base_dir=...)
        Loads the three Excel matrices (affiliation, agonism, submission)
        for one group. Returns a dict with keys 'affiliation', 'agonism',
        'submission'. Each value is a square DataFrame with rows=actor,
        cols=recipient, both indexed by monkey name (string).

    davids_score(win_matrix)
        Computes David's score (de Vries, Stevens & Vervaecke 2006,
        Animal Behaviour 71:585-592). Takes a "wins" matrix where
        entry (i, j) = number of times i dominated j. Uses the
        Dij = (aij + 1) / (aij + aji + 2) Bayesian correction so
        zero-cell dyads are handled sensibly. Returns a Series of
        David's scores, one per monkey; higher = more dominant.

    behavioral_pca(group_matrices_dict, z_score_within_group=True)
        Runs PCA on per-monkey node-level features (given/received
        for each of the three behaviors). Features are z-scored
        within each group before stacking, so scores are comparable
        across groups of different sizes. Returns (scores_df, pca,
        loadings_df).

Running this file as a script prints a comparison table of:
    - the 'Rank' column from monkeyinfo.csv
    - David's score from agonism-only
    - David's score from submission-only (transposed so it's a wins matrix)
    - David's score from combined (agonism given + submission received)
    - behavioral PC1

for every monkey with interaction data.

NOTE on Instigators: those matrices are still being collected, so results
involving Instigators are provisional and will change as data come in.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Paths — edit if your layout changes
# ---------------------------------------------------------------------------
BASE_DIR = "/home/connorlab/Documents/GitHub/Julie/social_data"
MONKEYINFO_PATH = os.path.join(BASE_DIR, "monkeyinfo.csv")

# Map of display name -> (folder name, file stem). Case-sensitive.
GROUP_PATHS = {
    "Zombies":     ("zombies_social_data",    "zombies"),
    "Best Frans":  ("bestfrans_social_data",  "bestfrans"),
    "Instigators": ("instigators_social_data", "instigators"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_group_matrices(group_name, base_dir=BASE_DIR):
    """
    Load affiliation / agonism / submission matrices for one group.

    Returns dict {'affiliation': df, 'agonism': df, 'submission': df}.
    Each df is indexed by actor name (str), columns are recipient names (str).
    """
    if group_name not in GROUP_PATHS:
        raise ValueError(f"Unknown group '{group_name}'. "
                         f"Known: {list(GROUP_PATHS)}")
    folder, stem = GROUP_PATHS[group_name]
    out = {}
    for behavior in ["affiliation", "agonism", "submission"]:
        path = os.path.join(base_dir, folder, f"{stem}_feature_df_{behavior}.xlsx")
        df = pd.read_excel(path).set_index("Focal Name")
        df.columns = [str(c).replace("Behavior Towards ", "").strip()
                      for c in df.columns]
        df.index = df.index.astype(str).str.strip()
        # Make sure rows and columns are in the same order
        common = [m for m in df.index if m in df.columns]
        missing_rows = set(df.columns) - set(df.index)
        missing_cols = set(df.index) - set(df.columns)
        if missing_rows or missing_cols:
            print(f"  [warn] {group_name} {behavior}: row/col mismatch "
                  f"(rows-missing={missing_rows}, cols-missing={missing_cols})")
        df = df.loc[common, common]
        out[behavior] = df.astype(float)
    return out


# ---------------------------------------------------------------------------
# David's score
# ---------------------------------------------------------------------------
def davids_score(win_matrix):
    """
    David's score with Bayesian-corrected dyadic dominance index.

    Input
    -----
    win_matrix : square DataFrame
        Entry (i, j) = number of times actor i "won over" recipient j.
        Diagonal is ignored (assumed zero). Must have matching row and
        column labels in the same order.

    Returns
    -------
    Series
        David's score per monkey, indexed by monkey name.
        Higher values indicate higher dominance.

    Reference
    ---------
    de Vries H., Stevens J.M.G., Vervaecke H. (2006). Measuring and testing
    the steepness of dominance hierarchies. Animal Behaviour 71: 585-592.

    Formulation (corrected dyadic index):
        D_ij = (a_ij + 1) / (a_ij + a_ji + 2)
    For each individual i:
        w_i  = sum over j of D_ij                  (weighted wins)
        w2_i = sum over j of (D_ij * w_j)          (weighted wins of those
                                                    one beat, weighted again)
        l_i  = sum over j of D_ji                  (weighted losses)
        l2_i = sum over j of (D_ji * l_j)
        DS_i = w_i + w2_i - l_i - l2_i
    """
    if not isinstance(win_matrix, pd.DataFrame):
        raise TypeError("win_matrix must be a DataFrame")
    if list(win_matrix.index) != list(win_matrix.columns):
        raise ValueError("win_matrix row and column labels must match "
                         "and be in the same order")

    A = win_matrix.values.astype(float).copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    # Dyadic dominance index with +1/+2 correction
    #   D[i, j] = (A[i, j] + 1) / (A[i, j] + A[j, i] + 2)
    # Note: even when i and j never interacted, this gives 0.5, which is
    # the intended "no information -> neutral" behavior.
    D = (A + 1.0) / (A + A.T + 2.0)
    np.fill_diagonal(D, 0.0)  # exclude self

    # Standard primary weighted sums
    w = D.sum(axis=1)            # sum over j of D[i, j]
    l = D.sum(axis=0)            # sum over j of D[j, i]

    # Second-order weighted sums
    w2 = D @ w                   # sum_j D[i,j] * w[j]
    l2 = D.T @ l                 # sum_j D[j,i] * l[j]

    ds = w + w2 - l - l2
    return pd.Series(ds, index=win_matrix.index, name="davids_score")


# ---------------------------------------------------------------------------
# Behavioral PCA
# ---------------------------------------------------------------------------
def behavioral_pca(group_matrices, z_score_within_group=True):
    """
    PCA on per-monkey node-level behavioral features, pooled across groups.

    Parameters
    ----------
    group_matrices : dict {group_name: dict_of_matrices}
        Output of load_group_matrices for each group, in one dict.
    z_score_within_group : bool
        If True (recommended), z-score each feature within its own group
        before stacking. This removes group-size and observation-time
        effects (e.g., a bigger group will have more interactions per
        monkey simply because there are more partners).

    Returns
    -------
    scores_df : DataFrame
        One row per monkey. Columns: group, aff_given, aff_received,
        ago_given, ago_received, sub_given, sub_received (the input
        features, z-scored if requested), plus PC1, PC2, PC3, ...
    pca : fitted sklearn.decomposition.PCA object
    loadings_df : DataFrame
        Loadings of features on each PC. Rows=features, cols=PC1, PC2, ...
    """
    rows = []
    for g, mats in group_matrices.items():
        aff, ago, sub = mats["affiliation"], mats["agonism"], mats["submission"]
        feats = pd.DataFrame(index=aff.index)
        feats["aff_given"]    = aff.sum(axis=1)
        feats["aff_received"] = aff.sum(axis=0)
        feats["ago_given"]    = ago.sum(axis=1)
        feats["ago_received"] = ago.sum(axis=0)
        feats["sub_given"]    = sub.sum(axis=1)
        feats["sub_received"] = sub.sum(axis=0)

        if z_score_within_group:
            # ddof=0 for consistency with sklearn StandardScaler
            feats = (feats - feats.mean()) / feats.std(ddof=0)

        feats.insert(0, "group", g)
        rows.append(feats)

    stacked = pd.concat(rows)
    feature_cols = ["aff_given", "aff_received", "ago_given",
                    "ago_received", "sub_given", "sub_received"]
    X = stacked[feature_cols].values

    pca = PCA().fit(X)
    scores = pca.transform(X)
    for k in range(scores.shape[1]):
        stacked[f"PC{k+1}"] = scores[:, k]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{k+1}" for k in range(pca.n_components_)],
    )
    return stacked, pca, loadings


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    info = pd.read_csv(MONKEYINFO_PATH)
    info["Name"] = info["Name"].astype(str).str.strip()
    info_idx = info.set_index("Name")

    print("Loading group matrices...")
    groups = {g: load_group_matrices(g) for g in GROUP_PATHS}
    for g, m in groups.items():
        print(f"  {g}: {m['agonism'].shape[0]} monkeys")
    print()

    # ---- David's scores, three variants per group --------------------------
    # Variant 1: agonism only.
    #   "i aggressed j" as a win of i over j.
    # Variant 2: submission transposed.
    #   "j submitted to i" is a win of i over j, so we use submission.T.
    # Variant 3: combined.
    #   Sum the two wins matrices — i.e. agonism + submission.T.
    ds_rows = []
    for g, m in groups.items():
        ago  = m["agonism"]
        sub  = m["submission"]
        sub_wins = sub.T          # row=actor-of-wins; actor "won" when recipient submitted
        combined = ago + sub_wins

        ds_ago = davids_score(ago)
        ds_sub = davids_score(sub_wins)
        ds_cmb = davids_score(combined)

        df = pd.DataFrame({
            "group": g,
            "DS_agonism":   ds_ago,
            "DS_submission": ds_sub,
            "DS_combined":  ds_cmb,
        })
        ds_rows.append(df)
    ds_all = pd.concat(ds_rows)

    # ---- Behavioral PCA ------------------------------------------------------
    pca_scores, pca_obj, loadings = behavioral_pca(groups,
                                                   z_score_within_group=True)
    print("PCA explained variance ratio:",
          np.round(pca_obj.explained_variance_ratio_, 3))
    print("Cumulative:                  ",
          np.round(np.cumsum(pca_obj.explained_variance_ratio_), 3))
    print()
    print("Feature loadings (first 3 PCs):")
    print(loadings.iloc[:, :3].round(2))
    print()

    # ---- Assemble comparison table ------------------------------------------
    comparison = ds_all.join(pca_scores[["PC1", "PC2", "PC3"]])
    comparison["official_rank"] = [info_idx.loc[m, "Rank"]
                                   if m in info_idx.index else np.nan
                                   for m in comparison.index]
    comparison = comparison[["group", "official_rank",
                             "DS_agonism", "DS_submission", "DS_combined",
                             "PC1", "PC2", "PC3"]]

    # ---- Print per group -----------------------------------------------------
    for g in GROUP_PATHS:
        sub = comparison[comparison["group"] == g].copy()
        sub = sub.sort_values("DS_combined", ascending=False)
        sub.round(2).to_csv(f'{g}_davids_score.csv', index=True)
        print(f"=== {g} ===")
        print(sub.round(2).to_string())
        print()

    # ---- Correlations with official rank (within each group) ---------------
    # Note: official_rank has 1 = top, so high dominance score should give
    # a negative correlation.
    print("Correlations with official Rank (1=top, negative r = expected):")
    for g in GROUP_PATHS:
        sub = comparison[(comparison["group"] == g)
                         & comparison["official_rank"].notna()]
        if len(sub) < 3:
            print(f"  {g}: too few monkeys with rank ({len(sub)})")
            continue
        line = f"  {g:12s} (n={len(sub)})"
        for col in ["DS_agonism", "DS_submission", "DS_combined", "PC1"]:
            r = np.corrcoef(sub[col], sub["official_rank"])[0, 1]
            line += f"  {col}: r={r:+.2f}"
        print(line)
    print()

    # ---- Correlations among the four dominance measures -------------------
    print("Within-group correlations among the dominance measures:")
    for g in GROUP_PATHS:
        sub = comparison[comparison["group"] == g]
        if len(sub) < 3:
            continue
        print(f"  {g}:")
        print(sub[["DS_agonism", "DS_submission", "DS_combined", "PC1"]]
              .corr().round(2).to_string())
        print()

    return comparison, loadings, pca_obj


if __name__ == "__main__":
    main()
