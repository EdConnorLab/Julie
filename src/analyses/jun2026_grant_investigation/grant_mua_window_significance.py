"""
Kruskal-Wallis / ANOVA permutation significance test on the response windows DETECTED for
the grant's UNSORTED (MUA) cells.

Pipeline:
  1. match the 37 grant MUA cells -> threshold-MUA NeuronIDs (32 unique neurons),
  2. run the in-house detector (threshold_window_detection) on those neurons,
  3. for each (neuron, detected window) run the permutation KW and/or ANOVA test across the
     stimulus monkeys (reusing analyses.preprocessing.detect_significant_windows_using_p*),
  4. FDR-correct (Benjamini-Hochberg) and write one combined per-window table.

This answers "are the detected windows actually significant?" -- i.e. does firing differ
across the stimulus monkeys within each detected window. It is the same test / permutation
scheme run_mua_preprocessing.py uses for the full list, just scoped to the grant MUA cells.

IMPORTANT: run with cfg.save=False so it does NOT overwrite the FULL-list significance pkls
(threshold_mua_Zombies_significant_windows_p*.pkl) with this 37-cell subset.

Must run on the machine with threshold_mua_spike_cache / the raw recordings.

    python grant_mua_window_significance.py [optional/output/path.csv]

Output columns (one row per detected (neuron, window)):
  NeuronID, WindowStart_ms, WindowEnd_ms, Date, Round No.,
  KW_stat, KW_p, KW_sig, KW_p_fdr, KW_sig_fdr,
  ANOVA_stat, ANOVA_p, ANOVA_sig, ANOVA_p_fdr, ANOVA_sig_fdr
  (*_sig = uncorrected p < alpha; *_sig_fdr = survives Benjamini-Hochberg FDR)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- edit these ----
N_PERMUTATIONS = 1000       # matches run_mua_preprocessing.py
ALPHA = 0.05
TESTS = ("KW", "ANOVA")     # subset to ("KW",) or ("ANOVA",) to run just one
# --------------------

KEYS = ["NeuronID", "WindowStart_ms", "WindowEnd_ms", "Date", "Round No."]


def _as_pair(ret):
    """detect_significant_windows_using_p* returns (all, sig) normally but a single frame on
    the empty path -- normalize to a pair."""
    return ret if isinstance(ret, tuple) else (ret, ret)


def _tidy(res, tag, alpha):
    """Rename a per-window results frame to test-prefixed columns and add the uncorrected
    significance flag."""
    import pandas as pd  # noqa: F401  (ensures pandas is importable in this scope)
    stat = "H-statistic" if "H-statistic" in res.columns else "F-statistic"
    r = res.rename(columns={
        stat: f"{tag}_stat",
        "p-value": f"{tag}_p",
        "p-value_corrected": f"{tag}_p_fdr",
        "p-value_significant": f"{tag}_sig_fdr",
    })
    r[f"{tag}_sig"] = r[f"{tag}_p"] < alpha
    keep = KEYS + [f"{tag}_stat", f"{tag}_p", f"{tag}_sig", f"{tag}_p_fdr", f"{tag}_sig_fdr"]
    return r[[c for c in keep if c in r.columns]]


def run_window_significance(n_permutations=N_PERMUTATIONS, alpha=ALPHA, tests=TESTS, out_csv=None):
    """Detect windows for the grant MUA cells and permutation-test each. Returns
    (combined_df, path)."""
    import pandas as pd
    from spike_count_connector import grant_mua_detected_windows, GROUP, DETECT_BIN_SIZE
    from analyses.preprocessing.preprocess_and_select_significant_neurons import (
        PreprocessConfig,
        detect_significant_windows_using_pKW,
        detect_significant_windows_using_pANOVA,
    )

    detected, source = grant_mua_detected_windows()
    if detected.empty:
        raise RuntimeError("[grant-mua][sig] no detected windows to test")
    n_neurons = detected["NeuronID"].nunique()
    print(f"[grant-mua][sig] testing {len(detected)} detected windows across {n_neurons} neurons "
          f"({n_permutations} permutations, alpha={alpha}, tests={tests})")

    # save=False so we never clobber the FULL-list threshold_mua_*_significant_windows_p*.pkl
    cfg = PreprocessConfig(group_name=GROUP, bin_size=DETECT_BIN_SIZE, save=False)

    runners = {
        "KW": detect_significant_windows_using_pKW,
        "ANOVA": detect_significant_windows_using_pANOVA,
    }
    results = {}
    for tag in tests:
        res, _sig = _as_pair(runners[tag](
            detected, cfg, source=source, n_permutations=n_permutations, alpha=alpha))
        results[tag] = res

    combined = None
    for tag in tests:
        res = results.get(tag)
        if res is None or res.empty:
            continue
        tidy = _tidy(res, tag, alpha)
        combined = tidy if combined is None else combined.merge(tidy, on=KEYS, how="outer")
    if combined is None:
        combined = pd.DataFrame(columns=KEYS)
    combined = combined.sort_values(["Date", "Round No.", "NeuronID", "WindowStart_ms"])

    out_csv = out_csv or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "output", "grant_mua_window_significance.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    combined.to_csv(out_csv, index=False)

    # ---- summary ----
    print("\n[grant-mua][sig] === summary ===")
    print(f"  detected windows: {len(detected)} across {n_neurons} neurons")
    for tag in tests:
        res = results.get(tag)
        if res is None or res.empty:
            print(f"  {tag}: 0 windows tested")
            continue
        n_tested = len(res)
        n_uncorr = int((res["p-value"] < alpha).sum())
        sig_col = [c for c in res.columns if c.endswith("_significant")][0]
        n_fdr = int(res[sig_col].fillna(False).sum())
        neurons_sig = res.loc[res["p-value"] < alpha, "NeuronID"].nunique()
        print(f"  {tag}: {n_tested} windows tested; "
              f"{n_uncorr} sig (uncorrected p<{alpha}) on {neurons_sig} neuron(s); "
              f"{n_fdr} sig after FDR")
    print(f"\n[grant-mua][sig] combined per-window table -> {out_csv}")
    return combined, out_csv


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else None
    run_window_significance(out_csv=out_csv)


if __name__ == "__main__":
    main()
