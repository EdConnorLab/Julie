import pandas as pd
from tqdm import tqdm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from analyses.single_neuron_analysis import plot_permutation_anova_distribution
from analyses.statistical_tests import perform_statistical_test_on_dataframe_rows, permutation_anova_test


def run_permutation_anova_by_window(df,
                                    category_col='MonkeyName',
                                    neuron_col='NeuronID',
                                    count_col='SpikeCount',
                                    window_start_col='WindowStart_ms',
                                    window_end_col='WindowEnd_ms',
                                    n_permutations=1000,
                                    alpha=0.05,
                                    plot=False):
    """
    Perform permutation ANOVA for each NeuronID and time window combination.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level spike count data. Must include NeuronID, SpikeCount,
        stimulus category column (e.g., MonkeyName), and window columns.
    """
    all_results = []

    group_cols = [neuron_col, window_start_col, window_end_col]
    grouped_df = df.groupby(group_cols)

    for (neuron, win_start, win_end), sub_df in tqdm(grouped_df, desc="Running Perm ANOVA per (Neuron, Window)"):
        grouped = sub_df.groupby(category_col)[count_col].apply(list)

        # Skip if not enough groups
        if len(grouped) < 2:
            continue

        test_input_df = pd.DataFrame([grouped])

        results, total_significant, details = perform_statistical_test_on_dataframe_rows(
            test_input_df,
            test_func=permutation_anova_test,
            num_permutations=n_permutations,
            alpha=alpha
        )

        for result in results:
            index, f_stat, p_value = result
            all_results.append({
                'NeuronID': neuron,
                'WindowStart_ms': win_start,
                'WindowEnd_ms': win_end,
                'F-statistic': f_stat,
                'p-value': p_value
            })

            if plot:
                detail = details.get(index, {})
                extras = detail.get('extras')
                if extras:
                    # print(extras)
                    perm_f_stats, = extras
                    plot_permutation_anova_distribution(
                        perm_f_stats=perm_f_stats,
                        observed_f_stat=detail['stat'],
                        neuron_id=neuron,
                        category_name=category_col,
                        output_dir="permutation_anova_plots"
                    )

    results_df = pd.DataFrame(all_results)
    significant_df = results_df[results_df['p-value'] < alpha]

    print("\nAll Results:")
    print(results_df)

    print("\nSignificant windows (p < 0.05):")
    print(significant_df)

    return results_df, significant_df


def run_glm_by_window(df,
                      formula="SpikeCount ~ C(MonkeyName)",
                      neuron_col="NeuronID",
                      window_start_col="WindowStart_ms",
                      window_end_col="WindowEnd_ms"):
    """
    Run GLM on each (NeuronID, Time Window) combination.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level spike counts. Must include NeuronID, SpikeCount,
        stimulus column (e.g., MonkeyName), and window columns.
    formula : str
        Patsy-style GLM formula (e.g., "SpikeCount ~ C(MonkeyName)")
    Returns
    -------
    results_df : pd.DataFrame
        GLM result summary per neuron-window combo.
    """
    all_results = []

    group_cols = [neuron_col, window_start_col, window_end_col]
    grouped = df.groupby(group_cols)

    for (neuron, win_start, win_end), sub_df in tqdm(grouped, desc="Running GLM per (Neuron, Window)"):
        if sub_df['SpikeCount'].sum() == 0:
            continue  # skip zero-activity windows

        try:
            model = smf.glm(formula=formula, data=sub_df, family=sm.families.Poisson()).fit()
            summary = model.summary2().tables[1].reset_index()
            summary['NeuronID'] = neuron
            summary['WindowStart_ms'] = win_start
            summary['WindowEnd_ms'] = win_end
            all_results.append(summary)
        except Exception as e:
            print(f"Error in neuron {neuron}, window {win_start}-{win_end}: {e}")
            continue

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        # print("\nGLM Results (windows):")
        # print(results_df)
        significant_df = results_df[results_df['P>|z|'] < 0.05]
        print("\nGLM Significant windows:")
        print(significant_df)
        return results_df, significant_df
    else:
        print("No valid (neuron, window) combinations found.")
        return pd.DataFrame()


def correct_glm_by_window_pvalues(glm_df, p_col='P>|z|'):
    """
    Apply FDR correction per NeuronID on windowed GLM results.

    Parameters
    ----------
    glm_df : DataFrame
        Output of run_glm_by_window. Must contain 'NeuronID' and p_col.
    p_col : str
        Column name for p-values to correct (default = 'P>|z|')

    Returns
    -------
    corrected_df : DataFrame
        Copy of input DataFrame with two new columns:
        - 'pval_corrected'
        - 'significant' (True/False)
    """
    df = glm_df.copy()

    # Group by NeuronID and extract min p-values (one per neuron)
    min_pvals_df = df.groupby('NeuronID')[p_col].min().reset_index()

    # Apply FDR correction across all neurons
    reject, pvals_corrected, _, _ = multipletests(min_pvals_df[p_col], method='fdr_bh')

    min_pvals_df['pval_corrected'] = pvals_corrected
    min_pvals_df['significant'] = reject

    return min_pvals_df

def get_significant_windows(glm_df, p_col='P>|z|', alpha=0.05):
    """
    Return significant (NeuronID, Time Window) rows from windowed GLM results.

    Parameters
    ----------
    glm_df : DataFrame
        Output of run_glm_by_window
    p_col : str
        p-value column (default = 'P>|z|')
    alpha : float
        Significance threshold (default = 0.05)

    Returns
    -------
    DataFrame with only significant windows
    """
    return glm_df[glm_df[p_col] < alpha].copy()

