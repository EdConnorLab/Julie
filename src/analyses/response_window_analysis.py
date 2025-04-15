import pandas as pd
from tqdm import tqdm

from analyses.glm_permutation_tests import plot_permutation_anova_distribution
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
            _, f_stat, p_value = result
            all_results.append({
                'NeuronID': neuron,
                'WindowStart_ms': win_start,
                'WindowEnd_ms': win_end,
                'F-statistic': f_stat,
                'p-value': p_value
            })

            if plot:
                detail = details.get(0, {})
                extras = detail.get('extras')
                if extras:
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

    return results_df


