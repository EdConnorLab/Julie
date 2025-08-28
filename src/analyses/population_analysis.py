import os

import numpy as np
from scipy.stats import zscore
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.spike_count import prepare_exploded_spike_data, filter_good_neurons, bin_spike_times



def run_glmm_social_score_across_neurons(spike_df, social_vector_df, social_col_name='AffiliationFrom_z'):
    """
    Run a population-level GLMM to test if spike count is modulated by social score,
    with NeuronID as random effect and social score as fixed effect.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame with columns ['NeuronID', 'MonkeyName', 'SpikeCount']
    social_vector_df : pd.DataFrame
        DataFrame with columns ['MonkeyName', social_col_name] (z-scored)
    social_col_name : str
        Name of the column in social_vector_df to use as predictor

    Returns
    -------
    summary : statsmodels Summary
        Summary of the fitted mixed model
    model : MixedLMResults
        Fitted model object
    """
    # Merge with social score
    merged_df = spike_df.merge(social_vector_df[['MonkeyName', social_col_name]], on='MonkeyName')

    # Ensure categorical
    merged_df['MonkeyName'] = merged_df['MonkeyName'].astype('category')
    merged_df['NeuronID'] = merged_df['NeuronID'].astype('category')

    # Fit mixed linear model
    formula = f"SpikeCount ~ {social_col_name}"
    model = mixedlm(formula, data=merged_df, groups=merged_df['NeuronID'])
    result = model.fit()

    return result.summary(), result

def run_population_glmm_by_social_score(df, social_vector, social_col_name='AffiliationTo_z'):

    # Step 3: merge with social vector
    df = df.merge(social_vector[['MonkeyName', social_col_name]], on='MonkeyName')

    # Step 4: fit mixed model
    df['MonkeyName'] = df['MonkeyName'].astype('category')
    model = mixedlm(f"SpikeCount ~ {social_col_name}", data=df, groups=df['MonkeyName'])
    result = model.fit()

    return result.summary(), result

def run_population_glmm_by_monkey_identity(
    df,
    formula="SpikeCount ~ C(MonkeyName)",
    neuron_col="NeuronID",
    monkey_filter=None,
    spike_col="SpikeTimes"
):
    """
    Run a population-level linear mixed model with neuron as a random effect
    and monkey identity as a fixed effect.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing one row per trial with spike times.
    formula : str, optional
        Patsy-style formula string for the fixed effects. Default: "SpikeCount ~ C(MonkeyName)".
    neuron_col : str, optional
        Column name for neuron identity. Used as the random effect group.
    monkey_filter : list or set, optional
        If provided, only rows where monkey_col is in this list will be included.
    spike_col : str, optional
        Column name containing spike times (as lists). Will be converted to spike counts.

    Returns
    -------
    result.summary() : statsmodels Summary
        Summary of the fitted mixed linear model.
    """

    # Convert spike times to spike counts
    df = df.copy()
    df['SpikeCount'] = df[spike_col].apply(len)

    # Optionally filter for specific monkeys
    monkey_col = formula.split("~")[1].strip().split("(")[-1].rstrip(")")
    monkey_col = monkey_col.replace("C(", "").replace(")", "").strip()
    # Filter by monkey_filter
    if monkey_filter is not None:
        if isinstance(monkey_filter, str):
            # Treat as MonkeyGroup name
            if 'MonkeyGroup' not in df.columns:
                raise ValueError("To filter by MonkeyGroup, 'MonkeyGroup' column must exist in the input dataframe.")
            df = df[df['MonkeyGroup'] == monkey_filter]
        elif isinstance(monkey_filter, (list, set)):
            df = df[df[monkey_col].isin(monkey_filter)]
        else:
            raise ValueError("monkey_filter must be a string (MonkeyGroup name), list, or set.")

    # Collapse to one row per neuron x monkey (average spike count)
    summary_df = (
        df.groupby([neuron_col, monkey_col])['SpikeCount']
        .mean()
        .reset_index()
    )

    # Ensure monkey variable is categorical (if needed)
    summary_df[monkey_col] = summary_df[monkey_col].astype('category')

    # Fit the mixed-effects model
    model = mixedlm(formula, data=summary_df, groups=summary_df[neuron_col])
    result = model.fit()

    return result.summary(), result


def plot_glmm_estimates_from_model(model, reference_label="(ref)"):
    """
    Plot estimated spike counts per monkey from a fitted MixedLMResults model object.

    Parameters
    ----------
    model : statsmodels.regression.mixed_linear_model.MixedLMResults
        A fitted model from statsmodels' mixedlm().
    reference_label : str, optional
        Label to assign to the reference level (intercept term).
    """

    # Get fixed effects and confidence intervals
    fixed_effects = model.fe_params
    conf_ints = model.conf_int()

    # Extract intercept
    intercept = fixed_effects["Intercept"]
    intercept_ci = conf_ints.loc["Intercept"]

    # Get all other monkey terms (e.g., C(MonkeyName)[T.143H])
    monkey_terms = [term for term in fixed_effects.index if term != "Intercept"]

    # Parse monkey names from terms
    monkeys = [term.split("[")[-1].strip("]") for term in monkey_terms]
    monkeys = [f"{model.model.data.orig_exog.columns[1]} {reference_label}"] + monkeys

    # Compute estimated firing rates and CIs
    estimates = [intercept] + [
        intercept + fixed_effects[term] for term in monkey_terms
    ]
    lower_bounds = [intercept_ci[0]] + [
        intercept + conf_ints.loc[term][0] for term in monkey_terms
    ]
    upper_bounds = [intercept_ci[1]] + [
        intercept + conf_ints.loc[term][1] for term in monkey_terms
    ]

    # Create DataFrame
    plot_df = pd.DataFrame({
        "Monkey": monkeys,
        "FiringRate": estimates,
        "LowerCI": lower_bounds,
        "UpperCI": upper_bounds
    })

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=plot_df['Monkey'], y=plot_df['FiringRate'],
        yerr=[plot_df['FiringRate'] - plot_df['LowerCI'], plot_df['UpperCI'] - plot_df['FiringRate']],
        fmt='o', capsize=5, linestyle='-', marker='o'
    )
    plt.xticks(rotation=45)
    plt.xlabel('Monkey Identity')
    plt.ylabel('Estimated Mean Spike Count')
    plt.title('Estimated Spike Count per Monkey (w/ 95% CI)')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return plot_df  # return the data used in plot for inspection if needed

def get_all_combined_exploded_spike_counts(group_name="Zombies", apply_filter=False, apply_binning=False, bin_size=0.05, location=None, use_sorted=False):
    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()

    all_dfs = []

    for _, row in metadata.iterrows():
        date = str(row['Date'].strftime("%Y-%m-%d"))
        round_no = int(row['Round No.'])

        exploded_df = prepare_exploded_spike_data(date, round_no, only_valid_channels=True, use_sorted=use_sorted)
        if apply_filter:
            good_neurons = filter_good_neurons(exploded_df)
            exploded_df = exploded_df[exploded_df["NeuronID"].isin(good_neurons)]
        if location == 'AMG' or location == 'ER' or location == 'Unknown':
            exploded_df = exploded_df[exploded_df["NeuronID"].str.startswith(location)]
        elif location == None:
            pass
        else:
            raise ValueError("location must be one of 'AMG', 'ER', 'Unknown', or None.")
        exploded_df = exploded_df[(exploded_df['MonkeyGroup'] == group_name) & (exploded_df['MonkeyName'] != "NewMonkey")]
        exploded_df['SpikeCount'] = exploded_df['SpikeTimes'].apply(len)

        if apply_binning:
            exploded_df = bin_spike_times(exploded_df, bin_size)

        all_dfs.append(exploded_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    # Running directional GLMMs on all neurons
    exploded_df = get_all_combined_exploded_spike_counts(apply_filter=True, location='ER')
    exploded_df['SpikeCount'] = exploded_df['SpikeTimes'].apply(len)

    subject_monkey_index = 6
    monkey_group_name = "Zombies"
    monkey_list = get_monkeys_by_default_order(monkey_group_name)

    base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'

    behavior_files = {
        "AffiliationTo": "zombies_feature_df_affiliation.xlsx",
        "AffiliationFrom": "zombies_feature_df_affiliation.xlsx",
        "SubmissionTo": "zombies_feature_df_submission.xlsx",
        "SubmissionFrom": "zombies_feature_df_submission.xlsx",
        "AgonismTo": "zombies_feature_df_agonism.xlsx",
        "AgonismFrom": "zombies_feature_df_agonism.xlsx",
    }

    for metric_name, file_name in behavior_files.items():
        print(f"\n===== Running GLMM for: {metric_name} =====")

        # Load matrix
        matrix = pd.read_excel(os.path.join(base_dir, file_name)).iloc[:, 1:].to_numpy()
        if 'From' in metric_name:
            matrix = matrix.T  # transpose if needed

        beh_vector = matrix[subject_monkey_index].copy()
        behavior_vector = np.delete(beh_vector, subject_monkey_index)
        monkey_vector = [m for i, m in enumerate(monkey_list) if i != subject_monkey_index]

        z_vector = zscore(behavior_vector)
        social_df = pd.DataFrame({
            'MonkeyName': monkey_vector,
            f'{metric_name}_z': z_vector
        })

        try:
            summary, model = run_glmm_social_score_across_neurons(
                exploded_df, social_df, social_col_name=f'{metric_name}_z'
            )
            print(summary)
        except Exception as e:
            print(f"[ERROR] GLMM failed for {metric_name}: {e}")
    ###  AMG  results summary! ###
    # # Step 1: manual summary of results (AMG only)
    # results = [
    #     {"Metric": "AffiliationTo", "Beta": 0.303, "CI_low": 0.060, "CI_high": 0.546, "p": 0.015},
    #     {"Metric": "AffiliationFrom", "Beta": 0.233, "CI_low": -0.010, "CI_high": 0.477, "p": 0.060},
    #     {"Metric": "SubmissionTo", "Beta": 0.388, "CI_low": 0.144, "CI_high": 0.631, "p": 0.002},
    #     {"Metric": "SubmissionFrom", "Beta": -0.175, "CI_low": -0.419, "CI_high": 0.070, "p": 0.162},
    #     {"Metric": "AgonismTo", "Beta": -0.230, "CI_low": -0.476, "CI_high": 0.017, "p": 0.068},
    #     {"Metric": "AgonismFrom", "Beta": 0.095, "CI_low": -0.149, "CI_high": 0.340, "p": 0.445},
    # ]
    # df = pd.DataFrame(results)
    # # Step 2: Plot
    # plt.figure(figsize=(10, 6))
    #
    # # Plot beta with 95% CI
    # x = np.arange(len(df))
    # y = df["Beta"]
    # yerr = [y - df["CI_low"], df["CI_high"] - y]
    #
    # bars = plt.bar(x, y, yerr=yerr, capsize=5, color="skyblue", edgecolor='k')
    #
    # # Add significance stars
    # for i, p in enumerate(df["p"]):
    #     if p < 0.001:
    #         star = "***"
    #     elif p < 0.01:
    #         star = "**"
    #     elif p < 0.05:
    #         star = "*"
    #     elif p < 0.1:
    #         star = "+"
    #     else:
    #         star = ""
    #     if star:
    #         plt.text(i, y[i] + 0.05, star, ha='center', va='bottom', fontsize=14)
    #
    # # Aesthetic
    # plt.axhline(0, color='gray', linestyle='--')
    # plt.xticks(x, df["Metric"], rotation=45)
    # plt.ylabel("β Estimate (GLMM)")
    # plt.title("AMG Population-Level GLMM: Social Metric Effects on SpikeCount")
    # plt.tight_layout()
    # plt.show()