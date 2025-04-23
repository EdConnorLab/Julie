from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import pandas as pd
from analyses.spike_count import prepare_exploded_spike_data


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


if __name__ == "__main__":
    date = "2023-09-26"
    round_no = 3
    exploded_df = prepare_exploded_spike_data(date, round_no, True)
    summary, model = run_population_glmm_by_monkey_identity(exploded_df, monkey_filter="Zombies")
    print(summary)
    plot_glmm_estimates_from_model(model)
