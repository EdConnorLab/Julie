from statsmodels.formula.api import mixedlm

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

if __name__ == "__main__":
    date = "2023-09-26"
    round_no = 1
    exploded_df = prepare_exploded_spike_data(date, round_no, True)
    summary, model = run_population_glmm_by_monkey_identity(exploded_df, monkey_filter="Zombies")
    print(summary)
