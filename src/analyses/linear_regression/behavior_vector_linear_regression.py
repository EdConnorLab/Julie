import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, zscore, spearmanr
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.spike_count import extract_spike_counts_from_windows, prepare_exploded_spike_data
from analyses.spike_rate import compute_mean_spike_rate_table

def run_linear_regression_using_sklearn(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y)


def run_marginal_vector_linear_regression_from_matrix(
    spike_df,
    behavior_matrix,
    behavior_name,
    group_name,
    monkey_list,
    subject_idx,
    use_spikerate=False,
    model_type='ols'
):
    """
    Run OLS or GLM per neuron using marginal vector from a behavior matrix (1 score per monkey).

    Parameters:
        spike_df: pd.DataFrame with ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'SpikeCount' or 'MeanSpikeRate']
        behavior_matrix: 2D np.ndarray (e.g., AffiliationTo matrix)
        behavior_name: str, name of behavior
        group_name: str, name of monkey group (e.g., "Zombies")
        monkey_list: list of monkey names in order
        subject_idx: index of subject monkey (to exclude)
        use_spikerate: if True, use 'MeanSpikeRate'; else, use 'SpikeCount'
        model_type: 'ols' or 'glm'
    Returns:
        pd.DataFrame with results per neuron
    """
    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'

    # 1. exclude subject monkey
    subject_monkey = monkey_list[subject_idx]
    monkeys = [m for m in monkey_list if m != subject_monkey]
    idxs = [i for i in range(len(monkey_list)) if i != subject_idx]

    # 2. compute marginal vector (mean across rows or columns)
    if 'From' in behavior_name:
        marginals = behavior_matrix[:, idxs].mean(axis=0)
    else:
        marginals = behavior_matrix[idxs, :].mean(axis=1)
    y = zscore(marginals)

    # 3. averaging spike data
    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'].isin(monkeys))
    ]
    mean_spikes = (
        filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col].mean()
    )
    spike_matrix = mean_spikes.pivot(index='NeuronID', columns='MonkeyName', values=value_col)

    # 4. regression per neuron
    for neuron_id, row in spike_matrix.iterrows():
        x_row = row[monkeys].values.astype(float)
        x = zscore(x_row)

        if np.var(x) < 1e-3 or np.var(y) < 1e-3:
            continue
        if len(x) != len(y):
            continue

        try:
            X = sm.add_constant(x)
            if model_type == 'ols':
                model = sm.OLS(y, X).fit()
                r_squared = model.rsquared
            else:
                model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                r_squared = None

            results.append({
                'NeuronID': neuron_id,
                'Behavior': behavior_name,
                'coef': model.params[1],
                'p_value': model.pvalues[1],
                'R-squared': r_squared
            })
        except Exception as e:
            print(f"Error for neuron {neuron_id}: {e}")
            continue

    return pd.DataFrame(results)



def run_directional_vector_linear_regression_window_level(
        spike_df,
        behavior_matrix,
        behavior_name,
        group_name,
        monkey_list,
        subject_idx,
        use_spikerate = False,
        model_type ='ols',
        permutation_test = False,
        n_perm = 10000):
    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    filtered_df = spike_df[(spike_df['MonkeyGroup'] == group_name) & (spike_df['MonkeyName'] != "NewMonkey")]
    # mean_spikes = filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col].mean()
    # spike_matrix = mean_spikes.pivot(index='NeuronID', columns='MonkeyName', values= value_col)
    mean_spikes = filtered_df.groupby(
        ['NeuronID', 'MonkeyName', 'WindowStart_ms', 'WindowEnd_ms'], as_index=False
    )[value_col].mean()

    # Create a unique key for each window
    mean_spikes['NeuronWindowID'] = (
            mean_spikes['NeuronID'].astype(str) + "_" +
            mean_spikes['WindowStart_ms'].astype(str) + "_" +
            mean_spikes['WindowEnd_ms'].astype(str)
    )

    spike_matrix = mean_spikes.pivot(index='NeuronWindowID', columns='MonkeyName', values=value_col)

    # Add back mapping from NeuronWindowID → NeuronID + Window
    id_map = mean_spikes[['NeuronWindowID', 'NeuronID', 'WindowStart_ms', 'WindowEnd_ms']].drop_duplicates()
    for unit_id, row in spike_matrix.iterrows():
        meta = id_map[id_map['NeuronWindowID'] == unit_id].iloc[0]
        neuron_id = meta['NeuronID']
        win_start = meta['WindowStart_ms']
        win_end = meta['WindowEnd_ms']
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue
            # z-score y
            raw_y = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_y, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            y = zscore(raw_y[mask])

            x_row = row.drop(index=[monkey_list[src_idx], monkey_list[subject_idx]], errors='ignore')
            x = x_row.values.astype(float)
            # check variance
            if np.var(x) < 1e-3 or np.var(y) < 1e-3:  # threshold adjustable (e.g. 0.0001)
                print(
                    f"Skipped {unit_id} {src_monkey}: variance too small (x_var={np.var(x):.6f}, y_var={np.var(y):.6f})")
                continue

            if len(x) != len(y):
                print(f"Length mismatch for {unit_id} (source: {monkey_list[src_idx]})")
                continue

            try:
                X = sm.add_constant(x)
                if model_type == 'ols':
                    model = sm.OLS(y, X).fit()
                    r_squared = model.rsquared
                elif model_type == 'glm':
                    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                    r_squared = None
                else:
                    raise ValueError("model_type must be 'ols' or 'glm'")

                # Permutation test
                if permutation_test and model_type == 'ols':
                    null_distribution = []
                    for _ in range(n_perm):
                        y_perm = np.random.permutation(y)
                        model_perm = sm.OLS(y_perm, X).fit()
                        null_distribution.append(model_perm.rsquared)
                    null_distribution = np.array(null_distribution)
                    p_perm = (np.sum(null_distribution >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None

                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': win_start,
                    'WindowEnd_ms': win_end,
                    'Behavior': behavior_name,
                    'Source_Monkey': monkey_list[src_idx],
                    'Model': model_type,
                    'R-squared': r_squared,
                    'coef': model.params[1],
                    'intercept': model.params[0],
                    'p_value': model.pvalues[1],
                    'p_perm': p_perm
                })

            except Exception as e:
                print(f"Error for {unit_id}, Monkey {monkey_list[src_idx]}: {e}")
                continue

    return pd.DataFrame(results)


def run_directional_vector_linear_regression_cell_level(
        spike_df,
        behavior_matrix,
        behavior_name,
        group_name,
        monkey_list,
        subject_idx,
        use_spikerate = False,
        model_type ='ols',
        plot = False,
        permutation_test = False,
        n_perm = 10000):
    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    filtered_df = spike_df[(spike_df['MonkeyGroup'] == group_name) & (spike_df['MonkeyName'] != "NewMonkey")]
    mean_spikes = filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col].mean()
    spike_matrix = mean_spikes.pivot(index='NeuronID', columns='MonkeyName', values= value_col)

    for neuron_id, row in spike_matrix.iterrows():
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue
            # y = np.delete(behavior_matrix[src_idx], [src_idx, subject_idx])
            # z-score y
            raw_y = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_y, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            y = zscore(raw_y[mask])

            x_row = row.drop(index=[monkey_list[src_idx], monkey_list[subject_idx]], errors='ignore')
            x = x_row.values.astype(float)
            # check variance
            if np.var(x) < 1e-3 or np.var(y) < 1e-3:  # threshold adjustable (e.g. 0.0001)
                print(
                    f"Skipped {neuron_id} {src_monkey}: variance too small (x_var={np.var(x):.6f}, y_var={np.var(y):.6f})")
                continue

            if len(x) != len(y):
                print(f"Length mismatch for {neuron_id} (source: {monkey_list[src_idx]})")
                continue

            try:
                X = sm.add_constant(x)
                if model_type == 'ols':
                    model = sm.OLS(y, X).fit()
                    r_squared = model.rsquared
                elif model_type == 'glm':
                    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                    r_squared = None
                else:
                    raise ValueError("model_type must be 'ols' or 'glm'")

                # Permutation test
                if permutation_test and model_type == 'ols':
                    null_distribution = []
                    for _ in range(n_perm):
                        y_perm = np.random.permutation(y)
                        model_perm = sm.OLS(y_perm, X).fit()
                        null_distribution.append(model_perm.rsquared)
                    null_distribution = np.array(null_distribution)
                    p_perm = (np.sum(null_distribution >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None

                results.append({
                    'NeuronID': neuron_id,
                    'Behavior': behavior_name,
                    'Source_Monkey': monkey_list[src_idx],
                    'Model': model_type,
                    'R-squared': r_squared,
                    'coef': model.params[1],
                    'intercept': model.params[0],
                    'p_value': model.pvalues[1],
                    'p_perm': p_perm
                })

                if plot and r_squared > 0.6:
                    y_pred = model.predict(X)

                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, label='True', color='blue', alpha=0.6)
                    plt.scatter(x, y_pred, color='green', label='Prediction', alpha=0.6)
                    plt.plot(x, y_pred, label='Fit', color='black', alpha=0.6)
                    plt.xlabel(f'{behavior_name} {monkey_list[src_idx]} (z-scored)')
                    plt.ylabel('Neural Response (z-scored)')
                    plt.title(f'{neuron_id} (R-sq {r_squared:.3f})')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()


            except Exception as e:
                print(f"Error for Neuron {neuron_id}, Monkey {monkey_list[src_idx]}: {e}")
                continue

    return pd.DataFrame(results)


def expand_cell_level_regression_results_with_spike_rates_per_stimulus(
        spike_df,
        regression_df,
        group_name,
        monkey_list,
        subject_idx,
        use_spikerate=True
):
    """
    Expands regression summary results by attaching mean spike rates per stimulus monkey.

    Parameters:
        spike_df (pd.DataFrame): Raw or processed DataFrame with 'NeuronID', 'MonkeyName', and either 'SpikeCount' or 'MeanSpikeRate'
        regression_df (pd.DataFrame): Output from run_directional_vector_linear_regression_cell_level
        group_name (str): Monkey group name (e.g., "Zombies")
        monkey_list (List[str]): List of all monkeys in group (ordered)
        subject_idx (int): Index of the subject monkey in monkey_list
        use_spikerate (bool): Whether to use 'MeanSpikeRate' or 'SpikeCount'

    Returns:
        pd.DataFrame: Expanded table with one row per stimulus monkey per regression result
    """

    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'

    # Filter for group and drop bad labels
    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'] != "NewMonkey")
        ]

    # Average spike rate per neuron per stimulus monkey
    mean_spikes = filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col].mean()

    # Build expanded table
    expanded_rows = []

    for _, row in regression_df.iterrows():
        neuron_id = row['NeuronID']
        behavior = row['Behavior']
        source = row['Source_Monkey']
        excluded = {monkey_list[subject_idx], source}

        for stim_monkey in monkey_list:
            if stim_monkey in excluded:
                continue

            # Get spike rate to this stimulus monkey for this neuron
            match = mean_spikes[
                (mean_spikes['NeuronID'] == neuron_id) &
                (mean_spikes['MonkeyName'] == stim_monkey)
                ]

            if match.empty:
                continue

            expanded_rows.append({
                'NeuronID': neuron_id,
                'Behavior': behavior,
                'Source_Monkey': source,
                'StimulusMonkey': stim_monkey,
                'MeanSpikeRate': match[value_col].values[0],
                'R-squared': row['R-squared'],
                'coef': row['coef'],
                'intercept': row['intercept'],
                'p_value': row['p_value']
            })

    return pd.DataFrame(expanded_rows)

def expand_window_level_regression_results_with_spike_rates_per_stimulus(
    spike_df,
    regression_df,
    group_name,
    monkey_list,
    subject_idx,
    use_spikerate=True
):
    """
    Expands window-level regression results with per-stimulus monkey mean spike rates.

    Parameters:
        spike_df (pd.DataFrame): DataFrame with 'NeuronID', 'MonkeyName', 'WindowStart_ms', 'WindowEnd_ms',
                                 and either 'SpikeCount' or 'MeanSpikeRate'
        regression_df (pd.DataFrame): Output from run_directional_vector_linear_regression_window_level
        group_name (str): Group name (e.g. "Zombies")
        monkey_list (List[str]): List of monkeys in group
        subject_idx (int): Index of subject monkey
        use_spikerate (bool): Use 'MeanSpikeRate' or 'SpikeCount'

    Returns:
        pd.DataFrame: Expanded table with one row per stimulus monkey per regression window
    """

    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'

    # Filter spike_df
    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'] != "NewMonkey")
    ]

    # Average spike rate per Neuron x StimulusMonkey x Time Window
    mean_spikes = filtered_df.groupby(
        ['NeuronID', 'MonkeyName', 'WindowStart_ms', 'WindowEnd_ms'],
        as_index=False
    )[value_col].mean()

    expanded_rows = []

    for _, row in regression_df.iterrows():
        neuron_id = row['NeuronID']
        behavior = row['Behavior']
        source = row['Source_Monkey']
        win_start = row['WindowStart_ms']
        win_end = row['WindowEnd_ms']
        excluded = {monkey_list[subject_idx], source}

        for stim_monkey in monkey_list:
            if stim_monkey in excluded:
                continue

            match = mean_spikes[
                (mean_spikes['NeuronID'] == neuron_id) &
                (mean_spikes['MonkeyName'] == stim_monkey) &
                (mean_spikes['WindowStart_ms'] == win_start) &
                (mean_spikes['WindowEnd_ms'] == win_end)
            ]

            if match.empty:
                continue

            expanded_rows.append({
                'NeuronID': neuron_id,
                'WindowStart_ms': win_start,
                'WindowEnd_ms': win_end,
                'Behavior': behavior,
                'Source_Monkey': source,
                'StimulusMonkey': stim_monkey,
                'MeanSpikeRate': match[value_col].values[0],
                'R-squared': row['R-squared'],
                'coef': row['coef'],
                'intercept': row['intercept'],
                'p_value': row['p_value']
            })

    return pd.DataFrame(expanded_rows)


def run_rsa_analysis(spike_df, behavior_matrix, behavior_name, monkey_list, subject_idx, method='cosine', use_rate = False):
    """
    Run RSA comparing neural and behavioral similarity (excluding subject monkey).

    Parameters:
        spike_df (pd.DataFrame): contains ['NeuronID', 'MonkeyName', 'SpikeCount' or 'MeanSpikeRate']
        behavior_matrix (np.ndarray): full social matrix (e.g., AffiliationTo)
        monkey_list (list): all monkeys (ordered like behavior_matrix)
        subject_idx (int): index of the subject monkey to exclude
        method (str): similarity metric ('correlation', 'cosine', etc.)

    Returns:
        r, p, neural_rsm, social_rsm
    """

    value_col = 'MeanSpikeRate' if use_rate else 'SpikeCount'

    # Step 0: Remove subject monkey
    subject_monkey = monkey_list[subject_idx]
    filtered_monkeys = [m for m in monkey_list if m != subject_monkey]

    # Filter neural data
    mean_df = (
        spike_df[spike_df['MonkeyName'].isin(filtered_monkeys)]
        .groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col]
        .mean()
    )

    neural_matrix = mean_df.pivot(index='NeuronID', columns='MonkeyName', values=value_col)
    neural_matrix = neural_matrix[filtered_monkeys].dropna()

    # Filter behavior matrix
    idxs = [monkey_list.index(m) for m in filtered_monkeys]
    behavior_matrix = behavior_matrix[np.ix_(idxs, idxs)]

    # Neural RSM
    neural_rsm = 1 - pd.DataFrame(
        squareform(pdist(neural_matrix.T, metric=method)),
        index=filtered_monkeys,
        columns=filtered_monkeys
    )

    # Social RSM
    social_rsm = 1 - pd.DataFrame(
        squareform(pdist(behavior_matrix.T, metric=method)),
        index=filtered_monkeys,
        columns=filtered_monkeys
    )

    # Flatten upper triangle for correlation
    mask = np.triu(np.ones_like(neural_rsm), k=1).astype(bool)
    neural_vec = neural_rsm.values[mask]
    social_vec = social_rsm.values[mask]
    r, p = spearmanr(neural_vec, social_vec)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(neural_rsm, ax=ax[0], cmap='viridis')
    ax[0].set_title("Neural RSM")
    sns.heatmap(social_rsm, ax=ax[1], cmap='viridis')
    ax[1].set_title("Social RSM")
    plt.suptitle(f"RSA for {behavior_name}: r = {r:.3f}, p = {p:.3g} using {method}", fontsize=14)
    plt.tight_layout()
    plt.show()

    return r, p, neural_rsm, social_rsm



if __name__ == "__main__":
    # Setup
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

    behavior_matrices = {
        name: pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy().T if 'From' in name
        else pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy()
        for name, fname in behavior_files.items()
    }

    # Load spike windows and compute spike counts
    cells_df = pd.read_excel('all_anova_passed_cells.xlsx')
    spike_df = extract_spike_counts_from_windows(cells_df)
    # exploded_df = prepare_exploded_spike_data("2023-09-26", 1, True)
    # print(exploded_df)
    # spike_df = compute_mean_spike_rate_table(exploded_df)
    #
    # subject_monkey_index = 6
    # all_results = []
    # for name, mat in behavior_matrices.items():
    #     results_df = run_directional_vector_linear_regression(
    #         spike_df, mat, name, monkey_group_name, monkey_list, subject_monkey_index, use_spikerate= True
    #     )
    #     # plot_clustered_neurons(results_df, name)
    #     plot_heatmap_r_squared(results_df, name)
    #     all_results.append(results_df)
    #
    # final_df = pd.concat(all_results, ignore_index=True)
    # print(final_df.head())

    rsa_results = []
    for name, mat in behavior_matrices.items():
        r, p, neural_rsm, social_rsm = run_rsa_analysis(
            spike_df=spike_df,
            behavior_matrix=mat,
            monkey_list=monkey_list,
            subject_idx=6,
            method='correlation',
            use_rate = True
        )

        rsa_results.append({
            'Behavior': name,
            'RSA_r': r,
            'RSA_p': p
        })

    rsa_df = pd.DataFrame(rsa_results)
    print(rsa_df.sort_values(by='RSA_r', ascending=False))
