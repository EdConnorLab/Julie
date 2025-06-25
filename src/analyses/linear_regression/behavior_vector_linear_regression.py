import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, zscore, spearmanr
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import plotly.express as px

from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.spike_count import extract_spike_counts_from_windows, load_exploded_data_from_cache
from analyses.spike_rate import compute_mean_spike_rate_table, compute_mean_spike_rate_for_cells


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
    for unit_id, row in tqdm(spike_matrix.iterrows(), total=len(spike_matrix), desc="Computing window-level linear regression with directional vectors"):
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

    for neuron_id, row in tqdm(spike_matrix.iterrows(), total=len(spike_matrix), desc="Computing cell-level linear regression with directional vectors"):
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

def run_stimulus_centric_pls_analysis(
    spike_df,
    behavior_matrices,
    behavior_names,
    group_name,
    monkey_list,
    subject_idx,
    use_spikerate=True,
    n_components=3
):
    """
    PLS analysis where behavioral predictors are based on stimulus monkeys.

    Parameters:
        spike_df: DataFrame with ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'SpikeCount' or 'MeanSpikeRate']
        behavior_matrices: dict of {behavior_name: np.ndarray} (10×10 directional matrices)
        behavior_names: list of behavior keys (e.g., 'AffiliationTo', 'AffiliationFrom', etc.)
        group_name: monkey group (e.g., 'Zombies')
        monkey_list: ordered list of 10 monkeys in the behavior matrices
        subject_idx: index of the subject monkey (to exclude)
        use_spikerate: if True, use 'MeanSpikeRate' column in spike_df
        n_components: number of PLS components

    Returns:
        pls_model: fitted PLSRegression model
        X_df: behavioral feature matrix (stimuli × features)
        Y_df: neural response matrix (stimuli × neurons)
        stimulus_monkeys: list of stimulus monkey names
    """

    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    subject_monkey = monkey_list[subject_idx]

    # Stimulus monkeys: all except subject
    stimulus_idxs = [i for i in range(len(monkey_list)) if i != subject_idx]
    stimulus_monkeys = [monkey_list[i] for i in stimulus_idxs]

    # Step 1: Build behavioral predictor matrix X (stimulus-centric)
    X_dicts = []

    for stim_idx in stimulus_idxs:
        stim_monkey = monkey_list[stim_idx]
        feature_dict = {}

        for bname in behavior_names:
            mat = behavior_matrices[bname]

            if 'To' in bname:
                vec = mat[:, stim_idx]  # other → stim
                ids = monkey_list
            elif 'From' in bname:
                vec = mat[stim_idx, :]  # stim → other
                ids = monkey_list
            else:
                raise ValueError(f"Behavior name must contain 'To' or 'From': {bname}")

            # Exclude subject and self
            mask = np.ones(len(monkey_list), dtype=bool)
            mask[subject_idx] = False
            mask[stim_idx] = False
            vec = vec[mask]
            ids_filtered = [id_ for i, id_ in enumerate(ids) if mask[i]]

            # Normalize and name
            zvec = zscore(vec)
            for target_id, val in zip(ids_filtered, zvec):
                colname = f"{bname}_{target_id}"
                feature_dict[colname] = val

        X_dicts.append(feature_dict)

    # Create DataFrame with consistent columns across all rows
    X_df = pd.DataFrame(X_dicts, index=stimulus_monkeys).fillna(0)

    # Step 2: Build neural response matrix Y (stimulus → neuron mean firing)
    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'].isin(stimulus_monkeys))
    ]
    value_table = filtered_df.groupby(['NeuronID', 'MonkeyName'])[value_col].mean().unstack()
    value_table = value_table.loc[:, stimulus_monkeys]  # ensure correct order

    # Z-score across stimulus monkeys (per neuron)
    Y = zscore(value_table.T, axis=0)
    Y_df = pd.DataFrame(Y, index=stimulus_monkeys, columns=value_table.index)

    # Step 3: PLS regression
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_df)
    Y_scaled = scaler_Y.fit_transform(Y_df)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    return pls, X_df, Y_df, stimulus_monkeys



def run_directional_vector_pls_analysis_cell_level(
    spike_df,
    behavior_matrices,
    behavior_names,
    group_name,
    monkey_list,
    subject_idx,
    use_spikerate=False,
    n_components=3
):
    """
    Run PLS analysis across neurons using directional behavior vectors as predictors.

    Parameters:
        spike_df: pd.DataFrame with ['NeuronID', 'MonkeyName', 'MonkeyGroup', 'SpikeCount' or 'MeanSpikeRate']
        behavior_matrices: dict of {behavior_name: np.ndarray} matrices (n_monkeys x n_monkeys)
        behavior_names: list of behavior names to include
        group_name: str, monkey group name (e.g., "Zombies")
        monkey_list: list of monkey names
        subject_idx: index of subject monkey (to exclude)
        use_spikerate: whether to use 'MeanSpikeRate' instead of 'SpikeCount'
        n_components: number of PLS components

    Returns:
        pls_model: fitted PLSRegression model
        X_df: DataFrame (n_samples x n_behaviors)
        Y_df: DataFrame (n_samples x n_neurons)
        sample_info: DataFrame with ['Source_Monkey', 'StimulusMonkey']
    """

    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    subject_monkey = monkey_list[subject_idx]
    monkeys = [m for m in monkey_list if m != subject_monkey]
    idxs = [i for i in range(len(monkey_list)) if i != subject_idx]

    # Step 1: build behavior predictor matrix (X)
    X_list = []
    sample_info = []
    for src_idx, src_monkey in enumerate(monkey_list):
        if src_idx == subject_idx:
            continue
        row = []
        mask = np.ones(len(monkey_list), dtype=bool)
        mask[[subject_idx, src_idx]] = False  # exclude self + subject

        for bname in behavior_names:
            mat = behavior_matrices[bname]
            vec = mat[src_idx][mask]  # directional vector from src
            row.append(zscore(vec))
        X_list.extend(np.stack(row, axis=1))  # append each stimulus monkey's slice
        for stim_idx, stim_monkey in enumerate(monkey_list):
            if stim_idx in (subject_idx, src_idx):
                continue
            sample_info.append({'Source_Monkey': src_monkey, 'StimulusMonkey': stim_monkey})
    X = np.vstack(X_list)
    X_df = pd.DataFrame(X, columns=behavior_names)

    # Step 2: build neural response matrix (Y)
    filtered_df = spike_df[
        (spike_df['MonkeyGroup'] == group_name) &
        (spike_df['MonkeyName'].isin(monkeys))
    ]
    value_table = filtered_df.groupby(['NeuronID', 'MonkeyName'])[value_col].mean().unstack()
    neuron_ids = value_table.index.tolist()

    Y_list = []
    for row in sample_info:
        stim_monkey = row['StimulusMonkey']
        y_row = value_table[stim_monkey].values  # all neurons' responses to this monkey
        Y_list.append(zscore(y_row))  # z-score per stimulus
    Y = np.vstack(Y_list)
    Y_df = pd.DataFrame(Y, columns=neuron_ids)

    # Step 3: run PLS
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_df)
    Y_scaled = scaler_Y.fit_transform(Y_df)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    return pls, X_df, Y_df, pd.DataFrame(sample_info)


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
                'p_value': row['p_value'],
                'p_perm': row['p_perm']
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
                'p_value': row['p_value'],
                'p_perm': row['p_perm']
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




def plot_pls_component_by_source(full_sample_df, component='XComp1'):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=full_sample_df, x='Source_Monkey', y=component)
    plt.title(f"{component} distribution by Source Monkey")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def scatter_pls_scores_by_source(full_sample_df, comp_x='XComp1', comp_y='XComp2'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=full_sample_df, x=comp_x, y=comp_y,
        hue='Source_Monkey', palette='tab10', s=70
    )
    plt.title(f"{comp_x} vs {comp_y}")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def scatter_3d_pls_by_stimulus_with_grid(full_sample_df):
    fig = px.scatter_3d(
        full_sample_df,
        x='XComp1',
        y='XComp2',
        z='XComp3',
        color='StimulusMonkey',
        title='PLS Component Space (colored by StimulusMonkey)',
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=5))

    # 축 스타일: grid on, zero lines, background
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='XComp1',
                showgrid=True,
                zeroline=True,
                backgroundcolor="rgba(240,240,240,0.95)"
            ),
            yaxis=dict(
                title='XComp2',
                showgrid=True,
                zeroline=True,
                backgroundcolor="rgba(240,240,240,0.95)"
            ),
            zaxis=dict(
                title='XComp3',
                showgrid=True,
                zeroline=True,
                backgroundcolor="rgba(240,240,240,0.95)"
            )
        )
    )
    fig.show()


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
    cells_df = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl')
    spike_df = compute_mean_spike_rate_for_cells(cells_df)
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

    ## Testing PLS
    pls_model, X_df, Y_df, info_df = run_directional_vector_pls_analysis_cell_level(
        spike_df=spike_df,
        behavior_matrices=behavior_matrices,
        behavior_names=["AffiliationTo", "AffiliationFrom", "SubmissionTo", "SubmissionFrom", "AgonismTo",
                        "AgonismFrom"],
        group_name="Zombies",
        monkey_list=monkey_list,
        subject_idx=6,
        use_spikerate=True,
        n_components=3
    )
    # 각 behavior의 salience
    salience = pd.DataFrame(pls_model.x_weights_, index=X_df.columns,
                 columns=[f"Comp{i + 1}" for i in range(pls_model.n_components)])

    # 각 neuron의 loading
    loading = pd.DataFrame(pls_model.y_weights_, index=Y_df.columns,
                 columns=[f"Comp{i + 1}" for i in range(pls_model.n_components)])
    print('----------------------------------')
    print(salience)
    print(loading)

    x_weights = pd.DataFrame(
        pls_model.x_weights_,
        index=X_df.columns,
        columns=[f"Comp{i + 1}" for i in range(pls_model.n_components)]
    )
    print("🧭 Behavior (X) weights:")
    print(x_weights.round(3))

    y_weights = pd.DataFrame(
        pls_model.y_weights_,
        index=Y_df.columns,
        columns=[f"Comp{i + 1}" for i in range(pls_model.n_components)]
    )
    print("🧠 Neuron (Y) weights:")
    print(y_weights.round(3))

    Y_pred = pls_model.predict(X_df)
    explained_var_per_neuron = np.var(Y_pred, axis=0) / np.var(Y_df.values, axis=0)
    print("📊 Variance explained per neuron (first few):")
    print(pd.Series(explained_var_per_neuron, index=Y_df.columns).round(3).sort_values(ascending=False).head())

    print(f"🔍 Mean explained variance across neurons: {explained_var_per_neuron.mean():.3f}")

    X_scores = pd.DataFrame(pls_model.x_scores_, columns=[f"XComp{i + 1}" for i in range(pls_model.n_components)])
    Y_scores = pd.DataFrame(pls_model.y_scores_, columns=[f"YComp{i + 1}" for i in range(pls_model.n_components)])

    print("📌 First few sample scores (X):")
    print(X_scores.head())

    full_sample_df = pd.concat([info_df.reset_index(drop=True), X_scores], axis=1)
    print(full_sample_df.head())

    # plot_pls_component_by_source(full_sample_df, component='XComp1')
    # plot_pls_component_by_source(full_sample_df, component='XComp2')
    # plot_pls_component_by_source(full_sample_df, component='XComp3')

    # scatter_pls_scores_by_source(full_sample_df, 'XComp1', 'XComp2')
    scatter_3d_pls_by_stimulus_with_grid(full_sample_df)

    y_weights = pd.DataFrame(
        pls_model.y_weights_,
        index=Y_df.columns,
        columns=[f"Comp{i + 1}" for i in range(pls_model.n_components)]
    )

    plt.figure(figsize=(10, 5))
    sns.histplot(y_weights["Comp1"], bins=30, kde=True)
    plt.title("Distribution of Neuron Loadings on Component 1")
    plt.xlabel("Loading")
    plt.grid(True)
    plt.show()


    ## Testing PLS with stimulus-centered
    pls_model, X_df, Y_df, stim_monkeys = run_stimulus_centric_pls_analysis(
        spike_df=spike_df,
        behavior_matrices=behavior_matrices,
        behavior_names=["AffiliationTo", "AffiliationFrom", "SubmissionTo", "SubmissionFrom", "AgonismTo",
                        "AgonismFrom"],
        group_name="Zombies",
        monkey_list=monkey_list,
        subject_idx=6,
        use_spikerate=True,
        n_components=3
    )

    plot_behavioral_weights(pls_model, X_df)
    plot_stimuli_in_component_space(pls_model, stim_monkeys)
    print('RESULTS')
    print(X_df)
    print(Y_df)
    print(pls_model.x_weights_)
    print(pls_model.y_weights_)
    interpret_behavioral_weights(pls_model, X_df, component=0, top_n=15, plot=True)
    # RSA
    # rsa_results = []
    # for name, mat in behavior_matrices.items():
    #     r, p, neural_rsm, social_rsm = run_rsa_analysis(
    #         spike_df=spike_df,
    #         behavior_matrix=mat,
    #         monkey_list=monkey_list,
    #         subject_idx=6,
    #         method='correlation',
    #         use_rate = True
    #     )
    #
    #     rsa_results.append({
    #         'Behavior': name,
    #         'RSA_r': r,
    #         'RSA_p': p
    #     })
    #
    # rsa_df = pd.DataFrame(rsa_results)
    # print(rsa_df.sort_values(by='RSA_r', ascending=False))
