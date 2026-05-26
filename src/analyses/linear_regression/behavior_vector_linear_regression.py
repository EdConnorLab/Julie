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
):
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
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared

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
        permutation_test = False,
        n_perm = 10000,
        random_state=42):
    """
    Run directional vector linear regression at the window level.

    For each neuron × window × source_monkey, regresses spike rates
    (across stimulus monkeys) against the source monkey's behavior vector.

    Returns one row per regression with full context:
        NeuronID, WindowStart_ms, WindowEnd_ms, Behavior, Source_Monkey,
        Stimulus_Monkeys (list), Behavior_Vector (raw), Neural_Response (raw),
        R_squared, coef, intercept, p_value, p_perm, Model
    """

    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    filtered_df = spike_df[(spike_df['MonkeyGroup'] == group_name) & (spike_df['MonkeyName'] != "NewMonkey")]

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

    for unit_id, row in tqdm(spike_matrix.iterrows(), total=len(spike_matrix),
                             desc="Window-level directional regression", dynamic_ncols=True):
        meta = id_map[id_map['NeuronWindowID'] == unit_id].iloc[0]
        neuron_id = meta['NeuronID']
        win_start = meta['WindowStart_ms']
        win_end = meta['WindowEnd_ms']

        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue

            # Raw behavior vector (excluding source and subject)
            raw_behavior = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_behavior, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            behavior_vec = raw_behavior[mask]

            # Stimulus monkeys and their spike rates
            stimulus_monkeys = [m for i, m in enumerate(monkey_list) if i not in (src_idx, subject_idx)]
            neural_response = row[stimulus_monkeys].values.astype(float)

            # Variance check
            if np.var(neural_response) < 1e-3 or np.var(behavior_vec) < 1e-3:
                continue
            if len(neural_response) != len(behavior_vec):
                print(f"Length mismatch for {unit_id} (source: {src_monkey})")
                continue

            try:
                # Z-score for regression
                x = zscore(behavior_vec)
                y = zscore(neural_response)
                X_design = sm.add_constant(x)

                model = sm.OLS(y, X_design).fit()
                r_squared = model.rsquared

                # Permutation test
                if permutation_test:
                    rng = np.random.default_rng(random_state)
                    null_dist = np.array([
                        sm.OLS(y, sm.add_constant(rng.permutation(x))).fit().rsquared
                        for _ in range(n_perm)
                    ])
                    p_perm = (np.sum(null_dist >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None


                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': win_start,
                    'WindowEnd_ms': win_end,
                    'Behavior': behavior_name,
                    'Source_Monkey': src_monkey,
                    'Stimulus_Monkeys': stimulus_monkeys,
                    'Behavior_Vector': behavior_vec,
                    'Neural_Response': neural_response,
                    'R_squared': r_squared,
                    'coef': model.params[1],
                    'intercept': model.params[0],
                    'p_value': model.pvalues[1],
                    'p_perm': p_perm,
                })

            except Exception as e:
                print(f"Error for {unit_id}, Monkey {src_monkey}: {e}")
                continue

    return pd.DataFrame(results)


def run_directional_vector_linear_regression_cell_level(
        spike_df,
        behavior_matrix,
        behavior_name,
        group_name,
        monkey_list,
        subject_idx,
        use_spikerate=False,
        plot=False,
        permutation_test=False,
        n_perm=10000,
        random_state=42):

    results = []
    value_col = 'MeanSpikeRate' if use_spikerate else 'SpikeCount'
    filtered_df = spike_df[(spike_df['MonkeyGroup'] == group_name) & (spike_df['MonkeyName'] != "NewMonkey")]
    mean_spikes = filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)[value_col].mean()
    spike_matrix = mean_spikes.pivot(index='NeuronID', columns='MonkeyName', values=value_col)
    spike_matrix = spike_matrix.reindex(columns=monkey_list)

    for neuron_id, row in tqdm(spike_matrix.iterrows(), total=len(spike_matrix),
                               desc="Cell-level directional regression"):
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue

            # Raw behavior vector (excluding source and subject)
            raw_behavior = behavior_matrix[src_idx].copy()
            mask = np.ones_like(raw_behavior, dtype=bool)
            mask[[src_idx, subject_idx]] = False
            behavior_vec = raw_behavior[mask]

            # Stimulus monkeys and their spike rates
            stimulus_monkeys = [m for i, m in enumerate(monkey_list) if i not in (src_idx, subject_idx)]
            neural_response = row[stimulus_monkeys].values.astype(float)

            # Variance check
            if np.var(neural_response) < 1e-3 or np.var(behavior_vec) < 1e-3:
                continue
            if len(neural_response) != len(behavior_vec):
                print(f"Length mismatch for {neuron_id} (source: {src_monkey})")
                continue

            try:
                # Z-score for regression
                x = zscore(behavior_vec)
                y = zscore(neural_response)
                X_design = sm.add_constant(x)


                model = sm.OLS(y, X_design).fit()
                r_squared = model.rsquared


                # Permutation test
                if permutation_test:
                    rng = np.random.default_rng(random_state)
                    null_dist = np.array([
                        sm.OLS(y, sm.add_constant(rng.permutation(x))).fit().rsquared
                        for _ in range(n_perm)
                    ])
                    p_perm = (np.sum(null_dist >= r_squared) + 1) / (n_perm + 1)
                else:
                    p_perm = None

                results.append({
                    'NeuronID': neuron_id,
                    'Behavior': behavior_name,
                    'Source_Monkey': src_monkey,
                    'Stimulus_Monkeys': stimulus_monkeys,
                    'Behavior_Vector': behavior_vec,
                    'Neural_Response': neural_response,
                    'R_squared': r_squared,
                    'coef': model.params[1],
                    'intercept': model.params[0],
                    'p_value': model.pvalues[1],
                    'p_perm': p_perm,
                })

                if plot and r_squared and r_squared > 0.6:
                    y_pred = model.predict(X_design)
                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, label='True', color='blue', alpha=0.6)
                    plt.plot(np.sort(x), y_pred[np.argsort(x)], label='Fit', color='black', alpha=0.6)
                    plt.xlabel(f'{behavior_name} {src_monkey} (z-scored)')
                    plt.ylabel('Neural Response (z-scored)')
                    plt.title(f'{neuron_id} (R²={r_squared:.3f})')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                print(f"Error for Neuron {neuron_id}, Monkey {src_monkey}: {e}")
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



def flatten_regression_results(df):
    """
    Flatten regression results from one-row-per-regression to one-row-per-stimulus-monkey.

    Works for both cell-level and window-level DataFrames (auto-detects window columns).

    Input schema (per row):
        Stimulus_Monkeys: list[str], Behavior_Vector: np.array, Neural_Response: np.array,
        plus scalar columns (NeuronID, Behavior, Source_Monkey, R_squared, coef, etc.)

    Output schema (per row):
        NeuronID, Behavior, Source_Monkey, Stimulus_Monkey, Behavior_Value, MeanSpikeRate,
        R_squared, coef, intercept, p_value, p_perm
        (+ WindowStart_ms, WindowEnd_ms if window-level)
    """
    has_windows = 'WindowStart_ms' in df.columns

    scalar_cols = ['NeuronID', 'Behavior', 'Source_Monkey',
                   'R_squared', 'coef', 'intercept', 'p_value', 'p_perm']
    if has_windows:
        scalar_cols = ['NeuronID', 'WindowStart_ms', 'WindowEnd_ms'] + scalar_cols[1:]

    rows = []
    for _, reg_row in df.iterrows():
        stim_monkeys = reg_row['Stimulus_Monkeys']
        behavior_vec = reg_row['Behavior_Vector']
        neural_resp = reg_row['Neural_Response']

        base = {col: reg_row[col] for col in scalar_cols}

        for stim_monkey, bval, spike_rate in zip(stim_monkeys, behavior_vec, neural_resp):
            row = {**base, 'Stimulus_Monkey': stim_monkey,
                   'Behavior_Value': bval, 'MeanSpikeRate': spike_rate}
            rows.append(row)
    return pd.DataFrame(rows)



def run_rsa_analysis(spike_df, behavior_matrix, behavior_name, monkey_list, subject_idx, method='correlation', use_rate = False):
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
    from sklearn.metrics.pairwise import cosine_similarity

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
    neural_rsm = pd.DataFrame(
        cosine_similarity(neural_matrix.T),
        index=filtered_monkeys,
        columns=filtered_monkeys
    )

    # Social RSM
    social_rsm = pd.DataFrame(
        cosine_similarity(behavior_matrix.T),
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
    cells_df = pd.read_pickle('/old/old_analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl')
    # spike_df = compute_mean_spike_rate_for_cells(cells_df)
    '''
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
'''
    np.set_printoptions(linewidth=np.inf)
    # RSA
    rsa_results = []
    for name, mat in behavior_matrices.items():
        r, p, neural_rsm, social_rsm = run_rsa_analysis(
            spike_df=spike_df,
            behavior_matrix=mat,
            behavior_name = name,
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
        social_dissim = 1 - social_rsm.values
        print(name)
        print(social_rsm)
        from sklearn.manifold import MDS

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        social_coords = mds.fit_transform(social_dissim)

        monkey_list_without_subject = monkey_list[:6] + monkey_list[7:]
        plt.figure(figsize=(6, 6))
        plt.scatter(social_coords[:, 0], social_coords[:, 1])

        for i, mon in enumerate(monkey_list_without_subject):
            plt.text(social_coords[i, 0], social_coords[i, 1], mon, fontsize=9)

        plt.title(f"MDS projection of social similarity for {name} ")
        plt.xlabel("MDS1")
        plt.ylabel("MDS2")
        plt.axis('equal')
        plt.show()

    rsa_df = pd.DataFrame(rsa_results)
    print(rsa_df.sort_values(by='RSA_r', ascending=False))
