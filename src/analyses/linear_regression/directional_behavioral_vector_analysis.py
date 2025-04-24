import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.spike_count import extract_spike_counts_from_windows

def run_linear_regression_using_sklearn(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y)


def run_directional_vector_linear_regression(spike_df, behavior_matrix, behavior_name, group_name, monkey_list, subject_idx):
    results = []
    filtered_df = spike_df[(spike_df['MonkeyGroup'] == group_name) & (spike_df['MonkeyName'] != "NewMonkey")]
    mean_spikes = filtered_df.groupby(['NeuronID', 'MonkeyName'], as_index=False)['SpikeCount'].mean()
    spike_matrix = mean_spikes.pivot(index='NeuronID', columns='MonkeyName', values='SpikeCount')

    for neuron_id, row in spike_matrix.iterrows():
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_idx:
                continue
            y = np.delete(behavior_matrix[src_idx], [src_idx, subject_idx])
            x_row = row.drop(index=[monkey_list[src_idx], monkey_list[subject_idx]], errors='ignore')
            x = x_row.values.astype(float)
            if len(x) != len(y):
                print(f"Length mismatch for {neuron_id} (source: {monkey_list[src_idx]})")
                continue

            coeff, intercept, r_squared = run_linear_regression_using_sklearn(x, y)
            if r_squared > 0.25:
                print(f"--- {neuron_id} | {monkey_list[src_idx]} | R² = {r_squared:.3f}")
                results.append({
                    'NeuronID': neuron_id,
                    'Behavior': behavior_name,
                    'Source_Monkey': monkey_list[src_idx],
                    'R-squared': r_squared
                })

    return pd.DataFrame(results)



def run_rsa_analysis(spike_df, behavior_matrix, monkey_list, subject_idx, method='cosine'):
    """
    Run RSA comparing neural and behavioral similarity (excluding subject monkey).

    Parameters:
        spike_df (pd.DataFrame): contains ['NeuronID', 'MonkeyName', 'SpikeCount']
        behavior_matrix (np.ndarray): full social matrix (e.g., AffiliationTo)
        monkey_list (list): all monkeys (ordered like behavior_matrix)
        subject_idx (int): index of the subject monkey to exclude
        method (str): similarity metric ('correlation', 'cosine', etc.)

    Returns:
        r, p, neural_rsm, social_rsm
    """

    # Step 0: Remove subject monkey
    subject_monkey = monkey_list[subject_idx]
    filtered_monkeys = [m for m in monkey_list if m != subject_monkey]

    # Filter neural data
    mean_df = (
        spike_df[spike_df['MonkeyName'].isin(filtered_monkeys)]
        .groupby(['NeuronID', 'MonkeyName'], as_index=False)['SpikeCount']
        .mean()
    )

    neural_matrix = mean_df.pivot(index='NeuronID', columns='MonkeyName', values='SpikeCount')
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
    r, p = pearsonr(neural_vec, social_vec)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(neural_rsm, ax=ax[0], cmap='viridis')
    ax[0].set_title("Neural RSM")
    sns.heatmap(social_rsm, ax=ax[1], cmap='viridis')
    ax[1].set_title("Social RSM")
    plt.suptitle(f"RSA: r = {r:.3f}, p = {p:.3g}", fontsize=14)
    plt.tight_layout()
    plt.show()

    return r, p, neural_rsm, social_rsm


def plot_heatmap_r_squared(df, behavior_type):
    """Heatmap: R² per NeuronID × Source Monkey."""
    pivot_df = df.pivot(index='NeuronID', columns='Source_Monkey', values='R-squared')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
    plt.title(f"R² Heatmap - {behavior_type}")
    plt.tight_layout()
    plt.show()

def plot_violin_r_squared(df, behavior_type):
    """Violin plot: distribution of R² per Source Monkey."""
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x='Source_Monkey', y='R-squared')
    plt.title(f"Distribution of R² per Source Monkey - {behavior_type}")
    plt.axhline(0.25, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_best_r2_bar(df, behavior_type):
    """Bar plot: each neuron's best-correlated source monkey."""
    best_df = df.sort_values('R-squared', ascending=False).drop_duplicates('NeuronID')
    plt.figure(figsize=(12, 5))
    sns.barplot(data=best_df, x='NeuronID', y='R-squared', hue='Source_Monkey')
    plt.title(f"Best R² per NeuronID by Source Monkey - {behavior_type}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_clustered_neurons_per_behavior(df, behavior_type):
    """Clustering neurons based on R² values across sources."""
    pivot_df = df.pivot(index='NeuronID', columns='Source_Monkey', values='R-squared').fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pivot_df)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(reduced)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap='tab10', s=60)
    plt.title(f"Neuron Clustering by R² Pattern {behavior_type}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Setup
    monkey_group_name = "Zombies"
    monkey_list = get_monkeys_by_default_order(monkey_group_name)
    base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'
    behavior_files = {
        "AffliationTo": "zombies_feature_df_affiliation.xlsx",
        "AffliationFrom": "zombies_feature_df_affiliation.xlsx",
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

    # subject_monkey_index = 6
    # all_results = []
    # for name, mat in behavior_matrices.items():
    #     results_df = run_directional_vector_linear_regression(
    #         spike_df, mat, name, monkey_group_name, monkey_list, subject_monkey_index
    #     )
    #     # plot_clustered_neurons(results_df, name)
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
            method='correlation'
        )

        rsa_results.append({
            'Behavior': name,
            'RSA_r': r,
            'RSA_p': p
        })

    rsa_df = pd.DataFrame(rsa_results)
    print(rsa_df.sort_values(by='RSA_r', ascending=False))


