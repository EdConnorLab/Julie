import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.express as px
matplotlib.use("Qt5Agg")
# -------------------------------
# 1. Load and Filter Results
# -------------------------------

def load_significant_ols_results(filename):
    results = pd.read_pickle(filename)
    significant_results = results[results['p_value'] < 0.05]
    print(f"{significant_results['NeuronID'].nunique()} significant neurons found")
    return significant_results

# -------------------------------
# 2. Filter Table for Pivot
# -------------------------------

def filter_table_for_pivot(df, column):
    if column in ['Behavior', 'Source_Monkey']:
        filtered_df = (
            df.sort_values('R-squared', ascending=False)
              .drop_duplicates(['NeuronID', column])
              .reset_index(drop=True)
        )
    elif column == 'Behavior_Source':
        df = df.copy()
        df['Behavior_Source'] = df['Behavior'] + '__' + df['Source_Monkey']
        filtered_df = (
            df.sort_values('R-squared', ascending=False)
              .drop_duplicates(['NeuronID', 'Behavior_Source'])
              .reset_index(drop=True)
        )
    else:
        raise ValueError("column must be one of 'Behavior', 'Source_Monkey', or 'Behavior_Source'")
    return filtered_df

# -------------------------------
# 3. Make Pivot Table
# -------------------------------

def make_pivot_table(filtered_df, column, min_nonzero=2):
    pivot_df = filtered_df.pivot(index='NeuronID', columns=column, values='R-squared').fillna(0)
    pivot_df = pivot_df[(pivot_df > 0).sum(axis=1) >= min_nonzero]
    return pivot_df

# -------------------------------
# 4. PCA with Scree/2D/3D Plot
# -------------------------------

def make_pca_model_and_projection(pivot_df, n_components=None, plot='scree'):
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(pivot_df)

    if plot == 'scree':
        var_ratio = pca.explained_variance_ratio_
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(var_ratio) + 1), var_ratio.cumsum(), marker='o')
        plt.title("PCA Scree Plot (Cumulative Variance Explained)")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif plot == '2d' and pca_coords.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], s=40)
        plt.title("PCA Projection (2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif plot == '3d' and pca_coords.shape[1] >= 3:
        fig = px.scatter_3d(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            z=pca_coords[:, 2],
            labels={"x": "PC1", "y": "PC2", "z": "PC3"},
            title="PCA Projection (3D - Interactive)",
            opacity=0.7
        )
        fig.show()

    return pca, pca_coords

# -------------------------------
# 5. t-SNE
# -------------------------------

def run_tsne(data, n_components=2, perplexity=30, random_state=42, plot=True):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='pca')
    tsne_coords = tsne.fit_transform(data)

    if plot and n_components == 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=40)
        plt.title("t-SNE Projection (2D)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return tsne_coords

# -------------------------------
# 6. KMeans Clustering
# -------------------------------

def run_kmeans(data, n_clusters=6, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=30)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels, kmeans

# -------------------------------
# 7. Cluster Visualization
# -------------------------------

def plot_clusters(coords, cluster_labels, title="Cluster Plot", annotate=False):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=cluster_labels, palette='tab10', s=60)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if annotate:
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, str(i), fontsize=8, alpha=0.6)

    plt.show()

# -------------------------------
# 8. Meta Info per Cluster
# -------------------------------


def plot_dominant_meta_per_cluster_subplots(pivot_df, cluster_labels, ncols=3):
    cluster_series = pd.Series(cluster_labels)
    pivot_df_clustered = pivot_df.copy()
    pivot_df_clustered['Cluster'] = cluster_labels
    n_clusters = cluster_series.nunique()
    # Figure setup
    ncols = ncols
    nrows = (n_clusters + ncols - 1) // ncols  # 자동 계산

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)

    axes = axes.flatten()
    for i, cluster_id in enumerate(sorted(cluster_series.unique())):
        ax = axes[i] if n_clusters > 1 else axes

        cluster_df = pivot_df_clustered[pivot_df_clustered['Cluster'] == cluster_id].drop(columns='Cluster')
        mean_r2 = cluster_df.mean().sort_values(ascending=True)

        sns.barplot(x=mean_r2.values, y=mean_r2.index, ax=ax)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("Mean R²")
        if i == 0:
            ax.set_ylabel("Behavior")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_behavior_distribution_per_cluster(df, cluster_labels, behavior_col='Behavior', behavior_map=None):
    """
    Plot stacked bar charts showing behavior type and directionality per cluster.

    Parameters:
    - df: DataFrame with behavior and cluster info
    - behavior_col: name of the column with behavior strings like 'AffliationFrom'
    - cluster_col: name of the cluster label column
    - behavior_map: dict mapping behavior string to (BehaviorType, Direction)
    """
    if behavior_map is None:
        behavior_map = {
            'AffliationFrom': ('Affliation', 'From'),
            'AffliationTo':   ('Affliation', 'To'),
            'SubmissionFrom': ('Submission', 'From'),
            'SubmissionTo':   ('Submission', 'To'),
            'AgonismFrom':    ('Agonism', 'From'),
            'AgonismTo':      ('Agonism', 'To'),
        }

    df = df.copy()
    df["Cluster"] = cluster_labels
    df['BehaviorType'] = df[behavior_col].map(lambda x: behavior_map.get(x, ('Unknown', 'None'))[0])
    df['Direction'] = df[behavior_col].map(lambda x: behavior_map.get(x, ('Unknown', 'None'))[1])

    # Type distribution
    type_by_cluster = pd.crosstab(df['Cluster'], df['BehaviorType'], normalize='index')
    type_by_cluster.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set2')
    plt.title("Behavior Type Distribution per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Direction distribution
    direction_by_cluster = pd.crosstab(df['Cluster'], df['Direction'], normalize='index')
    direction_by_cluster.plot(kind='bar', stacked=True, figsize=(6, 5), colormap='Set1')
    plt.title("Directionality (From vs To) per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.legend(title="Direction", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -------------------------------
# 11. Cross Factor R² Distribution Per Cluster
# -------------------------------
def plot_cross_meta_from_filtered_df(filtered_df, pivot_df, cluster_labels, value_col='R-squared', category_col=None, title_prefix="Cluster", color='teal'):
    """
    Given a filtered_df with 'NeuronID', merge in cluster info and plot cross-meta average R² per cluster.

    Parameters:
        filtered_df: long-form DataFrame with 'NeuronID' and [value_col, category_col]
        pivot_df: wide-form pivot_df used to generate cluster_labels
        cluster_labels: cluster assignments (same order as pivot_df)
        value_col: column with R² values
        category_col: feature (e.g., Behavior or SourceMonkey) to aggregate over
        title_prefix: subplot title prefix
        color: bar color
    """
    if category_col is None:
        raise ValueError("You must specify category_col (e.g., 'Behavior' or 'Source_Monkey')")

    # Step 1: attach cluster info
    pivot_df_clustered = pivot_df.copy().reset_index()
    pivot_df_clustered['Cluster'] = cluster_labels
    merged_df = filtered_df.merge(
        pivot_df_clustered[['NeuronID', 'Cluster']],
        on='NeuronID', how='left'
    )

    # Step 2: group and pivot
    group_df = merged_df.groupby(['Cluster', category_col])[value_col].mean().reset_index()
    pivot_mean = group_df.pivot(index='Cluster', columns=category_col, values=value_col).fillna(0)

    # Step 3: subplot plotting
    n_clusters = pivot_mean.shape[0]
    # Figure setup
    ncols = 3
    nrows = (n_clusters + ncols - 1) // ncols  # 자동 계산

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)

    axes = axes.flatten()
    for i, (cluster_id, row) in enumerate(pivot_mean.iterrows()):
        ax = axes[i]
        row_sorted = row.sort_values(ascending=True)
        sns.barplot(x=row_sorted.values, y=row_sorted.index, ax=ax, color=color)
        ax.set_title(f"{title_prefix} {cluster_id}")
        ax.set_xlabel("Mean R²")
        ax.set_ylabel(category_col if i == 0 else "")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = load_significant_ols_results('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results'
                                 '/good_neurons_directional_linear_regression_on_single_neurons.pkl')
    filtered_df = filter_table_for_pivot(df, column='Source_Monkey')
    # Source-monkey-centered
    pivot_df = make_pivot_table(filtered_df, column='Source_Monkey')


    # Visualize 2D PCA
    # pca, pca_coords = make_pca_model_and_projection(pivot_df, n_components=3, plot='2d')

    # # Only look at Scree plot
    _, pca_coords = make_pca_model_and_projection(pivot_df, plot='scree')
    # t-SNE after PCA
    tsne_coords = run_tsne(pca_coords, perplexity=20)
    cluster_labels, kmeans_model = run_kmeans(tsne_coords, n_clusters=8)

    # Align filtered_df with pivot_df
    filtered_df_unique = (
        filtered_df[filtered_df['NeuronID'].isin(pivot_df.index)]
        .drop_duplicates(subset='NeuronID')
        .sort_values('NeuronID')
        .reset_index(drop=True)
    )
    plot_clusters(tsne_coords, cluster_labels, title="t-SNE K-means Cluster Plot")
    plot_dominant_meta_per_cluster_subplots(pivot_df, cluster_labels)
    plot_cross_meta_from_filtered_df(filtered_df_unique, pivot_df, cluster_labels, category_col='Behavior')
    plot_behavior_distribution_per_cluster(filtered_df_unique, cluster_labels)



