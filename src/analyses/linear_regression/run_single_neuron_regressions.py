import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from analyses.enums.monkey_names import get_monkeys_by_default_order
from analyses.linear_regression.directional_behavioral_vector_analysis import run_directional_vector_linear_regression
from analyses.spike_rate import compute_mean_spike_rate_for_cells


def plot_single_neuron_profile(df, neuron_id):
    """
    Plot the full social encoding profile for a single neuron.
    """
    df = df[df['NeuronID'] == neuron_id]

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x='Source_Monkey', y='R-squared', hue='Behavior')
    plt.title(f"Neuron: {neuron_id} — social encoding profile")
    plt.ylabel("R²")
    plt.xlabel("Source Monkey")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # monkey_group_name = "Zombies"
    # monkey_list = get_monkeys_by_default_order(monkey_group_name)
    # base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'
    # behavior_files = {
    #     "AffliationTo": "zombies_feature_df_affiliation.xlsx",
    #     "AffliationFrom": "zombies_feature_df_affiliation.xlsx",
    #     "SubmissionTo": "zombies_feature_df_submission.xlsx",
    #     "SubmissionFrom": "zombies_feature_df_submission.xlsx",
    #     "AgonismTo": "zombies_feature_df_agonism.xlsx",
    #     "AgonismFrom": "zombies_feature_df_agonism.xlsx",
    # }
    #
    # behavior_matrices = {
    #     name: pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy().T if 'From' in name
    #     else pd.read_excel(os.path.join(base_dir, fname)).iloc[:, 1:].to_numpy()
    #     for name, fname in behavior_files.items()
    # }
    # sig_neurons = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/Zombies_significant_neurons_pANOVAorGLM_passed.pkl')
    # mean_spike_rate = compute_mean_spike_rate_for_cells(sig_neurons)
    # subject_monkey_index = 6
    # all_results = []
    # for name, mat in behavior_matrices.items():
    #     results_df = run_directional_vector_linear_regression(
    #         mean_spike_rate, mat, name, monkey_group_name, monkey_list, subject_monkey_index, use_spikerate=True
    #     )
    #     all_results.append(results_df)
    # all_results_df = pd.concat(all_results, ignore_index=True)
    # all_results_sorted = all_results_df.sort_values(by=['p_value','R-squared'], ascending=False)
    # significant_results = all_results_sorted[all_results_sorted['p_value'] < 0.05]
    # significant_results.to_excel('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_neurons_only_significant.xlsx')
    # all_results_sorted.to_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_single_neurons.pkl')
    # print(all_results_sorted)
    significant_results = pd.read_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/directional_linear_regression_on_single_neurons.pkl')
    significant_results = significant_results[significant_results['p_value'] < 0.05]

    # Simple Visualization 1: Bar plot
    # significant_results = significant_results[significant_results['R-squared'] > 0.5]
    # for neuron_id in significant_results['NeuronID'].unique():
    #     plot_single_neuron_profile(significant_results, neuron_id)

    # Simple Visualization 2: Count plot
    # best_monkey_df = significant_results.sort_values('R-squared', ascending=False).drop_duplicates('NeuronID')
    # plt.figure(figsize=(10, 5))
    # sns.countplot(data=best_monkey_df, x='Source_Monkey', order=best_monkey_df['Source_Monkey'].value_counts().index)
    # plt.title("Most-explained Source Monkey per neuron (best R²)")
    # plt.xlabel("Source Monkey")
    # plt.ylabel("Neuron Count")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    '''
    뉴런 하나당 6가지의 behavior 가 있고 각 behavior 당 9마리의 source monkey가 가능하잖아 
    (of course, not all of these r-squared values would be p-value with <0.05. some of them would just be 0) 
    그럼 그냥 6 x 9 = 54 개의 value를 가진 vector로 표현할 순 없어? 각 뉴런을 ㅎ 그담에 시각화
    '''
    # Step 1: 먼저 전체 significant result를 behavior × source_monkey 조합으로 정리
    filtered_df = significant_results.sort_values('R-squared', ascending=False)

    # Step 2: pivot (NeuronID × [Behavior|SourceMonkey] 조합 → column 이름을 합쳐서 사용)
    filtered_df['Behavior_Source'] = filtered_df['Behavior'] + '__' + filtered_df['Source_Monkey']

    pivot_df = filtered_df.pivot(index='NeuronID', columns='Behavior_Source', values='R-squared').fillna(0)

    # Optional: thresholding
    pivot_df[pivot_df < 0.05] = 0

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    # t-sne
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca')
    tsne_coords = tsne.fit_transform(scaled_data)

    # 시각화
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], s=60)
    plt.title("t-SNE on 6×9 Behavioral Encoding per Neuron")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    # PCA fitting for 54 vector
    pca = PCA()
    pca.fit(scaled_data)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Scree plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_var) + 1), cumulative_var, marker='o')
    plt.title("PCA Scree Plot (Cumulative Explained Variance)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.xticks(range(0, 55, 5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # now, reduce the dimensions to 20 and then do t-sne
    pca = PCA(n_components=25, random_state=42)
    pca_reduced = pca.fit_transform(scaled_data)

    # 2. t-SNE on PCA-reduced data
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca')
    tsne_coords = tsne.fit_transform(pca_reduced)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], s=60)
    plt.title("t-SNE on 6×9 Behavioral Encoding per Neuron")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()
    '''각 behavior 별로.. 최대 R² 인 source monkey만 남기고 vector 화 -- 이거 중요!!! 이거 됨!!!!'''

    # # 중복 제거: NeuronID × Behavior 조합당 최대 R²만 남기기
    # filtered_df = (
    #     significant_results
    #     .sort_values('R-squared', ascending=False)
    #     .drop_duplicates(['NeuronID', 'Behavior'])
    # )
    # filtered_df = filtered_df.reset_index(drop=True)
    # # Step 1: pivot (NeuronID × Behavior) wide format 만들기
    # pivot_df = filtered_df.pivot(index='NeuronID', columns='Behavior', values='R-squared').fillna(0)
    #
    # # pivot_df[pivot_df < 0.05] = 0
    # pivot_df = pivot_df[(pivot_df > 0).sum(axis=1) >= 2]
    #
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(pivot_df)
    # # t-SNE 수행
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    # tsne_coords = tsne.fit_transform(scaled_data)
    #
    # # KMeans 클러스터링 (ex: 6개 클러스터로 시도)
    # kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    # cluster_labels = kmeans.fit_predict(tsne_coords)
    #
    # # 플롯에 색상 추가
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], hue=cluster_labels, palette='tab10', s=60)
    # plt.title("t-SNE + KMeans Clustering of Neurons")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()
    #
    # cluster_series = pd.Series(cluster_labels, index=pivot_df.index, name='Cluster')
    # pivot_df_clustered = pivot_df.copy()
    # pivot_df_clustered['Cluster'] = cluster_series
    #
    # # 2. Cluster 5만 추출
    # cluster_5 = pivot_df_clustered[pivot_df_clustered['Cluster'] == 1].drop(columns='Cluster')
    #
    # # 3. 각 behavior별 평균 R² 구하기
    # mean_r2 = cluster_5.mean().sort_values(ascending=False)
    #
    # # 4. Bar plot으로 시각화
    # plt.figure(figsize=(8, 5))
    # sns.barplot(x=mean_r2.values, y=mean_r2.index, color='chocolate')
    # plt.title("Mean R² per Behavior in Cluster 1")
    # plt.xlabel("Mean R²")
    # plt.ylabel("Behavior")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()