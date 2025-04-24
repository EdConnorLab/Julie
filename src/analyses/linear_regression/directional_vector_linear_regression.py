import numpy as np
import pandas as pd
import os
from analyses.enums.monkey_names import Zombies, get_monkeys_by_rank
from analyses.spike_count import extract_spike_counts_from_windows
from sklearn.linear_model import LinearRegression


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


if __name__ == "__main__":
    # Setup
    monkey_group_name = "Zombies"
    monkey_list = get_monkeys_by_rank(monkey_group_name)
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

    subject_monkey_index = 6

    for name, mat in behavior_matrices.items():
        results_df = run_directional_vector_linear_regression(
            spike_df, mat, name, monkey_group_name, monkey_list, subject_monkey_index
        )
        print(results_df.head())
