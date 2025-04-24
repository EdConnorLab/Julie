import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from analyses.enums.monkey_names import Zombies, BestFrans
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from analyses.spike_count import extract_spike_counts_from_windows


def run_linear_regression_using_sklearn(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression().fit(x,y)
    return  model.coef_[0], model.intercept_, model.score(x, y) # coeff, intercept, r_squared


def run_directional_vector_linear_regression(all_spike_counts, behavior_table, behavior_name, monkey_group_name, monkey_list, subject_monkey_index):
    results = []
    all_spike_counts = all_spike_counts[all_spike_counts['MonkeyGroup'] == monkey_group_name]
    all_spike_counts= all_spike_counts[all_spike_counts['MonkeyName'] != "NewMonkey"]
    mean_spike_counts = all_spike_counts.groupby(['NeuronID', 'MonkeyName'], as_index=False)['SpikeCount'].mean()
    mean_spike_counts.rename(columns={'SpikeCount': 'MeanSpikeCount'}, inplace=True)
    spike_matrix = mean_spike_counts.pivot(index='NeuronID', columns='MonkeyName', values='MeanSpikeCount')

    for neuron_id, row in spike_matrix.iterrows():
        for src_idx, src_monkey in enumerate(monkey_list):
            if src_idx == subject_monkey_index:
                continue
            # Get y vector: source monkey’s behavior toward all others
            y = np.delete(behavior_table[src_idx, :], [src_idx, subject_monkey_index])
            # Get x vector: spike count of this neuron in response to those same monkeys
            spike_row = row.drop(index=[monkey_list[src_idx], monkey_list[subject_monkey_index]], errors='ignore')
            x = spike_row.values.astype(float)
            # Make sure lengths match
            if len(x) != len(y):
                print(f"Mismatch in lengths for {neuron_id} (source: {monkey_list[src_idx]})")
                continue  # skip if mismatch
            coeff, intercept, r_squared = run_linear_regression_using_sklearn(x, y)
            if r_squared > 0.25:
                print(f"--- Regression for {neuron_id} (source: {monkey_list[src_idx]}) --- R² = {r_squared:.3f}")
                results.append({
                    'NeuronID': neuron_id,
                    'Behavior': behavior_name,
                    'Source_Monkey': monkey_list[src_idx],
                    'R-squared': r_squared
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Zombies
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[-1]

    base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'
    zombies_affiliation_file_name = 'zombies_feature_df_affiliation.xlsx'
    zombies_submission_file_name = 'zombies_feature_df_submission.xlsx'
    zombies_agonism_file_name = 'zombies_feature_df_agonism.xlsx'

    zombies_affiliation_df = pd.read_excel(base_dir + zombies_affiliation_file_name)
    zombies_submission_df = pd.read_excel(base_dir + zombies_submission_file_name)
    zombies_agonism_df = pd.read_excel(base_dir + zombies_agonism_file_name)

    zombies_affiliation_to = zombies_affiliation_df.iloc[:, 1:].to_numpy()
    zombies_affiliation_from = zombies_affiliation_to.T

    zombies_submission_to = zombies_submission_df.iloc[:, 1:].to_numpy()
    zombies_submission_from = zombies_submission_to.T

    zombies_agonism_to = zombies_agonism_df.iloc[:, 1:].to_numpy()
    zombies_agonism_from = zombies_agonism_to.T

    # all_anova_passed_cells were created from combining
    # "/home/connorlab/Documents/GitHub/Julie/Cortana/Ed and ANOVA/Ed_window_cells_ANOVA_passed_Zombies.xlsx"
    # and
    # "/home/connorlab/Documents/GitHub/Julie/src/analyses/response_window_finder/window_cells_ANOVA_passed_to_keep.xlsx"
    cells_with_windows = pd.read_excel('all_anova_passed_cells.xlsx')
    all_spike_counts = extract_spike_counts_from_windows(cells_with_windows)
    subject_monkey_index = 6
    print(all_spike_counts)
    run_directional_vector_linear_regression(all_spike_counts, zombies_affiliation_to, "AffliationTo", "Zombies", zombies, subject_monkey_index)
