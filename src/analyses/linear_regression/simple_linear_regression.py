import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from analyses.enums.monkey_names import Zombies, BestFrans
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from analyses.response_window_processing import get_spike_count_for_single_neuron_with_time_window
from analyses.spike_count import extract_spike_counts_from_windows


def run_linear_regression_using_sklearn(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression().fit(x,y)
    return  model.coef_[0], model.intercept_, model.score(x, y) # coeff, intercept, r_squared



def plot_regression_scatter(x, y, coeff, intercept, r_squared, behavior_name, source_monkey, neuron_id, output_dir="regression_plots"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')

    # Regression line
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + coeff * x_vals
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeff[0]:.2f}x + {intercept[0]:.2f}')

    # Labels and title
    plt.xlabel('Mean Firing Rate')
    plt.ylabel(f'{behavior_name} Score')
    plt.title(f'Neuron {neuron_id} | {behavior_name} | {source_monkey}\nR² = {r_squared:.2f}')
    plt.legend()

    # Save plot
    filename = f"{behavior_name}_{source_monkey}_Neuron{neuron_id}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def run_linear_regression_analysis(spike_counts, behavior_table, behavior_name, subject_monkey_index):
    results = []
    for index, row in spike_counts.iterrows():
        date = row['Date']
        round_no = row['Round No.']
        time_window = row['Time Window']

        for sourcemonkey in range(len(zombies)):
            exclude_columns = ['Date', 'Round No.', 'Time Window']
            if sourcemonkey == subject_monkey_index:
                continue
            else:
                behavior_table_arr = behavior_table[sourcemonkey, :]
                y = np.delete(behavior_table_arr, [sourcemonkey, subject_monkey_index])
                exclude_columns.append(zombies[sourcemonkey])
                x = [value for key, value in row.items() if
                     key not in exclude_columns and isinstance(value, (int, float))]
            coeff, intercept, r_squared = run_linear_regression_using_sklearn(x, y)
            plot_regression_scatter(
                x=np.array(x),
                y=np.array(y),
                coeff=coeff,
                intercept=intercept,
                r_squared=r_squared,
                behavior_name=behavior_name,
                source_monkey=zombies[sourcemonkey],
                neuron_id=index
            )
            if r_squared > 0.25:
                print("")
                print(
                    f"-------- Linear Regression Results for {date} Round No.{round_no} {time_window} {index} -------")
                print(
                    f"------------------------------------{zombies[sourcemonkey]}--------------------------------------")
                print(f"r-squared: {r_squared}")
                results.append({
                    'Date': date,
                    'Round No.': round_no,
                    'Cell': str(index),
                    'Time Window': time_window,
                    'Behavior': behavior_name,
                    'Source_Monkey': zombies[sourcemonkey],
                    'R-squared': r_squared
                })
    results_df = pd.DataFrame(results)
    # print(results_df)
    # results_df.to_excel(behavior_name + '.xlsx')
    return results_df


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
    print(all_spike_counts)
    all_spike_counts.columns = all_spike_counts.columns.astype(str)
    subject_monkey_index = 6
    subject_monkey = '81G'
    zombies_without_subject_monkey = [item for item in zombies if item != subject_monkey]
    experimental_session_details = ['Date', 'Round No.', 'Time Window']
    zombies_spike_counts = all_spike_counts[zombies_without_subject_monkey + experimental_session_details]
    # get mean of spike counts
    zombies_spike_counts[zombies_without_subject_monkey] = zombies_spike_counts[zombies_without_subject_monkey].map(
        lambda x: sum(x) / len(x) if x else None)
    run_linear_regression_analysis(zombies_spike_counts, zombies_affiliation_to, "AffliationTo")
    # run_linear_regression_analysis(zombies_spike_counts, zombies_affiliation_from, "AffliationFrom")
    # run_linear_regression_analysis(zombies_spike_counts, zombies_submission_to, "SubmissionTo")
    # run_linear_regression_analysis(zombies_spike_counts, zombies_submission_from, "SubmissionFrom")
    # run_linear_regression_analysis(zombies_spike_counts, zombies_agonism_to, "AgonismTo")
    # run_linear_regression_analysis(zombies_spike_counts, zombies_agonism_from, "AgonismFrom")

'''an example to run for testing -- r squared has to be around 0.5092734647876507'''
# x = np.array([11.5, 8.444444444444445, 8.0, 8.444444444444445, 7.1, 7.3, 10.5, 5.777777777777778]).reshape(-1, 1)
# y = np.array([38, 43, 24, 18, 6, 26, 29, 4])
# coeff, intercept, r_squared = run_linear_regression_sklearn(x, y)
# my_coeff, my_intcpt, my_r_sq = run_linear_regression_mine(x, y)
# ed_coeff, ed_int, ed_r_sq = run_linear_regression_ed(x, y)
# print(f"Linear Regression Results")
# print(f"coeff -- sklearn, mine, ed's: {coeff}, {my_coeff}, {ed_coeff}")
# print(f"intercept -- sklearn, mine, ed's: {intercept}, {my_intcpt}, {ed_int}")
# print(f"r^2 -- sklearn, mine, ed's: {r_squared}, {my_r_sq}, {ed_r_sq}")
