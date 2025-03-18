import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from monkey_names import Zombies, BestFrans
from spike_count import get_spike_count_for_single_neuron_with_time_window

def run_linear_regression_using_sklearn(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    # Model
    model = LinearRegression()
    model.fit(x, y)
    # Coefficients
    coeff = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x, y)
    return coeff, intercept, r_squared

def run_linear_regression_manually(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    m_y = np.mean(y)
    m_x = np.mean(x)
    n = len(x)
    SS_xy = np.sum((x - m_x) * (y - m_y))
    SS_xx = np.sum((x - m_x) ** 2)
    # compute mean of x and y
    m_y /= float(n)
    m_x /= float(n)
    b_1_ed = SS_xy / SS_xx
    b_0_ed = m_y - b_1_ed * m_x
    ypred = []
    for ixy in range(0, len(y)):
        ypred.append(b_0_ed + b_1_ed * x[ixy])
    r_sq = explained_variance_score(y, ypred)
    return b_1_ed, b_0_ed, r_sq


if __name__ == "__main__":
    # Zombies
    zombies = [member.value for name, member in Zombies.__members__.items()]
    del zombies[-1]

    base_dir = '/home/connorlab/Documents/GitHub/Julie/behaviors/'
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

    # Cells
    cells_with_windows = pd.read_excel("/home/connorlab/Documents/GitHub/Julie/src/analyses/response_window_finder/cusum_sig_results.xlsx")
    all_spike_counts = get_spike_count_for_single_neuron_with_time_window(cells_with_windows)
    all_spike_counts.columns = all_spike_counts.columns.astype(str)
    subject_monkey_index = 6
    subject_monkey = '81G'
    zombies_without_subject_monkey = [item for item in zombies if item != subject_monkey]
    experimental_session_details = ['Date', 'Round No.', 'Time Window']
    zombies_spike_counts = all_spike_counts[zombies_without_subject_monkey + experimental_session_details]
    print('AFFILIATION TO ANALYSIS')
    # get mean of spike counts
    zombies_spike_counts[zombies_without_subject_monkey] = zombies_spike_counts[zombies_without_subject_monkey].map(lambda x: sum(x) / len(x) if x else None)

    count = 0
    count_r_squared = 0
    for index, row in zombies_spike_counts.iterrows():
        date = row['Date']
        round_no = row['Round No.']
        time_window = row['Time Window']

        for sourcemonkey in range(len(zombies)):
            exclude_columns = ['Date', 'Round No.', 'Time Window']
            if sourcemonkey == subject_monkey_index:
                continue
            else:
                zombies_affiliation_to_arr = zombies_affiliation_to[sourcemonkey, :]
                y = np.delete(zombies_affiliation_to_arr, [sourcemonkey, subject_monkey_index])
                exclude_columns.append(zombies[sourcemonkey])
                x = [value for key, value in row.items() if
                               key not in exclude_columns and isinstance(value, (int, float))]
            # print(f"x: {x}")
            # print(f"y: {y}")
            coeff, intercept, r_squared = run_linear_regression_using_sklearn(x, y)
            manual_coeff, manual_intercept, manual_r_squared = run_linear_regression_manually(x, y)
            if r_squared > 0.25:
                print("")
                print(f"-------- Linear Regression Results for {date} Round No.{round_no} {time_window} {index} -------")
                print(f"------------------------------------{zombies[sourcemonkey]}--------------------------------------")
                print(f"r^2 -- sklearn, ed's: {r_squared}, {manual_r_squared}")
                print("")
                count +=1
            if r_squared > 0.5:
                count_r_squared +=1
    print(count)
    print(count_r_squared)



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