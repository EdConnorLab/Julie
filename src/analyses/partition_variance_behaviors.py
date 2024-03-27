import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression

import spike_rate_analysis
from excel_data_reader import ExcelDataReader
from monkey_names import Monkey

excel_data_reader = ExcelDataReader(file_name='feature_df_submissive.xlsx')
beh = excel_data_reader.get_first_sheet()
beh = beh.iloc[:, 11:] # only extract the beh columns

Sm_arrow = beh.sum(axis=1) / (beh.shape[1] - 1)
Sarrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)

sum_Sm_arrow = Sm_arrow.sum() # this should be the same as sum_Sarrow_m = Sarrow_m.sum()

Sm_arrow_values = Sm_arrow.values.reshape(-1,1)
Sarrow_m_values = Sarrow_m.values.reshape(-1,1)

Sm_arrow_n = Sm_arrow_values * Sarrow_m_values.T

temp = Sm_arrow_n / sum_Sm_arrow
final = beh.values - temp
np.fill_diagonal(final, 0)
beh_final = pd.DataFrame(final, columns=beh.columns)
print(beh_final) # this table is RSm->n

# Get 81G
attraction_to_submission_81G = beh_final.iloc[:, 6]
submission_by_81G = pd.Series(beh_final.iloc[6, :].values)
general_submission = Sm_arrow
general_attraction_to_submission = pd.Series(Sarrow_m.values)

# Combine to make X
combined_df = pd.concat([attraction_to_submission_81G, submission_by_81G, general_submission, general_attraction_to_submission], axis=1,
                        keys=['attraction_to_submission_81G', 'submission_by_81G', 'general_submission', 'general_attraction_to_submission'])
combined_df.reset_index(drop=True, inplace=True)
numeric_array = combined_df.values
# Append a column of 1s to the array to represent the last column
numeric_array_with_ones = np.hstack((numeric_array, np.ones((numeric_array.shape[0], 1))))
X = numeric_array_with_ones

zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
zombies_df = pd.DataFrame({'Focal Name': zombies})

''' Looking at average spike rates'''
# spike_rates = spike_rate_analysis.compute_overall_average_spike_rates_for_each_round("2023-10-05", 1)
# matching_indices = zombies_df['Focal Name'].isin(spike_rates.index)
# matching_rows = spike_rates.loc[zombies_df.loc[matching_indices, 'Focal Name'].values]
# spike_rate_df = matching_rows.to_frame(name='Spike Rates')
# spike_rate_df['Focal Name'] = spike_rate_df.index
# spike_rate_df = pd.merge(zombies_df, spike_rate_df, on='Focal Name', how='left').fillna(0)
#
# # Extract values from a column as a NumPy array
# column_values = spike_rate_df['Spike Rates'].values
# # Convert the column values to a column matrix
# Y = column_values.reshape(-1, 1)

'''Looking at individual trial'''
spike_rates = spike_rate_analysis.get_raw_spike_rates_for_each_stimulus("2023-10-04", 3)
spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
print(spike_rates_zombies.head())

for index, row in spike_rates_zombies.iterrows():
    repeated_X_rows = []
    spike_rate_list = []
    row_index = 0
    print(f"for {index}, linear regression computed")
    for column_name, value in row.items():
        print(f"\t{len(value)} spike rates for {column_name} and {value.count(0)} of them are zeros")
        spike_rate_list.extend(value)
        repeated_X_rows.extend([X[row_index]] * len(value))
        row_index += 1
    Y = np.array(spike_rate_list).reshape(-1, 1)
    final_X = np.array(repeated_X_rows)

    model = OLS(Y, final_X)
    results = model.fit()
    print(results.summary())

# # Linear Regression
# lr = LinearRegression(fit_intercept=False)
# lr.fit(X, Y)
#
# print('coeff')
# print(lr.coef_)
# model = OLS(Y, X)
# results = model.fit()
# print(results.summary())