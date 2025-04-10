from sklearn.linear_model import LassoCV, MultiTaskLassoCV
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.multioutput import MultiOutputRegressor

import social_data_processor
import spike_count
from excel_data_reader import ExcelDataReader
from initial_4feature_lin_reg import construct_feature_matrix_from_behavior_data
from monkey_names import Monkey
from sklearn.model_selection import train_test_split

from spike_rate_computation import get_average_spike_rates_for_each_monkey

"""
Regressors
    1. Regressor group 1: genealogy
    2. Regressor group 2: frequency of social_data (affiliative, agonistic, submissive)
    standardize behavioral matrix (subtract mean and divide by standard deviation)

Response
    1. Entire trial (2 seconds)
    2. Time windowed (based on visual inspection)
    3. Set time window -- 300 ms to 1000 ms

"""
# Get frequency of social_data table
monkey = "81G"
agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file(
    'feature_df_agonism.xlsx')
X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(monkey, agon_beh, Sm_arrow_agon,
                                                                         Sarrow_m_agon, 'Agonism')
X_agon_aug = sm.add_constant(X_agon)

# Get genealogy matrix
excel_data_reader = ExcelDataReader(file_name='genealogy_matrix.xlsx')
genealogy_data = excel_data_reader.get_first_sheet()
genealogy_data['Focal Name'] = genealogy_data['Focal Name'].astype(str)  # To convert 7124 into string
genealogy_data = genealogy_data.drop(6)
X = genealogy_data.iloc[:, 1:]  # no need to standardize the genealogy data
X_aug = sm.add_constant(X)

spikes = spike_count.get_spike_count_for_each_trial("2023-10-04", 1)
# Get one neuron
one_neuron = spikes.iloc[0, :]
# get only zombies
zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]
zombies_columns = [col for col in zombies if col in one_neuron.index]
Y_zombies = one_neuron.loc[zombies_columns]
Y = np.vstack(Y_zombies.values)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_aug, Y, test_size=0.2, random_state=42)
# Initialize LassoCV
lasso = LassoCV(cv=5)
# Wrap LassoCV with MultiOutputRegressor
multi_output_lasso = MultiOutputRegressor(lasso)
# Fit the model
multi_output_lasso.fit(X_train, Y_train)
# Predict on the test set
Y_pred = multi_output_lasso.predict(X_test)
Y_pred_rounded = np.ceil(Y_pred).astype(int)

print("Predictions:\n", Y_pred)
print("Actual values:\n", Y_test)


# Let me try with average spike count and then simple LassoCV
spike_rates = get_average_spike_rates_for_each_monkey("2023-10-04", 1)
neuron = spike_rates.iloc[2, :]
zombies_columns = [col for col in zombies if col in neuron.index]
Y_avg = neuron.loc[zombies_columns]
lasso = LassoCV(cv=9).fit(X_aug, Y_avg)
betas = lasso.coef_
print("betas: ", betas)


######
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_agon_aug, Y, test_size=0.2, random_state=42)
lasso = LassoCV(cv=5)
multi_output_lasso = MultiOutputRegressor(lasso)
multi_output_lasso.fit(X_train, Y_train)
Y_pred = multi_output_lasso.predict(X_test)
print("Predictions for agon:\n", Y_pred)
print("Actual values for agon:\n", Y_test)


spike_rates = get_average_spike_rates_for_each_monkey("2023-10-04", 1)
spike_rates.to_csv('sample_spike_rates_for_poisson_glm.csv', index=False)
neuron = spike_rates.iloc[5, :]
zombies_columns = [col for col in zombies if col in neuron.index]
Y_avg = neuron.loc[zombies_columns]
lasso = LassoCV(cv=9).fit(X_agon_aug, Y_avg)
betas = lasso.coef_
print("betas for agon: ", betas)

#
# # X_scaled = sm.add_constant(X_scaled)
# # poisson_model = sm.GLM(Y, X_scaled, family=sm.families.Poisson()).fit()
# # print(poisson_model.summary())
#
# np.random.seed(42)
# # Create a synthetic dataset
# n = 100
# x1 = np.random.normal(size=n)
# x2 = np.random.normal(size=n)
# y = np.random.poisson(lam=np.exp(0.5 * x1 + 0.3 * x2))
# data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
# print(data.head())
#
#
# # Define the model formula
# formula = 'y ~ x1 + x2'
#
# # Fit the Poisson GLM
# model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
#
# # Print the model summary
# print(model.summary())
