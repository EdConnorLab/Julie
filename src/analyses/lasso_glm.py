import pandas as pd
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Lasso
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from initial_4feature_lin_reg import construct_feature_matrix_from_behavior_data, \
    get_metadata_for_ANOVA_passed_cells_time_windowed, compute_average_spike_rates_for_list_of_cells_with_time_windows
from monkey_names import Zombies
import social_data_processor

'''
Perform LassoCV (since there are only 9 data points, rather than doing Lasso) 
and pick out the non-zero coefficients and then use these for glm fit

Date Created: 07-02-2024
Last Modified: 07-02-2024

'''
# Get all dates and rounds from metadata
time_windowed_cells = get_metadata_for_ANOVA_passed_cells_time_windowed()

# Spike Rate
spike_rates = compute_average_spike_rates_for_list_of_cells_with_time_windows(time_windowed_cells)
zombies = [member.value for name, member in Zombies.__members__.items()]
zombies_columns = [col for col in zombies if col in spike_rates.columns]
zombies_spike_rates = spike_rates[zombies_columns]

# Get behavioral data
agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file(
    'feature_df_agonism.xlsx')
sub_beh, Sm_arrow_sub, Sarrow_m_sub = social_data_processor.partition_behavior_variance_from_excel_file(
    'feature_df_submission.xlsx')
aff_beh, Sm_arrow_aff, Sarrow_m_aff = social_data_processor.partition_behavior_variance_from_excel_file(
    'feature_df_affiliation.xlsx')


all_arrays = []
specific_feature_names = []
for zombie in zombies:
    X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(zombie, agon_beh, Sm_arrow_agon,
                                                                             Sarrow_m_agon, 'Agonism')
    X_sub, sub_feature_names = construct_feature_matrix_from_behavior_data(zombie, sub_beh, Sm_arrow_sub, Sarrow_m_sub,
                                                                           'Submission')
    X_aff, aff_feature_names = construct_feature_matrix_from_behavior_data(zombie, aff_beh, Sm_arrow_aff, Sarrow_m_aff,
                                                                           'Affiliation')
    arrays = np.hstack([X_agon[:, :2], X_sub[:, :2], X_aff[:, :2]])
    # this would give me 7124's attraction to agon, agon by 7124, general agon, general attraction to agon,
    # 7124 attraction to sub, sub by 7124, ...
    # and then 69X.. and so on ...
    all_arrays.append(arrays)

    feature_names_for_zombie = [f'{zombie}\'s attraction to Agonism', f'Agonism by the {zombie}',
                              f'{zombie}\'s attraction to Submission', f'Submission by the {zombie}',
                              f'{zombie}\'s attraction to Affiliation', f'Affiliation by the {zombie}']
    specific_feature_names.extend(feature_names_for_zombie)

subject_specific = np.hstack(all_arrays)

# get general social_data separately
cortana = '81G'
X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(cortana, agon_beh, Sm_arrow_agon,
                                                                         Sarrow_m_agon, 'Agonism')
X_sub, sub_feature_names = construct_feature_matrix_from_behavior_data(cortana, sub_beh, Sm_arrow_sub, Sarrow_m_sub,
                                                                       'Submission')
X_aff, aff_feature_names = construct_feature_matrix_from_behavior_data(cortana, aff_beh, Sm_arrow_aff, Sarrow_m_aff,
                                                                       'Affiliation')
general = np.hstack([X_agon[:, -2:], X_sub[:, -2:], X_aff[:, -2:]])

X = np.hstack([general, subject_specific])
df = pd.DataFrame(X)
df.astype(float)
df.to_excel('design_matrix.xlsx')
X = StandardScaler().fit_transform(X)
# Get all dates and rounds from metadata
time_windowed_cells = get_metadata_for_ANOVA_passed_cells_time_windowed()

# Names of features assuming you have them as a list or from a DataFrame
general_feature_names = ['General Agonism', 'General Attraction to Agonism', 'General Submission',
    'General Attraction to Submission', 'General Affiliation', 'General Attraction to Affiliation']

general_feature_names.extend(specific_feature_names)


for index, row in zombies_spike_rates.iterrows():
    y = row.values.flatten()
    lasso_cv = LassoCV(cv=5, max_iter=9000, tol=0.001)
    lasso_cv.fit(X, y)
    # print(lasso_cv.coef_)
    # print(lasso_cv.intercept_)
    # print(lasso_cv.alpha_)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    # plt.plot(lasso_cv.alphas_, mean_mse, marker='o')
    # plt.xlabel('Alpha')
    # plt.ylabel('Mean Squared Error')
    # plt.title('MSE vs. Alpha in LassoCV')
    # plt.xscale('log')  # Since alphas can vary on a large scale
    # plt.show()
    lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X, y)
    # Get the mask of non-zero coefficients
    non_zero_mask = lasso.coef_ != 0
    num_of_non_zero_coeff = sum(non_zero_mask)
    print("----------------------------------------------------------------------------------------------")
    print(f"-------------------------------{num_of_non_zero_coeff}---------------------------------------")
    print("----------------------------------------------------------------------------------------------")
    important_features = [name for name, m in zip(general_feature_names, non_zero_mask) if m]
    print("Important features:", important_features)
    X_selected = X[:, non_zero_mask]
    X_adj = sm.add_constant(X_selected)
    model = sm.GLM(y, X_adj, family=sm.families.Gaussian())
    results = model.fit()
    print(results.summary())


