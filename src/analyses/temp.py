from monkey_names import Zombies
from scipy.io import savemat

import social_data_processor
from initial_4feature_lin_reg import compute_average_spike_rates_for_list_of_cells_with_time_windows, \
    construct_feature_matrix_from_behavior_data, \
    get_spike_count_for_single_neuron_with_time_window, get_metadata_for_ANOVA_passed_cells_time_windowed

# Get all dates and rounds from metadata
time_windowed_cells = get_metadata_for_ANOVA_passed_cells_time_windowed()

# Spike Rate
spike_rates = compute_average_spike_rates_for_list_of_cells_with_time_windows(time_windowed_cells)
zombies = [member.value for name, member in Zombies.__members__.items()]
zombies_columns = [col for col in zombies if col in spike_rates.columns]
zombies_spike_rates = spike_rates[zombies_columns]

# Spike Count
spike_count = get_spike_count_for_single_neuron_with_time_window(time_windowed_cells)
zombies_spike_count = spike_count[zombies_columns]
# getting the average spike count
zombies_spike_count = zombies_spike_count.map(lambda x: sum(x) / len(x) if isinstance(x, list) else x)

# Get behavioral data -- start with Agonism
monkey = "81G"
agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file(
    'feature_df_agonism.xlsx')
X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(monkey, agon_beh, Sm_arrow_agon,
                                                                         Sarrow_m_agon, 'Agonism')
X_agon_dict = {}
for zombie in zombies:
    monkey = zombie
    X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(monkey, agon_beh, Sm_arrow_agon,
                                                                             Sarrow_m_agon, 'Agonism')
    X_agon_dict[f'subject_{monkey}'] = X_agon

savemat('windowed_cells_avg_spike_count.mat', {'y': zombies_spike_count})
savemat('agonistic_matrices.mat', X_agon_dict)
# one_neuron = zombies_spike_count.iloc[0, :]
# y = np.stack(one_neuron.values, axis=1)
# y = y.flatten()
# X_agon = np.repeat(X_agon, repeats=10, axis=0)
# X = StandardScaler().fit_transform(X_agon)
# # savemat('raw_data.mat', {'X': X, 'y': y})
# X = sm.add_constant(X[:, 0])
# model = sm.GLM(y, X, family=sm.families.Poisson())
# results = model.fit()
# print(results.summary())

# # Get one neuron for y and do GLM
# for index, row in zombies_spike_count.iterrows():
#     y = np.stack(row.values, axis=1)
#     y = y.flatten()
#     model = sm.GLM(y, X, family=sm.families.Poisson())
#     results = model.fit()
#     print(results.summary())
