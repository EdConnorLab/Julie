import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from analyses.enums.monkey_names import Zombies, BestFrans
from analyses.intan_data_processor.single_channel_analysis import read_pickle
from analyses.intan_data_processor.single_unit_analysis import read_sorted_data
from statsmodels.regression.linear_model import OLS

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.response_window_processing import get_spike_count_for_single_neuron_with_time_window
from analyses.spike_rate_computation import get_average_spike_rates_for_each_monkey, \
    compute_average_spike_rates_for_list_of_cells_with_time_windows


def construct_feature_matrix_from_behavior_data(monkey_of_interest, behavior_data, Sm_arrow, Sarrow_m, behavior_type):
    zombies = [member.value for name, member in Zombies.__members__.items()]
    subject_index = zombies.index(monkey_of_interest)
    subjects_attraction_to_behavior = behavior_data.iloc[:, subject_index]
    behavior_by_subject = pd.Series(behavior_data.iloc[subject_index, :].values)
    general_tendency_of_behavior = pd.Series(Sm_arrow)
    Sarrow_m = Sarrow_m.reset_index(drop=True)
    general_attraction_to_behavior = pd.Series(Sarrow_m)
    feat = pd.concat([subjects_attraction_to_behavior, behavior_by_subject,
                      general_tendency_of_behavior, general_attraction_to_behavior], axis=1,
                     keys=[f'subject\'s_attraction_to_{behavior_type}', f'{behavior_type}_by_subject',
                           f'general_{behavior_type}', f'general_attraction_to_{behavior_type}'])
    feat.reset_index(drop=True, inplace=True)
    feature_names = [f'Subject\'s attraction to {behavior_type}', f'{behavior_type} by the Subject',
                     f'General {behavior_type}', f'General attraction to {behavior_type}']
    feature_matrix = np.delete(feat.values, subject_index, axis=0)  # delete information on 81G
    # TODO: this is wrong!!!!  I should delete axis=6 (Cortana's) for general stuff! and others with the monkey_of_interest
    return feature_matrix, feature_names


def get_metadata_for_list_of_cells_with_time_window(sheet_name: str):
    metadata_reader = RecordingMetadataReader()
    raw_metadata = metadata_reader.get_raw_data()
    metadata_for_cell_list = raw_metadata.parse(sheet_name)
    metadata_for_cell_list = metadata_for_cell_list.dropna(subset=['Time Window Start', 'Time Window End'])
    metadata_for_cell_list['Time Window'] = metadata_for_cell_list.apply(
        lambda row: (row['Time Window Start'], row['Time Window End']), axis=1)
    metadata_subset = metadata_for_cell_list[['Date', 'Round No.', 'Cell', 'Time Window']]

    return metadata_subset


def get_metadata_for_ANOVA_passed_cells_time_windowed():
    metadata_reader = RecordingMetadataReader()
    raw_metadata = metadata_reader.get_raw_data()
    metadata_for_anova_passed = raw_metadata.parse('ANOVA_passed_windowed')
    metadata_for_anova_passed['Time Window'] = metadata_for_anova_passed['Time Window'].apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    metadata_anova_passed_subset = metadata_for_anova_passed[['Date', 'Round No.', 'Cell', 'Time Window']]
    return metadata_anova_passed_subset


def create_overall_plot_for_single_feature_linear_regression_analysis(feature_matrix, response):
    """
    Creates 2x2 plot for different behavior types (Agonism, Submission, Affiliation) for each neuron and combine
    them into one figure -- three 2x2 mixed_manual_raster_plots one for each behavior type

    @param feature_matrix:
    """
    zombies = Zombies.__members__.items()
    zombies.remove('81G')
    fig = plt.figure(figsize=(26, 10))
    outer_grid = GridSpec(1, 3, figure=fig)

    behavior_types = ['agonism', 'submission', 'affiliation']

    feature_names = [
        f"{descriptor} {behavior}" for behavior in behavior_types for descriptor in [
            f"81G's attraction to",
            f"{behavior} by 81G",
            f"General",
            f"General attraction to"
        ]
    ]
    for i in range(len(behavior_types)):
        inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[i])
        start_col = i * 4
        beh_type_specific_feature_mat = feature_matrix[:, start_col:start_col + 4]

        for col_index in range(beh_type_specific_feature_mat.shape[1]):
            one_feature = beh_type_specific_feature_mat[:, col_index].reshape(-1, 1)
            augmented_x = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
            model = OLS(response, augmented_x)
            results = model.fit()
            ax = plt.Subplot(fig, inner_grid[col_index])
            x = augmented_x[:, 0]
            ax.scatter(x, response)

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
            ax.plot(x_fit, y_fit, color='red')
            for n, label in enumerate(zombies):
                ax.text(x[n], response[n], label, ha='right', fontsize=6)
            # ax.set_xlabel(f"{feature_names[col_index]}")
            ax.set_ylabel(f"Average Firing Rate")
            ax.yaxis.set_label_coords(-0.1, 0.5)
            # ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
            fig.add_subplot(ax)
    plt.tight_layout(pad=15.0)
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.035, right=0.99, hspace=0.19, wspace=0.165)
    plt.show()


def run_single_feature_linear_regression_analysis_individual_cells(features, spike_rates, feature_names, behavior_type):
    for index, row in spike_rates.iterrows():
        date = row['Date']
        round_no = row['Round No.']
        window = row['Time Window']
        int_window = (int(window[0]), int(window[1]))
        fig = plt.figure(figsize=(10, 14))
        fig.suptitle(
            f"{behavior_type} plots for {date} Round {round_no} {index} for {int_window[0]}ms to {int_window[1]}ms")
        cleaned_row = row.drop(['Date', 'Round No.', 'Time Window'])
        labels = cleaned_row.index
        y = np.array(cleaned_row.values).astype(float).reshape(-1, 1)
        for col_index in range(features.shape[1]):
            one_feature = features[:, col_index].reshape(-1, 1)
            augmented_X = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
            model = OLS(y, augmented_X)
            results = model.fit()
            # print(results.summary())
            ax = fig.add_subplot(2, 2, col_index + 1)
            x = augmented_X[:, 0]
            ax.scatter(x, y)
            # Draw the fitted line
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
            ax.plot(x_fit, y_fit, color='red')
            for n, label in enumerate(labels):
                ax.text(x[n], y[n], label, ha='right', fontsize=6)
            ax.set_xlabel(f"{feature_names[col_index]}")
            ax.set_ylabel(f"Average Firing Rate")
            ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")

        # plt.show()
        fig.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                    f'time_windowed_cells/{behavior_type}_{date}_round{round_no}_{index} for {int_window}.png')
        plt.close()


def run_single_feature_linear_regression_analysis(X, metadata_for_regression, feature_names, behavior_type):
    all_results = []
    cell_count = 0
    for index, row in metadata_for_regression.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        location = row['Location']
        spike_rates = get_average_spike_rates_for_each_monkey(date, round_no)
        spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]
        labels = spike_rates_zombies.columns.tolist()
        for ind, zombies_row in spike_rates_zombies.iterrows():
            print(f"{date} round {round_no} : performing linear regression for {ind}")
            Y = np.array(zombies_row.values).reshape(-1, 1)
            fig = plt.figure(figsize=(10, 14))
            fig.suptitle(f"{date} Round #{round_no}: {ind}")
            cell_count += 1
            should_save_fig = False  # Initialize the flag

            for col_index in range(X.shape[1]):
                one_feature = X[:, col_index].reshape(-1, 1)
                augmented_X = np.hstack((one_feature, np.ones((one_feature.shape[0], 1))))
                model = OLS(Y, augmented_X)
                results = model.fit()
                ax = fig.add_subplot(2, 2, col_index + 1)
                x = augmented_X[:, 0]
                ax.scatter(x, Y)
                # Draw the fitted line
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = results.predict(np.column_stack((x_fit, np.ones_like(x_fit))))
                ax.plot(x_fit, y_fit, color='red')
                for n, label in enumerate(labels):
                    ax.text(x[n], Y[n], label, ha='right', fontsize=6)
                ax.set_xlabel(f"{feature_names[col_index]}")
                ax.set_ylabel(f"Average Firing Rate")
                ax.set_title(f"R-squared: {results.rsquared:.2f}, p-value {results.pvalues[0]:.6f}")
                # print(results.summary())
                if True:  # (results.rsquared > 0.7) and (results.pvalues[0] < 0.05):
                    should_save_fig = True
                    stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                               results.pvalues[0], results.params]
                    all_results.append(stat_params_to_be_saved)

            if should_save_fig:
                fig.savefig(f'/home/connorlab/Documents/GitHub/Julie/linear_regression_results/'
                            f'single_feature_plots/{behavior_type}_{date}_round{round_no}_{location}_{ind}.png')
                # plt.show()
    # final_df = pd.DataFrame(all_results, columns=['Date', 'Round No.', 'Cell', 'Feature Name', 'R-squared',
    #                                               'p-value', 'coefficients'])
    # final_df.to_excel(f'/home/connorlab/Documents/GitHub/Julie/all_cells_linear_regression/'
    #                   f'single_feature_{behavior_type}.xlsx', index=False)
    # print(f'cell count: {cell_count}')
    # return fig


def generate_r_squared_histogram_for_specific_population(X, feature_names, location):
    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()
    neural_population = metadata[metadata['Location'] == location]
    stat_param_list = []
    for index, row in neural_population.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round_no = row['Round No.']
        spike_rates = get_average_spike_rates_for_each_monkey(date, round_no)
        spike_rates_zombies = spike_rates[[col for col in zombies if col in spike_rates.columns]]

        for ind, zombies_row in spike_rates_zombies.iterrows():
            print(f"{date} round {round_no} : performing linear regression for {ind}")
            Y = np.array(zombies_row.values).reshape(-1, 1)

            for col_index in range(X.shape[1]):
                column = X[:, col_index].reshape(-1, 1)
                augmented_X = np.hstack((column, np.ones((column.shape[0], 1))))
                model = OLS(Y, augmented_X)
                results = model.fit()
                stat_params_to_be_saved = [date, round_no, ind, feature_names[col_index], results.rsquared,
                                           results.pvalues[0], results.params]
                stat_param_list.append(stat_params_to_be_saved)
    results_df = pd.DataFrame(stat_param_list, columns=['Date', 'Round', 'Neuron', 'Feature Name', 'R-squared',
                                                        'p-value', 'coefficients'])
    return results_df


if __name__ == '__main__':
    '''
    Date: 2024-04-28
    Last Modified: 2024-06-27
    Generating 12 plots for the list of cells that Ed picked out -- with time window
    '''

    zombies = [member.value for name, member in Zombies.__members__.items()]
    bestfrans = [member.value for name, member in BestFrans.__members__.items()]

    time_windowed_cells = get_metadata_for_list_of_cells_with_time_window("BestFrans_Cells")
    spike_rates = compute_average_spike_rates_for_list_of_cells_with_time_windows(time_windowed_cells)
    bestfrans_columns = [col for col in bestfrans if col in spike_rates.columns]
    bestfrans_columns.extend(["Date", "Round No.", "Time Window"])
    bestfrans_spike_rates = spike_rates[bestfrans_columns]
    bestfrans_spike_rates.to_excel('bestfrans_spike_rates_2nd_list_windowed.xlsx')

    spike_counts = get_spike_count_for_single_neuron_with_time_window(time_windowed_cells)
    bestfrans_columns = [col for col in bestfrans if col in spike_counts.columns]
    bestfrans_columns.extend(["Date", "Round No.", "Time Window"])
    bestfrans_spike_counts = spike_counts[bestfrans_columns]
    bestfrans_spike_counts.to_excel('bestfrans_spike_counts_2nd_list_windowed.xlsx')
    """
    Looking at each neuron using average spikes rate over 10 trials
    1D analysis for different types of social_data (affiliation, submission, aggression)

    """
    '''
    # Get all experimental round information
    metadata_for_regression = RecordingMetadataReader().get_metadata_for_preliminary_analysis()

    monkey = "81G"
    agon_beh, Sm_arrow_agon, Sarrow_m_agon = social_data_processor.partition_behavior_variance_from_excel_file(
        'zombies_feature_df_agonism.xlsx')
    sub_beh, Sm_arrow_sub, Sarrow_m_sub = social_data_processor.partition_behavior_variance_from_excel_file(
        'zombies_feature_df_submission.xlsx')
    aff_beh, Sm_arrow_aff, Sarrow_m_aff = social_data_processor.partition_behavior_variance_from_excel_file(
        'zombies_feature_df_affiliation.xlsx')

    X_agon, agon_feature_names = construct_feature_matrix_from_behavior_data(monkey, agon_beh, Sm_arrow_agon,
                                                                             Sarrow_m_agon, 'Agonism')
    X_sub, sub_feature_names = construct_feature_matrix_from_behavior_data(monkey, sub_beh, Sm_arrow_sub, Sarrow_m_sub,
                                                                           'Submission')
    X_aff, aff_feature_names = construct_feature_matrix_from_behavior_data(monkey, aff_beh, Sm_arrow_aff, Sarrow_m_aff,
                                                                           'Affiliation')

    run_single_feature_linear_regression_analysis_individual_cells(X_agon, sorted_spike_rates_zombies,
                                                                   agon_feature_names, 'Agonism')
    run_single_feature_linear_regression_analysis_individual_cells(X_sub, sorted_spike_rates_zombies,
                                                                   sub_feature_names, 'Submission')
    run_single_feature_linear_regression_analysis_individual_cells(X_aff, sorted_spike_rates_zombies,
                                                                   aff_feature_names, 'Affiliation')
    # run_single_feature_linear_regression_analysis(X_agon, metadata_for_regression, agon_feature_names, 'agonism')
    # run_single_feature_linear_regression_analysis(X_sub, metadata_for_regression, sub_feature_names, 'submission')
    # run_single_feature_linear_regression_analysis(X_aff, metadata_for_regression, aff_feature_names, 'affiliation')
    '''
    """
    Generating R-squared histograms
    
    """

    # location = 'Amygdala'
    # agon_df = generate_r_squared_histogram_for_specific_population(X_sub, sub_feature_names, location)
    # agon_sorted = agon_df.sort_values(by='R-squared', ascending=False)
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle(f"R-squared values for Submission in {location}", fontsize=16)
    # axs = axs.flatten()
    # plot_number = 0
    #
    # for feature_name, group in agon_sorted.groupby('Feature Name'):
    #     filtered_group = group[group['R-squared'] > 0.1]
    #
    #     if not filtered_group.empty and plot_number < 4:
    #         ax = axs[plot_number]
    #         data = filtered_group['R-squared']
    #         counts, bins, patches = ax.hist(data, bins=20)
    #         top_20_percent = np.percentile(data, 80)
    #         for count, bin, patch in zip(counts, bins, patches):
    #             if bin >= top_20_percent:
    #                 patch.set_facecolor('red')
    #         ax.set_title(f'{feature_name}', fontsize=10)
    #         ax.set_xlabel('R-squared')
    #         ax.set_ylabel('Number of Cells')
    #         ax.grid(False)
    #         plot_number += 1
    #
    # plt.savefig(f'/home/connorlab/Documents/GitHub/Julie/all_cells_linear_regression/single_feature_r_squared_histograms/{location}_Submission_marked')
    # plt.show()
