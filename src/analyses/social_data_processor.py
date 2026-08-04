import numpy as np
import pandas as pd
from analyses.data_readers.excel_data_reader import ExcelDataReader
from analyses.enums.monkey_names import Zombies, BestFrans, Instigators

from data_readers.social_data_reader import SocialDataReader
from enums.behaviors import AffiliativeBehaviors as Affiliative
from enums.behaviors import AgonisticBehaviors as Agonistic
from enums.behaviors import IndividualBehaviors as Individual
from enums.behaviors import SubmissiveBehaviors as Submissive


def extract_specific_social_behavior(social_data, social_behavior):
    if isinstance(social_behavior, (Agonistic or Submissive or Affiliative or Individual)):
        specific_behavior = social_data[social_data['Behavior'].str.contains(social_behavior.value, case=False)]
        specific_behavior = specific_behavior[['Focal Name', 'Social Modifier', 'Behavior', 'Behavior Abbrev']]
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    elif isinstance(social_behavior, list):
        specific_behavior = pd.DataFrame()
        for beh in social_behavior:
            temp = social_data[social_data['Behavior'].str.contains(beh.value, case=False)]
            extracted_behavior = temp[['Focal Name', 'Social Modifier', 'Behavior', 'Behavior Abbrev']]
            specific_behavior = pd.concat([specific_behavior, extracted_behavior])
            # print("Length of specific behavior updated to")
            # print(specific_behavior.shape[0])
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    else:
        raise ValueError('Invalid social behavior')
    return subset_df


def expand_rows_on_comma(df):
    # Splitting values with comma and creating new rows
    new_rows = []
    for index, row in df.iterrows():
        if ',' in row['Social Modifier']:
            modifiers = row['Social Modifier'].split(',')
            for modifier in modifiers:
                new_rows.append({col: row[col] for col in df.columns})
                new_rows[-1]['Social Modifier'] = modifier
        else:
            new_rows.append({col: row[col] for col in df.columns})
    return new_rows


def combine_edge_lists(edge_list1, edge_list2):
    # Concatenate the two edge lists
    combined_edge_list = pd.concat([edge_list1, edge_list2])
    # Group by 'Focal Name' and 'Social Modifier' and sum the weights
    combined_edge_list = combined_edge_list.groupby(['Focal Name', 'Social Modifier']).sum().reset_index()
    return combined_edge_list


def generate_edge_list_from_extracted_interactions(interaction_df):
    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edge_list = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    return edge_list


def generate_edge_list_from_pairwise_interactions(interaction_df):
    if (interaction_df['Behavior Abbrev'].isin(['ISU', 'AOS'])).all():
        # Switch actor and receiver for the submissive social_data
        temp = interaction_df['Focal Name'].copy()
        interaction_df['Focal Name'] = interaction_df['Social Modifier']
        interaction_df['Social Modifier'] = temp
    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edge_list = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    print(f'edge list {edge_list}')
    return edge_list


def partition_behavior_variance_from_excel_file(file_name):
    excel_data_reader = ExcelDataReader(file_name=file_name)
    beh = excel_data_reader.get_first_sheet()
    beh = beh.iloc[:, 1:]  # extract only the values
    beh_final, Sm_arrow_values, Sarrow_m_values = partition_behavior_variance(beh)
    return beh_final, Sm_arrow_values, Sarrow_m_values


def partition_behavior_variance(beh: pd.DataFrame):
    Sm_arrow = beh.sum(axis=1) / (beh.shape[1] - 1)  # average frequency of the monkey m submitting to other monkeys
    Sarrow_m = beh.sum(axis=0) / (beh.shape[0] - 1)  # average frequency of other monkeys submitting to the monkey m

    sum_Sm_arrow = Sm_arrow.sum()  # this should be the same as sum_Sarrow_m = Sarrow_m.sum()

    Sm_arrow_values = Sm_arrow.values.reshape(-1, 1)
    Sarrow_m_values = Sarrow_m.values.reshape(-1, 1)

    RSm_n = beh.values - ((Sm_arrow_values * Sarrow_m_values.T) / sum_Sm_arrow)
    np.fill_diagonal(RSm_n, 0)
    beh_final = pd.DataFrame(RSm_n, columns=beh.columns)
    # print(beh_final)  # this table is RSm->n
    return beh_final, Sm_arrow, Sarrow_m


def generate_feature_matrix_from_edge_list(edge_list, monkey_group):
    feature_df = pd.DataFrame({'Focal Name': monkey_group})
    for monkey in monkey_group:
        fill_in = pd.DataFrame({'Focal Name': monkey_group, 'Social Modifier': monkey, 'weight': 0})
        dim = edge_list[edge_list['Social Modifier'] == monkey]
        one_dim = pd.merge(fill_in, dim, how='left', on=['Focal Name', 'Social Modifier'])
        one_dim['weight'] = (one_dim['weight_x'] + one_dim['weight_y']).fillna(0)
        one_dim.drop(['weight_x', 'weight_y'], axis=1, inplace=True)
        one_dim['weight'] = one_dim['weight'].astype(int)
        one_dim.rename(columns={'weight': f'Behavior Towards {monkey}'}, inplace=True)
        one_dim.drop('Social Modifier', axis=1, inplace=True)
        feature_df = pd.merge(feature_df, one_dim, on='Focal Name')
    return feature_df


if __name__ == '__main__':
    zombies = [member.value for name, member in Zombies.__members__.items()]
    bestfrans = [member.value for name, member in BestFrans.__members__.items()]
    instigators = [member.value for name, member in Instigators.__members__.items()]
    del instigators[-1]
    del bestfrans[-1]
    del zombies[-1]
    social_data_reader = SocialDataReader(file_name="instigators_social_data/Instigators_Social_Data_20260804.xlsx")
    social_data = social_data_reader.social_data

    # Agonistic
    agonistic_behaviors = list(Agonistic)
    agon = extract_specific_social_behavior(social_data, agonistic_behaviors)
    # Counting specific social_data and saving the data table as excel
    agon_behavior_specific_count = agon.groupby('Behavior').apply(lambda x: x.groupby(['Focal Name', 'Social Modifier']).size()).reset_index(name='Count')
    agon_behavior_specific_count.columns = ['Behavior', 'Actor', 'Receiver', 'Count']
    agon_behavior_specific_count.to_excel('instigators_frequency_of_specific_behavior_agonism.xlsx')
    edge_list_agon = generate_edge_list_from_extracted_interactions(agon)
    agonistic_feature_df = generate_feature_matrix_from_edge_list(edge_list_agon, instigators)
    agonistic_feature_df.to_excel('instigators_feature_df_agonism.xlsx', index=False)

    # Submissive
    submissive_behaviors = list(Submissive)
    sub = extract_specific_social_behavior(social_data, submissive_behaviors)
    # Counting specific social_data and saving the data table as excel
    sub_behavior_specific_count = sub.groupby('Behavior').apply(lambda x: x.groupby(['Focal Name', 'Social Modifier']).size()).reset_index(name='Count')
    sub_behavior_specific_count.columns = ['Behavior', 'Actor', 'Receiver', 'Count']
    sub_behavior_specific_count.to_excel('instigators_frequency_of_specific_behavior_submission.xlsx')
    edge_list_sub = generate_edge_list_from_extracted_interactions(sub)
    submissive_feature_df = generate_feature_matrix_from_edge_list(edge_list_sub, instigators)
    submissive_feature_df.to_excel('instigators_feature_df_submission.xlsx', index=False)
    #
    # Affiliative
    affiliative_behaviors = list(Affiliative)
    aff = extract_specific_social_behavior(social_data, affiliative_behaviors)
    # Counting specific social_data and saving the data table as excel
    aff_behavior_specific_count = aff.groupby('Behavior').apply(lambda x: x.groupby(['Focal Name', 'Social Modifier']).size()).reset_index(name='Count')
    aff_behavior_specific_count.columns = ['Behavior', 'Actor', 'Receiver', 'Count']
    aff_behavior_specific_count.to_excel('instigators_frequency_of_specific_behavior_affiliation.xlsx')
    edge_list_aff = generate_edge_list_from_extracted_interactions(aff)
    affiliative_feature_df = generate_feature_matrix_from_edge_list(edge_list_aff, instigators)
    affiliative_feature_df.to_excel('instigators_feature_df_affiliation.xlsx', index=False)


    '''
    # Get genealogy matrix
    # excel_data_reader = ExcelDataReader(file_name='zombies_genealogy_matrix.xlsx')
    # genealogy_data = excel_data_reader.get_first_sheet()
    # print(genealogy_data)
    # genealogy_data['Focal Name'] = genealogy_data['Focal Name'].astype(str)
    '''
