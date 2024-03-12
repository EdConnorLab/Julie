import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import behaviors
from behaviors import AgonisticBehaviors as Agonistic
from behaviors import SubmissiveBehaviors as Submissive
from behaviors import AffiliativeBehaviors as Affiliative
from behaviors import IndividualBehaviors as Individual
from monkey_names import Monkey
import spike_rate_analysis


def read_social_data_and_validate():
    # Load raw data file
    current_dir = os.getcwd()
    file_path = '/home/connorlab/Documents/GitHub/Julie/resources/ZombiesFinalRawData.xlsx'
    raw_social_data = read_raw_social_data(file_path)

    # Clean raw data
    social_data = clean_raw_social_data(raw_social_data)

    # Check number of monkeys recorded for each day
    validate_number_of_monkeys(social_data)

    # Check number of interval datapoints for each day
    validate_number_of_interval_datapoints(social_data)
    return social_data


def read_raw_social_data(filepath):
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    # print(sheet_names)
    last_sheet = sheet_names[-1]
    df = xl.parse(last_sheet)
    # print(df.columns)
    return df


def clean_raw_social_data(raw_social_data):
    # Rename columns for convenience
    raw_social_data.rename(columns={'All Occurrence Value': 'Behavior'}, inplace=True)
    raw_social_data.rename(columns={'All Occurrence Behavior Social Modifier': 'Social Modifier'}, inplace=True)
    raw_social_data.rename(columns={'All Occurrence Space Use Coordinate XY': 'Space Use'}, inplace=True)

    # Rename the last column to Time
    raw_social_data.columns = [*raw_social_data.columns[:-1], 'Time']

    # Combine Year, Month, Day columns into VideoDate
    raw_social_data['VideoDate'] = (
                raw_social_data.iloc[:, -4].astype(str).str.zfill(2) + raw_social_data.iloc[:, -3].astype(
            str).str.zfill(2)
                + raw_social_data.iloc[:, -2].astype(str).str.zfill(2))
    # print(df.columns)

    social_data = raw_social_data[
        ['Observer', 'Focal Name', 'Behavior', 'Social Modifier', 'Space Use', 'VideoDate', 'Time']].copy()
    # Remove parentheses and extract monkey ids
    social_data['Social Modifier'] = social_data['Social Modifier'].str.replace(r'^.*?\((.*?)\).*|^(.+)$',
                                                                      lambda m: m.group(1) if m.group(
                                                                          1) is not None else m.group(2), regex=True)
    social_data['Focal Name'] = social_data['Focal Name'].str.replace(r'^.*?\((.*?)\).*|^(.+)$',
                                                                      lambda m: m.group(1) if m.group(
                                                                          1) is not None else m.group(2), regex=True)
    print(social_data)
    return social_data

def validate_number_of_monkeys(social_data):
    # For dates before 06/13/2022, 10 monkeys
    # after 06/13/2022, 8 monkeys
    # exception: 05/19/2022, 8 monkeys
    grouped = social_data.groupby('VideoDate')['Focal Name'].agg(['nunique', 'unique']).reset_index()

    for index, row in grouped.iterrows():
        video_date = row['VideoDate']
        # unique_values = ', '.join(row['unique'])
        count = row['nunique']
        # print(f"VideoDate: {video_date}, # of unique monkeys: {count}")

        expected_count = 10 if video_date < '20220613' else 8
        if video_date == '20220519':
            expected_count = 8
        else:
            expected_count = 10 if video_date < '20220613' else 8

        if count != expected_count:
            raise ValueError(
                f"Unexpected number of monkeys ({count}) observed on {video_date}. Expected: {expected_count} monkeys.")
        else:
            print(f"Validation passed! Valid number of monkeys for {video_date}")

def validate_number_of_interval_datapoints(social_data):
    if 'Behavior Abbrev' not in social_data.columns:
        social_data['Behavior Abbrev'] = social_data['Behavior'].str[:4].str.replace(' ', '')
        social_data['Behavior'] = social_data['Behavior'].str[4:]

    ''' CHECK NUMBER OF INTERVAL DATA '''
    # Create a mask to check if 'Behavior Abbrev' starts with 'I'
    mask = social_data['Behavior Abbrev'].str.startswith('I')

    # Filter the DataFrame to include only rows where 'Behavior Abbrev' starts with 'I'
    interval = social_data[mask].groupby('VideoDate')['Behavior Abbrev'].count().reset_index()
    filtered = interval[(interval['Behavior Abbrev'] != 120) & (interval['Behavior Abbrev'] != 96)]
    result = social_data[mask].groupby(['VideoDate', 'Focal Name']).size().reset_index(name='Count')
    filtered_result = result[(result['Count'] != 12)]

    if filtered.empty:
        print("Validation passed! Valid number of interval datapoints for all dates :)")
    else:
        raise ValueError(f'Invalid number of interval datapoints! : {filtered}')
        raise ValueError(f'Monkey specific interval datapoint count: {filtered_result}')


def extract_specific_social_behavior(social_data, social_behavior):
    if isinstance(social_behavior, (behaviors.AgonisticBehaviors or behaviors.SubmissiveBehaviors
                                    or behaviors.AffiliativeBehaviors or behaviors.IndividualBehaviors)):
        specific_behavior = social_data[social_data['Behavior'].str.contains(social_behavior.value, case=False)]
        specific_behavior = specific_behavior[['Focal Name', 'Social Modifier', 'Behavior']]
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    elif isinstance(social_behavior, list):
        specific_behavior = pd.DataFrame()
        for beh in social_behavior:
            temp = social_data[social_data['Behavior'].str.contains(beh.value, case=False)]
            extracted_behavior = temp[['Focal Name', 'Social Modifier', 'Behavior']]
            specific_behavior = pd.concat([specific_behavior, extracted_behavior])
            # print("Length of specific behavior updated to")
            # print(specific_behavior.shape[0])
        subset = specific_behavior.dropna(subset=['Social Modifier'])  # remove all nan
        subset = expand_rows_on_comma(subset)
        subset_df = pd.DataFrame(subset)
    else:
        raise ValueError('Invalid social behavior')
    return subset_df


def extract_specific_interaction_type(social_data, social_interaction_type):
    social_data['Behavior Abbrev'] = social_data['Behavior'].str[:4].str.replace(' ', '')
    social_data = social_data[['Focal Name', 'Social Modifier', 'Behavior Abbrev']]


    if social_interaction_type.lower() == 'agonistic':
        subset = social_data[(social_data['Behavior Abbrev'] == 'AOA') | (social_data['Behavior Abbrev'] == 'IAG')]
    elif social_interaction_type.lower() == 'submissive':
        subset = social_data[(social_data['Behavior Abbrev'] == 'AOS') | (social_data['Behavior Abbrev'] == 'ISU')]
    elif social_interaction_type.lower() == 'affiliative':
        subset = social_data[(social_data['Behavior Abbrev'] == 'IAF')]
    else:
        raise ValueError("Invalid social interaction type. Please provide 'affiliative', 'submissive', or 'agonistic'.")

    subset = subset.dropna(subset=['Social Modifier'])  # remove all nan
    subset = expand_rows_on_comma(subset)
    subset_df = pd.DataFrame(subset)
    interaction_df = subset_df[['Focal Name', 'Social Modifier', 'Behavior Abbrev']]

    return interaction_df

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

def generate_edgelist_from_extracted_interactions(interaction_df):
    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edgelist = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    return edgelist

def generate_edgelist_from_pairwise_interactions(interaction_df):

    if (interaction_df['Behavior Abbrev'].isin(['ISU', 'AOS'])).all():
        # Switch actor and receiver for the submissive behaviors
        temp = interaction_df['Focal Name'].copy()
        interaction_df['Focal Name'] = interaction_df['Social Modifier']
        interaction_df['Social Modifier'] = temp

    interaction_df = interaction_df[['Focal Name', 'Social Modifier']]
    edgelist = interaction_df.groupby(['Focal Name', 'Social Modifier']).size().reset_index(name='weight')
    print(f'edge list {edgelist}')

    return edgelist


def combine_edgelists(edgelist1, edgelist2):
    # Concatenate the two edge lists
    combined_edgelist = pd.concat([edgelist1, edgelist2])
    # Group by 'Focal Name' and 'Social Modifier' and sum the weights
    combined_edgelist = combined_edgelist.groupby(['Focal Name', 'Social Modifier']).sum().reset_index()
    return combined_edgelist


def read_genealogy_matrix(genealogy_file):
    pass


if __name__ == '__main__':

    # Agonistic
    social_data = read_social_data_and_validate()
    agonistic_behaviors = list(Agonistic)
    agon = extract_specific_social_behavior(social_data, agonistic_behaviors)
    edge_list_agon = generate_edgelist_from_extracted_interactions(agon)

    # Submissive
    submissive_behaviors = list(Submissive)
    sub = extract_specific_social_behavior(social_data, submissive_behaviors)
    edge_list_sub = generate_edgelist_from_extracted_interactions(sub)

    # Affiliative
    affiliative_behaviors = list(Affiliative)
    aff = extract_specific_social_behavior(social_data, affiliative_behaviors)
    edge_list_aff = generate_edgelist_from_extracted_interactions(aff)

    # Get genealogy matrix
    file_path = '/home/connorlab/Documents/GitHub/Julie/resources/genealogy_matrix.xlsx'
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    genealogy_df = xl.parse(sheet_names[0])
    print(genealogy_df)
    genealogy_df['Focal Name'] = genealogy_df['Focal Name'].astype(str)

    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]

    zombies_df = pd.DataFrame({'Focal Name': zombies})
    temp = pd.DataFrame({'Focal Name': zombies})

    for zombie in zombies:
        fill_in = pd.DataFrame({'Focal Name': zombies, 'Social Modifier': zombie, 'weight': 0})
        dim = edge_list_agon[edge_list_agon['Social Modifier'] == zombie]
        one_dim = pd.merge(fill_in, dim, how='left', on=['Focal Name', 'Social Modifier'])
        one_dim['weight'] = (one_dim['weight_x'] + one_dim['weight_y']).fillna(0)
        one_dim.drop(['weight_x','weight_y'], axis=1, inplace=True)
        one_dim['weight'] = one_dim['weight'].astype(int)
        one_dim.rename(columns={'weight': f'Agonistic Behavior Towards {zombie}'}, inplace=True)
        one_dim.drop('Social Modifier', axis=1, inplace=True)
        temp = pd.merge(temp, one_dim, on='Focal Name')
    cols_to_normalize = temp.select_dtypes(include='int64').columns[1:]
    # Normalize selected columns
    normalized_cols = temp[cols_to_normalize] / temp[cols_to_normalize].max()
    # Concatenate normalized columns with the first column and non-integer columns
    normalized_df = pd.concat([temp.iloc[:, 0], normalized_cols, temp.select_dtypes(exclude='int64')], axis=1)
    genealogy_df = pd.merge(genealogy_df, temp, on='Focal Name')

    # Normalize
    cols_to_normalize = genealogy_df.select_dtypes(include='int64').columns[1:]
    normalized_cols = genealogy_df[cols_to_normalize] / genealogy_df[cols_to_normalize].max()
    normalized_df = pd.concat([genealogy_df.iloc[:, 0], normalized_cols, genealogy_df.select_dtypes(exclude='int64')], axis=1)
    print(normalized_df.shape)

    # Get only numbers
    numeric_columns = normalized_df.iloc[:, 1:-1]
    # Convert selected numeric columns to a NumPy array
    numeric_array = numeric_columns.values
    # Append a column of 1s to the array to represent the last column
    numeric_array_with_ones = np.hstack((numeric_array, np.ones((numeric_array.shape[0], 1))))
    # Compute pseudoinverse of the resulting array
    X = numeric_array_with_ones
    Xpinv= np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)

    print(Xpinv.shape)

    # Get Spike Rate -- Y
    ER_population_spike_rate = spike_rate_analysis.compute_population_spike_rate_for_ER()
    matching_indices = zombies_df['Focal Name'].isin(ER_population_spike_rate.index)
    matching_rows = ER_population_spike_rate.loc[zombies_df.loc[matching_indices, 'Focal Name'].values]
    spike_rate_df = matching_rows.to_frame(name='Spike Rates')
    spike_rate_df['Focal Name'] = spike_rate_df.index
    spike_rate_df = pd.merge(zombies_df, spike_rate_df, on='Focal Name', how='left').fillna(0)

    # Extract values from a column as a NumPy array
    column_values = spike_rate_df['Spike Rates'].values
    # Convert the column values to a column matrix
    Y = column_values.reshape(-1, 1)

    beta = Xpinv @ Y
    print('beta')
    print(beta)

    lr = LinearRegression()
    lr.fit(X, Y)
    print('coeff')
    print(lr.coef_)


    # submissive_behavior_list = list(Submissive)
    # submissive = extract_specific_social_behavior(social_data, submissive_behavior_list)
    # edgelist_submissive = generate_edgelist_from_extracted_interactions(submissive)
    #
    # print(edgelist_submissive['Social Modifier'].unique())

