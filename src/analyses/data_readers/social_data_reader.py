import os
from pathlib import Path

from excel_data_reader import ExcelDataReader


class SocialDataReader(ExcelDataReader):

    def __init__(self, file_name):
        super().__init__(subfolder='behaviors', file_name=file_name)
        self.raw_social_data = self.get_raw_social_data()
        self.social_data = self.clean_raw_social_data()

    def get_raw_social_data(self):
        if self.xl is None:
            raise ValueError("Excel file not loaded.")
        # Get the last sheet in the Excel file
        sheet_names = self.xl.sheet_names
        raw_social_data = self.xl.parse(sheet_names[-1])
        return raw_social_data

    def clean_raw_social_data(self):
        # Rename columns for convenience
        self.raw_social_data.rename(columns={'All Occurrence Value': 'Behavior'}, inplace=True)
        self.raw_social_data.rename(columns={'All Occurrence Behavior Social Modifier': 'Social Modifier'},
                                    inplace=True)
        self.raw_social_data.rename(columns={'All Occurrence Space Use Coordinate XY': 'Space Use'}, inplace=True)

        # Rename the last column to Time
        self.raw_social_data.columns = [*self.raw_social_data.columns[:-1], 'Time']

        # Combine Year, Month, Day columns into VideoDate
        self.raw_social_data['VideoDate'] = (
                self.raw_social_data.iloc[:, -4].astype(str).str.zfill(2) + self.raw_social_data.iloc[:, -3].astype(
            str).str.zfill(2)
                + self.raw_social_data.iloc[:, -2].astype(str).str.zfill(2))

        self.social_data = self.raw_social_data[
            ['Observer', 'Focal Name', 'Behavior', 'Social Modifier', 'Space Use', 'VideoDate', 'Time']].copy()
        # Remove parentheses and extract monkey ids
        self.social_data['Social Modifier'] = self.social_data['Social Modifier'].str.replace(r'^.*?\((.*?)\).*|^(.+)$',
                                                                                    lambda m: m.group(1) if m.group(
                                                                                        1) is not None else m.group(2),
                                                                                    regex=True)
        self.social_data['Focal Name'] = self.social_data['Focal Name'].str.replace(r'^.*?\((.*?)\).*|^(.+)$',
                                                                          lambda m: m.group(1) if m.group(
                                                                              1) is not None else m.group(2),
                                                                          regex=True)

        if 'Behavior Abbrev' not in self.social_data.columns:
            self.social_data['Behavior Abbrev'] = self.social_data['Behavior'].str[:4].str.replace(' ', '')
            self.social_data['Behavior'] = self.social_data['Behavior'].str[4:]

        return self.social_data

    def validate_number_of_monkeys(self, social_data):
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

    def validate_number_of_interval_datapoints(self, social_data):

        """ CHECK NUMBER OF INTERVAL DATA """
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

