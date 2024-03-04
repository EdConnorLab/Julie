import os
from pathlib import Path

import pandas as pd
from clat.intan.channels import Channel


class RecordingMetadataReader:

    def __init__(self):
        raw_data_file_name = 'Cortana_Recording_Metadata.xlsx'
        dir = '/home/connorlab/Documents/GitHub/Julie/resources'
        file_path = Path(os.path.join(dir, raw_data_file_name))
        xl = pd.ExcelFile(file_path)
        channels_tab = xl.parse('Channels')
        locations_tab = xl.parse('Location')
        recording_metadata = pd.merge(channels_tab, locations_tab, on=['Date', 'Round No.'])
        self.xl = xl
        self.recording_metadata = recording_metadata

    def get_metadata(self):
        return self.recording_metadata

    def get_valid_channels(self, date, round_number) -> list:
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == date) & (self.recording_metadata['Round No.'] == round_number)]
        channels = matching_round['Channels1'].apply(lambda x: [int(i.strip()) for i in x.split(',')]).tolist()
        enum_channels = [Channel(f'C-{channel:03}') for channel in channels[0]]
        print(enum_channels)
        return enum_channels

    def get_pickle_filenames_for_specific_date(self, date):
        matching_date = self.recording_metadata[(self.recording_metadata['Date'] == date)]
        filenames = matching_date['Pickle File Name']
        return list(filenames)

    def get_pickle_filename_for_specific_round(self, date, round_number):
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == date) & (self.recording_metadata['Round No.'] == round_number)]
        filename = matching_round['Pickle File Name'].iloc[0]
        return str(filename)

    def get_intan_folder_names_for_specific_date(self, date):
        matching_date = self.recording_metadata[(self.recording_metadata['Date'] == date)]
        folder_names = matching_date['Folder Name']
        return list(folder_names)

    def get_intan_folder_name_for_specific_round(self, date, round_number):
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == date) & (self.recording_metadata['Round No.'] == round_number)]
        folder_name = matching_round['Folder Name'].iloc[0]
        return str(folder_name)


if __name__ == "__main__":
    reader = RecordingMetadataReader()
    metadata = reader.recording_metadata
    filename = reader.get_pickle_filename_for_specific_round("2023-10-27", 1)
    ER_data = metadata[metadata['Location'] == 'ER']
    AMG_data = metadata[metadata['Location'] == 'Amygdala']
    # print("JH 12: \n", ER_data)
    # print("JH 32192: \n", AMG_data)
    # get_valid_channels("10-10-2023", "1696957915096002_231010_131155")
