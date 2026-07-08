import os
from datetime import datetime, date
from pathlib import Path

import pandas as pd
from clat.intan.channels import Channel

from analyses.data_readers.excel_data_reader import ExcelDataReader
from compile.compile_common import INTAN_BASE_PATH
from project_util import SUBJECT_MONKEY

def standardize_date(date_input):
    if isinstance(date_input, str):
        try:
            return datetime.strptime(date_input.split(' ')[0], '%Y-%m-%d').date()
        except ValueError:
            raise ValueError("Invalid date format. Please use YYYY-MM-DD.")
    elif isinstance(date_input, datetime):
        return date_input.date()
    elif isinstance(date_input, date):
        return date_input
    else:
        print(date_input)
        print(type(date_input))
        raise TypeError("Date must be either a datetime object or a string in YYYY-MM-DD format.")


class RecordingMetadataReader(ExcelDataReader):

    def __init__(self):
        super().__init__(subfolder='recording_metadata', file_name='Cortana_Recording_Metadata.xlsx')
        self.recording_metadata = self.get_metadata()
        self.recording_metadata['Date'] = pd.to_datetime(self.recording_metadata['Date'])

    def get_metadata(self):
        if self.xl is None:
            raise ValueError("Excel file not loaded.")

        channels_tab = self.xl.parse('Channels')
        locations_tab = self.xl.parse('Location')
        recording_metadata = pd.merge(channels_tab, locations_tab, on=['Date', 'Round No.'])
        return recording_metadata

    def get_curated_channels(self, date, round_number) -> list:
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == str(date)) & (
                    self.recording_metadata['Round No.'] == int(round_number))]
        channels = matching_round['Channels1'].apply(
            lambda x: [x] if isinstance(x, int) else [int(i.strip()) for i in str(x).split(',')]).tolist()
        enum_channels = [Channel(f'C-{channel:03}') for channel in channels[0]]
        return enum_channels

    def get_pickle_filenames_for_specific_date(self, date):
        matching_date = self.recording_metadata[(self.recording_metadata['Date'] == date)]
        filenames = matching_date['Pickle File Name']
        return list(filenames)

    def get_pickle_filename_for_specific_round(self, date, round_number):
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == str(date)) & (
                    self.recording_metadata['Round No.'] == int(round_number))]
        filename = matching_round['Pickle File Name'].iloc[0]
        return str(filename) + ".pkl"

    def get_intan_folder_names_for_specific_date(self, date):
        matching_date = self.recording_metadata[(self.recording_metadata['Date'] == date)]
        folder_names = matching_date['Folder Name']
        return list(folder_names)

    def get_intan_folder_name_for_specific_round(self, date, round_number):
        matching_round = self.recording_metadata[
            (self.recording_metadata['Date'] == str(date)) & (
                    self.recording_metadata['Round No.'] == int(round_number))]
        folder_name = matching_round['Folder Name'].iloc[0]
        return str(folder_name)

    def get_metadata_for_brain_region(self, brain_region):
        if brain_region == 'ER' or brain_region == 'Entorhinal':
            return self.recording_metadata[(self.recording_metadata['Location'] == 'ER')]
        elif brain_region == 'Amygdala' or brain_region == 'AMG':
            return self.recording_metadata[(self.recording_metadata['Location'] == 'Amygdala')]
        else:
            raise ValueError('Brain region should be ER or AMG')

    def get_metadata_for_spike_analysis(self, date, round_number, monkey=SUBJECT_MONKEY):
        date = standardize_date(date)
        round_number = int(round_number)
        pickle_filename = self.get_pickle_filename_for_specific_round(date, round_number)
        compiled_dir = (Path(__file__).resolve().parent.parent.parent.parent / monkey / 'compiled')
        pickle_filepath = os.path.join(compiled_dir, pickle_filename)
        valid_channels = set(self.get_curated_channels(date, round_number))

        # for sorted rounds
        intan_dir = self.get_intan_folder_name_for_specific_round(date, round_number)
        round_dir_path = Path(INTAN_BASE_PATH) / monkey / str(date) / intan_dir

        return pickle_filepath, valid_channels, round_dir_path

    def get_metadata_for_preliminary_analysis(self):
        return self.xl.parse('InitialRegression')



#
# if __name__ == "__main__":
#     reader = RecordingMetadataReader()
#     data = reader.recording_metadata
#     filename = reader.get_pickle_filename_for_specific_round("2023-10-27", 1)
#     ER_data = data[data['Location'] == 'ER']
#     AMG_data = data[data['Location'] == 'Amygdala']
#     print("JH 12: \n", ER_data)
#     print("JH 32192: \n", AMG_data)
