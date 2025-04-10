from pathlib import Path

import pandas as pd
from clat.intan.channels import Channel

from channel_enum_resolvers import convert_to_enum
from recording_metadata_reader import RecordingMetadataReader
from single_channel_analysis import read_pickle, plot_raster_for_monkeys_by_rank


def plot_rasters_for_specific_round(date, round_no):
    metadata_reader = RecordingMetadataReader()
    experiment_data_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_no)
    channels = metadata_reader.get_valid_channels(date, round_no)
    script_dir = Path(__file__).parent
    file_path = (script_dir / '..' / '..' / '..' / 'Cortana' / 'compiled' / experiment_data_filename).resolve()
    raw_data = read_pickle(file_path)
    for channel in channels:
        print("Working on channel %s" % channel)
        plot_raster_for_monkeys_by_rank(raw_data, channel, date, round_no, save=True)

def plot_rasters_for_specific_round_and_channel(date, round_no, channels):
    metadata_reader = RecordingMetadataReader()
    experiment_data_filename = metadata_reader.get_pickle_filename_for_specific_round(date, round_no)
    script_dir = Path(__file__).parent
    file_path = (script_dir / '..' / '..' / '..' / 'Cortana' / 'compiled' / experiment_data_filename).resolve()
    raw_data = read_pickle(file_path)
    for channel in channels:
        print("Working on channel %s" % channel)
        plot_raster_for_monkeys_by_rank(raw_data, channel, date, round_no, save=True)



if __name__ == '__main__':
    cells_to_be_plotted = pd.read_excel("/home/connorlab/Documents/GitHub/Julie/src/analyses/response_window_finder/CUSUM_window_cells_ANOVA_passed.xlsx")
    subset_df = cells_to_be_plotted[['Date', 'Round No.', 'Cell']]
    all_cells = subset_df.drop_duplicates(subset=['Date', 'Round No.', 'Cell'])

    reversed_df = all_cells.groupby(['Date', 'Round No.']).agg(list)
    for index, row in reversed_df.iterrows():
        row['Cell']= [convert_to_enum(item) for item in row['Cell']]
        date = index[0]
        round_no = index[1]
        plot_rasters_for_specific_round_and_channel(date, round_no, row['Cell'])
        # plot_rasters_for_specific_round_and_channel("2023-10-04", 3, [Channel.C_002])
