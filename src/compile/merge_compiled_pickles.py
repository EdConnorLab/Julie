import os
import pickle

import pandas as pd


def main():
    round_one = ["1696954960635002_231010_122242_round1_1_special.pkl",
                 "1696957097095212_231010_125817_round1_2_special.pkl",
                 ]
    round_two = ["1696888890257195_231009_180130_round2_1_special.pkl",
                 "1696891705894032_231009_184826_round2_2_special.pkl"]
    round_three = ["1698427250062585_231027_132051_round1_1.pkl",
                   "1698428614957224_231027_134335_round1_2.pkl"]
    round_four = [".pkl",
                  "1698436475645557_231027_155436_round4_2.pkl"]

    experiment_data_filenames = round_three

    experiment_names = [experiment_data_filename.split(".")[0] for experiment_data_filename in
                        experiment_data_filenames]

    file_paths = ["/home/connorlab/Documents/GitHub/Julie/compiled/%s" % experiment_data_filename for
                  experiment_data_filename in experiment_data_filenames]

    data = add_pickled_dataframes(file_paths)
    print("Combined Dataframe number of trials:", len(data))
    combined_filename = "&".join(experiment_names) + ".pkl"
    save_dir = "//"
    save_path = os.path.join(save_dir, combined_filename)
    data.to_pickle(save_path)


def add_pickled_dataframes(paths):
    result = None

    for path in paths:
        with open(path, 'rb') as file:
            data = pickle.load(file)
            if result is None:
                result = data
            else:
                # Concatenate the DataFrames vertically (later ones below earlier ones)
                result = pd.concat([result, data], axis=0, ignore_index=True)

    return result


if __name__ == '__main__':
    main()
