
"""
single_unit_analysis.py
"""

import os
import pickle

import pandas as pd
from clat.intan.rhd import load_intan_rhd_format

from analyses.raster_plotting import plot_raster_by_group


def main():
    date = "2023-10-05"
    round_name = "231005_round2"
    cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
    round_path = os.path.join(cortana_path, date, round_name)

    sorted_data = read_sorted_data(round_path)

    for unit in sorted_data["SpikeTimes"].iloc[0]:
        plot_raster_for_unit(sorted_data, unit, experiment_name=round_name)


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_manually_sorted_spikes(path):
    """Load a dict of manually sorted spike indices from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


# ── Data loading ─────────────────────────────────────────────────────────────

def read_sorted_data(round_path,
                     manually_sorted_spikes_filename="sorted_spikes.pkl",
                     compiled_trials_filename="compiled.pkl"):
    """Load compiled trials and attach per-unit spike timestamps."""
    raw = pd.read_pickle(os.path.join(round_path, compiled_trials_filename)).reset_index(drop=True)
    sorted_spikes = load_manually_sorted_spikes(os.path.join(round_path, manually_sorted_spikes_filename))

    rhd_path = os.path.join(round_path, "info.rhd")
    sample_rate = load_intan_rhd_format.read_data(rhd_path)["frequency_parameters"]["amplifier_sample_rate"]

    return calculate_spike_timestamps(raw, sorted_spikes, sample_rate)


def calculate_spike_timestamps(df, spike_indices_by_unit_by_channel, sample_rate):
    """
    Add a 'SpikeTimes' column: dict mapping unit names to spike-time lists
    filtered to each trial's epoch.
    """
    def _for_row(epoch_start_stop):
        epoch_start, epoch_stop = epoch_start_stop
        result = {}
        for channel, units in reversed(spike_indices_by_unit_by_channel.items()):
            for unit_name, spike_indices in units.items():
                key = f"{channel}_{unit_name}"
                result[key] = [
                    idx / sample_rate
                    for idx in spike_indices
                    if epoch_start <= idx / sample_rate < epoch_stop
                ]
        return result

    out = df.copy(deep=True)
    out["SpikeTimes"] = out["EpochStartStop"].apply(_for_row)
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_raster_for_unit(data, unit, experiment_name=None, save=False):
    """Plot a ranked raster for a single sorted unit."""
    unit_data = data.copy()
    spike_col = f"SpikeTimes_{unit}"
    unit_data[spike_col] = data["SpikeTimes"].apply(lambda x: x.get(unit, []))

    save_path = None
    if experiment_name and save:
        save_dir = os.path.join("/png_raster_plots/", experiment_name)
        save_path = os.path.join(save_dir, f"{experiment_name}_{unit}.png")

    plot_raster_by_group(
        unit_data, spike_col,
        title=f"Raster Plots (by Rank): {unit}",
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
#
# import os
#
# import matplotlib
# import pandas as pd
# from clat.intan.rhd import load_intan_rhd_format
# from matplotlib import pyplot as plt
# from analyses.enums.monkey_names import get_monkeys_by_rank
# # matplotlib.use("Qt5Agg")
#
#
# def main():
#     date = "2023-10-05"
#     round = "231005_round2"
#     sorted_spikes_filename = "sorted_spikes.pkl"
#
#     cortana_path = "/home/connorlab/Documents/IntanData/Cortana"
#     round_path = os.path.join(cortana_path, date, round)
#     compiled_trials_filepath = os.path.join(round_path, "compiled.pkl")
#     experiment_name = os.path.basename(os.path.dirname(compiled_trials_filepath))
#     raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)
#
#     # TODO: specify which sorting pickle to use and which units to plot, then add them to dataframe
#     rhd_file_path = os.path.join(round_path, "info.rhd")
#     sorted_spikes_filepath = os.path.join(cortana_path, date, round, sorted_spikes_filename)
#     sorted_spikes = pd.read_pickle(sorted_spikes_filepath)
#     sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
#     sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)
#
#     for unit, data in sorted_data['SpikeTimes'][0].items():
#         plot_raster_for_monkeys(sorted_data, unit, experiment_name=experiment_name)
#     # plt.show()
#
#
# def read_sorted_data(round_path, sorted_spikes_filename='sorted_spikes.pkl', compiled_trials_filename='compiled.pkl'):
#     compiled_trials_filepath = os.path.join(round_path, compiled_trials_filename)
#     raw_trial_data = pd.read_pickle(compiled_trials_filepath).reset_index(drop=True)
#     rhd_file_path = os.path.join(round_path, 'info.rhd')
#     sorted_spikes_filepath = os.path.join(round_path, sorted_spikes_filename)
#     sorted_spikes = pd.read_pickle(sorted_spikes_filepath)
#     sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']
#     sorted_data = calculate_spike_timestamps(raw_trial_data, sorted_spikes, sample_rate)
#     return sorted_data
#
#
# def calculate_spike_timestamps(df: pd.DataFrame, spike_indices_by_unit_by_channel: dict, sample_rate: int):
#     """
#     Calculates spike timestamps for each row in the DataFrame.
#
#     Parameters:
#     - df: Pandas DataFrame with a column 'EpochStartStop' containing tuples of (epoch_start, epoch_stop)
#     - spike_indices_by_unit_by_channel: Dictionary of channels to a dict of Units to spike indices
#     - sample_rate: The sample rate for the spike indices
#
#     Returns:
#     - new_df: A new DataFrame with an additional column containing the calculated spike timestamps
#     """
#
#     def single_row_calculation(epoch_start_stop):
#         epoch_start, epoch_stop = epoch_start_stop
#         spikes_tstamps_by_unit = {}
#
#         for channel, spike_indices_by_unit in reversed(spike_indices_by_unit_by_channel.items()):
#             for unit_name, spike_indices in spike_indices_by_unit.items():
#                 new_unit_name = f"{channel}_{unit_name}"
#
#                 # Filter and convert spike indices to timestamps
#                 spike_times = [spike_index / sample_rate for spike_index in spike_indices]
#                 valid_spike_times = []
#                 valid_spike_times = [
#                     spike_time
#                     for spike_time in spike_times
#                     if epoch_start <= spike_time < epoch_stop
#                 ]
#
#                 spikes_tstamps_by_unit[new_unit_name] = valid_spike_times
#
#         return spikes_tstamps_by_unit
#
#     new_df = df.copy(deep=True)
#     new_df['SpikeTimes'] = new_df['EpochStartStop'].apply(single_row_calculation)
#     return new_df
#
#
# def extract_target_unit_data(unit, data):
#     # Get SpikeTimes for channel
#     unit_data = data.copy()
#     unit_data[f'SpikeTimes_{unit}'] = data['SpikeTimes'].apply(lambda x: x[unit] if unit in x else [])
#     return unit_data
#
#
# def plot_raster_for_monkeys(raw_data, unit, experiment_name=None):
#     unit_data = extract_target_unit_data(unit, raw_data)
#     unique_monkey_groups = unit_data['MonkeyGroup'].dropna().unique().tolist()
#     N = len(unit_data)
#
#     max_rows = 0
#     for group_name in unique_monkey_groups:
#         group_data = unit_data[unit_data['MonkeyGroup'] == group_name]
#         unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
#         max_rows = max(max_rows, len(unique_monkeys))
#
#     fig = plt.figure(figsize=(15 * len(unique_monkey_groups), 45 * max_rows))
#
#     for col_idx, group_name in enumerate(unique_monkey_groups):
#         group_data = unit_data[unit_data['MonkeyGroup'] == group_name]
#         unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
#
#         for row_idx, monkey_name in enumerate(unique_monkeys):
#             monkey_data = group_data[group_data['MonkeyName'] == monkey_name]
#
#             ax = fig.add_subplot(max_rows, len(unique_monkey_groups), row_idx * len(unique_monkey_groups) + col_idx + 1)
#
#             filtered_spike_times_list = []
#             for idx, row in monkey_data.iterrows():
#                 spike_times = row[f'SpikeTimes_{unit}']
#                 epoch_start, epoch_stop = row['EpochStartStop']
#
#                 # Filter spikes based on EpochStartStop and subtract the epoch start time
#                 filtered_spike_times = [spike - epoch_start for spike in spike_times if
#                                         epoch_start <= spike <= epoch_stop]
#                 filtered_spike_times_list.append(filtered_spike_times)
#
#             ax.eventplot(filtered_spike_times_list, color='black', linewidths=0.5)
#             ax.set_xlim(0, 2.0)
#             ax.set_yticks([len(filtered_spike_times_list)])
#             # Place the title text to the right of the subplot
#             ax.text(1.05, 0.5, f"{monkey_name}", transform=ax.transAxes, ha='left', va='center', fontsize=14)
#
#         fig.text(0.5 / len(unique_monkey_groups) + col_idx / len(unique_monkey_groups), 0.95, f'{group_name}',
#                  ha='center', va='center')
#
#     # fig.text(0.5, 0.01, 'Monkey Groups', ha='center', va='center')
#     fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center', rotation='horizontal')
#     fig.text(0.99, 0.95, f'N: {N}', ha='right', va='bottom')
#     fig.suptitle(f'Raster Plots for Individual Monkeys: Channel: {unit}')
#
#     plt.subplots_adjust(hspace=1.0, wspace=1.0)
#
#     plt.show()
#     ## SAVE PLOTS
#     base_save_dir = "/png_raster_plots/"
#     if experiment_name is not None:
#         save_dir = os.path.join(base_save_dir, experiment_name)
#         os.makedirs(save_dir, exist_ok=True)
#
#         # Save individual plot
#         individual_save_path_png = os.path.join(save_dir, f"{experiment_name}_{unit}.png")
#         individual_save_path_svg = os.path.join(save_dir, f"{experiment_name}_{unit}.svg")
#         # fig.savefig(individual_save_path_svg)
#         print(f"saving individual plots to {individual_save_path_png}")
#         fig.savefig(individual_save_path_png)
#
#     return fig
#
#
# def plot_raster_for_monkeys_by_rank(raw_data, unit, experiment_name=None):
#     unit_data = extract_target_unit_data(unit, raw_data)
#     unique_monkey_groups = unit_data['MonkeyGroup'].dropna().unique().tolist()
#     N = len(unit_data)
#
#     max_rows = 0
#     for group_name in unique_monkey_groups:
#         group_data = unit_data[unit_data['MonkeyGroup'] == group_name]
#         unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
#         max_rows = max(max_rows, len(unique_monkeys))
#
#     fig = plt.figure(figsize=(15 * len(unique_monkey_groups), 45 * max_rows))
#
#     for col_idx, group_name in enumerate(unique_monkey_groups):
#         group_data = unit_data[unit_data['MonkeyGroup'] == group_name]
#         unique_monkeys = group_data['MonkeyName'].dropna().unique().tolist()
#         unique_monkeys_in_order = get_monkeys_by_rank(group_name)
#         monkeys_list = [item for item in unique_monkeys_in_order if item in unique_monkeys]
#
#         for row_idx, monkey_name in enumerate(monkeys_list):
#             monkey_data = group_data[group_data['MonkeyName'] == monkey_name]
#
#             ax = fig.add_subplot(max_rows, len(unique_monkey_groups),
#                                  row_idx * len(unique_monkey_groups) + col_idx + 1)
#
#             filtered_spike_times_list = []
#             for idx, row in monkey_data.iterrows():
#                 spike_times = row[f'SpikeTimes_{unit}']
#                 epoch_start, epoch_stop = row['EpochStartStop']
#
#                 # Filter spikes based on EpochStartStop and subtract the epoch start time
#                 filtered_spike_times = [spike - epoch_start for spike in spike_times if
#                                         epoch_start <= spike <= epoch_stop]
#                 filtered_spike_times_list.append(filtered_spike_times)
#
#             ax.eventplot(filtered_spike_times_list, color='black', linewidths=0.5)
#             ax.set_xlim(0, 2.0)
#             ax.set_yticks([len(filtered_spike_times_list)])
#             # Place the title text to the right of the subplot
#             ax.text(1.05, 0.5, f"{monkey_name}", transform=ax.transAxes, ha='left', va='center', fontsize=14)
#
#         fig.text(0.5 / len(unique_monkey_groups) + col_idx / len(unique_monkey_groups), 0.95, f'{group_name}',
#                  ha='center', va='center')
#
#     # fig.text(0.5, 0.01, 'Monkey Groups', ha='center', va='center')
#     fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center', rotation='horizontal')
#     fig.text(0.99, 0.95, f'N: {N}', ha='right', va='bottom')
#     fig.suptitle(f'Raster Plots for Individual Monkeys: Channel: {unit}')
#
#     plt.subplots_adjust(hspace=1.0, wspace=1.0)
#
#     plt.show()
#
#     return fig
#
#
# if __name__ == '__main__':
#     main()
