from typing import List, Tuple

import numpy as np
from channel_enum_resolvers import get_value_from_dict_with_channel
from single_channel_analysis import get_spike_count
from spike_rate_computation import get_raw_data_and_channels_from_files, get_raw_spike_tstamp_data


def get_inter_trial_intervals(trial_epochs: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    inter_trial_intervals = []
    total_inter_trial_duration = 0
    for i in range(0, len(trial_epochs)):
        if i == 0:
            # start, _ = trial_epoch=[i]
            # pre_experiment_interval = (0, start)
            # total_inter_trial_duration += start
            # inter_trial_intervals.append(pre_experiment_interval)
            continue
        else:
            _, stop = trial_epochs[i - 1]
            start, _ = trial_epochs[i]
            inter_trial_interval = (stop, start)
            total_inter_trial_duration += (start - stop)
            inter_trial_intervals.append(inter_trial_interval)
    return inter_trial_intervals, total_inter_trial_duration


def get_all_spikes_within_interval_by_channel(spike_timestamps_by_channel, intervals: list) -> dict:
    spikes_by_channel = {}
    for channel, timestamps in spike_timestamps_by_channel.items():
        spike_times = []
        for interval in intervals:
            start, stop = interval
            for timestamp in timestamps:
                if start < timestamp < stop:
                    spike_times.append(timestamp)
        spikes_by_channel[channel] = spike_times

    return spikes_by_channel


def compute_spontaneous_firing_rate_by_channel(spontaneous_spikes_by_channel, total_inter_trial_duration,
                                               valid_channels):
    spontaneous_firing_rate_by_channel = {}
    for channel in valid_channels:
        spikes = get_value_from_dict_with_channel(channel, spontaneous_spikes_by_channel)
        if spikes is not None:
            spontaneous_spike_count = len(spikes)
            spontaneous_firing_rate = spontaneous_spike_count / total_inter_trial_duration
            spontaneous_firing_rate_by_channel[channel] = spontaneous_firing_rate
            # print(f"spon spike rate (sp/s) for {channel} is {spontaneous_firing_rate} with total spikes {spontaneous_spike_count}")
            # print(f"for 50 ms... {spike_rate * 0.05}")
        else:
            spontaneous_firing_rate_by_channel[channel] = 0
    return spontaneous_firing_rate_by_channel


def get_average_spontaneous_firing_rate(date, round_no):
    # Get raw spike data
    raw_spike_timestamp_data, sample_rate = get_raw_spike_tstamp_data(date, round_no)
    # Get trial epochs
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
    trial_intervals = raw_unsorted_data['EpochStartStop']
    inter_trial_intervals, total_duration = get_inter_trial_intervals(trial_intervals.tolist())
    spontaneous_spikes_by_channel = get_all_spikes_within_interval_by_channel(raw_spike_timestamp_data,
                                                                              inter_trial_intervals)
    average_spontaneous_firing_rate_by_channel = compute_spontaneous_firing_rate_by_channel(
        spontaneous_spikes_by_channel, total_duration,
        valid_channels)
    return average_spontaneous_firing_rate_by_channel


def compute_standard_deviation_of_spontaneous_spike_count_for_time_chunk(inter_trial_intervals, valid_channels,
                                                                         raw_spike_tstamp_data, chunk_size):
    spike_count_std_dev_by_channel = {}
    for channel in valid_channels:
        data = get_value_from_dict_with_channel(channel, raw_spike_tstamp_data)
        spike_count_for_each_channel = []
        for inter_trial_interval in inter_trial_intervals:
            start, end = inter_trial_interval
            time_chunks = [start + i * chunk_size for i in
                           range(int((end - start) / chunk_size) + 1)]
            for i in range(len(time_chunks) - 1):
                time_range = (time_chunks[i], time_chunks[i + 1])
                spike_count = get_spike_count(data, time_range)
                spike_count_for_each_channel.append(spike_count)
        std_dev = np.std(spike_count_for_each_channel)
        spike_count_std_dev_by_channel[channel] = std_dev
    # print(f"{spike_count_std_dev_by_channel}")

    return spike_count_std_dev_by_channel


if __name__ == "__main__":
    date = "2023-09-26"
    round_no = 1

    '''
    spon_firing_rate_by_channel = get_average_spontaneous_firing_rate(date, round_no)
    # Get raw spike data
    raw_spike_timestamp_data, sample_rate = get_raw_spike_tstamp_data(date, round_no)

    # Get trial epochs
    raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
    trial_intervals = raw_unsorted_data['EpochStartStop']

    inter_trial_intervals, total_duration= get_inter_trial_intervals(trial_intervals.tolist())
    spon_spikes_by_channel = get_all_spikes_within_interval_by_channel(raw_spike_timestamp_data, inter_trial_intervals)
    spon_firing_rate_by_channel = compute_spontaneous_firing_rate_by_channel(spon_spikes_by_channel, total_duration, valid_channels)



    #### Just checking trial spike rate

    total_trial_duration = 0
    for trial_interval in trial_intervals:
        trial_start, trial_stop = trial_interval
        total_trial_duration += trial_stop - trial_start

    trial_spikes_by_channel = {}
    for channel, trial_spike_tstamps in raw_spike_timestamp_data.items():
        trial_spike_times = []
        for trial_interval in trial_intervals:
            trial_start, trial_stop = trial_interval
            for trial_spike_tstamp in trial_spike_tstamps:
                if trial_start < trial_spike_tstamp < trial_stop:
                    trial_spike_times.append(trial_spike_tstamp)
        trial_spikes_by_channel[channel] = trial_spike_times

    trial_firing_rate_by_channel = {}
    for channel in valid_channels:
        final_trial_spikes = get_value_from_dict_with_channel(channel, trial_spikes_by_channel)
        if final_trial_spikes is not None:
            trial_spike_count = len(final_trial_spikes)
            trial_firing_rate = trial_spike_count / total_trial_duration
            trial_firing_rate_by_channel[channel] = trial_firing_rate
            print(f"spike rate (sp/s) for {channel} is {trial_firing_rate} with total spikes {trial_spike_count}")

    for key in spon_firing_rate_by_channel.keys():
        if key in trial_firing_rate_by_channel.keys():
            spon_firing_rate = spon_firing_rate_by_channel[key]
            trial_firing_rate = trial_firing_rate_by_channel[key]
            print(f"{key}: spon {spon_firing_rate} and trial {trial_firing_rate}")
            if spon_firing_rate < trial_firing_rate:
                print(f"trial increases the firing rate by {trial_firing_rate-spon_firing_rate:.2f}.")
            else:
                pass
    '''
