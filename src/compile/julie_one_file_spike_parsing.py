import os
from dataclasses import dataclass

from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels
from clat.intan.rhd import load_intan_rhd_format
from clat.intan.spike_file import fetch_spike_tstamps_from_file


@dataclass
class OneFileParser:

    def parse(self, intan_file_path: str):
        spike_path = os.path.join(intan_file_path, "spike.dat")
        digital_in_path = os.path.join(intan_file_path, "digitalin.dat")
        notes_path = os.path.join(intan_file_path, "notes.txt")

        spike_tstamps_for_channels, sample_rate = fetch_spike_tstamps_from_file(spike_path)
        stim_epochs_from_markers = epoch_using_marker_channels(digital_in_path, false_negative_correction_duration=2)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(notes_path,
                                                                   stim_epochs_from_markers)

        filtered_spikes_for_channels_by_task_id = {}
        epoch_start_stop_times_by_task_id = {}
        for task_id, epoch in epochs_for_task_ids.items():
            filtered_spikes_for_channels = {}
            for channel, tstamps in spike_tstamps_for_channels.items():
                passed_filter = []
                for spike_tstamp in tstamps:
                    spike_index = int(spike_tstamp * sample_rate)
                    if epoch[0] <= spike_index <= epoch[1]:
                        passed_filter.append(spike_tstamp)
                filtered_spikes_for_channels[channel] = passed_filter

            epoch_start = epoch[0] / sample_rate
            epoch_end = epoch[1] / sample_rate
            epoch_start_stop_times_by_task_id[task_id] = (epoch_start, epoch_end)
            filtered_spikes_for_channels_by_task_id[task_id] = filtered_spikes_for_channels
        return filtered_spikes_for_channels_by_task_id, epoch_start_stop_times_by_task_id, sample_rate

    def parse_with_peristimulus_spikes(self, intan_file_path: str, pre_stimulus_time: float = 0.5,
                                       post_stimulus_time: float = 0.3):
        spike_path = os.path.join(intan_file_path, "spike.dat")
        digital_in_path = os.path.join(intan_file_path, "digitalin.dat")
        notes_path = os.path.join(intan_file_path, "notes.txt")

        spike_tstamps_for_channels, sample_rate = fetch_spike_tstamps_from_file(spike_path)
        stim_epochs_from_markers = epoch_using_marker_channels(digital_in_path, false_negative_correction_duration=2)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(notes_path,
                                                                   stim_epochs_from_markers)

        filtered_spikes_for_channels_by_task_id = {}
        unfiltered_spikes_for_channels_by_task_id = {}
        epoch_start_stop_times_by_task_id = {}
        for task_id, epoch in epochs_for_task_ids.items():
            filtered_spikes_for_channels = {}
            unfiltered_spikes_for_channels = {}
            for channel, tstamps in spike_tstamps_for_channels.items():
                passed_filter = []
                unfiltered = []
                for spike_tstamp in tstamps:
                    spike_index = int(spike_tstamp * sample_rate)
                    pre_stimulus_time_index = int(pre_stimulus_time * sample_rate)
                    post_stimulus_time_index = int(post_stimulus_time * sample_rate)
                    if epoch[0] - pre_stimulus_time_index <= spike_index <= epoch[1] + post_stimulus_time_index:
                        unfiltered.append(spike_tstamp)
                    if epoch[0] <= spike_index <= epoch[1]:
                        passed_filter.append(spike_tstamp)
                filtered_spikes_for_channels[channel] = passed_filter
                unfiltered_spikes_for_channels[channel] = unfiltered

            epoch_start = epoch[0] / sample_rate
            epoch_end = epoch[1] / sample_rate
            epoch_start_stop_times_by_task_id[task_id] = (epoch_start, epoch_end)
            filtered_spikes_for_channels_by_task_id[task_id] = filtered_spikes_for_channels
            unfiltered_spikes_for_channels_by_task_id[task_id] = unfiltered_spikes_for_channels
        return unfiltered_spikes_for_channels_by_task_id, filtered_spikes_for_channels_by_task_id, epoch_start_stop_times_by_task_id, sample_rate

    def parse_without_filtering_spikes(self, intan_file_path: str):
        spike_path = os.path.join(intan_file_path, "spike.dat")
        digital_in_path = os.path.join(intan_file_path, "digitalin.dat")
        notes_path = os.path.join(intan_file_path, "notes.txt")
        rhd_file_path = os.path.join(intan_file_path, "info.rhd")
        sample_rate = load_intan_rhd_format.read_data(rhd_file_path)["frequency_parameters"]['amplifier_sample_rate']

        spike_tstamps_for_channels, sample_rate = fetch_spike_tstamps_from_file(spike_path)
        stim_epochs_from_markers = epoch_using_marker_channels(digital_in_path, false_negative_correction_duration=2)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(notes_path,
                                                                   stim_epochs_from_markers)

        epoch_start_stop_times_by_task_id = {}
        for task_id, epoch in epochs_for_task_ids.items():
            epoch_start = epoch[0] / sample_rate
            epoch_end = epoch[1] / sample_rate
            epoch_start_stop_times_by_task_id[task_id] = (epoch_start, epoch_end)

        return spike_tstamps_for_channels, epoch_start_stop_times_by_task_id, sample_rate
