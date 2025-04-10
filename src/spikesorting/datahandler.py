import os
import pickle
import re
from typing import Dict, List

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QWidget, QInputDialog, QLineEdit
from clat.intan.amplifiers import read_amplifier_data_with_mmap
from clat.intan.channels import Channel
from clat.intan.rhd import load_intan_rhd_format
from scipy.signal import butter, filtfilt
from windowsort.datahandler import SortingConfigManager
from windowsort.drift import DriftingTimeAmplitudeWindow
from windowsort.units import Unit


class InputDataManager:
    def __init__(self, intan_file_directory):
        self.intan_file_directory = intan_file_directory
        self.voltages_by_channel: dict[Channel, np.ndarray] = {}
        self.sample_rate = None  # You can initialize this from info.rhd if needed
        self.read_data()

    def read_data(self):
        # Paths for original and preprocessed data
        info_rhd_path = os.path.join(self.intan_file_directory, "info.rhd")
        amplifier_dat_path = os.path.join(self.intan_file_directory, "amplifier.dat")
        self.preprocessed_dat_path = os.path.join(self.intan_file_directory, "preprocessed_data.dat")

        # Extract information from info.rhd
        data = load_intan_rhd_format.read_data(info_rhd_path)
        amplifier_channels = data['amplifier_channels']
        self.sample_rate = data['frequency_parameters']['amplifier_sample_rate']

        # Check if preprocessed data already exists
        if os.path.exists(self.preprocessed_dat_path):
            print("Preprocessed data found. Loading...")
            self.voltages_by_channel = read_amplifier_data_with_mmap(self.preprocessed_dat_path, amplifier_channels)
        else:
            print("Preprocessed data not found. Preprocessing and saving...")
            # Load original data
            self.voltages_by_channel = read_amplifier_data_with_mmap(amplifier_dat_path, amplifier_channels)
            # Preprocess and save the data
            self.preprocess_data()
            del self.voltages_by_channel  # Delete the original data to save memory
            self.voltages_by_channel = read_amplifier_data_with_mmap(self.preprocessed_dat_path, amplifier_channels)

    def preprocess_data(self):
        # New dictionary to hold preprocessed data
        preprocessed_data = {}

        # Iterate through channels, filter data, and store in new dict
        for channel, voltages in self.voltages_by_channel.items():
            filtered_voltages = self._highpass_filter(voltages, cutoff=300)
            preprocessed_data[channel] = filtered_voltages.astype(
                'int16')  # Convert back to int16

        # Save preprocessed data to a new binary file
        self._save_preprocessed_data(preprocessed_data, self.preprocessed_dat_path)

    def _highpass_filter(self, data, cutoff=300, order=5):
        b, a = self._butter_highpass(cutoff, self.sample_rate, order=order)
        y = filtfilt(b, a, data)
        return y

    def _butter_highpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _save_preprocessed_data(self, preprocessed_data, output_file_path, chunk_size=10000):
        # Get the number of channels and samples
        num_channels = len(preprocessed_data)
        num_samples = len(next(iter(preprocessed_data.values())))

        with open(output_file_path,
                  'wb') as f:  # 'wb' mode for binary writing
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = start_idx + chunk_size

                # Create an empty NumPy array to hold chunk of the data
                current_chunk_size = min(chunk_size, num_samples - start_idx)
                data_chunk = np.empty((current_chunk_size, num_channels), dtype='int16')

                # Populate the chunk with your preprocessed data
                for i, (channel, data) in enumerate(preprocessed_data.items()):
                    # Convert the data to a format that, when multiplied by 0.195, gives the correct microvolt values
                    data_chunk[:, i] = (data[start_idx:end_idx] / 0.195).astype('int16')

                    # Save the chunk to the binary file
                f.write(data_chunk.tobytes())

        print(f"Preprocessed Data saved to {output_file_path}.")


class SortedSpikeExporter:
    def __init__(self, *, save_directory):
        self.thresholded_spike_indices_by_channel = {}  # Keyed by channel, each value is a list of spike times
        self.sorted_spikes_by_unit_by_channel = {}  # Keyed by channel, each value is a dict of unit name to spike times
        self.save_directory = save_directory

    def update_thresholded_spikes(self, channel, thresholded_spike_indices):
        self.thresholded_spike_indices_by_channel[channel] = thresholded_spike_indices

    def save_thresholded_spikes(self):
        filename = os.path.join(self.save_directory, "thresholded_spikes.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.thresholded_spike_indices_by_channel, f)
            # print(f"Saved {len(self.thresholded_spikes_by_channel.items())} thresholded spikes to {self.filename}")
        print(self.thresholded_spike_indices_by_channel)

    def save_sorted_spikes(self, spikes_by_unit: Dict[str, np.ndarray], channel, label=None):
        base_filename = "sorted_spikes"
        if label is not None:
            filename = base_filename + "_" + label + ".pkl"
        else:
            filename = base_filename + ".pkl"

        filename = os.path.join(self.save_directory, filename)

        # First, check if the file already exists.
        if os.path.exists(filename):
            # Load the existing data.
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = {}

            # Update the specific channel's data in-memory.
        existing_data[channel] = spikes_by_unit

        # Now, save the updated data back to the file.
        with open(filename, 'wb') as f:
            pickle.dump(existing_data, f)

        for unit_name, spikes in spikes_by_unit.items():
            print(f"Saved {len(spikes)} spikes for unit {unit_name} to {filename}")


class JulieSortingConfigManager(SortingConfigManager):
    current_sorting_config_path: str = None

    def __init__(self, *, save_directory, voltage_time_plot, spike_plot, sort_panel, data_exporter):
        self.voltage_time_plot = voltage_time_plot
        self.spike_plot = spike_plot
        self.sort_panel = sort_panel
        self.data_exporter = data_exporter
        self.save_directory = save_directory
        self._set_current_sorting_config_path(os.path.join(self.save_directory, "sorting_config.pkl"))

    def open_current_sorting_config(self):
        channel = self.spike_plot.current_channel
        config = self._open_sorting_config(self.current_sorting_config_path, channel)
        self._apply_config(config)

    def open_selected_sorting_config(self):
        channel = self.spike_plot.current_channel
        print(f"Loading sorting config for channel {channel}")
        config = self._select_sorting_config(channel, self.sort_panel)
        self._apply_config(config)

    def save(self):
        channel = self.spike_plot.current_channel
        sorted_spikes_by_unit = self.sort_panel.sort_all_spikes(channel)

        file_label = self._get_current_file_label()
        # Use the DataExporter to save the sorted spikes
        self.data_exporter.save_sorted_spikes(sorted_spikes_by_unit, channel, label=file_label)
        self._save_sorting_config(channel, self.spike_plot.amp_time_windows,
                                  self.spike_plot.units,
                                  self.spike_plot.current_threshold_value,
                                  label=file_label)

    def save_as(self):
        channel = self.spike_plot.current_channel
        sorted_spikes_by_unit = self.sort_panel.sort_all_spikes(channel)

        file_label = self._query_file_label()
        print(file_label)

        # Use the DataExporter to save the sorted spikes
        self.data_exporter.save_sorted_spikes(sorted_spikes_by_unit, channel, label=file_label)
        self._save_sorting_config(channel, self.spike_plot.amp_time_windows,
                                  self.spike_plot.units,
                                  self.spike_plot.current_threshold_value,
                                  label=file_label)

    def _apply_config(self, config):
        if config:
            # Add threshold
            threshold = config['threshold']
            self.voltage_time_plot.update_threshold(threshold)
            self.voltage_time_plot.threshold_line.setValue(threshold)

            self.sort_panel.clear_all_unitpanels()
            self.spike_plot.clear_amp_time_windows()
            self.spike_plot.clear_units()

            # Add the amp time windows
            for window in config['amp_time_windows']:
                self.spike_plot.load_amp_time_window(window)

            self.unit_counter = 0
            for logical_expression, unit_name, color in config['units']:
                unit = Unit(logical_expression, unit_name, color)
                self.sort_panel.load_unit(unit)

            self.spike_plot.updatePlot()
            self.spike_plot.sortSpikes()

    def _query_file_label(self):
        # Open Input Dialog to get the filename extension
        text, ok = QInputDialog.getText(self.sort_panel, 'Input Dialog', 'Enter filename label:', QLineEdit.Normal, "")

        if ok and text:
            return text

    def _get_current_file_label(self):
        pattern = r"sorting_config_(.*?).pkl"
        match = re.search(pattern, self.current_sorting_config_path)
        return match.group(1) if match else None

    def _save_sorting_config(self, channel, amp_time_windows: List[DriftingTimeAmplitudeWindow], units, threshold,
                             label=None):
        base_filename = "sorting_config"
        if label is not None:
            filename = base_filename + "_" + label + ".pkl"
        else:
            filename = base_filename + ".pkl"
        filename = os.path.join(self.save_directory, filename)
        self._set_current_sorting_config_path(filename)

        try:
            with open(filename, 'rb') as f:
                all_configs = pickle.load(f)
        except FileNotFoundError:
            all_configs = {}

        all_configs[channel] = {
            'amp_time_windows': [window.time_control_points for window in amp_time_windows],
            'units': [(unit.logical_expression, unit.unit_name, unit.color) for unit in units],
            'threshold': threshold
        }

        with open(filename, 'wb') as f:
            pickle.dump(all_configs, f)

        print("Saved sorting configs to: ", filename)

    def _select_sorting_config(self, channel: Channel, parent_widget: QWidget):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(parent_widget, "Open File", self.save_directory,
                                                  "Sorting Config Files (sorting_config*.pkl);;All Files (*)",
                                                  options=options)

        return self._open_sorting_config(filename, channel)

    def _set_current_sorting_config_path(self, filename):
        self.current_sorting_config_path = filename

    def _open_sorting_config(self, filename, channel):
        if filename:
            try:
                with open(filename, 'rb') as f:
                    all_configs = pickle.load(f)
                self._set_current_sorting_config_path(filename)
                for key, value in all_configs.items():
                    if key.value == channel.value:
                        return value
                # return all_configs.get(channel, None)
            except FileNotFoundError:
                print(f"Configuration file {filename} not found.")
                return None
            except Exception as e:
                print(f"An error occurred while loading the configuration file: {e}")
                return None
