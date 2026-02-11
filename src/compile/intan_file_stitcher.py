
import os
import shutil
import tkinter as tk
from tkinter import simpledialog

import tkfilebrowser



class IntanFileStitcher:
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths

    def read_append_write(self, filename, output_folder):
        with open(os.path.join(output_folder, filename), 'wb') as output_file:
            for folder in self.folder_paths:
                input_file_path = os.path.join(folder, filename)
                with open(input_file_path, 'rb') as input_file:
                    shutil.copyfileobj(input_file, output_file)

    def append_notes(self, filename, output_folder):
        cumulative_last_index = 0
        with open(os.path.join(output_folder, filename), 'w') as output_file:
            for folder in self.folder_paths:
                digital_in_path = os.path.join(folder, 'digitalin.dat')
                number_of_samples = read_number_of_samples_from_digital_in(digital_in_path)

                input_file_path = os.path.join(folder, filename)
                with open(input_file_path, 'r') as input_file:
                    lines = input_file.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        index, timestamp, info = line.split(", ")
                        new_index = int(index) + cumulative_last_index
                        output_file.write(f"{new_index}, {timestamp}, {info}\n\n")  # Added two extra newlines

                # After processing all notes from this folder, update the cumulative index
                cumulative_last_index += number_of_samples
    def copy_auxiliary_files(self, filename, output_folder):
        source_path = os.path.join(self.folder_paths[0], filename)
        destination_path = os.path.join(output_folder, filename)
        shutil.copyfile(source_path, destination_path)

    def stitch_files(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.append_notes('notes.txt', output_folder)

        files_to_stitch = ['amplifier.dat', 'digitalin.dat']
        for filename in files_to_stitch:
            self.read_append_write(filename, output_folder)


        auxiliary_files = ['info.rhd', 'info.rhs', 'settings.xml']
        for filename in auxiliary_files:
            try:
                self.copy_auxiliary_files(filename, output_folder)
            except FileNotFoundError:
                print(f"Warning: {filename} not found in {self.folder_paths[0]}")

        self.create_merge_info(output_folder)

    def create_merge_info(self, output_folder):
        with open(os.path.join(output_folder, "mergeinfo.txt"), 'w') as f:
            for folder in self.folder_paths:
                folder_name = os.path.basename(folder)
                f.write(f"{folder_name}\n")

    def stitch_spike_dat(self, output_folder):
        """
        Stitch multiple spike.dat files together, updating timestamps correctly.
        This handles the special binary format of spike.dat files.
        """
        # Initialize structures to store data from all folders
        all_spikes = {}
        last_sample_rate = None

        # First read all spike data from all folders
        file_offsets = {}  # To track time offsets for each file
        max_timestamp = 0

        print("Processing spike.dat files...")

        # First pass: read all files and determine timestamp offsets
        for i, folder in enumerate(self.folder_paths):
            input_file_path = os.path.join(folder, 'spike.dat')
            if not os.path.exists(input_file_path):
                print(f"Warning: spike.dat not found in {folder}")
                continue

            if i == 0:
                file_offsets[folder] = 0  # First file has no offset
            else:
                file_offsets[folder] = max_timestamp  # Offset for subsequent files

            # Read the spike file and extract data
            spikes, sample_rate = self.read_intan_spike_file(input_file_path, no_artifacts=False)

            if last_sample_rate is not None and sample_rate != last_sample_rate:
                print(f"Warning: Sample rate mismatch between folders: {last_sample_rate} vs {sample_rate}")

            last_sample_rate = sample_rate

            # Process each channel in the spike data
            for channel_data in spikes:
                channel_name = channel_data[0]  # Native channel name
                custom_name = channel_data[1]  # Custom channel name
                timestamps = channel_data[2]  # Timestamps
                spike_ids = channel_data[3]  # Spike IDs

                # Get snapshots if they exist
                snapshots = channel_data[4] if len(channel_data) > 4 else None

                # Update timestamps based on file offset
                adjusted_timestamps = [t + file_offsets[folder] / sample_rate for t in timestamps]

                # Find the maximum timestamp for the next offset calculation
                if adjusted_timestamps:
                    current_max = max(adjusted_timestamps)
                    max_timestamp = max(max_timestamp, current_max * sample_rate)

                # Initialize channel in all_spikes if not already present
                if channel_name not in all_spikes:
                    all_spikes[channel_name] = {
                        'custom_name': custom_name,
                        'timestamps': [],
                        'spike_ids': [],
                        'snapshots': [] if snapshots else None
                    }

                # Append data for this channel
                all_spikes[channel_name]['timestamps'].extend(adjusted_timestamps)
                all_spikes[channel_name]['spike_ids'].extend(spike_ids)
                if snapshots:
                    all_spikes[channel_name]['snapshots'].extend(snapshots)

        # Now write the combined data to a new spike.dat file
        output_file_path = os.path.join(output_folder, 'spike.dat')

        if all_spikes and last_sample_rate:
            self.write_intan_spike_file(output_file_path, all_spikes, last_sample_rate)
            print(f"Created stitched spike.dat file at {output_file_path}")
        else:
            print("No spike data found to stitch.")
def open_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    default_folder = "/home/r2_allen/Documents/JulieIntanData/Cortana"
    # default_folder = "/run/user/1003/gvfs/sftp:host=172.30.6.58/home/connorlab/Documents/IntanData"

    folder_paths = tkfilebrowser.askopendirnames(initialdir=default_folder, title="Select Folders to Stitch")
    folder_paths = list(folder_paths)  # Convert tuple to list

    if folder_paths:  # If folders were selected
        output_folder_name = simpledialog.askstring("Output Folder", "Enter the name for the output folder:")

        if output_folder_name:  # If an output folder name was provided
            # Get the parent directory of each selected folder
            parent_directories = [os.path.dirname(folder) for folder in folder_paths]

            # Get the common parent directory of all selected folders
            common_parent_directory = os.path.commonprefix(parent_directories)

            # Append the output folder name to the common parent directory
            final_output_folder_path = os.path.join(common_parent_directory, output_folder_name)

            # Run the stitcher
            stitcher = IntanFileStitcher(folder_paths)
            stitcher.stitch_files(final_output_folder_path)

def read_number_of_samples_from_digital_in(full_file_name) -> int:
    fid = open(full_file_name, 'rb')
    filesize = os.path.getsize(full_file_name)

    num_samples = filesize // 2  # uint16 = 2 bytes

    fid.close()

    return num_samples
def main():
    open_gui()

if __name__ == '__main__':
    main()