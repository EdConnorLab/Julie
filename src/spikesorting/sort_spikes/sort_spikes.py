import argparse
import os
from datetime import datetime

import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from clat.intan.rhd import load_intan_rhd_format
from probeinterface import generate_linear_probe

from compile.compile_common import INTAN_BASE_PATH

# 32-channel linear probe: maps contact index → Intan native_order
PROBE_CHANNEL_ORDER = np.array([
    25, 6, 21, 10, 26, 5, 20, 11, 22, 9, 27, 4, 19, 12, 28, 3,
    24, 7, 18, 13, 29, 2, 17, 14, 31, 0, 23, 8, 30, 1, 16, 15,
])


def get_recording_session_info(intan_file_dir):
    info_rhd_path = os.path.join(intan_file_dir, "info.rhd")
    data = load_intan_rhd_format.read_data(info_rhd_path)
    enabled_channels = data['amplifier_channels']
    sample_rate = data['frequency_parameters']['amplifier_sample_rate']
    return sample_rate, enabled_channels


def compute_device_channel_index(enabled_channels):
    native_to_recidx = {}
    for rec_idx, ch in enumerate(enabled_channels):
        native_to_recidx[ch["native_order"]] = rec_idx

    device_channel_idx = []
    for native_order in PROBE_CHANNEL_ORDER:
        rec_idx = native_to_recidx.get(native_order, -1)
        device_channel_idx.append(rec_idx)
    return device_channel_idx


def build_intan_session_path(date_str, round_no):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    round_folder = f"{date_obj.strftime('%y%m%d')}_round{round_no}"
    return os.path.join(INTAN_BASE_PATH, date_str, round_folder)


def load_and_preprocess_recording(intan_dir):
    sampling_frequency, channels = get_recording_session_info(intan_dir)

    recording = si.read_binary(
        os.path.join(intan_dir, "amplifier.dat"),
        sampling_frequency=sampling_frequency,
        dtype=np.int16,
        num_channels=len(channels),
        gain_to_uV=0.195,
        offset_to_uV=0.0,
    )

    probe = generate_linear_probe(
        num_elec=32, ypitch=65,
        contact_shapes="circle", contact_shape_params={"radius": 20},
    )
    device_channel_idx = compute_device_channel_index(channels)
    probe.set_device_channel_indices(device_channel_idx)
    recording = recording.set_probe(probe)

    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed = spre.common_reference(recording_f, reference="global", operator="median")
    return recording_preprocessed


def run_sorters(recording_preprocessed, intan_dir):
    sorting_KS4 = ss.run_sorter(
        sorter_name="kilosort4", recording=recording_preprocessed,
        docker_image="spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0",
        folder=os.path.join(intan_dir, "kilosort4_output"),
        remove_existing_folder=True, verbose=False,
    )
    # Note: if you have to drop one sorter due to memory, drop TDC — its output is largest
    sorting_TDC = ss.run_sorter(
        sorter_name="tridesclous", recording=recording_preprocessed,
        folder=os.path.join(intan_dir, "tridesclous_output"),
        remove_existing_folder=True, verbose=False,
    )
    sorting_MS5 = ss.run_sorter(
        sorter_name="mountainsort5", recording=recording_preprocessed,
        docker_image="spikeinterface/mountainsort5-base:latest",
        folder=os.path.join(intan_dir, "mountainsort5_output"),
        remove_existing_folder=True, verbose=False,
    )
    return sorting_KS4, sorting_TDC, sorting_MS5


def create_analyzers(sorting_KS4, sorting_TDC, sorting_MS5, recording_preprocessed, intan_dir):
    extensions = ["random_spikes", "waveforms", "templates", "correlograms"]

    analyzer_KS4 = si.create_sorting_analyzer(
        sorting=sorting_KS4, recording=recording_preprocessed,
        format='binary_folder', overwrite=True,
        folder=os.path.join(intan_dir, 'analyzer_KS4_binary'),
    )
    analyzer_TDC = si.create_sorting_analyzer(
        sorting=sorting_TDC, recording=recording_preprocessed,
        format='binary_folder', overwrite=True,
        folder=os.path.join(intan_dir, 'analyzer_TDC_binary'),
    )
    analyzer_MS5 = si.create_sorting_analyzer(
        sorting=sorting_MS5, recording=recording_preprocessed,
        format='binary_folder', overwrite=True,
        folder=os.path.join(intan_dir, 'analyzer_MS5_binary'),
    )

    analyzer_KS4.compute(extensions)
    analyzer_TDC.compute(extensions)
    analyzer_MS5.compute(extensions)

    return analyzer_KS4, analyzer_TDC, analyzer_MS5


def run_spike_sorting(date_str, round_no):
    intan_dir = build_intan_session_path(date_str, round_no)
    print(f"Sorting session: {os.path.basename(intan_dir)}")

    recording_preprocessed = load_and_preprocess_recording(intan_dir)
    sorting_KS4, sorting_TDC, sorting_MS5 = run_sorters(recording_preprocessed, intan_dir)
    create_analyzers(sorting_KS4, sorting_TDC, sorting_MS5, recording_preprocessed, intan_dir)

    print(f"Done sorting {date_str} round {round_no}")


if __name__ == '__main__':
    si.set_global_job_kwargs(n_jobs=4, chunk_duration="1s")

    p = argparse.ArgumentParser(description="Run 3-sorter spike sorting on an Intan session")
    p.add_argument("--date", required=True, help="e.g. 2023-09-26")
    p.add_argument("--round", type=int, required=True, dest="round_no")
    args = p.parse_args()

    run_spike_sorting(args.date, args.round_no)
