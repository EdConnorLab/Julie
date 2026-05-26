"""
compile_posthoc_manually_sorted.py — Compile neural data using post-hoc manually sorted spikes.

Reads epoch boundaries from Intan marker channels + livenotes,
and spike clusters from a pre-sorted pickle file.
"""

import os
import pickle
from datetime import date

from clat.compile.task.cached_task_fields import CachedTaskDatabaseField
from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels
from clat.intan.rhd import load_intan_rhd_format

from compile.compile_common import (
    make_connections, intan_experiment_path, collect_task_ids, base_fields,
)



def _load_sorted_spikes(path: str) -> dict:
    """Load a dict of manually sorted spike indices from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


class EpochStartStopTimesField(CachedTaskDatabaseField):
    """Converts sample-index epoch boundaries to seconds."""

    def __init__(self, conn, epoch_indices_by_task_id: dict, sample_rate: float):
        self.epoch_indices_by_task_id = epoch_indices_by_task_id
        self.sample_rate = sample_rate
        super().__init__(conn)

    def get(self, task_id: int):
        try:
            start_idx, stop_idx = self.epoch_indices_by_task_id[task_id]
        except KeyError:
            return None

        start_sec = start_idx / self.sample_rate
        stop_sec = stop_idx / self.sample_rate
        if stop_sec - start_sec < 1:
            print(f"Warning: epoch for task {task_id} is < 1 s ({stop_sec - start_sec:.3f} s)")
        return start_sec, stop_sec

    def get_name(self):
        return "EpochStartStop"


class ManuallySortedSpikeTStampField(EpochStartStopTimesField):
    """Extracts per-unit spike timestamps that fall within each task's epoch."""

    def __init__(self, conn, spike_indices_by_unit_by_channel: dict,
                 sample_rate: float, epoch_indices_by_task_id: dict):
        self.spike_indices_by_unit_by_channel = spike_indices_by_unit_by_channel
        super().__init__(conn, epoch_indices_by_task_id, sample_rate)

    def get(self, task_id: int):
        epoch = self.get_cached_super(
            task_id, EpochStartStopTimesField,
            self.epoch_indices_by_task_id, self.sample_rate,
        )
        if epoch is None:
            return None

        epoch_start, epoch_stop = epoch
        spikes_by_unit: dict[str, list[float]] = {}

        for channel, units in self.spike_indices_by_unit_by_channel.items():
            for unit_name, spike_indices in units.items():
                key = f"{channel}_{unit_name}"
                for idx in spike_indices:
                    tstamp = idx / self.sample_rate
                    if epoch_start <= tstamp < epoch_stop:
                        spikes_by_unit.setdefault(key, []).append(tstamp)

        return spikes_by_unit

    def get_name(self):
        return "SortedSpikeTimes"



def compile_data(*, experiment_name: str, day: date):
    """Compile post-hoc manually sorted neural data for a single experiment session."""
    conn_xper, conn_photo = make_connections(day)
    task_ids = collect_task_ids(conn_xper, day)

    exp_path = intan_experiment_path(day, experiment_name)

    # Read sample rate from Intan header
    rhd = load_intan_rhd_format.read_data(os.path.join(exp_path, "info.rhd"))
    sample_rate = rhd["frequency_parameters"]["amplifier_sample_rate"]

    # Build epoch map from marker channels + livenotes
    stim_epochs = epoch_using_marker_channels(
        os.path.join(exp_path, "digitalin.dat"),
        false_negative_correction_duration=10,
    )
    epochs_by_task_id = map_task_id_to_epochs_with_livenotes(
        os.path.join(exp_path, "notes.txt"),
        stim_epochs,
    )

    # Load pre-sorted spikes
    sorted_spikes = _load_sorted_spikes(os.path.join(exp_path, "sorted_spikes.pkl"))

    # Assemble fields
    fields = base_fields(conn_xper, conn_photo)
    fields.append(EpochStartStopTimesField(conn_xper, epochs_by_task_id, sample_rate))
    fields.append(ManuallySortedSpikeTStampField(conn_xper, sorted_spikes, sample_rate, epochs_by_task_id))

    # Compile, clean, and save
    data = fields.to_data(task_ids)
    data = data[data["EpochStartStop"].notna()]
    data.to_pickle(os.path.join(exp_path, "compiled.pkl"))

    print(data.to_string())
    return data


if __name__ == "__main__":
    compile_data(
        experiment_name="231030_round2",
        day=date(2023, 10, 30),
    )