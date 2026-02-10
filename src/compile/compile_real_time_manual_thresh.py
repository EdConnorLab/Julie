"""
compile_real_time_manual_thresholded.py — Compile neural data with manual spike thresholding.

Supports two modes:
  - Per-experiment: a single Intan file for the whole experimental session (provide experiment_filename).
  - Per-trial: one Intan file per trial (provide start_time and end_time).
"""
from datetime import date, time

from compile.compile_common import (
    FULL_DAY, make_connections, intan_day_path, intan_experiment_path,
    collect_task_ids, base_fields,
)
from compile.julie_intan_file_per_experiment_fields import (
    SpikeTimesForChannelsField_Experiment,
    EpochStartStopField_Experiment,
)
from compile.julie_intan_file_per_trial_fields import (
    SpikeTimesForChannelsField,
    EpochStartStopField,
)
from compile.julie_one_file_spike_parsing import OneFileParser


# ── Read helpers ─────────────────────────────────────────────────────────────

def _read_per_experiment(day: date, experiment_name: str,
                         start_time: time, end_time: time):
    conn_xper, conn_photo = make_connections(day)
    task_ids = collect_task_ids(conn_xper, day, start_time, end_time)

    file_path = intan_experiment_path(day, experiment_name)
    _, filtered_spikes, epoch_start_stop, _ = OneFileParser().parse_with_peristimulus_spikes(file_path)

    fields = base_fields(conn_xper, conn_photo)
    fields.append(SpikeTimesForChannelsField_Experiment(conn_xper, filtered_spikes))
    fields.append(EpochStartStopField_Experiment(conn_xper, epoch_start_stop))
    return fields.to_data(task_ids)


def _read_per_trial(day: date, start_time: time, end_time: time):
    conn_xper, conn_photo = make_connections(day)
    task_ids = collect_task_ids(conn_xper, day, start_time, end_time)

    fields = base_fields(conn_xper, conn_photo)
    fields.append(SpikeTimesForChannelsField(intan_data_path=intan_day_path(day)))
    fields.append(EpochStartStopField(intan_data_path=intan_day_path(day)))
    return fields.to_data(task_ids)


# ── Public API ───────────────────────────────────────────────────────────────

def compile_data(day: date,
                 start_time: time = None,
                 end_time: time = None,
                 experiment_filename: str = None):
    """
    Compile live-thresholded neural recording data.

    If `experiment_filename` is provided → per-experiment mode (single file).
    Otherwise → per-trial mode (start_time & end_time required).
    """
    if experiment_filename is not None:
        data = _read_per_experiment(
            day, experiment_filename,
            start_time or FULL_DAY[0],
            end_time or FULL_DAY[1],
        )
    else:
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time are required when no experiment_filename is provided.")
        data = _read_per_trial(day, start_time, end_time)

    data = data[data["RawSpikeTimes"].notna()]
    print(data.to_string())
    return data


if __name__ == "__main__":
    compile_data(
        day=date(2023, 11, 7),
        experiment_filename="231107_round1",
    )