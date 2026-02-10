import os
from datetime import datetime, time, date

import pytz
from clat.compile.task.cached_task_fields import CachedTaskFieldList
from clat.compile.task.classic_database_task_fields import TaskIdField
from clat.compile.task.compile_task_id import PngSlideIdCollector
from clat.util import time_util
from clat.util.connection import Connection

from src.compile.julie_database_fields import FileNameField, MonkeyIdField, MonkeyNameField, MonkeyGroupField
from compile.julie_intan_file_per_experiment_fields import SpikeTimesForChannelsField_Experiment, EpochStartStopField_Experiment
from compile.julie_intan_file_per_trial_fields import SpikeTimesForChannelsField, EpochStartStopField
from compile.julie_one_file_spike_parsing import OneFileParser

# ── Constants ────────────────────────────────────────────────────────────────

TIMEZONE = pytz.timezone("US/Eastern")
INTAN_BASE_PATH = "/home/connorlab/Documents/IntanData/Cortana"
DB_HOST = "172.30.6.59"
FULL_DAY = (time(0, 0, 0), time(23, 59, 59))

# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_unix_range(day: date, start_time: time, end_time: time) -> tuple[float, float]:
    """Convert a day + start/end times to a (start_unix, end_unix) tuple."""
    start = TIMEZONE.localize(datetime.combine(day, start_time))
    end = TIMEZONE.localize(datetime.combine(day, end_time))
    return time_util.to_unix(start), time_util.to_unix(end)


def _make_connections(day: date):
    """Create the xper recording and photo_metadata DB connections for a given day."""
    date_no_hyphens = day.strftime("%Y%m%d")
    conn_xper = Connection(f"{date_no_hyphens}_recording", host=DB_HOST)
    conn_photo = Connection("photo_metadata", host=DB_HOST)
    return conn_xper, conn_photo


def _collect_task_ids(conn_xper, day: date, start_time: time, end_time: time):
    """Collect completed task IDs within the given time window."""
    time_range = _to_unix_range(day, start_time, end_time)
    return PngSlideIdCollector(conn_xper).collect_complete_task_ids(time_range)


def _base_fields(conn_xper, conn_photo) -> CachedTaskFieldList:
    """Return the common metadata fields shared by both read paths."""
    fields = CachedTaskFieldList()
    fields.append(TaskIdField(conn_xper))
    fields.append(FileNameField(conn_xper=conn_xper))
    fields.append(MonkeyIdField(conn_xper=conn_xper, conn_photo=conn_photo))
    fields.append(MonkeyNameField(conn_xper=conn_xper, conn_photo=conn_photo))
    fields.append(MonkeyGroupField(conn_xper=conn_xper, conn_photo=conn_photo))
    return fields


# ── Read functions ───────────────────────────────────────────────────────────

def read_per_experiment(day: date, experiment_name: str,
                        start_time: time = FULL_DAY[0],
                        end_time: time = FULL_DAY[1]):
    """Compile data from a single Intan file generated per experiment."""
    conn_xper, conn_photo = _make_connections(day)
    task_ids = _collect_task_ids(conn_xper, day, start_time, end_time)

    intan_file_path = os.path.join(INTAN_BASE_PATH, day.strftime("%Y-%m-%d"), experiment_name)
    parser = OneFileParser()
    (unfiltered_spikes, filtered_spikes,
     epoch_start_stop, sample_rate) = parser.parse_with_peristimulus_spikes(intan_file_path)

    fields = _base_fields(conn_xper, conn_photo)
    fields.append(SpikeTimesForChannelsField_Experiment(conn_xper, filtered_spikes))
    fields.append(EpochStartStopField_Experiment(conn_xper, epoch_start_stop))
    # fields.append(PeriStimulusSpikeTimesForChannelsField_Experiment(conn_xper, unfiltered_spike_tstamps_for_channels_by_task_id))

    return fields.to_data(task_ids)


def read_per_trial(day: date, start_time: time, end_time: time):
    """Compile data from individual Intan files generated per trial."""
    conn_xper, conn_photo = _make_connections(day)
    task_ids = _collect_task_ids(conn_xper, day, start_time, end_time)

    intan_data_path = os.path.join(INTAN_BASE_PATH, day.strftime("%Y-%m-%d"))

    fields = _base_fields(conn_xper, conn_photo)
    fields.append(SpikeTimesForChannelsField(intan_data_path=intan_data_path))
    fields.append(EpochStartStopField(intan_data_path=intan_data_path))

    return fields.to_data(task_ids)


# ── Main entry point ────────────────────────────────────────────────────────

def compile_data(day: date,
                 start_time: time = None,
                 end_time: time = None,
                 experiment_filename: str = None):
    """
    Compile neural recording data for a given day.

    If `experiment_filename` is provided, reads from a single file per experiment
    (start_time / end_time are optional, defaulting to the full day).

    Otherwise, reads from per-trial files and `start_time` / `end_time` are required.
    """
    if experiment_filename is not None:
        data = read_per_experiment(day, experiment_filename, start_time or FULL_DAY[0], end_time or FULL_DAY[1])
        filename = f"{experiment_filename}.pkl"
    else:
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time are required when no experiment_filename is provided.")
        data = read_per_trial(day, start_time, end_time)
        filename = f"{day:%Y-%m-%d}_{start_time:%H-%M-%S}_to_{end_time:%H-%M-%S}.pkl"

    data = data[data["RawSpikeTimes"].notna()]
    print(data.to_string())
    return data


if __name__ == "__main__":
    compile_data(
        day=date(2023, 11, 7),
        experiment_filename="231107_round1",
    )