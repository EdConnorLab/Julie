
import os
from datetime import datetime, time, date

import pytz
from clat.compile.task.cached_task_fields import CachedTaskFieldList
from clat.compile.task.classic_database_task_fields import TaskIdField
from clat.compile.task.compile_task_id import PngSlideIdCollector
from clat.util import time_util
from clat.util.connection import Connection

from project_util import SUBJECT_MONKEY
from compile.julie_database_fields import (
    FileNameField, MonkeyIdField, MonkeyNameField, MonkeyGroupField,
)


TIMEZONE = pytz.timezone("US/Eastern")
INTAN_BASE_PATH = "/home/connorlab/Documents/IntanData"
DB_HOST = "172.30.6.59"
FULL_DAY = (time(0, 0, 0), time(23, 59, 59))


# ── Connection / path helpers ────────────────────────────────────────────────

def make_connections(day: date):
    """Create the xper-recording and photo_metadata DB connections."""
    date_key = day.strftime("%Y%m%d")
    conn_xper = Connection(f"{date_key}_recording", host=DB_HOST)
    conn_photo = Connection("photo_metadata", host=DB_HOST)
    return conn_xper, conn_photo


def intan_day_path(day: date, monkey: str = SUBJECT_MONKEY) -> str:
    return os.path.join(INTAN_BASE_PATH, monkey, day.strftime("%Y-%m-%d"))


def intan_experiment_path(day: date, experiment_name: str, monkey: str = SUBJECT_MONKEY) -> str:
    return os.path.join(intan_day_path(day, monkey), experiment_name)


# ── Time utilities

def to_unix_range(day: date, start_time: time, end_time: time) -> tuple[float, float]:
    """Convert day + start/end wall-clock times to a (start_unix, end_unix) pair."""
    start = TIMEZONE.localize(datetime.combine(day, start_time))
    end = TIMEZONE.localize(datetime.combine(day, end_time))
    return time_util.to_unix(start), time_util.to_unix(end)


# ── Task ID collection

def collect_task_ids(conn_xper, day: date,
                     start_time: time = FULL_DAY[0],
                     end_time: time = FULL_DAY[1]):
    """Return completed task IDs within the given time window."""
    time_range = to_unix_range(day, start_time, end_time)
    return PngSlideIdCollector(conn_xper).collect_complete_task_ids(time_range)


### Common task field list

def base_fields(conn_xper, conn_photo) -> CachedTaskFieldList:
    """Metadata fields shared across all compilation pipelines."""
    fields = CachedTaskFieldList()
    fields.append(TaskIdField(conn_xper))
    fields.append(FileNameField(conn_xper=conn_xper))
    fields.append(MonkeyIdField(conn_xper=conn_xper, conn_photo=conn_photo))
    fields.append(MonkeyNameField(conn_xper=conn_xper, conn_photo=conn_photo))
    fields.append(MonkeyGroupField(conn_xper=conn_xper, conn_photo=conn_photo))
    return fields