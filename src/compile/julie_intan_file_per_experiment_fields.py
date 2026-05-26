from clat.compile.task.cached_task_fields import CachedTaskDatabaseField
from clat.compile.task.task_field import TaskField


class SpikeTimesForChannelsField_Experiment(CachedTaskDatabaseField):
    def __init__(self, conn, spike_times_for_channels_by_task_id):
        super().__init__(conn)
        self.spike_times_for_channels_by_task_id = spike_times_for_channels_by_task_id

    def get(self, task_id: int):
        if task_id not in self.spike_times_for_channels_by_task_id:
            return None
        return self.spike_times_for_channels_by_task_id[task_id]

    def get_name(self):
        return "RawSpikeTimes"


class EpochStartStopField_Experiment(CachedTaskDatabaseField):
    def __init__(self, conn, epoch_start_stop_by_task_id):
        super().__init__(conn)
        self.epoch_start_stop_by_task_id = epoch_start_stop_by_task_id

    def get(self, task_id: int):
        if task_id not in self.epoch_start_stop_by_task_id:
            return None
        return self.epoch_start_stop_by_task_id[task_id]

    def get_name(self):
        return "EpochStartStop"


class PeriStimulusSpikeTimesForChannelsField_Experiment(CachedTaskDatabaseField):
    def __init__(self, conn, peristimulus_spike_times_for_channels_by_task_id):
        self.peristimulus_spike_times_for_channels_by_task_id = peristimulus_spike_times_for_channels_by_task_id
        super().__init__(conn)

    def get(self, task_id: int):
        if task_id not in self.peristimulus_spike_times_for_channels_by_task_id:
            return None
        return self.peristimulus_spike_times_for_channels_by_task_id[task_id]

    def get_name(self):
        return "PeristimulusSpikeTimes"