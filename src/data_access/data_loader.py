import os
import warnings
import pickle
import pandas as pd
from clat.intan.spike_file import fetch_spike_tstamps_from_file
from clat.intan.rhd import load_intan_rhd_format
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from project_util import canonical_monkey_name


def get_raw_spike_tstamp_data(date, round_number):
    reader = RecordingMetadataReader()
    _, curated_channels, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_number)
    spike_path = os.path.join(round_dir_path, "spike.dat")
    spike_tstamps_for_channels, sample_rate = fetch_spike_tstamps_from_file(spike_path)
    return spike_tstamps_for_channels, sample_rate


def normalize_trial_columns(df, source_path=""):
    """Rename the task-id column to 'TaskField' if it arrived as 'TaskId'.

    compiled.pkl column names track the installed clat version: newer
    TaskIdField.get_name() returns "TaskId" where older ones returned
    "TaskField". Downstream (combine_unsorted_with_sorted) keys trials on
    'TaskField', so a pickle compiled under a newer clat fails with
    KeyError: 'TaskField'. Normalize here, at the single point where compiled
    trial data enters the pipeline, so both vintages load the same way.
    """
    if "TaskField" in df.columns:
        return df
    if "TaskId" in df.columns:
        return df.rename(columns={"TaskId": "TaskField"})
    raise KeyError(
        f"Compiled trial data{' at ' + str(source_path) if source_path else ''} has "
        f"neither 'TaskField' nor 'TaskId'; got {list(df.columns)}"
    )


def normalize_monkey_names(df, source_path=""):
    """Canonicalise MonkeyName so one monkey is never two.

    The recording database now returns the trailing letter of seven stimulus
    monkeys in lower case ('114j' where monkeyinfo.csv and every social workbook
    say '114J'). Trial metadata compiled before and after that change therefore
    disagrees, and because the exploded cache draws its unsorted rows from one
    compiled.pkl and its manually-sorted rows from another, a single session can
    carry both spellings. Downstream, ``linear_regression`` inner-merges on
    MonkeyName and Ed's scripts test ``== '87J'`` -- both drop the mismatches
    without a word.

    Applied wherever compiled trial data enters the pipeline, and again at the
    SpikeSource load boundary so caches already written with the lower-case
    spelling still line up.
    """
    if df is None or getattr(df, "empty", True) or "MonkeyName" not in df.columns:
        return df
    canonical = df["MonkeyName"].map(canonical_monkey_name)
    if canonical.equals(df["MonkeyName"]):
        return df
    out = df.copy()
    out["MonkeyName"] = canonical
    return out


def load_manually_sorted_spikes(path):
    """Load a dict of manually sorted spike indices from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def calculate_spike_timestamps(df, spike_indices_by_unit_by_channel, sample_rate,
                               pre_stimulus_time=0.0):
    """
    Add a 'SpikeTimes' column: dict mapping unit names to spike-time lists
    filtered to each trial's epoch.

    ``pre_stimulus_time`` (seconds) widens the lower bound to
    ``epoch_start - pre_stimulus_time`` (clamped to recording start) so a
    pre-stimulus baseline is retained; these manual-sort spike indices are stored
    un-windowed in sorted_spikes.pkl, so the baseline is genuinely recoverable.
    pre_stimulus_time=0.0 reproduces the original strict window.
    """
    def _for_row(epoch_start_stop):
        epoch_start, epoch_stop = epoch_start_stop
        lower_bound = max(epoch_start - pre_stimulus_time, 0.0)  # clamp to recording start
        result = {}
        for channel, units in reversed(spike_indices_by_unit_by_channel.items()):
            for unit_name, spike_indices in units.items():
                key = f"{channel}_{unit_name}"
                result[key] = [
                    idx / sample_rate
                    for idx in spike_indices
                    if lower_bound <= idx / sample_rate < epoch_stop
                ]
        return result

    out = df.copy(deep=True)
    out["SpikeTimes"] = out["EpochStartStop"].apply(_for_row)
    return out



def read_sorted_data(round_path,
                     manually_sorted_spikes_filename="sorted_spikes.pkl",
                     compiled_trials_filename="compiled.pkl",
                     pre_stimulus_time=0.0):
    """Load compiled trials and attach per-unit spike timestamps."""
    compiled_path = os.path.join(round_path, compiled_trials_filename)
    raw = pd.read_pickle(compiled_path).reset_index(drop=True)
    raw = normalize_monkey_names(normalize_trial_columns(raw, compiled_path), compiled_path)
    sorted_spikes = load_manually_sorted_spikes(os.path.join(round_path, manually_sorted_spikes_filename))

    rhd_path = os.path.join(round_path, "info.rhd")
    sample_rate = load_intan_rhd_format.read_data(rhd_path)["frequency_parameters"]["amplifier_sample_rate"]

    return calculate_spike_timestamps(raw, sorted_spikes, sample_rate,
                                      pre_stimulus_time=pre_stimulus_time)



def get_raw_spike_tstamp_data(date, round_number):
    reader = RecordingMetadataReader()
    _, curated_channels, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_number)
    spike_path = os.path.join(round_dir_path, "spike.dat")
    spike_tstamps_for_channels, sample_rate = fetch_spike_tstamps_from_file(spike_path)
    return spike_tstamps_for_channels, sample_rate


def load_raw_data(date, round_number, pre_stimulus_time=0.0):
    reader = RecordingMetadataReader()
    pickle_filepath, curated_channels, round_path = reader.get_metadata_for_spike_analysis(date, round_number)
    raw_trial_data = normalize_monkey_names(
        normalize_trial_columns(pd.read_pickle(pickle_filepath), pickle_filepath),
        pickle_filepath)

    sorted_file = round_path / 'sorted_spikes.pkl'
    if sorted_file.exists():
        sorted_data = read_sorted_data(round_path, pre_stimulus_time=pre_stimulus_time)
    else:
        sorted_data = None

    return raw_trial_data, curated_channels, sorted_data


def _assert_same_session(raw_unsorted_data, sorted_data):
    """Refuse to merge two trial tables that describe different recordings.

    The unsorted stream comes from ``<data>/<monkey>/compiled/<PickleFileName>.pkl``
    and the manually-sorted stream from ``<intan>/<date>/<round>/compiled.pkl`` --
    two separate files that are only assumed to describe the same round. When they
    do not, the merge below has no matching keys and degenerates into a
    concatenation, producing a cache that looks fine and is not: 2023-10-27 round 4
    ended up holding round 3's 370 trials beside its own 323, with the manually
    sorted unit's spikes windowed against the wrong stimuli.

    Disjoint task-id sets mean the wrong compiled.pkl is in one of the two places.
    That is a data-placement error to fix, not something to average over.
    """
    sorted_ids = set(sorted_data['TaskField'])
    unsorted_ids = set(raw_unsorted_data['TaskField'])
    shared = sorted_ids & unsorted_ids
    if not shared:
        raise ValueError(
            "the manually-sorted and unsorted trial tables share no task ids "
            f"({len(sorted_ids)} sorted vs {len(unsorted_ids)} unsorted). They are "
            "not the same round -- check which compiled.pkl is sitting in the Intan "
            "round folder before rebuilding this session.")
    if shared != sorted_ids or shared != unsorted_ids:
        warnings.warn(
            f"trial tables only partly overlap: {len(shared)} shared task id(s), "
            f"{len(sorted_ids - unsorted_ids)} sorted-only, "
            f"{len(unsorted_ids - sorted_ids)} unsorted-only. Trials missing from "
            f"one stream will carry only the other stream's channels.")

    # Same trial, two different epochs -> the streams disagree on trial timing and
    # the trial will appear twice in the output, once per stream.
    def epochs_by_task(df):
        out = {}
        for t, e in zip(df['TaskField'], df['EpochStartStop']):
            out.setdefault(t, set()).add(tuple(e))
        return out

    sorted_epochs, unsorted_epochs = epochs_by_task(sorted_data), epochs_by_task(raw_unsorted_data)
    clashes = [t for t in shared if not (sorted_epochs[t] & unsorted_epochs[t])]
    if clashes:
        warnings.warn(
            f"{len(clashes)} task id(s) carry different epochs in the two trial "
            f"tables, e.g. {clashes[:3]}; those trials will be split across two rows.")


def combine_unsorted_with_sorted(raw_unsorted_data, sorted_data):
    if sorted_data is None or sorted_data.empty:
        return raw_unsorted_data

    _assert_same_session(raw_unsorted_data, sorted_data)

    # Step 1: Collect base channel names from sorted_data
    sorted_base_channels = set()
    for spike_dict in sorted_data['SpikeTimes']:
        for ch in spike_dict.keys():
            base_channel = ch.rsplit('_Unit', 1)[0]  # Remove unit info to get base channel
            sorted_base_channels.add(base_channel)

    # Trials are keyed on (task id, epoch) ONLY. MonkeyId/Name/Group used to be part
    # of the key, which meant a metadata difference between the two compiled.pkl
    # vintages -- MonkeyId as int here and str there, '114J' here and '114j' there --
    # split one trial into two rows, one carrying the sorted units and one the
    # unsorted channels. Stimulus identity is a function of the task id, so it can
    # never legitimately disambiguate two trials; it only ever caused false splits.
    def trial_key(row):
        return (row['TaskField'], row['EpochStartStop'])

    meta_by_key = {}

    # Step 2: Prepare unsorted spikes, excluding channels already present in sorted_data
    unsorted_spike_dicts = {}
    for _, row in raw_unsorted_data.iterrows():
        key = trial_key(row)
        meta_by_key.setdefault(key, (row['MonkeyId'], row['MonkeyName'], row['MonkeyGroup']))
        spike_times = unsorted_spike_dicts.setdefault(key, {})

        for ch, spikes in row['SpikeTimes'].items():
            ch_str = str(ch)  # Convert enum to string
            if ch_str not in sorted_base_channels:
                spike_times[ch] = spikes

    # Step 3: Prepare sorted spikes, using the same keys
    combined_spike_dicts = {}
    for _, row in sorted_data.iterrows():
        key = trial_key(row)
        meta_by_key.setdefault(key, (row['MonkeyId'], row['MonkeyName'], row['MonkeyGroup']))
        combined_spike_dicts[key] = row['SpikeTimes'].copy()

    # Step 4: Merge unsorted spikes into sorted spikes
    for key, unsorted_spikes in unsorted_spike_dicts.items():
        if key in combined_spike_dicts:
            combined_spike_dicts[key].update(unsorted_spikes)
        else:
            combined_spike_dicts[key] = unsorted_spikes

    # Step 5: Reconstruct final DataFrame
    combined_rows = []
    for (taskfield, epoch), spike_times in combined_spike_dicts.items():
        monkeyid, monkeyname, monkeygroup = meta_by_key[(taskfield, epoch)]
        combined_rows.append({
            'TaskField': taskfield,
            'MonkeyId': monkeyid,
            'MonkeyName': monkeyname,
            'MonkeyGroup': monkeygroup,
            'SpikeTimes': spike_times,
            'EpochStartStop': epoch
        })

    combined_df = pd.DataFrame(combined_rows)
    return combined_df


def load_and_combine_data(date, round_no, pre_stimulus_time=0.0):
    """Load and combine raw unsorted and sorted spike data.

    ``pre_stimulus_time`` widens only the manually-sorted stream (from
    sorted_spikes.pkl, which holds un-windowed indices). Unsorted spikes come from
    the already-clipped compiled.pkl and keep the strict [onset, offset] window.
    """
    raw_unsorted_data, _, sorted_data = load_raw_data(date, round_no, pre_stimulus_time=pre_stimulus_time)
    combined_data = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    return combined_data


# --- Data explosion (wide → long format) ---
def explode_spike_data(combined_data, date, round_no, curated_channels_only=False):
    """Explode spike times into long-format DataFrame with metadata."""
    reader = RecordingMetadataReader()
    rows = []
    for _, row in combined_data.iterrows():
        for channel_enum, spike_list in row['SpikeTimes'].items():
            rows.append({
                'TaskField': row['TaskField'],
                'MonkeyId': row['MonkeyId'],
                'MonkeyGroup': row['MonkeyGroup'],
                'MonkeyName': row['MonkeyName'],
                'Channel': channel_enum,
                'SpikeTimes': spike_list,
                'EpochStartStop': row['EpochStartStop']
            })
    exploded_df = pd.DataFrame(rows)

    sample_channel = str(exploded_df['Channel'].iloc[0])
    has_unit = "_Unit" in sample_channel

    def normalize_channel(ch):
        ch_str = str(ch) if not hasattr(ch, 'value') else str(ch)
        return ch_str.split("_Unit")[0] if has_unit else ch_str

    metadata = reader.get_metadata_for_preliminary_analysis()

    exploded_df['BaseChannel'] = exploded_df['Channel'].apply(normalize_channel)
    exploded_df['Date'] = date
    exploded_df['Round No.'] = round_no
    # add locations!
    metadata['Date'] = metadata['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    exploded_df = exploded_df.merge(
        metadata[['Date', 'Round No.', 'Location']],
        on=['Date', 'Round No.'],
        how='left'
    )
    exploded_df['Location'] = exploded_df['Location'].fillna('Unknown')

    exploded_df['NeuronID'] = (
            exploded_df['Location'].astype(str) + "_" +
            exploded_df['Date'].astype(str) + "_" +
            exploded_df['Round No.'].astype(str) + "_" +
            exploded_df['Channel'].astype(str)
    )
    if curated_channels_only:
        warnings.warn(
            "curated_channels_only in explode_spike_data is deprecated; "
            "filtering is now handled at load time",
            DeprecationWarning,
        )

    return exploded_df


if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1
    raw_unsorted_data, valid_channels, sorted_data = load_raw_data(date, round_no)
    print(raw_unsorted_data)
    print(sorted_data)
    combined = combine_unsorted_with_sorted(raw_unsorted_data, sorted_data)
    print(combined)
