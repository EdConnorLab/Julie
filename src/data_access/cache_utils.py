from pathlib import Path

import pandas as pd

from data_access.data_loader import load_and_combine_data, explode_spike_data
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from project_util import PROJECT_BASE_PATH, SUBJECT_MONKEY

PROJECT_ROOT = Path(PROJECT_BASE_PATH)

class GenericCacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, label, ext="pkl"):
        return self.cache_dir / f"{label}.{ext}"

    def invalidate(self, label, ext="pkl"):
        path = self._get_cache_path(label, ext)
        if path.exists():
            path.unlink()


class SortedSpikeCacheManager(GenericCacheManager):
    def __init__(self, monkey: str = SUBJECT_MONKEY, cache_subdir: str = "sorted_spike_cache"):
        cache_dir = PROJECT_ROOT / monkey / cache_subdir
        self.summary_dir = PROJECT_ROOT / monkey / "sorted_spike_summary"
        super().__init__(cache_dir)

    def _summary_path(self, date: str, round_no: int) -> Path:
        # matches: 231024_round1_sorting_summary.txt
        yymmdd = date.replace("-", "")[2:]
        return self.summary_dir / f"{yymmdd}_round{round_no}_sorting_summary.txt"

    def _summary_says_no_units(self, path: Path) -> bool:
        if not path.exists():
            return False
        text = path.read_text().lower()
        return "no units in agreement" in text

    def load_or_compute(self, date, round_no, force_recompute=False):
        label = f"{date}_round_{round_no}"
        pkl_path = self._get_cache_path(label)

        if pkl_path.exists() and not force_recompute:
            return pd.read_pickle(pkl_path)

        summary_path = self._summary_path(date, round_no)
        if self._summary_says_no_units(summary_path):
            return None  # valid session, nothing to analyze

        raise FileNotFoundError(f"No sorted cache found for {label}")

class ExplodedSpikeCacheManager(GenericCacheManager):
    def __init__(self, monkey: str = SUBJECT_MONKEY):
        cache_dir = PROJECT_ROOT / monkey / "exploded_spike_cache"
        super().__init__(cache_dir)

    def _filter_curated(self, df: pd.DataFrame, date: str, round_no: int) -> pd.DataFrame:
        """
        Filter to curated channels only (view-layer filter).
        Assumes df has BaseChannel (your explode_spike_data adds it).
        """
        if df is None or df.empty:
            return pd.DataFrame()

        reader = RecordingMetadataReader()
        curated_channels = reader.get_curated_channels(date, round_no)
        curated_channel_list = [str(ch) for ch in curated_channels]

        if not curated_channel_list:
            return pd.DataFrame()

        if "BaseChannel" not in df.columns:
            # fallback: try to derive from Channel
            if "Channel" in df.columns:
                base = df["Channel"].astype(str).str.split("_Unit", n=1).str[0]
                return df.loc[base.isin(curated_channel_list)].copy()
            return pd.DataFrame()

        return df.loc[df["BaseChannel"].isin(curated_channel_list)].copy()

    def load_or_compute(self, date, round_no, curated_channels_only=False, force_recompute=False):
        """
        Canonical cache = ALL channels.
        curated_channels_only only affects the returned df, never the cached contents.
        """
        label = f"{date}_round_{round_no}"
        path = self._get_cache_path(label)

        # 1) Load canonical cache if exists
        if path.exists() and not force_recompute:
            df_all = pd.read_pickle(path)
            return self._filter_curated(df_all, date, round_no) if curated_channels_only else df_all

        # 2) Compute ALL channels (ignore curated flag when writing)
        print(f"[Cache] Using file: {path}")
        combined_data = load_and_combine_data(date, round_no)

        # IMPORTANT: compute canonical ALL-channels explosion
        df_all = explode_spike_data(combined_data, date, round_no, curated_channels_only=False)
        df_all.to_pickle(path)

        # 3) Return filtered view if requested
        return self._filter_curated(df_all, date, round_no) if curated_channels_only else df_all

# TODO: This class needs to be double-checked -- not useful for Cortana's data due to high noise
class ThresholdSpikeCacheManager(GenericCacheManager):
    """
    Cache for threshold-detected spikes from raw amplifier.dat.
    Uses Quian Quiroga (2004) method: threshold = -multiplier * median(|signal|) / 0.6745
    """
    def __init__(self, monkey: str = SUBJECT_MONKEY):
        cache_dir = PROJECT_ROOT / monkey / "threshold_spike_cache"
        super().__init__(cache_dir)

    def load_or_compute(self, date, round_no, *,
                        threshold_multiplier=4.5,
                        force_recompute=False):
        label = f"{date}_round_{round_no}_thr{threshold_multiplier}"
        path = self._get_cache_path(label)

        if path.exists() and not force_recompute:
            return pd.read_pickle(path)

        print(f"[ThresholdCache] Computing threshold spikes for {date} round {round_no} ...")
        df = self._compute(date, round_no, threshold_multiplier)
        if df is not None and not df.empty:
            df.to_pickle(path)
        return df

    def _compute(self, date, round_no, threshold_multiplier):
        import os
        from clat.intan.rhd import load_intan_rhd_format
        from clat.intan.amplifiers import read_amplifier_data_with_mmap
        from data_access.threshold_detection import detect_spikes_for_recording

        reader = RecordingMetadataReader()
        pickle_filepath, _, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_no)

        # Load trial metadata from compiled.pkl
        raw_trials = pd.read_pickle(pickle_filepath)
        if raw_trials is None or raw_trials.empty:
            return None

        # Load amplifier data
        info_path = os.path.join(round_dir_path, "info.rhd")
        amp_path = os.path.join(round_dir_path, "amplifier.dat")
        preprocessed_path = os.path.join(round_dir_path, "preprocessed_data.dat")

        rhd = load_intan_rhd_format.read_data(info_path)
        sample_rate = rhd['frequency_parameters']['amplifier_sample_rate']
        amp_channels = rhd['amplifier_channels']

        # Prefer preprocessed if available (already highpass filtered)
        if os.path.exists(preprocessed_path):
            voltages = read_amplifier_data_with_mmap(preprocessed_path, amp_channels)
            apply_filter = False
        else:
            voltages = read_amplifier_data_with_mmap(amp_path, amp_channels)
            apply_filter = True

        spike_times_by_channel, info = detect_spikes_for_recording(
            voltages, sample_rate,
            threshold_multiplier=threshold_multiplier,
            apply_filter=apply_filter,
        )

        for ch, ch_info in info.items():
            print(f"  {ch}: {ch_info['n_spikes']} spikes (thr={ch_info['threshold']:.1f})")

        # Build per-trial DataFrame matching the compiled.pkl format
        rows = []
        for _, trial in raw_trials.iterrows():
            epoch_start, epoch_stop = trial['EpochStartStop']
            trial_spikes = {}
            for channel, all_times in spike_times_by_channel.items():
                trial_spikes[channel] = [
                    t for t in all_times if epoch_start <= t < epoch_stop
                ]

            rows.append({
                'TaskField': trial['TaskField'],
                'MonkeyId': trial['MonkeyId'],
                'MonkeyName': trial['MonkeyName'],
                'MonkeyGroup': trial['MonkeyGroup'],
                'SpikeTimes': trial_spikes,
                'EpochStartStop': trial['EpochStartStop'],
            })

        combined = pd.DataFrame(rows)
        return explode_spike_data(combined, date, round_no)


class ThresholdMUASpikeCacheManager(GenericCacheManager):
    """Cache for multi-unit activity (MUA) from OFFLINE MAD/RMS negative-crossing
    detection on amplifier.dat / preprocessed_data.dat. Mirrors
    ThresholdSpikeCacheManager but calls detect_mad_spikes_for_recording. Kept as a
    separate class so the existing threshold pipeline is untouched."""
    def __init__(self, monkey: str = SUBJECT_MONKEY):
        cache_dir = PROJECT_ROOT / monkey / "threshold_mua_spike_cache"
        super().__init__(cache_dir)

    def load_or_compute(self, date, round_no, *, noise_method='mad',
                        threshold_multiplier=4.0, refractory_ms=1.0,
                        force_recompute=False):
        label = f"{date}_round_{round_no}_{noise_method}{threshold_multiplier}_ref{refractory_ms}"
        path = self._get_cache_path(label)
        if path.exists() and not force_recompute:
            return pd.read_pickle(path)
        print(f"[ThresholdMUACache] Computing MUA spikes for {date} round {round_no} ...")
        df = self._compute(date, round_no, noise_method, threshold_multiplier, refractory_ms)
        if df is not None and not df.empty:
            df.to_pickle(path)
        return df

    def _compute(self, date, round_no, noise_method, threshold_multiplier, refractory_ms):
        import os
        from clat.intan.rhd import load_intan_rhd_format
        from clat.intan.amplifiers import read_amplifier_data_with_mmap
        from data_access.threshold_detection import detect_mad_spikes_for_recording

        reader = RecordingMetadataReader()
        pickle_filepath, _, round_dir_path = reader.get_metadata_for_spike_analysis(date, round_no)

        raw_trials = pd.read_pickle(pickle_filepath)
        if raw_trials is None or raw_trials.empty:
            return None

        info_path = os.path.join(round_dir_path, "info.rhd")
        amp_path = os.path.join(round_dir_path, "amplifier.dat")
        preprocessed_path = os.path.join(round_dir_path, "preprocessed_data.dat")

        rhd = load_intan_rhd_format.read_data(info_path)
        sample_rate = rhd['frequency_parameters']['amplifier_sample_rate']
        amp_channels = rhd['amplifier_channels']

        # Prefer preprocessed (already highpass filtered) as the existing manager does.
        if os.path.exists(preprocessed_path):
            voltages = read_amplifier_data_with_mmap(preprocessed_path, amp_channels)
            apply_filter = False
        else:
            voltages = read_amplifier_data_with_mmap(amp_path, amp_channels)
            apply_filter = True

        spike_times_by_channel, info = detect_mad_spikes_for_recording(
            voltages, sample_rate,
            noise_method=noise_method,
            threshold_multiplier=threshold_multiplier,
            refractory_ms=refractory_ms,
            apply_filter=apply_filter,
        )
        for ch, ch_info in info.items():
            print(f"  {ch}: {ch_info['n_spikes']} spikes (thr={ch_info['threshold']:.1f})")

        rows = []
        for _, trial in raw_trials.iterrows():
            epoch_start, epoch_stop = trial['EpochStartStop']
            trial_spikes = {}
            for channel, all_times in spike_times_by_channel.items():
                trial_spikes[channel] = [t for t in all_times if epoch_start <= t < epoch_stop]
            rows.append({
                'TaskField': trial['TaskField'],
                'MonkeyId': trial['MonkeyId'],
                'MonkeyName': trial['MonkeyName'],
                'MonkeyGroup': trial['MonkeyGroup'],
                'SpikeTimes': trial_spikes,
                'EpochStartStop': trial['EpochStartStop'],
            })
        combined = pd.DataFrame(rows)
        return explode_spike_data(combined, date, round_no)


class BehaviorMatrixCacheManager(GenericCacheManager):
    def __init__(self, monkey: str = SUBJECT_MONKEY):
        cache_dir = PROJECT_ROOT / monkey / "behavior_matrix_cache"
        super().__init__(cache_dir)

    def load_or_cache(self, xlsx_path, label, transpose=False, drop_first_col=True, force_recompute=False, ext='pkl'):
        '''Load or cache behavior matrix from Excel, saving under label name.'''

        if label is None:
            label = Path(xlsx_path).stem

        path = self._get_cache_path(label, ext)

        if path.exists() and not force_recompute:
            return pd.read_pickle(path)

        project_root = Path(__file__).resolve().parents[2]  # /home/.../Julie
        full_excel_path = project_root / "social_data" / xlsx_path
        full_excel_path = full_excel_path.resolve()

        df = pd.read_excel(full_excel_path)

        if drop_first_col:
            df.set_index(df.columns[0], inplace=True)

        if transpose:
            df = df.T

        df.to_pickle(path)
        return df

if __name__ == "__main__":
    bm_cache = BehaviorMatrixCacheManager()
    # ⬇️ Submission example with custom label
    submission_df = bm_cache.load_or_cache(
        xlsx_path="zombies_social_data/zombies_feature_df_affiliation.xlsx",
        label="submission_",
        transpose=True
    )






