from pathlib import Path

import pandas as pd

from data_access.data_loader import load_and_combine_data, explode_spike_data
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader


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
    def __init__(self):
        cache_dir = Path(__file__).resolve().parents[2] / "Cortana" / "sorted_spike_cache"
        self.summary_dir = Path(__file__).resolve().parents[2] / "Cortana" / "sorted_spike_summary"
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
    def __init__(self):
        cache_dir = Path(__file__).resolve().parents[2] / "Cortana" / "exploded_spike_cache"
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

class BehaviorMatrixCacheManager(GenericCacheManager):
    def __init__(self):
        cache_dir = Path(__file__).resolve().parents[2] / "Cortana" / "behavior_matrix_cache"
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






