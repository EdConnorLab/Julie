from pathlib import Path

import pandas as pd

from analyses.data_loader import load_and_combine_data, explode_spike_data

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

class ExplodedSpikeCacheManager(GenericCacheManager):
    def __init__(self):
        cache_dir = Path(__file__).resolve().parents[2] / "Cortana" / "exploded_spike_cache"
        super().__init__(cache_dir)

    def load_or_compute(self, date, round_no, only_valid_channels=False, force_recompute=False):
        label = f"{date}_round_{round_no}{'_validOnly' if only_valid_channels else ''}"
        path = self._get_cache_path(label)
        if path.exists() and not force_recompute:
            return pd.read_pickle(path)
        print(f"[Cache] Using file: {path}")
        combined_data = load_and_combine_data(date, round_no)
        exploded_df = explode_spike_data(combined_data, date, round_no, only_valid_channels)
        exploded_df.to_pickle(path)

        return exploded_df


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
        full_excel_path = project_root / "social_data" / xlsx_path  # 상대경로 붙이기
        full_excel_path = full_excel_path.resolve()  # 절대경로로 변환

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






