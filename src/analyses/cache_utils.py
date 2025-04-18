from pathlib import Path

import pandas as pd

from analyses.spike_count import load_and_combine_data, explode_spike_data

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
        label = f"{date}_round_{round_no}"
        path = self._get_cache_path(label)
        if path.exists() and not force_recompute:
            return pd.read_pickle(path)
        print(f"[Cache] Using file: {path}")
        combined_data = load_and_combine_data(date, round_no)
        exploded_df = explode_spike_data(combined_data, date, round_no, only_valid_channels)
        exploded_df.to_pickle(path)

        return exploded_df

class GLMResultCacheManager(GenericCacheManager):
    def __init__(self):
        cache_dir = Path(__file__).resolve().parents[2] / "Cortana" / "glm_cache"
        super().__init__(cache_dir)


if __name__ == "__main__":
    spike_cache = ExplodedSpikeCacheManager()
    spike_cache.load_or_compute("2023-09-26",3)





