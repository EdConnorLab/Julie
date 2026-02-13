from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import pandas as pd

from data_access.cache_utils import ExplodedSpikeCacheManager, SortedSpikeCacheManager, ThresholdSpikeCacheManager


class SpikeSource(Protocol):
    name: str
    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]: ...


@dataclass(frozen=True)
class MixedManualSpikeSource:
    """
    Mixed: exploded cache that includes unsorted + manually sorted.
    Returns None when empty.
    """
    curated_channels_only: bool = False
    force_recompute: bool = False
    name: str = "mixed_manual"

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        df = ExplodedSpikeCacheManager().load_or_compute(
            date,
            round_no,
            curated_channels_only=self.curated_channels_only,
            force_recompute=self.force_recompute,
        )
        if df is None or getattr(df, "empty", True):
            return None
        return df

##
@dataclass(frozen=True)
class SISortedSpikeSource:
    """
    SpikeInterface-sorted cache.
    Returns None when no units / missing.
    """
    force_recompute: bool = False
    name: str = "si_sorted"

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        df = SortedSpikeCacheManager().load_or_compute(
            date,
            round_no,
            force_recompute=self.force_recompute,
        )
        if df is None or getattr(df, "empty", True):
            return None
        return df




@dataclass(frozen=True)
class ThresholdSpikeSource:
    """
    Unsorted spikes detected from raw amplifier.dat using the Quian Quiroga (2004)
    threshold method:  threshold = -multiplier * median(|signal|) / 0.6745

    Parameters
    ----------
    threshold_multiplier : float
        Sigma multiplier (default 4 → -4*sigma_noise).  Typical range: 4-5.
    force_recompute : bool
        Re-run detection even if a cache exists.
    """
    threshold_multiplier: float = 4.5
    force_recompute: bool = False
    name: str = "threshold"

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        df = ThresholdSpikeCacheManager().load_or_compute(
            date,
            round_no,
            threshold_multiplier=self.threshold_multiplier,
            force_recompute=self.force_recompute,
        )
        if df is None or getattr(df, "empty", True):
            return None
        return df