from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol

import pandas as pd

from data_access.cache_utils import ExplodedSpikeCacheManager, SortedSpikeCacheManager
from data_access.data_loader import normalize_monkey_names
from data_access.spike_window import (
    EXPLODED_CACHE_SUBDIR, MUA_CACHE_SUBDIR, SORTED_CACHE_SUBDIR, STRICT,
    clip_spike_window,
)


class SpikeSource(Protocol):
    name: str
    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]: ...


def _finalize(df, cache_subdir, pre_stimulus_time):
    """The load boundary: normalise names, then clip to the requested window.

    Both steps belong here rather than in each analysis. Names because trial
    metadata compiled at different times spells seven monkeys differently and an
    inner merge on MonkeyName drops the mismatches silently; the window because
    the caches hold a 1000 ms baseline that would inflate every raw spike count
    and firing rate downstream unless it is clipped off by default.
    """
    if df is None or getattr(df, "empty", True):
        return None
    df = normalize_monkey_names(df)
    return clip_spike_window(df, pre_stimulus_time, cache_subdir=cache_subdir)


class _WindowMixin:
    def with_pre_stimulus(self, seconds: float):
        """A copy of this source that keeps ``seconds`` of pre-stimulus baseline.

        e.g. ``SISortedSpikeSource().with_pre_stimulus(1.0)`` for a raster or a
        response-window detector that needs a real baseline. Everything else gets
        the strict window and does not have to know the cache is wider.
        """
        return replace(self, pre_stimulus_time=float(seconds))


@dataclass(frozen=True)
class MixedManualSpikeSource(_WindowMixin):
    """
    Mixed: exploded cache that includes unsorted + manually sorted.
    Returns None when empty.

    NOTE on ``pre_stimulus_time``: only the manually-sorted units carry a real
    baseline here. Unsorted channels come from compiled.pkl, already clipped to
    [onset, offset], so their pre-window is empty by construction.
    """
    curated_channels_only: bool = False
    force_recompute: bool = False
    name: str = "mixed_manual"
    cache_subdir: str = EXPLODED_CACHE_SUBDIR
    pre_stimulus_time: float = STRICT

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        df = ExplodedSpikeCacheManager(cache_subdir=self.cache_subdir).load_or_compute(
            date,
            round_no,
            curated_channels_only=self.curated_channels_only,
            force_recompute=self.force_recompute,
        )
        return _finalize(df, self.cache_subdir, self.pre_stimulus_time)


##
@dataclass(frozen=True)
class SISortedSpikeSource(_WindowMixin):
    """
    SpikeInterface-sorted cache.
    Returns None when no units / missing.
    """
    force_recompute: bool = False
    name: str = "si_sorted"
    cache_subdir: str = SORTED_CACHE_SUBDIR
    pre_filtered: bool = False
    pre_stimulus_time: float = STRICT

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        try:
            df = SortedSpikeCacheManager(cache_subdir=self.cache_subdir).load_or_compute(
                date,
                round_no,
                force_recompute=self.force_recompute,
            )
        except FileNotFoundError:
            if self.pre_filtered:
                return None
            raise
        return _finalize(df, self.cache_subdir, self.pre_stimulus_time)


@dataclass(frozen=True)
class ThresholdMUASpikeSource(_WindowMixin):
    """Multi-unit activity (MUA) from OFFLINE MAD/RMS negative-crossing detection on
    the raw amplifier signal (threshold_detection.detect_mad_spikes).

    An alternative to online spike.dat MUA: online thresholds are hand-set during
    recording and drift as cells move / across 32 channels, so they can be
    inaccurate. Here the threshold is recomputed from the data per channel
    (median/RMS noise * multiplier), so it doesn't rely on the live settings.

    Replaces the old Quian-Quiroga ThresholdSpikeSource (now removed).
    pre_filtered=True so the single-unit ISI QC is skipped (this is multiunit).
    """
    noise_method: str = "mad"          # 'mad' = median(|v|)/0.6745, or 'rms'
    threshold_multiplier: float = 4.0
    refractory_ms: float = 1.0
    force_recompute: bool = False
    pre_filtered: bool = True
    name: str = "threshold_mua"
    cache_subdir: str = MUA_CACHE_SUBDIR
    pre_stimulus_time: float = STRICT

    def load(self, date: str, round_no: int) -> Optional[pd.DataFrame]:
        from data_access.cache_utils import ThresholdMUASpikeCacheManager
        df = ThresholdMUASpikeCacheManager(cache_subdir=self.cache_subdir).load_or_compute(
            date,
            round_no,
            noise_method=self.noise_method,
            threshold_multiplier=self.threshold_multiplier,
            refractory_ms=self.refractory_ms,
            force_recompute=self.force_recompute,
        )
        return _finalize(df, self.cache_subdir, self.pre_stimulus_time)
