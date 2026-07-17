# rsa_pkl_config.py
"""
RSA configs for running RSA on a hand-picked set of (NeuronID, window) pairs
loaded from a pickle file.

The pkl is expected to be a DataFrame with columns:
    NeuronID, WindowStart_ms, WindowEnd_ms, H-statistic, p-value
(e.g. the si_sorted_*_significant_windows_pKW_passed.pkl files)
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from analyses.population_analysis.rsa.core.rsa_config import RSAConfig, SocialRSAConfig


# ──────────────────────────────────────────────────────────
# Pkl-specific knobs (shared by both flavors via mixin pattern;
# but Python dataclasses don't compose cleanly, so we just
# declare the fields twice in the two subclasses below.)
# ──────────────────────────────────────────────────────────


@dataclass
class PklRSAConfig(RSAConfig):
    """
    RSAConfig + pkl-window options.

    Modes
    -----
    pkl_window_mode = 'per_neuron'
        Each NeuronID in the pkl uses its own (WindowStart_ms, WindowEnd_ms).
        cfg.window is ignored for the firing-rate computation, but
        cfg.window IS still used by upstream filters (visual-responsiveness,
        peak-latency) — set those to False if you don't want them.

    pkl_window_mode = 'common'
        Pkl is used purely as a NeuronID whitelist.  All selected neurons
        use cfg.window.  Equivalent to the existing neuron_id_filter_pkl
        path, but with the optional p-value filter on top.

    Filtering
    ---------
    pkl_filter_significant = True  →  keep rows with p-value <  pkl_p_threshold
    pkl_filter_significant = False →  keep every row in the pkl

    Duplicates
    ----------
    If a NeuronID appears in multiple rows (different windows), the row with
    the smallest p-value is kept.  Other rows are discarded.
    """

    # ── Pkl source ──
    pkl_path: Optional[str] = None
    pkl_neuron_id_col: str = 'NeuronID'
    pkl_window_start_col: str = 'WindowStart_ms'
    pkl_window_end_col: str = 'WindowEnd_ms'
    pkl_p_value_col: str = 'p-value'

    # ── How to use the pkl ──
    pkl_window_mode: str = 'per_neuron'      # 'per_neuron' | 'common'
    pkl_filter_significant: bool = False     # if True, keep only rows with p < pkl_p_threshold
    pkl_p_threshold: float = 0.05

    # Override: with per-neuron windows the standard cfg.window check
    # doesn't apply.  We instead check that every pkl window fits inside
    # min_epoch_duration at load time (see rsa_pkl_core.load_pkl_windows).
    def validate(self):
        if self.pkl_path is None:
            raise ValueError("pkl_path must be set on PklRSAConfig")
        if self.pkl_window_mode not in ('per_neuron', 'common'):
            raise ValueError(
                f"pkl_window_mode must be 'per_neuron' or 'common', "
                f"got {self.pkl_window_mode!r}")
        if self.pkl_window_mode == 'common':
            # Only validate cfg.window in common-window mode
            super().validate()


@dataclass
class PklSocialRSAConfig(SocialRSAConfig):
    """
    SocialRSAConfig + pkl-window options.  See PklRSAConfig for semantics.
    """

    pkl_path: Optional[str] = None
    pkl_neuron_id_col: str = 'NeuronID'
    pkl_window_start_col: str = 'WindowStart_ms'
    pkl_window_end_col: str = 'WindowEnd_ms'
    pkl_p_value_col: str = 'p-value'

    pkl_window_mode: str = 'per_neuron'
    pkl_filter_significant: bool = False
    pkl_p_threshold: float = 0.05

    def validate(self):
        if self.pkl_path is None:
            raise ValueError("pkl_path must be set on PklSocialRSAConfig")
        if self.pkl_window_mode not in ('per_neuron', 'common'):
            raise ValueError(
                f"pkl_window_mode must be 'per_neuron' or 'common', "
                f"got {self.pkl_window_mode!r}")
        if self.pkl_window_mode == 'common':
            super().validate()
        else:
            # In per-neuron mode, skip the cfg.window check but still
            # validate transform_social_behavior etc. from the parent.
            valid = {None, 'rank', 'log'}
            if self.transform_social_behavior not in valid:
                raise ValueError(
                    f"transform_social_behavior must be one of {valid}, "
                    f"got {self.transform_social_behavior!r}")
