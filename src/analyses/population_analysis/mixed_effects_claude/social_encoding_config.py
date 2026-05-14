# social_encoding_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class SocialEncodingConfig:
    # ── Data ────────────────────────────────────────────────────────────
    data_path: str = '/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered'
    monkey_info_path: str = '/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv'
    region: str = 'ALL'              # 'AMG', 'ER', or 'ALL'
    session: Optional[str] = None    # None = pool across sessions (pseudo-pop)

    # ── Firing-rate window (seconds relative to epoch start) ────────────
    window: Tuple[float, float] = (0.200, 0.600)
    min_epoch_duration: float = 2.0

    # ── Trial filtering ─────────────────────────────────────────────────
    min_reps_per_monkey: int = 3
    exclude_groups: List[str] = field(default_factory=list)

    # ── Subject ─────────────────────────────────────────────────────────
    subject_name: str = '81G'

    # ── Groups to compare ──────────────────────────────────────────────
    groups: List[str] = field(default_factory=lambda: ['Zombies', 'Instigators'])

    # ── Social-feature settings ─────────────────────────────────────────
    behavior_types: List[str] = field(
        default_factory=lambda: ['affiliation', 'agonism', 'submission']
    )
    feature_mode: str = 'full_profile'   # 'full_profile' | 'to_subject' | 'summary'
    n_feature_pcs: int = 2               # PCA components on behavioral profile

    # ── Pseudo-population ───────────────────────────────────────────────
    pseudo_population: bool = True       # pool neurons across sessions

    # ── Normalization (applied per neuron before analysis) ──────────────
    normalization: str = None            # None | 'soft' | 'zscore'
    soft_normalize_const: float = 5.0

    # ── Stats ───────────────────────────────────────────────────────────
    n_permutations: int = 5000
    rng_seed: int = 42

    # ── Plotting / saving ──────────────────────────────────────────────
    save_plots: bool = True
    save_dir: str = 'social_encoding_results'

    group_colors: dict = field(default_factory=lambda: {
        'Zombies':         '#9467bd',
        'Best Frans':      '#d62728',
        'Instigators':     '#2ca02c',
        'Stranger Things': '#1f77b4',
    })

    def validate(self):
        if self.window[0] >= self.window[1]:
            raise ValueError(f"window start ({self.window[0]}) must be < end ({self.window[1]})")
        if self.n_feature_pcs < 1:
            raise ValueError("n_feature_pcs must be >= 1")
        valid_modes = ('full_profile', 'to_subject', 'from_subject', 'summary')
        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode must be one of {valid_modes}")
