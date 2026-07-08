# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class TrajectoryConfig:
    # Data
    base_dir: Path = Path(__file__).resolve().parents[4]  # go up to repo root (Julie)
    data_path: Path = base_dir / "Cortana" / "sorted_spike_cache_filtered"
    region: str = 'ALL'              # 'AMG', 'ER', or 'ALL'
    session: Optional[str] = None    # None = multi-session; else e.g. 'ER_2023-11-28_4'

    # Binning
    bin_width: float = 0.050
    min_epoch_duration: float = 2.2

    # Averaging
    trial_averaged: bool = True      # must be True when session is None
    min_reps_per_monkey: int = 3

    # Peak alignment (trial-averaged only)
    peak_align: bool = False
    peak_fraction: float = 0.3
    drop_onset_outliers: bool = True
    outlier_percentile: float = 95

    # By-condition analysis (used by run_trajectory_by_condition.py)
    analysis: Optional[str] = None   # 'identity' | 'group' | 'familiarity' | 'sex' | 'rank'

    # PCA
    n_components: int = 3

    # Plotting
    smoothing_sigma: float = 2.0
    group_colors: dict = field(default_factory=lambda: {
        'Stranger Things': '#1f77b4',
        'Best Frans':      '#d62728',
        'Instigators':     '#2ca02c',
        'Zombies':         '#9467bd',
    })
    group_cmaps: dict = field(default_factory=lambda: {
        'Stranger Things': 'Blues',
        'Best Frans':      'Reds',
        'Instigators':     'Greens',
        'Zombies':         'Purples',
    })

    def validate(self):
        if self.session is None and not self.trial_averaged:
            raise ValueError("Multi-session mode requires trial_averaged=True")
        if self.peak_align and not self.trial_averaged:
            raise ValueError("peak_align requires trial_averaged=True")
        # Non-averaged PCA only makes sense when every condition has a comparable trial budget.
        # which is true for 'identity' (each stimulus monkey has the same number of reps)
        # but not for group analyses where groups have different numbers of member monkeys.
        if self.analysis is not None and self.analysis != 'identity' \
                and not self.trial_averaged:
            raise ValueError(
                f"analysis='{self.analysis}' requires trial_averaged=True "
                f"(unbalanced trial counts across conditions). "
                f"Only analysis='identity' supports non-averaged mode.")