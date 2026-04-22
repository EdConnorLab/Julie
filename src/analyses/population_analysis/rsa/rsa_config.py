# rsa_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class RSAConfig:
    # Data
    data_path: str = '/sorted_spike_cache_filtered'
    monkey_info_path: str = '/social_data/monkeyinfo.csv'
    region: str = 'ALL'              # 'AMG', 'ER', or 'ALL'
    session: Optional[str] = None    # None = all sessions

    # Firing-rate window (seconds relative to epoch start)
    window: Tuple[float, float] = (0.200, 0.600)

    # Trial filtering
    min_epoch_duration: float = 2.0   # drop trials shorter than this
    min_reps_per_monkey: int = 3      # need at least this many reps per identity
    exclude_groups: List[str] = field(default_factory=list)  # e.g. ['Stranger Things']

    # RSA
    neural_metric: str = 'correlation'   # 'correlation' (1-r) or 'euclidean'
    model_factors: List[str] = field(default_factory=lambda: [
        'group', 'sex', 'age_bin', 'rank'
    ])
    # Additional options: 'age_continuous', 'familiarity'

    # PCA dimensionality reduction before RSA
    pca_before_rsa: bool = False
    pca_n_components: Optional[int] = None    # fixed # of PCs; None → use variance threshold
    pca_var_threshold: float = 0.90           # keep PCs explaining this fraction of variance
                                              # (only used when pca_n_components is None)

    # Stats
    n_permutations: int = 0           # 0 = skip permutation test
    rng_seed: int = 42

    # Pseudo-population (pool neurons across sessions within a region)
    pseudo_population: bool = False   # False = per-session RSA

    # Plotting
    save_plots: bool = True
    save_dir: str = 'rsa_results'

    group_colors: dict = field(default_factory=lambda: {
        'Zombies':         '#9467bd',
        'Best Frans':      '#d62728',
        'Instigators':     '#2ca02c',
        'Stranger Things': '#1f77b4',
    })

    def validate(self):
        if self.window[0] >= self.window[1]:
            raise ValueError(f"window start ({self.window[0]}) must be < end ({self.window[1]})")
        if self.window[1] > self.min_epoch_duration:
            raise ValueError(
                f"window end ({self.window[1]}) exceeds min_epoch_duration "
                f"({self.min_epoch_duration}). Increase min_epoch_duration or shrink window.")
