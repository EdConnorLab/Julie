# rsa_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class RSAConfig:
    """
    Base RSA configuration.

    Used directly by run_rsa.py (model-factor RSA).
    Subclassed by SocialRSAConfig for run_rsa_social.py.
    """

    # ── Data ──
    data_path: str = '/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered'
    monkey_info_path: str = '/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv'
    region: str = 'ALL'              # 'AMG', 'ER', or 'ALL'
    session: Optional[str] = None    # None = all sessions → pseudo-population mode
                                     # 'session_id' = single-session mode

    # ── Firing-rate window (seconds relative to epoch start) ──
    window: Tuple[float, float] = (0.200, 0.600)

    # ── Trial filtering ──
    min_epoch_duration: float = 2.0   # drop trials shorter than this
    min_reps_per_monkey: int = 3      # need at least this many reps per identity
    exclude_identities: List[str] = field(default_factory=list)
    exclude_groups: List[str] = field(default_factory=list)  # e.g. ['Stranger Things']

    # ── Neural RDM ──
    neural_metric: str = 'correlation'   # 'correlation' (1-pearson_r) or 'euclidean' or 'cosine' or 'mahalanobis'

    # ── Normalization (applied per neuron before building RDM) ──
    normalization: str = None            # None   = raw firing rates
                                         # 'soft' = divide by (range + const); conservative equalization
                                         # 'zscore' = zscore firing rates
    soft_normalize_const: float = 5.0    # only used when normalization='soft'

    # ── Visual responsiveness filter ──
    # Keeps only neurons with significant stimulus-driven responses.
    # Compares firing in vr_response_window to vr_baseline_window (0–100 ms,
    # before visual responses reach MTL) using Wilcoxon signed-rank test.
    visual_responsiveness_filter: bool = False
    vr_baseline_window: Tuple[float, float] = (0.0, 0.100)    # 0–100 ms
    vr_response_window: Tuple[float, float] = (0.150, 0.600)  # 150–600 ms
    vr_alpha: float = 0.2
    vr_require_increase: bool = True   # only excitatory responses

    # ── Peak-latency filter ──
    # For each neuron, compute per-identity PSTHs (Gaussian-smoothed),
    # pick the peak time of the strongest-responding identity, and keep
    # only neurons whose peak time lies inside peak_latency_range.
    peak_latency_filter: bool = False
    peak_latency_range: Tuple[float, float] = (0.200, 0.500)         # inclusive, seconds
    peak_latency_search_window: Tuple[float, float] = (0.0, 0.800)   # where to look for the peak
    peak_latency_bin_size: float = 0.020                              # PSTH bin width, seconds
    peak_latency_smooth_sigma: float = 0.020                          # Gaussian σ, seconds

    # ── Pkl-based NeuronID whitelist ──
    # Path to a pickled DataFrame; only neurons whose NeuronID appears in
    # its 'NeuronID' column are kept.  None = disabled.
    neuron_id_filter_pkl: Optional[str] = None

    # ── Partial RSA (run_rsa.py): regress out these model RDMs before testing each factor ──
    # e.g. ['familiarity', 'group'] → for each target factor, partial out
    # familiarity and group, then correlate residuals.
    # Factors listed here are auto-built even if not in model_factors.
    partial_out: List[str] = field(default_factory=list)

    # ── Stats ──
    # n_permutations  : p-value for "is ρ > 0?" within each group.
    #                   Shuffles neural RDM rows+cols to build a null distribution of ρ.
    n_permutations: int = 0           # 0 = skip permutation test
    rng_seed: int = 42

    # ── Plotting ──
    save_plots: bool = True
    save_dir: str = 'rsa_results'
    group_colors: dict = field(default_factory=lambda: {
        'Zombies':         '#9467bd',
        'Best Frans':      '#d62728',
        'Instigators':     '#2ca02c',
        'Stranger Things': '#1f77b4',
    })

    # ── Model RSA only (run_rsa.py) ──
    model_factors: List[str] = field(default_factory=lambda: [
        'group', 'sex', 'age_bin', 'rank'
    ])
    # Additional options: 'age_continuous', 'familiarity'
    rdm_sort_mode: str = 'by_factor'   # 'by_factor' = each subplot sorted by its own factor
                                        # 'by_group'  = all subplots sorted by group → sex → age

    def validate(self):
        if self.window[0] >= self.window[1]:
            raise ValueError(f"window start ({self.window[0]}) must be < end ({self.window[1]})")
        if self.window[1] > self.min_epoch_duration:
            raise ValueError(
                f"window end ({self.window[1]}) exceeds min_epoch_duration "
                f"({self.min_epoch_duration}). Increase min_epoch_duration or shrink window.")
        if self.peak_latency_filter:
            if self.peak_latency_range[0] >= self.peak_latency_range[1]:
                raise ValueError("peak_latency_range start must be < end")
            sw = self.peak_latency_search_window
            if sw[0] >= sw[1]:
                raise ValueError("peak_latency_search_window start must be < end")


@dataclass
class SocialRSAConfig(RSAConfig):
    """
    Extended config for social behavior RSA (run_rsa_social.py).

    Adds social-specific parameters on top of the shared base.
    """

    # ── Social profile RDMs ──
    # transform_social_behavior :
    #   None   = raw interaction counts
    #   'rank' = rank-transform each profile vector before distance
    #   'log'  = log1p-transform the interaction matrix before profiling
    transform_social_behavior: Optional[str] = None

    # ── Partial out rank from social RSA (|rank_i - rank_j| as confound within each group) ──
    partial_out_rank: bool = False

    # ── Sensitivity analysis: exclude specific monkeys ──
    # List of monkey IDs to remove before RSA (e.g. ['7124'] to drop the alpha).
    # Removes them from identities, neural RDM rows/cols, and social RDMs.
    exclude_identities: List[str] = field(default_factory=list)

    # ── Social RSA stats ──
    # between_group_permutations : p-value for "does ρ differ between groups?"
    #   Pools within-group pairs from both groups, shuffles group labels,
    #   recomputes Δρ = ρ_A − ρ_B to test if the observed difference is above chance.
    between_group_permutations: int = 0     # 0 = skip between-group Δρ test

    # n_bootstrap : 95% confidence interval on ρ within each group (not a p-value).
    #   Resamples pairs with replacement to quantify estimation uncertainty.
    n_bootstrap: int = 0                    # 0 = skip bootstrap CI on ρ

    def validate(self):
        super().validate()
        valid = {None, 'rank', 'log'}
        if self.transform_social_behavior not in valid:
            raise ValueError(
                f"transform_social_behavior must be one of {valid}, "
                f"got {self.transform_social_behavior!r}")
