# decode_config.py
"""Unified configuration for all decoding and RSA scripts."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DecodeConfig:
    """
    Shared configuration for identity / group / pairwise / social / RSA decoding.

    Parameters
    ----------
    data_path : str
        Directory containing per-neuron .pkl files.
    region : str
        Brain region filter: 'AMG', 'ER', or 'ALL'.
    session : str or None
        Restrict to a single recording session (e.g. 'ER_2023-11-28_4').
    exclude_monkeys : list[str]
        Monkey names to exclude (e.g. ['NewMonkey', '81G']).

    response_window : tuple[float, float] or None
        (start_sec, end_sec) relative to epoch onset, e.g. (0.0, 0.5).
        Must be within 0–2 sec.  None = use the full epoch duration.

    min_trials : int
        Minimum trials per identity per neuron (0 = no filter).
    min_epoch_duration : float
        Drop trials whose epoch is shorter than this (seconds).

    n_pseudo_draws : int
        Number of random trial-pairings to average over.
    n_cv_folds : int
        Cross-validation folds.
    n_pca : int or None
        PCA components before classification (None / 0 = skip).
    logreg_C : float
        Inverse regularisation strength for LogisticRegression.
    logreg_max_iter : int
        Max solver iterations.

    skip_perm : bool
        If True, skip all permutation tests.
    n_permutations : int
        Number of permutations when running tests.

    output_dir : str
        Where to write results and figures.
    random_seed : int
        Master random seed.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = '/Cortana/sorted_spike_cache_filtered'
    region: str = 'ALL'
    session: Optional[str] = None
    exclude_monkeys: list = field(default_factory=lambda: ['NewMonkey', '70G', '79G', '42Z', '144H'])
    exclude_monkey_groups: list = field(default_factory=lambda: [])

    # ── Response window ──────────────────────────────────────────────────
    response_window: Optional[Tuple[float, float]] = None  # (start_sec, end_sec)

    # ── Neuron / trial filtering ─────────────────────────────────────────
    min_trials: int = 0
    min_epoch_duration: float = 1.0

    # ── Pseudo-population ────────────────────────────────────────────────
    n_pseudo_draws: int = 10

    # ── Decoding ─────────────────────────────────────────────────────────
    n_cv_folds: int = 5
    n_pca: Optional[int] = 50
    logreg_C: float = 1.0
    logreg_max_iter: int = 2000

    # ── Permutation test ─────────────────────────────────────────────────
    skip_perm: bool = False
    n_permutations: int = 1000

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = './decode_output'
    random_seed: int = 42

    # ──────────────────────────────────────────────────────────────────────
    def validate(self):
        if self.n_pca is not None and self.n_pca <= 0:
            self.n_pca = None
        if self.min_trials < 0:
            raise ValueError("min_trials must be >= 0")
        if self.n_cv_folds < 2:
            raise ValueError("n_cv_folds must be >= 2")
        if self.response_window is not None:
            lo, hi = self.response_window
            if not (0.0 <= lo < hi <= 2.0):
                raise ValueError(
                    f"response_window must satisfy 0 <= start < end <= 2.0, "
                    f"got ({lo}, {hi})"
                )
