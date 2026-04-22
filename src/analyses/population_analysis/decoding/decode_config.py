# decode_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecodeIdentityConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = '/sorted_spike_cache_filtered'
    region: str = 'ALL'                # 'AMG', 'ER', or 'ALL'
    session: Optional[str] = None      # None = all sessions; else e.g. 'ER_2023-11-28_4'
    min_epoch_duration: float = 2.2    # drop trials shorter than this (seconds)

    # ── Neuron filtering ──────────────────────────────────────────────────
    min_trials: int = 0                # min trials per identity per neuron (0 = no filter)

    # ── Pseudo-population ─────────────────────────────────────────────────
    n_pseudo_draws: int = 10           # number of random trial-pairings to average over

    # ── Decoding ──────────────────────────────────────────────────────────
    n_cv_folds: int = 5
    n_pca: Optional[int] = 50         # None or 0 = skip PCA
    logreg_C: float = 1.0             # inverse regularisation strength
    logreg_max_iter: int = 2000

    # ── Permutation test ──────────────────────────────────────────────────
    skip_perm: bool = False            # True = skip permutation test entirely
    n_permutations: int = 1000

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = './decode_identity_output'
    random_seed: int = 42

    # ──────────────────────────────────────────────────────────────────────
    def validate(self):
        if self.n_pca is not None and self.n_pca <= 0:
            self.n_pca = None
        if self.min_trials < 0:
            raise ValueError("min_trials must be >= 0")
        if self.n_cv_folds < 2:
            raise ValueError("n_cv_folds must be >= 2")
