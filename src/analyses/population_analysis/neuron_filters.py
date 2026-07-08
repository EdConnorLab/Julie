# neuron_filters.py
"""
Optional neuron-level filters applied to the spike DataFrame
before pseudo-population assembly / RSA / encoding analysis.

Two filters available, toggled via config:
  1) Peak-latency filter   (cfg.peak_latency_filter)
       For each neuron, build a per-identity PSTH (pooling trials of that
       identity), Gaussian-smooth, find each identity's peak time.
       Pick the peak time of the *strongest-responding identity*
       (the one with the highest single-bin smoothed firing rate).
       Keep the neuron if that peak time falls inside
       cfg.peak_latency_range.

  2) Pkl NeuronID whitelist  (cfg.neuron_id_filter_pkl)
       Load a pickled DataFrame, take its 'NeuronID' column, and keep
       only neurons whose IDs appear there.

Both filters operate on the raw spike DataFrame (one row per
neuron × trial, with columns NeuronID, MonkeyName, SpikeTimes,
EpochStartStop), so they work uniformly for RSA pseudopop,
RSA single-session, and the social-encoding pipeline.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ──────────────────────────────────────────────────────────
# 1. Peak-latency filter
# ──────────────────────────────────────────────────────────

def compute_peak_latency_per_neuron(
    df: pd.DataFrame,
    search_window: Tuple[float, float] = (0.0, 0.800),
    bin_size: float = 0.020,
    smooth_sigma_sec: float = 0.020,
) -> Dict[str, dict]:
    """
    For each neuron, compute per-identity PSTHs and return the peak time
    of the strongest-responding identity.

    Parameters
    ----------
    df : DataFrame
        Must contain columns NeuronID, MonkeyName, SpikeTimes, EpochStartStop.
        Spike times are relative to recording origin; epoch start is
        EpochStartStop[0].
    search_window : (start_s, end_s)
        Time window (relative to epoch start) in which to look for the peak.
    bin_size : float
        PSTH bin width in seconds.
    smooth_sigma_sec : float
        Gaussian smoothing sigma in seconds (applied along time).

    Returns
    -------
    per_neuron : dict {neuron_id: {'peak_time': float,
                                   'peak_rate': float,
                                   'preferred_identity': str}}
    """
    win_start, win_end = search_window
    bins = np.arange(win_start, win_end + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    sigma_bins = smooth_sigma_sec / bin_size

    per_neuron: Dict[str, dict] = {}

    for nid, ndf in df.groupby('NeuronID', sort=False):
        best_rate = -np.inf
        best_time = np.nan
        best_identity = None

        for monkey, mdf in ndf.groupby('MonkeyName', sort=False):
            # Collect spike times relative to epoch start, across this
            # identity's trials, restricted to the search window.
            spike_chunks = []
            n_trials = 0
            for spk, epoch in zip(mdf['SpikeTimes'].values,
                                  mdf['EpochStartStop'].values):
                rel = np.asarray(spk) - epoch[0]
                rel = rel[(rel >= win_start) & (rel < win_end)]
                spike_chunks.append(rel)
                n_trials += 1

            if n_trials == 0:
                continue

            all_spk = (np.concatenate(spike_chunks)
                       if spike_chunks else np.array([]))
            counts, _ = np.histogram(all_spk, bins=bins)
            psth = counts / n_trials / bin_size               # spikes/sec
            psth_smooth = gaussian_filter1d(psth, sigma=sigma_bins)

            peak_idx = int(np.argmax(psth_smooth))
            peak_rate = float(psth_smooth[peak_idx])

            if peak_rate > best_rate:
                best_rate = peak_rate
                best_time = float(bin_centers[peak_idx])
                best_identity = monkey

        per_neuron[nid] = {
            'peak_time': best_time,
            'peak_rate': best_rate if np.isfinite(best_rate) else np.nan,
            'preferred_identity': best_identity,
        }

    return per_neuron


def filter_neurons_by_peak_latency(
    df: pd.DataFrame,
    latency_range: Tuple[float, float],
    search_window: Tuple[float, float] = (0.0, 0.800),
    bin_size: float = 0.020,
    smooth_sigma_sec: float = 0.020,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Keep only neurons whose preferred-identity peak time lies in
    `latency_range`.

    Returns
    -------
    df_filt : DataFrame
        Subset of df containing only the surviving neurons.
    per_neuron : dict
        Per-neuron diagnostics (peak_time, peak_rate, preferred_identity)
        for *all* input neurons (useful for downstream plotting / QC).
    """
    if latency_range[0] >= latency_range[1]:
        raise ValueError(f"latency_range start ({latency_range[0]}) must be "
                         f"< end ({latency_range[1]})")

    per_neuron = compute_peak_latency_per_neuron(
        df, search_window=search_window,
        bin_size=bin_size, smooth_sigma_sec=smooth_sigma_sec)

    lo, hi = latency_range
    keep_ids = [nid for nid, info in per_neuron.items()
                if np.isfinite(info['peak_time']) and lo <= info['peak_time'] <= hi]

    df_filt = df[df['NeuronID'].isin(keep_ids)].reset_index(drop=True)

    if verbose:
        n_total = len(per_neuron)
        print(f"  Peak-latency filter "
              f"[{lo*1000:.0f}–{hi*1000:.0f} ms, "
              f"search {search_window[0]*1000:.0f}–{search_window[1]*1000:.0f} ms, "
              f"bin {bin_size*1000:.0f} ms, σ {smooth_sigma_sec*1000:.0f} ms]: "
              f"kept {len(keep_ids)}/{n_total} neurons")

    return df_filt, per_neuron


# ──────────────────────────────────────────────────────────
# 2. PKL NeuronID whitelist
# ──────────────────────────────────────────────────────────

def filter_neurons_by_pkl(
    df: pd.DataFrame,
    pkl_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only neurons whose NeuronID appears in the 'NeuronID' column of
    the pickled DataFrame at `pkl_path`.
    """
    pkl_df = pd.read_pickle(pkl_path)
    if 'NeuronID' not in pkl_df.columns:
        raise ValueError(
            f"pkl file {pkl_path!r} has no 'NeuronID' column "
            f"(found: {list(pkl_df.columns)})")

    whitelist = set(pkl_df['NeuronID'].astype(str).unique())
    before = df['NeuronID'].nunique()
    df_filt = df[df['NeuronID'].astype(str).isin(whitelist)].reset_index(drop=True)
    after = df_filt['NeuronID'].nunique()

    if verbose:
        print(f"  PKL NeuronID filter ({pkl_path}): "
              f"kept {after}/{before} neurons "
              f"(pkl whitelist contains {len(whitelist)} unique IDs)")

    return df_filt


# ──────────────────────────────────────────────────────────
# 3. Unified entry point
# ──────────────────────────────────────────────────────────

def apply_neuron_filters(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Apply optional neuron-level filters in sequence based on config flags.

    Order: pkl whitelist first (cheaper), then peak-latency filter on the
    remainder.  If both are on, the result is the intersection.

    Required cfg attributes (all optional, default = off):
      cfg.neuron_id_filter_pkl    : Optional[str]   path to a pickled DataFrame
      cfg.peak_latency_filter     : bool
      cfg.peak_latency_range      : (float, float)  inclusive, seconds
      cfg.peak_latency_search_window : (float, float) seconds
      cfg.peak_latency_bin_size   : float           seconds
      cfg.peak_latency_smooth_sigma : float         seconds
    """
    # ── 1. pkl whitelist ─────────────────────────────────────────
    pkl_path = getattr(cfg, 'neuron_id_filter_pkl', None)
    if pkl_path:
        df = filter_neurons_by_pkl(df, pkl_path)

    # ── 2. peak-latency filter ──────────────────────────────────
    if getattr(cfg, 'peak_latency_filter', False):
        df, _ = filter_neurons_by_peak_latency(
            df,
            latency_range=cfg.peak_latency_range,
            search_window=getattr(cfg, 'peak_latency_search_window',
                                  (0.0, 0.800)),
            bin_size=getattr(cfg, 'peak_latency_bin_size', 0.020),
            smooth_sigma_sec=getattr(cfg, 'peak_latency_smooth_sigma',
                                     0.020),
        )

    return df
