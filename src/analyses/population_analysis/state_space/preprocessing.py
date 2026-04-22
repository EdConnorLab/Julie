# preprocessing.py
"""
Churchland-style preprocessing for trajectory analysis.

Pipeline (applied in order):
    1. Convert spike counts to firing rates (Hz)
    2. Soft-normalize each neuron:  r / (range + softnorm_const)
    3. Mean-center across conditions at each timepoint:
           r_hat[c, t, n] = r[c, t, n] - mean_c r[c, t, n]

Operates on the matrix produced by build_matrix:
    averaged:     rows = (monkey, time_bin),       cols = neurons
    non-averaged: rows = (monkey, rep, time_bin),  cols = neurons

For the non-averaged case, the across-condition mean is computed from the
rep-averaged response (so a single mean trace per neuron per timepoint),
then subtracted from every rep. This matches what you'd get if you mean-
centered the trial-averaged data and is the standard approach.
"""
import numpy as np


def preprocess(pca_matrix, info, cfg, softnorm_const=5.0,
               soft_normalize=True, mean_center=False):
    """
    Returns a new matrix of the same shape as pca_matrix, preprocessed.

    Parameters
    ----------
    pca_matrix : ndarray
        Output of build_matrix. Spike counts per bin.
    info : dict
        Output of build_matrix. Must contain n_bins, bin_width, n_conditions (could be monkeys or social groups or some other conditions)
        trial_averaged, and (if non-averaged) global_min_reps.
    cfg : TrajectoryConfig
    softnorm_const : float
        Soft-normalization constant in Hz. Churchland uses 5.
    soft_normalize : bool
    mean_center : bool
    """
    n_bins = info['n_bins']
    n_conditions = info['n_conditions']
    bin_width = info['bin_width']
    n_neurons = pca_matrix.shape[1]

    # 1. counts -> Hz
    rates = pca_matrix / bin_width

    if info['trial_averaged']:
        # (n_conditions, n_bins, n_neurons)
        R = rates.reshape(n_conditions, n_bins, n_neurons)

        if soft_normalize:
            # range over (conditions, time) per neuron
            rng = R.max(axis=(0, 1)) - R.min(axis=(0, 1))      # (n_neurons,)
            R = R / (rng + softnorm_const)

        if mean_center:
            cond_mean = R.mean(axis=0, keepdims=True)          # (1, n_bins, n_neurons)
            R = R - cond_mean

        out = R.reshape(n_conditions * n_bins, n_neurons)

    else:
        n_reps = info['global_min_reps']
        # (n_conditions, n_reps, n_bins, n_neurons)
        R = rates.reshape(n_conditions, n_reps, n_bins, n_neurons)

        # Trial-averaged view used for computing stats
        R_avg = R.mean(axis=1)  # (n_conditions, n_bins, n_neurons)

        if soft_normalize:
            rng = R_avg.max(axis=(0, 1)) - R_avg.min(axis=(0, 1))   # (n_neurons,)
            denom = rng + softnorm_const
            R = R / denom
            R_avg = R_avg / denom

        if mean_center:
            cond_mean = R_avg.mean(axis=0, keepdims=True)       # (1, n_bins, n_neurons)
            # broadcast over reps: insert rep axis
            R = R - cond_mean[:, None, :, :]

        out = R.reshape(n_conditions * n_reps * n_bins, n_neurons)

    # Diagnostics
    print(f"Preprocess: Hz | softnorm={'on' if soft_normalize else 'off'} "
          f"(c={softnorm_const}) | mean_center={'on' if mean_center else 'off'}")
    print(f"  matrix range after preprocess: [{out.min():.3f}, {out.max():.3f}]")
    return out