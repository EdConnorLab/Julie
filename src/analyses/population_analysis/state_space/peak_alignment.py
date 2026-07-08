import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def peak_align(pca_matrix, row_meta_df, info, cfg):
    """Peak-onset align neurons. Trial-averaged only."""
    if not info['trial_averaged']:
        raise NotImplementedError("peak_align supports trial-averaged mode only")

    n_bins = info['n_bins']
    n_conditions = info['n_conditions']
    n_neurons = pca_matrix.shape[1]

    data_3d = pca_matrix.reshape(n_conditions, n_bins, n_neurons)
    mean_profile = data_3d.mean(axis=0)
    peak_rates = mean_profile.max(axis=0)
    thresholds = cfg.peak_fraction * peak_rates
    above = mean_profile >= thresholds[None, :]
    onset_bins = np.argmax(above, axis=0)
    silent = peak_rates == 0
    onset_bins[silent] = 0

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(onset_bins, bins=np.arange(-0.5, n_bins + 0.5, 1), edgecolor='black')
    ax.axvline(np.median(onset_bins), color='red', linestyle='--',
               label=f'median={np.median(onset_bins):.0f}')
    cutoff = None
    if cfg.drop_onset_outliers:
        cutoff = np.percentile(onset_bins[~silent], cfg.outlier_percentile)
        ax.axvline(cutoff, color='orange', linestyle='--',
                   label=f'{cfg.outlier_percentile}th pct={cutoff:.0f}')
    ax.set_xlabel('Onset bin'); ax.set_ylabel('# neurons')
    ax.set_title('Peak-onset distribution'); ax.legend()
    fig.tight_layout()

    keep = ~silent
    if cfg.drop_onset_outliers:
        keep &= onset_bins <= cutoff
    print(f"Peak align: keeping {keep.sum()}/{n_neurons} neurons")

    data_3d = data_3d[:, :, keep]
    onset_bins = onset_bins[keep]
    info['neuron_ids'] = [n for n, k in zip(info['neuron_ids'], keep) if k]

    min_on, max_on = onset_bins.min(), onset_bins.max()
    aligned_start = -min_on
    aligned_end = n_bins - 1 - max_on
    n_aligned = aligned_end - aligned_start + 1
    if n_aligned <= 0:
        raise ValueError("No overlap window after peak alignment")

    n_kept = int(keep.sum())
    aligned = np.zeros((n_conditions, n_aligned, n_kept))
    for i in range(n_kept):
        s = aligned_start + onset_bins[i]
        aligned[:, :, i] = data_3d[:, s:s + n_aligned, i]

    pca_matrix = aligned.reshape(n_conditions * n_aligned, n_kept)

    # Rebuild row_meta_df with aligned bin indices
    has_group = 'group' in row_meta_df.columns
    if has_group:
        c2g = dict(zip(row_meta_df['condition'], row_meta_df['group']))
    aligned_idx = np.arange(aligned_start, aligned_end + 1)
    rows = [(c, int(b)) for c in info['conditions'] for b in aligned_idx]
    new_meta = pd.DataFrame(rows, columns=['condition', 'aligned_bin'])
    new_meta['time_from_onset_s'] = new_meta['aligned_bin'] * cfg.bin_width
    if has_group:
        new_meta['group'] = new_meta['condition'].map(c2g)

    info['n_bins'] = n_aligned
    info['peak_aligned'] = True
    info['aligned_bin_indices'] = aligned_idx
    return pca_matrix, new_meta, info