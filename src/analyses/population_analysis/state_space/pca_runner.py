# pca_runner.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca(pca_matrix, info, cfg):
    X = pca_matrix

    pca = PCA(n_components=cfg.n_components)
    scores = pca.fit_transform(X)

    n_bins = info['n_bins']
    n_conditions = info['n_conditions']

    if info['trial_averaged']:
        # rows = (condition, time_bin)
        scores_3d = scores.reshape(n_conditions, n_bins, cfg.n_components)
    else:
        # rows = (condition, rep, time_bin) — collapse reps after PCA
        n_reps = info['global_min_reps']
        scores_4d = scores.reshape(n_conditions, n_reps, n_bins, cfg.n_components)
        scores_3d = scores_4d.mean(axis=1)  # (n_conditions, n_bins, n_components)

    var = pca.explained_variance_ratio_
    print("Variance: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))

    pca_full = PCA().fit(X)
    return dict(pca=pca, pca_full=pca_full, scaler=None,
                scores_3d=scores_3d, var=var, n_units=n_conditions)


def plot_scree(pca_result, n_show=50):
    ratios = pca_result['pca_full'].explained_variance_ratio_
    n_show = min(n_show, len(ratios))
    x = np.arange(1, n_show + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # axes[0].bar(x, ratios[:n_show] * 100)
    axes[0].plot(x, ratios[:n_show] * 100, 'o-')
    axes[0].set_xlabel('PC'); axes[0].set_ylabel('% variance'); axes[0].set_title('Scree')

    cum = np.cumsum(ratios[:n_show]) * 100
    axes[1].plot(x, cum, 'o-')
    axes[1].axhline(80, color='gray', ls='--', alpha=0.5, label='80%')
    axes[1].axhline(90, color='gray', ls=':', alpha=0.5, label='90%')
    axes[1].set_xlabel('# components'); axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative'); axes[1].legend()
    fig.tight_layout()
    fig.savefig('scree.png')
    return fig