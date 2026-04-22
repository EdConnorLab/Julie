# rsa_plotting.py
"""
Plotting functions for RSA results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import os


def _group_sort_index(result):
    """Return a sort index: group → sex (F before M) → age (adult before juvenile).
    Used by all RDM plotting functions so row/col order is consistent."""
    model_labels = result['model_labels']
    identities = result['identities']
    n = len(identities)

    # Build a sort-key tuple per identity
    sex_order = {'F': 0, 'M': 1}
    age_order = {'adult': 0, 'juvenile': 1}

    keys = []
    for i in range(n):
        g = model_labels['group'][i] if 'group' in model_labels else ''
        s = model_labels.get('sex', [None]*n)[i]
        a = model_labels.get('age_bin', [None]*n)[i]
        keys.append((
            g,
            sex_order.get(s, 9),   # F=0, M=1, missing=9
            age_order.get(a, 9),   # adult=0, juvenile=1, missing=9
        ))

    sort_idx = np.array(sorted(range(n), key=lambda i: keys[i]))
    return sort_idx


def _apply_sort(rdm, identities, model_labels, sort_idx):
    """Apply a sort index to an RDM and its labels."""
    rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
    ids_sorted = [identities[i] for i in sort_idx]
    groups_sorted = ([model_labels['group'][i] for i in sort_idx]
                     if 'group' in model_labels
                     else [None] * len(identities))
    return rdm_sorted, ids_sorted, groups_sorted


def plot_neural_rdm(result, cfg, suffix='', save=True, save_dir=None):
    """Plot neural RDM heatmap with identity labels colored by group."""
    rdm = result['neural_rdm']
    identities = result['identities']
    model_labels = result['model_labels']

    fig, ax = plt.subplots(figsize=(10, 9))

    sort_idx = _group_sort_index(result)
    rdm_sorted, ids_sorted, groups_sorted = _apply_sort(
        rdm, identities, model_labels, sort_idx)

    im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    metric_label = '1 - r' if cfg.neural_metric == 'correlation' else 'Euclidean dist'
    cbar.set_label(metric_label)

    ax.set_xticks(range(len(ids_sorted)))
    ax.set_yticks(range(len(ids_sorted)))
    ax.set_xticklabels(ids_sorted, rotation=90, fontsize=7)
    ax.set_yticklabels(ids_sorted, fontsize=7)

    # Color tick labels by group
    for i, (tick_x, tick_y) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        g = groups_sorted[i]
        if g in cfg.group_colors:
            tick_x.set_color(cfg.group_colors[g])
            tick_y.set_color(cfg.group_colors[g])

    # Draw group boundaries
    if 'group' in model_labels:
        boundaries = []
        prev = groups_sorted[0]
        for i, g in enumerate(groups_sorted):
            if g != prev:
                boundaries.append(i - 0.5)
                prev = g
        for b in boundaries:
            ax.axhline(b, color='k', lw=1, alpha=0.7)
            ax.axvline(b, color='k', lw=1, alpha=0.7)

    # Legend
    handles = [mpatches.Patch(color=c, label=g) for g, c in cfg.group_colors.items()
               if g in set(groups_sorted)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=8)

    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"
    title = f"Neural RDM ({metric_label}) | {result['session']} | {win_str}"
    if suffix:
        title += f" | {suffix}"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"neural_rdm_{result['session']}_{win_str}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig


def plot_model_rdms(result, cfg, save=True, save_dir=None):
    """Plot all model RDMs side by side, sorted to match neural RDM."""
    model_rdms = result['model_rdms']
    identities = result['identities']
    model_labels = result['model_labels']
    n_models = len(model_rdms)

    sort_idx = _group_sort_index(result)
    _, ids_sorted, groups_sorted = _apply_sort(
        result['neural_rdm'], identities, model_labels, sort_idx)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (factor, rdm) in zip(axes, model_rdms.items()):
        rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
        im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(factor, fontsize=11)
        ax.set_xticks(range(len(ids_sorted)))
        ax.set_yticks(range(len(ids_sorted)))
        ax.set_xticklabels(ids_sorted, rotation=90, fontsize=5)
        ax.set_yticklabels(ids_sorted, fontsize=5)

        # Color tick labels by group
        for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
            g = groups_sorted[i]
            if g in cfg.group_colors:
                tx.set_color(cfg.group_colors[g])
                ty.set_color(cfg.group_colors[g])

    fig.suptitle(f"Model RDMs | {result['session']}", fontsize=12)
    fig.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"model_rdms_{result['session']}.png"),
                    dpi=150, bbox_inches='tight')
    return fig


def plot_rsa_bar(results_list, cfg, save=True, save_dir=None):
    """
    Bar chart of RSA correlations across sessions (or a single pseudo-pop result).

    Parameters
    ----------
    results_list : list of result dicts (from run_rsa_session or run_rsa_pseudopop)
    """
    factors = cfg.model_factors

    # Collect rho values: shape (n_sessions, n_factors)
    session_names = []
    rho_matrix = []
    for r in results_list:
        if r is None:
            continue
        session_names.append(r['session'])
        rho_matrix.append([r['comparisons'][f]['rho'] for f in factors])
    rho_matrix = np.array(rho_matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(factors) * 2), 5))

    if len(rho_matrix) == 1:
        # Single result (pseudo-pop or one session): simple bar chart
        bars = ax.bar(factors, rho_matrix[0], color='steelblue', edgecolor='k', alpha=0.8)
        ax.set_title(f"RSA | {session_names[0]}")
        # Add significance stars if permutation test was run
        for i, f in enumerate(factors):
            p = results_list[0]['comparisons'][f].get('p_val')
            if p is not None:
                star = _p_to_stars(p)
                ax.text(i, rho_matrix[0, i] + 0.01, star, ha='center', fontsize=12)
    else:
        # Multiple sessions: bar per factor with individual session dots + mean ± SEM
        x = np.arange(len(factors))
        means = np.nanmean(rho_matrix, axis=0)
        sems = np.nanstd(rho_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(rho_matrix), axis=0))

        ax.bar(x, means, yerr=sems, color='steelblue', edgecolor='k',
               alpha=0.6, capsize=4, label='mean ± SEM')
        # Overlay individual session dots
        for si in range(len(rho_matrix)):
            jitter = np.random.default_rng(si).uniform(-0.15, 0.15, len(factors))
            ax.scatter(x + jitter, rho_matrix[si], color='k', s=15, alpha=0.5, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.set_title(f"RSA across {len(rho_matrix)} sessions")

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Spearman ρ (neural vs model RDM)')
    ax.set_xlabel('Model factor')
    fig.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'rsa_bar.png'), dpi=150, bbox_inches='tight')
    return fig


def plot_mds(result, cfg, color_by='group', save=True, save_dir=None):
    """
    2D MDS of neural RDM, colored by a factor.
    """
    rdm = result['neural_rdm']
    identities = result['identities']
    model_labels = result['model_labels']

    if color_by not in model_labels:
        print(f"  Cannot color by '{color_by}': not in model_labels")
        return None

    labels = model_labels[color_by]

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=cfg.rng_seed,
              normalized_stress='auto')
    coords = mds.fit_transform(rdm)

    fig, ax = plt.subplots(figsize=(8, 7))

    unique_labels = sorted(set(l for l in labels if l is not None and l is not np.nan))
    # Use group_colors if coloring by group, else auto
    if color_by == 'group':
        cmap = cfg.group_colors
    else:
        default_colors = plt.cm.tab10.colors
        cmap = {l: default_colors[i % 10] for i, l in enumerate(unique_labels)}

    for i, (x, y) in enumerate(coords):
        lab = labels[i]
        color = cmap.get(lab, 'gray')
        ax.scatter(x, y, color=color, s=60, edgecolors='k', lw=0.5, zorder=5)
        ax.annotate(identities[i], (x, y), fontsize=6, textcoords='offset points',
                    xytext=(4, 4))

    handles = [mpatches.Patch(color=cmap.get(l, 'gray'), label=str(l)) for l in unique_labels]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('MDS dim 1')
    ax.set_ylabel('MDS dim 2')
    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"
    ax.set_title(f"MDS of neural RDM | color={color_by} | {result['session']} | {win_str}")
    fig.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"mds_{color_by}_{result['session']}.png"),
                    dpi=150, bbox_inches='tight')
    return fig


def _sort_by_factor(result, factor):
    """Sort index for a factor, with secondary sort by name.
    Handles both categorical (group, sex) and numeric (rank, age_continuous) factors."""
    model_labels = result['model_labels']
    identities = result['identities']
    n = len(identities)
    labels = model_labels.get(factor, [None] * n)

    def _sort_key(i):
        val = labels[i]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return (1, float('inf'), identities[i])  # NaN/None last
        if isinstance(val, (int, float)):
            return (0, val, identities[i])
        return (0, str(val), identities[i])

    sort_idx = np.array(sorted(range(n), key=_sort_key))
    sorted_labels = [labels[i] for i in sort_idx]
    sorted_ids = [identities[i] for i in sort_idx]
    return sort_idx, sorted_labels, sorted_ids


def _is_continuous_factor(labels):
    """Check if a factor's labels are numeric (continuous/ordinal) vs categorical."""
    vals = [l for l in labels if l is not None and not (isinstance(l, float) and np.isnan(l))]
    return len(vals) > 0 and all(isinstance(v, (int, float)) for v in vals)


def plot_neural_rdm_multi_sort(result, cfg, factors=None,
                               save=True, save_dir=None):
    """
    Plot the same neural RDM N ways, each sorted by a different factor.
    Automatically handles categorical (group, sex, age_bin, familiarity) and
    continuous/ordinal (rank, age_continuous) factors.

    Parameters
    ----------
    result : dict from run_rsa_session / run_rsa_pseudopop
    cfg : RSAConfig
    factors : list of str or None
        Which factors to sort by (one subplot each). None → cfg.model_factors.
    """
    if factors is None:
        factors = cfg.model_factors

    rdm = result['neural_rdm']
    n_panels = len(factors)

    # Predefined color maps for known categorical factors
    categorical_color_maps = {
        'group': cfg.group_colors,
        'sex': {'F': '#e377c2', 'M': '#17becf'},
        'age_bin': {'adult': '#ff7f0e', 'juvenile': '#2ca02c'},
        'familiarity': {'familiar': '#d62728', 'unfamiliar': '#1f77b4'},
    }

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    metric_label = '1 - r' if cfg.neural_metric == 'correlation' else 'Euclidean dist'
    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"

    # Shared color scale across subplots
    vmin = np.nanmin(rdm[np.triu_indices_from(rdm, k=1)])
    vmax = np.nanmax(rdm[np.triu_indices_from(rdm, k=1)])

    for ax, factor in zip(axes, factors):
        if factor not in result['model_labels']:
            ax.set_title(f"{factor} (not available)")
            ax.axis('off')
            continue

        sort_idx, sorted_labels, sorted_ids = _sort_by_factor(result, factor)
        rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
        continuous = _is_continuous_factor(sorted_labels)

        im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(sorted_ids)))
        ax.set_yticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=90, fontsize=6)
        ax.set_yticklabels(sorted_ids, fontsize=6)

        if continuous:
            # --- Continuous / ordinal factor: gradient-colored tick labels ---
            valid_vals = [l for l in sorted_labels
                         if l is not None and not (isinstance(l, float) and np.isnan(l))]
            lo, hi = min(valid_vals), max(valid_vals)
            gradient_cmap = plt.cm.viridis

            for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
                val = sorted_labels[i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    normed = (val - lo) / (hi - lo) if hi > lo else 0.5
                    c = gradient_cmap(normed)
                else:
                    c = 'gray'
                tx.set_color(c)
                ty.set_color(c)

            # Small gradient colorbar as legend
            sm = plt.cm.ScalarMappable(cmap=gradient_cmap,
                                       norm=plt.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cbar_factor = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.01,
                                       location='bottom', aspect=30)
            cbar_factor.set_label(factor, fontsize=8)

        else:
            # --- Categorical factor: discrete colored labels + boundaries ---
            cmap_cat = categorical_color_maps.get(factor, {})
            if not cmap_cat:
                unique = sorted(set(str(l) for l in sorted_labels
                                    if l is not None))
                colors = plt.cm.tab10.colors
                cmap_cat = {l: colors[i % 10] for i, l in enumerate(unique)}

            for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
                lab = sorted_labels[i]
                c = cmap_cat.get(lab, cmap_cat.get(str(lab), 'k'))
                tx.set_color(c)
                ty.set_color(c)

            # Draw boundaries where label changes
            boundaries = []
            prev = sorted_labels[0]
            for i, lab in enumerate(sorted_labels):
                if lab != prev:
                    boundaries.append(i - 0.5)
                    prev = lab
            for b in boundaries:
                ax.axhline(b, color='k', lw=1.2, alpha=0.7)
                ax.axvline(b, color='k', lw=1.2, alpha=0.7)

            # Legend
            unique_in_plot = list(dict.fromkeys(sorted_labels))
            handles = [mpatches.Patch(
                           color=cmap_cat.get(l, cmap_cat.get(str(l), 'gray')),
                           label=str(l))
                       for l in unique_in_plot if l is not None]
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.02),
                      fontsize=8, ncol=min(len(handles), 4), frameon=False)

        ax.set_title(f"sorted by {factor}", fontsize=11)

    fig.suptitle(f"Neural RDM ({metric_label}) | {result['session']} | {win_str}",
                 fontsize=13, y=1.02)

    fig.tight_layout()
    fig.subplots_adjust(right=0.92)

    # Shared colorbar for the RDM dissimilarity scale
    cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(metric_label)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"neural_rdm_multi_{result['session']}_{win_str}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig


def _p_to_stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'
