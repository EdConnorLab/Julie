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


_METRIC_LABELS = {
    'correlation': '1 - r',
    'euclidean': 'Euclidean dist',
    'cosine': 'Cosine dist',
    'mahalanobis': 'Mahalanobis dist',
}


def _metric_label(metric):
    return _METRIC_LABELS.get(metric, metric)


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
    metric_label = _metric_label(cfg.neural_metric)
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


def _get_sort_for_factor(result, factor, cfg):
    """Pick sort index based on cfg.rdm_sort_mode.
    'by_factor' → sort by this factor; 'by_group' → sort by group/sex/age."""
    if getattr(cfg, 'rdm_sort_mode', 'by_factor') == 'by_group':
        sort_idx = _group_sort_index(result)
        model_labels = result['model_labels']
        identities = result['identities']
        sorted_ids = [identities[i] for i in sort_idx]
        labels = model_labels.get(factor, [None] * len(identities))
        sorted_labels = [labels[i] for i in sort_idx]
        # For by_group mode, use group labels for coloring/boundaries
        groups_sorted = ([model_labels['group'][i] for i in sort_idx]
                         if 'group' in model_labels
                         else [None] * len(identities))
        return sort_idx, sorted_labels, sorted_ids, groups_sorted
    else:
        sort_idx, sorted_labels, sorted_ids = _sort_by_factor(result, factor)
        return sort_idx, sorted_labels, sorted_ids, None


def plot_model_rdms(result, cfg, save=True, save_dir=None):
    """Plot each model RDM. Sort controlled by cfg.rdm_sort_mode."""
    model_rdms = result['model_rdms']
    model_labels = result['model_labels']
    n_models = len(model_rdms)

    categorical_color_maps = {
        'group': cfg.group_colors,
        'sex': {'F': '#e377c2', 'M': '#17becf'},
        'age_bin': {'adult': '#ff7f0e', 'juvenile': '#2ca02c'},
        'familiarity': {'familiar': '#d62728', 'unfamiliar': '#1f77b4'},
    }
    by_group = getattr(cfg, 'rdm_sort_mode', 'by_factor') == 'by_group'

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    for ax, (factor, rdm) in zip(axes, model_rdms.items()):
        sort_idx, sorted_labels, sorted_ids, groups_sorted = \
            _get_sort_for_factor(result, factor, cfg)
        rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]

        im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(factor, fontsize=11)
        ax.set_xticks(range(len(sorted_ids)))
        ax.set_yticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=90, fontsize=5)
        ax.set_yticklabels(sorted_ids, fontsize=5)

        if by_group and groups_sorted is not None:
            # Color by group, draw group boundaries
            for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
                g = groups_sorted[i]
                if g in cfg.group_colors:
                    tx.set_color(cfg.group_colors[g])
                    ty.set_color(cfg.group_colors[g])
            boundaries = []
            prev = groups_sorted[0]
            for i, g in enumerate(groups_sorted):
                if g != prev:
                    boundaries.append(i - 0.5)
                    prev = g
            for b in boundaries:
                ax.axhline(b, color='k', lw=1.2, alpha=0.7)
                ax.axvline(b, color='k', lw=1.2, alpha=0.7)
        elif _is_continuous_factor(sorted_labels):
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
        else:
            cmap_cat = categorical_color_maps.get(factor, {})
            if not cmap_cat:
                unique = sorted(set(str(l) for l in sorted_labels if l is not None))
                colors = plt.cm.tab10.colors
                cmap_cat = {l: colors[i % 10] for i, l in enumerate(unique)}
            for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
                lab = sorted_labels[i]
                c = cmap_cat.get(lab, cmap_cat.get(str(lab), 'k'))
                tx.set_color(c)
                ty.set_color(c)
            boundaries = []
            prev = sorted_labels[0]
            for i, lab in enumerate(sorted_labels):
                if lab != prev:
                    boundaries.append(i - 0.5)
                    prev = lab
            for b in boundaries:
                ax.axhline(b, color='k', lw=1.2, alpha=0.7)
                ax.axvline(b, color='k', lw=1.2, alpha=0.7)

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
    factors = getattr(cfg, 'model_factors', [])

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
        Which factors to sort by (one subplot each). None → getattr(cfg, 'model_factors', []).
    """
    if factors is None:
        factors = getattr(cfg, 'model_factors', [])

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

    metric_label = _metric_label(cfg.neural_metric)
    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"

    # Shared color scale across subplots
    vmin = np.nanmin(rdm[np.triu_indices_from(rdm, k=1)])
    vmax = np.nanmax(rdm[np.triu_indices_from(rdm, k=1)])

    for ax, factor in zip(axes, factors):
        if factor not in result['model_labels']:
            ax.set_title(f"{factor} (not available)")
            ax.axis('off')
            continue

        sort_idx, sorted_labels, sorted_ids, groups_sorted = \
            _get_sort_for_factor(result, factor, cfg)
        rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
        by_group = groups_sorted is not None
        continuous = _is_continuous_factor(sorted_labels)

        im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(sorted_ids)))
        ax.set_yticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=90, fontsize=6)
        ax.set_yticklabels(sorted_ids, fontsize=6)

        if by_group:
            # by_group mode: color by group, draw group boundaries
            for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
                g = groups_sorted[i]
                if g in cfg.group_colors:
                    tx.set_color(cfg.group_colors[g])
                    ty.set_color(cfg.group_colors[g])
            boundaries = []
            prev = groups_sorted[0]
            for i, g in enumerate(groups_sorted):
                if g != prev:
                    boundaries.append(i - 0.5)
                    prev = g
            for b in boundaries:
                ax.axhline(b, color='k', lw=1.2, alpha=0.7)
                ax.axvline(b, color='k', lw=1.2, alpha=0.7)
            # Legend
            unique_groups = list(dict.fromkeys(groups_sorted))
            handles = [mpatches.Patch(color=cfg.group_colors.get(g, 'gray'), label=str(g))
                       for g in unique_groups if g is not None]
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.12),
                      fontsize=6, ncol=min(len(handles), 4), frameon=False)

        elif continuous:
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

            # Small gradient colorbar below the plot (using inset_axes so subplot isn't resized)
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cbar_ax_inset = inset_axes(ax, width="60%", height="4%",
                                       loc='lower center', borderpad=-4.5)
            sm = plt.cm.ScalarMappable(cmap=gradient_cmap,
                                       norm=plt.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cbar_factor = fig.colorbar(sm, cax=cbar_ax_inset, orientation='horizontal')
            cbar_factor.set_label(factor, fontsize=7)
            cbar_factor.ax.tick_params(labelsize=6)

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
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.12),
                      fontsize=6, ncol=min(len(handles), 4), frameon=False)

        ax.set_title(f"sorted by {factor}", fontsize=11)

    fig.suptitle(f"Neural RDM for {cfg.region} ({metric_label}) | {result['session']} | {win_str} | nor={cfg.normalization}",
                 fontsize=13, y=1.02)

    fig.tight_layout()
    fig.subplots_adjust(right=0.92, bottom=0.15)

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


def plot_variance_quartile_rdms(diag_results, cfg, sort_factors=None,
                                 save=True, save_dir=None):
    """
    Grid of neural RDM heatmaps: columns = variance quartiles,
    rows = sort factors.  Lets you visually inspect whether structural
    patterns (e.g. one group with high 1-r, age gradient) persist or
    vanish as you include lower-variance neurons.

    Parameters
    ----------
    diag_results : list of dict
        Output of variance_quartile_diagnostic.  Each entry must contain
        'quantile', 'n_neurons', 'neural_rdm', 'identities',
        'model_labels', 'model_rdms', 'comparisons'.
    cfg : RSAConfig
    sort_factors : list of str or None
        Which factors to sort by (one row each).  None → getattr(cfg, 'model_factors', []).
    """
    if sort_factors is None:
        sort_factors = getattr(cfg, 'model_factors', [])

    n_cols = len(diag_results)
    n_rows = len(sort_factors)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols + 1.5, 5 * n_rows + 1),
                             squeeze=False)

    metric_label = _metric_label(cfg.neural_metric)
    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"

    # Shared color scale across all panels
    all_rdms = [e['neural_rdm'] for e in diag_results]
    vmin = min(np.nanmin(r[np.triu_indices_from(r, k=1)]) for r in all_rdms)
    vmax = max(np.nanmax(r[np.triu_indices_from(r, k=1)]) for r in all_rdms)

    # Predefined color maps for known categorical factors
    categorical_color_maps = {
        'group': cfg.group_colors,
        'sex': {'F': '#e377c2', 'M': '#17becf'},
        'age_bin': {'adult': '#ff7f0e', 'juvenile': '#2ca02c'},
        'familiarity': {'familiar': '#d62728', 'unfamiliar': '#1f77b4'},
    }

    im = None  # will hold last imshow for shared colorbar

    for col_i, entry in enumerate(diag_results):
        q_pct = f"Top {entry['quantile']*100:.0f}%"
        n_neur = entry['n_neurons']
        rdm = entry['neural_rdm']

        for row_i, factor in enumerate(sort_factors):
            ax = axes[row_i, col_i]

            if factor not in entry['model_labels']:
                ax.set_title(f"{factor} (n/a)")
                ax.axis('off')
                continue

            # Reuse existing sort logic — entry has the same keys as a result dict
            sort_idx, sorted_labels, sorted_ids, groups_sorted = \
                _get_sort_for_factor(entry, factor, cfg)
            rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
            continuous = _is_continuous_factor(sorted_labels)

            im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal',
                           vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(sorted_ids)))
            ax.set_yticks(range(len(sorted_ids)))
            ax.set_xticklabels(sorted_ids, rotation=90, fontsize=5)
            ax.set_yticklabels(sorted_ids, fontsize=5)

            # ── Tick coloring & boundaries (same logic as plot_neural_rdm_multi_sort) ──
            if groups_sorted is not None:
                # by_group mode
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                  ax.get_yticklabels())):
                    g = groups_sorted[i]
                    if g in cfg.group_colors:
                        tx.set_color(cfg.group_colors[g])
                        ty.set_color(cfg.group_colors[g])
                boundaries = []
                prev = groups_sorted[0]
                for i, g in enumerate(groups_sorted):
                    if g != prev:
                        boundaries.append(i - 0.5)
                        prev = g
                for b in boundaries:
                    ax.axhline(b, color='k', lw=1, alpha=0.7)
                    ax.axvline(b, color='k', lw=1, alpha=0.7)

            elif continuous:
                valid_vals = [l for l in sorted_labels
                              if l is not None and not (isinstance(l, float) and np.isnan(l))]
                lo, hi = min(valid_vals), max(valid_vals)
                gradient_cmap = plt.cm.viridis
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                  ax.get_yticklabels())):
                    val = sorted_labels[i]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        normed = (val - lo) / (hi - lo) if hi > lo else 0.5
                        c = gradient_cmap(normed)
                    else:
                        c = 'gray'
                    tx.set_color(c)
                    ty.set_color(c)

            else:
                # Categorical
                cmap_cat = categorical_color_maps.get(factor, {})
                if not cmap_cat:
                    unique = sorted(set(str(l) for l in sorted_labels if l is not None))
                    colors = plt.cm.tab10.colors
                    cmap_cat = {l: colors[i % 10] for i, l in enumerate(unique)}
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                  ax.get_yticklabels())):
                    lab = sorted_labels[i]
                    c = cmap_cat.get(lab, cmap_cat.get(str(lab), 'k'))
                    tx.set_color(c)
                    ty.set_color(c)
                boundaries = []
                prev = sorted_labels[0]
                for i, lab in enumerate(sorted_labels):
                    if lab != prev:
                        boundaries.append(i - 0.5)
                        prev = lab
                for b in boundaries:
                    ax.axhline(b, color='k', lw=1, alpha=0.7)
                    ax.axvline(b, color='k', lw=1, alpha=0.7)

            # ── Annotate rho for this factor in the corner ──
            rho = entry['comparisons'].get(factor, {}).get('rho', np.nan)
            ax.text(0.02, 0.98, f"ρ={rho:+.3f}", transform=ax.transAxes,
                    fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            # Column header (top row only)
            if row_i == 0:
                ax.set_title(f"{q_pct}  (n={n_neur})", fontsize=10)

            # Row label (left column only)
            if col_i == 0:
                ax.set_ylabel(f"sorted by {factor}", fontsize=10)

    fig.suptitle(f"Variance-quartile diagnostic ({metric_label}) | {win_str}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(right=0.92, bottom=0.08)

    # Shared colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(metric_label)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"variance_quartile_rdms_{win_str}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig

def plot_random_subset_rdms(random_results, cfg, reference_entry=None,
                            sort_factors=None, save=True, save_dir=None):
    """
    Grid of neural RDM heatmaps for random neuron subsets, with an
    optional reference column (typically top-X% by variance from
    variance_quartile_diagnostic) shown leftmost for direct comparison.

    Columns: [reference] (optional) + random_draw_1 ... random_draw_N
    Rows:    sort factors (group, familiarity, sex, age_continuous, ...)

    Parameters
    ----------
    random_results : list of dict
        Output of random_subset_diagnostic.
    cfg : RSAConfig
    reference_entry : dict or None
        A single entry from variance_quartile_diagnostic (typically the
        top-25% quartile = diag[0]) shown as the leftmost column.
    sort_factors : list of str or None
        None → getattr(cfg, 'model_factors', []).
    """
    if sort_factors is None:
        sort_factors = getattr(cfg, 'model_factors', [])

    panels = []
    if reference_entry is not None:
        panels.append(('reference', reference_entry))
    for r in random_results:
        panels.append(('random', r))

    n_cols = len(panels)
    n_rows = len(sort_factors)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols + 1.5, 5 * n_rows + 1),
                             squeeze=False)

    metric_label = _metric_label(cfg.neural_metric)
    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"

    # Shared color scale across all panels
    all_rdms = [p[1]['neural_rdm'] for p in panels]
    vmin = min(np.nanmin(r[np.triu_indices_from(r, k=1)]) for r in all_rdms)
    vmax = max(np.nanmax(r[np.triu_indices_from(r, k=1)]) for r in all_rdms)

    categorical_color_maps = {
        'group': cfg.group_colors,
        'sex': {'F': '#e377c2', 'M': '#17becf'},
        'age_bin': {'adult': '#ff7f0e', 'juvenile': '#2ca02c'},
        'familiarity': {'familiar': '#d62728', 'unfamiliar': '#1f77b4'},
    }

    im = None

    for col_i, (panel_type, entry) in enumerate(panels):
        rdm = entry['neural_rdm']
        n_neur = entry['n_neurons']

        if panel_type == 'reference':
            q = entry.get('quantile', None)
            header = (f"Top {q*100:.0f}% by variance  (n={n_neur})"
                      if q is not None else f"Reference  (n={n_neur})")
        else:
            header = f"Random draw {entry['draw_index']+1}  (n={n_neur})"

        for row_i, factor in enumerate(sort_factors):
            ax = axes[row_i, col_i]

            if factor not in entry['model_labels']:
                ax.set_title(f"{factor} (n/a)")
                ax.axis('off')
                continue

            sort_idx, sorted_labels, sorted_ids, groups_sorted = \
                _get_sort_for_factor(entry, factor, cfg)
            rdm_sorted = rdm[np.ix_(sort_idx, sort_idx)]
            continuous = _is_continuous_factor(sorted_labels)

            im = ax.imshow(rdm_sorted, cmap='RdYlBu_r', aspect='equal',
                           vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(sorted_ids)))
            ax.set_yticks(range(len(sorted_ids)))
            ax.set_xticklabels(sorted_ids, rotation=90, fontsize=5)
            ax.set_yticklabels(sorted_ids, fontsize=5)

            # Tick coloring + boundary lines (mirrors plot_variance_quartile_rdms)
            if groups_sorted is not None:
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                 ax.get_yticklabels())):
                    g = groups_sorted[i]
                    if g in cfg.group_colors:
                        tx.set_color(cfg.group_colors[g])
                        ty.set_color(cfg.group_colors[g])
                boundaries = []
                prev = groups_sorted[0]
                for i, g in enumerate(groups_sorted):
                    if g != prev:
                        boundaries.append(i - 0.5)
                        prev = g
                for b in boundaries:
                    ax.axhline(b, color='k', lw=1, alpha=0.7)
                    ax.axvline(b, color='k', lw=1, alpha=0.7)

            elif continuous:
                valid_vals = [l for l in sorted_labels
                              if l is not None and not (isinstance(l, float) and np.isnan(l))]
                lo, hi = min(valid_vals), max(valid_vals)
                gradient_cmap = plt.cm.viridis
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                 ax.get_yticklabels())):
                    val = sorted_labels[i]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        normed = (val - lo) / (hi - lo) if hi > lo else 0.5
                        c = gradient_cmap(normed)
                    else:
                        c = 'gray'
                    tx.set_color(c)
                    ty.set_color(c)

            else:
                cmap_cat = categorical_color_maps.get(factor, {})
                if not cmap_cat:
                    unique = sorted(set(str(l) for l in sorted_labels if l is not None))
                    colors = plt.cm.tab10.colors
                    cmap_cat = {l: colors[i % 10] for i, l in enumerate(unique)}
                for i, (tx, ty) in enumerate(zip(ax.get_xticklabels(),
                                                 ax.get_yticklabels())):
                    lab = sorted_labels[i]
                    c = cmap_cat.get(lab, cmap_cat.get(str(lab), 'k'))
                    tx.set_color(c)
                    ty.set_color(c)
                boundaries = []
                prev = sorted_labels[0]
                for i, lab in enumerate(sorted_labels):
                    if lab != prev:
                        boundaries.append(i - 0.5)
                        prev = lab
                for b in boundaries:
                    ax.axhline(b, color='k', lw=1, alpha=0.7)
                    ax.axvline(b, color='k', lw=1, alpha=0.7)

            # rho annotation
            rho = entry['comparisons'].get(factor, {}).get('rho', np.nan)
            ax.text(0.02, 0.98, f"ρ={rho:+.3f}", transform=ax.transAxes,
                    fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            if row_i == 0:
                ax.set_title(header, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"sorted by {factor}", fontsize=10)

    fig.suptitle(f"Random-subset diagnostic ({metric_label}) | {win_str}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(right=0.92, bottom=0.08)

    if im is not None:
        cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(metric_label)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"random_subset_rdms_{win_str}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig


def plot_containment_metric(containment_results, cfg, save=True, save_dir=None):
    """
    Line plot: containment metric across variance quartiles, one line per group.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    groups = list(containment_results[0]['groups'].keys())
    quartiles = [r['quantile'] for r in containment_results]
    x_labels = [f"Top {q*100:.0f}%" for q in quartiles]

    # Panel 1: containment (within - between)
    ax = axes[0]
    for g in groups:
        vals = [r['groups'][g]['containment'] for r in containment_results]
        ax.plot(quartiles, vals, 'o-', color=cfg.group_colors.get(g, 'gray'),
                label=g, lw=2, markersize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_ylabel('within mean − between mean  (1-r)')
    ax.set_title('Containment')
    ax.legend(fontsize=8)
    ax.set_xticks(quartiles)
    ax.set_xticklabels(x_labels)

    # Panel 2: within-group mean
    ax = axes[1]
    for g in groups:
        vals = [r['groups'][g]['within_mean'] for r in containment_results]
        ax.plot(quartiles, vals, 'o-', color=cfg.group_colors.get(g, 'gray'),
                label=g, lw=2, markersize=8)
    ax.set_title('Within-group mean 1-r')
    ax.set_xticks(quartiles)
    ax.set_xticklabels(x_labels)

    # Panel 3: between-group mean
    ax = axes[2]
    for g in groups:
        vals = [r['groups'][g]['between_mean'] for r in containment_results]
        ax.plot(quartiles, vals, 'o-', color=cfg.group_colors.get(g, 'gray'),
                label=g, lw=2, markersize=8)
    ax.set_title('Between-group mean 1-r')
    ax.set_xticks(quartiles)
    ax.set_xticklabels(x_labels)

    win_str = f"{int(cfg.window[0]*1000)}-{int(cfg.window[1]*1000)}ms"
    norm_str = cfg.normalization or 'no_normalization'
    fig.suptitle(f'Containment metric | {cfg.region} | {norm_str} | {win_str}',
                 fontsize=13)
    fig.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"containment_metric_{norm_str}_{win_str}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig