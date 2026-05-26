# plotting.py
"""
Plotting for trajectory analysis.

Row metadata contract:
    row_meta_df must have columns 'condition' and 'time_bin' (+ 'time_bin_start_s').
    Optionally a 'group' column for sub-grouping conditions into colored families
    (e.g. condition=MonkeyName, group=MonkeyGroup). If 'group' is absent, each
    condition is treated as its own group.

info contract:
    info['conditions'] : ordered list of condition labels
    info['n_bins']     : int
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

PLOT_SAVE_DIR = '/analysis_results/state_space_trajectory'


def _save_mpl(fig, name, save, save_dir):
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{name}.png'),
                    dpi=200, bbox_inches='tight')


def _save_plotly(fig, name, save, save_dir):
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.write_html(os.path.join(save_dir, f'{name}.html'))


def _safe(s):
    return str(s).replace(' ', '_').lower()


def _clean_scene(var):
    axis = dict(showbackground=True, backgroundcolor='rgb(240,240,240)',
                gridcolor='rgb(120,120,120)', gridwidth=1,
                zerolinecolor='rgb(80,80,80)', showspikes=False)
    return dict(
        xaxis=dict(title=f'PC1 ({var[0]:.1%})', **axis),
        yaxis=dict(title=f'PC2 ({var[1]:.1%})', **axis),
        zaxis=dict(title=f'PC3 ({var[2]:.1%})', **axis),
    )


def _time_axis(row_meta_df, info):
    n_bins = info['n_bins']
    first = row_meta_df.iloc[:n_bins]
    if 'time_from_onset_s' in first.columns:
        return first['time_from_onset_s'].values
    return first['time_bin_start_s'].values


def _smooth(traj, sigma):
    return np.column_stack([gaussian_filter1d(traj[:, i], sigma=sigma)
                            for i in range(traj.shape[1])])


# ── Trajectory computation ──────────────────────────────────────────────────
def compute_trajectories(pca_result, row_meta_df, info, cfg):
    """
    Returns
    -------
    cond_trajs  : dict {condition: (n_bins, n_comp)}   one trajectory per condition
    group_trajs : dict {group: (n_bins, n_comp)}       mean across conditions in each group
    cond_to_group : dict {condition: group}            identity if no 'group' col
    """
    scores_3d = pca_result['scores_3d']      # (n_conditions, n_bins, n_comp)
    n_bins = info['n_bins']

    # One row per condition from row_meta_df (take first bin of each)
    cond_meta = row_meta_df.iloc[::n_bins].reset_index(drop=True)

    if 'group' in cond_meta.columns:
        cond_to_group = dict(zip(cond_meta['condition'], cond_meta['group']))
    else:
        cond_to_group = {c: c for c in info['conditions']}

    cond_trajs = {}
    for i, c in enumerate(info['conditions']):
        cond_trajs[c] = _smooth(scores_3d[i], cfg.smoothing_sigma)

    group_trajs = {}
    groups = sorted(set(cond_to_group.values()), key=str)
    for g in groups:
        members = [c for c in info['conditions'] if cond_to_group[c] == g]
        stacked = np.stack([cond_trajs[c] for c in members], axis=0)
        group_trajs[g] = stacked.mean(axis=0)

    return cond_trajs, group_trajs, cond_to_group


# ── Color utilities ─────────────────────────────────────────────────────────
def _conditions_by_group(c2g):
    out = {}
    for c, g in c2g.items():
        out.setdefault(g, []).append(c)
    return out


def _distinct_condition_colors(conditions):
    palette = [plt.get_cmap('tab20')(i) for i in range(20)] + \
              [plt.get_cmap('tab20b')(i) for i in range(20)]
    return {c: palette[i % len(palette)] for i, c in enumerate(conditions)}


def _shaded_group_colors(c2g, cfg):
    groups = _conditions_by_group(c2g)
    colors = {}
    for g, conds in groups.items():
        cmap = plt.get_cmap(cfg.group_cmaps.get(g, 'Greys'))
        n = len(conds)
        for i, c in enumerate(conds):
            colors[c] = cmap(0.3 + 0.6 * (i / max(n - 1, 1)))
    return colors

_AUTO_GROUP_COLORS = {}

def _group_color(cfg, g):
    if g in cfg.group_colors:
        return cfg.group_colors[g]
    if g not in _AUTO_GROUP_COLORS:
        palette = plt.get_cmap('tab10').colors
        _AUTO_GROUP_COLORS[g] = palette[len(_AUTO_GROUP_COLORS) % len(palette)]
    return _AUTO_GROUP_COLORS[g]


def _rgba_str(c):
    a = c[3] if len(c) > 3 else 1
    return f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{a})'


def _add_plotly_endpoints(fig, traj, color, label, legendgroup=None):
    import plotly.graph_objects as go
    for idx, sym, tag in [(0, 'circle', 'start'), (-1, 'diamond', 'end')]:
        fig.add_trace(go.Scatter3d(
            x=[traj[idx, 0]], y=[traj[idx, 1]], z=[traj[idx, 2]],
            mode='markers',
            marker=dict(size=6, symbol=sym, color=color,
                        line=dict(width=1, color='black')),
            legendgroup=legendgroup, showlegend=False,
            hovertext=f'{label} — {tag}', hoverinfo='text'))


# Group-mean plots: One trajectory per group
def plot_group_mean_3d_mpl(group_trajs, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for g, traj in group_trajs.items():
        c = _group_color(cfg, g)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, linewidth=2.5, label=str(g))
        ax.scatter(*traj[0, :3], color=c, s=100, marker='o', edgecolors='black')
        ax.scatter(*traj[-1, :3], color=c, s=100, marker='s', edgecolors='black')
    ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_zlabel(f'PC3 ({var[2]:.1%})')
    ax.set_title(f'Group-mean (3D) {suffix}'); ax.legend(); ax.view_init(25, 135)
    fig.tight_layout()
    _save_mpl(fig, f'group_mean_3d_{_safe(suffix)}', save, save_dir)
    return fig


def plot_group_mean_2d_mpl(group_trajs, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    fig, ax = plt.subplots(figsize=(9, 7))
    for g, traj in group_trajs.items():
        c = _group_color(cfg, g)
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.5, label=str(g))
        ax.scatter(traj[0, 0], traj[0, 1], color=c, s=100, marker='o', edgecolors='black')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=c, s=100, marker='s', edgecolors='black')
    ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_title(f'Group-mean (2D) {suffix}'); ax.legend()
    fig.tight_layout()
    _save_mpl(fig, f'group_mean_2d_{_safe(suffix)}', save, save_dir)
    return fig


def plot_group_mean_3d_plotly(group_trajs, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    import plotly.graph_objects as go
    fig = go.Figure()
    for g, traj in group_trajs.items():
        c = _group_color(cfg, g)
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(width=5, color=c),
            name=str(g), hoverinfo='name'))
        _add_plotly_endpoints(fig, traj, c, str(g), legendgroup=str(g))
    fig.update_layout(title=f'Group-mean {suffix}', scene=_clean_scene(var),
                      width=1100, height=800)
    _save_plotly(fig, f'group_mean_3d_plotly_{_safe(suffix)}', save, save_dir)
    return fig

def plot_pc_vs_time_group_mean(group_trajs, var, cfg, row_meta_df, info,
                               pcs=(1, 2, 3), suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    """One figure with a subplot per PC. pcs is 1-indexed."""
    t = _time_axis(row_meta_df, info)
    n = len(pcs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    for i, pc in enumerate(pcs):
        ax = axes.flat[i]
        idx = pc - 1
        for g, traj in group_trajs.items():
            c = _group_color(cfg, g)
            ax.plot(t, traj[:, idx], color=c, linewidth=2, label=str(g))
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(f'PC{pc} ({var[idx]:.1%})')
        ax.set_xlabel('Time (s)')
    for j in range(n, nrows * ncols):
        axes.flat[j].axis('off')
    axes.flat[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'PC vs time — group-mean {suffix}')
    fig.tight_layout()
    _save_mpl(fig, f'pc_vs_time_group_mean_{_safe(suffix)}', save, save_dir)
    return fig


# Per-group plots (individual conditions within a group): Only useful when there's sub-grouping such as identity
def plot_per_group_3d_mpl(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    figs = {}
    for g, conds in _conditions_by_group(c2g).items():
        colors = _distinct_condition_colors(conds)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        for c in conds:
            t = cond_trajs[c]; col = colors[c]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, linewidth=2, label=str(c))
            ax.scatter(*t[0, :3], color=col, s=60, marker='o', edgecolors='black')
            ax.scatter(*t[-1, :3], color=col, s=60, marker='s', edgecolors='black')
        ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_zlabel(f'PC3 ({var[2]:.1%})')
        ax.set_title(f'{g} — conditions (3D) {suffix}')
        ax.legend(fontsize=7, ncol=2); ax.view_init(25, 135)
        fig.tight_layout()
        _save_mpl(fig, f'per_group_3d_{_safe(g)}_{_safe(suffix)}', save, save_dir)
        figs[g] = fig
    return figs



def plot_pc_vs_time_per_group(cond_trajs, c2g, var, cfg, row_meta_df, info,
                              pcs=(1, 2, 3), suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    """One figure per group; subplots are PCs."""
    t = _time_axis(row_meta_df, info)
    n = len(pcs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    figs = {}
    for g, conds in _conditions_by_group(c2g).items():
        colors = _distinct_condition_colors(conds)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows),
                                 sharex=True, squeeze=False)
        for i, pc in enumerate(pcs):
            ax = axes.flat[i]
            idx = pc - 1
            for c in conds:
                ax.plot(t, cond_trajs[c][:, idx],
                        color=colors[c], linewidth=1.5, label=str(c))
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_title(f'PC{pc} ({var[idx]:.1%})')
            ax.set_xlabel('Time (s)')
        for j in range(n, nrows * ncols):
            axes.flat[j].axis('off')
        axes.flat[0].legend(fontsize=6, ncol=2, loc='best')
        fig.suptitle(f'PC vs time — {g} {suffix}')
        fig.tight_layout()
        _save_mpl(fig, f'pc_vs_time_{_safe(g)}_{_safe(suffix)}', save, save_dir)
        figs[g] = fig
    return figs

def plot_per_group_2d_mpl(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    figs = {}
    for g, conds in _conditions_by_group(c2g).items():
        colors = _distinct_condition_colors(conds)
        fig, ax = plt.subplots(figsize=(9, 7))
        for c in conds:
            t = cond_trajs[c]; col = colors[c]
            ax.plot(t[:, 0], t[:, 1], color=col, linewidth=2, label=str(c))
            ax.scatter(t[0, 0], t[0, 1], color=col, s=60, marker='o', edgecolors='black')
            ax.scatter(t[-1, 0], t[-1, 1], color=col, s=60, marker='s', edgecolors='black')
        ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_title(f'{g} — conditions (2D) {suffix}'); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        _save_mpl(fig, f'per_group_2d_{_safe(g)}_{_safe(suffix)}', save, save_dir)
        figs[g] = fig
    return figs


def plot_per_group_3d_plotly(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    import plotly.graph_objects as go
    figs = {}
    for g, conds in _conditions_by_group(c2g).items():
        colors = _distinct_condition_colors(conds)
        fig = go.Figure()
        for c in conds:
            t = cond_trajs[c]
            col = _rgba_str(colors[c])
            fig.add_trace(go.Scatter3d(
                x=t[:, 0], y=t[:, 1], z=t[:, 2],
                mode='lines', line=dict(width=4, color=col),
                name=str(c), hoverinfo='name', legendgroup=str(c)))
            _add_plotly_endpoints(fig, t, col, str(c), legendgroup=str(c))
        fig.update_layout(title=f'{g} — conditions {suffix}',
                          scene=_clean_scene(var), width=1100, height=800)
        _save_plotly(fig, f'per_group_3d_plotly_{_safe(g)}_{_safe(suffix)}', save, save_dir)
        figs[g] = fig
    return figs



# All conditions shaded by group
def plot_all_shaded_3d_mpl(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    colors = _shaded_group_colors(c2g, cfg)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for c, t in cond_trajs.items():
        col = colors[c]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, linewidth=1.5, alpha=0.7)
        ax.scatter(*t[0, :3], color=col, s=40, marker='o', edgecolors='black')
        ax.scatter(*t[-1, :3], color=col, s=40, marker='s', edgecolors='black')
    handles = [Line2D([0], [0],
               color=plt.get_cmap(cfg.group_cmaps.get(g, 'Greys'))(0.6),
               linewidth=3, label=str(g)) for g in sorted(set(c2g.values()), key=str)]
    ax.legend(handles=handles)
    ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_zlabel(f'PC3 ({var[2]:.1%})')
    ax.set_title(f'All conditions shaded by group (3D) {suffix}')
    ax.view_init(25, 135); fig.tight_layout()
    _save_mpl(fig, f'all_shaded_3d_{_safe(suffix)}', save, save_dir)
    return fig


def plot_all_shaded_2d_mpl(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    colors = _shaded_group_colors(c2g, cfg)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c, t in cond_trajs.items():
        col = colors[c]
        ax.plot(t[:, 0], t[:, 1], color=col, linewidth=1.5, alpha=0.7)
        ax.scatter(t[0, 0], t[0, 1], color=col, s=40, marker='o', edgecolors='black')
        ax.scatter(t[-1, 0], t[-1, 1], color=col, s=40, marker='s', edgecolors='black')
    handles = [Line2D([0], [0],
               color=plt.get_cmap(cfg.group_cmaps.get(g, 'Greys'))(0.6),
               linewidth=3, label=str(g)) for g in sorted(set(c2g.values()), key=str)]
    ax.legend(handles=handles)
    ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_title(f'All conditions shaded by group (2D) {suffix}')
    fig.tight_layout()
    _save_mpl(fig, f'all_shaded_2d_{_safe(suffix)}', save, save_dir)
    return fig


def plot_all_shaded_3d_plotly(cond_trajs, c2g, var, cfg, suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    import plotly.graph_objects as go
    colors = _shaded_group_colors(c2g, cfg)
    fig = go.Figure()
    for g, conds in _conditions_by_group(c2g).items():
        for c in conds:
            t = cond_trajs[c]
            col = _rgba_str(colors[c])
            fig.add_trace(go.Scatter3d(
                x=t[:, 0], y=t[:, 1], z=t[:, 2],
                mode='lines', line=dict(width=3, color=col),
                name=str(c), hoverinfo='name',
                legendgroup=str(g), legendgrouptitle_text=str(g)))
            _add_plotly_endpoints(fig, t, col, str(c), legendgroup=str(g))
    fig.update_layout(title=f'All conditions shaded by group {suffix}',
                      scene=_clean_scene(var), width=1200, height=900)
    _save_plotly(fig, f'all_shaded_3d_plotly_{_safe(suffix)}', save, save_dir)
    return fig



def plot_pc_vs_time_all_shaded(cond_trajs, c2g, var, cfg, row_meta_df, info,
                               pcs=(1, 2, 3), suffix='', save=False, save_dir=PLOT_SAVE_DIR):
    """All conditions on one figure, shaded by group; subplots are PCs."""
    t = _time_axis(row_meta_df, info)
    colors = _shaded_group_colors(c2g, cfg)
    n = len(pcs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    for i, pc in enumerate(pcs):
        ax = axes.flat[i]
        idx = pc - 1
        for c, traj in cond_trajs.items():
            ax.plot(t, traj[:, idx], color=colors[c], linewidth=1.2, alpha=0.75)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(f'PC{pc} ({var[idx]:.1%})')
        ax.set_xlabel('Time (s)')
    for j in range(n, nrows * ncols):
        axes.flat[j].axis('off')
    handles = [Line2D([0], [0],
               color=plt.get_cmap(cfg.group_cmaps.get(g, 'Greys'))(0.6),
               linewidth=3, label=str(g)) for g in sorted(set(c2g.values()), key=str)]
    axes.flat[0].legend(handles=handles, fontsize=8, loc='best')
    fig.suptitle(f'PC vs time — all conditions {suffix}')
    fig.tight_layout()
    _save_mpl(fig, f'pc_vs_time_all_shaded_{_safe(suffix)}', save, save_dir)
    return fig


def plot_psth_group_mean(raw_matrix, preproc_matrix, row_meta_df, info, cfg,
                         smooth_sigma_s=0.050, suffix='', save=False,
                         save_dir=PLOT_SAVE_DIR):
    """
    PSTH: population-averaged firing rate vs time, one trace per condition/group.

    Parameters
    ----------
    raw_matrix : ndarray (n_conditions * n_bins, n_neurons)
        Pre-preprocessing mean spike counts per bin (output of build_matrix_by_condition).
    preproc_matrix : ndarray, same shape
        Post-preprocessing matrix (soft-norm / mean-center applied).
    row_meta_df : DataFrame with 'condition' and 'time_bin' columns.
    info : dict with 'conditions', 'n_bins', 'bin_width'.
    cfg : TrajectoryConfig (uses cfg.group_cmaps if present for colors).
    smooth_sigma_s : float or None
        Gaussian smoothing SD in seconds along time axis. None/0 disables.

    Plots two panels side by side:
        (left)  raw firing rate (Hz), mean ± SEM across neurons
        (right) preprocessed units, mean ± SEM across neurons
    """
    conditions = list(info['conditions'])
    n_bins = info['n_bins']
    bin_width = info['bin_width']
    n_cond = len(conditions)

    def _reshape(mat):
        # (n_cond * n_bins, n_neurons) -> (n_cond, n_bins, n_neurons),
        # respecting the row order in row_meta_df.
        n_neurons = mat.shape[1]
        out = np.full((n_cond, n_bins, n_neurons), np.nan)
        cond_to_idx = {c: i for i, c in enumerate(conditions)}
        for row_idx, (c, b) in enumerate(zip(row_meta_df['condition'],
                                             row_meta_df['time_bin'])):
            out[cond_to_idx[c], int(b), :] = mat[row_idx]
        return out

    raw_3d = _reshape(raw_matrix) / bin_width  # spike counts/bin -> Hz
    pre_3d = _reshape(preproc_matrix)

    # Optional smoothing along time axis (axis=1)
    if smooth_sigma_s and smooth_sigma_s > 0:
        sigma_bins = smooth_sigma_s / bin_width
        raw_3d = gaussian_filter1d(raw_3d, sigma=sigma_bins, axis=1)
        pre_3d = gaussian_filter1d(pre_3d, sigma=sigma_bins, axis=1)

    # Population mean and SEM across neurons
    def _mean_sem(arr3d):
        mean = arr3d.mean(axis=2)  # (n_cond, n_bins)
        sem = arr3d.std(axis=2, ddof=1) / np.sqrt(arr3d.shape[2])  # (n_cond, n_bins)
        return mean, sem

    raw_m, raw_s = _mean_sem(raw_3d)
    pre_m, pre_s = _mean_sem(pre_3d)

    t = np.arange(n_bins) * bin_width

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # Pick a color per condition: prefer group_cmaps if available
    cmaps = getattr(cfg, 'group_cmaps', {}) or {}

    def _color(cond, i):
        name = cmaps.get(cond)
        if name is None:
            return plt.get_cmap('tab10')(i % 10)
        return plt.get_cmap(name)(0.6)

    for ax, (mean, sem, ylabel, title) in zip(
            axes,
            [(raw_m, raw_s, 'Firing rate (Hz)', 'Raw PSTH'),
             (pre_m, pre_s, 'Preprocessed (a.u.)', 'Preprocessed PSTH')]):
        for i, c in enumerate(conditions):
            col = _color(c, i)
            ax.plot(t, mean[i], color=col, lw=2, label=str(c))
            ax.fill_between(t, mean[i] - sem[i], mean[i] + sem[i],
                            color=col, alpha=0.2, linewidth=0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    axes[0].legend(fontsize=8, loc='best', frameon=False)

    smooth_txt = f' (σ={int(smooth_sigma_s * 1000)} ms)' if smooth_sigma_s else ''
    fig.suptitle(f'PSTH — population mean ± SEM across {raw_3d.shape[2]} neurons'
                 f'{smooth_txt}  {suffix}')
    fig.tight_layout()
    _save_mpl(fig, f'psth_group_mean_{_safe(suffix)}', save, save_dir)
    return fig