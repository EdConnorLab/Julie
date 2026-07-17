# noise_ceiling_pc_pipeline.py
"""
Split-half noise ceiling for the NEURAL RDM used by the social-PC geometry RSA
(social_pc_predictions.py::permutation_test_geometry).

That pipeline builds the neural RDM as:
    X_sc = StandardScaler().fit_transform(pop_matrix)   # per-neuron z-score over identities
    neural_for_rsa = X_sc            (N_NEURAL_PCS_RSA=None, full space)
                   = PCA(k).fit_transform(X_sc)   (if N_NEURAL_PCS_RSA=k)
    d_neural = pdist(neural_for_rsa, metric='euclidean')
and correlates d_neural with the top-2 social-PC distances via Spearman.

This script asks: how reproducible is d_neural across independent halves of the
trials?  The Spearman-Brown-corrected split-half reliability gives a noise
ceiling = sqrt(r_full): the largest |Spearman rho| the neural RDM could show
against ANY model (including a perfect one). Compare the observed RSA rho from
social_pc_predictions.py against this ceiling:
    obs_rho ~ ceiling  -> model ~ as good as the (noisy) data permits (power statement)
    obs_rho << ceiling -> reproducible structure the model is missing
    ceiling ~ 0        -> the neural RDM is noise at this trial count; the RSA rho
                          (significant or not) is not interpretable.

NOTE: this mirrors the PC-pipeline neural-RDM CONSTRUCTION but builds the
pseudo-population directly from the spike cache (identities present in every
session, session-dropped at MIN_REPS). If your build_population_vectors uses a
different neuron/identity set, align REGION/WINDOW/MIN_REPS/GROUP below to match
the run that produced the significant rho.
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config (edit to match the run that produced the significant rho) ──────────
DATA_PATH = '/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered'
INFO_PATH = '/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv'
GROUP        = 'Zombies'
SUBJECT      = '81G'            # dropped (subject, never a stimulus) — matches SocialEncodingConfig
REGIONS      = ['AMG', 'ER']
WINDOW       = (0.300, 0.500)
MIN_EPOCH    = 1.3
MIN_REPS     = 3               # per-session floor; matches SocialEncodingConfig.min_reps_per_monkey
N_NEURAL_PCS_RSA = [None, 5]   # None = full standardized space (pipeline default); ints add sensitivity
N_SPLITS     = 1000
SEED         = 42
OUT_DIR      = 'noise_ceiling_pc_pipeline'


def load_region(region):
    fr = [pd.read_pickle(f) for f in sorted(glob.glob(f'{DATA_PATH}/*.pkl'))]
    df = pd.concat(fr, ignore_index=True)
    df = df[df['MonkeyName'] != 'NewMonkey'].dropna(subset=['MonkeyGroup']).copy()
    df['session'] = df['Date'].astype(str) + '_' + df['Round No.'].astype(str)
    df = df[df['Location'] == region]
    df['dur'] = df['EpochStartStop'].map(lambda x: x[1] - x[0])
    df = df[df['dur'] >= MIN_EPOCH]
    df['MonkeyName'] = df['MonkeyName'].astype(str)
    # per-trial (per-row) firing rate in the window
    es = df['EpochStartStop'].map(lambda x: x[0]).to_numpy()
    sp = df['SpikeTimes'].to_numpy(dtype=object)
    cnt = np.empty(len(df))
    for i, (s, t0) in enumerate(zip(sp, es)):
        a = np.asarray(s)
        cnt[i] = np.count_nonzero((a >= t0 + WINDOW[0]) & (a < t0 + WINDOW[1]))
    df['rate'] = cnt / (WINDOW[1] - WINDOW[0])
    return df


def build_cells(df, identities):
    """(neuron=(session,NeuronID), identity) -> array of per-trial rates,
    over sessions where every identity has >= MIN_REPS trials."""
    sessions = sorted(df['session'].unique())
    def ntr(s, m):
        return df[(df.session == s) & (df.MonkeyName == m)]['TaskField'].nunique()
    kept = [s for s in sessions if all(ntr(s, m) >= MIN_REPS for m in identities)]
    dfk = df[df.session.isin(kept)]
    cells = {}
    for (s, nid, mk), g in dfk.groupby(['session', 'NeuronID', 'MonkeyName']):
        cells[((s, nid), mk)] = g['rate'].to_numpy()
    neurons = sorted({k[0] for k in cells}, key=lambda x: (str(x[0]), str(x[1])))
    return cells, neurons, len(sessions), len(kept)


def neuron_set(cells, neurons, identities, min_trials):
    return [nk for nk in neurons
            if all(len(cells.get((nk, m), [])) >= min_trials for m in identities)]


def neural_rdm_vec(pop, n_pcs):
    """pop: (n_identities, n_neurons) mean matrix -> condensed euclidean RDM,
    mirroring permutation_test_geometry (StandardScaler [+ optional PCA])."""
    Xsc = StandardScaler().fit_transform(pop)
    if n_pcs is None:
        Z = Xsc
    else:
        k = min(n_pcs, pop.shape[0] - 1, pop.shape[1])
        Z = PCA(n_components=k).fit_transform(Xsc)
    return pdist(Z, metric='euclidean')


def reliability(cells, neurons, identities, n_pcs, n_splits=N_SPLITS, seed=SEED):
    n_id, n_ne = len(identities), len(neurons)
    arrs = [[cells[(nk, m)] for m in identities] for nk in neurons]
    rng = np.random.default_rng(seed)
    r_split = np.empty(n_splits)
    for s in range(n_splits):
        A = np.empty((n_id, n_ne)); B = np.empty((n_id, n_ne))
        for k in range(n_ne):
            for i in range(n_id):
                r = arrs[k][i]; nn = len(r)
                p = rng.permutation(nn); h = nn // 2
                A[i, k] = r[p[:h]].mean()
                B[i, k] = r[p[h:]].mean()
        dA = neural_rdm_vec(A, n_pcs)
        dB = neural_rdm_vec(B, n_pcs)
        r_split[s], _ = spearmanr(dA, dB)
    r_half = float(np.nanmean(r_split))
    lo, hi = np.nanpercentile(r_split, [2.5, 97.5])
    r_full = 2 * r_half / (1 + r_half)
    ceiling = float(np.sqrt(r_full)) if r_full > 0 else float('nan')
    return dict(r_half=r_half, ci=(float(lo), float(hi)), r_full=r_full,
                ceiling=ceiling, dist=r_split, n=n_ne)


def main():
    info = pd.read_csv(INFO_PATH); info['Name'] = info['Name'].astype(str)
    members = set(info.loc[info['Group Name'] == GROUP, 'Name']) - {SUBJECT}
    os.makedirs(OUT_DIR, exist_ok=True)
    hist = {}
    for region in REGIONS:
        df = load_region(region)
        df = df[df['MonkeyName'].isin(members)].reset_index(drop=True)
        identities = sorted(members & set(df['MonkeyName']))
        cells, neurons, n_sess, n_kept = build_cells(df, identities)
        perm = neuron_set(cells, neurons, identities, 2)
        rest = neuron_set(cells, neurons, identities, 6)
        print(f"\n{'='*72}\n{region}: {len(identities)} identities, "
              f"{n_kept}/{n_sess} sessions kept, {len(neurons)} pseudopop neurons "
              f"(permissive>=2: {len(perm)}, restrictive>=6: {len(rest)})")
        for n_pcs in N_NEURAL_PCS_RSA:
            lab = 'full' if n_pcs is None else f'{n_pcs}PC'
            for setname, nset in [('permissive', perm), ('restrictive', rest)]:
                if len(nset) < 3:
                    continue
                res = reliability(cells, nset, identities, n_pcs)
                print(f"  [{lab:>4}] {setname:11s} n={res['n']:3d}  "
                      f"r_half={res['r_half']:+.3f} "
                      f"(95% {res['ci'][0]:+.2f},{res['ci'][1]:+.2f})  "
                      f"r_full={res['r_full']:+.3f}  ceiling={res['ceiling']:.3f}")
                if n_pcs is None and setname == 'permissive':
                    hist[region] = res

    fig, axes = plt.subplots(1, len(REGIONS), figsize=(6 * len(REGIONS), 4.5),
                             squeeze=False)
    for ax, region in zip(axes[0], REGIONS):
        res = hist.get(region)
        if res is None:
            continue
        ax.hist(res['dist'], bins=40, color='indianred', edgecolor='k', alpha=0.85)
        ax.axvline(res['r_half'], color='k', lw=2, label=f"r_half={res['r_half']:.3f}")
        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f"{region}: PC-pipeline neural RDM split-half\n"
                     f"(full space, ceiling={res['ceiling']:.3f}, n={res['n']})",
                     fontsize=10)
        ax.set_xlabel('Spearman r (RDM_A vs RDM_B)'); ax.set_ylabel('# splits')
        ax.legend(fontsize=8)
    fig.suptitle('Neural RDM reliability — social-PC (Euclidean/StandardScaler) construction',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'noise_ceiling_pc_pipeline.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nsaved {OUT_DIR}/noise_ceiling_pc_pipeline.png")


if __name__ == '__main__':
    main()
