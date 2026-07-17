# social_pc_vs_rank.py
"""
Is the "social PC" really a rank / age / sex readout?

Rebuilds the social PCs EXACTLY as social_pc_predictions.py does
(load_social_features -> drop subject -> reduce_features, full_profile, PCA=2)
and correlates PC1/PC2 with the monkeyinfo Rank, Age, and Sex. Also reports
Rank~Age and the Sex split, so you can trace the alternative explanation:
if social_PC1 ~ Rank, and Rank ~ Age/Sex (both visually available on the
stimulus), the "social" PC RSA/decoding result may be a rank or perceptual
readout rather than a social-relationship one.

Run from the mixed_effects_claude directory (same as social_pc_predictions.py),
with src/ on PYTHONPATH.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

from social_encoding_config import SocialEncodingConfig
from social_encoding_analysis import load_social_features, reduce_features

GROUPS = ['Zombies']        # add 'Best Frans', 'Instigators' if wanted
OUT_DIR = 'social_pc_vs_rank'


def analyze_group(group, cfg, info):
    print(f"\n{'='*72}\nGROUP: {group}\n{'='*72}")
    features = load_social_features(group, cfg)          # rows = monkeys, cols = behavior profile
    if cfg.subject_name in features.index:
        features = features.drop(cfg.subject_name)
    features_pc, pca, scaler = reduce_features(features, n_components=cfg.n_feature_pcs)
    # reduce_features already prints variance-explained and loadings

    # align with monkeyinfo
    meta = info.set_index('Name')
    df = features_pc.copy()
    for col in ['Rank', 'Age', 'Sex']:
        df[col] = [meta.loc[m, col] if m in meta.index else np.nan for m in df.index]
    df = df.dropna(subset=['Rank'])
    print(f"\n  n = {len(df)} monkeys with rank info")
    print(df.round(3).to_string())

    pcs = [c for c in df.columns if c.startswith('social_PC')]
    print(f"\n  Spearman correlations (social PC vs metadata), n={len(df)}:")
    for pc in pcs:
        for var in ['Rank', 'Age']:
            v = pd.to_numeric(df[var], errors='coerce')
            m = v.notna()
            if m.sum() >= 3:
                rho, p = spearmanr(df.loc[m, pc], v[m])
                flag = '  <-- strong' if abs(rho) >= 0.7 and p < 0.05 else ''
                print(f"    {pc} vs {var:5s}: rho={rho:+.3f}  p={p:.3f}{flag}")
        # Sex: Mann-Whitney U of PC across sexes
        sexes = df['Sex'].dropna().unique()
        if len(sexes) == 2:
            g0 = df.loc[df['Sex'] == sexes[0], pc]
            g1 = df.loc[df['Sex'] == sexes[1], pc]
            if len(g0) >= 1 and len(g1) >= 1:
                try:
                    u, p = mannwhitneyu(g0, g1, alternative='two-sided')
                    print(f"    {pc} vs Sex ({sexes[0]} n={len(g0)} / "
                          f"{sexes[1]} n={len(g1)}): MWU p={p:.3f} "
                          f"(median {sexes[0]}={g0.median():+.2f}, {sexes[1]}={g1.median():+.2f})")
                except ValueError:
                    pass

    # the alternative-explanation chain: Rank ~ Age, and Sex composition
    rank = pd.to_numeric(df['Rank'], errors='coerce')
    age = pd.to_numeric(df['Age'], errors='coerce')
    m = rank.notna() & age.notna()
    if m.sum() >= 3:
        rho, p = spearmanr(rank[m], age[m])
        print(f"\n  Rank vs Age: rho={rho:+.3f}  p={p:.3f}")
    print(f"  Sex composition: {df['Sex'].value_counts().to_dict()}")

    # scatter PC1 vs Rank, coloured by sex
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sex_colors = {'M': '#1f77b4', 'F': '#d62728'}
    for _, r in df.iterrows():
        ax.scatter(r['Rank'], r['social_PC1'],
                   c=sex_colors.get(r['Sex'], 'gray'), s=90, edgecolors='k', zorder=5)
        ax.annotate(r.name, (r['Rank'], r['social_PC1']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    rho, p = spearmanr(df['social_PC1'], pd.to_numeric(df['Rank'], errors='coerce'))
    ax.set_xlabel('Rank (David\'s score)'); ax.set_ylabel('social_PC1')
    ax.set_title(f"{group}: social_PC1 vs Rank  (Spearman rho={rho:+.2f}, p={p:.3f})")
    handles = [plt.Line2D([], [], marker='o', ls='', mfc=c, mec='k', label=s)
               for s, c in sex_colors.items()]
    ax.legend(handles=handles, title='Sex', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'pc1_vs_rank_{group}.png'),
                dpi=150, bbox_inches='tight')
    print(f"  saved {OUT_DIR}/pc1_vs_rank_{group}.png")


def main():
    cfg = SocialEncodingConfig()          # defaults: full_profile, subject 81G, PCA=2
    info = pd.read_csv(cfg.monkey_info_path)
    info['Name'] = info['Name'].astype(str)
    for group in GROUPS:
        analyze_group(group, cfg, info)


if __name__ == '__main__':
    main()
