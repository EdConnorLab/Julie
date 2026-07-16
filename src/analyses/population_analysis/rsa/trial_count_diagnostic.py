# trial_count_diagnostic.py
"""
Audit trials (distinct TaskField) per monkey per session, per brain region.

Use this to decide the per-session `min_reps_per_monkey` floor and to see which
sessions run_rsa_pseudopop will drop (it keeps only sessions where every
candidate identity has >= min_reps trials, so the pseudo-population is a clean
rectangle with the same neurons for every monkey).

Run from the rsa/ directory (or with src/ on PYTHONPATH):
    python trial_count_diagnostic.py --region ER --group Zombies --floor 7
Defaults for data/monkeyinfo paths come from rsa_config.SocialRSAConfig.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from rsa_config import SocialRSAConfig


def load_trials(data_path, min_epoch_duration):
    """One row per trial (TaskField) after the pipeline's base cleaning."""
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(data_path, '*.pkl')))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['MonkeyName'] != 'NewMonkey']
    df = df.dropna(subset=['MonkeyGroup']).copy()
    df['session'] = df['Date'].astype(str) + '_' + df['Round No.'].astype(str)
    df['dur'] = df['EpochStartStop'].apply(lambda x: x[1] - x[0])
    df = df[df['dur'] >= min_epoch_duration]
    df['MonkeyName'] = df['MonkeyName'].astype(str)
    return df.drop_duplicates('TaskField')[['TaskField', 'MonkeyName',
                                            'Location', 'session', 'dur']]


def counts_matrix(sub):
    """session x monkey -> distinct TaskField count."""
    m = (sub.groupby(['session', 'MonkeyName'])['TaskField']
            .nunique().unstack(fill_value=0).astype(int).sort_index())
    return m[sorted(m.columns)]


def audit(region, sub, floor):
    M = counts_matrix(sub)
    if M.empty:
        print(f"== {region}: no trials ==\n"); return
    n_sess, n_monk = M.shape
    present_all = [c for c in M.columns if (M[c] > 0).all()]
    print(f"== {region}: {n_sess} sessions x {n_monk} monkeys "
          f"({len(present_all)} present in all sessions) ==")

    # current pseudopop rule: session-dropping at `floor`
    ids = present_all
    if len(ids) < 3:
        print(f"  <3 identities present in all sessions; cannot build pseudopop.\n")
        return
    kept, dropped = [], []
    for s in M.index:
        worst = int(M.loc[s, ids].min())
        (kept if worst >= floor else dropped).append(s)
    print(f"  floor {floor}/session -> keep {len(kept)}/{n_sess} sessions, "
          f"{len(ids)} identities")
    for s in dropped:
        thin = {m: int(M.loc[s, m]) for m in ids if M.loc[s, m] < floor}
        print(f"    drop {s}: " + ", ".join(f"{m}={c}" for m, c in sorted(thin.items())))

    # trade curve across floors
    print("  sessions kept at floors:", end=" ")
    for f in [1, 3, 5, 7, 10]:
        k = int(((M[ids] >= f).all(axis=1)).sum())
        print(f"{f}:{k}", end="  ")
    print("\n")


def main():
    cfg = SocialRSAConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default=cfg.data_path)
    ap.add_argument('--info', default=cfg.monkey_info_path)
    ap.add_argument('--min-epoch', type=float, default=1.3)
    ap.add_argument('--floor', type=int, default=7, help='per-session trial floor')
    ap.add_argument('--group', default=None, help='restrict to one Group Name')
    ap.add_argument('--region', default=None, help='restrict to one Location')
    args = ap.parse_args()

    info = pd.read_csv(args.info)
    info['Name'] = info['Name'].astype(str)
    known = set(info['Name'])
    grp = info.set_index('Name')['Group Name'].to_dict()

    trials = load_trials(args.data_path, args.min_epoch)
    trials = trials[trials['MonkeyName'].isin(known)]
    if args.group:
        members = {m for m, g in grp.items() if g == args.group}
        trials = trials[trials['MonkeyName'].isin(members)]

    regions = ([args.region] if args.region
               else sorted(trials['Location'].dropna().unique().tolist()))
    print(f"min_epoch_duration={args.min_epoch}, floor={args.floor}, "
          f"group={args.group or 'ALL'}\n")
    for region in regions:
        audit(region, trials[trials['Location'] == region], args.floor)


if __name__ == '__main__':
    main()
