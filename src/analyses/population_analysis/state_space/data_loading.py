# data_loading.py
from neuron_selection_curves import load_all_trials


def load_and_filter(cfg):
    df = load_all_trials(cfg.data_path)
    df = df[df['MonkeyName'] != 'NewMonkey']
    df = df.dropna(subset=['MonkeyGroup'])

    df = df.copy()

    # Build session ID from date + round (region-agnostic)
    df['session'] = df['Date'].astype(str) + '_' + df['Round No.'].astype(str)

    # Filter by region (affects which neurons, not which sessions)
    if cfg.region != 'ALL':
        df = df[df['Location'] == cfg.region]

    if cfg.session is not None:
        df = df[df['session'] == cfg.session]
        if len(df) == 0:
            raise ValueError(f"No data for session '{cfg.session}' with region='{cfg.region}'")

    # Drop short trials
    durs = df['EpochStartStop'].apply(lambda x: x[1] - x[0])
    df = df[durs >= cfg.min_epoch_duration].reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['session'].nunique()} sessions, "
          f"{df['TaskField'].nunique()} trials, {df['NeuronID'].nunique()} neurons")
    return df
