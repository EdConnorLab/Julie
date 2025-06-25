import math

import numpy as np
import pandas as pd
from dPCA.dPCA import dPCA
from collections import defaultdict

from matplotlib import pyplot as plt

from analyses.population_analysis import load_all_filtered_spike_data
from analyses.spike_rate import compute_mean_spike_rate_table

# Parameters
BIN_SIZE_MS = 100
WINDOW_MS = 2000
N_BINS = WINDOW_MS // BIN_SIZE_MS

# Step 1: bin spikes per trial per neuron
def bin_spikes_per_trial(exploded_df):
    binned_dict = defaultdict(lambda: np.zeros(N_BINS))

    for _, row in exploded_df.iterrows():
        neuron = row['NeuronID']
        monkey = row['MonkeyName']
        group = row['MonkeyGroup']
        epoch_start, epoch_end = row['EpochStartStop']
        spike_times = np.array(row['SpikeTimes']) * 1000  # convert to ms

        aligned_spikes = spike_times - epoch_start * 1000
        bin_edges = np.linspace(0, WINDOW_MS, N_BINS + 1)
        counts, _ = np.histogram(aligned_spikes, bins=bin_edges)

        key = (neuron, monkey, group)
        binned_dict[key] += counts

    return binned_dict

# Step 2: reshape into (neurons, conditions, time)
def build_tensor(binned_dict):
    neurons = sorted(set(k[0] for k in binned_dict.keys()))
    monkeys = sorted(set(k[1] for k in binned_dict.keys()))
    groups = set(k[2] for k in binned_dict.keys())

    tensor = np.zeros((len(neurons), len(monkeys), len(groups), N_BINS))

    neuron_idx = {n: i for i, n in enumerate(neurons)}
    monkey_idx = {m: i for i, m in enumerate(monkeys)}
    group_idx =  {g: i for i, g in enumerate(groups)}

    for (neuron, monkey, group), vec in binned_dict.items():
        i = neuron_idx[neuron]
        j = monkey_idx[monkey]
        k = group_idx[group]
        tensor[i, j, k, :] = vec

    return tensor, neurons, monkeys, groups

# Step 3: run dPCA
def run_dpca(tensor):
    dpca = dPCA(labels='gmt', regularizer=1e-5)
    dpca.protect = ['t']
    Z = dpca.fit_transform(tensor)
    return dpca, Z


def plot_dpca_clean(Z, monkey_list, group_list, max_components=3):
    for label, components in Z.items():
        if components.ndim != 4:
            print(f"Skipping {label}")
            continue

        n_components = min(max_components, components.shape[0])
        n_groups = components.shape[1]
        n_monkeys = components.shape[2]
        time_len = components.shape[3]

        plt.figure(figsize=(10, 3 * n_components))

        for comp_idx in range(n_components):
            plt.subplot(n_components, 1, comp_idx + 1)

            for group_idx in range(n_groups):
                group = group_list[group_idx]
                monkey_ids = monkeys_by_group[group]
                for monkey_idx, monkey_id in enumerate(monkey_ids):
                    try:
                        plt.plot(components[comp_idx, group_idx, monkey_idx, :],
                                 label=f"{monkey_id} | {group}")
                    except IndexError:
                        print(f"⚠️ Skipping missing data for monkey {monkey_id} in group {group}")
                        continue

            plt.title(f"dPCA | {label} comp {comp_idx + 1}")
            plt.xlabel("Time bin")
            plt.ylabel("Component value")
            plt.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


def compute_trial_avg_spikecount(exploded_df):
    spike_table = []

    for _, row in exploded_df.iterrows():
        neuron = row['NeuronID']
        monkey = row['MonkeyName']
        start, end = row['EpochStartStop']
        duration = end - start
        count = len(row['SpikeTimes'])
        rate = count / duration
        spike_table.append((neuron, monkey, rate))

    df = pd.DataFrame(spike_table, columns=['NeuronID', 'MonkeyName', 'SpikeRate'])
    pivot = df.groupby(['NeuronID', 'MonkeyName']).mean().reset_index()
    return pivot.pivot(index='NeuronID', columns='MonkeyName', values='SpikeRate')

def run_behavioral_dpca(X, behavior_labels):
    """
    X: spike rate matrix (neurons × monkeys)
    behavior_labels: list or Series of behavior bins per monkey (same order as X.columns)
    """
    unique_bins = sorted(set(behavior_labels))
    bin_to_idx = {b: i for i, b in enumerate(unique_bins)}
    condition_labels = [bin_to_idx[behavior_labels[m]] for m in X.columns]

    # Collapse by bin (bin × neuron): (n_neurons, n_conditions)
    X_collapsed = np.stack([
        X.loc[:, [m for m in X.columns if behavior_labels[m] == b]].mean(axis=1).values
        for b in unique_bins
    ], axis=1)  # shape: (neurons, bins)

    dpca = dPCA(labels='b', regularizer=1e-5)
    Z = dpca.fit_transform(X_collapsed)

    return dpca, Z, unique_bins


def plot_dpca_components_grid(Z, factor_key='a', bin_labels=['low', 'high']):
    comps = Z[factor_key]
    n_comps = comps.shape[0]

    # 몇 행 × 몇 열로 나눌지 결정
    ncols = 2
    nrows = math.ceil(n_comps / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()  # 2D → 1D

    for i in range(n_comps):
        axs[i].bar(bin_labels, comps[i])
        axs[i].set_title(f"dPCA | {factor_key} comp {i + 1}")
        axs[i].set_ylim(comps.min(), comps.max())  # y축 일관되게

    # 나머지 subplot은 비워놓기
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


# Usage example (after loading exploded_df)
if __name__ == "__main__":
    exploded_df = load_all_filtered_spike_data(
        location='AMG',
        filter_config={"apply": True}
    )
    binned = bin_spikes_per_trial(exploded_df)
    tensor, neuron_list, monkey_list, group_list = build_tensor(binned)
    tensor = np.transpose(tensor, (1,2,0,3))
    dpca, Z = run_dpca(tensor)
    # plot_dpca_clean(Z, list(monkey_list), list(group_list), max_components=3)
    monkeys_by_group = defaultdict(list)

    for (_, monkey, group), _ in binned.items():
        if monkey not in monkeys_by_group[group]:
            monkeys_by_group[group].append(monkey)
    for label, components in Z.items():
        if components.ndim != 4:
            print(f"Skipping {label}")
            continue

        n_components = min(3, components.shape[0])
        n_groups = components.shape[1]
        n_monkeys = components.shape[2]
        time_len = components.shape[3]

        plt.figure(figsize=(10, 3 * n_components))
        group_list = list(group_list)
        for comp_idx in range(n_components):
            plt.subplot(n_components, 1, comp_idx + 1)

            for group_idx in range(n_groups):
                group = group_list[group_idx]
                monkey_ids = monkeys_by_group[group]
                for monkey_idx, monkey_id in enumerate(monkey_ids):
                    try:
                        plt.plot(components[comp_idx, group_idx, monkey_idx, :],
                                 label=f"{monkey_id} | {group}")
                    except IndexError:
                        print(f"⚠️ Skipping missing data for monkey {monkey_id} in group {group}")
                        continue

            plt.title(f"dPCA | {label} comp {comp_idx + 1}")
            plt.xlabel("Time bin")
            plt.ylabel("Component value")
            #plt.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
    # mean_spike_rate = compute_mean_spike_rate_table(exploded_df)
    # X = mean_spike_rate.pivot(index='NeuronID', columns='MonkeyName', values='MeanSpikeRate')
    # # Load your behavioral summary
    # behavior_df = pd.read_excel("/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/zombies_marginals.xlsx")
    # behavior_df['MonkeyName'] = behavior_df['MonkeyName'].astype(str)
    # behavior_df = behavior_df.set_index('MonkeyName')
    # # 예시: 세 가지 behavior를 tertile bin으로 나누기
    # label_a = pd.qcut(behavior_df['AffiliationTo'], q=2, labels=[0, 1])
    # label_b = pd.qcut(behavior_df['SubmissionTo'], q=2, labels=[0, 1])
    # label_c = pd.qcut(behavior_df['AgonismTo'], q=2, labels=[0, 1])
    #
    # # 모든 label을 하나의 array로 묶기
    # # shape = (3, n_conditions) → (n_factors, n_stimuli)
    # labels_array = np.vstack([
    #     label_a.loc[X.columns].values,
    #     label_b.loc[X.columns].values,
    #     label_c.loc[X.columns].values
    # ])
    #
    # bins = [0, 1]
    # X_collapsed = np.stack([
    #     X.loc[:, [m for m in X.columns if label_a[m] == b]].mean(axis=1).values
    #     for b in bins
    # ], axis=1)  # shape = (neurons, bins)
    #
    # # Run dPCA with only this factor
    # labels_array = np.array([bins])  # or np.array([[0, 1]])
    # dpca = dPCA(labels='a', regularizer=1e-5)
    # Z = dpca.fit_transform(X_collapsed, labels_array)
    # plot_dpca_components_grid(Z, factor_key='a', bin_labels=['low','high'])