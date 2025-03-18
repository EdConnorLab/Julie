"""
These methods are mainly based on
an article in https://brainandmind.fandom.com/wiki/Automatic_Burst_Detection
and a paper "Burst detection methods" by E. Cotterill and SJ Eglen (2019)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.special import factorial
from clat.intan.channels import Channel

from channel_enum_resolvers import is_channel_in_dict, get_value_from_dict_with_channel
from monkey_names import Zombies
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files, get_spike_rates_for_each_trial



# Rank Surprise Burst Detection
def exhaustive_surprise_maximization(spike_train, limit, threshold):
    """
    Exhaustive Surprise Maximization algorithm for burst detection.
    Parameters:
        spike_train: array of spike times.
        limit: largest ISI allowed in a burst (percentile threshold).
        threshold: minimum RS to consider a burst valid.
    Returns:
        detected_bursts: list of tuples, each containing (start_index, end_index, RS).
    """

    def compute_rank_surprise(isi, q, u, isi_count):
        """
        Compute the Rank Surprise (RS) statistic.
        Parameters:
            isi: list of inter-spike intervals (ISIs).
            q: number of ISIs in the burst.
            u: sum of ranks of the burst ISIs.
            isi_count: total number of ISIs in the spike train.
        Returns:
            RS: Rank Surprise statistic value.
        """
        # Compute probability P(Tq <= u)
        probability = 0
        for k in range((u - q) // isi_count + 1):
            num = (-1) ** k * factorial(u - k * isi_count)
            den = (
                    factorial(k)
                    * factorial(q - k)
                    * factorial(u - k * isi_count - q)
            )
            probability += num / den
        probability /= isi_count ** q

        # Rank Surprise statistic
        RS = -np.log(probability)
        return RS

    isi = np.diff(spike_train)
    isi_count = len(isi)
    ranks = np.argsort(np.argsort(isi)) + 1  # Compute ranks (1-based index)
    marked_indices = np.where(isi <= limit)[0]  # Mark ISIs below the limit

    detected_bursts = []
    while len(marked_indices) > 0:
        best_RS = 0
        best_burst = None

        # Test all contiguous sequences of marked indices
        for start_idx, end_idx in combinations(marked_indices, 2):
            if end_idx - start_idx + 1 > 1:  # Minimum of two spikes in a burst
                burst_isis = ranks[start_idx:end_idx + 1]
                u = sum(burst_isis)
                q = len(burst_isis)
                RS = compute_rank_surprise(isi, q, u, isi_count)
                if RS > best_RS:
                    best_RS = RS
                    best_burst = (start_idx, end_idx, RS)

        # If a burst is found with RS above the threshold, store it and remove ISIs
        if best_burst and best_RS > threshold:
            detected_bursts.append(best_burst)
            start_idx, end_idx, _ = best_burst
            marked_indices = marked_indices[
                ~((marked_indices >= start_idx) & (marked_indices <= end_idx))
            ]
        else:
            break

    return detected_bursts

# Simple Burst Detection
def detect_bursts(spike_times, isi_threshold):
    # Compute inter-spike intervals
    isi = np.diff(spike_times)
    # Identify bursts based on ISI threshold
    bursts = np.where(isi < isi_threshold)[0]
    return bursts

# Simplified Poisson Surprise Burst Detection
def simplified_poisson_surprise(spike_times, t_start, t_end, threshold=5):
    """
    Detect bursts in a spike train using the Poisson Surprise method.

    Parameters:
    - spike_times (array-like): Array of spike times (in seconds).
    - t_start (float): Start time of the observation period (in seconds).
    - t_end (float): End time of the observation period (in seconds).
    - threshold (float): Poisson Surprise threshold for defining a burst.

    Returns:
    - bursts (list of tuples): Detected bursts as (start_time, end_time, surprise_value).
    """
    spike_times = np.array(spike_times)
    if len(spike_times) == 0:
        return []

    bursts = []
    mean_rate = len(spike_times) / (t_end - t_start)  # Expected rate under Poisson assumption
    n_spikes = len(spike_times)

    for i in range(n_spikes):
        for j in range(i + 1, n_spikes + 1):
            # Define burst interval
            burst_start = spike_times[i]
            burst_end = spike_times[j - 1]
            n_burst_spikes = j - i
            duration = burst_end - burst_start

            if duration <= 0:
                continue

            # Calculate surprise value
            expected_count = mean_rate * duration
            surprise = -np.log10(poisson.sf(n_burst_spikes - 1, expected_count))

            # Check if burst qualifies
            if surprise > threshold:
                bursts.append((burst_start, burst_end, surprise))

    return bursts


# # Example Usage
# spike_times = [0.1, 0.12, 0.15, 0.6, 0.62, 0.63, 1.0, 1.02, 1.05]
# t_start, t_end = 0.0, 2.0  # Observation period (seconds)
# threshold = 5.0
#
# bursts = simplified_poisson_surprise(spike_times, t_start, t_end, threshold)
# print("Detected bursts:")
# for burst in bursts:
#     print(f"Start: {burst[0]:.3f}, End: {burst[1]:.3f}, Surprise: {burst[2]:.3f}")

def poisson_surprise(spike_times, mean_firing_rate, s_threshold=10):
    """
    Detect bursts in a spike train using the Poisson Surprise method.

    Parameters:
    - spike_times (array-like): Array of spike times (in seconds).
    - mean_firing_rate (float): Mean firing rate of the spike train (spikes per second).
    - s_threshold (float): Threshold for declaring a burst (default = 10).

    Returns:
    - bursts (list of tuples): Detected bursts as (start_time, end_time, S_value).
    """

    def calculate_surprise(k, tau, mean_firing_rate):
        lambda_tau = mean_firing_rate * tau
        prob = poisson.sf(k - 1, lambda_tau)  # CCDF directly computed
        return -np.log(prob) if prob > 0 else float('inf')

    bursts = []
    n_spikes = len(spike_times)

    i = 0
    while i < n_spikes:
        # Start of a potential burst
        j = i + 2  # Minimum sequence length is 3
        # Extend the potential burst sequence as long as:
        # 1. We haven't reached the end of the spike train.
        # 2. The average ISI for the sequence remains shorter than half the average ISI of the entire spike train.
        while j < n_spikes and ((spike_times[j] - spike_times[i]) / (j - i + 1)) < (2 / mean_firing_rate):
            j += 1

        # Skip if no valid burst was formed (i.e., j didn't increment)
        if j <= i + 2:  # Only two spikes considered, not a valid burst
            i += 1  # Move to the next spike as the starting point
            continue

        # Evaluate surprise S for the initial sequence
        while j <= n_spikes:
            tau = spike_times[j - 1] - spike_times[i]  # Interval length
            if tau <= 0:
                break
            k = j - i  # Spike count in the interval
            s_value = calculate_surprise(k, tau, mean_firing_rate)

            if s_value > s_threshold:
                # Extend burst by adding more spikes if S increases
                while j < n_spikes:
                    next_tau = spike_times[j] - spike_times[i]
                    next_s_value = calculate_surprise(k + 1, next_tau, mean_firing_rate)
                    if next_s_value > s_value:
                        s_value = next_s_value
                        k += 1
                        j += 1
                    else:
                        break

                # Remove spikes from the start if it increases S
                while i < j - 2:
                    next_tau = spike_times[j - 1] - spike_times[i + 1]
                    next_s_value = calculate_surprise(k - 1, next_tau, mean_firing_rate)
                    if next_s_value > s_value:
                        s_value = next_s_value
                        k -= 1
                        i += 1
                    else:
                        break

                # Save the burst
                bursts.append((spike_times[i], spike_times[j - 1], s_value))
                break
            else:
                break

        # Move to the next potential burst
        i += 1

    return bursts


def extract_specific_spike_times_from_raw_unsorted_data(raw_unsorted_data, monkeys, channels):
    all_spike_times = pd.DataFrame(index=[])
    for monkey in monkeys:
        monkey_data = raw_unsorted_data[raw_unsorted_data['MonkeyName'] == monkey]
        monkey_specific_spike_times = {}
        for channel in channels:
            spike_times = []
            for index, row in monkey_data.iterrows():
                if is_channel_in_dict(channel, row['SpikeTimes']):
                    data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
                    spike_times.append(data)
                else:
                    print(f"No data for {channel} in row {index}")
            monkey_specific_spike_times[channel] = all_spike_times
        #
        # # Add monkey-specific spike rates to the DataFrame
        all_spike_times[monkey] = pd.Series(monkey_specific_spike_times)

    return all_spike_times

zombies = [member.value for name, member in Zombies.__members__.items()]
del zombies[6]
del zombies[-1]

date = "2023-09-26"
round_no = 1
raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
# spike_times = extract_specific_spike_times_from_raw_unsorted_data(raw_unsorted_data, zombies, [Channel.C_027, Channel.C_014])
yanny = raw_unsorted_data[raw_unsorted_data['MonkeyName'] == "7124"]
epoch = yanny['EpochStartStop']
start, stop = epoch.iloc[0]
channel = Channel.C_027
yanny_spike_times = []
min_length = float('inf')
for index, row in yanny.iterrows():
    if is_channel_in_dict(channel, row['SpikeTimes']):
        data = get_value_from_dict_with_channel(channel, row['SpikeTimes'])
        yanny_spike_times.append(data)
        if len(data) < min_length:
            min_length = len(data)  # Update minimum length
# print(yanny_spike_times)
truncated_spike_times = [spike[:min_length] for spike in yanny_spike_times]

# Convert list of lists into a 2D NumPy array
yanny_spike_times_array = np.array(truncated_spike_times)
# print(yanny_spike_times.shape)

# for index, row in spike_counts.iterrows():
#     if str(index) == "Channel.C_027":
#         s_threshold = 3
        # data = row['total_sum']
        # target_mean = np.mean(data) # baseline mean
        # mean_firing_rate = target_mean/time_chunk_size
        # bursts = poisson_surprise(spike_times, mean_firing_rate, s_threshold)

# Example Usage
spike_times = yanny_spike_times_array[0,:]
spike_count = len(spike_times)
mean_firing_rate = len(spike_times) / (spike_times[-1] - spike_times[0])  # Spikes per second
s_threshold = 5
spike_times_array = np.array(spike_times)

# Create a figure and axes
# fig, ax = plt.subplots()

# # Plot vertical lines at spike times
# ax.vlines(spike_times_array, 0, 1, color='k')
# # Show the plot
# plt.show()


bursts = poisson_surprise(spike_times, mean_firing_rate, s_threshold)
print("Poisson Surprise ---- Detected bursts:")
for burst in bursts:
    print(f"Start: {burst[0]:.3f}, End: {burst[1]:.3f}, S-value: {burst[2]:.2f}")



# Example usage
spike_train = spike_times
limit = np.percentile(np.diff(spike_train), 75)  # 75th percentile ISI threshold
threshold = 5  # Minimum RS threshold

bursts = exhaustive_surprise_maximization(spike_train, limit, threshold)
print("Exhaustive Surprise Maximization ---- Detected bursts:")
for burst in bursts:
    print(f"Start: {burst[0]}, End: {burst[1]}, RS: {burst[2]:.2f}")


bursts = detect_bursts(spike_times, isi_threshold = 0.3)
print("Simple ---- Detected bursts:")
print(bursts)
# for burst in bursts:
#     print(f"Start: {burst[0]}, End: {burst[1]}, RS: {burst[2]:.2f}")

bursts = simplified_poisson_surprise(spike_times, start, stop, threshold=5)
print("Simplified Poisson ---- Detected bursts:")
for burst in bursts:
    print(f"Start: {burst[0]}, End: {burst[1]}, RS: {burst[2]:.2f}")