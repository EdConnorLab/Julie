import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.response_window_analysis import run_permutation_anova_by_window, run_glm_by_window, run_permutation_kruskal_wallis_by_window
from analyses.response_window_finder.threshold_window_detection import detect_response_windows_for_session
from analyses.single_neuron_analysis import run_permutation_anova
from analyses.spike_count import prepare_binned_spike_data, aggregate_trial_level, extract_spike_counts_from_windows

def select_all_significant_neurons_using_KW(metadata, group_name="Zombies", bin_size=0.05,
                                                        analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/", save=True, use_sorted=False):
    all_results = []
    for ind, row in metadata.iterrows():
        date = str(row['Date'].strftime('%Y-%m-%d'))
        round = int(row['Round No.'])
        binned_data = prepare_binned_spike_data(date, round, bin_size, True, use_sorted=use_sorted)

        if binned_data is None or binned_data.empty:
            print(f"Skipping {date} Round {round}: no good neurons")
            continue  # 🔁 skip this round

        monkey_group_df = binned_data[binned_data['MonkeyGroup'] == group_name]
        group_specific_trial_df = aggregate_trial_level(monkey_group_df)
        results, sig_results = run_permutation_anova(group_specific_trial_df, category_col='MonkeyName', plot=False)
        all_results.append(results)
    final_df = pd.concat(all_results, ignore_index=True)
    # print(final_df.head())
    # Keep neurons that passed GLM **OR** permANOVA
    significant_df = final_df[(final_df["GLM_significant"]) | (final_df["Permutation_significant"])]
    print('All significant neurons passing GLM or permANOVA:')
    print(significant_df.head())
    if save:
        significant_df.to_pickle(analysis_cache_dir + f"{group_name}_significant_neurons_pANOVAorGLM_passed.pkl")
    return significant_df

def select_all_significant_neurons_using_pANOVA(metadata, group_name="Zombies", bin_size=0.05,
                                                analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/", save=True, use_sorted=False):
    all_results = []
    for ind, row in metadata.iterrows():
        date = str(row['Date'].strftime('%Y-%m-%d'))
        round = int(row['Round No.'])
        binned_data = prepare_binned_spike_data(date, round, bin_size, True, use_sorted=use_sorted)

        if binned_data is None or binned_data.empty:
            print(f"Skipping {date} Round {round}: no good neurons")
            continue  # 🔁 skip this round

        monkey_group_df = binned_data[binned_data['MonkeyGroup'] == group_name]
        group_specific_trial_df = aggregate_trial_level(monkey_group_df)
        perm_anova_results, perm_anova_sig_results = run_permutation_anova(group_specific_trial_df, category_col='MonkeyName', plot=False)
        all_results.append(perm_anova_results)
    final_df = pd.concat(all_results, ignore_index=True)
    # print(final_df.head())
    # Keep neurons that passed permANOVA
    significant_df = final_df[final_df["Permutation_significant"]]
    print('All significant neurons passing permANOVA:')
    print(significant_df.head())
    if save:
        significant_df.to_pickle(analysis_cache_dir + f"{group_name}_significant_neurons_pANOVA_passed.pkl")
    return significant_df


def detect_all_response_windows_for_all_neurons(metadata, group_name="Zombies",
                                                analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/",
                                                save=True, use_sorted=False):
    all_results = []
    for ind, row in metadata.iterrows():
        date = str(row['Date'].strftime('%Y-%m-%d'))
        round = int(row['Round No.'])
        results = detect_response_windows_for_session(date, round, monkey_group=group_name, plot=False)
        all_results.append(results)
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(by=['NeuronID'])
    print(final_df.head())
    print(f"{final_df.shape[0]} windows detected")
    if use_sorted:
        file_name = f"sorted_units_{group_name}_response_windows.pkl"
    else:
        file_name = f"{group_name}_response_windows.pkl"
    if save:
        final_df.to_pickle(analysis_cache_dir + file_name)
    return final_df

def detect_significant_windows_using_pANOVA(response_window_fpath, monkey_group,
                                            analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/",
                                            save = True):
    windows = pd.read_pickle(response_window_fpath)
    spike_counts_windows = extract_spike_counts_from_windows(windows)
    group_specific_spike_counts_for_windows = spike_counts_windows[spike_counts_windows['MonkeyGroup'] == monkey_group]
    panova_results, _ = run_permutation_anova_by_window(group_specific_spike_counts_for_windows,
                                                        category_col='MonkeyName',
                                                        neuron_col='NeuronID',
                                                        count_col='SpikeCount',
                                                        window_start_col='WindowStart_ms',
                                                        window_end_col='WindowEnd_ms',
                                                        n_permutations=1000,
                                                        alpha=0.05,
                                                        plot=False)
    significant_windows = panova_results[panova_results["Permutation_significant"]]
    print(f"{significant_windows.shape[0]} significant windows detected (permutation ANOVA)")
    if save:
        significant_windows.to_pickle(analysis_cache_dir + f"{monkey_group}_significant_windows_pANOVA_passed.pkl")
    return significant_windows

def detect_significant_windows_using_KW(response_window_fpath, monkey_group,
                                            analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/",
                                            save = True):
    windows = pd.read_pickle(response_window_fpath)
    spike_counts_windows = extract_spike_counts_from_windows(windows)
    group_specific_spike_counts_for_windows = spike_counts_windows[spike_counts_windows['MonkeyGroup'] == monkey_group]
    results_df, significant_df = run_permutation_kruskal_wallis_by_window(group_specific_spike_counts_for_windows,
                                             category_col='MonkeyName',
                                             neuron_col='NeuronID',
                                             count_col='SpikeCount',
                                             window_start_col='WindowStart_ms',
                                             window_end_col='WindowEnd_ms',
                                             n_permutations=10000,
                                             alpha=0.05,
                                             plot=False)
    print(significant_df.head())
    print(f"{significant_df.shape[0]} significant windows detected (permutation KW) ")
    if save:
        significant_df.to_pickle(analysis_cache_dir + f"{monkey_group}_significant_windows_KW_passed.pkl")
    return significant_df

if __name__ == "__main__":

    ## find sig cells and detect windows
    analysis_cache_dir = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/"
    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()
    monkey_group = "Zombies"
    # significant_df = select_all_significant_neurons_using_glm_and_pANOVA(metadata, group_name = monkey_group, analysis_cache_dir=analysis_cache_dir,save=False)
    # significant_df.to_pickle(analysis_cache_dir + f"{monkey_group}_significant_neurons_pANOVAorGLM_passed.pkl")
    # windows = detect_all_response_windows_for_all_neurons(metadata, group_name = monkey_group, analysis_cache_dir=analysis_cache_dir)
    sig_windows_kw = detect_significant_windows_using_KW(f"/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_response_windows.pkl", monkey_group=monkey_group)
    # sig_windows_anova = detect_significant_windows_using_pANOVA(f"/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_response_windows.pkl", monkey_group=monkey_group)


