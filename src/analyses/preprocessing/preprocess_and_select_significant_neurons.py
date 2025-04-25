import pandas as pd

from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.response_window_finder.threshold_window_detection import detect_response_windows_for_session
from analyses.single_neuron_analysis import run_glm, run_permutation_anova, merge_glm_and_permutation_anova
from analyses.spike_count import prepare_binned_spike_data, aggregate_trial_level

def select_all_significant_neurons_using_glm_and_pANOVA(metadata, group_name="Zombies", bin_size=0.05,
                                                        analysis_cache_dir="/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/"):
    all_results = []
    for ind, row in metadata.iterrows():
        date = str(row['Date'].strftime('%Y-%m-%d'))
        round = int(row['Round No.'])
        binned_data = prepare_binned_spike_data(date, round, bin_size, True)
        zombies_df = binned_data[binned_data['MonkeyGroup'] == group_name]
        formula = "SpikeCount ~ C(MonkeyName)"  # Stimulus identity
        glm_results = run_glm(zombies_df, formula=formula)
        # print(glm_results.head())
        zombies_trial_df = aggregate_trial_level(zombies_df)
        perm_anova_results = run_permutation_anova(zombies_trial_df, category_col='MonkeyName', plot=False)
        # print(perm_anova_results.head())
        results = merge_glm_and_permutation_anova(glm_results, perm_anova_results)
        # print(results.head())
        all_results.append(results)
    final_df = pd.concat(all_results, ignore_index=True)
    # print(final_df.head())
    # Keep neurons that passed GLM or permANOVA
    significant_df = final_df[(final_df["GLM_significant"]) | (final_df["Permutation_significant"])]
    print('All significant neurons passing GLM or permANOVA:')
    print(significant_df.head())
    significant_df.to_pickle(analysis_cache_dir + f"{group_name}_significant_neurons_pANOVAorGLM_passed.pkl")


def main():
    analysis_cache_dir = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/"
    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()
    # select_all_significant_neurons_using_glm_and_pANOVA(metadata, analysis_cache_dir=analysis_cache_dir)
    all_results = []
    group_name = "Zombies"
    for ind, row in metadata.iterrows():
        date = str(row['Date'].strftime('%Y-%m-%d'))
        round = int(row['Round No.'])
        results = detect_response_windows_for_session(date, round, monkey_group=group_name, plot=False)
        all_results.append(results)
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_pickle(analysis_cache_dir + f"{group_name}_all_threshold_detected_windows.pkl")
    print(final_df)




if __name__ == "__main__":
    main()