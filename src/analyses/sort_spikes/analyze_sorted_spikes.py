import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw

from analyses.sort_spikes.sort_spikes import get_recording_session_info



def write_summary_and_exit(msg="no units in agreement"):
    # Always restore stdout before leaving
    print(msg)
    with open(f"{round_folder}_sorting_summary.txt", "w") as f:
        f.write(msg + "\n")
    print("No units in agreement, skipping further analysis.")
    sys.exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)   # e.g., 2023-10-10
    p.add_argument("--round", type=int, required=True)
    args = p.parse_args()

    date = args.date
    round = args.round

    date_obj = datetime.strptime(date, "%Y-%m-%d")
    date_str = date_obj.strftime("%y%m%d")  # → "230926"

    # build paths safely
    round_folder = f"{date_str}_round{round}"
    base_dir = os.path.join(
        "/home/connorlab/Documents/IntanData/Cortana", date, round_folder
    )

    sampling_frequency, channels = get_recording_session_info(base_dir)
    # Load Sorting Objects
    sorting_KS4 = ss.read_sorter_folder(os.path.join(base_dir, 'kilosort4_output'))
    sorting_MS5 = ss.read_sorter_folder(os.path.join(base_dir, 'mountainsort5_output'))
    sorting_TDC = ss.read_sorter_folder(os.path.join(base_dir, 'tridesclous_output'))

    # Load Sorting Analyzers
    analyzer_KS4 = si.load_sorting_analyzer(os.path.join(base_dir, 'analyzer_KS4_binary'))
    analyzer_MS5 = si.load_sorting_analyzer(os.path.join(base_dir, 'analyzer_MS5_binary'))
    analyzer_TDC = si.load_sorting_analyzer(os.path.join(base_dir, 'analyzer_TDC_binary'))

    # Compare sorters
    comp_multi = sc.compare_multiple_sorters(
        sorting_list=[sorting_TDC, sorting_MS5, sorting_KS4], name_list=["tdc", "ms5", "ks4"])
    sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=3)

    ## Plot
    import matplotlib.pyplot as plt

    # Plot agreement across all sorters
    fig1 = sw.plot_multicomparison_agreement(comp_multi)
    fig1.figure.savefig(base_dir + "_agreement_overall.png", bbox_inches="tight")

    # Plot agreement by sorter
    fig2 = sw.plot_multicomparison_agreement_by_sorter(comp_multi)
    fig2.figure.savefig(base_dir + "_agreement_by_sorter.png", bbox_inches="tight")

    plt.close("all")
    # Check the channels matching up
    tdc_peak = si.get_template_extremum_channel(analyzer_TDC, peak_sign='both', outputs='index')
    ms5_peak = si.get_template_extremum_channel(analyzer_MS5, peak_sign='both', outputs='index')
    ks4_peak = si.get_template_extremum_channel(analyzer_KS4, peak_sign='both', outputs='index')
    consensus_maps = sorting_agreement.get_property('unit_ids')
    if isinstance(consensus_maps, np.ndarray):
        consensus_maps = consensus_maps.tolist()

    if consensus_maps is None or len(consensus_maps) == 0:
        write_summary_and_exit()
    # Save the original stdout
    original_stdout = sys.stdout
    summary_file = f"{round_folder}_sorting_summary.txt"

    with open(f"{round_folder}_sorting_summary.txt", "w") as f:
        sys.stdout = f
        base_channel = []
        for consensus_map in consensus_maps:
            tdc_uid = consensus_map['tdc']
            ms5_uid = consensus_map['ms5']
            ks4_uid = consensus_map['ks4']

            tdc_ch = tdc_peak[tdc_uid]
            ms5_ch = ms5_peak[ms5_uid]
            ks4_ch = ks4_peak[ks4_uid]
            base_channel.append((tdc_uid, tdc_ch))

            if not (tdc_ch == ms5_ch == ks4_ch):
                print("Channels do NOT match!")
                print(
                    f"TDC unit_id {tdc_uid} Ch {tdc_ch}, "
                    f"MS5 unit_id {ms5_uid} Ch {ms5_ch}, "
                    f"KS4 unit_id {ks4_uid} Ch {ks4_ch}"
                )


        # Choose which one to use
        MS5_unit_ids = sorting_MS5.get_unit_ids()
        KS4_unit_ids = sorting_KS4.get_unit_ids()
        main_sorting_results_source = sorting_KS4
        main_sorting_results_units = KS4_unit_ids
        from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
        from clat.intan.marker_channels import epoch_using_marker_channels

        stim_epochs_from_markers = epoch_using_marker_channels(os.path.join(base_dir, "digitalin.dat"), false_negative_correction_duration=2)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(os.path.join(base_dir, "notes.txt"), stim_epochs_from_markers)


        from clat.intan.channels import Channel
        chan_count = {}
        sorted_results = []
        for idx, chan in base_channel:
            channel = Channel[f"C_{chan:03}"]
            if chan not in chan_count:
                chan_count[chan] = 0
            chan_count[chan] +=1
            sorted_results.append({
                'BaseChannel': channel,
                'SpikeIdx': main_sorting_results_source.get_unit_spike_train(unit_id=main_sorting_results_units[idx]),
                'Channel': f"{channel}_Unit {chan_count[chan]}"
            })
        sorted_df = pd.DataFrame(sorted_results)
        print(f"Using... {main_sorting_results_source} for getting the original channels that sorted units belong to")
        print(f"Units in agreement: {sorted_df.shape[0]}")

    # close summary text file
    sys.stdout = original_stdout
    print(f"Units in agreement: {sorted_df.shape[0]}")
    # reformat sorted spikes df
    sorted_spikes_with_taskid = []
    for task_id, epoch_idx in epochs_for_task_ids.items():
        epoch_start, epoch_stop = epoch_idx
        epoch = (epoch_start/sampling_frequency, epoch_stop/sampling_frequency)
        for _,row in sorted_df.iterrows():
            spike_indices = row['SpikeIdx']
            mask = (epoch_start <= spike_indices) & (spike_indices <= epoch_stop)
            selected_spikes = spike_indices[mask]
            spikes = selected_spikes / sampling_frequency
            sorted_spikes_with_taskid.append({
                'TaskField': task_id,
                'SpikeTimes': spikes,
                'Channel': row['Channel'],
                'BaseChannel': row['BaseChannel'],
                'EpochStartStop': epoch
            })

    sorted_spikes = pd.DataFrame(sorted_spikes_with_taskid)
    pkl_file_name = date + "_round_" + str(round) + ".pkl"
    print(f"Reading {pkl_file_name} from exploded spike cache for getting sorted spikes")
    unsorted_exploded_spikes_dir = "/home/connorlab/Documents/GitHub/Julie/Cortana/exploded_spike_cache/"
    unsorted = pd.read_pickle(unsorted_exploded_spikes_dir + pkl_file_name)
    print(f"shape of unsorted dataframe {unsorted.shape}")
    print(f"num of unique TaskID from unsorted: {unsorted['TaskField'].nunique()}")
    print(f"num of unique TaskID from sorted_spikes: {sorted_spikes['TaskField'].nunique()}")

    meta_cols = ['TaskField', 'MonkeyGroup', 'MonkeyId', 'MonkeyName', 'Date', 'Round No.', 'Location']
    metadata_raw = unsorted[meta_cols]
    check = (unsorted
             .groupby('TaskField')[['MonkeyGroup', 'MonkeyId', 'MonkeyName', 'Date', 'Round No.', 'Location']]
             .nunique())
    print("Max distinct values within a TaskField:\n", check.max())  # should be 1 for a clean many-to-one

    metadata_uniq = metadata_raw.drop_duplicates(subset=['TaskField'])

    merged_sorted_spikes_df = sorted_spikes.merge(
        metadata_uniq, on='TaskField', how='left', validate='many_to_one'
    )
    print(f"shape of sorted dataframe {sorted_spikes.shape}")
    print(f"shape of merged dataframe {merged_sorted_spikes_df.shape}")
    file_name = date + "_round_" + str(round) + ".pkl"
    merged_sorted_spikes_df.to_pickle('/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache/' + pkl_file_name)
    print(f"newly generated sorted spike cache saved as... {pkl_file_name}")
