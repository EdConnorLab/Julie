import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
from clat.intan.channels import Channel
from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels

from analyses.cache_utils import SortedSpikeCacheManager, ExplodedSpikeCacheManager
from analyses.sort_spikes.sort_spikes import get_recording_session_info, build_intan_session_path


def load_sorting_results(intan_dir):
    sorting_KS4 = ss.read_sorter_folder(os.path.join(intan_dir, 'kilosort4_output'))
    sorting_MS5 = ss.read_sorter_folder(os.path.join(intan_dir, 'mountainsort5_output'))
    sorting_TDC = ss.read_sorter_folder(os.path.join(intan_dir, 'tridesclous_output'))

    analyzer_KS4 = si.load_sorting_analyzer(os.path.join(intan_dir, 'analyzer_KS4_binary'))
    analyzer_MS5 = si.load_sorting_analyzer(os.path.join(intan_dir, 'analyzer_MS5_binary'))
    analyzer_TDC = si.load_sorting_analyzer(os.path.join(intan_dir, 'analyzer_TDC_binary'))

    return (sorting_KS4, sorting_TDC, sorting_MS5), (analyzer_KS4, analyzer_TDC, analyzer_MS5)


def find_consensus_units(sortings, analyzers, intan_dir):
    sorting_KS4, sorting_TDC, sorting_MS5 = sortings
    analyzer_KS4, analyzer_TDC, analyzer_MS5 = analyzers

    comp_multi = sc.compare_multiple_sorters(
        sorting_list=[sorting_TDC, sorting_MS5, sorting_KS4],
        name_list=["tdc", "ms5", "ks4"],
    )
    sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=3)

    # Save agreement plots
    fig1 = sw.plot_multicomparison_agreement(comp_multi)
    fig1.figure.savefig(intan_dir + "_agreement_overall.png", bbox_inches="tight")
    fig2 = sw.plot_multicomparison_agreement_by_sorter(comp_multi)
    fig2.figure.savefig(intan_dir + "_agreement_by_sorter.png", bbox_inches="tight")
    plt.close("all")

    # Get consensus unit maps
    consensus_maps = sorting_agreement.get_property('unit_ids')
    if isinstance(consensus_maps, np.ndarray):
        consensus_maps = consensus_maps.tolist()

    if consensus_maps is None or len(consensus_maps) == 0:
        return None, []

    # Check channel agreement across sorters and build unit table
    tdc_peak = si.get_template_extremum_channel(analyzer_TDC, peak_sign='both', outputs='index')
    ms5_peak = si.get_template_extremum_channel(analyzer_MS5, peak_sign='both', outputs='index')
    ks4_peak = si.get_template_extremum_channel(analyzer_KS4, peak_sign='both', outputs='index')

    summary_lines = []
    base_channel_list = []
    for consensus_map in consensus_maps:
        tdc_uid, ms5_uid, ks4_uid = consensus_map['tdc'], consensus_map['ms5'], consensus_map['ks4']
        tdc_ch, ms5_ch, ks4_ch = tdc_peak[tdc_uid], ms5_peak[ms5_uid], ks4_peak[ks4_uid]
        base_channel_list.append((ks4_uid, ks4_ch))

        if not (tdc_ch == ms5_ch == ks4_ch):
            summary_lines.append(
                f"Channel mismatch: TDC unit {tdc_uid} Ch {tdc_ch}, "
                f"MS5 unit {ms5_uid} Ch {ms5_ch}, KS4 unit {ks4_uid} Ch {ks4_ch}"
            )

    # Use KS4 as the primary source for spike trains
    chan_count = {}
    sorted_results = []
    for unit_idx, chan_idx in base_channel_list:
        channel = Channel[f"C_{chan_idx:03}"]
        chan_count[chan_idx] = chan_count.get(chan_idx, 0) + 1
        sorted_results.append({
            'BaseChannel': channel,
            'SpikeIdx': sorting_KS4.get_unit_spike_train(unit_id=sorting_KS4.get_unit_ids()[unit_idx]),
            'Channel': f"{channel}_Unit {chan_count[chan_idx]}",
        })

    sorted_df = pd.DataFrame(sorted_results)
    summary_lines.append(f"Using KS4 for spike trains. Units in agreement: {sorted_df.shape[0]}")
    return sorted_df, summary_lines


def assign_spikes_to_trials(sorted_df, intan_dir, sampling_frequency):
    stim_epochs = epoch_using_marker_channels(
        os.path.join(intan_dir, "digitalin.dat"),
        false_negative_correction_duration=2,
    )
    epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(
        os.path.join(intan_dir, "notes.txt"), stim_epochs,
    )

    rows = []
    for task_id, (epoch_start, epoch_stop) in epochs_for_task_ids.items():
        epoch = (epoch_start / sampling_frequency, epoch_stop / sampling_frequency)
        for _, unit_row in sorted_df.iterrows():
            spike_indices = unit_row['SpikeIdx']
            mask = (epoch_start <= spike_indices) & (spike_indices <= epoch_stop)
            rows.append({
                'TaskField': task_id,
                'SpikeTimes': spike_indices[mask] / sampling_frequency,
                'Channel': unit_row['Channel'],
                'BaseChannel': unit_row['BaseChannel'],
                'EpochStartStop': epoch,
            })

    return pd.DataFrame(rows)


def merge_with_metadata(sorted_spikes_df, date_str, round_no):
    exploded_cache = ExplodedSpikeCacheManager()
    unsorted = exploded_cache.load_or_compute(date_str, round_no)

    meta_cols = ['TaskField', 'MonkeyGroup', 'MonkeyId', 'MonkeyName', 'Date', 'Round No.', 'Location']
    metadata_uniq = unsorted[meta_cols].drop_duplicates(subset=['TaskField'])

    merged = sorted_spikes_df.merge(metadata_uniq, on='TaskField', how='left', validate='many_to_one')
    merged['NeuronID'] = (
        merged['Location'].astype(str) + "_" +
        merged['Date'].astype(str) + "_" +
        merged['Round No.'].astype(str) + "_" +
        merged['Channel'].astype(str)
    )
    return merged


def write_summary(summary_dir, round_folder, lines):
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{round_folder}_sorting_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary written to {summary_path}")


def analyze_sorted_spikes(date_str, round_no):
    intan_dir = build_intan_session_path(date_str, round_no)
    sampling_frequency, _ = get_recording_session_info(intan_dir)

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    round_folder = f"{date_obj.strftime('%y%m%d')}_round{round_no}"

    cache_mgr = SortedSpikeCacheManager()

    # Load sorter outputs and find consensus
    sortings, analyzers = load_sorting_results(intan_dir)
    sorted_df, summary_lines = find_consensus_units(sortings, analyzers, intan_dir)

    if sorted_df is None:
        msg = "No units in agreement, skipping further analysis."
        write_summary(str(cache_mgr.summary_dir), round_folder, [msg])
        print(msg)
        sys.exit(0)

    print(f"Units in agreement: {sorted_df.shape[0]}")

    # Assign spikes to trial epochs
    sorted_spikes_df = assign_spikes_to_trials(sorted_df, intan_dir, sampling_frequency)
    print(f"Sorted spikes shape: {sorted_spikes_df.shape}")

    # Merge with metadata from exploded spike cache
    merged = merge_with_metadata(sorted_spikes_df, date_str, round_no)
    print(f"Merged shape: {merged.shape}")

    # Save to sorted spike cache
    pkl_path = cache_mgr._get_cache_path(f"{date_str}_round_{round_no}")
    merged.to_pickle(pkl_path)
    print(f"Saved sorted spike cache: {pkl_path}")

    # Write summary
    write_summary(str(cache_mgr.summary_dir), round_folder, summary_lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze SI-sorted spikes: consensus validation and cache generation")
    p.add_argument("--date", required=True, help="e.g. 2023-09-26")
    p.add_argument("--round", type=int, required=True, dest="round_no")
    args = p.parse_args()

    analyze_sorted_spikes(args.date, args.round_no)
