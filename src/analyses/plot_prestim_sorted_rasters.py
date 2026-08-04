"""
plot_prestim_sorted_rasters.py — Plot SI-sorted rasters with a pre-stimulus baseline.

HOW TO RUN (PyCharm): open this file and click the green Run button. No CLI args —
just edit the CONFIG block below and re-run. (Requires `src` marked as a Sources
Root in PyCharm, which you already have for the other scripts.)

PREREQUISITE: a pre-stimulus cache built by analyze_sorted_spikes.py, e.g.
    python -m spikesorting.sort_spikes.analyze_sorted_spikes \
        --date 2023-09-26 --round 2 --pre-stimulus-time 1.0
writes Cortana/sorted_spike_cache_pre1000ms/2023-09-26_round_2.pkl.
PRE_STIMULUS_TIME below must match the window that cache was built with.
"""
from pathlib import Path

from analyses.raster_plotting import plot_stacked_raster
from data_access.spike_source import SISortedSpikeSource
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

# ===== CONFIG — edit me =======================================================
DATE = "2023-11-20"
ROUND_NO = 2
PRE_STIMULUS_TIME = 1.0                        # seconds before onset; must match the cache
CACHE_SUBDIR = "sorted_spike_cache_pre1000ms"  # the variant you generated
XLIM = 2.2                                     # right edge of the time axis (seconds after onset)
# Column layout: each inner list = social groups (top->bottom) for one column,
# drawn left->right. Set to None to stack all groups in a single (tall) column.
COLUMNS = [["Zombies", "Stranger Things"], ["Best Frans", "Instigators"]]
EXCLUDE = []           # monkey names to leave out entirely, e.g. ["144H", "81G"]
LABEL = "trials"       # next to each name: "trials" (count), "rank" (#N), "both", or "none"
SAVE = False           # False = show each figure interactively; True = write PNGs and move on
MIN_TRIALS = 5         # skip neurons with fewer trials
ONLY_NEURON = None     # set to a NeuronID string to plot just one; None = plot all
# ==============================================================================


def main():
    source = SISortedSpikeSource(cache_subdir=CACHE_SUBDIR,
                                 pre_stimulus_time=PRE_STIMULUS_TIME)
    df = source.load(DATE, ROUND_NO)
    if df is None:
        print(f"No SI-sorted data in '{CACHE_SUBDIR}' for {DATE} round {ROUND_NO}. "
              f"Generate it first with analyze_sorted_spikes.py --pre-stimulus-time.")
        return

    neuron_ids = df["NeuronID"].dropna().unique().tolist()
    if ONLY_NEURON is not None:
        neuron_ids = [n for n in neuron_ids if n == ONLY_NEURON]
    print(f"{DATE} round {ROUND_NO}: {len(neuron_ids)} neuron(s) to plot "
          f"(pre-stimulus {int(PRE_STIMULUS_TIME * 1000)} ms)")

    save_dir = Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "raster_plots" / CACHE_SUBDIR
    for neuron_id in neuron_ids:
        neuron_df = df[df["NeuronID"] == neuron_id]
        if len(neuron_df) < MIN_TRIALS:
            print(f"  skip {neuron_id}: only {len(neuron_df)} trials")
            continue
        n_spikes = sum(len(s) for s in neuron_df["SpikeTimes"])
        print(f"  plotting {neuron_id} — {len(neuron_df)} trials, {n_spikes} spikes")
        save_path = str(save_dir / f"{neuron_id}.png") if SAVE else None
        plot_stacked_raster(
            neuron_df,
            xlim=XLIM,
            pre_stimulus_time=PRE_STIMULUS_TIME,
            columns=COLUMNS,
            exclude_monkeys=EXCLUDE,
            annotate=LABEL,
            title=f"SI-sorted raster (pre-stim {int(PRE_STIMULUS_TIME * 1000)} ms): {neuron_id}",
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
