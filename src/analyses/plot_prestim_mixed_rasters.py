"""
plot_prestim_mixed_rasters.py — stacked pre-stimulus rasters for the exploded
(manual-sorted + unsorted) cells, one figure per channel/unit.

HOW TO RUN (PyCharm): open this file and click the green Run button. No CLI args —
edit the CONFIG block and re-run. (Requires `src` as a Sources Root, which you
already have.)

With BUILD_IF_MISSING=True this script builds the pre-stimulus exploded variant on
first run (writes Cortana/exploded_spike_cache_pre{ms}ms/) and plots it; later runs
just read it. Same stacked layout as the SI-sorted rasters.

NOTE: only manually-sorted units get a real pre-stimulus baseline (their indices
live un-windowed in sorted_spikes.pkl). Unsorted channels come from the already-
clipped compiled.pkl and have NO pre-stimulus spikes, so they are plotted from t=0
(no empty [-pre, 0) band that could be mistaken for silence). For pre-stimulus
baselines on unsorted channels, use plot_prestim_mua_rasters.py instead. spike.dat
is not used.
"""
from pathlib import Path

from analyses.raster_plotting import plot_stacked_raster
from data_access.exploded_peristim_builder import build_exploded_peristim_cache, peristim_cache_subdir
from data_access.spike_source import MixedManualSpikeSource
from project_util import PROJECT_BASE_PATH, SUBJECT_MONKEY

# ===== CONFIG — edit me =======================================================
DATE = "2023-09-26"
ROUND_NO = 2
PRE_STIMULUS_TIME = 1.0          # seconds of baseline before onset (manual-sorted units only)
BUILD_IF_MISSING = True          # build the variant cache if it isn't on disk yet
FORCE_REBUILD = False            # rebuild even if the variant exists

XLIM = 2.2                                     # right edge of the time axis (seconds after onset)
COLUMNS = [["Zombies", "Best Frans"], ["Instigators", "Stranger Things"]]
EXCLUDE = []           # monkey names to leave out entirely, e.g. ["144H", "81G"]
LABEL = "trials"       # next to each name: "trials" (count), "rank" (#N), "both", or "none"
CURATED_ONLY = False   # True = only curated channels from the recording metadata
SAVE = False           # False = show each figure interactively; True = write PNGs and move on
MIN_TRIALS = 7         # skip neurons/channels with fewer trials
ONLY_NEURON = None     # set to a NeuronID string to plot just one; None = plot all
# ==============================================================================


def main():
    cache_subdir = peristim_cache_subdir(PRE_STIMULUS_TIME)
    if BUILD_IF_MISSING or FORCE_REBUILD:
        print(f"Building/checking {cache_subdir} for {DATE} round {ROUND_NO} ...")
        build_exploded_peristim_cache(DATE, ROUND_NO, PRE_STIMULUS_TIME,
                                      cache_subdir=cache_subdir, force=FORCE_REBUILD)

    source = MixedManualSpikeSource(cache_subdir=cache_subdir, curated_channels_only=CURATED_ONLY)
    df = source.load(DATE, ROUND_NO)
    if df is None:
        print(f"No exploded data in '{cache_subdir}' for {DATE} round {ROUND_NO}.")
        return

    neuron_ids = df["NeuronID"].dropna().unique().tolist()
    if ONLY_NEURON is not None:
        neuron_ids = [n for n in neuron_ids if n == ONLY_NEURON]
    print(f"{DATE} round {ROUND_NO}: {len(neuron_ids)} channel(s)/unit(s) to plot "
          f"(pre-stimulus {int(PRE_STIMULUS_TIME * 1000)} ms)")

    save_dir = Path(PROJECT_BASE_PATH) / SUBJECT_MONKEY / "raster_plots" / cache_subdir
    for neuron_id in neuron_ids:
        neuron_df = df[df["NeuronID"] == neuron_id]
        if len(neuron_df) < MIN_TRIALS:
            print(f"  skip {neuron_id}: only {len(neuron_df)} trials")
            continue
        # Manually-sorted units (Channel carries a "_Unit" suffix) have real
        # pre-stimulus spikes from sorted_spikes.pkl; unsorted channels come from the
        # clipped compiled.pkl and have none. Show the pre-stimulus axis only for the
        # manual units — plot unsorted channels from t=0 so their empty [-pre, 0) band
        # isn't mistaken for silence (use plot_prestim_mua_rasters.py for unsorted pre-stim).
        is_manual = "_Unit" in neuron_id
        pre = PRE_STIMULUS_TIME if is_manual else 0.0
        n_spikes = sum(len(s) for s in neuron_df["SpikeTimes"])
        print(f"  plotting {neuron_id} [{'manual' if is_manual else 'unsorted'}] "
              f"— {len(neuron_df)} trials, {n_spikes} spikes")
        pre_label = f"pre-stim {int(pre * 1000)} ms" if pre > 0 else "no pre-stim (unsorted)"
        save_path = str(save_dir / f"{neuron_id}.png") if SAVE else None
        plot_stacked_raster(
            neuron_df,
            xlim=XLIM,
            pre_stimulus_time=pre,
            columns=COLUMNS,
            exclude_monkeys=EXCLUDE,
            annotate=LABEL,
            title=f"Mixed manual/unsorted raster ({pre_label}): {neuron_id}",
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
