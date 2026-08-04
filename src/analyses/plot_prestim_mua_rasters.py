"""
plot_prestim_mua_rasters.py — stacked pre-stimulus rasters for the UNSORTED channels,
using offline MUA (multi-unit activity) detected from amplifier.dat.

For unsorted channels (no manual/SI sort), channel-level MUA is the natural signal.
This builds the pre-stimulus MUA cache on first run (via mua_peristim_builder, which
uses the session's own digitalin for correct epochs on stitched sessions) and plots
one stacked raster per channel — same layout as the SI-sorted / mixed drivers.

HOW TO RUN (PyCharm): open this file and click Run — edit the CONFIG block below.
Needs the raw Intan session on disk (amplifier.dat etc.), so run it on the rig.
"""
from pathlib import Path

from analyses.raster_plotting import plot_stacked_raster
from data_access.mua_peristim_builder import build_mua_peristim_cache
from project_util import DATA_BASE_PATH, SUBJECT_MONKEY

# ===== CONFIG — edit me =======================================================
DATE = "2023-09-26"
ROUND_NO = 2
PRE_STIMULUS_TIME = 1.0          # seconds of pre-stimulus baseline

# MUA detection (same detector as the grant ThresholdMUASpikeSource)
NOISE_METHOD = "mad"             # 'mad' = median(|v|)/0.6745, or 'rms'
THRESHOLD_MULTIPLIER = 4.0
REFRACTORY_MS = 1.0
FORCE_REBUILD = False            # True = re-detect even if the variant cache exists

XLIM = 2.2                                     # right edge of the time axis (seconds after onset)
COLUMNS = [["Zombies", "Best Frans"], ["Instigators", "Stranger Things"]]
EXCLUDE = []           # monkey names to leave out entirely, e.g. ["144H", "81G"]
LABEL = "trials"       # next to each name: "trials" (count), "rank" (#N), "both", or "none"
SAVE = False           # False = show each figure interactively; True = write PNGs and move on
MIN_TRIALS = 5         # skip channels with fewer trials
ONLY_CHANNELS = None   # e.g. ["Channel.C_005", "Channel.C_026"] to restrict; None = all channels
# ==============================================================================


def main():
    df, path = build_mua_peristim_cache(
        DATE, ROUND_NO, PRE_STIMULUS_TIME,
        noise_method=NOISE_METHOD, threshold_multiplier=THRESHOLD_MULTIPLIER,
        refractory_ms=REFRACTORY_MS, force=FORCE_REBUILD)
    if df is None:
        print(f"No MUA data for {DATE} round {ROUND_NO}.")
        return

    neuron_ids = df["NeuronID"].dropna().unique().tolist()
    if ONLY_CHANNELS is not None:
        keep = {str(c) for c in ONLY_CHANNELS}
        neuron_ids = [n for n in neuron_ids if any(n.endswith(c) for c in keep)]
    print(f"{DATE} round {ROUND_NO}: {len(neuron_ids)} channel(s) to plot "
          f"(MUA {NOISE_METHOD} x{THRESHOLD_MULTIPLIER}, pre-stimulus {int(PRE_STIMULUS_TIME * 1000)} ms)")

    save_dir = Path(DATA_BASE_PATH) / SUBJECT_MONKEY / "raster_plots" / f"mua_pre{int(PRE_STIMULUS_TIME * 1000)}ms"
    for neuron_id in neuron_ids:
        neuron_df = df[df["NeuronID"] == neuron_id]
        if len(neuron_df) < MIN_TRIALS:
            print(f"  skip {neuron_id}: only {len(neuron_df)} trials")
            continue
        n_spikes = sum(len(s) for s in neuron_df["SpikeTimes"])
        print(f"  plotting {neuron_id} — {len(neuron_df)} trials, {n_spikes} MUA spikes")
        save_path = str(save_dir / f"{neuron_id}.png") if SAVE else None
        plot_stacked_raster(
            neuron_df,
            xlim=XLIM,
            pre_stimulus_time=PRE_STIMULUS_TIME,
            columns=COLUMNS,
            exclude_monkeys=EXCLUDE,
            annotate=LABEL,
            title=f"MUA raster ({NOISE_METHOD} x{THRESHOLD_MULTIPLIER}, pre-stim {int(PRE_STIMULUS_TIME * 1000)} ms): {neuron_id}",
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
