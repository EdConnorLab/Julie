"""
validate_si_sorting.py
======================
Experiment 1 of the sorting-validation series.

GOAL (hypothesis-driven):
    For a given recording session, decide whether the 3-sorter *consensus* units
    that feed your downstream analysis are real, well-isolated single neurons, and
    whether the pipeline's channel labelling / spike assignment holds up.

WHAT IT DOES (each maps to a hypothesis):
    H1  per-unit waveform panel (template + raw-waveform overlay on peak channel,
        spatial footprint, auto-correlogram)  + a quality-metrics table
        (snr, isi_violations, rp_contamination, firing_rate, presence_ratio,
         amplitude_cutoff, num_spikes)
    H2  split detection  -> template-similarity matrix + get_potential_auto_merge
        merge/contamination -> flagged from ISI/rp metrics
    H3  cross-sorter consistency -> KS4/MS5/TDC template overlay + peak-channel check
    H4  channel map -> prints SI recording-index  ->  Intan native_order / name,
        so you can confirm whether "C_018" in a NeuronID is the same physical
        electrode as "C_018" in the manual / grant list.
    (bug check) verifies KS4 get_unit_ids()[uid] == uid, i.e. whether the positional
        indexing used in analyze_sorted_spikes.find_consensus_units is safe.

This script does NOT touch the manual sort (that is Experiment 2). It needs only:
    amplifier.dat, info.rhd, analyzer_KS4_binary / _MS5_ / _TDC_ for the session.

RUN (in your repo env, where spikeinterface + clat + your package are importable):
    python validate_si_sorting.py --date 2023-11-20 --round 1
    python validate_si_sorting.py --date 2023-11-20 --round 2

Outputs land in <intan_dir>/_sorting_validation/ :
    quality_metrics_<sorter>.csv, consensus_units.csv, channel_map.txt,
    unit_<uid>_panel.png, split_candidates.txt (+ pair plots), summary.txt

NOTE: written against the SpikeInterface API your pipeline already uses
(si.load_sorting_analyzer, analyzer.compute, get_template_extremum_channel,
compare_multiple_sorters). Version-fragile calls (quality metrics, auto-merge)
are wrapped in try/except and will print what failed rather than crash.
I could not execute this here (SI not installed in my sandbox) — run it and send
back any traceback / the PNGs.
"""
import argparse
import os
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.comparison as sc

# --- reuse YOUR pipeline code so paths/preprocessing are identical -------------
from spikesorting.sort_spikes.sort_spikes import (
    build_intan_session_path,
    load_and_preprocess_recording,
    get_recording_session_info,
)
from compile.compile_common import SUBJECT_MONKEY

SORTER_FOLDERS = {"ks4": "analyzer_KS4_binary",
                  "ms5": "analyzer_MS5_binary",
                  "tdc": "analyzer_TDC_binary"}

# extensions we need for metrics / plots (some already saved by sort_spikes.py)
NEEDED_EXT = ["random_spikes", "waveforms", "templates", "noise_levels",
              "spike_amplitudes", "correlograms", "template_similarity"]
QM_NAMES = ["snr", "isi_violation", "rp_violation", "firing_rate",
            "presence_ratio", "amplitude_cutoff", "num_spikes"]


# ------------------------------------------------------------------ loading ---
def load_analyzers(intan_dir, recording):
    analyzers = {}
    for key, folder in SORTER_FOLDERS.items():
        path = os.path.join(intan_dir, folder)
        az = si.load_sorting_analyzer(path)
        # binary_folder analyzers may load without a recording attached; re-attach
        # so noise_levels / spike_amplitudes can be (re)computed for SNR etc.
        try:
            if not az.has_recording():
                az.set_temporary_recording(recording)
        except Exception:
            try:
                az.set_temporary_recording(recording)
            except Exception:
                pass
        analyzers[key] = az
    return analyzers


def ensure_extensions(az):
    have = set(az.get_saved_extension_names()) | set(az.get_loaded_extension_names())
    todo = [e for e in NEEDED_EXT if e not in have]
    if todo:
        try:
            az.compute(todo)
        except Exception:
            # compute one-by-one so a single failing ext doesn't kill the rest
            for e in todo:
                try:
                    az.compute(e)
                except Exception as ex:
                    print(f"    [warn] could not compute '{e}': {ex}")


def quality_metrics(az):
    try:
        az.compute("quality_metrics", metric_names=QM_NAMES)
        qm = az.get_extension("quality_metrics").get_data()
        return qm
    except Exception as ex:
        print(f"    [warn] quality_metrics failed ({ex}); falling back to ISI only")
        # minimal fallback: ISI-violation fraction per unit from the spike train
        srt = az.sorting
        fs = srt.get_sampling_frequency()
        rows = {}
        for uid in srt.get_unit_ids():
            st = srt.get_unit_spike_train(unit_id=uid) / fs
            if st.size >= 2:
                isis = np.diff(np.sort(st))
                frac = float(np.mean(isis < 0.002))
            else:
                frac = np.nan
            rows[uid] = {"isi_violation_fraction_<2ms": frac,
                         "num_spikes": int(st.size)}
        return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------- consensus map --
def consensus_units(analyzers):
    """Return list of dicts: {'ks4':uid,'ms5':uid,'tdc':uid} for 3-sorter agreement,
    plus the multicomparison object (for agreement scores)."""
    comp = sc.compare_multiple_sorters(
        sorting_list=[analyzers["tdc"].sorting,
                      analyzers["ms5"].sorting,
                      analyzers["ks4"].sorting],
        name_list=["tdc", "ms5", "ks4"],
    )
    agree = comp.get_agreement_sorting(minimum_agreement_count=3)
    maps = agree.get_property("unit_ids")
    if maps is None:
        maps = []
    maps = list(maps)
    return maps, comp


# --------------------------------------------------------------- plotting -----
def peak_index(az, uid):
    d = si.get_template_extremum_channel(az, peak_sign="both", outputs="index")
    return int(d[uid])


def unit_panel(analyzers, cmap, out_png, note=""):
    """One consensus unit: KS4 template+overlay+footprint+ACG, plus MS5/TDC template."""
    az = analyzers["ks4"]
    uid = cmap["ks4"]
    fs = az.sorting.get_sampling_frequency()
    temp_ext = az.get_extension("templates")
    tmpl = temp_ext.get_unit_template(unit_id=uid)              # (nsamp, nchan)
    pk = int(np.argmax(np.max(np.abs(tmpl), axis=0)))
    t_ms = (np.arange(tmpl.shape[0]) - tmpl.shape[0] // 2) / fs * 1000.0

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"consensus unit  KS4={cmap.get('ks4')} MS5={cmap.get('ms5')} "
                 f"TDC={cmap.get('tdc')}   peak_ch(idx)={pk}   {note}")

    # (1) raw-waveform overlay on peak channel
    try:
        wf = az.get_extension("waveforms").get_waveforms_one_unit(unit_id=uid)  # (n,nsamp,nchan)
        sel = wf[np.random.choice(wf.shape[0], min(60, wf.shape[0]), replace=False)]
        ax[0].plot(t_ms, sel[:, :, pk].T, color="0.7", lw=0.4)
    except Exception as ex:
        ax[0].text(0.1, 0.5, f"waveforms n/a\n{ex}", transform=ax[0].transAxes)
    ax[0].plot(t_ms, tmpl[:, pk], "k", lw=2)
    ax[0].set_title("peak-channel waveforms + template"); ax[0].set_xlabel("ms")

    # (2) spatial footprint (template on every channel)
    off = np.arange(tmpl.shape[1]) * (np.abs(tmpl).max() * 1.2)
    ax[1].plot(tmpl + off, color="C0", lw=0.8)
    ax[1].set_title("spatial footprint (all channels)"); ax[1].set_yticks([])

    # (3) cross-sorter template overlay on each sorter's own peak channel
    for key, col in [("ks4", "k"), ("ms5", "C1"), ("tdc", "C2")]:
        try:
            a2 = analyzers[key]; u2 = cmap[key]
            tt = a2.get_extension("templates").get_unit_template(unit_id=u2)
            p2 = int(np.argmax(np.max(np.abs(tt), axis=0)))
            tt = tt[:, p2]
            tt = tt / (np.abs(tt).max() + 1e-9)
            ax[2].plot(np.linspace(t_ms[0], t_ms[-1], tt.size), tt, col, lw=1.6,
                       label=f"{key} u{u2}")
        except Exception as ex:
            print(f"    [warn] cross-sorter template {key}: {ex}")
    ax[2].legend(fontsize=8); ax[2].set_title("cross-sorter template (norm.)"); ax[2].set_xlabel("ms")

    # (4) auto-correlogram
    try:
        cc = az.get_extension("correlograms")
        ccg, bins = cc.get_data()
        ids = list(az.sorting.get_unit_ids())
        i = ids.index(uid)
        centers = (bins[:-1] + bins[1:]) / 2
        ax[3].bar(centers, ccg[i, i], width=(bins[1] - bins[0]))
        ax[3].axvline(0, color="r", lw=0.6)
    except Exception as ex:
        ax[3].text(0.1, 0.5, f"ACG n/a\n{ex}", transform=ax[3].transAxes)
    ax[3].set_title("auto-correlogram"); ax[3].set_xlabel("ms")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- split detect --
def detect_splits(az, out_dir, summary):
    """Flag KS4 unit pairs that look like one oversplit neuron."""
    try:
        from spikeinterface.curation import get_potential_auto_merge
        try:
            pairs = get_potential_auto_merge(az, preset="similarity_correlograms")
        except TypeError:
            pairs = get_potential_auto_merge(az)  # older signature
        summary.append(f"auto-merge suggested pairs (possible oversplits): {list(pairs)}")
        with open(os.path.join(out_dir, "split_candidates.txt"), "w") as f:
            f.write("Pairs that get_potential_auto_merge thinks are one neuron:\n")
            for p in pairs:
                f.write(f"  {p}\n")
        return list(pairs)
    except Exception as ex:
        summary.append(f"[warn] auto-merge unavailable ({ex}); "
                       f"inspect template_similarity matrix manually")
        return []


# ------------------------------------------------------------------- driver ---
def run(date, round_no, monkey=SUBJECT_MONKEY, cells_of_interest=None):
    intan_dir = build_intan_session_path(date, round_no, monkey)
    out_dir = os.path.join(intan_dir, "_sorting_validation")
    os.makedirs(out_dir, exist_ok=True)
    summary = [f"=== {date} round {round_no} ==="]

    rec = load_and_preprocess_recording(intan_dir)
    analyzers = load_analyzers(intan_dir, rec)

    # H4: channel map  (SI recording index -> Intan native_order / name)
    _, channels = get_recording_session_info(intan_dir)
    with open(os.path.join(out_dir, "channel_map.txt"), "w") as f:
        f.write("SI_index\tnative_order\tcustom_name\n")
        for i, ch in enumerate(channels):
            f.write(f"{i}\t{ch.get('native_order')}\t{ch.get('custom_channel_name')}\n")
    summary.append("channel_map.txt written -> confirm SI 'C_xxx' == Intan name "
                   "before trusting cross-list / manual matching")

    # bug check on KS4 positional indexing
    ks4_ids = list(analyzers["ks4"].sorting.get_unit_ids())
    contiguous = all(ks4_ids[i] == i for i in range(len(ks4_ids)))
    summary.append(f"KS4 unit ids contiguous 0..N-1 ? {contiguous}  "
                   f"(if False, get_unit_ids()[uid] indexing in "
                   f"analyze_sorted_spikes is UNSAFE)")

    # metrics per sorter
    for key, az in analyzers.items():
        ensure_extensions(az)
        qm = quality_metrics(az)
        try:
            qm.insert(0, "peak_ch_idx",
                      [peak_index(az, u) for u in az.sorting.get_unit_ids()])
        except Exception:
            pass
        qm.to_csv(os.path.join(out_dir, f"quality_metrics_{key}.csv"))
    summary.append("quality_metrics_{ks4,ms5,tdc}.csv written")

    # consensus units
    maps, comp = consensus_units(analyzers)
    summary.append(f"3-sorter consensus units: {len(maps)}")
    rows = []
    for cmap in maps:
        note = ""
        try:
            pk = peak_index(analyzers["ks4"], cmap["ks4"])
            name = f"C_{pk:03d}"
            if cells_of_interest and name in cells_of_interest:
                note = f"<< matches cell-of-interest {name}"
            rows.append({"ks4": cmap["ks4"], "ms5": cmap["ms5"], "tdc": cmap["tdc"],
                         "ks4_peak_ch_idx": pk, "note": note})
            unit_panel(analyzers, cmap,
                       os.path.join(out_dir, f"unit_ks4_{cmap['ks4']}_panel.png"), note)
        except Exception:
            summary.append("panel error:\n" + traceback.format_exc())
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "consensus_units.csv"), index=False)

    # split / merge
    detect_splits(analyzers["ks4"], out_dir, summary)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nAll outputs -> {out_dir}")


# ============================================================================
#  EDIT HERE, then just hit  ▶ Run  in PyCharm  (no terminal / no arguments).
#  Each entry: (date, round, [channels to flag if a consensus unit peaks there])
#    r1: your list has C_018 ; grant multiunit is C_013
#    r2: grant multiunit is C_006
# ============================================================================
SESSIONS = [
    ("2023-11-20", 1, ["C_018", "C_013"]),
    # ("2023-11-20", 2, ["C_006"]),
]
MONKEY = SUBJECT_MONKEY   # subject folder name ("Cortana")

if __name__ == "__main__":
    for _date, _round, _cells in SESSIONS:
        print(f"\n########## {_date} round {_round} ##########")
        try:
            run(_date, _round, MONKEY, cells_of_interest=_cells)
        except Exception:
            print(traceback.format_exc())   # keep going to the next session
