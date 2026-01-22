from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple
import pandas as pd


from analyses.response_window_finder.threshold_window_detection import detect_response_windows_for_session
from analyses.single_neuron_analysis import run_permutation_anova, run_permutation_kruskal_wallis
from analyses.spike_count import (
    prepare_binned_spike_data,
    aggregate_trial_level,
    extract_spike_counts_from_windows
)
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.response_window_analysis import run_permutation_kruskal_wallis_by_window, run_permutation_anova_by_window
from analyses.spike_source import SpikeSource, SISortedSpikeSource, MixedManualSpikeSource


@dataclass(frozen=True)
class PreprocessConfig:
    group_name: str = "Zombies"
    bin_size: float = 0.05
    analysis_cache_dir: Path = Path("/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/")
    save: bool = True


def iter_sessions_from_metadata(metadata: pd.DataFrame) -> Iterable[Tuple[str, int]]:
    """
    Expected metadata columns:
      - 'Date' (datetime-like)
      - 'Round No.' (int-like)
    """
    for _, r in metadata.iterrows():
        date = str(r["Date"].strftime("%Y-%m-%d"))
        round_no = int(r["Round No."])
        yield date, round_no


def _save_pickle(df: pd.DataFrame, out_path: Path, save: bool) -> None:
    if not save:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(str(out_path))


def _select_all_significant_neurons(
    metadata: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
    test_runner: Callable[..., Tuple[pd.DataFrame, pd.DataFrame]],
    out_name: str,
) -> pd.DataFrame:
    sig_parts = []

    for date, round_no in iter_sessions_from_metadata(metadata):
        binned = prepare_binned_spike_data(
            date,
            round_no,
            cfg.bin_size,
            source=source
        )

        if binned is None or binned.empty:
            print(f"Skipping {date} Round {round_no}: no good neurons")
            continue

        group_df = binned[binned["MonkeyGroup"] == cfg.group_name]
        if group_df.empty:
            continue

        trial_df = aggregate_trial_level(group_df)
        _, sig_df = test_runner(trial_df, category_col="MonkeyName", plot=False)

        if sig_df is not None and not sig_df.empty:
            sig_parts.append(sig_df)

    out = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    _save_pickle(out, cfg.analysis_cache_dir / out_name, cfg.save)
    return out


def select_all_significant_neurons_using_pKW(
    metadata: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
) -> pd.DataFrame:
    return _select_all_significant_neurons(
        metadata,
        cfg,
        source=source,
        test_runner=run_permutation_kruskal_wallis,
        out_name=f"{source.name}_{cfg.group_name}_significant_neurons_pKW_passed.pkl",
    )


def select_all_significant_neurons_using_pANOVA(
    metadata: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
) -> pd.DataFrame:
    return _select_all_significant_neurons(
        metadata,
        cfg,
        source=source,
        test_runner=run_permutation_anova,
        out_name=f"{source.name}_{cfg.group_name}_significant_neurons_pANOVA_passed.pkl",
    )


def detect_all_response_windows_for_all_neurons(
    metadata: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
) -> pd.DataFrame:
    parts = []
    use_sorted = source.name == "si_sorted"

    for date, round_no in iter_sessions_from_metadata(metadata):
        df = detect_response_windows_for_session(
            date,
            round_no,
            source = source,
            monkey_group=cfg.group_name,
            plot=False,
        )
        if df is None or getattr(df, "empty", True):
            continue
        parts.append(df)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty and "NeuronID" in out.columns:
        out = out.sort_values(by=["NeuronID"])

    out_name = f"{source.name}_{cfg.group_name}_response_windows.pkl"
    _save_pickle(out, cfg.analysis_cache_dir / out_name, cfg.save)
    print("all windows")
    print(out.shape)
    print(out)
    return out


def detect_significant_windows_using_pANOVA(
    windows: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    ) -> pd.DataFrame:
    spike_counts_windows = extract_spike_counts_from_windows(windows, source=source)

    group_df = spike_counts_windows[spike_counts_windows["MonkeyGroup"] == cfg.group_name]
    if group_df.empty:
        out = pd.DataFrame()
        _save_pickle(out, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pANOVA_passed.pkl", cfg.save)
        return out

    panova_results, _ = run_permutation_anova_by_window(
        group_df,
        category_col="MonkeyName",
        neuron_col="NeuronID",
        count_col="SpikeCount",
        window_start_col="WindowStart_ms",
        window_end_col="WindowEnd_ms",
        n_permutations=n_permutations,
        alpha=alpha,
        plot=False,
    )

    sig = panova_results[panova_results["p-value"]<alpha] if not panova_results.empty else pd.DataFrame()
    _save_pickle(sig, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pANOVA_passed.pkl", cfg.save)
    return sig


def detect_significant_windows_using_pKW(
    windows: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    source: SpikeSource,
    n_permutations: int = 10000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    spike_counts_windows = extract_spike_counts_from_windows(windows, source=source)

    group_df = spike_counts_windows[spike_counts_windows["MonkeyGroup"] == cfg.group_name]
    if group_df.empty:
        out = pd.DataFrame()
        _save_pickle(out, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl", cfg.save)
        return out

    _, significant_df = run_permutation_kruskal_wallis_by_window(
        group_df,
        category_col="MonkeyName",
        neuron_col="NeuronID",
        count_col="SpikeCount",
        window_start_col="WindowStart_ms",
        window_end_col="WindowEnd_ms",
        n_permutations=n_permutations,
        alpha=alpha,
        plot=False,
    )

    _save_pickle(significant_df, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl", cfg.save)
    print("significant windows based on pKW")
    print(significant_df.shape)
    print(significant_df)
    return significant_df


if __name__ == "__main__":
    analysis_cache_dir = "/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/"

    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()

    cfg = PreprocessConfig(group_name="Zombies", bin_size=0.05, analysis_cache_dir=Path(analysis_cache_dir), save=False)

    # pick ONE source for the whole run
    # source: SpikeSource = SISortedSpikeSource()
    source: SpikeSource = MixedManualSpikeSource(curated_channels_only=True)

    # 1) significant neurons
    # sig_neurons = select_all_significant_neurons_using_pKW(metadata, cfg, source=source)

    # 2) detect windows
    windows = detect_all_response_windows_for_all_neurons(metadata, cfg, source=source)

    source: SpikeSource = SISortedSpikeSource()
    # 3) significant windows
    sig_windows_kw = detect_significant_windows_using_pKW(windows, cfg, source=source)
