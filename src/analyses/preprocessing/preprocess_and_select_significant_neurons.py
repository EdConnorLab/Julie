from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple
import pandas as pd
from statsmodels.stats.multitest import multipletests

from analyses.response_window_finder.threshold_window_detection import detect_response_windows_for_session
from analyses.single_neuron_analysis import run_permutation_anova, run_permutation_kruskal_wallis, \
    multiple_comparison_test
from analyses.spike_count import (
    prepare_binned_spike_data,
    aggregate_trial_level,
    extract_spike_counts_from_windows
)
from analyses.data_readers.recording_metadata_reader import RecordingMetadataReader
from analyses.response_window_analysis import run_permutation_kruskal_wallis_by_window, run_permutation_anova_by_window
from data_access.spike_source import SpikeSource, SISortedSpikeSource


@dataclass(frozen=True)
class PreprocessConfig:
    group_name: str = "Zombies"
    bin_size: float = 0.05
    analysis_cache_dir: Path = Path("/Cortana/analysis_cache/")
    save: bool = True


def iter_sessions_from_metadata(metadata: pd.DataFrame) -> Iterable[Tuple[str, int]]:
    """
    Expected metadata columns:
      - 'Date' (datetime-like)
      - 'Round No.' (int-like)
    """
    for _, r in metadata.iterrows():
        date_val = r["Date"]
        date_val = pd.to_datetime(date_val)
        date = date_val.strftime("%Y-%m-%d")
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
    out_name_sig: str,
    out_name_all: str,
    alpha: float = 0.05,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    all_parts = []

    for date, round_no in iter_sessions_from_metadata(metadata):
        binned = prepare_binned_spike_data(
            date,
            round_no,
            cfg.bin_size,
            source=source
        )

        if binned is None or binned.empty:
            print(f"Skipping {date} Round {round_no}: no neurons found")
            continue

        group_df = binned[binned["MonkeyGroup"] == cfg.group_name]
        if group_df.empty:
            continue

        trial_df = aggregate_trial_level(group_df)
        results_df, _ = test_runner(trial_df, category_col="MonkeyName",
                                    n_permutations=n_permutations, random_state = random_state, plot=False)

        if results_df is not None and not results_df.empty:
            results_df["Date"] = date
            results_df["Round No."] = round_no
            all_parts.append(results_df)

    if not all_parts:
        print("No valid neurons found.")
        return pd.DataFrame(), pd.DataFrame()

    all_results = pd.concat(all_parts, ignore_index=True)

    # FDR correction on ALL p-values
    all_results = multiple_comparison_test(all_results)
    # Save all results (includes both uncorrected and FDR-corrected)
    _save_pickle(all_results, cfg.analysis_cache_dir / out_name_all, cfg.save)

    # Significant based on uncorrected permutation p-value
    sig_results = all_results[all_results["p-value"] < alpha]
    _save_pickle(sig_results, cfg.analysis_cache_dir / out_name_sig, cfg.save)

    # Count FDR survivors for reporting
    sig_col = [c for c in all_results.columns if c.endswith("_significant")][0]
    n_fdr = all_results[sig_col].sum()

    print(f"Total neurons tested: {len(all_results)}")
    print(f"Significant after permutation test (p < {alpha}): {len(sig_results)}")
    print(f"Significant after FDR correction: {n_fdr}")

    return all_results, sig_results


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
        out_name_all=f"{source.name}_{cfg.group_name}_all_neurons_pKW.pkl",
        out_name_sig=f"{source.name}_{cfg.group_name}_significant_neurons_pKW_passed.pkl",
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
        out_name_all=f"{source.name}_{cfg.group_name}_all_neurons_pANOVA.pkl",
        out_name_sig=f"{source.name}_{cfg.group_name}_significant_neurons_pANOVA_passed.pkl",
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
        return out, out

    results_df, _ = run_permutation_anova_by_window(
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

    # FDR correction on ALL p-values
    results_df = multiple_comparison_test(results_df)
    # Save all results (includes both uncorrected and FDR-corrected)
    _save_pickle(results_df, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_all_windows_pANOVA_tested.pkl",
                 cfg.save)

    # Significant based on uncorrected permutation p-value
    sig_results = results_df[results_df["p-value"] < alpha]
    _save_pickle(sig_results,
                 cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pANOVA_passed.pkl",
                 cfg.save)
    print("significant windows based on pANOVA")
    print(sig_results.shape)
    print(sig_results)
    # Count FDR survivors for reporting
    sig_col = [c for c in results_df.columns if c.endswith("_significant")][0]
    n_fdr = results_df[sig_col].sum()

    print(f"Total windows tested: {len(results_df)}")
    print(f"Significant windows after permutation test (p < {alpha}): {len(sig_results)}")
    print(f"Significant windows after FDR correction: {n_fdr}")

    return results_df, sig_results


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

    results_df, _ = run_permutation_kruskal_wallis_by_window(
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

    # FDR correction on ALL p-values
    results_df = multiple_comparison_test(results_df)
    # Save all results (includes both uncorrected and FDR-corrected)
    _save_pickle(results_df, cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_all_windows_pKW_tested.pkl",
                 cfg.save)

    # Significant based on uncorrected permutation p-value
    sig_results = results_df[results_df["p-value"] < alpha]
    _save_pickle(sig_results,
                 cfg.analysis_cache_dir / f"{source.name}_{cfg.group_name}_significant_windows_pKW_passed.pkl",
                 cfg.save)
    print("significant windows based on pKW")
    print(sig_results.shape)
    print(sig_results)
    # Count FDR survivors for reporting
    sig_col = [c for c in results_df.columns if c.endswith("_significant")][0]
    n_fdr = results_df[sig_col].sum()

    print(f"Total windows tested: {len(results_df)}")
    print(f"Significant windows after permutation test (p < {alpha}): {len(sig_results)}")
    print(f"Significant windows after FDR correction: {n_fdr}")

    return results_df, sig_results


if __name__ == "__main__":
    analysis_cache_dir = "/Cortana/analysis_cache/"

    reader = RecordingMetadataReader()
    metadata = reader.get_metadata_for_preliminary_analysis()

    cfg = PreprocessConfig(group_name="Zombies", bin_size=0.05, analysis_cache_dir=Path(analysis_cache_dir), save=True)

    # pick ONE source for the whole run
    source: SpikeSource = SISortedSpikeSource(cache_subdir="sorted_spike_cache_filtered", pre_filtered=True)
    # source: SpikeSource = MixedManualSpikeSource(curated_channels_only=True)

    # # 1) significant neurons
    # all_neurons, sig_neurons = select_all_significant_neurons_using_pKW(metadata, cfg, source=source)
    all_neurons, sig_neurons = select_all_significant_neurons_using_pANOVA(metadata, cfg, source=source)

    #2) detect windows
    windows = detect_all_response_windows_for_all_neurons(metadata, cfg, source=source)

    # 3) significant windows
    res_window_kw, sig_windows_kw = detect_significant_windows_using_pKW(windows, cfg, source=source)
    res_window_panova, sig_windows_panova = detect_significant_windows_using_pANOVA(windows, cfg, source=source)
