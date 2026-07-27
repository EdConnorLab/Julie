"""
scoring.py — score detected windows against the user's hand-annotated truth.

For one cell and one method we match predicted windows to truth windows by
temporal overlap (IoU), then tally:

* **hit**            a truth window matched by some predicted window (IoU >= thr)
* **miss**           a truth window with no matching prediction
* **false alarm**    a predicted window matching no truth window
* **onset error**    |pred_start - truth_start| for matched pairs (ms)
* **IoU**            intersection-over-union for matched pairs

Cells the user marks as "no response" (empty truth) score a **correct rejection**
when a method predicts nothing, and a false alarm per spurious window.

Aggregated per method across cells: precision, recall, F1, mean IoU, mean onset
error, and correct-rejection rate — the scoreboard that picks the winner.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Window = Tuple[float, float]


def iou(a: Window, b: Window) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def overlap(a: Window, b: Window) -> float:
    """Overlap duration (s) of two windows (0 if disjoint)."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def coverage(truth: List[Window], pred: List[Window]) -> float:
    """Fraction of the total truth duration covered by ANY predicted window.

    Rewards capturing the response even when a method splits it into pieces;
    complements strict IoU. NaN when there is no truth window.
    """
    total = sum(hi - lo for lo, hi in truth)
    if total <= 0:
        return float("nan")
    covered = 0.0
    for lo, hi in truth:
        # union of predicted overlaps with this truth window
        segs = sorted((max(lo, p0), min(hi, p1)) for p0, p1 in pred if overlap((lo, hi), (p0, p1)) > 0)
        cur = lo
        for s0, s1 in segs:
            if s1 <= cur:
                continue
            covered += s1 - max(s0, cur)
            cur = max(cur, s1)
    return covered / total


def match_windows(truth: List[Window], pred: List[Window], iou_thresh: float):
    """Greedy IoU matching. Returns (matches, missed_truth_idx, fa_pred_idx).

    ``matches`` is a list of ``(t_idx, p_idx, iou, onset_err_s)``.
    """
    pairs = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            v = iou(t, p)
            if v >= iou_thresh:
                pairs.append((v, ti, pi))
    pairs.sort(reverse=True)  # highest IoU first
    used_t, used_p, matches = set(), set(), []
    for v, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi)
        matches.append((ti, pi, v, abs(truth[ti][0] - pred[pi][0])))
    missed = [ti for ti in range(len(truth)) if ti not in used_t]
    fa = [pi for pi in range(len(pred)) if pi not in used_p]
    return matches, missed, fa


def score_cell(cell_key: str, method: str, truth: List[Window],
               pred: List[Window], iou_thresh: float) -> dict:
    matches, missed, fa = match_windows(truth, pred, iou_thresh)
    no_resp = len(truth) == 0
    # "off-target" false alarms: predicted windows that overlap NO truth window at
    # all (a fairer count than strict FA, which also penalises sub-windows that
    # land inside the response but weren't the single best match).
    off_target = sum(1 for p in pred if not any(overlap(p, t) > 0 for t in truth))
    return {
        "cell_key": cell_key,
        "method": method,
        "n_truth": len(truth),
        "n_pred": len(pred),
        "hits": len(matches),
        "misses": len(missed),
        "false_alarms": len(fa),
        "off_target_fa": off_target,
        "coverage": coverage(truth, pred),
        "correct_rejection": bool(no_resp and len(pred) == 0),
        "mean_iou": float(np.mean([m[2] for m in matches])) if matches else np.nan,
        "mean_onset_err_ms": float(np.mean([m[3] for m in matches]) * 1000) if matches else np.nan,
    }


def score_all(
    results_by_cell: Dict[str, Dict[str, List[Window]]],
    truth_by_cell: Dict[str, List[Window]],
    methods: Sequence[str],
    *,
    iou_thresh: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every (cell, method) and aggregate per method.

    ``results_by_cell``: cell_key -> {method -> [windows]}.
    ``truth_by_cell``:   cell_key -> [windows] (only scored cells need be present).
    Returns ``(per_cell_df, scoreboard_df)``.
    """
    rows = []
    for cell_key, truth in truth_by_cell.items():
        preds = results_by_cell.get(cell_key, {})
        for method in methods:
            rows.append(score_cell(cell_key, method, truth,
                                   preds.get(method, []), iou_thresh))
    per_cell = pd.DataFrame(rows)

    agg_rows = []
    for method in methods:
        sub = per_cell[per_cell["method"] == method]
        hits = int(sub["hits"].sum())
        misses = int(sub["misses"].sum())
        fa = int(sub["false_alarms"].sum())
        n_noresp = int((sub["n_truth"] == 0).sum())
        cr = int(sub["correct_rejection"].sum())
        off_target = int(sub["off_target_fa"].sum())
        precision = hits / (hits + fa) if (hits + fa) else np.nan
        recall = hits / (hits + misses) if (hits + misses) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else np.nan)
        agg_rows.append({
            "method": method,
            "hits": hits,
            "misses": misses,
            "false_alarms": fa,
            "off_target_fa": off_target,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "mean_coverage": float(sub["coverage"].mean(skipna=True)),
            "mean_iou": float(sub["mean_iou"].mean(skipna=True)),
            "mean_onset_err_ms": float(sub["mean_onset_err_ms"].mean(skipna=True)),
            "no_response_cells": n_noresp,
            "correct_rejections": cr,
        })
    scoreboard = pd.DataFrame(agg_rows).sort_values(
        ["F1", "recall"], ascending=False, na_position="last").reset_index(drop=True)
    return per_cell, scoreboard


def _fmt_windows(windows: List[Window]) -> str:
    return "; ".join(f"{int(a * 1000)}-{int(b * 1000)}" for a, b in windows) or "-"


def diagnostic_table(
    results_by_cell: Dict[str, Dict[str, List[Window]]],
    truth_by_cell: Dict[str, List[Window]],
    methods: Sequence[str],
    *,
    meta_by_cell: Optional[Dict[str, dict]] = None,
    iou_thresh: float = 0.3,
) -> pd.DataFrame:
    """One row per cell: the truth window(s) next to EACH method's raw detected
    window(s), with per-method hit / off-target / coverage. Makes over-detection
    (a response split into several windows) visible at a glance.
    """
    rows = []
    for cell_key, truth in truth_by_cell.items():
        preds = results_by_cell.get(cell_key, {})
        row = {"cell_key": cell_key}
        if meta_by_cell and cell_key in meta_by_cell:
            row.update(meta_by_cell[cell_key])
        row["truth_ms"] = _fmt_windows(truth)
        for m in methods:
            pw = preds.get(m, [])
            matches, missed, fa = match_windows(truth, pw, iou_thresh)
            off = sum(1 for p in pw if not any(overlap(p, t) > 0 for t in truth))
            row[f"{m}_windows_ms"] = _fmt_windows(pw)
            row[f"{m}_hit"] = len(matches)
            row[f"{m}_fa"] = len(fa)
            row[f"{m}_offtarget"] = off
            row[f"{m}_coverage"] = round(coverage(truth, pw), 2) if truth else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scoreboard(scoreboard: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart of precision / recall / F1 per method."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = scoreboard["method"].tolist()
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 3, 4.5))
    for i, (col, color) in enumerate([("precision", "#2F80ED"),
                                      ("recall", "#27AE60"),
                                      ("F1", "#EB5757")]):
        ax.bar(x + (i - 1) * w, scoreboard[col].fillna(0).to_numpy(), w,
               label=col, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"Detector scoreboard (IoU-matched)")
    ax.legend()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
