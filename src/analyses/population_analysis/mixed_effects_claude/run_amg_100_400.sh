#!/bin/bash
# run_overnight.sh
#
# Run PCs sweep (full_profile + summary, 2–5 PCs) for AMG only.
# Window fixed at 100–400 ms (Munuera 2018 first epoch).
#
# Usage:
#   chmod +x run_overnight.sh
#   nohup ./run_overnight.sh > overnight_master.log 2>&1 &
#
#   Or with screen/tmux:
#   screen -S sweep
#   ./run_overnight.sh
#   # Ctrl+A, D to detach

# DO NOT use set -e — we want to continue even if one combo fails
# Use headless matplotlib backend (must be set before Python imports pyplot)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_analysis

echo "Using python: $(which python)"
export MPLBACKEND=Agg
export PYTHONPATH="/home/connorlab/Documents/GitHub/Julie/src:$PYTHONPATH"

SCRIPT="/home/connorlab/Documents/GitHub/Julie/src/analyses/population_analysis/mixed_effects_claude/run_social_encoding_sweep.py"
PERMS=5000
MIN_PCS=2
MAX_PCS=5
WINDOW="100-400"
LOG_DIR="overnight_logs"
FAILED=0

mkdir -p "$LOG_DIR"

# ── Grid: AMG only, single window ──
REGIONS=("AMG")

TOTAL=${#REGIONS[@]}
COUNT=0
START_TIME=$(date +%s)

echo "========================================================================"
echo "  Social Encoding Overnight Run"
echo "  Started: $(date)"
echo "  Window: ${WINDOW} ms"
echo "  Regions: ${REGIONS[*]}"
echo "  Feature modes: full_profile, summary"
echo "  PCs: $MIN_PCS–$MAX_PCS"
echo "  Permutations: $PERMS"
echo "========================================================================"
echo ""

for REGION in "${REGIONS[@]}"; do
    COUNT=$((COUNT + 1))
    LABEL="${REGION}_${WINDOW}"
    LOG_FILE="${LOG_DIR}/${LABEL}.log"

    echo "────────────────────────────────────────────────────────────"
    echo "  [$COUNT/$TOTAL] region=$REGION  window=$WINDOW"
    echo "  Log: $LOG_FILE"
    echo "  Started: $(date)"
    echo "────────────────────────────────────────────────────────────"

    python3 "$SCRIPT" pcs \
        --region "$REGION" \
        --window "$WINDOW" \
        --perms "$PERMS" \
        --min-pcs "$MIN_PCS" \
        --max-pcs "$MAX_PCS" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "  *** [$COUNT/$TOTAL] FAILED (exit code $EXIT_CODE) — $(date) ***"
        echo ""
        FAILED=$((FAILED + 1))
    else
        echo ""
        echo "  [$COUNT/$TOTAL] DONE — $(date)"
        echo ""
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "========================================================================"
echo "  ALL DONE"
echo "  Finished: $(date)"
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Completed: $((TOTAL - FAILED))/$TOTAL    Failed: $FAILED/$TOTAL"
echo ""
echo "  Results saved to:"
echo "    social_encoding_pcs_sweep/{region}/${WINDOW}ms/"
echo ""
echo "  Logs saved to:"
echo "    $LOG_DIR/"
echo "========================================================================"
echo ""

# ── Print combined summary from all CSV files ──
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    COMBINED RESULTS SUMMARY                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

for f in social_encoding_pcs_sweep/*/*/pcs_sweep_results.csv; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        column -t -s',' "$f"
        echo ""
    fi
done
