#!/usr/bin/env bash
# run_sorting_in_batches.sh
# Make executable:  chmod +x run_sorting_in_batches.sh
# Run:              ./run_sorting_in_batches.sh
# Run overnight:    nohup ./run_sorting_in_batches.sh >/dev/null 2>&1 &
#     (all output is already saved under $LOG_DIR, so nohup.out is not needed)
#
# What it does:
#   For every (date, round) in COMBOS it runs sort_spikes.py then
#   analyze_sorted_spikes.py, writing every output folder (analyzer_*_binary,
#   *_output, ...) onto the SSD at /data. Nothing is deleted afterwards.
#
#   Crucially, ONE bad session never stops the run: if a session's data folder
#   is missing, or either step throws ANY error, it is recorded and the script
#   moves on to the next session. A summary of everything that was skipped is
#   printed (and saved) at the very end so you can check it in the morning.
#
# NOTE: no `set -e` here on purpose -- we must NOT abort the whole overnight run
# when a single session fails. Errors are caught explicitly per session below.
set -uo pipefail

############### EDIT THESE ###############
# (Optional) conda/mamba env
# source ~/.bashrc

source /home/connorlab/Documents/GitHub/Julie/venv/bin/activate

PROJECT_ROOT="/home/connorlab/Documents/GitHub/Julie"

# Where the Intan sessions live and where all outputs are written (the new SSD).
# This must match sort_spikes.py's SORT_SPIKES_INTAN_BASE_PATH; it is passed to
# both scripts explicitly so this file is the single source of truth.
INTAN_BASE="/data/IntanData"
MONKEY="Cortana"

# Where run logs + the skipped-sessions summary go. Kept on /data (lots of room)
# rather than the low-space system disk.
LOG_BASE="${INTAN_BASE}/sorting_logs"
##########################################

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

PY1="$PROJECT_ROOT/src/spikesorting/sort_spikes/sort_spikes.py"
PY2="$PROJECT_ROOT/src/spikesorting/sort_spikes/analyze_sorted_spikes.py"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
run_stamp="$(date "+%Y%m%d_%H%M%S")"

# Per-run log directory (fall back to the project dir if /data isn't writable).
LOG_DIR="${LOG_BASE}/run_${run_stamp}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  LOG_DIR="${PROJECT_ROOT}/sorting_logs/run_${run_stamp}"
  mkdir -p "$LOG_DIR"
fi
MASTER_LOG="${LOG_DIR}/run.log"
SUMMARY="${LOG_DIR}/SUMMARY.txt"

# Echo a progress line to the console AND the master log.
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

# ── Preflight: fail fast (BEFORE the long run) on an obviously broken setup, so
#    you don't come back to 100+ identical failures caused by e.g. an inactive
#    venv. This does not run any session; it just sanity-checks the environment.
preflight_ok=1
for pyf in "$PY1" "$PY2"; do
  if [[ ! -f "$pyf" ]]; then
    log "PREFLIGHT ERROR: script not found: $pyf"
    preflight_ok=0
  fi
done
if ! python3 -c "import spikeinterface" >>"$MASTER_LOG" 2>&1; then
  log "PREFLIGHT ERROR: 'import spikeinterface' failed -- is the venv activated and are deps installed?"
  log "   (check the 'source .../venv/bin/activate' line near the top of this script)"
  preflight_ok=0
fi
if [[ ! -d "$INTAN_BASE" ]]; then
  log "PREFLIGHT ERROR: INTAN_BASE not found: $INTAN_BASE  (is the SSD mounted?)"
  preflight_ok=0
fi
if [[ $preflight_ok -ne 1 ]]; then
  log "Aborting before the batch run due to the preflight error(s) above."
  exit 1
fi

# (date, round) pairs
COMBOS=(
"2023-09-26 1"
"2023-09-26 2"
"2023-09-26 3"
"2023-09-27 1"
"2023-09-27 2"
"2023-09-27 3"
"2023-09-28 1"
"2023-09-29 1"
"2023-09-29 2"
"2023-09-29 3"
"2023-09-29 4"
"2023-10-03 1"
"2023-10-03 3"
"2023-10-03 4"
"2023-10-04 1"
"2023-10-04 2"
"2023-10-04 3"
"2023-10-04 4"
"2023-10-05 1"
"2023-10-05 2"
"2023-10-06 1"
"2023-10-06 2"
"2023-10-09 3"
"2023-10-10 2"
"2023-10-10 3"
"2023-10-11 1"
"2023-10-11 2"
"2023-10-11 3"
"2023-10-24 1"
"2023-10-24 2"
"2023-10-27 1"
"2023-10-27 2"
"2023-10-27 3"
"2023-10-27 4"
"2023-10-30 1"
"2023-10-30 2"
"2023-10-31 1"
"2023-10-31 2"
"2023-10-31 3"
"2023-11-08 1"
"2023-11-08 2"
"2023-11-08 3"
"2023-11-08 4"
"2023-11-11 1"
"2023-11-11 2"
"2023-11-12 1"
"2023-11-12 2"
"2023-11-17 1"
"2023-11-17 2"
"2023-11-20 1"
"2023-11-20 2"
"2023-11-22 1"
"2023-11-22 2"
"2023-11-22 3"
"2023-11-22 4"
"2023-11-25 1"
"2023-11-25 2"
"2023-11-25 3"
"2023-11-27 1"
"2023-11-27 2"
"2023-11-27 3"
"2023-11-28 1"
"2023-11-28 2"
"2023-11-28 3"
"2023-11-28 4"
"2023-12-05 1"
"2023-12-05 2"
"2023-12-05 3"
"2023-12-05 4"
"2023-12-07 1"
"2023-12-07 2"
"2023-12-07 3"
"2023-12-07 4"
"2023-12-11 1"
"2023-12-11 2"
"2023-12-11 3"
"2023-12-11 4"
"2023-12-14 1"
"2023-12-14 2"
"2023-12-14 3"
"2023-12-14 4"
"2023-12-18 1"
"2023-12-18 2"
"2023-12-18 3"
)

# Result trackers
OK_COUNT=0
MISSING=()   # session folder not found on disk -> nothing was run
FAILED=()    # data present but a step errored -> see per-session log

total="${#COMBOS[@]}"
start_human="$(ts)"
log "Starting batch run: ${total} sessions. Logs -> ${LOG_DIR}"

session_idx=0
for combo in "${COMBOS[@]}"; do
  # Skip accidental blank entries.
  [[ -z "${combo// /}" ]] && continue
  read -r date round <<<"$combo"
  session_idx=$((session_idx + 1))

  date_short="$(date -d "$date" +%y%m%d)"
  base_dir="${INTAN_BASE}/${MONKEY}/${date}/${date_short}_round${round}"
  session_log="${LOG_DIR}/${date}_round${round}.log"

  log "=== [${session_idx}/${total}] $date round $round ==="

  # 0. Does the raw session folder even exist? If not, record & skip.
  if [[ ! -d "$base_dir" ]]; then
    log "    MISSING data folder: $base_dir  (skipping)"
    MISSING+=("${date} round ${round}  ->  ${base_dir}")
    continue
  fi

  # 1. Sorting (writes *_output/ and analyzer_*_binary/ into the session folder).
  echo "===== sort_spikes: $date round $round @ $(ts) =====" >>"$session_log"
  python3 "$PY1" --date "$date" --round "$round" \
      --monkey "$MONKEY" --intan-base-path "$INTAN_BASE" >>"$session_log" 2>&1
  rc_sort=$?
  if [[ $rc_sort -ne 0 ]]; then
    log "    FAILED at sort_spikes (exit ${rc_sort}). See ${session_log}"
    FAILED+=("${date} round ${round}  ->  [sort_spikes exit ${rc_sort}]  log: ${session_log}")
    continue
  fi

  # 2. Analysis (consensus + sorted-spike cache; reads the analyzers from step 1).
  echo "===== analyze_sorted_spikes: $date round $round @ $(ts) =====" >>"$session_log"
  python3 "$PY2" --date "$date" --round "$round" \
      --monkey "$MONKEY" --intan-base-path "$INTAN_BASE" >>"$session_log" 2>&1
  rc_analyze=$?
  if [[ $rc_analyze -ne 0 ]]; then
    log "    FAILED at analyze_sorted_spikes (exit ${rc_analyze}). See ${session_log}"
    FAILED+=("${date} round ${round}  ->  [analyze_sorted_spikes exit ${rc_analyze}]  log: ${session_log}")
    continue
  fi

  OK_COUNT=$((OK_COUNT + 1))
  log "    OK -- finished $date round $round"
done

# ─────────────────────────── Final summary ──────────────────────────────────
{
  echo "========================================================================"
  echo " RUN SUMMARY"
  echo "   started : ${start_human}"
  echo "   finished: $(ts)"
  echo "========================================================================"
  echo "   Total sessions : ${total}"
  echo "   Completed OK   : ${OK_COUNT}"
  echo "   Failed         : ${#FAILED[@]}"
  echo "   Missing data   : ${#MISSING[@]}"
  echo ""

  if ((${#MISSING[@]})); then
    echo "--- MISSING (no data folder on ${INTAN_BASE}; nothing was run) ---"
    for m in "${MISSING[@]}"; do echo "   ${m}"; done
    echo ""
  fi

  if ((${#FAILED[@]})); then
    echo "--- FAILED (data present, but a step errored; open the log to see why) ---"
    for f in "${FAILED[@]}"; do echo "   ${f}"; done
    echo ""
  fi

  if ((${#FAILED[@]} == 0 && ${#MISSING[@]} == 0)); then
    echo "All ${total} sessions completed successfully."
  else
    echo "To retry a single session by hand, e.g.:"
    echo "   python3 \"${PY1}\" --date 2023-09-26 --round 1 --monkey ${MONKEY} --intan-base-path ${INTAN_BASE}"
  fi

  echo ""
  echo "Per-session logs : ${LOG_DIR}"
  echo "This summary     : ${SUMMARY}"
  echo "========================================================================"
} | tee "$SUMMARY"

# Exit 0 even if some sessions were skipped -- the run itself completed. The
# summary above (and $SUMMARY) is the record of what to check.
exit 0
