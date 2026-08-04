#!/usr/bin/env bash
# run_analyze_in_batches.sh
# Make executable:  chmod +x run_analyze_in_batches.sh
# Run:              ./run_analyze_in_batches.sh
# Run detached:     nohup ./run_analyze_in_batches.sh >/dev/null 2>&1 &
#
# ANALYZE ONLY -- this never runs sort_spikes.py.
#
# Use this when sorting already completed (the sorter/analyzer folders are on
# disk) but analyze_sorted_spikes.py failed, e.g. the whole overnight batch died
# at the analyze step. Re-running the full sorting script would DELETE and redo
# every sorter output (run_sorters uses remove_existing_folder=True) and cost
# many hours, so this script skips straight to the analysis.
#
# Like the sorting batch script: one failing session never stops the run, and a
# summary of MISSING / NOT_SORTED / FAILED / INCOMPLETE sessions is printed and
# saved at the end.
set -uo pipefail

############### EDIT THESE ###############
source /home/connorlab/Documents/GitHub/Julie/venv/bin/activate

PROJECT_ROOT="/home/connorlab/Documents/GitHub/Julie"
INTAN_BASE="/data/IntanData"
MONKEY="Cortana"
LOG_BASE="${INTAN_BASE}/sorting_logs"
# Data lives outside the repo so git checkouts cannot delete it.
DATA_ROOT="${JULIE_DATA_PATH:-/home/connorlab/Documents/JulieData}/${MONKEY}"
##########################################

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

PY2="$PROJECT_ROOT/src/spikesorting/sort_spikes/analyze_sorted_spikes.py"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
run_stamp="$(date "+%Y%m%d_%H%M%S")"

LOG_DIR="${LOG_BASE}/analyze_${run_stamp}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  LOG_DIR="${PROJECT_ROOT}/sorting_logs/analyze_${run_stamp}"
  mkdir -p "$LOG_DIR"
fi
MASTER_LOG="${LOG_DIR}/run.log"
SUMMARY="${LOG_DIR}/SUMMARY.txt"
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

# ── Preflight: catch an obviously broken setup before looping over 85 sessions.
preflight_ok=1
if [[ ! -f "$PY2" ]]; then
  log "PREFLIGHT ERROR: script not found: $PY2"; preflight_ok=0
fi
if ! python3 -c "import spikeinterface" >>"$MASTER_LOG" 2>&1; then
  log "PREFLIGHT ERROR: 'import spikeinterface' failed -- is the venv activated?"; preflight_ok=0
fi
if [[ ! -d "$INTAN_BASE" ]]; then
  log "PREFLIGHT ERROR: INTAN_BASE not found: $INTAN_BASE (is the SSD mounted?)"; preflight_ok=0
fi
# analyze_sorted_spikes needs the compiled trial pickles for its metadata merge.
if [[ ! -d "${DATA_ROOT}/compiled" ]]; then
  log "PREFLIGHT WARNING: ${DATA_ROOT}/compiled is missing."
  log "   analyze will fail with FileNotFoundError on every session. Restore it with:"
  log "   git restore --source=2975202^ -- ${MONKEY}/compiled/"
fi
if [[ $preflight_ok -ne 1 ]]; then
  log "Aborting before the batch run due to the preflight error(s) above."
  exit 1
fi

# (date, round) pairs -- same list as run_sorting_in_batches.sh
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

# Sorter outputs that must already exist for analysis to be possible.
REQUIRED_SORTED_DIRS=( kilosort4_output tridesclous_output mountainsort5_output
                       analyzer_KS4_binary analyzer_TDC_binary analyzer_MS5_binary )

# Result trackers
OK_COUNT=0
NO_UNITS=()     # ran fine, but no consensus units (valid outcome, no pickle written)
MISSING=()      # session data folder absent
NOT_SORTED=()   # session present but sorter/analyzer outputs absent -> sort it first
FAILED=()       # analyze errored
INCOMPLETE=()   # analyze exited 0 but produced no cache pickle and no summary

total="${#COMBOS[@]}"
start_human="$(ts)"
log "Starting ANALYZE-ONLY batch: ${total} sessions. Logs -> ${LOG_DIR}"

session_idx=0
for combo in "${COMBOS[@]}"; do
  [[ -z "${combo// /}" ]] && continue
  read -r date round <<<"$combo"
  session_idx=$((session_idx + 1))

  date_short="$(date -d "$date" +%y%m%d)"
  base_dir="${INTAN_BASE}/${MONKEY}/${date}/${date_short}_round${round}"
  session_log="${LOG_DIR}/${date}_round${round}.log"

  # Outputs analyze_sorted_spikes writes on success:
  cache_pkl="${DATA_ROOT}/sorted_spike_cache/${date}_round_${round}.pkl"
  summary_txt="${DATA_ROOT}/sorted_spike_summary/${date_short}_round${round}_sorting_summary.txt"

  log "=== [${session_idx}/${total}] $date round $round ==="

  if [[ ! -d "$base_dir" ]]; then
    log "    MISSING data folder: $base_dir (skipping)"
    MISSING+=("${date} round ${round}  ->  ${base_dir}")
    continue
  fi

  # Analysis needs the sorter output; don't burn a Python start-up to find out.
  missing_sorted=""
  for d in "${REQUIRED_SORTED_DIRS[@]}"; do
    [[ -d "${base_dir}/${d}" ]] || missing_sorted+="${d} "
  done
  if [[ -n "$missing_sorted" ]]; then
    log "    NOT SORTED -- missing: ${missing_sorted}(run the sorting script for this session)"
    NOT_SORTED+=("${date} round ${round}  ->  missing: ${missing_sorted}")
    continue
  fi

  echo "===== analyze_sorted_spikes: $date round $round @ $(ts) =====" >>"$session_log"
  python3 "$PY2" --date "$date" --round "$round" \
      --monkey "$MONKEY" --intan-base-path "$INTAN_BASE" >>"$session_log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    log "    FAILED (exit ${rc}). See ${session_log}"
    FAILED+=("${date} round ${round}  ->  [exit ${rc}]  log: ${session_log}")
    continue
  fi

  # Verify it actually saved something. A session with no consensus units writes
  # only the summary and exits 0 -- that is a valid outcome, not a failure.
  if [[ -s "$cache_pkl" ]]; then
    OK_COUNT=$((OK_COUNT + 1))
    log "    OK -- cache written: ${cache_pkl}"
  elif [[ -s "$summary_txt" ]] && grep -qi "no units in agreement" "$summary_txt"; then
    NO_UNITS+=("${date} round ${round}")
    log "    OK -- no units in agreement (nothing to cache)"
  else
    log "    INCOMPLETE -- exit 0 but no cache pickle and no summary found"
    INCOMPLETE+=("${date} round ${round}  ->  expected: ${cache_pkl}  (log: ${session_log})")
  fi
done

# ─────────────────────────── Final summary ──────────────────────────────────
{
  echo "========================================================================"
  echo " ANALYZE-ONLY RUN SUMMARY"
  echo "   started : ${start_human}"
  echo "   finished: $(ts)"
  echo "========================================================================"
  echo "   Total sessions     : ${total}"
  echo "   Cache written OK   : ${OK_COUNT}"
  echo "   No units (valid)   : ${#NO_UNITS[@]}"
  echo "   Failed (errored)   : ${#FAILED[@]}"
  echo "   Incomplete output  : ${#INCOMPLETE[@]}"
  echo "   Not sorted yet     : ${#NOT_SORTED[@]}"
  echo "   Missing data       : ${#MISSING[@]}"
  echo ""

  if ((${#FAILED[@]})); then
    echo "--- FAILED (open the log to see why) ---"
    for f in "${FAILED[@]}"; do echo "   ${f}"; done
    echo ""
  fi
  if ((${#INCOMPLETE[@]})); then
    echo "--- INCOMPLETE (no error, but nothing saved) ---"
    for i in "${INCOMPLETE[@]}"; do echo "   ${i}"; done
    echo ""
  fi
  if ((${#NOT_SORTED[@]})); then
    echo "--- NOT SORTED (run the sorting script for these first) ---"
    for n in "${NOT_SORTED[@]}"; do echo "   ${n}"; done
    echo ""
  fi
  if ((${#MISSING[@]})); then
    echo "--- MISSING (no data folder on ${INTAN_BASE}) ---"
    for m in "${MISSING[@]}"; do echo "   ${m}"; done
    echo ""
  fi
  if ((${#NO_UNITS[@]})); then
    echo "--- NO UNITS IN AGREEMENT (valid result, nothing cached) ---"
    for u in "${NO_UNITS[@]}"; do echo "   ${u}"; done
    echo ""
  fi

  if ((${#FAILED[@]} == 0 && ${#INCOMPLETE[@]} == 0 && ${#NOT_SORTED[@]} == 0 && ${#MISSING[@]} == 0)); then
    echo "All ${total} sessions analyzed successfully."
  else
    echo "To retry one session by hand:"
    echo "   python3 \"${PY2}\" --date 2023-09-26 --round 1 --monkey ${MONKEY} --intan-base-path ${INTAN_BASE}"
  fi

  echo ""
  echo "Per-session logs : ${LOG_DIR}"
  echo "This summary     : ${SUMMARY}"
  echo "========================================================================"
} | tee "$SUMMARY"

exit 0
