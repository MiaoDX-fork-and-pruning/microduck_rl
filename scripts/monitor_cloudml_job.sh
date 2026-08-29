#!/usr/bin/env bash
set -u -o pipefail

JOB_ID=${1:?usage: $0 JOB_ID [interval_seconds] [output_prefix]}
INTERVAL_SECONDS=${2:-1800}
OUTPUT_PREFIX=${3:-/dongxu/microduck_rl/runs/velocity-full-20260829-0620}
POD_NAME=${POD_NAME:-}
EXECUTOR=${EXECUTOR_BIN:-/home/mi/executor/exe}
LOG_FILE=${MONITOR_LOG:-"cloudml/monitor-${JOB_ID}.log"}

mkdir -p "$(dirname "$LOG_FILE")"
log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

terminal_state() {
  case "$1" in
    succeed|failed|stopped|exception|stopping) return 0 ;;
    *) return 1 ;;
  esac
}

while :; do
  describe=$($EXECUTOR compute cloudml cml -- custom_train describe "$JOB_ID" 2>&1)
  state=$(printf '%s\n' "$describe" | awk -F'"' '/"state"/{print $4; exit}')
  runtime=$(printf '%s\n' "$describe" | awk -F'"' '/"runTime"/{print $4; exit}')
  pod=${POD_NAME:-$(printf '%s\n' "$describe" | awk -F'"' '/"podName"/{print $4; exit}')}
  log "job=$JOB_ID state=${state:-unknown} runtime=${runtime:-unknown} pod=${pod:-unknown}"

  if [ -n "$pod" ]; then
    logs=$($EXECUTOR compute cloudml cml -- custom_train logs "$JOB_ID" --pod_name "$pod" --lines 120 2>&1 || true)
    summary=$(printf '%s\n' "$logs" | rg 'Learning iteration|Mean reward|Steps per second|nan_state|ETA' | tail -6 || true)
    [ -n "$summary" ] && while IFS= read -r line; do log "train $line"; done <<< "$summary"
  fi

  probe=$($EXECUTOR storage juicefs probe \
    --url "https://cloud.mioffice.cn/juicefs/vol-detail?cluster=wlcb-cloudml&name=robot-intelligent-planning-data&path=${OUTPUT_PREFIX#/}" \
    --markers model_250.pt --max-depth 6 --json 2>&1 || true)
  hit=$(printf '%s\n' "$probe" | rg -o '"hit_count": [0-9]+' | head -1 || true)
  log "juicefs model_250.pt ${hit:-probe_failed}"

  terminal_state "$state" && { log "terminal state reached: $state"; break; }
  sleep "$INTERVAL_SECONDS"
done
