#!/usr/bin/env bash
# Engine adapter for NInfer (see engines/adapters/CONTRACT.md):
#
#   ninfer.sh <runtime ninfer-serve> <artifact.ninfer> <model-id> [ninfer-serve flags...]
#
# Makes room on the card, then exec's the runtime so llama-swap's SIGTERM reaches it
# directly. FreeToken is stopped through its settings helper (which disarms its crash
# watchdog); a stray NInfer started outside llama-swap is stopped by exact process name.
set -uo pipefail
RUNTIME="${1:?usage: ninfer.sh <runtime> <artifact> <model-id> [flags...]}"
ARTIFACT="${2:?artifact}"
MODEL_ID="${3:?model-id}"
shift 3
HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"
SERVER="${FREETOKEN_SERVER:-http://127.0.0.1:2020}"
PROC="${NINFER_PROCESS_NAME:-ninfer-serve}"

log() { printf '[ninfer.sh] %s\n' "$*" >&2; }

# "state has_job watchdog_live"; "helper-down - False" when the helper does not answer.
snapshot() {
  curl -s --max-time 5 "$HELPER/api/status" | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: print('helper-down - False'); sys.exit(0)
w=d.get('autoRestart') or {}
print(d['server']['state'], bool(d.get('currentJob')), bool(w.get('armed') and w.get('enabled', True) and not w.get('gave_up')))"
}

read -r state has_job live <<<"$(snapshot)"
if [ "$state" = helper-down ]; then
  if curl -s --max-time 3 -o /dev/null "$SERVER/health"; then
    log "FreeToken answers on $SERVER but the helper is down; refusing"
    exit 1
  fi
elif [ "$state" != unreachable ] || [ "$has_job" = True ] || [ "$live" = True ]; then
  log "stopping FreeToken (state: $state) to free the card"
  curl -s --max-time 10 -X POST "$HELPER/api/server/stop" >/dev/null
  for _ in $(seq 1 "${NINFER_STOP_POLLS:-90}"); do
    read -r state has_job _ <<<"$(snapshot)"
    [ "$state" = unreachable ] && [ "$has_job" = False ] && break
    sleep 2
  done
  if [ "$state" != unreachable ] || [ "$has_job" != False ]; then
    log "FreeToken did not stop (state: $state); refusing to start"
    exit 1
  fi
fi

if pgrep -x "$PROC" >/dev/null; then
  log "stopping a $PROC started outside llama-swap"
  pkill -TERM -x "$PROC"
  for _ in $(seq 1 30); do pgrep -x "$PROC" >/dev/null || break; sleep 1; done
  pkill -KILL -x "$PROC" 2>/dev/null
  sleep 2
  pgrep -x "$PROC" >/dev/null && { log "$PROC would not stop; refusing"; exit 1; }
fi

exec "$RUNTIME" "$ARTIFACT" --model-id "$MODEL_ID" "$@"
