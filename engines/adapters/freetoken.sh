#!/usr/bin/env bash
# llama-swap `cmd` for a FreeToken model: boots it through the settings helper instead of a
# bare `ft serve`, so the page's fit check, memory governor, KV parking and crash watchdog
# stay in charge. A bare `ft serve` under llama-swap fights the helper: the watchdog adopts
# any server serving on the port and restarts it after llama-swap kills it.
#
#   freetoken.sh <model folder>    boot that model, then stay alive while it serves
#
# llama-swap's SIGTERM (unload or swap) becomes a page Stop, which also disarms the watchdog.
# Point llama-swap's checkEndpoint at /ready?model=<folder name>: /health is 200 while the
# weights still load, and without ?model= the model being swapped out would pass the check.
set -uo pipefail

HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"
SERVER="${FREETOKEN_SERVER:-http://127.0.0.1:2020}"
MODEL_PATH="${1:?usage: freetoken.sh <model folder>}"
MODEL_PATH="${MODEL_PATH%/}"
MODEL_NAME="$(basename "$MODEL_PATH")"  # what /ready?model= matches: the folder name

log() { printf '[freetoken.sh] %s\n' "$*" >&2; }

# One /api/status call -> "state job watchdog_live": a consistent snapshot. The watchdog only
# brings a crashed server back when it is armed AND enabled AND has not given up (it stays
# "armed" with auto-restart off, and after its hourly budget runs out).
snapshot() {
  curl -s --max-time 5 "$HELPER/api/status" | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('helper-down - False'); sys.exit(0)
w = d.get('autoRestart') or {}
job = (d.get('currentJob') or {}).get('jobId') or '-'
live = bool(w.get('armed') and w.get('enabled', True) and not w.get('gave_up'))
print(d['server']['state'], job, live)"
}

# 0 when the server's loaded model is not this one. Asks the server itself, not the saved
# settings: a settings edit alone must not count as a model switch. A server that predates
# /ready (404) cannot prove it holds this model, so it counts as another.
other_model_loaded() {
  local out
  out=$(curl -s --max-time 5 -w '\n%{http_code}' --get --data-urlencode "model=$MODEL_NAME" "$SERVER/ready")
  case "$out" in *'another model is loaded'*|*$'\n'404) return 0 ;; *) return 1 ;; esac
}

# wait_job <job id> <stage wanted>: 0 when reached, 1 on any other end. No time cap: a slow
# boot is ended by llama-swap's healthCheckTimeout, whose SIGTERM reaches on_term.
wait_job() {
  local job="$1" want="$2" stage missing=0
  while true; do
    stage=$(curl -s --max-time 5 -w '\n%{http_code}' "$HELPER/api/server/jobs/$job" | python3 -c "import json,sys
body, _, code = sys.stdin.read().rpartition('\n')
if code.strip() == '404': print('gone'); sys.exit(0)
try: print(json.loads(body).get('stage') or '')
except Exception: print('')")
    [ "$stage" = "$want" ] && return 0
    case "$stage" in
      failed|stopped) return 1 ;;
      gone)  # the helper restarted and forgot its jobs: judge by what is running instead
        missing=$((missing + 1))
        if [ "$missing" -ge 2 ]; then
          read -r state job_now _ <<<"$(snapshot)"
          [ "$want" = stopped ] && [ "$state" = unreachable ] && [ "$job_now" = - ] && return 0
          return 1
        fi ;;
    esac
    sleep 2
  done
}

stop_server() {
  local job
  job=$(curl -s --max-time 10 -X POST "$HELPER/api/server/stop" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('jobId') or '')
except Exception: print('')")
  [ -n "$job" ] && wait_job "$job" stopped
}

on_term() {
  log "unload requested: stopping FreeToken through the helper"
  if stop_server; then exit 0; fi
  log "Stop did not confirm; the card may still be held (see the settings page)"
  exit 1
}
trap on_term TERM INT

# Another engine may hold the card if it was started outside llama-swap.
if pgrep -x ninfer-serve >/dev/null; then
  log "stopping a NInfer server that holds the card"
  pkill -TERM -x ninfer-serve
  for _ in $(seq 1 30); do pgrep -x ninfer-serve >/dev/null || break; sleep 1; done
  pkill -KILL -x ninfer-serve 2>/dev/null
  sleep 2
  pgrep -x ninfer-serve >/dev/null && { log "NInfer would not stop"; exit 1; }
fi

read -r state job_now _ <<<"$(snapshot)"
[ "$state" = helper-down ] && { log "settings helper not reachable at $HELPER"; exit 1; }
if [ "$state" = serving ] && [ "$job_now" = - ] && ! other_model_loaded; then
  log "already serving $MODEL_PATH; adopting it"
else
  # Also stop when only a job is in flight: a page or watchdog boot that has not bound its
  # port yet reads "unreachable", and a new ModelPath saved under it could boot the wrong model.
  if [ "$state" != unreachable ] || [ "$job_now" != - ]; then
    log "stopping the running FreeToken model first (state: $state, job: $job_now)"
    stop_server || { log "could not stop the running server"; exit 1; }
  fi
  body=$(python3 -c 'import json,sys; print(json.dumps({"settings": {"ModelPath": sys.argv[1]}}))' "$MODEL_PATH")
  saved=$(curl -s --max-time 30 -X PUT -H 'content-type: application/json' -d "$body" "$HELPER/api/settings")
  case "$saved" in *'"status":"saved"'*) ;; *) log "settings refused: $saved"; exit 1 ;; esac
  job=$(curl -s --max-time 30 -X POST "$HELPER/api/server/start" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('jobId') or '')
except Exception: print('')")
  [ -n "$job" ] || { log "helper did not accept Start"; exit 1; }
  log "booting $MODEL_PATH (job $job)"
  # The wait runs in the background so a SIGTERM during the boot reaches on_term at once.
  wait_job "$job" serving & wait $!
  if [ $? -ne 0 ]; then
    log "boot did not reach serving (job $job); stopping what is left of it"
    stop_server
    exit 1
  fi
fi

# Stay alive while the helper keeps THIS model up. A crash the watchdog is restarting counts
# as up. We end when the server is gone for good, or when the page switched to another model
# (llama-swap would otherwise keep sending this model's name to it).
gone=0
while true; do
  sleep 5 & wait $!
  read -r state job_now live <<<"$(snapshot)"
  if [ "$state" = unreachable ] && [ "$job_now" = - ] && [ "$live" != True ]; then
    gone=$((gone + 1))
    [ "$gone" -ge 3 ] && { log "FreeToken is no longer running"; exit 0; }
  else
    gone=0
  fi
  if [ "$state" = serving ] && other_model_loaded; then
    log "the settings page switched to another model; releasing $MODEL_PATH"
    exit 0
  fi
done
