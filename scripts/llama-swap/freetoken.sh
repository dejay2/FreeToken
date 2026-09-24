#!/usr/bin/env bash
# llama-swap `cmd` for a FreeToken model: boots it through the settings helper instead of a
# bare `ft serve`, so the page's fit check, memory governor, KV parking and crash watchdog
# stay in charge. A bare `ft serve` under llama-swap fights the helper: the watchdog adopts
# any server serving on the port and restarts it after llama-swap kills it.
#
#   freetoken.sh <model folder>    boot that model, then stay alive while it serves
#
# llama-swap's SIGTERM (unload or swap) becomes a page Stop, which also disarms the watchdog.
# Point llama-swap's checkEndpoint at /ready: /health is 200 while the weights still load.
set -uo pipefail

HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"
MODEL_PATH="${1:?usage: freetoken.sh <model folder>}"

log() { printf '[freetoken.sh] %s\n' "$*" >&2; }

json_get() {  # json_get <python expr on d>; reads JSON from stdin
  python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: print(''); sys.exit(0)
v=$1
print('' if v is None else v)"
}

server_state() { curl -s --max-time 5 "$HELPER/api/status" | json_get "d['server']['state']"; }
current_job() { curl -s --max-time 5 "$HELPER/api/status" | json_get "(d.get('currentJob') or {}).get('jobId')"; }
armed() { curl -s --max-time 5 "$HELPER/api/status" | json_get "d['autoRestart'].get('armed')"; }
saved_model() { curl -s --max-time 5 "$HELPER/api/settings" | json_get "d['settings'].get('ModelPath')"; }

wait_job() {  # wait_job <job id> <stage wanted>; 0 when reached, 1 on failed/stopped/timeout
  local job="$1" want="$2" stage i
  for i in $(seq 1 180); do
    stage=$(curl -s --max-time 5 "$HELPER/api/server/jobs/$job" | json_get "d.get('stage')")
    [ "$stage" = "$want" ] && return 0
    case "$stage" in failed|stopped) [ "$want" != "$stage" ] && return 1 ;; esac
    sleep 2
  done
  return 1
}

stop_server() {
  local job
  job=$(curl -s --max-time 10 -X POST "$HELPER/api/server/stop" | json_get "d.get('jobId')")
  [ -n "$job" ] && wait_job "$job" stopped
}

on_term() {
  log "unload requested: stopping FreeToken through the helper"
  stop_server || log "stop job did not confirm; check the settings page"
  exit 0
}
trap on_term TERM INT

# Another engine may hold the card if it was started outside llama-swap.
if pgrep -f ninfer-serve >/dev/null; then
  log "stopping a NInfer server that holds the card"
  pkill -TERM -f ninfer-serve
  for _ in $(seq 1 30); do pgrep -f ninfer-serve >/dev/null || break; sleep 1; done
fi

state=$(server_state)
if [ "$state" = "serving" ] && [ "$(saved_model)" = "$MODEL_PATH" ]; then
  log "already serving $MODEL_PATH; adopting it"
else
  if [ -n "$state" ] && [ "$state" != "unreachable" ]; then
    log "stopping the running FreeToken model first (state: $state)"
    stop_server || { log "could not stop the running server"; exit 1; }
  fi
  # The engine swapped out (NInfer reads ~21 GB of weights buffered) leaves its file cache in
  # the WSL VM, and the boot pins ~53 GiB on top. Clean cache only; needs passwordless sudo,
  # skipped otherwise.
  if sudo -n true 2>/dev/null; then
    sync && sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' && log "dropped the Linux file cache before boot"
  fi
  body=$(python3 -c 'import json,sys; print(json.dumps({"settings": {"ModelPath": sys.argv[1]}}))' "$MODEL_PATH")
  saved=$(curl -s --max-time 30 -X PUT -H 'content-type: application/json' -d "$body" "$HELPER/api/settings")
  [ "$(printf '%s' "$saved" | json_get "d.get('status')")" = "saved" ] || { log "settings refused: $saved"; exit 1; }
  job=$(curl -s --max-time 30 -X POST "$HELPER/api/server/start" | json_get "d.get('jobId')")
  [ -n "$job" ] || { log "helper did not accept Start"; exit 1; }
  log "booting $MODEL_PATH (job $job)"
  # The wait runs in the background so a SIGTERM during the boot reaches on_term at once.
  wait_job "$job" serving & wait $! || { log "boot did not reach serving (job $job)"; exit 1; }
fi

# Stay alive while the helper keeps a server up. A crash the watchdog is restarting counts
# as up; only a stopped server with nothing in flight and the watchdog disarmed ends us.
gone=0
while true; do
  sleep 5 & wait $!
  if [ "$(server_state)" = "unreachable" ] && [ -z "$(current_job)" ] && [ "$(armed)" != "True" ]; then
    gone=$((gone + 1))
    [ "$gone" -ge 3 ] && { log "FreeToken is no longer running"; exit 0; }
  else
    gone=0
  fi
done
