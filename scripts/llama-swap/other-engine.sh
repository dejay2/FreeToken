#!/usr/bin/env bash
# llama-swap `cmd` wrapper for a non-FreeToken engine on the same card:
#
#   other-engine.sh <server command...>
#
# A FreeToken model started from the settings page (outside llama-swap) would still hold the
# card, and the other engine would fail to allocate. Stop it through the helper first (a page
# Stop also disarms its crash watchdog), then exec the command so llama-swap's SIGTERM
# reaches the engine itself. If FreeToken will not go, refuse rather than start on top of it.
set -uo pipefail
HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"

snapshot() {  # "state has_job"; "helper-down -" when the helper does not answer
  curl -s --max-time 5 "$HELPER/api/status" | python3 -c "import json,sys
try: d=json.load(sys.stdin); print(d['server']['state'], bool(d.get('currentJob')))
except Exception: print('helper-down -')"
}

read -r state has_job <<<"$(snapshot)"
if [ "$state" != unreachable ] || [ "$has_job" = True ]; then
  if [ "$state" = helper-down ]; then
    # No helper, so no watchdog either; only a server it left behind could hold the port.
    curl -s --max-time 3 -o /dev/null http://127.0.0.1:2020/health \
      && { printf '[other-engine.sh] FreeToken answers on :2020 but the helper is down; refusing\n' >&2; exit 1; }
  else
    printf '[other-engine.sh] stopping FreeToken (state: %s) to free the card\n' "$state" >&2
    curl -s --max-time 10 -X POST "$HELPER/api/server/stop" >/dev/null
    for _ in $(seq 1 90); do
      read -r state has_job <<<"$(snapshot)"
      [ "$state" = unreachable ] && [ "$has_job" = False ] && break
      sleep 2
    done
    if [ "$state" != unreachable ] || [ "$has_job" != False ]; then
      printf '[other-engine.sh] FreeToken did not stop (state: %s); refusing to start\n' "$state" >&2
      exit 1
    fi
  fi
fi
exec "$@"
