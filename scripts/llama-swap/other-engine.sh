#!/usr/bin/env bash
# llama-swap `cmd` wrapper for a non-FreeToken engine on the same card:
#
#   other-engine.sh <server command...>
#
# A FreeToken model started from the settings page (outside llama-swap) would still hold the
# card, and the other engine would fail to allocate. Stop it through the helper first (a page
# Stop also disarms its crash watchdog), then exec the command so llama-swap's SIGTERM
# reaches the engine itself.
set -uo pipefail
HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"

state=$(curl -s --max-time 5 "$HELPER/api/status" \
  | python3 -c "import json,sys
try: print(json.load(sys.stdin)['server']['state'])
except Exception: print('')")
if [ -n "$state" ] && [ "$state" != "unreachable" ]; then
  printf '[other-engine.sh] stopping FreeToken (state: %s) to free the card\n' "$state" >&2
  curl -s --max-time 10 -X POST "$HELPER/api/server/stop" >/dev/null
  for _ in $(seq 1 90); do
    s=$(curl -s --max-time 5 "$HELPER/api/status" | python3 -c "import json,sys
try: d=json.load(sys.stdin); print(d['server']['state'], bool(d.get('currentJob')))
except Exception: print('')")
    [ "$s" = "unreachable False" ] && break
    sleep 2
  done
fi
exec "$@"
