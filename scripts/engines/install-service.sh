#!/usr/bin/env bash
# Point the llama-swap systemd --user unit at the frozen build. Keeps the old unit as a backup.
set -euo pipefail
BIN="${FREETOKEN_ENGINES_BIN:-$HOME/.local/share/freetoken-engines/bin}/llama-swap"
CONF="$HOME/llama-swap/config.yaml"
UNIT="$HOME/.config/systemd/user/llama-swap.service"
[ -x "$BIN" ] || { echo "build first: scripts/engines/build.sh" >&2; exit 1; }
[ -f "$CONF" ] || { echo "missing $CONF (copy engines/config/config.example.yaml and fill in paths)" >&2; exit 1; }
mkdir -p "$(dirname "$UNIT")"
[ -f "$UNIT" ] && cp "$UNIT" "$UNIT.bak-$(date +%Y%m%d-%H%M%S)"
cat > "$UNIT" <<EOF
[Unit]
Description=llama-swap (FreeToken frozen build): one address for all local models (port 2040)
After=freetoken-settings.service

[Service]
WorkingDirectory=%h/llama-swap
ExecStart=$BIN --config $CONF --listen 127.0.0.1:2040 --watch-config
Restart=on-failure
KillMode=mixed
TimeoutStopSec=200

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable llama-swap >/dev/null
echo "installed; apply with: systemctl --user restart llama-swap"
