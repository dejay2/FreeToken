#!/usr/bin/env bash
# Build the frozen engines on the serving box (WSL). Safe to re-run.
#   scripts/engines/build.sh            # everything
#   scripts/engines/build.sh llama-swap # one part: llama-swap | ninfer
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${FREETOKEN_ENGINES_BIN:-$HOME/.local/share/freetoken-engines/bin}"
GO_VERSION=1.27.1
NODE_VERSION=24.9.0
mkdir -p "$OUT" "$HOME/.local"

need_go() {
  if ! "$HOME/.local/go/bin/go" version 2>/dev/null | grep -q "go$GO_VERSION "; then
    echo "== installing Go $GO_VERSION"
    local tmp; tmp=$(mktemp -d)
    curl -sfL "https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz" | tar xz -C "$tmp"
    rm -rf "$HOME/.local/go" && mv "$tmp/go" "$HOME/.local/go" && rm -rf "$tmp"
  fi
  export PATH="$HOME/.local/go/bin:$PATH"
}

need_node() {
  local dir="$HOME/.local/node-v$NODE_VERSION-linux-x64"
  if [ ! -x "$dir/bin/node" ]; then
    echo "== installing Node $NODE_VERSION"
    curl -sfL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" | tar xJ -C "$HOME/.local"
  fi
  export PATH="$dir/bin:$PATH"
}

build_llama_swap() {
  need_go; need_node
  cd "$REPO/engines/llama-swap"
  echo "== llama-swap UI"
  (cd ui && npm ci --no-audit --no-fund && npm run build)
  echo "== llama-swap tests"
  go test -short -count=1 ./internal/...
  echo "== llama-swap binary"
  go build -tags embed_ui -ldflags "-X main.version=frozen-v257-freetoken -X main.commit=$(git -C "$REPO" rev-parse --short HEAD)" -o "$OUT/llama-swap.new" .
  mv "$OUT/llama-swap.new" "$OUT/llama-swap"
  "$OUT/llama-swap" --version
}

build_ninfer() {
  export PATH=/usr/local/cuda/bin:$PATH
  for d in "$REPO"/engines/ninfer*; do
    [ -f "$d/CMakeLists.txt" ] || continue
    echo "== $(basename "$d")"
    cmake -S "$d" -B "$d/build" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$d/build" -j "$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))"
  done
}

case "${1:-all}" in
  llama-swap) build_llama_swap ;;
  ninfer) build_ninfer ;;
  all) build_llama_swap; build_ninfer ;;
  *) echo "usage: $0 [all|llama-swap|ninfer]" >&2; exit 2 ;;
esac
