# Frozen engines

Plain copies of other projects that this repo builds and runs. They change only when we
change them. Each folder has a FROZEN.md with its source commit and our patch list.

| Folder | What | Source |
|---|---|---|
| llama-swap/ | the one-address model switcher (Go + web UI) | mostlygeek/llama-swap v257 |
| adapters/ | start/stop/ready scripts, one per engine (see adapters/CONTRACT.md) | ours |
| config/ | example switcher config; the live one is ~/llama-swap/config.yaml on the box | ours |

Build everything on the serving box with `scripts/engines/build.sh`.
