# Frozen engines

Plain copies of other projects that this repo builds and runs. They change only when we
change them. Each folder has a FROZEN.md with its source commit and our patch list.

| Folder | What | Source |
|---|---|---|
| llama-swap/ | the one-address model switcher (Go + web UI) | mostlygeek/llama-swap v257 |
| ninfer/ | QUASAR-capable inference runtime (mobile fork); runs quasar-27b, fable-27b, twin-27b | MirkoCovizzi/ninfer-rtx5090-mobile @ d4bc75dbc7066109c3d9692ed564e5904a849ba0 |
| adapters/ | start/stop/ready scripts, one per engine (see adapters/CONTRACT.md) | ours |
| config/ | example switcher config; the live one is ~/llama-swap/config.yaml on the box | ours |

Whether NInfer stays a single frozen copy (`ninfer/`) or gains a second, `ninfer-upstream/`
(Neroued/ninfer @ f76e19c0), is pending the live Fable/Twin speed check on the box (see
`engines/ninfer/FROZEN.md`). If the mobile build falls short of 95% of upstream's tok/s on
either model, `ninfer-upstream/` is added and this table gets a second row.

Build everything on the serving box with `scripts/engines/build.sh`.
