# Frozen engines

Plain copies of other projects that this repo builds and runs. They change only when we
change them. Each folder has a FROZEN.md with its source commit and our patch list.

| Folder | What | Source |
|---|---|---|
| llama-swap/ | the one-address model switcher (Go + web UI) | mostlygeek/llama-swap v257 |
| ninfer/ | QUASAR-capable inference runtime (mobile fork); runs quasar-27b only | MirkoCovizzi/ninfer-rtx5090-mobile @ d4bc75dbc7066109c3d9692ed564e5904a849ba0 |
| ninfer-upstream/ | upstream inference runtime; runs fable-27b, twin-27b (NInfer v3 artifacts) | Neroued/ninfer @ f76e19c0fbd026c86f46005acf2c80c54084bade |
| adapters/ | start/stop/ready scripts, one per engine (see adapters/CONTRACT.md) | ours |
| config/ | example switcher config; the live one is ~/llama-swap/config.yaml on the box | ours |
| (generated) | ~/llama-swap/config.yaml is written by the settings page (daemon/settings/swap_config.py) from ~/.config/freetoken/registry.json | ours |

llama-swap patch P5 keeps a loaded model through a config reload when its entry did not change.

Two NInfer copies are frozen, not one. The live check on the serving box (task-6-brief.md
Steps 1-2) found `engines/ninfer`'s mobile build refuses the Fable and Twin artifacts at
startup: `FATAL server failed during startup | artifact magic is not NInfer v2`. Those
artifacts are NInfer v3, converted with upstream Neroued/ninfer at
f76e19c0fbd026c86f46005acf2c80c54084bade, and the mobile fork's artifact loader reads only v2.
Per the spec's fallback, `engines/ninfer-upstream/` freezes that same upstream commit to run
fable-27b and twin-27b, while `engines/ninfer/` stays the runtime for quasar-27b (it carries
the QUASAR binding the upstream tree lacks). See `engines/ninfer/FROZEN.md` and
`engines/ninfer-upstream/FROZEN.md`.

Build everything on the serving box with `scripts/engines/build.sh`.
