# Engine adapter contract

An adapter is the program llama-swap runs as a model's `cmd`. Every adapter must:

1. **Make room.** Stop anything else holding the GPU through that engine's proper stop path.
   FreeToken is stopped through its settings helper (`POST /api/server/stop`), which also
   disarms its crash watchdog. If the card cannot be freed, exit non-zero without starting.
2. **Start** the engine on its fixed local port and stay in the foreground while it runs,
   either by exec'ing the engine or by monitoring it.
3. **Be ready only when chats are answered.** The model's `checkEndpoint` must answer 200 only
   then. FreeToken uses `/ready?model=<folder>`; NInfer uses `/health`, which it opens only
   after "engine ready".
4. **Stop fully on SIGTERM.** Exit once the card is released; exit non-zero if the stop failed.

Adding an engine means writing one adapter against this contract, adding a build step to
`scripts/engines/build.sh` if needed, and adding config entries.

FreeToken's adapter takes `--profile model-<registry id>` from the control panel's generated
config. It pushes the model's effective settings (`GET /api/panel/models/<id>/effective`) into
that helper profile (`PUT /api/profiles/<id>` with `replace: true`) and activates it before the
boot. It adopts a running server only when that server runs the same folder, on the same
profile, and the push reported no change.

FreeToken's `sleeping` state (the card given back, the model still loaded) counts as running
this model: the adapter adopts it, and any other adapter stops it fully before starting
(`ninfer.sh` treats every state but `unreachable` as holding the card).
