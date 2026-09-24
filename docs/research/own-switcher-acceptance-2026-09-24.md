# Own switcher (frozen llama-swap + NInfer): live acceptance, 2026-09-24

Box: Windows 11, RTX 5090, 93.6 GB RAM, WSL `vllm` (cap 76 GB). Build: branch feat/own-switcher
(a58d079 + bfd7879), binary `frozen-v257-freetoken`, from a box worktree `~/FreeToken-own-switcher`.
The service unit now runs `~/.local/share/freetoken-engines/bin/llama-swap` (old unit and config
kept as `*.bak-*` / `config.yaml.bak-before-frozen`).

| # | Check | Result |
|---|---|---|
| 1 | QUASAR through tailnet 12020 | pass (45 s incl. a 25 s memory-gate wait, then answered) |
| 2 | Latest wins: FreeToken requested, QUASAR 10 s later | pass: first caller 409 `model_superseded` after 10.0 s, QUASAR answered 17.4 s later, FreeToken never started, helper `unreachable`, watchdog disarmed |
| 3 | QUASAR → Fable → Twin → QUASAR swaps; speed | pass; frozen `engines/ninfer` equals the original `ninfer-mobile` build under identical conditions (code 261-285, chat 177-184, long review 228-229 tok/s for both). The box ran about 10% below this morning's absolute numbers for both builds |
| 4 | Memory gate | pass, live: QUASAR waited 25 s (Windows free 4.1 → 4.6+ GB after the FreeToken stop). FreeToken refused with 503 `not_enough_memory` after 300 s (Windows free peaked at 56.9 GB; the rule needs 60 + 6) |
| 5 | `top_k: 40` to QUASAR | pass (clamped to 20) |
| 6 | Fable `ttl: 60` | pass: unloaded 83 s after the request, 0 `ninfer-serve` processes, card back to 1.8 GB |
| 7 | FreeToken full boot, memory measured (gate bypassed for this one boot) | answered in 161.5 s. Linux used peaks at 62 GB and settles at 60. Windows free went 57.5 → **0.3 GB** minimum, steady 5-6 GB with FreeToken loaded. `ramNeedGB: 62` |
| 8 | `wsl --shutdown`, then start | pass: `llama-swap` and `freetoken-settings` active, 5 models, nothing running, card 2.1 GB, tailnet 12020 lists all models |

## Findings

- **NInfer needs two runtimes.** The QUASAR-capable mobile fork (d4bc75db) refuses Fable and
  Twin with `artifact magic is not NInfer v2`, because they are v3 artifacts from upstream
  f76e19c0. Both are now frozen, as `engines/ninfer` (QUASAR) and `engines/ninfer-upstream`
  (Fable, Twin). Fable on the frozen upstream build ran at 190-229 tok/s.
- **The gate's FreeToken refusal was correct.** A FreeToken boot costs about 57-60 GB of Windows
  memory. With Windows itself holding about 30 GB, loading it would take Windows to 0.3 GB,
  which is the freeze zone. Jay's decision: keep the 6 GB cushion. FreeToken loads only when the
  PC has room; otherwise it waits up to 5 minutes and answers 503 with the numbers.
