# Anthropic system-message order and checkpoint reuse

## Observed cause

Real agent requests append system messages containing changing token budgets. The previous
Anthropic adapter merged every system message into the opening system prompt. Even when
the agent only appended to its history, the rendered prefix changed near the beginning.
An exact recurrent-state checkpoint near the old prompt's end could no longer match.

A private nine-request sample was replayed through the serving Qwen3.8-Flash-Next tokenizer.
In its first main-turn transition, only 14,100 of the old 105,459 tokens remained an exact
prefix. Preserving chronological system turns produced a 105,619-token previous prompt,
all of which remained a prefix of the next main turn. The subsequent main turns similarly
preserved their complete previous prompts. This diagnoses the sampled traffic; it does not
attribute every earlier cache miss to this cause. Conversation age is not an eligibility gate.

The private sample is not a repository fixture. Tests use synthetic archives and budget updates.

## Compatibility

The Anthropic converter merges only leading system messages and retains later system turns
in order. A private rendering hint carries that intent through the existing tokenization
message, including the count-tokens path. The renderer consumes the hint before calling
any checkpoint template or custom encoder.

For the recognized Qwen ChatML structure, the renderer substitutes only the late-system
position guard. It renders that turn using the template's existing system-content validator
and ChatML system markers. Tools, reasoning, other validation, and the surrounding template
remain intact. The derived template is request-local; no model file or shared tokenizer is
mutated. Named/default/tool-specific template selection still follows transformers.

Unknown templates and custom encoders retain the previous system-hoisting behavior. This
limits the compatibility change to the template structure validated here. No system message
is silently discarded or converted into a user message. Ordinary requests without later
system messages follow their existing rendering path.

## Validation

The regression suite renders and encodes through real Jinja/transformers components. It
covers exact checkpoint prefixes, system-role chronology, reasoning and tool results,
unchanged ordinary prompts, legacy fallback, custom encoders, selected templates, and
unrelated validation failures. The implementation also reproduces the offline rendering
projection for all nine sampled requests with the full serving tokenizer. Six ordinary
renderings, with/without tools and with thinking enabled/disabled/default, are unchanged.

The full server/tokenizer CPU run has one existing failure:
`test_ple_backend_is_exposed_by_the_server_cli` expects a pinned PLE default where the Linux
CLI selects disk. It also fails on the unchanged parent revision. It is unrelated to this fix.

For GPU acceptance, use `scripts/bench/kv_ram_agent_live.py` with RAM parking and cache
reporting enabled. It generates two synthetic long archives, asks the model to report
their exact facts through a tool call, carries forward its reasoning/tool history, appends
system budget updates, and runs both concurrent continuations and serial revisits. It checks
frontend/local/generated token-count agreement, cached input, bounded tail prefill, unchanged
instance, and complete RAM attention/state transfers on serial revisits. An idle-only cache
rebuild at the current state-slot count clears GPU prefixes before those serial requests,
so their host-transfer checks do not depend on incidental GPU memory pressure or idle delay.
Initial responses must contain reasoning, which is carried into the next requests. The
benchmark records partial results if a later assertion fails. Run it against an
idle validation server: unrelated traffic can evict a family or replace the global transfer
diagnostic before the test reads it. `--check-prefix` runs only local tokenization.

Initial cold fills, rewritten historical content, and memory-driven family eviction can
still cause misses. A concurrent branch may arrive before the first checkpoint exists or
diverge before the newest endpoint; an older matching endpoint is then needed. Preserving
system order makes valid checkpoints reusable, rather than making every request a hit.
