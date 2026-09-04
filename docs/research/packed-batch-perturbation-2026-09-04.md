# Packed-batch logit perturbation, Qwen3.8-Flash-Next NVFP4, RTX 5090

Date: 2026-09-04. Follow-up 7 of `measurements-kv-context-parallel-live-2026-09-03.md`.
Research only — no product code was changed and nothing here asks for a change.

## Question

D1 (2026-09-03) closed the "crossed reply" scare: nothing leaked between requests. It also
reported that a mixed batch changes the token run of essentially every request in it (15/15
with distinct-word prompts) while solo answers are stable. The open question was how often and
how far greedy decoding actually moves when the same prompt runs at batch size 1 versus
batch size 4.

## Method

Driver: `scripts/bench/kv_batch_perturbation.py`, plus a re-run of D1's own
`scripts/bench/kv_concurrency_perturb.py` for reconciliation.

The server exposes no logprobs — `server/openai_api.py:630` rejects `logprobs` outright and
the chat request model carries no `top_logprobs` — so the top-2 gap and the max absolute logit
shift the follow-up asked for cannot be read from the API. The fallback is text: first-token
match, full-text match, and the token index at which the batched and solo answers first
diverge.

Three prompt classes, ten prompts each, so the flip rate can be split by how tied the top-2
choice is:

- `digit` — D1's marker probes (`request-N-marker-700N`); the answer ends in a number whose
  neighbours are one token away.
- `word` — distinct-word passwords restated in the prompt.
- `ordinary` — plain factual questions with no planted answer.

Each probe's prefix is warmed once. Then, per repeat, the probe is issued through a barrier
together with three fillers that carry a fresh nonce, so they re-prefill cold and the probe
sits in the mixed batch shape D1 used (long cold prefill packed beside a warm extend). The
same prompt is then re-issued alone against the same warm cache and the two answers compared.

Because a zero flip count is worthless if the four requests never actually shared a step, the
driver also records serial time / wall time per group: 1.0 means the requests ran one after
another, 4.0 means all four were in flight for their whole life.

Profile A throughout (`kv_dtype=bf16`, `parking=off`, `moe_cache_size=1116`), port 2020,
temperature 0, `top_p` 1, thinking off. Server was not restarted between runs.

```
python scripts/bench/kv_batch_perturbation.py --model-path <checkpoint> --per-class 10 --repeats 5
python scripts/bench/kv_batch_perturbation.py --model-path <checkpoint> --per-class 2 --repeats 3 \
    --probe-tokens 8192 --filler-tokens 8192 --max-tokens 48 --ignore-eos
python scripts/bench/kv_concurrency_perturb.py --model-path <checkpoint> --family word --runs 15
```

## Numbers

| run | probe context | fillers | stop | pairs | first-token flips | full-text diffs | answer flips | batch concurrency |
|---|---|---|---|---|---|---|---|---|
| A | ~30 tokens | 3 x 2,048 cold | natural (`<\|im_end\|>`) | 150 | 0 | 0 | 0 | 3.58 median, 3.39 min |
| B | 8,192 tokens | 3 x 8,192 cold | forced 48 tokens | 18 | 0 | 0 | 0 | 4.00 median, 4.00 min |
| C (D1's driver) | 8,192 tokens | 1 x 8,192 cold + 2 warm extends | forced 48 tokens | 15 | 0 | **15** | 0 | not instrumented |

Run A by class, 50 pairs each:

| class | first-token flips | full-text diffs | answer verdict flips |
|---|---|---|---|
| digit | 0 / 50 | 0 / 50 | 0 / 50 |
| word | 0 / 50 | 0 / 50 | 0 / 50 |
| ordinary | 0 / 50 | 0 / 50 | n/a (no planted answer) |

## Reconciling run C with runs A and B

Run C reproduces D1's 15/15 exactly at today's `HEAD`, so nothing has silently changed since
yesterday. But splitting each answer at its first `<|im_end|>` shows where the difference
lives:

```
pairs: 15  differ_full: 15  differ_before_first_im_end: 0
distinct answers before <|im_end|>: ['cobalt']
```

All fifteen divergences are in tokens generated **after the model ended its turn**. D1's
driver sets `ignore_eos` and takes 48 forced tokens, so every comparison includes ~35 tokens
of post-turn continuation, where the distribution is close to flat and any perturbation
decides the token. The answer itself was byte-identical in all fifteen. That is why runs A and
B, which compare what the server would really return, find nothing.

## Conclusion

The packed-batch perturbation is real, reproducible and confined to the region no client ever
sees. Across 168 paired batch-1 vs batch-4 comparisons on profile A — with the four requests
measurably in flight together (serial/wall 3.39 to 4.00) — not one first token, finished
answer or answer verdict changed, for near-tied digit labels, distinct-word labels or ordinary
questions alike. The only 15/15 divergence observed needed `ignore_eos` to force generation
past `<|im_end|>` first. Greedy decoding at batch size 4 is therefore stable at the scale that
matters here, and **no code change is warranted**: there is nothing to fix, and adding tie
tolerance to `engine/sample.py` would cost work and buy nothing measurable. D1's original
digit-marker failure remains explained by the probe, not the engine.

## Follow-ups

1. No logprobs. The top-2 gap and max absolute logit shift asked for in the follow-up cannot
   be measured through the API; a `top_logprobs` implementation would turn this from a flip
   count into a margin distribution.
2. Run B's zero is weaker than run A's. Serial/wall overlap proves the requests were in flight
   together, not that they shared a scheduler step; with three 8,192-token chunked prefills the
   probe's decode may have been admitted around them.
3. Run A does not recreate D1's original failure, which needed four markers one digit apart in
   the same batch. That remains a property of the probe, and the corrected probe
   (`scripts/bench/kv_concurrency_distinct.py`) is the one to use.
