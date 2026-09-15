# Prompt cache startup and short-tail latency

Approved scope: persist registered local prompts, warm after model startup, and avoid full-bank transfers for short prompt tails without changing Triton NVFP4 prefill arithmetic.

## Implementation
- [x] Add a bounded Triton NVFP4 short-prefill movement path: ensure routed experts in GPU slots, then use the existing prefill GEMM with slot IDs and slot-bank extent. Preserve decode/MTP paths, CPU/hybrid/disk fallbacks and large prefills. Test movement, capacity boundaries, exact GPU numerical equivalence and interleaved streaming/decode.
- [x] Add bounded, atomic, private, model-scoped storage of successful named registrations. Restore/re-tokenize and warm on startup without generation; deletion removes saved definitions. Expose saved/startup status and failures. Do not persist recent requests implicitly.
- [x] Verify on CPU where supported and RTX 5090 with real kernels. Run controlled live cached-tail benchmarks before/after and compare deterministic output. Preserve the registered prompt and explicitly record any production profile adjustment.
- [x] Review, integrate into mtp-upstream-merge, deploy, restart, confirm startup restores the actual saved prompt, generation succeeds and latency improves.

Baseline: 5ba9960. Local focused suite: 31 pass, 3 CUDA skips, 22 existing CPU routing failures (sgl top-k CPU implementation unavailable). GPU numerical checks are required. Existing service uses Triton NVFP4, TP1, 48 offloaded layers, RAM parking, dynamic KV and a 5952-token named system prefix. Baseline warm-tail one-token measurements: 5.01s after idle, 1.60s and 1.46s on repeats; all reuse 5952/6011 input tokens.

Machine paths, private prompts and raw live evidence stay outside tracked files.

Pre-deployment checks: storage/API/startup/settings checks pass; GPU tests 39 passed / 4 skipped, including exact prefill equality. Read-only review found overlap/prefetch and persistence cancellation/budget races; guards and serialized persistence fixed them and added regression coverage. The unchanged base reproduces all 24 failures in the legacy small-prefill movement suite.


## Production regression follow-up

The first live sparse-prefill attempt exposed two cases absent from the small fixtures:
query-space dedup compilation for 590 routes, and the installed sgl grouping kernel failing
with thousands of cache slots. Sparse prefill now uses the bounded expert-domain lookup
and Triton grouping. GPU regression coverage includes 512 experts, 7,279 slots, high slot
indices, 59 rows, top-k 10, and the real 2,560/640 hidden/intermediate dimensions. Both
first and repeated forwards match full-layer prefill exactly. The focused GPU suite passes
24 tests with 4 backend skips; cache store/startup/API tests pass 39 tests. Two additional
warmup tests pass locally and on the GPU host, checking dummy-state isolation and cleanup
on success and failure. Read-only reviews found no remaining blockers.

Live tail checks at the corrected grouping revision reused 5,952 of 6,011 input tokens.
Repeated first-token times were 0.32–0.37s versus the prior 1.36–2.22s. The capped 64-token
responses matched the baseline output hash exactly. The initial post-boot request took
2.84s; a follow-up adds short-prefill kernel warmup before serving. The local startup KV
floor was raised from 16,384 to 32,768 to fit the usual input plus 16K output reservation
without an immediate pool resize. Final fresh-boot measurements follow below.


Final fresh boot at `a5053f5`: short-prefill kernel warmup ran for 64 dummy tokens in
2.017s before serving; saved-prompt restoration reported ready without errors. The first
real cached request (16K output allowance) reached its first token in 0.909s and visible
answer text in 1.199s. The next request took 0.364s / 0.658s; capped 64-token repeats took
0.320s / 0.617s and 0.333s / 0.675s. All reused 5,952 input tokens. Output hashes matched
the preceding revision for both 16K requests, and matched the original baseline for the
64-token requests. The pool remained at its 32,768-token startup floor without a resize.
Normal service is running from `mtp-upstream-merge`; DFlash remains stopped. Startup
preloading reduces first-use work but does not promise identical first/repeat latency:
new tail routing and generation still require computation and may fetch expert weights.
