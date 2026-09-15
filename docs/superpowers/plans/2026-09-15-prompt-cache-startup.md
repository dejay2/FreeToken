# Prompt cache startup and short-tail latency

Approved scope: persist registered local prompts, warm after model startup, and avoid full-bank transfers for short prompt tails without changing Triton NVFP4 prefill arithmetic.

## Implementation
- [x] Add a bounded Triton NVFP4 short-prefill movement path: ensure routed experts in GPU slots, then use the existing prefill GEMM with slot IDs and slot-bank extent. Preserve decode/MTP paths, CPU/hybrid/disk fallbacks and large prefills. Test movement, capacity boundaries, exact GPU numerical equivalence and interleaved streaming/decode.
- [x] Add bounded, atomic, private, model-scoped storage of successful named registrations. Restore/re-tokenize and warm on startup without generation; deletion removes saved definitions. Expose saved/startup status and failures. Do not persist recent requests implicitly.
- [ ] Verify on CPU where supported and RTX 5090 with real kernels. Run controlled live cached-tail benchmarks before/after and compare deterministic output. Preserve the existing production profile and registered prompt during deployment.
- [ ] Review, integrate into mtp-upstream-merge, deploy, restart, confirm startup restores the actual saved prompt, generation succeeds and latency improves.

Baseline: 5ba9960. Local focused suite: 31 pass, 3 CUDA skips, 22 existing CPU routing failures (sgl top-k CPU implementation unavailable). GPU numerical checks are required. Existing service uses Triton NVFP4, TP1, 48 offloaded layers, RAM parking, dynamic KV and a 5952-token named system prefix. Baseline warm-tail one-token measurements: 5.01s after idle, 1.60s and 1.46s on repeats; all reuse 5952/6011 input tokens.

Machine paths, private prompts and raw live evidence stay outside tracked files.

Pre-deployment checks: storage/API/startup/settings checks pass; GPU tests 39 passed / 4 skipped, including exact prefill equality. Read-only review found overlap/prefetch and persistence cancellation/budget races; guards and serialized persistence fixed them and added regression coverage. The unchanged base reproduces all 24 failures in the legacy small-prefill movement suite.
