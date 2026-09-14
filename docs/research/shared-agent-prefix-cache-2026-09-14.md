# Shared agent prefixes: investigation and proposed plan

Research date: 2026-09-14. Code reviewed: `mtp-upstream-merge` at `b793433`. Status: baseline investigation followed by an implemented v1 on `feat/local-agent-prefix-cache`; see [usage and validation](../local-agent-prefix-cache.md). The live server has not been reconfigured or restarted. Confirmed product scope (Jay): people running locally, with agents sharing one FreeToken server and cache held on that machine's GPU, RAM or SSD. Multi-server sharing and distributed storage are excluded from the roadmap entirely. New local restart-persistence features can follow the first version. The user's 25,000-token system prompt is an illustrative workload, not a measured Claude Code prompt size.

## Recommendation

Build **reusable agent presets on FreeToken's existing prefix cache and checkpoint stores**. Warm a common prompt once, share its GPU attention pages between agents, and keep resumable copies in RAM or on SSD. Add a checkpoint at the shared branching point, coordination so concurrent cold arrivals do not repeat the same prefill, named versions, and bounded retention.

This fork already implements much of the storage machinery. Replacing it with LMCache would introduce an engine integration project before fixing the immediate agent-sharing gaps. Use LMCache as prior art for local cache reuse, storage and transfer design; no integration is planned.

## What the reusable numbers actually are

There are three different things to store:

| Representation | What it saves | Limitation |
|---|---|---|
| Prompt text | Repeated loading/transport of text | The model must still process it. |
| Token IDs (integers) | Repeated tokenization; 25,000 int32 IDs occupy about 100 KB | Almost all model prefill work remains. |
| Computed inference state | Repeating the model's expensive processing of the prefix | Model-specific tensors; much larger than the text. |

The third is the useful optimization. A transformer computes keys and values for attention during **prefill**, then reuses them during **decode**, which generates the answer. For this fork's Qwen hybrid model, a resumable checkpoint also includes Gated DeltaNet recurrent/convolution state, the QSA compressed index, FP8 scales where applicable, and PLE sibling state. Calling it a saved inference checkpoint is more complete than just “decoded prompt.”

Four requests can point to the same immutable attention pages while maintaining separate changing state and tails. Every request still has its full logical context length. Sharing does not make 25k context count as zero, eliminate attention work during generation, or copy an agent's tools, filesystem, and task state. It applies when the agents use the same compatible FreeToken model; FreeToken cannot import the private KV state of Anthropic-hosted Claude.

## What existed in the baseline checkout

| Capability | Evidence | Meaning for this proposal |
|---|---|---|
| Physical sharing of resident prefix pages | `python/freetoken/scheduler/prefill.py:94-142`; `kvcache/radix_cache.py:141-158` | Matched requests use the same indices in different page-table rows. Reference counts protect shared pages. |
| Deduplication after a cold batch | `scheduler/cache.py:625-651`; `tests/scheduler/test_commit_repoints_page_table.py:39-84` | Duplicate pages can be freed and request rows repointed after computation. This does not recover repeated prefill time. |
| RAM and persistent SSD checkpoints | `kvcache/park_store.py`; `README.md:112-184` | `--kv-park off|ram|ssd`; default off. GPU prefix reuse is distinct from this parking flag. |
| Shared parked segments | `park_store.py:1-23`; `tests/kvcache/test_park_store.py:2262` | Parent-linked checkpoints share unchanged page payloads; each endpoint retains its own complete state. |
| Exact prompt checkpoints before state reclamation | `scheduler/cache.py:674-746` | Both RAM and SSD save the final-prefill endpoint. Arbitrary earlier branching points are not guaranteed to have state. |
| Restore validation | `park_store.py:387-519`, manifest/restore paths | Chained keys, full token comparison, layout identity, checksums, and restart recovery already exist. |
| Usage and parking diagnostics | `server/anthropic_api.py:396-407`; `server/responses_api.py:459-466`; `README.md:156-164` | Cache usage reporting is opt-in with `--enable-cache-report`; `/v1/cache/status` exposes parking diagnostics. |

The modes are currently alternatives: `ram` keeps page-locked buffers and loses them on restart; `ssd` persists files and uses bounded RAM transfer windows. It is not yet an automatic GPU → resident RAM cache → SSD cache hierarchy. Operating-system file caching is separate.

Existing hardware evidence is encouraging: the [2026-09-10 RAM report](kv-ram-conversation-switching-2026-09-10.md) records eight full restores of roughly 200k tokens in 1.47–1.86 seconds, processing only 48–114 new prompt tokens on each revisit. Six checkpoints occupied 5.59 GiB. It also documents different expert-cache sizes between comparison boots, so response times do not establish a controlled speedup ratio. Those are previous measurements, not benchmarks rerun for this investigation.

## The two gaps that matter most

**Simultaneous cold arrivals.** Admission matches pending requests before their forward pass (`scheduler/prefill.py:333-365`). The scheduler deliberately skips publishing intermediate prefill chunks and commits at final-prefill completion (`scheduler/scheduler.py:486-547`). Four cold agents can therefore process duplicate prefixes before the cache is visible. The code explains an overlap/double-free hazard behind that intermediate-chunk restriction; removing the restriction is not a safe implementation shortcut.

**A shared prefix needs a state checkpoint at its endpoint.** The hybrid matcher backs up to a node with a live recurrent-state snapshot (`kvcache/hybrid_radix_cache.py:85-97`). A checkpoint after an entire prompt or answer cannot generally be rewound to the earlier end of the common system prompt. Each admitted agent receives a private live state and two ping-pong slots (`scheduler/prefill.py:100-142`), initialized from the shared snapshot before execution (`scheduler/scheduler.py:967-977`). Attention-page sharing alone cannot remove that state requirement.

The Anthropic request models currently have no explicit `cache_control` field for content blocks (`server/anthropic_models.py:34-58`). Supporting client cache-boundary hints would be new work; the mere presence of that field in a client request does not prove FreeToken acts on it.

There is a narrower existing semantic checkpoint hook: `--enable-special-token-ckpt` can preserve a boundary after a single-token tool-call opener (`server/args.py:756-767`; `scheduler/cache.py:448-473`). It is off by default and is not a general system-prompt/skill boundary or named-cache API.

## vLLM and LMCache comparison

vLLM implements automatic prefix caching. Its block identity includes the preceding prefix, block tokens, and extra identity such as adapters or isolation salts; it caches full blocks. Its current native offloading documentation also describes a CPU tier and optional filesystem/object-store tiers, so persistent offloading is no longer solely an LMCache differentiator. [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), [vLLM offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/).

LMCache is a separate KV management layer with a daemon and storage backends including local RAM and disk. Its current multiprocess vLLM connector supports hybrid cache groups, including GDN state transferred as opaque pages. This is relevant architecture to learn from, but does not establish compatibility with FreeToken's custom QSA, PLE, and pool layouts. [LMCache repository](https://github.com/LMCache/LMCache), [hybrid model documentation](https://docs.lmcache.ai/mp/hybrid_models.html).

| Approach | Benefit | Cost / judgment |
|---|---|---|
| **Extend the native cache — recommended** | Keeps the working model-specific serializers, scheduler ownership, and existing RAM/SSD files. Directly addresses agent fanout. | Requires explicit checkpoint scheduling and a small registry/retention layer. |
| Integrate LMCache (evaluated, not planned) | Local RAM/disk cache management and a separate cache process. | Requires a FreeToken adapter, compatibility contract, transfer ownership, packaging, and fresh validation of every state component. The existing native stores already serve the local storage goal. |
| Move serving to vLLM + LMCache | Gets the supported ecosystem integration. | A different serving stack; FreeToken's custom model/offload behavior needs a separate parity investigation. Outside this proposal. |

## Cached skills: exact recipes, with context attached

Recommended conceptual prompt order, where the client and template allow it: stable base and tool definitions → stable project instructions → selected agent role and skill bundle → changing conversation/task. Inspect the actual rendered token stream first: templates may place tools differently, and existing message order must remain semantically intact.

Cache a tree of complete recipes: a shared base; a coding child containing that base plus a coding skill; a review child containing that base plus review instructions. Agents using one child can share that longer prefix. Children share their common ancestor's attention pages.

An unchanged skill inserted after different histories has different incoming context, positions, and recurrent state. It cannot simply be pasted in as the same computed tensor blob. LMCache's CacheBlend selectively recomputes tokens to recover non-prefix reuse; its hybrid documentation says content-aware CacheBlend/CacheGen features do not apply to the opaque recurrent pages. Treat arbitrary skill insertion as a separate research project. [CacheBlend](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html), [hybrid caveats](https://docs.lmcache.ai/mp/hybrid_models.html#caveats).

Only compile recipes actually used. A name such as `reviewer-v3` is an alias to a model-bound recipe and checkpoint; it is not a substitute for comparing the real prefix. Editing the base invalidates descendant recipes; editing one skill invalidates the recipes containing it.

## Sizing the four-agent example

For conventional attention the basic KV formula is `tokens × layers × KV heads × head width × 2 × bytes per value`; hybrid models need their actual pool geometry and state sizes. The README documents this fork's Qwen TP1 geometry as **13,248 B/token for FP8** and **25,344 B/token for BF16**, including its index and FP8 scales.

With a 64-token page, 25,000 common tokens provide 24,960 reusable tokens (390 pages); the remaining 40 tokens are recomputed per agent. For four agents, sharing saves 74,880 token positions: **0.924 GiB FP8** or **1.767 GiB BF16**, before per-agent state, private tails, padding, metadata, and transfer buffers. This is approximately 75% of duplicated common-prefix attention storage, not 75% of total server memory.

With 2,000 unique tokens per agent, rounded attention allocations are 108,032 token positions without sharing versus 33,152 with sharing. State is additional: the recorded Qwen checkpoint state is 115,642,376 bytes (about 110.28 MiB), and live execution uses multiple state slots per agent. Capacity planning must query actual geometry, not assume one state slot or this model's byte cost universally.

FreeToken preallocates its GPU pool. Deduplication increases free pages within that pool; it does not immediately reduce `nvidia-smi` memory use. The [approved dynamic-pool design](../superpowers/specs/2026-09-12-dynamic-kv-pool-design.md) is a separate planned mechanism for reallocating idle capacity to expert slots. This checkout has the design and implementation plan, not its implementation.

Restoring from SSD/RAM avoids prefill but is not instantaneous. Estimate `lookup + bytes / measured effective restore bandwidth + state installation`, then benchmark against cold prefill. Prefer measured end-to-end restore time over raw PCIe microbenchmarks. This box's expert/PLE traffic also uses RAM and SSD bandwidth. Keep the active shared prefix on GPU while agents run; reading it from SSD on every decoded token is a different, much more expensive design.

## Delivery plan

**Confirmed v1 scope: steps 1–3 on one local FreeToken server.** Share resident GPU prefixes, use the existing RAM parking path when configured, coordinate concurrent cold arrivals, and provide named preset controls. Preset registration/coordination can be process-local in v1; restarting may require re-registering and warming presets. Existing SSD parking remains supported, with regression checks where shared code changes affect it. Step 4 extends local durability with preset metadata and a combined RAM/SSD policy. All planned work stays on the user's machine; there is no multi-server or distributed-storage phase. Jay subsequently approved implementation with “carry on then”; the isolated implementation follows steps 1–3.

### 1. Establish the four-agent baseline

Extend the existing benchmark patterns in `scripts/bench/kv_ram_agent_live.py` and `tests/scheduler/test_commit_repoints_page_table.py`. Run 1/2/4/8 agents with a measured 25k common prefix plus different tails. Compare cold concurrent launch, sequential warm-up then fanout, and GPU eviction plus RAM restore within the same server process. A controlled SSD-restart comparison belongs to the later persistence phase.

Record rendered longest-common-prefix length, reusable checkpoint boundary, prompt tokens actually forwarded, GPU/RAM/SSD hit tokens, physical unique pages, private state slots, peak allocated pool bytes, per-agent TTFT p50/p95, restore latency, and total completion time. Hold model, KV dtype, expert slots, batch settings, and output budget fixed. Use synthetic tool-shaped requests first; locally inspect real client prompt differences without putting private prompt content into general logs. Do not equate summed API input tokens with physical allocations.

### 2. Add explicit common-prefix checkpoints and one prefill owner

Add a small checkpoint coordinator near `scheduler/prefill.py`, with ownership/publication handled through `scheduler/cache.py`. For the first release, target the existing text-only QSA/GDN hybrid path; reject unsupported cache families explicitly.

An internal prefill-only work item processes the exact reusable token prefix, with no fabricated assistant turn and no generated output. End at a supported page/GDN boundary: `floor(requested_prefix / boundary_alignment) × boundary_alignment`, where alignment comes from the runtime. Short leftovers become each agent's normal suffix. Capture every state component at this exact boundary, settle device work, publish a refcounted immutable checkpoint, then release waiting requests to their normal match/admission path. Keep an ownership lease until waiters have acquired references; normal scheduling still budgets their private pages and state slots.

Use a single in-flight entry per model/namespace/exact-prefix identity. Concurrent requests for that prefix wait for one computation or restore, while unrelated requests remain schedulable. A cancelled waiter releases only its own reservation. A failed leader releases its resources and lets waiters use ordinary cold prefill; no incomplete checkpoint becomes visible. Coalesce restore as well as compute. Do not route this through the old intermediate `ChunkedReq` commit path.

Start with explicitly registered prefixes. Later automatic discovery can use the longest common prefix of pending requests, after usefulness and fairness are measured. This keeps the first scheduler change bounded.

### 3. Add named presets and bounded retention

Proposed additive management API under `/v1/cache/prefixes`: register, inspect/status, warm, and delete aliases. Registration supplies a canonical request in an existing FreeToken request format plus the desired prefix token count; the server renders/tokenizes through the ordinary path, validates the count, and returns the actual aligned boundary and immutable prefix ID. A local inspection helper reports common-prefix lengths between representative complete requests, making the boundary reviewable. The management path uses existing access controls.

Clients continue sending ordinary complete requests; transparent matching remains the default. The alias is for warming/inspection and is never trusted as proof that arbitrary request bytes match. This avoids requiring Claude Code to adopt a new request protocol. A future compatible client can reference a stored recipe to reduce transport, but the server must reconstruct exactly the same full input.

Store name, version, parent recipe, canonical source/rendering identity, model identity, checkpoint ID, supported boundary, last use, physical bytes, and residency. Keep names separate from payload storage. Deleting an alias releases its retention reference; shared payload is collected only after its last alias, child, active request, and transfer reference is gone.

Use bounded preferred retention with expiry and a configurable byte budget; long conversations must still be admitted. Reuse the dynamic-pool design's existing RAM-expiry policy when implemented rather than introducing competing timers. GPU preference is releasable at safe points before pool rebuild; ordinary live-request locks remain mandatory. Missing/expired payload means the alias can be warmed again.

Bind v1 preset identity to the active model/execution configuration, attention backend/state layout, tokenizer/template recipe identity, and namespace; retain full token validation. Before adding durable preset metadata, strengthen persistent identity with a versioned cache producer/serialization contract. Retain local checkpoint-file validation so changes to the installed weights, model settings or cache format invalidate incompatible entries. Exact copy bytes and stable generation numerics are different guarantees.

Optional client `cache_control` hints can later identify useful boundaries and retention preferences. Preserve the metadata through request conversion and rendering before claiming support; never let a hint bypass matching or memory limits.

### 4. Combine RAM speed with SSD durability, if measured demand justifies it

Keep existing `off`, `ram`, and `ssd` behavior intact. Add an explicit combined policy later, with one logical checkpoint identity represented in multiple tiers. Promote hot entries through a bounded RAM budget, persist selected reusable presets to SSD in the background, and let ordinary LRU/expiry discard colder conversations. Reuse existing segment/checksum machinery and bounded copy buffers.

Declare a checkpoint durable only after the SSD payload and manifest commit; RAM readiness and durable readiness are different status fields. Cancellation, disk full, checksum failure, or pressure falls back to a shorter valid hit/cold prefill. Count each tier's actual bytes and shared-segment ownership accurately. Avoid permanently pinning every named recipe or persisting every transient token.

## Acceptance gates for v1

1. Four cold arrivals for one registered base forward that base only once; four private suffixes still run. Resident requests reference the same common page IDs and distinct mutable state.
2. A 25,000-token base on the documented 64-token geometry shares 390 pages; boundary tails are accounted for. No promise of a zero-token warm prefill.
3. Every restored component is byte-identical to its saved source. Compare scores and a task/answer suite against cold execution under fixed settings; GDN numerical differences can occur across scheduling/kernel paths, so token equality alone is not a universal correctness contract. [LMCache's numerical caveat](https://docs.lmcache.ai/mp/hybrid_models.html#caveats).
4. Cancellation, simultaneous first use, eviction during transfer, model/execution identity mismatch, namespace mismatch, and repeated warm/delete cycles produce valid fallback and no leaked pages/state slots. Existing SSD paths receive regression coverage where affected; new durability guarantees are deferred.
5. The existing cache-rebuild path invalidates GPU handles and safely rehydrates retained RAM checkpoints. Admission counts unique resident shared pages once while reserving each private tail/state requirement. Reuse that contract when the separate dynamic-pool implementation arrives; v1 does not depend on shipping it.
6. A real four-agent workload improves TTFT and peak unique-page demand against the same fixed configuration; a no-reuse workload has no material scheduling regression. Set numeric latency targets from baseline data, not headline speedups from another engine.

The later persistence phase additionally gates durable preset metadata on restart, full-disk, corruption, changed model/template versions, and interrupted-write recovery tests.

## Verification limits of this investigation

Source paths and existing regression/live-test records were reviewed. The implementation adds exact preparation, cold-arrival coordination, bounded aliases/leases and named controls. An existing project virtualenv supplies PyTorch and pytest; CPU tests now cover four-agent physical sharing, exact RAM/SSD byte restoration without another forward, cancellation, rebuilds and transport. No new GPU benchmark was run and the live server was not restarted. The reproducible live benchmark is `scripts/bench/agent_prefix_live.py`; real-model GPU correctness and performance remain unverified. Detailed test evidence is recorded in the [implementation plan](../superpowers/plans/2026-09-14-local-agent-prefix-cache.md).
