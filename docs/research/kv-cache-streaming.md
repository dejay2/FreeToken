# KV Cache Streaming in LLM Inference

Research report, 2026-09-02. Compiled from web sources listed at the end.

## 1. Background: What Is the KV Cache, and Why Does It Get So Big?

When a transformer generates text, each new token's attention layer needs to look back at the keys (K) and values (V) computed for every previous token, in every layer and every attention head. Recomputing those K/V projections from scratch for the entire prefix on every new token would be quadratically wasteful, so inference engines **cache** them the first time they're computed and reuse them for every subsequent token. This is the **KV cache**: a per-sequence, per-layer, per-head store of key and value tensors.

The KV cache exists purely as an optimization to avoid recomputation. The trade-off is memory: the cache grows linearly with sequence length, and it has to be read in full (or in large part) on every decoding step.

**Rough size formula** (per NVIDIA's technical blog and corroborated by several sources):

```
KV cache size (bytes) = 2 x L x h x d x p x n
```

where `L` = number of layers, `h` = number of KV heads, `d` = head dimension, `p` = bytes per value (2 for FP16/BF16, 1 for FP8/INT8), `n` = sequence length, and the leading `2` accounts for storing both K and V. Multiply by batch size for total memory across concurrent sequences.

Concrete example: for **Llama-3-70B at a 128K context**, the FP16 KV cache is reported at **41 GB**, larger than the ~35 GB the quantized model weights themselves might take, and larger than most single GPUs' entire memory. This is why KV cache, not model weights, is often the actual capacity bottleneck for long-context or high-concurrency serving. When the cache doesn't fit, providers either truncate context, evict older cache entries and recompute them, or turn to some form of "streaming."

## 2. What "KV Cache Streaming" Actually Means

The phrase is used loosely to describe several different techniques. In descending order of how commonly the term is used today:

### (a) Tiered offloading: GPU <-> CPU RAM <-> disk/NVMe
**This is the most common meaning** in current systems documentation (vLLM, LMCache, TensorRT-LLM, DeepSpeed). Instead of keeping every token's K/V in GPU memory, the cache is spilled to slower-but-larger tiers (pinned CPU RAM, then local NVMe, then sometimes remote object storage) and streamed back into GPU memory layer-by-layer or block-by-block as needed, overlapping the transfer with compute on other layers or requests. DeepSpeed ZeRO-Inference, FlexGen, HeadInfer, and KVPR exist specifically to make single-GPU inference of huge models possible this way.

### (b) Cross-node network streaming: prefill/decode disaggregation
Production serving stacks often split a request into a compute-heavy **prefill** phase (processing the prompt) and a memory-bandwidth-bound **decode** phase (generating tokens one at a time), run on separate GPU pools. The freshly computed KV cache is then **streamed over the network**, usually RDMA, from the prefill node to the decode node. Mooncake (Moonshot AI's Kimi backend) and NVIDIA Dynamo (using its NIXL transfer library) are the flagship examples.

### (c) Prefix / prompt cache reuse
This is about **persisting** KV cache across separate requests. If two requests, or the same user's next chat turn, share an identical prefix (a system prompt, a long document, a repeated agent context), the KV cache for that shared prefix can be saved and looked up again instead of recomputed. vLLM's automatic prefix caching, SGLang's RadixAttention (radix tree for exact and partial matching), LMCache (a pluggable tiered store for vLLM), Mooncake Store, NVIDIA Dynamo's KV Cache Manager, and llama.cpp's slot save/restore API all implement variants. This category overlaps heavily with (a) and (b): reused cache is often stored using the same offload and transfer machinery.

### (d) StreamingLLM is a different thing
**StreamingLLM** (Xiao et al., MIT, ICLR 2024) uses "streaming" to mean serving an **unbounded input stream** of tokens with a **fixed-size** KV cache, by permanently keeping a handful of initial "attention sink" tokens plus a rolling window of recent tokens, and discarding everything else. Early tokens absorb a disproportionate share of attention weight, so evicting them causes a perplexity blowup, but keeping about 4 sink tokens fixes it. **This is a cache-eviction technique, not a cache-transfer technique.** Nothing moves between memory tiers or machines; content is thrown away and never comes back.

### (e) Related: compression and enabling data structures
Quantizing the KV cache (FP8, INT8, or INT4/2-bit schemes like KIVI, MiniKV) shrinks it 2-4x and makes any streaming approach cheaper, since there's less data to move. **PagedAttention** (vLLM) isn't streaming, but its block-based, non-contiguous layout is the enabling data structure nearly everything above relies on. **InfiniGen** (OSDI '24) speculatively predicts which tokens' KV entries will matter for the next attention step and prefetches only those from host memory.

## 3. How It Works Mechanically

**Memory layout.** Nearly every modern system stores the KV cache in fixed-size **paged blocks** (commonly 16-256 tokens each; vLLM 0.12 grew physical blocks to 0.5-2.5 MB for better DMA efficiency), tracked via a per-sequence **block table** mapping logical token positions to physical block addresses, like virtual memory paging in an OS. This lets a block be shared between sequences, evicted independently, or moved to another tier without touching the rest.

**What triggers a transfer.**
- *Offload direction* (GPU -> CPU -> disk): GPU memory pressure forces eviction of a block still worth keeping (an idle chat session, or completed prefill blocks). TensorRT-LLM's priority-aware LRU manager offloads soon-to-be-evicted-but-reusable blocks to host memory instead of discarding them.
- *Load direction* (CPU/disk -> GPU, or peer GPU -> GPU): a cache lookup hit. A new request's prefix matches blocks in a slower tier, or a decode node needs the cache a prefill node just produced.

**Overlapping transfer with compute.** Transfers run asynchronously on DMA copy engines (`cudaMemcpyAsync` with pinned host memory, separate CUDA streams from the compute stream) so a block can move over PCIe while the GPU computes attention or MLP layers for other tokens or requests. Layer-by-layer or chunk-by-chunk prefetch (start moving layer N+1's cache while layer N computes) is the standard pattern in FlexGen and HeadInfer. NVIDIA's NIXL transfers are documented as non-blocking so prefill GPUs keep serving during the handoff.

**Bandwidth math.** Decoding is memory-bandwidth-bound: one token requires reading essentially all model weights plus the relevant KV cache from GPU HBM (~2-3.35 TB/s on H100/H200). Compare the pipes used for streaming:

| Link | Approx. bandwidth |
|---|---|
| GPU HBM (on-chip) | ~2,000-3,350 GB/s |
| NVLink (H100, per GPU aggregate) | ~900 GB/s |
| PCIe Gen5 x16 | ~128 GB/s |
| PCIe Gen4 x16 | ~64 GB/s |
| RDMA over 400 Gbps RoCE (Mooncake, x8 NICs) | ~190 GB/s aggregate |
| NVMe SSD (sequential) | ~5-14 GB/s per drive |

HBM is roughly 20-50x faster than PCIe, and PCIe is far faster than a single NVMe drive. A single decode step's compute budget (tens of milliseconds for a large model) is nowhere near enough time to pull a large KV cache across PCIe from host RAM, let alone disk. So CPU/GPU offload of the *active* cache genuinely slows per-token latency unless most of the transfer is hidden behind other layers' compute, or only a speculative subset is fetched (InfiniGen), or the workload is throughput-oriented with large batches (FlexGen). vLLM's benchmarking: CPU-offloaded prefix hits cut time-to-first-token by 2-22x (avoiding recomputation dominates), but the offload path is not competitive with cache resident in HBM for latency-critical single requests. Its biggest win is aggregate throughput under concurrent load. NVIDIA separately reports up to 14x faster TTFT from KV cache reuse for long shared prompts.

## 4. Concrete Systems and Papers

| System | Org | What it streams | Mechanism |
|---|---|---|---|
| [vLLM](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) prefix caching + [KV offloading connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) | vLLM project | GPU<->CPU offload; cross-engine transfer via connector API | Async Connector API, `cudaMemcpyAsync`, pinned memory; PagedAttention blocks |
| [LMCache](https://github.com/lmcache/lmcache) ([tech report](https://lmcache.ai/tech_report.pdf)) | LMCache/vLLM ecosystem | Tiered store: CPU RAM, local disk, Redis, S3, Mooncake, NIXL | Plug-in KV layer for vLLM; non-prefix reuse via CacheBlend |
| [SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) RadixAttention | LMSYS | Automatic prefix reuse within/across requests | Radix tree indexing shared token prefixes |
| [Mooncake](https://arxiv.org/abs/2407.00079) | Moonshot AI (Kimi) | Prefill->decode transfer; distributed KV pool over DRAM/SSD | Transfer Engine over RDMA/RoCE, NVMe-oF, NVLink |
| [NVIDIA Dynamo](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving) / [NIXL](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/) | NVIDIA | Prefill<->decode GPU-to-GPU cache transfer | NIXL over RDMA/NVLink, non-blocking |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html) KV cache manager | NVIDIA | Block reuse + host-memory offload | Priority-aware LRU, offload before eviction |
| [FlexGen](https://arxiv.org/abs/2303.06865) | Stanford/CMU/UC Berkeley et al. | GPU/CPU/disk offload of weights + KV, for single-GPU throughput | LP-optimized tensor placement, 4-bit compression |
| [InfiniGen](https://arxiv.org/abs/2406.19707) (OSDI '24) | Seoul National Univ. | Speculative partial KV prefetch from host memory | Rehearsal-based importance prediction |
| [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) | Microsoft | GPU/CPU/NVMe hierarchical offload | Extends ZeRO-Infinity to inference |
| [HeadInfer](https://arxiv.org/abs/2502.12574) | - | Head-wise (not layer-wise) CPU offload | Finer-grained overlap unit |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) slot save/restore | ggml-org | Save/restore full KV state to/from disk per session | `--slot-save-path`, REST API |
| [MLX](https://arxiv.org/abs/2511.05502) | Apple | No PCIe offload needed: unified memory | CPU/GPU share one memory pool; rotating cache for bounded context |
| StreamingLLM | MIT (Han Lab) | *Not* transfer: bounded-size cache via attention sinks | Eviction/compression, different problem |

Broadly: vLLM, LMCache, TensorRT-LLM, DeepSpeed, FlexGen, HeadInfer, InfiniGen do CPU/disk offload; Mooncake and NVIDIA Dynamo/NIXL do network transfer between nodes; vLLM prefix caching, SGLang RadixAttention, LMCache, Mooncake Store, TensorRT-LLM KV reuse, llama.cpp do prefix/prompt reuse. Most production systems combine two or three.

## 5. Practical Considerations and Trade-offs

- **Cache hit vs. miss latency**: on a hit, avoiding recomputation of a long shared prefix is where the big wins are (vLLM reports 2-22x TTFT improvement, NVIDIA up to 14x). On a miss, or when retrieval cost approaches recompute cost, offloading adds pure overhead (tens of milliseconds per CPU-GPU round trip has been cited) with no benefit.
- **Where it helps**: long, shared, static system prompts; multi-turn chat where history is resent each turn; RAG and agentic workloads that repeatedly re-send large tool/document context; high-concurrency serving where GPU memory is the binding constraint.
- **Where it doesn't**: short, unique prompts with little reuse; strict single-request latency SLOs where async overlap can't fully hide the cost; workloads where nearly every token of context differs between requests.
- **Consumer hardware**: on a single consumer GPU plus system RAM and NVMe, llama.cpp's slot save/restore and FlexGen-style offload let you run bigger models or longer contexts than VRAM alone allows, at the cost of slower first-token latency. Apple Silicon's unified memory sidesteps the PCIe bottleneck entirely, though with lower raw bandwidth than a high-end discrete GPU.
- **Compression compounds with streaming**: FP8/INT8 KV quantization roughly halves the data to move with near-zero accuracy loss on most benchmarks; INT4 and below trade more accuracy for a further ~2x reduction.

## Sources

- [Mastering LLM Techniques: Inference Optimization - NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [KV cache offloading - LLM Inference Handbook (Modular/BentoML)](https://handbook.modular.com/inference-optimization/kv-cache-offloading)
- [HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading](https://arxiv.org/pdf/2502.12574)
- [LMCache tech report](https://lmcache.ai/tech_report.pdf)
- [LMCache GitHub](https://github.com/lmcache/lmcache)
- [LMCache docs - Local storage](https://docs.lmcache.ai/kv_cache/local_storage.html)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (arXiv)](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub README](https://github.com/kvcache-ai/Mooncake/blob/main/README.md)
- [Mooncake Joins PyTorch Ecosystem](https://pytorch.org/blog/mooncake-joins-pytorch-ecosystem/)
- [Fast and Expressive LLM Inference with RadixAttention and SGLang - LMSYS](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [Efficient Streaming Language Models with Attention Sinks (StreamingLLM, arXiv)](https://arxiv.org/abs/2309.17453)
- [StreamingLLM GitHub (MIT Han Lab)](https://github.com/mit-han-lab/streaming-llm)
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM, arXiv)](https://arxiv.org/pdf/2309.06180)
- [Disaggregated Serving - NVIDIA Dynamo Documentation](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving)
- [NVIDIA NIXL and Disaggregated Inference - Spheron Blog](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/)
- [KV Cache System - TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- [KV Cache Offloading - TensorRT-LLM examples](https://nvidia.github.io/TensorRT-LLM/examples/llm_kv_cache_offloading.html)
- [Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- [FlexGen (arXiv)](https://arxiv.org/abs/2303.06865)
- [InfiniGen (OSDI '24, USENIX)](https://www.usenix.org/conference/osdi24/presentation/lee)
- [InfiniGen (arXiv)](https://arxiv.org/abs/2406.19707)
- [ZeRO-Inference: Democratizing massive model inference - DeepSpeed](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [llama.cpp discussion - reprocessing large prompts / slot save-restore](https://github.com/ggml-org/llama.cpp/discussions/18244)
- [NVLink vs PCIe - Jarvislabs](https://jarvislabs.ai/ai-faqs/what-are-the-key-differences-between-nvlink-and-pcie)
- [PCIe Gen4 and Gen5 in Servers - ServerMall](https://servermall.com/blog/pcie-gen4-gen5-bandwidth-and-bottlenecks/)
- [Inside vLLM's New KV Offloading Connector - vLLM Blog](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- [KV Offloading Usage Guide - vLLM docs](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- [The State of FP8 KV-Cache and Attention Quantization in vLLM - vLLM Blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [INT4/INT8 KV Cache - lmdeploy docs](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- [oMLX: Apple Silicon-Optimized LLM Inference with Two-Tier KV Caching - Better Stack](https://betterstack.com/community/guides/ai/omlx-apple-silicon/)
- [Production-Grade Local LLM Inference on Apple Silicon - arXiv](https://arxiv.org/pdf/2511.05502)
