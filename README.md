<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## This fork: Windows PLE-mmap, still-picture serving, and the MTP spike

This fork (github.com/dejay2/FreeToken) tracks upstream FlashML-org/FreeToken and adds three features developed and tested on Windows 11 with an RTX 5090 for Qwen3.8-Flash-Next-NVFP4. The branch `mtp-upstream-merge` contains the upstream main branch plus this work.

### Full-context and concurrency budget

The Windows launcher defaults to a 262,144-token context and KV pool with four
active requests. On the tested RTX 5090, the machine-local `boot-2020.ps1` recipe
uses BF16 with **4,188 total expert slots** and **1,116 streaming-LRU slots**, and
turns integrated MTP off because MTP accepts only one active request. Four chats
share the pool, so their worst-case equal share is 65,536 total prompt-plus-answer
tokens each. The optional FP8 recipe uses **5,332 total expert slots** and **2,260
streaming-LRU slots**. Its corrected ledger is 13,248 B/token × 262,144 tokens,
with page rounding, the post-cache reserve, and headroom subtracted per
`cache_budget.py`. See the [Windows guide](docs/windows-qwen38-flash-next-mmap.md)
for the full comparison and budget assumptions.

### Experimental FP8 QSA KV storage

Main QSA K/V storage is BF16 by default. Select the experimental E4M3 path at server startup
with `--kv-dtype fp8`, `FREETOKEN_KV_DTYPE=fp8`, or the Windows launcher's `-KVDtype fp8`.
The unset/`bf16` path allocates the original tensors and uses the original raw-copy store. FP8 is
currently accepted only for the QSA pool and requires a CUDA GPU with native E4M3 support
(compute capability 8.9 or newer); other cache families reject it instead of silently changing
their storage.

The compressed QSA index remains BF16, so `qsa/score.py` and its selected-token path are unchanged.
Each main K/V write chooses a separate FP32 scale for every stored token and local KV head, then
stores `clamp(value / scale, -448, 448)` as E4M3. This online layout never changes the scale of an
older cache row. At the shipping TP1 geometry (12 QSA layers, 2 KV heads, width 256), its exact
paged costs are:

| Main QSA K/V mode | Main K+V | BF16 compressed index | K/V scale metadata | Total per token |
|---|---:|---:|---:|---:|
| BF16 (default) | 24,576 B | 768 B | 0 B | **25,344 B** |
| FP8 E4M3 | 12,288 B | 768 B | 192 B | **13,248 B** |

The layer-3 capture measured 2.20654501% aggregate output relative-L2 error and 3.78707871%
worst-head error with per-head scaling; E4M3 arithmetic predicts about 2.4% aggregate rounding
error. These numbers justify defect-detection bounds, not production acceptance: the offline
selection proxy matched captured production selection only 29.4921875%. Final acceptance still
requires the first 48 temperature-0 generated tokens to match BF16 exactly and a coherent answer
to a 250,000-plus-token prompt. Until that live check passes, BF16 remains the supported default.

### PLE table backends

Qwen3.8-Flash-Next has a 47.7 GiB n-gram (PLE) lookup table. Choose where it lives with `--ple-backend` or the launcher's parameter:

| Backend | Description | Default |
|---------|-------------|---------|
| `disk` | Linux-only io_uring row store (reads rows straight from checkpoint files) | Linux |
| `mmap` | Demand-pages the safetensors table with a bounded LRU row cache; includes a Windows PrefetchVirtualMemory fast path | Windows |
| `pinned` | Preloads the whole table into page-locked host RAM | non-Linux |

The launcher script `start-qwen38-flash-next-mmap-windows.ps1` defaults to `--ple-backend mmap`.

### KV prefix parking

Hybrid QSA/GDN conversations can keep completed prefixes outside GPU memory and restore them on
the next matching turn instead of recomputing. The feature is **off by default**. In `off` mode
FreeToken constructs no parking store, copy stream, page-locked buffer, worker thread, directory,
or manifest, and the existing cache paths do not add a device synchronization.

Choose `--kv-park off|ram|ssd`, `FREETOKEN_KV_PARK`, or the Windows launcher's `-KVPark`:

| Setting | Environment / Windows launcher | Default | Meaning |
|---|---|---:|---|
| `--kv-park-idle-ms` | `FREETOKEN_KV_PARK_IDLE_MS` / `-KVParkIdleMs` | `0` | Park at the first idle scheduler point; raise it to retain recent prefixes on the GPU. |
| `--kv-park-min-tokens` | `FREETOKEN_KV_PARK_MIN_TOKENS` / `-KVParkMinTokens` | `8192` | Smallest prefix worth parking; it must be a multiple of the model's page size. |
| `--kv-park-ram-gib` | `FREETOKEN_KV_PARK_RAM_GIB` / `-KVParkRAMGiB` | `2` | LRU budget for full entries in page-locked host RAM. |
| `--kv-park-ssd-dir` | `FREETOKEN_KV_PARK_SSD_DIR` / `-KVParkSSDDir` | `~/.cache/freetoken/kv-park` | Persistent SSD root; tensor-parallel ranks use separate subdirectories. |
| `--kv-park-ssd-gib` | `FREETOKEN_KV_PARK_SSD_GIB` / `-KVParkSSDGiB` | `32` | On-disk LRU budget per tensor-parallel rank. |
| `--kv-park-window-mib` | `FREETOKEN_KV_PARK_WINDOW_MIB` / `-KVParkWindowMiB` | `256` | Size of each of two shared page-locked SSD windows; restore alternates both to overlap disk reads with GPU copies (512 MiB total by default). |

`ram` retains exact QSA K/V, compressed-index, FP8-scale (when enabled), GDN and PLE sibling-state
bytes until its RAM LRU drops them. A single bounded background worker performs the copy/write;
the scheduler keeps source pages and the state slot owned until device-to-host copying finishes.
A full worker queue falls back to ordinary cache eviction instead of blocking request admission.

`ssd` keeps the same bytes in version-2 files with a 4-KiB header, SHA-256 payload checksum, and
atomic manifest. Files survive a server restart; a missing or invalid manifest is rebuilt by
scanning valid headers, and incomplete temp files left by dead writers are removed before new
writes begin. The persistent fingerprint pins one Hub snapshot before model loading and covers the
active safetensors or FTW index and referenced shards. One payload pass both verifies the checksum
and feeds restore through the two alternating windows, while later manifest work may finish after
the source GPU storage is released. Each tensor-parallel rank has its own files, and ranks agree
on the reusable length and restore result before installing a parked prefix in the normal hybrid
radix cache.

Both stores compare the full token IDs after their rolling content hash, so a collision or stale
checkpoint/layout entry is a miss, not incorrect output. The Windows launcher always passes the
selected mode explicitly, so `-KVPark off` overrides an inherited `FREETOKEN_KV_PARK`.
`GET /v1/cache/status` reports the mode, parked entry count and bytes, hits, misses, and the latest
restore time under `parking`.

Parking currently applies only to the hybrid QSA/GDN radix cache. Picture/private prefixes remain
uncached. A parked hit must beat the live GPU match by one page in RAM mode or 4096 tokens in SSD
mode (the measured transfer break-even), and restore leaves one GPU page for the current turn's
tail; otherwise FreeToken uses an ordinary cold prefill.

### Still-picture (vision) serving

Vision support is opt-in and disabled by default. Enable it with environment variables or launcher flags:

- `FREETOKEN_LOAD_VISION=1` or the launcher's `-EnableVision` flag loads the vision tower and multimodal embedder (~1 GiB of bf16 GPU weights).
- `-VisionExecution layer-stream|gpu` selects where picture components run: `layer-stream` (the launcher's default) keeps vision weights on CPU and stages bounded components per encode; `gpu` runs everything on the GPU (and is the default when `FREETOKEN_VISION_EXECUTION` is set directly without the launcher).
- `-VisionPackagesPath` points to the installed packages (Pillow, TorchVision).
- Install vision dependencies: `pip install "freetoken[vision]"` (pulls pillow>=11,<13 and torchvision>=0.26,<0.27) or run the helper script `install-qwen38-vision-deps-windows.ps1`.

### MTP speculative decoding (feasibility spike)

MTP (Multi-Token Prediction) speculative decoding is a feasibility spike and is off by default. It requires:

- `FREETOKEN_MTP_PRIVATE_ROOT`: path to a private spike root (not in the repo) containing MTP head weights derived from the checkpoint and pre-converted NVFP4 expert banks. Required whenever any MTP mode is active.
- Key environment variables for configuration:
  - `FREETOKEN_MTP_SPECULATE` (0|1): Enable integrated speculative decoding.
  - `FREETOKEN_MTP_SPEC_DEPTH`: Maximum draft chain length (1-5, default 5).
  - `FREETOKEN_MTP_SPEC_GRAPH` (0|1): Capture speculation cycles in CUDA graphs.
  - `FREETOKEN_MTP_SPEC_EMA_ALPHA`: Smoothing factor for acceptance EMA (default 0.3).
  - `FREETOKEN_MTP_SPEC_MIN_EMITTED`: Minimum emitted tokens per cycle to keep speculating (defaults to the measured break-even, capped at depth+1).
  - `FREETOKEN_MTP_SPEC_COST_AWARE` (0|1): Adapt the acceptance bar to measured wall time (default 1).
  - `FREETOKEN_MTP_SHADOW` (0|1): Observer-only mode (no integration).
  - `FREETOKEN_DENSE_QUANT` (int8|): W8A16 quantization for dense projections.
  - `FREETOKEN_EMBED_HOST` (0|1): Keep token embeddings in pinned host RAM.
- Limitation: `--ple-backend disk` is incompatible with MTP (the capture path does not enter forward_host_ctx); use `mmap` or `pinned` instead.
- Tests in `tests/spike/` skip unless `FREETOKEN_MTP_PRIVATE_ROOT` points at a valid private root.

### Running the tests on Windows

There is no uv build on Windows. Tests run with the FreeToken Desktop venv's Python
(it already ships torch + CUDA). Two things must be on `PYTHONPATH`, in this order:
the repo's `python/` (the Desktop venv carries its own installed `freetoken`, which
would otherwise shadow the checkout) and the `scripts/windows-ple-mmap` shim
(`sitecustomize.py`: selector event loop, ZMQ over loopback TCP, kernel DLL reuse).

```powershell
$env:PYTHONPATH = "python;scripts\windows-ple-mmap"
python -m pytest tests/ -m "not slow"
```

GPU tests skip themselves when CUDA is not available; `-m "not slow"` deselects long-running tests (big kernel sweeps, real-checkpoint reads).

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
