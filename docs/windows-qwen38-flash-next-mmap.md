# Qwen3.8 Flash Next with SSD-backed PLE on native Windows

This is an **unofficial, Desktop-assisted command-line setup** for running
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
from FreeToken PR [#279](https://github.com/FlashML-org/FreeToken/pull/279) on
native Windows. The model's 47.7 GiB PLE/n-gram lookup table is memory-mapped
from SSD instead of copied into RAM or VRAM.

FreeToken Desktop supplies the Windows Python runtime, DLLs, CUDA sources, and
libraries. The Desktop application does not need to be open and its files are
not changed, but this server **cannot be selected or configured in the Desktop
UI**. This is also **not a standalone Windows FreeToken package**; FreeToken
currently publishes Linux wheels only, and the request for a Windows wheel is
still open as [issue #130](https://github.com/FlashML-org/FreeToken/issues/130).

## Tested system

| Component | Tested value |
|---|---|
| OS | Microsoft Windows 11 Pro, build 26200 |
| CPU | AMD Ryzen 9 9950X3D (16 cores) |
| System RAM | 95.6 GiB |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| NVIDIA driver | 610.62 |
| CUDA toolkit | 13.1.115 |
| SSD | Samsung 990 PRO 4 TB NVMe |
| FreeToken Desktop runtime | 0.1.2 |
| Python | 3.12.14 |
| Model checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (125.96 GiB observed) |
| Fork base | PR #279 commit `feaeaa31c0cea385a1c9ee107d4b1053f83b35db` |

Expect the model to consume nearly all of a 32 GB GPU and most of a 96 GB RAM
machine under long prompts. A fast local NVMe SSD is important. Smaller GPUs or
less RAM were not tested.

## Prerequisites

1. Windows 11 with an NVIDIA CUDA 13-capable GPU and current driver.
2. [FreeToken Desktop](https://github.com/FlashML-org/FreeToken) installed, with
   its engine/runtime installation completed. The UI may then be closed.
3. CUDA Toolkit 13.x with `nvcc.exe` available through `CUDA_PATH` or PyTorch's
   CUDA discovery.
4. This fork's `windows-ple-mmap` branch.
5. The complete Hugging Face checkpoint on a local SSD.

The launcher checks the model directory, Desktop Python, installed Windows
kernels, and CUDA toolkit before starting workers. It never edits the Desktop
installation.

## Start the server

Open PowerShell in the fork checkout. Read the model path rather than embedding
a machine-specific path in a shared script:

```powershell
$ModelPath = Read-Host 'Full path to Qwen3.8-Flash-Next-NVFP4'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath
```

The defaults are:

- `127.0.0.1:2020` only;
- `--ple-backend mmap`;
- full 262,144-token usable context;
- automatic GPU expert-cache sizing;
- one active request, with extra requests queued;
- serial expert loading for the proven Windows path (`-ExpertLoad`).

Do not change the host to a public address unless authentication and network
security are added separately.

### Speed-focused 50K allocation

The smaller KV allocation leaves room for 6,002 experts on the GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ContextTokens 50048
```

FreeToken allocates 782 pages of 64 tokens (50,048 total). One page is reserved,
so 49,984 tokens are usable by requests.

### Full 256K allocation

The default reserves the model's full advertised context and leaves room for
4,063 GPU-resident experts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ContextTokens 262144
```

FreeToken allocates 4,097 pages (262,208 total). After the reserved page,
262,144 tokens are usable.

### Expert loading

`-ExpertLoad` selects how the MoE expert banks are read into host RAM and maps
straight onto the server's `--expert-load`:

| Value      | Behaviour                                                                         |
| ---------- | --------------------------------------------------------------------------------- |
| `serial`   | Low-memory reclaimable read, one shard at a time. The launcher default.            |
| `parallel` | Cache-bypassing multi-threaded read (`FILE_FLAG_NO_BUFFERING` on Windows).         |
| `auto`     | Let the loader pick; resolves to `parallel` when the expert tensors are scattered. |

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ExpertLoad parallel
```

`parallel` and `auto` both fall back to the serial build when the unbuffered
reader is unavailable (`FREETOKEN_WIN_UNBUFFERED_IO=0`) or the expert quant has
no parallel provider; the boot log always names the build it took:

```
INFO expert banks: slow path (serial build)
```

### Expert routing statistics

`-CollectRoutingStats` boots with `--moe-collect-decode-freq`, which accumulates a
per-(MoE layer, expert) decode routing histogram and serves it at
`GET /v1/cache/routing`. It is boot-time only: the counters are device-side
`scatter_add_`s that must exist before CUDA graph capture so graph replay re-runs
them, and arming them afterwards would only ever see eager steps. There is no need
to disable CUDA graphs.

```powershell
curl.exe "http://127.0.0.1:2020/v1/cache/routing"
curl.exe "http://127.0.0.1:2020/v1/cache/routing?reset=true"   # window the next workload
```

The response carries `summary` (working set, `experts_for_90pct`, normalized
entropy, the oracle hit rate at the current slot count), `per_layer` realized miss
rates, and `decode_freq` — the raw `[num_layers, num_experts]` histogram.
`reset=true` zeroes the counters after reading. Without the boot flag the route
answers 409 rather than a page of zeros.

Two caveats when reading the numbers: graph capture contributes a handful of
warm-up counts before the first real token, and under MTP speculation the
histogram also counts the routing of draft tokens that were later rejected.

## Wait until it is serving

`GET /health` answers 200 from the moment uvicorn binds the port, which is
long before the weights are in memory — it reports the load in its *body*
(`{"status": "loading", "phase": "expert_banks", "progress": {...}}`), not in
its status code. Do not treat a 200 from `/health` as readiness.

The readiness signal is `GET /v1/cache/status` reporting `state == "serving"`.
On this box the expert banks alone take ~45 s, so allow a 15-minute timeout:

```powershell
$deadline = (Get-Date).AddMinutes(15)
while ((Get-Date) -lt $deadline) {
  try {
    $state = (Invoke-RestMethod http://127.0.0.1:2020/v1/cache/status).state
    if ($state -eq 'serving') { "serving"; break }
  } catch { }
  Start-Sleep -Seconds 2
}
```

The other `state` values are `loading`, `rebuilding` (a live pool resize),
`stopping` and `failed`. While the state is not `serving` the chat routes
answer 503, so a request fired off a bare `/health` 200 will be rejected.

## Check the API

In a second PowerShell window:

```powershell
curl.exe http://127.0.0.1:2020/v1/models
curl.exe http://127.0.0.1:2020/v1/cache/status
```

A text request:

```powershell
$Body = @{
  model = 'Qwen3.8-Flash-Next-NVFP4'
  messages = @(
    @{ role = 'system'; content = 'Answer concisely.' }
    @{ role = 'user'; content = 'What is an SSD-backed memory map?' }
  )
  max_tokens = 256
  reasoning_effort = 'low'
} | ConvertTo-Json -Depth 6

Invoke-RestMethod `
  -Uri http://127.0.0.1:2020/v1/chat/completions `
  -Method Post `
  -ContentType 'application/json' `
  -Body $Body
```

The tested renderer accepts `system`, `user`, `assistant`, and tool messages; it
rejects the OpenAI `developer` role. This guide covers text input only.

Run the public smoke test after startup:

```powershell
$DesktopPython = Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'

& $DesktopPython .\benchmarks\run_qwen38_mmap_smoke.py `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4 `
  --expected-context 262144
```

It checks model information, sequential text generation, streamed reasoning,
content and usage, a parsed tool call, live context, and final server health.
Use `--expected-context 49984` for the 50K allocation.

## Measured 50K versus 256K trade-off

These are controlled throughput measurements, not model-quality tests. Raw
request-level data is in
[`benchmarks/qwen38-flash-next-rtx5090-windows-mmap.json`](../benchmarks/qwen38-flash-next-rtx5090-windows-mmap.json).

### Memory split and startup

| Setting | Total KV pool | Usable context | GPU-cached experts | Fresh startup |
|---|---:|---:|---:|---:|
| Speed-focused 50K | 50,048 | 49,984 | 6,002 | 95.912 s |
| Full 256K | 262,208 | 262,144 | 4,063 | 96.044 s |

### Streaming speed

| Setting | Prompt case | Mean actual prompt | Mean TTFT | Mean decode | Mean TPOT | Mean end-to-end |
|---|---|---:|---:|---:|---:|---:|
| 50K | Short | 524.7 tokens | 2.16 s | 58.7 tok/s | 17.05 ms | 10.87 s |
| 256K | Short | 524.7 tokens | 2.11 s | 54.5 tok/s | 18.39 ms | 11.51 s |
| 50K | Long | 8,204.3 tokens | 6.41 s | 63.5 tok/s | 15.85 ms | 14.51 s |
| 256K | Long | 8,204.3 tokens | 5.98 s | 59.1 tok/s | 17.13 ms | 14.73 s |
| 50K | Context check | 32,780 tokens | 19.22 s | 59.2 tok/s | 16.90 ms | 21.37 s |
| 256K | Context check | 32,780 tokens | 18.87 s | 55.1 tok/s | 18.14 ms | 21.18 s |

The full-context setting decoded about 6.8–7.2% more slowly in this test because
1,939 fewer experts fit in the GPU cache. Startup time and time to first token
were effectively unchanged or slightly better in the measured full-context
runs.

### Peak device observations

| Setting | Prompt case | Peak system RAM used | Peak GPU VRAM used | Peak SSD reads |
|---|---|---:|---:|---:|
| 50K | Short | 87.9 GiB | 29.0 GiB | 0.51 GB/s |
| 256K | Short | 87.5 GiB | 29.1 GiB | 0.47 GB/s |
| 50K | Long | 89.4 GiB | 31.1 GiB | 3.21 GB/s |
| 256K | Long | 89.1 GiB | 31.1 GiB | 3.14 GB/s |
| 50K | Context check | 91.2 GiB | 31.1 GiB | 3.21 GB/s |
| 256K | Context check | 90.7 GiB | 31.1 GiB | 3.28 GB/s |

RAM is the whole-system Windows counter, VRAM is the whole-GPU NVIDIA counter,
and SSD reads are the whole physical disk containing the model. They are not
per-process counters. Normal background OS activity may contribute, although no
other intentional disk/GPU workload ran during the test.

## Benchmark method

For each memory setting:

1. Start a fresh server and wait for `state: serving`.
2. Run one unrecorded 256-input/128-output warm-up.
3. Run three deterministic 512-input/512-output requests.
4. Run three deterministic 8,192-input/512-output requests.
5. Run one deterministic 32,768-input/128-output context check.

Seed `3805090`, reasoning off, temperature 0, top-k 1, ignored EOS, streaming,
and one active request were used. The current FreeToken API treats `max_tokens`
as an exclusive cap (requesting N reports N-1 completion tokens), so the helper
requests one extra API slot and verifies exactly 512 or 128 measured completion
tokens. Chat-template overhead explains why actual prompt usage is slightly
above the generated input size.

To repeat one setting:

```powershell
New-Item -ItemType Directory -Force .\results | Out-Null

& $DesktopPython .\benchmarks\bench_qwen38_mmap_windows.py `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4 `
  --tokenizer $ModelPath `
  --context-label 262144 `
  --seed 3805090 `
  --output .\results\benchmark-262144.json
```

For the 50K server, use `--context-label 49984`. The JSON preserves every raw
timing, usage count, prompt/output hash, memory snapshot summary, and disk sample.

## Troubleshooting

### Desktop runtime not found

Install FreeToken Desktop and complete its engine installation. If it is in a
nonstandard location, pass the exact runtime explicitly:

```powershell
.\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -DesktopPython (Read-Host 'Full path to the Desktop Python executable')
```

### CUDA toolkit not found

Install CUDA Toolkit 13.x and set `CUDA_PATH` to its root. A driver alone is not
enough because first-use kernels may need `nvcc.exe`.

### Port already occupied

Inspect the local ports before stopping anything:

```powershell
Get-NetTCPConnection -State Listen |
  Where-Object LocalPort -ge 2020 |
  Where-Object LocalPort -le 2026 |
  Select-Object LocalAddress, LocalPort, OwningProcess
```

Confirm the owning command is a stale FreeToken process before stopping it.
The server uses the requested API port and nearby worker ports through `+6`.

### First request is slower

First-use CUDA kernel compilation and SSD page faults can make initial activity
slower. Wait for `state: serving` and warm up once before comparing throughput.

## Known limits

- PR #279 is unmerged at the time of this measurement.
- FreeToken Desktop is required as the Windows engine delivery mechanism.
- The server is command-line only and cannot be configured in Desktop.
- This path is text-only as tested; image request parts are not supported here.
- The default allows one active request because a 32 GB GPU has little headroom.
- The compatibility bridge is Windows-specific and not an official FreeToken
  standalone package.
- Performance is one machine/configuration, not a general RTX 5090 guarantee.

## Attribution

SSD-backed PLE mmap support is from PR #279 by
[Ruslan Suleimanov (`ryseek`)](https://github.com/ryseek). The Windows bridge
adapts relevant prior approaches from closed PR
[#232](https://github.com/FlashML-org/FreeToken/pull/232) by
[Max K (`MaxKerkula`)](https://github.com/MaxKerkula) and its contributors,
including loopback TCP worker links, the Selector event loop, installed Windows
kernel reuse, compiler handling, and a parameter-based launcher.
