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
- a 262,144-token context and a 262,144-token KV pool;
- automatic GPU expert-cache sizing (`-MoECacheSize 0`);
- four active requests, with additional requests queued;
- parallel (cache-bypassing) expert loading (`-ExpertLoad`).

The machine-local `boot-2020.ps1` recipe pins the full-context BF16 budget shown below
and turns integrated MTP off. The generic launcher keeps the expert total automatic so
an operator can choose the total for the selected KV storage mode.

Do not change the host to a public address unless authentication and network
security are added separately.

### Speed-focused 50K allocation

The smaller KV allocation leaves room for 6,002 experts on the GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ContextTokens 50048 `
  -KVCacheTokens 50048
```

FreeToken allocates 782 pages of 64 tokens (50,048 total). One page is reserved,
so 49,984 tokens are usable by requests.

### Full 256K / four-request allocation

The full-context recipe reserves 262,144 tokens for the shared KV pool and allows
four active requests. The machine-local `boot-2020.ps1` uses the BF16 row below:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ContextTokens 262144 `
  -KVCacheTokens 262144 `
  -MaxRunningRequests 4 `
  -MoECacheSize 4188 `
  -GpuOwnedLayers auto `
  -KVDtype bf16 `
  -CudaGraphMaxBS 4
```

| KV storage | Context / pool tokens | Active requests | Total expert slots | Streaming LRU slots | Decode estimate |
|---|---:|---:|---:|---:|---:|
| BF16 (default) | 262,144 / 262,144 | 4 | **4,188** | **1,116** | **~60.2 tok/s** |
| FP8 (optional) | 262,144 / 262,144 | 4 | **5,350** | **2,278** | **~67.0 tok/s** |

The total includes the 3,072 slot-equivalents charged to six GPU-owned layers.
The LRU number is what remains for streaming layers. These token-rate figures are
estimates from linear interpolation of the 2026-09-02 cache sweep, whose 6,750-slot
reference measured 73.1 tok/s; see
[`measurements-moe-cache-sweep-2026-09-02.md`](research/measurements-moe-cache-sweep-2026-09-02.md).
They are not a new live measurement, so recheck `/v1/cache/status` and one decode
on the target machine before treating them as a guarantee.

One chat may use nearly the whole 262,144-token model and pool limit when it is
alone. Four active chats have a worst-case equal share of `262144 / 4 = 65,536`
total tokens each. Prompt tokens plus answer tokens must fit that share; the pool
is shared, not four separate full-length pools.

Integrated MTP is a one-request feature. The four-request recipe turns it off before
launch, including the resident draft head, so inherited MTP settings cannot reject
the configuration or consume its VRAM budget. FP8 remains an optional experiment:
BF16 is the default, and an FP8 boot must use the 5,350-slot line only after its
live quality check passes.

### Expert loading

`-ExpertLoad` selects how the MoE expert banks are read into host RAM and maps
straight onto the server's `--expert-load`:

| Value      | Behaviour                                                                          |
| ---------- | ---------------------------------------------------------------------------------- |
| `parallel` | Cache-bypassing multi-threaded read (`FILE_FLAG_NO_BUFFERING`). The default.        |
| `serial`   | The older one-shard-at-a-time read. Lower peak RAM.                                 |
| `auto`     | Let the loader pick; resolves to `parallel` when the expert tensors are scattered.  |

Measured on the tested system, same flags back to back: the expert-load phase takes
**43 s serial vs 37 s parallel** (-14 %), and standby stays flat across the load
either way while 64 GiB of banks are pinned. Boot-to-serving is a wash (73.1 s vs
73.2 s), so the win is in the load phase, not the headline number.

Parallel costs about **1.4 GiB more peak RAM** for its whole-shard buffers (peak
physical used 85.4 -> 86.8 GiB of 95.6; minimum available 10.3 -> 8.8 GiB). The
loader has a low-RAM fallback to serial, but it reads `/proc/meminfo` and so never
trips on Windows. On a machine with less RAM, ask for serial explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -ExpertLoad serial
```

`parallel` and `auto` both fall back to the serial build when the unbuffered
reader is unavailable (`FREETOKEN_WIN_UNBUFFERED_IO=0`) or the expert quant has
no parallel provider; the boot log always names the build it took:

```
INFO expert banks: slow path (parallel build)
```

### GPU-owned MoE layers

`-GpuOwnedLayers` passes `--moe-gpu-owned-layers`. The named MoE layers keep all 512
experts permanently resident in VRAM and allocate **no pinned host bank at all**, so each
one hands 1.322 GiB of host RAM back and costs 1.322 GiB of VRAM (about 512 LRU slots).
Host RAM is the binding constraint on the tested system (95.6 GiB, ~89 GiB commit), and a
pinned bank cannot be partially released on Windows -- never allocating it is the only way
to give the RAM back.

| Value | Meaning |
| --- | --- |
| *(empty)* | Off. The default. |
| `auto` | The six hungriest layers by measured decode miss rate: `0, 1, 2, 6, 7, 22`. |
| `auto:N` | The first N of that ranked list (`1, 6, 0, 2, 7, 22, 10, 13, 5, 18, ...`). |
| `0,1,2` | An explicit MoE-layer id list. |
| `6` / `0.125` | A count (evenly strided) or a fraction, as `--moe-cpu-layers` reads them. |

The ranking comes from four decode captures on this box
(`docs/research/routing-skew-2026-09-02/`) and is a fixed built-in list, not a runtime
heuristic.

**`-MoECacheSize` is the total expert-slot budget, and the owned layers are charged to
it** -- you do NOT lower it yourself. Keep the number that works without the flag (6750
here) and the LRU shrinks by 512 slots per owned layer, so the card holds exactly the same
MoE bytes with or without `-GpuOwnedLayers`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -GpuOwnedLayers auto `
  -MoECacheSize 6750
```

`auto` therefore needs `-MoECacheSize` **>= 4096**: its six layers charge 6 x 512 = 3072
slots, and the prefill-overlap floor keeps 2 x 512 = 1024 slots for the 42 streaming layers.
Anything lower refuses to boot, naming the size to raise to:

```
ValueError: --moe-cache-size 4095 is the TOTAL expert-slot budget, and 6 GPU-owned MoE
layer(s) charge 3072 slots of it (6 x 512 experts), leaving 1023 for the LRU -- but the
streaming layers need at least 1024. Raise --moe-cache-size (launcher: -MoECacheSize) to at
least 4096, or own fewer layers.
```

In general the floor is `512 * owned_layers + 1024` (or `+ 512` with prefill overlap off).
It is a floor, not a recommendation: 4096 leaves 24 slots per streaming layer, and the
measured working set is ~449 experts per layer. `-MoECacheSize 6750` is what the box was
measured at.

The boot log says exactly what happened:

```
INFO --moe-cache-size 6750 is the total MoE expert-slot budget: 6 GPU-owned layer(s) hold 3072 of those slots, leaving 3678 for the streaming-layer LRU
INFO MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); LRU cache 3678 slots for 42 streaming layers
```

Adding the owned layers on top of the budget instead of charging them to it is what made
the first live run twice as slow: 6 owned layers plus 4400 LRU slots is +1.85 GiB of VRAM,
which left the card 569 MiB free at decode peak and halved throughput even though it moved
11 % fewer expert rows per step. Measurements:
`docs/research/gpu-owned-layers-speed-diagnosis-2026-09-02.md`.

`GET /v1/cache/routing` reports the owned layers as `resident: true` with a null
`miss_rate` (a resident layer cannot miss, and reporting `0.0` would read as a perfectly
cacheable streaming layer), and the `summary` block describes the streaming cache only.
The flag needs `--moe-backend offload`, refuses to overlap with `--moe-cpu-layers`, and is
not supported on an FTW packed checkpoint.


### The VRAM ledger and the post-cache reserve

The expert cache is sized before the engine allocates the resident MTP draft head (2.17 GiB
measured), the CUDA-graph pools and the picture layer-stream workspace, so those bytes have
to be reserved in advance or the card ends up oversubscribed at decode peak -- which is what
halved throughput in live run 2 (569 MiB free, 70.4 -> 34.6 tok/s).

| Launcher | Flag | Default |
| --- | --- | --- |
| `-MoEVramReserveBytes` | `--moe-vram-reserve-bytes` | `-1` = auto: 0.75 GiB for the graph pools, plus 2.25 GiB for the draft head when speculation is on |
| `-MoECacheHeadroomBytes` | `--moe-cache-headroom-bytes` | `-1` = the engine default, 1.5 GiB of free VRAM left after every reservation |

Both are respected by `--moe-cache-auto` (they join the fixed budget before the MoE-vs-KV
split) **and** by an explicit `-MoECacheSize`, which refuses to boot rather than silently
shrinking. The refusal quotes the **total you typed** and names the largest **total** that
fits, so the size it names can be pasted straight back into `-MoECacheSize` (live boot A,
2026-09-02, `-GpuOwnedLayers auto -MoECacheSize 6750`):

```
ValueError: --moe-cache-size 6750 (3678 LRU slots after 6 GPU-owned MoE layers take 3072,
8517058560 B resident) plus 4831838208 B of post-cache reservations
(--moe-vram-reserve-bytes + --moe-cache-headroom-bytes) needs 23546078208 B of the
20861318737 B MoE budget. Either lower --moe-cache-size (launcher: -MoECacheSize) to 5781
slots, or own at most 4 layer(s) at this cache size.
```

The parenthetical is the split, not a second budget: 6750 buys 3072 slots of GPU-owned
residency and 3678 LRU slots. Both numbers the message hands you -- 6750 and 5781 -- are
totals in the unit `-MoECacheSize` takes. This check needs the loaded bank geometry, so it
fires after the expert-bank read (~40 s), not at config time; the LRU-floor refusal above
fires immediately.

Pass `-MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0` to restore the pre-2026-09 sizing
(nothing reserved) if a boot refuses a size you know fits.

Once the KV pool is sized the boot log prints one **VRAM ledger** block naming every term
of the plan, so a boot that will page is visible in the log rather than only in `nvidia-smi`
at decode peak:

```
INFO VRAM ledger (30.42 GiB on the card):
INFO   weights               13.10 GiB
INFO   KV cache               4.21 GiB
INFO   GDN state pool         0.42 GiB
INFO   GPU-owned MoE layers   7.93 GiB (6 layers [0, 1, 2, 6, 7, 22])
INFO   MoE LRU cache          9.49 GiB (3678 slots)
INFO   post-cache reserve     3.00 GiB (MTP draft head, graphs, vision)
INFO   headroom               1.50 GiB
INFO   unaccounted           -9.23 GiB
INFO   named total           39.65 GiB
```

`unaccounted` is what the ledger cannot name -- allocator slack and activations. A negative
number means the plan is over the card: that is the number to read when a boot starts paging.
(The figures above are illustrative, not a measured boot.)


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

These historical runs predate the four-request BF16/FP8 budget above and retain
their original one-request settings. Use the full-context table above for the
current recipe; keep this section for the measured 50K-versus-256K reference.

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

The usual cause is a leftover `python.exe ... spawn_main` child: the launcher was killed
from the console, its `multiprocessing` children outlived it as orphans, and they still
hold the ZMQ side ports, so the next boot dies with

```
ZMQError: Address in use (tcp://127.0.0.1:2033)
```

Use the stop script rather than a `taskkill` by image name -- `ft.exe`, the FreeToken
Desktop daemon on port 1900, also has `freetoken.cli serve` in its command line, and killing
it takes the Windows runtime down with it:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\stop-qwen38-flash-next-windows.ps1 -Port 2020
```

It selects `python.exe` / `pythonw.exe` running `freetoken.cli serve` on that port, every
python descendant of those to any depth, and any orphaned `spawn_main` python whose parent
no longer exists; it never touches a non-python process. It prints what it will kill, kills
it, then waits (up to `-TimeoutSeconds`, default 120) until `nvidia-smi` reports less than
`-VramFreeThresholdMB` (default 3072) in use **and** nothing is listening on the port or the
nine ZMQ side ports above it, and reports which of the two is still holding. Exit code 0
means settled; 1 means it timed out. `-Port 0` (the default) takes every FreeToken server on
the box; `-DryRun` prints the selection and stops.

Waiting matters: the card is only really free once the driver has torn the context down, and
booting into a half-released card is how the last expert bank dies in `cudaHostRegister
failed ... out of memory`.

To inspect the ports by hand instead:

```powershell
Get-NetTCPConnection -State Listen |
  Where-Object LocalPort -ge 2020 |
  Where-Object LocalPort -le 2029 |
  Select-Object LocalAddress, LocalPort, OwningProcess
```

Confirm the owning command is a stale FreeToken process before stopping it.
The server uses the requested API port and nearby worker ports through `+9`.

### First request is slower

First-use CUDA kernel compilation and SSD page faults can make initial activity
slower. Wait for `state: serving` and warm up once before comparing throughput.

## Known limits

- PR #279 is unmerged at the time of this measurement.
- FreeToken Desktop is required as the Windows engine delivery mechanism.
- The server is command-line only and cannot be configured in Desktop.
- This path is text-only as tested; image request parts are not supported here.
- The full-context recipe allows four active requests only within the shared 262,144-token pool;
  four worst-case equal shares are 65,536 total tokens each.
- Integrated MTP remains a one-request feature and is disabled by the four-request recipe.
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
