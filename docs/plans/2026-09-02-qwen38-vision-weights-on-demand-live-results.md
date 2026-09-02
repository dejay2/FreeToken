# Live results: picture weights served from the SSD on demand

Live GPU verification of branch `vision-on-demand` (worktree `D:\FreeToken-vision-on-demand`,
tip `2872727`, six commits over `750d83d`) against the accepted server built from
`D:\FreeToken` (`mtp-upstream-merge`, `8591caf`). Box: Windows 11, RTX 5090 32 GB, 95.6 GiB
RAM. One server at a time, port 2020. Nothing committed, merged or pushed; `D:\Models` never
written. The accepted server was restored at the end and answers.

**Verdict: not safe to merge as it stands.** The feature is correct, fast and free of
regressions — but it does not deliver its purpose. The measured saving is **347 MiB of
private commit and zero working set, zero standby**, against a design criterion of "at least
800 MiB lower". Two defects were found: the boot log always reports `backing=ram`, and the
memory model of the copy-on-write mapping cancels most of the heap saving. Details and the
one-line fix for the first are below.

## 1. What was booted

Four boots, all with the accepted flag set from `boot-2020.ps1` — `-ContextTokens 65536
-KVCacheTokens 65536 -MoECacheSize 6750 -MaxRunningRequests 1 -DenseQuant int8 -EmbedHost
-EnableCacheReport -VisionExecution layer-stream`, the same MTP environment (integrated
speculation, verify graphs, resident nvfp4 draft head) and the same private root, manifest
and picture packages. The only differences between boots are the ones named.

| # | build | picture weights | boot → `state == "serving"` |
| --- | --- | --- | --- |
| 0 | `D:\FreeToken` accepted (`8591caf`) | `ram` (flag does not exist) | already up (booted 07:58) |
| 1 | worktree `2872727` | `-VisionWeights mmap` | 76.9 s |
| 2 | worktree `2872727` | `-VisionWeights ram` | 73.1 s |
| 3 | worktree `2872727` | `-VisionWeights mmap` (repeat) | 71.0 s |
| 4 | worktree `2872727` | no `-EnableVision` (text only) | 67.2 s |

Boot 2 is the correct baseline for every memory comparison: boot 0 could only be sampled
after 37 minutes of idle, by which time Windows had trimmed its working set, so its numbers
are not comparable at serving.

## 2. Side-by-side

### 2.1 Engine process memory at serving, before any picture

The engine is the `multiprocessing` spawn child that holds the model (the largest working set
among the `freetoken.cli serve` process tree).

| metric | branch `ram` | branch `mmap` #1 | branch `mmap` #3 | text-only | mmap − ram |
| --- | --- | --- | --- | --- | --- |
| private bytes (`PrivateMemorySize64`, commit) | 101,912.5 MiB | 101,564.2 MiB | 101,565.4 MiB | 100,703.5 MiB | **−347.5 MiB** |
| working set (`WorkingSet64`) | 68,656.4 MiB | 68,656.9 MiB | 68,663.3 MiB | 68,656.7 MiB | **+0.5 / +6.9 MiB** |
| private working set | 3,315.4 MiB | 3,317.0 MiB | 3,323.4 MiB | 3,316.9 MiB | **+1.6 / +8.0 MiB** |
| peak working set | 70,051.5 MiB | 70,059.8 MiB | 70,064.8 MiB | 70,049.5 MiB | +8.3 / +13.3 MiB |
| system standby list | 10.05 GiB | 7.29 GiB | 10.04 GiB | 10.15 GiB | −0.01 GiB (#3) |

The two `mmap` boots agree to 1.2 MiB on private bytes, so −347.5 MiB is a real number, not
boot noise. (The 7.29 GiB standby figure on boot 1 is confounded: it was sampled right after
the previous server's teardown; boot 3's 10.04 GiB against `ram`'s 10.05 GiB is the clean
comparison.)

The text-only column explains the shape of the result. The whole picture tower costs
**+1,209 MiB** of private commit in `ram` mode and **+862 MiB** in `mmap` mode over a boot
that builds no tower at all. The 856 MiB heap copy really does go away — but a
`mmap(..., ACCESS_COPY)` view charges Windows commit for its full 856 MiB reservation
whether or not a page is ever written, so the private-bytes saving nets out at 347 MiB. This
is design risk 4 landing harder than the design allowed for.

Worse for the stated purpose: the working set and private working set are **identical to
within 8 MiB across all four boots, including the text-only one**. At idle the 856 MiB
picture heap is not resident in `ram` mode either — Windows has already trimmed it — so
there is nothing in the working set left to hand back to the standby list. The standby list
itself measures the same (10.04 vs 10.05 GiB). The 47.7 GiB PLE table gets no more room than
it had.

### 2.2 Picture encode latency

From the server log line `Picture encoder request <uid>: <s> seconds`. The test picture is a
1920×1080 screenshot-like PNG generated for this run (build report with a large verification
code, a bar chart and a console block; ~2,000 picture tokens, prompt 2,109-2,114 tokens). The
"retained screenshot" row is the 1920×1280 fixture from the private evidence tree used by the
layer-streaming acceptance.

| sequence | accepted `ram` (boot 0) | branch `ram` (boot 2) | branch `mmap` (boot 1) | branch `mmap` (boot 3) |
| --- | --- | --- | --- | --- |
| cold (first picture after boot) | 18.389 s¹ | 1.122 s | **0.767 s** | **0.744 s** |
| warm 2 | 0.509 s | 0.322 s | 0.507 s | 0.318 s |
| warm 3 | 0.296 s | 0.303 s | 0.299 s | 0.295 s |
| warm 4 | — | 0.291 s | 0.291 s | — |
| warm 5 | — | 0.289 s | 0.292 s | — |
| retained screenshot | 0.376 s | 0.363 s | 0.856 s / 0.562 s² | 0.368 s |
| small fixtures (smoke suite) | — | — | 0.086-0.325 s | — |

¹ Boot 0's first picture came 37 minutes after boot, so its resident heap had been paged out
and the encode paid to fault 856 MiB back from the pagefile. It is the honest cold number for
the accepted build in that state, not a like-for-like against boots 1-3, and it is the one
measurement in this whole exercise that **breaks the 6 s screenshot budget** — on the
accepted build, not on the branch.
² Taken after the smoke suite, 40 s idle; the immediately-following repeat was 0.562 s.

Every encode on the branch, cold or warm, in either mode, is inside the 6 s budget with a
large margin. `mmap` is *faster* cold than `ram` (0.744-0.767 s vs 1.122 s), reproducibly:
the admission-time `PrefetchVirtualMemory` starts the read before the encode needs the bytes,
while `ram` mode's first encode has to fault its own trimmed heap back in. Warm times are
indistinguishable between the two modes (0.29-0.32 s once past the second encode).

### 2.3 Text throughput after picture work

`ab_send.py <port> <label> 2` (2 reps, median), run after the picture work in every case.
Output tokens/second, and time to first token in seconds.

| build | numbers | essay | code | 8k-chat | 8k-greedy | cold-7k TTFT | warm-turn TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| accepted `ram` | 137.9 | 76.3 | 92.4 | 72.7 | 72.2 | 5.93 s | 1.54 s |
| branch `ram` | 139.2 | 78.1 | 93.4 | 74.6 | 72.5 | 5.84 s | 1.39 s |
| branch `mmap` | 142.5 | 77.2 | 94.7 | 75.8 | 74.1 | 5.51 s | 1.37 s |

Every decode number is far above the 50 tok/s floor, and `mmap` is at or slightly above the
accepted build on every one of them.

### 2.4 Geometry

`/v1/cache/status` `geometry` objects compared field by field:

```
baseline==ram-branch: True   baseline==mmap-branch: True   ram==mmap: True
```

All three: `num_pages=1024`, `page_size=64` → **65,536 usable tokens**;
`moe_cache_size=6750`; `num_mamba_slots=8`; `num_experts=512`; `num_moe_layers=48`;
`cache_budget_bytes=23,564,753,305`; identical `unit_bytes`. Nothing about sizing moved.

## 3. Two defects

### 3.1 The placement report always says `backing=ram` (cosmetic, one-line fix)

Boot 1 and boot 3 logs, verbatim:

```
[2026-09-02|08:40:22|core|rank=0] INFO     Picture weights: mapped 333 tensors, 897862112 bytes in 1 window(s) of model-bf16-00001.safetensors
[2026-09-02|08:40:30|core|rank=0] INFO     Token embedding: host-resident, bytes=1271398400, device=cpu
Picture weights: mode=layer-stream, backing=ram, tensors=333, bytes=897862112, devices=cpu
```

The mapping is built (line 1), but the report says `backing=ram`. Cause: the engine calls
`weight_placement_report()` at `python/freetoken/engine/engine.py:404`, twenty lines
**before** it calls `load_host_tables(config)` at `engine.py:424` — and
`_attach_picture_weight_source` (`models/qwen4_exp/model.py:341`) runs inside
`load_host_tables`. At report time `Qwen4VisionModel._weight_source` is still `None`, so
`weight_backing()` (`models/qwen4_exp/vision.py:228-230`) correctly returns `"ram"`.

The report is therefore wrong in `mmap` mode on every boot, and criterion 1 / checklist D
fail as written. The fix is to attach the source before the report — either move
`_attach_picture_weight_source` out of `load_host_tables` into the loader path, or have
`weight_backing()` consult `mmap_vision_weights(model_path)` directly rather than the
attached attribute. This is a reporting defect only: section 3.2's probe proves the mapping
and the prefetch handshake are live.

### 3.2 The copy-on-write mapping does not return memory to the standby list

Measured in section 2.1: −347 MiB private commit, ±8 MiB working set, ±0.01 GiB standby.
Criterion 3 asked for "at least 800 MiB lower". The stated purpose of the work — "return 856
MiB to the Windows standby list, which is what feeds the 47.7 GiB memory-mapped PLE table" —
is not achieved on this box, because (a) `ACCESS_COPY` charges commit for the whole window
and (b) the resident copy it replaces was already trimmed out of the working set at idle.

This is not a bug in the implementation; it is the design's memory model meeting Windows'
commit accounting. Options, none of them tried here: map the extent read-only rather than
copy-on-write (needs a `torch.frombuffer` source that tolerates a read-only buffer, and the
`pos_embed` carve-out already exists for the one operand that is not a memcpy source), or
accept the feature for its cold-latency win and drop the RAM claim.

## 4. Live proof that the extent really is mapped and is never written through

Checklist E asks whether every `visual.*` tensor still has its `data_ptr()` inside the mapped
window after a real encode. The serving process exposes no introspection endpoint, so this
was answered from outside with `VirtualQueryEx` over the engine's address space, grouping
committed `MEM_MAPPED` regions by allocation base and resolving each to its file with
`GetMappedFileNameW`. A copy-on-write page that has been written converts from
`PAGE_WRITECOPY` to `PAGE_READWRITE`, so the protection histogram is a direct test.

Branch `mmap`, at serving, before any picture:

```
pid=93584  MEM_MAPPED allocations >= 1.0 MiB
  base=0x026898b00000 total=      856.3 MiB  WRITECOPY=856.3MiB
      file=\Device\HarddiskVolume6\Models\Qwen3.8-Flash-Next-NVFP4\model-bf16-00001.safetensors
```

Branch `mmap`, after four encodes (two pictures, two fixtures) — byte for byte the same:

```
  base=0x026898b00000 total=      856.3 MiB  WRITECOPY=856.3MiB
```

Branch `ram`, same probe, same filter:

```
  base=0x0235a98f0000 total=     1214.1 MiB  WRITECOPY=1214.1MiB
      file=\Device\HarddiskVolume6\...\model-bf16-00001.safetensors
```

Text-only boot: **no mapping of that shard at all** (zero regions).

Three things follow. The 856.3 MiB window exists exactly once, is the size the design
predicts, and is aligned as described. **Not one byte of it converted to `READWRITE` across
four encodes** — nothing wrote through a view, so the saving that does exist is not silently
negated. And `mmap` mode removes the 1,214 MiB whole-shard mapping that `ram` mode leaves
open, confirming that `iter_weights` skips the second CPU `safetensors` handle.

## 5. Checklist A-K

| item | verdict | evidence |
| --- | --- | --- |
| **A** two GPU-gated tests | **PASS** | `tests/models/qwen4_exp/test_vision.py`: 21 passed with `torch.cuda.is_available()` True, device_count 1. Named tests run explicitly: `test_layer_stream_matches_gpu_reference_uses_all_blocks_once_and_cleans_up` PASSED, `test_streamed_encode_from_mapped_sources_prefetches_exactly_once` PASSED. |
| **B** H2D from mapped sources | **PASS** | Nine encodes from mapped weights across two `mmap` boots. Answers identical to `ram` word for word (verification code `738214`, all nine numeric fields transcribed correctly). No H2D slowdown: warm encodes 0.291-0.318 s in both modes. Option (a2)'s pinned bounce buffer is not needed. |
| **C** RAM saving ≥ 800 MiB | **FAIL** | −347.5 MiB private commit; +0.5 / +6.9 MiB working set; +1.6 / +8.0 MiB private working set; −0.01 GiB standby. Section 2.1. Reproduced across two `mmap` boots agreeing to 1.2 MiB. |
| **D** placement report `backing=mmap` | **FAIL** (report) / **PASS** (mapping) | Log reads `backing=ram`; cause and fix in section 3.1. The new INFO line is present and exact: `Picture weights: mapped 333 tensors, 897862112 bytes in 1 window(s) of model-bf16-00001.safetensors`. |
| **E** post-encode containment | **PASS** (by address-space probe) | Section 4: 856.3 MiB `WRITECOPY`, 0 bytes `READWRITE`, unchanged after four encodes. The literal `data_ptr()` walk is **not performable** on a live server — no introspection endpoint — so this is the strongest available live substitute; the per-tensor walk is covered off-GPU by `test_vision_weights_ckpt.py`. |
| **F** screenshot latency ≤ 6 s | **PASS** | Cold 0.767 s / 0.744 s, warm 0.289-0.507 s, retained screenshot 0.368-0.856 s. Section 2.2. Encodes 3-5 are within 3% of each other; encode 2 is still warming in both modes (0.507 `mmap` / 0.322 `ram`), the same shape the accepted build shows (0.509 → 0.296). |
| **G** text ≥ 50 tok/s | **PASS** | `mmap` after pictures: 142.5 / 77.2 / 94.7 / 75.8 / 74.1 tok/s, all above the accepted build. Section 2.3. Not run *before* any picture on this pass — the "before" leg is skipped, see section 6. |
| **H** sizing unchanged | **PASS** (relative) | Geometry byte-identical across all three servers (section 2.4). The design's absolute figures (4,097 pages, 262,144 usable, 4,063 experts) belong to a different launch configuration; this run used the accepted `boot-2020.ps1` geometry of 1,024 pages / 65,536 tokens / 6,750 pinned expert slots and proves it does not move. |
| **I** failure paths | **PASS** | Malformed data URL → HTTP 400 `invalid picture data URL`; valid base64 of non-picture bytes → 400 `cannot identify image file`; missing file path → 400 `could not read picture file ...`. A direct text request immediately after all three answered normally. `-VisionWeights ram` on the branch reproduces the accepted behaviour (encode times, geometry, throughput all match; the only textual difference is the new `backing=` field). Text-only boot: banner `Picture weights: disabled`, no tower, **no mapping of the shard at all**, one text request answered. |
| **J** regressions | **PASS** | `run_qwen38_vision_smoke.py` against the `mmap` server: `PASS` on picture source data / direct_path / file_url / loopback_http, cache-private cross-picture requests, two-picture order, **32K picture-bearing chunking**, bounded picture errors, text after picture failures, 32K text-only chunking. One assertion had to be relaxed to run at all — see section 6. Retained pi screenshot answered correctly, and the following 7,082-token pi text request answered in 4.21 s. |
| **K** standby list | **PASS** (measured; answer is "no effect") | Engine private bytes and working set immediately after the last picture: 102,807.9 / 71,336.1 MiB. Sixty seconds later: 102,807.9 / 71,208.1 MiB — private unchanged, working set −128 MiB, system standby 3.74 → 3.81 GiB, free+zero 0.03 → 3.08 GiB, committed 214.01 → 210.73 GiB. `PrefetchVirtualMemory` leaves nothing permanently charged and needs no `EmptyWorkingSet` knob. It also confers no standby benefit, because there was none to confer (section 3.2). |

Growth check demanded by the brief — private bytes must not grow by ~856 MiB across the
picture work, which would mean a write through the copy-on-write view:

| build | private at serving | private after the pictures | growth |
| --- | --- | --- | --- |
| branch `ram` (6 pictures) | 101,912.5 MiB | 103,033.5 MiB | +1,121.0 MiB |
| branch `mmap` #1 (5 pictures + smoke) | 101,564.2 MiB | 102,580.7 MiB | +1,016.5 MiB |
| branch `mmap` #3 (4 pictures) | 101,565.4 MiB | 102,651.9 MiB | +1,086.5 MiB |

`mmap` grows **less** than `ram` over comparable work, and `ram` has no mapping to write
through, so the growth is the picture path's ordinary allocation, not COW faults. The
address-space probe (section 4) settles it independently.

## 6. What was not done, and why

- **Checklist G's "before any picture" leg.** Only the after-picture leg was measured on each
  boot, to keep the number of 75 s boots down. The after-picture number is the one the
  criterion is really about (it is the leg that could regress) and it passes with a 24 tok/s
  margin on the slowest prompt.
- **`benchmarks/run_qwen38_vision_smoke.py` unmodified.** It asserts
  `geometry["num_pages"] == 4097 and usable == 262144` (line 270), which is the 262,144-token
  launch configuration, not the `boot-2020.ps1` configuration this exercise had to reproduce.
  It fails that assertion on the accepted server too. A copy in the scratchpad with that one
  assertion relaxed to `num_pages * page_size >= 65536` ran the full suite; everything after
  the assertion is the benchmark's own unmodified code.
- **Checklist E's literal `data_ptr()` walk on the live tower.** Not performable — see the E
  row. Replaced by the address-space probe.
- **A five-back-to-back cold-cache screenshot run with the page cache genuinely dropped.**
  Windows offers no supported way to drop the standby list for one file, and the box was not
  rebooted. "Cold" here means "first picture after the process started", which is what the
  checklist's ordering asks for.
- **Bit-identity of streamed features between mapped and resident weights (criterion 8) on
  the live tower.** The features never leave the engine process. It is covered off-GPU by the
  branch's own tests; live, the identical answers in B are the available evidence.

## 7. Exact commands

Scratchpad root (`$S`) is
`C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad`;
the helpers written for this run are under `$S\vision`.

```powershell
# Stop the server: kills only python.exe whose command line has "freetoken.cli serve",
# plus its python descendants, never the ft.exe daemon.
powershell -NoProfile -ExecutionPolicy Bypass -File $S\stop-server.ps1
# then wait for nvidia-smi < 3 GB and settle 50-55 s

# Boot the branch. boot-vod.ps1 is boot-2020.ps1 with the private root hard-coded under
# D:\FreeToken\.local, -VisionWeights passed through, and readiness on
# /v1/cache/status state == "serving".
powershell -NoProfile -ExecutionPolicy Bypass -File $S\vision\boot-vod.ps1 `
    -Port 2020 -Root D:\FreeToken-vision-on-demand -VisionWeights mmap -Tag vod-mmap
powershell -NoProfile -ExecutionPolicy Bypass -File $S\vision\boot-vod.ps1 `
    -Port 2020 -VisionWeights ram -Tag vod-ram
powershell -NoProfile -ExecutionPolicy Bypass -File $S\vision\boot-vod.ps1 `
    -Port 2020 -TextOnly -Tag vod-textonly

# Restore the accepted server
powershell -NoProfile -ExecutionPolicy Bypass `
    -File <ple-mmap-mtp-spike scratchpad>\boot-2020.ps1 -Port 2020 -Root D:\FreeToken
```

```powershell
# Checklist A, from the worktree, with the GPU free
$env:PYTHONPATH = "D:\FreeToken-vision-on-demand\scripts\windows-ple-mmap;" +
                  "D:\FreeToken-vision-on-demand\python;$S\pytest-site"
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest `
    tests\models\qwen4_exp\test_vision.py -q -p no:cacheprovider --timeout=900
```

```
# Measurement helpers (all in $S\vision, all read-only against the server)
python make_pic.py                      # writes the 1920x1080 verify-shot.png
python pic_send.py 2020 <path|url> "<prompt>" <max_tokens>
python mem.py <label> [jsonl]           # engine process private/WS/private-WS + standby list
python vq.py <engine pid> <min MiB> [name filter]   # VirtualQueryEx address-space probe
python $S\ab_send.py 2020 <label> 2     # the text benchmark
python run_vision_smoke_relaxed.py --base-url http://127.0.0.1:2020/v1 \
    --fixture-dir <private fixture dir> --evidence-out <private evidence file>
```

Log excerpts quoted above come from `$S\vision\server-vod-{mmap,mmap2,ram,textonly}.out.log`
and the accepted server's `server-2020.out.log`.

## 8. Recommendation

Hold the merge. The code is sound — correct pictures, no regression anywhere, a cold-encode
*improvement*, no write-through, clean failure paths, an honest fallback — but it is sold on a
saving that this box does not show, and it ships a placement report that contradicts its own
behaviour. Two things to settle first:

1. Fix the `backing=` report (section 3.1). One-line ordering change; cheap.
2. Decide what the feature is for (section 3.2). If the RAM claim matters, try a read-only
   mapping so no commit is charged, and re-measure. If the cold-latency win is enough on its
   own — 0.75 s versus 1.12 s, and versus 18.4 s on an idle-trimmed accepted server — then
   restate criterion 3 to what is achievable and merge on that basis.
