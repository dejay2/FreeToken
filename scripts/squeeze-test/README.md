# Squeeze Test Kit

The squeeze test kit exercises the FreeToken memory governor by generating realistic memory pressure on the host and GPU while traffic is continuously sent to the model server. It verifies that the server steps expert layers down and up smoothly without dropping requests or raising HTTP errors.

---

## Tool Overview

| Tool | Environment | Description |
|---|---|---|
| `grab.py` | Windows native (`C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe`) | Allocates VRAM via `torch.empty(device="cuda")` and host RAM via `bytearray` in stepped amounts, holds at peak, and releases in reverse. Samples memory status every second to CSV. Supports `--dry-run` to log timetables without allocating. |
| `hammer.py` | WSL / Linux (`python3`) | Fires continuous short chat requests (`max_tokens 64`, temperature 0) against `--url http://127.0.0.1:2020` for `--seconds 600`. Logs per-request latency, token throughput, and errors. Reports summary metrics (request counts, failures, p50 tok/s, min tok/s, max first-token wait). |
| `bench.py` | WSL / Linux (`python3`) | Executes three 200-token completions at temperature 0 (300-word lighthouse keeper prompt), reporting usage and median decode tok/s and wall time. Supports `--label` prefix for pre-/post-comparison. |
| `stub_server.py` | Any (`python3`) | Lightweight local HTTP server answering `/v1/chat/completions` with fixed OpenAI-shaped responses and usage blocks for offline devbox verification. |

---

## Exact Run Recipe for Done-Checklist Items 1–5

### 0. Pre-flight & Baseline (Item 5: Un-squeezed Speed Baseline)

Before introducing any artificial memory pressure, verify server health and record the un-squeezed baseline speed.

1. In WSL on the serving box (`5090`), check server health:
   ```bash
   curl -s http://127.0.0.1:2020/health
   ```
2. Run baseline benchmark:
   ```bash
   python3 scripts/squeeze-test/bench.py --url http://127.0.0.1:2020 --label baseline
   ```

---

### 1. Start Continuous Traffic (Item 4: Zero Failed Requests During Moves)

In a WSL terminal on the serving box, start `hammer.py` to stream continuous requests throughout the squeeze test:

```bash
python3 scripts/squeeze-test/hammer.py \
  --url http://127.0.0.1:2020 \
  --seconds 600 \
  --max-tokens 64 \
  --log hammer.csv
```

Keep this running in the foreground or in a dedicated pane.

---

### 2. Monitor Server and Governor Logs

In a second WSL terminal, tail the settings supervisor logs and query cache residency:

```bash
# Watch governor step actions and cushion triggers
journalctl --user -u freetoken-settings -f
```

To inspect layer residency on demand (owned, pinned, disk):
```bash
curl -s http://127.0.0.1:2020/v1/cache/residency | jq .
```

---

### 3. Apply Memory Pressure from Windows (Items 1 & 2: VRAM and RAM Squeeze)

From the devbox (or Windows terminal), launch `grab.py` using the Windows FreeToken Python venv.

```bash
# Via SSH from devbox:
ssh 5090 'C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe C:\Users\jay\FreeToken\scripts\squeeze-test\grab.py --vram-steps 2,4,8,12 --ram-steps 4,8,16 --hold 60 --step-interval 30 --log grab.csv'
```

What occurs during this run:
- **Card squeeze (Item 1)**: As VRAM allocations step up (2 -> 4 -> 8 -> 12 GB), the governor detects card cushion violations (< 1.5 GB free), issuing `POST /v1/cache/step` to move `gpu_owned` layers to `pinned`, then shrink slot cache. Free card memory stays above the cushion, and hammer requests continue without 503 errors.
- **RAM squeeze (Item 2)**: As host RAM allocations step up (4 -> 8 -> 16 GB), the governor detects main memory cushion violations (< 4.0 GB free), stepping `pinned` layers to `disk`. The model continues decoding with routed experts streamed from the SSD. Chat remains responsive (slower tok/s, zero errors).

---

### 4. Memory Release & Speed Recovery (Item 3: Step-up to Original Speed)

Once `grab.py` completes its peak hold, it releases RAM and VRAM in reverse order (12 -> 8 -> 4 -> 2 -> 0 GB).

- The governor observes sustained free headroom above cushion + 1 rung + margin for the hold-off period (60s).
- It steps layers back up (`disk` -> `pinned` -> `gpu_owned`) and restores slot caches.
- Once `grab.py` finishes and `hammer.py` completes its run, verify recovery:
  ```bash
  python3 scripts/squeeze-test/bench.py --url http://127.0.0.1:2020 --label post-release
  ```
  The median decode tok/s should match the `[baseline]` number within normal measurement noise.

---

### 5. Verification Checklist Summary

1. **Card squeeze held cushion (Item 1)**: `grab.csv` and `journalctl` show VRAM steps down; free VRAM remains above cushion.
2. **RAM squeeze spilled to disk (Item 2)**: `GET /v1/cache/residency` shows layers in `"disk"`; responses continue without failure.
3. **Speed recovered after release (Item 3)**: `bench.py [post-release]` tok/s matches `[baseline]`.
4. **Zero failed requests (Item 4)**: `hammer.py` final summary line reports `failed 0`.
5. **Baseline untouched (Item 5)**: When idle/un-squeezed, normal serving speed is identical to today's numbers.

---

## How to Read the CSVs Side by Side

Both `grab.py` and `hammer.py` log rows stamped with `t` representing the Unix epoch seconds (`time.time()`). Because WSL2 and Windows host share the exact system hardware clock, rows can be directly correlated by `t`.

### `grab.csv` Schema
```csv
t,phase,vram_free_mb,win_free_mb,grabbed_vram_gb,grabbed_ram_gb
1757255740.00,step_up,15200.0,32400.0,2.00,4.00
1757255741.00,step_up,15195.0,32380.0,2.00,4.00
```

- `t`: Wall-clock Unix timestamp (seconds).
- `phase`: Current timetable phase (`step_up`, `hold`, `step_down`).
- `vram_free_mb`: Free VRAM reported by `nvidia-smi`.
- `win_free_mb`: Free physical RAM reported by Windows `Win32_OperatingSystem`.
- `grabbed_vram_gb`: Active CUDA memory allocated by the grabber.
- `grabbed_ram_gb`: Active host RAM allocated by the grabber.

### `hammer.csv` Schema
```csv
t,http_status,wall_s,completion_tokens,tok_s,error
1757255740.45,200,0.450,64,142.2,
1757255741.12,200,0.670,64,95.5,
```

- `t`: Completion wall-clock Unix timestamp.
- `http_status`: HTTP response status code (200, 503, 0 for timeout).
- `wall_s`: Elapsed request wall time in seconds.
- `completion_tokens`: Generated token count.
- `tok_s`: Request decode throughput (`completion_tokens / wall_s`).
- `error`: Error description if failed (empty on success).

### Side-by-Side Analysis

You can merge the two files in Python to evaluate system behavior under pressure:

```python
import pandas as pd

df_grab = pd.read_csv("grab.csv")
df_hammer = pd.read_csv("hammer.csv")

# Merge on nearest timestamp
merged = pd.merge_asof(
    df_hammer.sort_values("t"),
    df_grab.sort_values("t"),
    on="t",
    direction="nearest",
)

# Inspect throughput and errors by grabber phase and allocation level
summary = merged.groupby(["phase", "grabbed_vram_gb", "grabbed_ram_gb"]).agg(
    requests=("http_status", "count"),
    failed=("error", lambda s: (s.fillna("") != "").sum()),
    p50_tok_s=("tok_s", "median"),
    min_tok_s=("tok_s", "min"),
    mean_wall_s=("wall_s", "mean"),
)
print(summary)
```

---

## Local Verification on Devbox (Stub Mode)

When developing without a GPU or when the serving box is in use, verify the kit using `stub_server.py`:

```bash
# 1. Start the stub server on port 12020:
python3 scripts/squeeze-test/stub_server.py --port 12020 &
STUB_PID=$!

# 2. Run hammer against the stub server:
python3 scripts/squeeze-test/hammer.py --url http://127.0.0.1:12020 --seconds 5

# 3. Run bench against the stub server:
python3 scripts/squeeze-test/bench.py --url http://127.0.0.1:12020 --label stub-test

# 4. Run grab in dry-run mode:
python3 scripts/squeeze-test/grab.py --vram-steps 1 --ram-steps 1 --hold 5 --step-interval 2 --dry-run

# 5. Stop stub server:
kill $STUB_PID
```
