# KV parking benchmark — 2026-09-03

This is the M1 synthetic one-TP-rank measurement from the P0 specification. No model or serving code was loaded. The port-2020 server was stopped with the approved stop script while the `M1r` GPU lock was held; it was not restarted.

- GPU lock owner: `M1r`
- GPU: `NVIDIA GeForce RTX 5090`
- Pre- and post-run checks require no 202x/203x listeners, no `ft serve` Python processes, and no orphaned `spawn_main` workers.
- Repetitions: `3` measured after one warm-up per size and method
- Recompute denominator: `1740` prompt tokens/second
- SSD path: `D:\kv-parking-bench`; reads use `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` through the existing Windows reader, so the read number is not a standby-cache read.
- SSD queue: two 4-KiB-aligned pinned `256`-MiB windows; the file bytes are not counted as resident host RAM.
- Each method and size has one warm-up and exactly three measured repetitions; no measured row was discarded or replaced.
- M1 command: `$env:PYTHONPATH='D:\FreeToken\python;D:\FreeToken\scripts\windows-ple-mmap'; & 'C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe' 'D:\FreeToken\scripts\bench\kv_parking_bench.py' --job-id M1r --prefix-tokens 8192 65536 262144 --repetitions 3 --ssd-dir 'D:\kv-parking-bench' --pinned-window-mib 256 --recompute-tokens-per-second 1740 --output 'D:\FreeToken\docs\research\kv-parking-bench-2026-09-03.md'`

## Geometry and resident memory

| prefix_tokens | kv_index_bytes | state_bytes | payload_bytes | payload_gib | vram_peak_pinned_ram_bytes | vram_peak_ssd_bytes | host_ram_pinned_ram_bytes | host_ram_ssd_bytes |
|---|---|---|---|---|---|---|---|---|
| 8,192 | 207,618,048 | 115,642,376 | 323,260,424 | 0.301 | 2,547,580,928.000 | 2,545,483,776.000 | 323,260,424.000 | 536,879,104.000 |
| 65,536 | 1,660,944,384 | 115,642,376 | 1,776,586,760 | 1.655 | 5,466,816,512.000 | 5,464,719,360.000 | 1,776,586,760.000 | 536,879,104.000 |
| 262,144 | 6,643,777,536 | 115,642,376 | 6,759,419,912 | 6.295 | 15,432,482,816.000 | 15,430,385,664.000 | 6,759,419,912.000 | 536,879,104.000 |

The VRAM columns are the maximum process-visible used VRAM during each method's source-plus-fresh-destination run. The host-RAM columns count the active pinned snapshot for `pinned_ram`, and the two bounded SSD queue windows for `ssd`.

## Raw repetitions

| prefix_tokens | payload_bytes | payload_gib | method | repetition | d2h_ms | ssd_write_ms | ssd_write_gib_s | ssd_read_ms | ssd_read_gib_s | h2d_ms | park_wall_ms | restore_wall_ms | recompute_ms | restore_speedup | peak_pinned_bytes | process_working_set_delta_bytes | physical_disk_read_bytes | checksum_match | peak_vram_used_bytes | vram_used_delta_bytes | host_ram_used_bytes | outlier | outlier_fields | outlier_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 323,260,424 | 0.301 | pinned_ram | 1 | 5.713 | — | — | — | — | 5.694 | 5.804 | 5.796 | 4,708.046 | 812.306 | 323,260,424 | 558,137,344 | — | yes | 2,547,580,928 | 394,264,576 | 323,260,424 | no | — | — |
| 8,192 | 323,260,424 | 0.301 | pinned_ram | 2 | 5.698 | — | — | — | — | 5.665 | 5.786 | 5.758 | 4,708.046 | 817.596 | 323,260,424 | 558,170,112 | — | yes | 2,547,580,928 | 394,264,576 | 323,260,424 | no | — | — |
| 8,192 | 323,260,424 | 0.301 | pinned_ram | 3 | 5.697 | — | — | — | — | 5.740 | 5.780 | 5.876 | 4,708.046 | 801.274 | 323,260,424 | 558,170,112 | — | yes | 2,547,580,928 | 394,264,576 | 323,260,424 | no | — | — |
| 8,192 | 323,260,424 | 0.301 | ssd | 1 | 5.792 | 196.558 | 1.532 | 56.275 | 5.350 | 6.205 | 196.558 | 58.779 | 4,708.046 | 80.098 | 536,879,104 | 1,074,081,792 | 323,260,424 | yes | 2,545,483,776 | 381,681,664 | 536,879,104 | no | — | — |
| 8,192 | 323,260,424 | 0.301 | ssd | 2 | 5.697 | 189.588 | 1.588 | 56.079 | 5.368 | 6.070 | 189.588 | 58.470 | 4,708.046 | 80.521 | 536,879,104 | 1,074,081,792 | 323,260,424 | yes | 2,545,483,776 | 381,681,664 | 536,879,104 | no | — | — |
| 8,192 | 323,260,424 | 0.301 | ssd | 3 | 5.751 | 189.526 | 1.588 | 54.786 | 5.495 | 5.988 | 189.526 | 56.886 | 4,708.046 | 82.763 | 536,879,104 | 1,074,081,792 | 323,260,424 | yes | 2,545,483,776 | 381,681,664 | 536,879,104 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | pinned_ram | 1 | 31.132 | — | — | — | — | 30.813 | 31.220 | 30.923 | 37,664.368 | 1,218.013 | 1,776,586,760 | 2,214,596,608 | — | yes | 5,466,816,512 | 1,841,299,456 | 1,776,586,760 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | pinned_ram | 2 | 31.071 | — | — | — | — | 30.848 | 31.162 | 30.958 | 37,664.368 | 1,216.620 | 1,776,586,760 | 2,214,596,608 | — | yes | 5,466,816,512 | 1,841,299,456 | 1,776,586,760 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | pinned_ram | 3 | 31.149 | — | — | — | — | 30.862 | 31.246 | 30.971 | 37,664.368 | 1,216.105 | 1,776,586,760 | 2,214,596,608 | — | yes | 5,466,816,512 | 1,841,299,456 | 1,776,586,760 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | ssd | 1 | 31.456 | 1,048.834 | 1.578 | 304.581 | 5.432 | 32.432 | 1,048.834 | 310.933 | 37,664.368 | 121.133 | 536,879,104 | 151,552 | 1,776,586,760 | yes | 5,464,719,360 | 1,841,299,456 | 536,879,104 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | ssd | 2 | 31.686 | 1,071.267 | 1.545 | 310.413 | 5.330 | 33.380 | 1,071.267 | 317.011 | 37,664.368 | 118.811 | 536,879,104 | 151,552 | 1,776,586,760 | yes | 5,464,719,360 | 1,841,299,456 | 536,879,104 | no | — | — |
| 65,536 | 1,776,586,760 | 1.655 | ssd | 3 | 31.504 | 1,050.811 | 1.575 | 303.316 | 5.455 | 244.296 | 1,050.811 | 337.826 | 37,664.368 | 111.490 | 536,879,104 | 167,936 | 1,776,586,760 | yes | 5,464,719,360 | 1,841,299,456 | 536,879,104 | yes | h2d_ms | likely transient WDDM/CUDA scheduling stall during device transfer |
| 262,144 | 6,759,419,912 | 6.295 | pinned_ram | 1 | 118.145 | — | — | — | — | 163.171 | 118.238 | 163.296 | 150,657.471 | 922.602 | 6,759,419,912 | 8,589,950,976 | — | yes | 15,432,482,816 | 6,824,132,608 | 6,759,419,912 | yes | h2d_ms,restore_wall_ms | likely transient WDDM/CUDA scheduling stall during device transfer |
| 262,144 | 6,759,419,912 | 6.295 | pinned_ram | 2 | 118.306 | — | — | — | — | 117.415 | 118.398 | 117.545 | 150,657.471 | 1,281.695 | 6,759,419,912 | 8,589,950,976 | — | yes | 15,432,482,816 | 6,824,132,608 | 6,759,419,912 | no | — | — |
| 262,144 | 6,759,419,912 | 6.295 | pinned_ram | 3 | 118.115 | — | — | — | — | 120.277 | 118.210 | 120.406 | 150,657.471 | 1,251.248 | 6,759,419,912 | 8,589,950,976 | — | yes | 15,432,482,816 | 6,824,132,608 | 6,759,419,912 | no | — | — |
| 262,144 | 6,759,419,912 | 6.295 | ssd | 1 | 118.979 | 4,022.890 | 1.565 | 1,178.649 | 5.341 | 125.879 | 4,022.890 | 1,191.749 | 150,657.471 | 126.417 | 536,879,104 | 0 | 6,759,419,912 | yes | 15,430,385,664 | 6,824,132,608 | 536,879,104 | no | — | — |
| 262,144 | 6,759,419,912 | 6.295 | ssd | 2 | 119.239 | 4,054.275 | 1.553 | 1,173.407 | 5.365 | 127.035 | 4,054.275 | 1,185.746 | 150,657.471 | 127.057 | 536,879,104 | 0 | 6,759,419,912 | yes | 15,430,385,664 | 6,824,132,608 | 536,879,104 | no | — | — |
| 262,144 | 6,759,419,912 | 6.295 | ssd | 3 | 118.981 | 4,017.983 | 1.567 | 1,181.643 | 5.327 | 127.876 | 4,017.983 | 1,194.711 | 150,657.471 | 126.104 | 536,879,104 | 0 | 6,759,419,912 | yes | 15,430,385,664 | 6,824,132,608 | 536,879,104 | no | — | — |

## Outlier notes

- `ssd` prefix `65536` repetition `3`: `h2d_ms` was an outlier; likely transient WDDM/CUDA scheduling stall during device transfer.
- `pinned_ram` prefix `262144` repetition `1`: `h2d_ms,restore_wall_ms` was an outlier; likely transient WDDM/CUDA scheduling stall during device transfer.

For SSD, `d2h_ms` is an isolated two-window CUDA-event probe so disk scheduling cannot pollute the transfer cell; `ssd_write_ms` is the actual sequential-write wall time. `ssd_read_ms` is the sum of the physical unbuffered `ReadFile` spans; `restore_wall_ms` is the end-to-end read-plus-H2D wall time with the two-window overlap.

## Median and maximum across the three measured repetitions

| prefix_tokens | payload_bytes | payload_gib | method | repetitions | d2h_median_ms | d2h_max_ms | ssd_write_median_ms | ssd_write_max_ms | ssd_write_gib_s_median | ssd_read_median_ms | ssd_read_max_ms | ssd_read_gib_s_median | h2d_median_ms | h2d_max_ms | park_wall_median_ms | park_wall_max_ms | restore_wall_median_ms | restore_wall_max_ms | recompute_ms | restore_speedup_median | restore_speedup_max | peak_pinned_bytes_max | process_working_set_delta_bytes_max | physical_disk_read_bytes_min | physical_disk_read_bytes_max | checksum_all | peak_vram_used_bytes_max | vram_used_delta_bytes_max | host_ram_used_bytes_max | outlier_repetitions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 323,260,424 | 0.301 | pinned_ram | 3 | 5.698 | 5.713 | — | — | — | — | — | — | 5.694 | 5.740 | 5.786 | 5.804 | 5.796 | 5.876 | 4,708.046 | 812.306 | 817.596 | 323,260,424.000 | 558,170,112.000 | — | — | yes | 2,547,580,928.000 | 394,264,576.000 | 323,260,424.000 | — |
| 8,192 | 323,260,424 | 0.301 | ssd | 3 | 5.751 | 5.792 | 189.588 | 196.558 | 1.588 | 56.079 | 56.275 | 5.368 | 6.070 | 6.205 | 189.588 | 196.558 | 58.470 | 58.779 | 4,708.046 | 80.521 | 82.763 | 536,879,104.000 | 1,074,081,792.000 | 323,260,424.000 | 323,260,424.000 | yes | 2,545,483,776.000 | 381,681,664.000 | 536,879,104.000 | — |
| 65,536 | 1,776,586,760 | 1.655 | pinned_ram | 3 | 31.132 | 31.149 | — | — | — | — | — | — | 30.848 | 30.862 | 31.220 | 31.246 | 30.958 | 30.971 | 37,664.368 | 1,216.620 | 1,218.013 | 1,776,586,760.000 | 2,214,596,608.000 | — | — | yes | 5,466,816,512.000 | 1,841,299,456.000 | 1,776,586,760.000 | — |
| 65,536 | 1,776,586,760 | 1.655 | ssd | 3 | 31.504 | 31.686 | 1,050.811 | 1,071.267 | 1.575 | 304.581 | 310.413 | 5.432 | 33.380 | 244.296 | 1,050.811 | 1,071.267 | 317.011 | 337.826 | 37,664.368 | 118.811 | 121.133 | 536,879,104.000 | 167,936.000 | 1,776,586,760.000 | 1,776,586,760.000 | yes | 5,464,719,360.000 | 1,841,299,456.000 | 536,879,104.000 | 3 |
| 262,144 | 6,759,419,912 | 6.295 | pinned_ram | 3 | 118.145 | 118.306 | — | — | — | — | — | — | 120.277 | 163.171 | 118.238 | 118.398 | 120.406 | 163.296 | 150,657.471 | 1,251.248 | 1,281.695 | 6,759,419,912.000 | 8,589,950,976.000 | — | — | yes | 15,432,482,816.000 | 6,824,132,608.000 | 6,759,419,912.000 | 1 |
| 262,144 | 6,759,419,912 | 6.295 | ssd | 3 | 118.981 | 119.239 | 4,022.890 | 4,054.275 | 1.565 | 1,178.649 | 1,181.643 | 5.341 | 127.035 | 127.876 | 4,022.890 | 4,054.275 | 1,191.749 | 1,194.711 | 150,657.471 | 126.417 | 127.057 | 536,879,104.000 | 0.000 | 6,759,419,912.000 | 6,759,419,912.000 | yes | 15,430,385,664.000 | 6,824,132,608.000 | 536,879,104.000 | — |

## P0 decision

P0 qualification — `pinned_ram`: QUALIFIES at 65,536 tokens — median restore 30.958 ms <= 1883 ms, maximum 30.971 ms <= 2000 ms, all checksums passed.
P0 qualification — `ssd`: QUALIFIES at 65,536 tokens — median restore 317.011 ms <= 1883 ms, maximum 337.826 ms <= 2000 ms, all checksums passed.
Winner under P0's rule: **ssd** — both qualify; lower measured host-RAM cost at 262,144 tokens wins (ssd).
