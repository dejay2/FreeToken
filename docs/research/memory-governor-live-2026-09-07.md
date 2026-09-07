# Memory governor: live acceptance on the RTX 5090 WSL box, 2026-09-07

Branch `memory-governor` at 8da8e4c. Daily 262k profile (MaxRunningRequests 1, MoECacheSize auto,
no owned layers, MTP off), governor on with cushions 1.5 GB card / 4 GB Windows RAM. Traffic:
`scripts/squeeze-test/hammer.py` (64-token chats, one at a time) inside WSL; pressure:
`scripts/squeeze-test/grab.py` run natively on Windows with
`C:\Users\jay\AppData\Local\Programs\Python\Python312\python.exe` (the Desktop venv launcher
cannot spawn; the `\\wsl.localhost` share is unreadable from an ssh session, so the script
was copied to `C:\Users\jay\`). Bench: `bench.py`, three 200-token completions, median.

## Results against the job card

| # | Item | Result |
|---|---|---|
| 1 | Card squeeze: steps down, cushion held, chat continues | PASS. Grab 2/4/8 GB: shelf 6429 -> 5405 -> 4381 slots (512-slot rungs), free card memory 2.65 -> 2.7 -> 1.6 GB, one owned layer unparked first. |
| 2 | RAM squeeze: experts to the SSD, chat continues slowly | PASS. Grab 6/8 GB: 11 layers spilled one per step, Windows free recovered from 0.3 GB, decode 2.6-14 tok/s with 4-17 disk layers, first-token wait up to 28 s. |
| 3 | Release: back to today's speed | PASS with a caveat: 53.3 tok/s after every layer was home (baseline 55.8), but the automatic recall stalled at 2-5 disk layers (see flap below); the last four were recalled by hand through `POST /v1/cache/step`. |
| 4 | Zero failed requests during moves | PASS after fix 8da8e4c: 169/169 and 25/25 on the last two runs. Before the fix, 10 of 63 chats got HTTP 503 from the adapters' front-door gate. |
| 5 | No squeeze: speed matches today | PASS: 55.3 vs 55.8 tok/s; boot leaves 4.0 GB free on the card (headroom 1.5 GiB applied) instead of 0.6 GB. |
| 6 | Real game | Not run (Jay's check). |
| 7 | Existing checks | PASS on the devbox: only the pre-existing failures remain (3 in settings, 1 in server, 149 in scheduler, all failing at 046d328 too). |

## Defects found live and fixed on the branch

| Commit | Defect |
|---|---|
| 6978700 | `residency_report()` keyed layers by int; msgpack `strict_map_key` in the tokenizer workers killed `freetoken-detokenizer-0` on the first `/v1/cache/residency` and took the API down. |
| 50fec68 | The helper's systemd PATH has no `/mnt/c/Windows`, so `powershell.exe` failed and the RAM axis read the VM's MemAvailable (16.5 GiB) while Windows had 2.8 GB. Absolute path fallback. |
| c21d396 | After `systemctl --user restart freetoken-settings` the adopted server had no governor loop; the loop now starts with the helper. |
| 8da8e4c | OpenAI/Anthropic/Responses routes answered 503 on `maintenance_state == "rebuilding"` before `new_user`'s wait queue; they now wait out a rebuild (120 s cap). |

## Other measurements

- Disk copy: 63.46 GB written in 213 s (320 MB/s) at first boot, skipped afterwards.
- Boot with the new code: 215-335 s to healthy (was 140 s); the difference is not yet attributed.
- Step down cost: each move is one rebuild of a few seconds; requests queue.
- Decode with any DISK layer runs eager (graphs deferred): 8-11 tok/s with 2-4 disk layers,
  2.6 tok/s worst case seen with 17.

## The recall flap (open, governor left OFF overnight)

After the grab released, Windows free memory rose to 10 GB, the governor recalled one layer per
minute, and then stalled at 2-5 disk layers with Windows free hovering at 4.7-4.9 GB, right on
the cushion. Each recall reads 1.33 GB from the disk copy through the VM page cache and grows
the pinned set by 1.33 GB, so Windows free drops ~2.7 GB per recall while WSL returns memory to
Windows only slowly (`/proc/meminfo` showed 24 GB MemAvailable inside the VM with 4.9 GB free on
Windows). The next tick sees the RAM axis below the cushion and spills a layer again. With the
governor off, four manual recalls brought every layer home and full speed returned; Windows free
was 2.2 GB afterwards with all 48 layers pinned.

Fixes to make before the governor stays on by default:

1. Read the disk copy with O_DIRECT (recall and per-step gather) so recalls do not grow the
   page cache; drop the layer's pages after a spill/recall (`posix_fadvise DONTNEED`).
2. RAM-axis step-up hysteresis: require free RAM >= cushion + 2 rungs + margin, and hold the
   RAM axis for one interval after a recall so a recall can never trip its own cushion.
3. Consider `.wslconfig` `[experimental] autoMemoryReclaim=gradual` so WSL hands freed memory
   back to Windows on its own; `drop_caches` needs root and could not be tested from ssh.
4. Faster recall once memory is clearly free: more than one rung per minute, or a burst when
   free RAM exceeds the threshold by several rungs, since a single disk layer costs the CUDA
   graphs.
5. Windows launcher parameters for the three new dials (R3 finding 5), and `/v1/cache/residency`
   should wait like the chat routes instead of answering 503 during a rebuild.
