# WSL cache reclamation without expert rebuilds

The September 16 model download left about 60.3 GiB of its unused checkpoint in
Linux's page cache. Windows headroom stayed close to the governor's 4 GiB
cushion despite Linux reporting roughly 67 GiB available. The governor spilled
all 36 non-GPU expert layers and could not begin restoring them until Windows
headroom recovered. Recovery eventually started before the live cache probes;
it must not be attributed to those probes.

## Behaviour

`daemon/settings/memory_reclaim.py` adds a separate reclaim actuator to the
existing governor. It issues no cache-step, model restart or settings-write
requests. Emergency expert spills and the existing stable-headroom hold remain
independent of it.

- On WSL, probe when Windows headroom is below the normal RAM recall threshold
  plus 256 MiB, Linux has sufficient available memory, and inactive file cache
  exceeds a 512 MiB reserve plus a minimum probe.
- Start at 256 MiB. Wait five seconds after completion and compare Windows free
  RAM and inactive file cache with the pre-probe readings. Increase to 512 MiB
  and at most 1 GiB only when both readings improve materially. Measurements are
  observational and can include unrelated activity; they are not attribution.
- Ineffective probes back off for 10, 30, then 60 seconds and reset to 256 MiB.
- One worker subprocess at a time. An independent watchdog kills it after ten
  seconds, even when the placement governor is blocked on a rebuild. Shutdown
  prevents new workers and reaps the existing child. An uninterruptible child
  retains the slot until it exits.
- The integrated `/api/status` governor object includes `reclaim` with the state,
  retry delay, batch size, observed gains and last actuator result.
- `FREETOKEN_CACHE_RECLAIM=0` disables automatic probes. Disabling the memory
  governor also disables its reclaim controller.

## Actuators and limitations

First try `memory.reclaim` with `swappiness=0` on the current user's delegated
systemd cgroup, covering that user's services. Never retry with an unqualified
request that could swap anonymous memory, and never change swap limits or global
VM settings. Other users' and system services' cgroups are outside this scope.

The serving machine's `6.6.114.1-microsoft-standard-WSL2` kernel rejects this
option with EINVAL. Its fallback is therefore deliberately narrower: scan
sibling model directories of the roots returned by the live `/v1/models` API,
and advise away cached regions of inactive `.safetensors` files. Exclude active
roots, symlinks, hardlinks to active weights, and partial file names. This is not
general reclamation of arbitrary application caches on that older kernel.

The fallback uses mincore without reading/faulting the weight data, scans at
most 8 GiB of file address ranges per probe, rotates its cursor and counts
observed page eviction. It never deletes files. Cache advice does not guarantee
an immediate return of physical memory to Windows. The live server's model
files are protected by the fallback; the general kernel reclaim path uses the
kernel's own cache selection.

The built-in downloader also flushes and advises away each completed weight
shard before downloading the next. Errors in optional cache advice do not fail
a download. External `hf download` commands do not receive this hook, although
their completed files can be handled by the governor's fallback.

## Activation without restarting the existing model

The integrated worker starts with the settings helper. For an already-running
helper, this temporary watcher can activate reclamation without restarting it:

```bash
PYTHONPATH=python .venv/bin/python -m freetoken.daemon.settings.memory_reclaim --watch 2031 2020
```

It reads the existing helper's memory measurements and configured recovery
threshold. Once the helper exposes its integrated `reclaim` status, the watcher
closes its worker and exits. Run only one temporary watcher per server. Its
activity is logged; it does not inject status into an older helper.

## Validation

- 80 targeted settings/download/governor/reclaim tests pass, including the old
  kernel rejection, partial reclaim, active hardlink exclusions, adaptation,
  backoff, subprocess timeout without further governor ticks, shutdown, downloader
  integration and temporary-watcher handoff.
- The broader settings/daemon run had four failures (browse fallback, two
  checkpoint scan-count cases, and architecture metadata). All four reproduce
  on unchanged base commit `297789d`; they are not introduced by this change.
- Ruff passes for the new module and tests; existing files have unrelated lint
  findings. `git diff --check` passes.
- On the live RTX 5090 WSL machine, a staged worker detected EINVAL and used the
  inactive-file fallback. Two probes completed in 227 ms and 175 ms, evicting
  approximately 3.1 MiB and 1.1 MiB from mostly cold scanned ranges.
- A controlled 64 MiB scratch-file probe evicted all 64 MiB, preserved the file
  size, and finished in 113 ms. The model remained healthy, decoding, with the
  same instance ID before and after. The scratch file was removed afterward.

This validates the live actuator and runtime continuity. It does not establish
end-to-end speed recovery under a repeat 135 GB download, nor a full-rate cgroup
reclaim result on this kernel, which lacks the required option.
