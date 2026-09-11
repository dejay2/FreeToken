# Detecting and investigating a stopped scheduler

`/health` and `/v1/cache/status` expose an `inference` object with active requests,
the last phase, seconds since progress, completed prefill tokens, a deadline and a
latched `stuck` verdict. Completed prefill chunks and sampled tokens advance the
clock. Idle time and maintenance time do not consume an active request's deadline.
Control replies and cache-parking updates do not count as inference progress.

This detects **whole-scheduler silence**: 600 seconds with active requests and no
completed chunk or token. A healthy request can mask a separate lost request; this
is not a per-request timeout. Long prompts can run indefinitely while their chunks
complete. Failed initial dispatches are removed from active accounting; disconnected
requests remain active until the backend acknowledges their abort.

A runtime scheduler exception is reported through the supervisor queue even while
the process is still alive. The scheduler flushes the error for at most one second,
then exits without CUDA synchronization or Python tensor destruction. Keyboard
interrupts retain graceful shutdown. This improves failure handling; it does not
identify or repair the kernel that caused an asynchronous CUDA exception.

The settings helper captures one incident on the first failed probe in an outage,
before its existing restart policy runs. Bundles default to `logs/incidents/` beside
the server log; set `FREETOKEN_INCIDENT_DIR` on the helper to override this. The most
recent path is available as `autoRestart.last_incident` in the helper's status.
Five bundles are retained. Directories are created with mode 0700 and files with
0600 on POSIX. The bundle contains readiness, a 128 KiB log tail, GPU status,
process wait states, RAM use, and best-effort kernel logs/native stacks. On WSL it
also queries recent Windows NVIDIA events. Diagnostic commands
have time and output limits and share an eight-second budget. A separate process
bounds the entire capture to ten seconds, including filesystem access. Missing tools,
ptrace restrictions and capture failures do not prevent recovery. The server log
can contain user data: inspect locally and review before sharing any bundle.

The helper still respects Stop, loading and lifecycle jobs, with at most three
automatic restart attempts per hour. Running processes receive six failed probes
instead of three (ten seconds between probes). A stalled inference therefore
normally takes about eleven minutes to trigger recovery; explicit worker failures
are detected sooner. These limits avoid treating long cold work as a crash.

For an illegal GPU address, the Python traceback can point to the next CUDA
synchronization instead of the faulty kernel. Enable NVIDIA exception diagnostics
for a controlled reproduction and inspect the faulting kernel/PC before changing
engine behavior. See [NVIDIA GPU core dump documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-core-dump-support).
GPU core dumps have separate retention and privacy requirements from these small
helper bundles. Do not attach another CUDA debugger or terminate the process while
a dump is being generated.

Two failure modes found during the September 2026 investigation have targeted
regression coverage. PLE's bounded token-index cache could evict warmup tensors
still referenced by a CUDA graph after enough distinct prefill shapes. Captured
indices now retain tensor ownership for the model's lifetime; uncaptured shapes
remain bounded. The GPU test churns the cache, checks ownership before replay,
and compares replay results with the reference hash for decode and verification
shapes. The captured live fault was a context read in `_ple_row_ids_kernel`;
the cache-lifetime defect was reproduced separately on the GPU.

RAM/SSD parking can also temporarily lock a shared prefix while detached leaves
are copied. When those copies finish, allocation now retries eviction of the
newly unlocked prefix before asserting that space is insufficient. A deterministic
forked-prefix test checks this sequence and verifies page conservation. Keep the
incident monitor enabled: these fixes do not establish that every possible cause
of scheduler silence has been eliminated.
