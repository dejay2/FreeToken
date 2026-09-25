# FreeToken Sleep (phase 1): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A **Sleep** button (and `POST /v1/sleep`). It gives the graphics card back for a game
while the model stays in PC memory. The next chat, or **Wake**, brings it back in about half a
minute, against 149-161 s for a cold load.

**Architecture:**
- Sleep happens in-process, at the scheduler's idle safe point, and composes primitives the memory
  governor already runs live. Owned expert layers go to the SSD expert copy (`Engine._move_layer`).
  A new `OffloadMoeCache.release_slots()` frees the slot cache. KV shrinks to 1 page after its
  conversations are parked, GDN shrinks to its padding slot, and the CUDA graphs and MTP head are
  dropped. The dense weights stay on the card; that is phase 1.
- Wake reverses it in a safe order and ends with a two-prefill self-check. A card that is too
  busy (a game) refuses **before anything is allocated**. A wake that fails midway goes back to
  sleep and answers "rejected".
- A new message quartet `CacheSleep*` travels the existing control path. The API keeps an
  `asleep` flag, and its public state word is `"sleeping"`. A chat to a sleeping model wakes it.
- Helper: `/api/server/sleep|wake`. The watchdog treats `sleeping` as alive. While asleep the
  governor only spills to the SSD. The panel gets Sleep and Wake buttons. `freetoken.sh` adopts a
  sleeping server. **No llama-swap patch:** to it a sleeping FreeToken stays "ready", and any
  other model's load unloads it fully (Jay's rule).

**Tech Stack:** Python 3.13 (torch, FastAPI, pytest), bash adapters, plain browser JavaScript
(node for tests), systemd --user in WSL `vllm` on the serving box.

**Spec:** `docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md`. Phase 2 (dense weights to an
SSD sleep file) is **not** in this plan; see spec section 4.

**Provenance (2026-09-25, devbox):** before this plan was committed, its diffs were applied with
`patch -p1` in task order, and its new files copied, onto a clean copy of `fedf802`.
Every new test passed there (`49 passed, 2 skipped`; the two skips are the CUDA tests). The
full devbox suite (`tests/scheduler tests/server tests/engine tests/moe tests/tokenizer
tests/settings tests/daemon tests/engines`) had **the same 136 failures as `fedf802`, and no new
ones**. The box-only parts are still unproven: the GPU tests (Task 12), the real `SpecDraftHead`
rebuild, and live timing (Task 13).

## Global Constraints

- **Worktree:** `/home/jay/projects/FreeToken/.worktrees/feat-freetoken-sleep`, branch
  `feat/freetoken-sleep` (from `mtp-upstream-merge` at `fedf802`). The PR targets
  `mtp-upstream-merge`. Push only to `origin` (`dejay2/FreeToken`). `upstream` is fetch-only.
  Never force-push.
- **Implementers make no git writes**: no add, commit, checkout, switch, reset, clean, stash, rm,
  mv or push. The controller commits after each task's review, with the message given in that
  task. Every commit message is `type: subject` and ends with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  (2026-09-23: helpers ran reset/clean and wiped untracked files; memory `feedback-subagent-git-safety`.)
- **Leave the untracked `.lavish/` and `docs/research/ninfer-qwen-flash-nvfp4-2026-09-14.md` alone.**
- **Test command** (from the worktree root; the worktree has its own `.venv`):
  `PYTHONPATH=python .venv/bin/python -m pytest <paths> -q -p no:cacheprovider`.
- **The devbox suites already fail in places.** At `fedf802`, `tests/scheduler tests/server
  tests/engine tests/moe` had 123 failures and `tests/settings tests/daemon tests/engines` had 13,
  none of them about sleep. Task 0 saves the failure list. **"Green" means no failure outside
  that list**, never a total count.
- **GPU-dependent tests skip on the devbox** (no GPU). The controller runs them on the box (Task 12).
- **The box:** the Windows 11 PC with an RTX 5090, WSL distro `vllm`, checkout `/home/jay/FreeToken`. The devbox reaches it
  with `ssh 5090` plus a stdin script into `wsl -d vllm -e bash -l` (memory
  `reference-wsl-ssh-route`). Model server on `127.0.0.1:2020`, helper on `:2031`, llama-swap on
  `:2040`.
- **Box rules (Jay granted server up/down freedom for this job):**
  - Before any box action, run `curl -s 127.0.0.1:2040/running` and the helper's `/api/status`.
    **Never evict a model Jay is using** (a request in flight, or a chat in the last 10 minutes
    in `/v1/requests`). Wait, or ask.
  - **At most 3 FreeToken boots per live session.** Announce each one before it starts. Watch
    Windows free RAM through the boot:
    `powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"`.
    Stop and report if it stays under 2 GB for more than 60 s.
  - Leave the box as found: same model loaded (or unloaded), same branch
    (`mtp-upstream-merge` after the merge).
- **Comments carry measurements** (repo convention): when code encodes a number (a margin, a
  timeout), the comment says where the number came from.
- **Plain words for Jay** on the page: labels and messages in everyday language. Technical
  detail goes in tooltips and logs.
- **Tests mirror subsystems**, not models (CLAUDE.md): engine tests in `tests/engine`,
  scheduler tests in `tests/scheduler`, and so on.
- **Codex reviews:** use the `codex-review` skill with model **Astra** at thinking **xhigh**
  (the `codex-models` skill has the slug). **Two rounds at most.** List every finding at once,
  and rule on each one against the card, not the reviewer's bar (memory
  `feedback-review-rounds-cap`).

## Review Focus

These are the five failure modes most likely to hurt Jay. Each is pinned by named tests; a
reviewer checks those tests exist and fail without the fix.

1. **Waking while a game holds the card.** The refusal must come before anything is allocated.
   The model stays asleep, and a chat gets a plain 503 that tells Jay to close the game. The
   server must never crash or latch failed. Pinned by
   `test_wake_is_refused_before_touching_anything_while_a_game_holds_the_card` (Task 3),
   `test_a_refused_auto_wake_answers_the_held_chats_in_plain_words` (Task 5), and
   `test_a_chat_while_a_game_holds_the_card_gets_a_plain_refusal` (Task 6).
2. **A wake that fails partway** (an out-of-memory error after allocation began, because a game grabbed the card between the check and the
   allocation). It must go back to a consistent asleep state and reply "rejected". Only if that
   also fails may it latch "failed" and let the watchdog restart. Pinned by
   `test_a_wake_that_fails_midway_goes_back_to_sleep_and_is_refused` and
   `test_a_wake_that_cannot_go_back_to_sleep_raises_wake_failed` (Task 3), and
   `test_a_sleep_that_fails_after_teardown_latches_failed` (Task 5).
3. **A chat that reaches a sleeping scheduler** (it passed the API gate before the sleep reply
   landed; tokenizer workers do not preserve order). It must never be admitted against the
   1-page pool, because `_admit_user_msg`'s clip would drop it as too long. It is held, one
   auto-wake runs, and it is admitted after. An abort removes it. Pinned by
   `test_a_chat_that_reaches_a_sleeping_scheduler_is_held_and_wakes_it` and
   `test_an_abort_removes_a_held_chat` (Task 5).
4. **The residency round trip.** Owned and RAM-parked layers come back byte-identical.
   `_ram_parked_layers` and `_ram_spilled_layers` must be right, prefill overlap must resume,
   and graphs must be recaptured with the boot sizes. A layer the governor spilled *while
   asleep* stays on the SSD, with graphs deferred, exactly as a live spill does today. Pinned by
   `test_wake_restores_the_geometry_and_the_owned_banks_byte_for_byte`,
   `test_ram_parked_layers_are_still_parked_after_a_wake`,
   `test_prefill_overlap_is_off_asleep_and_back_awake` and
   `test_a_ram_squeeze_while_asleep_spills_to_the_ssd_and_the_layer_stays_there` (Task 3).
   The live greedy-equality bench covers it too (Task 13).
5. **The helper and switcher read "sleeping" correctly.** The watchdog never restarts a sleeping
   server. The governor never steps the card or recalls a layer while asleep. `freetoken.sh`
   adopts its own sleeping server instead of rebooting it. `ninfer.sh` fully stops a sleeping
   FreeToken before loading NInfer (Jay's rule). Pinned by
   `test_the_watchdog_treats_a_sleeping_server_as_alive`,
   `test_asleep_the_governor_never_recalls_even_with_ram_to_spare` (Task 7),
   `test_freetoken_adopts_its_own_sleeping_server_without_rebooting` and
   `test_ninfer_fully_stops_a_sleeping_freetoken_first` (Task 8).

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `python/freetoken/moe/offload_cache.py` | modify | `release_slots()`: free the slot cache to size 0 |
| `python/freetoken/engine/engine.py` | modify | extract `_graph_bs_for_recapture` / `_recapture_graphs` (pure refactor); `sleep_snapshot`, `sleep_preflight`, `sleep`, `wake`, `asleep_rebuild` |
| `python/freetoken/engine/sleep.py` | new | `SleepSnapshot`, `check_can_sleep`, `sleep_engine`, `release_to_sleep`, `wake_engine`, `asleep_rebuild`, `SleepRefused`, `WakeFailed` |
| `python/freetoken/message/{backend,tokenizer,frontend,__init__}.py` | modify | `CacheSleepBackendMsg`, `CacheSleepMsg`, `CacheSleepResultMsg`, `CacheSleepReply` |
| `python/freetoken/tokenizer/server.py` | modify | passthrough both ways |
| `python/freetoken/scheduler/scheduler.py` | modify | queue/execute sleep and wake; held chats and auto-wake; idle work paused while asleep; asleep-only governor step |
| `python/freetoken/server/api_server.py` | modify | `asleep` flag, `ensure_awake`, `dispatch_sleep`, `/v1/sleep`, `/v1/wake`, status word |
| `python/freetoken/server/control_api.py` | modify | `public_state`; `/health` and `/ready` know "sleeping" |
| `python/freetoken/daemon/settings/process_manager.py` | modify | `sleep_server(action)` |
| `python/freetoken/daemon/settings/app.py` | modify | `/api/server/sleep|wake`; panel wiring |
| `python/freetoken/daemon/settings/watchdog.py` | modify | `sleeping` is alive |
| `python/freetoken/daemon/settings/governor.py` | modify | `_tick_asleep`: RAM-down only |
| `python/freetoken/daemon/settings/memory_reclaim.py` | modify | reclaim runs while sleeping |
| `python/freetoken/daemon/settings/panel.py` | modify | `sleep` field on rows, `sleep_model`, routes |
| `python/freetoken/daemon/settings/static/panel.js`, `index.html` | modify | Sleep / Wake buttons, "Asleep (graphics card free)" |
| `engines/adapters/freetoken.sh` | modify | adopt a sleeping server |
| `scripts/bench/sleep_bench.py` | new | live acceptance bench (stdlib) |
| `tests/moe/test_offload_release_slots.py` | new | Task 1 |
| `tests/engine/test_graph_recapture_helpers.py` | new | Task 2 |
| `tests/engine/test_engine_sleep.py` | new | Task 3 (CPU harness) |
| `tests/server/test_sleep_messages.py` | new | Task 4 |
| `tests/scheduler/test_sleep_scheduler.py` | new | Task 5 |
| `tests/server/test_sleep_api.py` | new | Task 6 |
| `tests/settings/test_sleep_helper.py` | new | Task 7 |
| `tests/engines/test_adapters_sleep.py` | new | Task 8 |
| `tests/settings/test_panel_sleep.py` | new | Task 9 |
| `tests/engine/test_sleep_gpu.py` | new | Task 11 (box) |
| `README.md`, `CONTEXT.md`, `engines/adapters/CONTRACT.md` | modify | Task 10 |
| `docs/research/freetoken-sleep-acceptance-2026-09-XX.md` | new | Task 13 (live results) |

Task order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15. Tasks 7-9 depend only
on the wire contract (Task 6) and may run in parallel once Task 6 is committed.

---

### Task 0: Baseline (controller only)

- [ ] **Step 1: Commit the spec and this plan on the branch**

```bash
cd /home/jay/projects/FreeToken/.worktrees/feat-freetoken-sleep
git status --short   # expect only the two docs (plus the untracked .lavish/ in the main checkout)
git add docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md docs/superpowers/plans/2026-09-25-freetoken-sleep.md
git commit -m "docs: design and plan for FreeToken sleep (free the card, keep the model in RAM)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Save the devbox failure baseline**

```bash
cd /home/jay/projects/FreeToken/.worktrees/feat-freetoken-sleep
mkdir -p /tmp/claude-1000/sleep-baseline
PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler tests/server tests/engine tests/moe tests/tokenizer \
  tests/settings tests/daemon tests/engines -q -p no:cacheprovider 2>&1 | grep '^FAILED' | sort \
  > /tmp/claude-1000/sleep-baseline/failed.txt
wc -l /tmp/claude-1000/sleep-baseline/failed.txt   # about 136 at fedf802
```

Every later "run the suite" step compares with
`grep '^FAILED' | sort | comm -13 /tmp/claude-1000/sleep-baseline/failed.txt -`. The expected
output is empty.

---

### Task 1: `OffloadMoeCache.release_slots()`

**Files:**
- Modify: `python/freetoken/moe/offload_cache.py` (new method before `set_alphas`)
- Test: `tests/moe/test_offload_release_slots.py`

**Interfaces:**
- Produces: `OffloadMoeCache.release_slots() -> int`, which returns the bytes freed. Afterwards
  `cache_size == 0`, `bank_caches == {}`, `banks == []`, fused copy descriptors cleared,
  `slot_for_id` all -1, and prefill-overlap views torn down. The `prefill_overlap` flag is left
  as it was, so `rebuild()` re-inits the buffers. `bank_sources` and `resident_banks` are
  untouched. `rebuild(n)` afterwards gives a cold cache of size n.

- [ ] **Step 1: Write the failing tests**

`tests/moe/test_offload_release_slots.py`:

```python
"""OffloadMoeCache.release_slots: sleep frees the whole slot cache; rebuild brings it back cold."""

from __future__ import annotations

import torch

from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache

E = 4  # experts per layer
ROW = 8 * 8 * 4  # one float32 [8, 8] row per bank


def _cache(cache_size: int = 16, owned=(1,), overlap: bool = False) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=4, num_experts=E, cache_size=cache_size, device=torch.device("cpu"),
        prefill_overlap=overlap,
    )
    sources = {
        "gate_up": [torch.randn(E, 8, 8) for _ in range(4)],
        "down": [torch.randn(E, 8, 8) for _ in range(4)],
    }
    residency = [
        HostResidency.GPU_OWNED.value if i in owned else HostResidency.PINNED.value for i in range(4)
    ]
    cache.set_bank_sources(sources, layer_residency=residency, gpu_owned_layers=frozenset(owned))
    return cache


def test_release_frees_every_slot_row_and_keeps_the_banks():
    cache = _cache()
    sources = {n: list(v) for n, v in cache.bank_sources.items()}
    owned = cache.resident_banks[1]
    freed = cache.release_slots()
    assert freed == 16 * ROW * 2  # two banks
    assert cache.cache_size == 0 and cache.bank_caches == {} and cache.banks == []
    assert bool((cache.slot_for_id == -1).all())
    assert cache.id_of_slot.numel() == 0 and cache.usage.numel() == 0
    assert not cache._copy_fused_ok
    for name, per_layer in sources.items():
        assert all(a is b for a, b in zip(per_layer, cache.bank_sources[name]))
    assert cache.resident_banks[1] is owned


def test_rebuild_after_release_is_a_cold_cache_of_the_old_size():
    cache = _cache()
    cache.release_slots()
    cache.rebuild(16)
    assert cache.cache_size == 16
    assert cache.bank_caches["gate_up"].shape == (16, 8, 8)
    assert bool((cache.id_of_slot == -1).all()) and int(cache.usage.sum()) == 0


def test_release_tears_down_prefill_overlap_and_rebuild_restores_it():
    cache = _cache(cache_size=2 * E, owned=(), overlap=True)
    assert cache.prefill_bank_buffers
    cache.release_slots()
    assert cache.prefill_bank_buffers == []
    cache.rebuild(2 * E)
    assert cache.prefill_overlap and len(cache.prefill_bank_buffers) == 2


def test_a_second_release_frees_nothing():
    cache = _cache()
    cache.release_slots()
    assert cache.release_slots() == 0
```

- [ ] **Step 2: Run them to see them fail**

`PYTHONPATH=python .venv/bin/python -m pytest tests/moe/test_offload_release_slots.py -q -p no:cacheprovider`
Expected: 4 failed, `AttributeError: 'OffloadMoeCache' object has no attribute 'release_slots'`.

- [ ] **Step 3: Implement**

```diff
--- a/python/freetoken/moe/offload_cache.py
+++ b/python/freetoken/moe/offload_cache.py
@@ -811,6 +811,31 @@
         if self.prefill_overlap:
             self._init_prefill_overlap_buffers()
 
+    def release_slots(self) -> int:
+        """Sleep: free the GPU slot cache and return the bytes it held.
+
+        Every streaming expert copy on the card goes; the host banks (``bank_sources``), the
+        GPU-owned layers' ``resident_banks`` and the per-layer bookkeeping stay, so
+        :meth:`rebuild` brings the cache back cold at any size. ``rebuild``'s own floor
+        (``num_experts`` slots, 1.32 GiB for Qwen3.8; 2x that with prefill overlap) is why
+        sleep needs this instead. Between forwards only, like ``rebuild``
+        (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md section 3.2).
+        """
+        freed = sum(int(t.numel()) * t.element_size() for t in self.bank_caches.values())
+        self._teardown_prefill_overlap()  # its views alias the slot cache
+        self.banks = []
+        self.bank_caches = {}
+        self.cache_size = 0
+        self._build_fused_copy_plan()  # no banks: resets the descriptors and returns
+        self.slot_for_id.fill_(-1)
+        self.id_of_slot = torch.empty((0,), dtype=torch.int32, device=self.device)
+        self.usage = torch.empty((0,), dtype=torch.int64, device=self.device)
+        self._reset_prefetch()
+        if self.device.type == "cuda":
+            torch.cuda.synchronize(self.device)
+            torch.cuda.empty_cache()
+        return freed
+
     def set_alphas(
         self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
     ) -> None:
```

- [ ] **Step 4: Run the tests and the neighbours**

`PYTHONPATH=python .venv/bin/python -m pytest tests/moe/test_offload_release_slots.py tests/moe/test_disk_banks.py tests/engine/test_memory_step.py -q -p no:cacheprovider`
Expected: `56 passed, 1 skipped`, or thereabouts; no failures.

- [ ] **Step 5: Controller commit**

`git add python/freetoken/moe/offload_cache.py tests/moe/test_offload_release_slots.py && git commit -m "feat(moe): release the whole slot cache for sleep" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"`

---

### Task 2: Extract the graph re-capture from `rebuild_runtime_cache` (pure refactor)

Wake must re-capture graphs exactly the way a rebuild does, including the deferral while a layer
is on the SSD. The block moves into two methods. `rebuild_runtime_cache` behaviour is unchanged.

**Files:**
- Modify: `python/freetoken/engine/engine.py` (`rebuild_runtime_cache` step 0 and step 4; two new methods before `_rearm_spec_graphs`)
- Test: `tests/engine/test_graph_recapture_helpers.py`

**Interfaces:**
- Produces: `Engine._graph_bs_for_recapture() -> list[int]` (the boot-resolved sizes; records
  `_deferred_graph_bs`) and `Engine._recapture_graphs(config, prior_graph_bs, free_min) -> None`
  (a new `GraphRunner`; sets `_graphs_deferred`).

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_graph_recapture_helpers.py`:

```python
"""The two helpers extracted from rebuild_runtime_cache step 0 and step 4 (shared with wake)."""

from __future__ import annotations

from types import SimpleNamespace

import freetoken.engine.engine as engine_mod
from freetoken.engine.engine import Engine


def test_graph_bs_for_recapture_reads_the_live_runner_then_the_parked_set():
    eng = SimpleNamespace(_graphs_deferred=None, graph_runner=SimpleNamespace(graph_bs_list=[1, 2]))
    assert Engine._graph_bs_for_recapture(eng) == [1, 2] and eng._deferred_graph_bs == [1, 2]
    eng._graphs_deferred, eng.graph_runner = "deferred until no disk layers", SimpleNamespace(graph_bs_list=[])
    assert Engine._graph_bs_for_recapture(eng) == [1, 2]
    no_graphs = SimpleNamespace(_graphs_deferred=None, graph_runner=SimpleNamespace(graph_bs_list=[]))
    assert Engine._graph_bs_for_recapture(no_graphs) == []  # a graphs-off boot stays off


def test_recapture_defers_while_a_layer_is_on_the_ssd(monkeypatch):
    built = []
    monkeypatch.setattr(engine_mod, "GraphRunner", lambda **kw: built.append(kw) or SimpleNamespace(**kw))
    config = SimpleNamespace(page_size=64, cuda_graph_max_bs=4,
                             model_config=SimpleNamespace(vocab_size=10, model_is_mrope=False))
    eng = SimpleNamespace(max_seq_len=4096, stream=None, device=None, model=None, attn_backend=None,
                          dummy_req=None, moe_offload_cache=SimpleNamespace(has_disk_layers=True))
    Engine._recapture_graphs(eng, config, [1, 2], free_min=5)
    assert built[-1]["cuda_graph_bs"] == [] and eng._graphs_deferred == "deferred until no disk layers"
    eng.moe_offload_cache.has_disk_layers = False
    Engine._recapture_graphs(eng, config, [1, 2], free_min=5)
    assert built[-1]["cuda_graph_bs"] == [1, 2] and eng._graphs_deferred is None
    assert built[-1]["max_seq_len"] == 4096 and built[-1]["free_memory"] == 5
```

- [ ] **Step 2: Run to see them fail** (`AttributeError ... _graph_bs_for_recapture`).

- [ ] **Step 3: Implement.** Move the two blocks verbatim. The moved comment keeps its
  reasoning, and nothing else changes.

```diff
--- a/python/freetoken/engine/engine.py
+++ b/python/freetoken/engine/engine.py
@@ -2373,20 +2373,7 @@
 
         torch.cuda.synchronize(self.device)
         self._report_maintenance_progress("rebuild:validated")
-        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
-        # off free memory, which is far smaller now that the caches are resident (post-cache
-        # free << startup pre-load free), so re-deriving it here would silently drop large
-        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
-        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
-        # While DISK layers exist the live runner was built with no graphs (below), so the
-        # boot-resolved set is parked in _deferred_graph_bs and read back from there. Keyed on
-        # the deferral marker, not on list truthiness: a boot that disabled graphs has an
-        # empty list that must stay empty (None would re-derive the auto set).
-        if getattr(self, "_graphs_deferred", None):
-            prior_graph_bs = self._deferred_graph_bs
-        else:
-            prior_graph_bs = self.graph_runner.graph_bs_list
-            self._deferred_graph_bs = list(prior_graph_bs)
+        prior_graph_bs = self._graph_bs_for_recapture()
         # Point of no return for the scheduler's rollback logic: from here the live graphs and
         # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
         # untouched (no rollback needed); after it, only a rebuild restores service.
@@ -2419,12 +2406,43 @@
         self._report_maintenance_progress("rebuild:pools")
         # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
         self._refresh_seq_state(config)
-        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
         # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
         #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
         gc.collect()
         free_min = self._sync_get_memory()[0]
         self._report_maintenance_progress("rebuild:capture")
+        self._recapture_graphs(config, prior_graph_bs, free_min)
+        # Re-arm the speculative widths on BOTH branches. Until 2026-09-08 this sat inside the
+        # capture branch only, so the spill that deferred the graphs dropped the runner and the
+        # recall found nothing to re-arm: with MTP on, every later decode step ran eager
+        # (measured 20 tok/s against 50 before the spill, RTX 5090, depth 3). Arming while the
+        # graphs are deferred is harmless: _capture_decode_graph checks can_use_cuda_graph first.
+        self._rearm_spec_graphs()
+        self._report_maintenance_progress("rebuild:captured")
+
+    def _graph_bs_for_recapture(self) -> list[int]:
+        """The CUDA-graph batch sizes resolved at boot, for a re-capture after a teardown.
+
+        The auto heuristic keys off free memory, which is far smaller once the caches are
+        resident (post-cache free << startup pre-load free), so re-deriving it after a rebuild
+        would silently drop large batch sizes. While DISK layers exist the live runner was built
+        with no graphs, so the boot-resolved set is parked in ``_deferred_graph_bs`` and read
+        back from there. Keyed on the deferral marker, not on list truthiness: a boot that
+        disabled graphs has an empty list that must stay empty (None would re-derive the auto
+        set). Shared by rebuild_runtime_cache and sleep (engine/sleep.py).
+        """
+        if getattr(self, "_graphs_deferred", None):
+            return self._deferred_graph_bs
+        prior = self.graph_runner.graph_bs_list
+        self._deferred_graph_bs = list(prior)
+        return prior
+
+    def _recapture_graphs(self, config, prior_graph_bs, free_min: int) -> None:
+        """Build a fresh GraphRunner against the current tensors: no graphs while any MoE layer
+        is on the SSD (a disk layer forces eager decode), else the boot-resolved sizes.
+        Extracted unchanged from rebuild_runtime_cache step 4 so wake (engine/sleep.py) takes
+        the identical path."""
+        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
         has_disk = self.moe_offload_cache is not None and getattr(self.moe_offload_cache, "has_disk_layers", False)
         if has_disk:
             self.graph_runner = GraphRunner(
@@ -2450,7 +2468,7 @@
                 device=self.device,
                 model=self.model,
                 attn_backend=self.attn_backend,
-                cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set (see above)
+                cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set
                 cuda_graph_max_bs=config.cuda_graph_max_bs,
                 free_memory=free_min,
                 max_seq_len=aligned_max_seq_len,
@@ -2459,13 +2477,6 @@
                 moe_offload_cache=self.moe_offload_cache,
                 mrope=config.model_config.model_is_mrope,
             )
-        # Re-arm the speculative widths on BOTH branches. Until 2026-09-08 this sat inside the
-        # capture branch only, so the spill that deferred the graphs dropped the runner and the
-        # recall found nothing to re-arm: with MTP on, every later decode step ran eager
-        # (measured 20 tok/s against 50 before the spill, RTX 5090, depth 3). Arming while the
-        # graphs are deferred is harmless: _capture_decode_graph checks can_use_cuda_graph first.
-        self._rearm_spec_graphs()
-        self._report_maintenance_progress("rebuild:captured")
 
     def _rearm_spec_graphs(self) -> None:
         """Re-arm (not re-capture) the speculative verify widths armed at boot.
```

- [ ] **Step 4: Run the tests plus every rebuild test**

`PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_graph_recapture_helpers.py tests/engine/test_memory_step.py tests/engine/test_spec_rearm_after_rebuild.py tests/moe/test_disk_banks.py tests/scheduler/test_cache_rebuild.py tests/engine/test_kv_dynamic_engine.py -q -p no:cacheprovider`
Expected: all pass (at `fedf802` plus this change: `64 passed, 1 skipped` for the first five files).

- [ ] **Step 5: Controller commit** — `refactor(engine): share the graph re-capture between rebuild and wake`

---

### Task 3: `engine/sleep.py` and the Engine methods

**Files:**
- Create: `python/freetoken/engine/sleep.py`
- Modify: `python/freetoken/engine/engine.py` (class default `sleep_snapshot` and four thin methods after `__init__`)
- Test: `tests/engine/test_engine_sleep.py`

**Interfaces:**
- Consumes: `Engine._move_layer`, `OffloadMoeCache.release_slots/rebuild/is_gpu_owned_layer`,
  `Engine._resize_kv_pool`, `LinearStatePool.rebuild`, `SpecStateLadder.rebind`,
  `Engine._refresh_seq_state`, `Engine._graph_bs_for_recapture/_recapture_graphs` (Task 2),
  `Engine._rearm_spec_graphs`, `Engine._capture_spec_graphs_at_boot`, `Engine._warmup_prefill`,
  `Engine.snapshot_pool_budget`, `Engine._sync_get_memory`, `Engine._report_maintenance_progress`.
- Produces:
  - `Engine.sleep_snapshot: SleepSnapshot | None`. None means awake; this is the one source of truth.
  - `Engine.sleep_preflight()` raises `SleepRefused` with nothing freed.
  - `Engine.sleep() -> dict` and `Engine.wake() -> dict`. Each returns
    `{"asleep", "released_bytes", "vram_free_bytes", "elapsed_s", "note"}`, where `note` is
    `"already asleep"` / `"already awake"` for a no-op and None otherwise.
  - `Engine.asleep_rebuild(**rebuild_kwargs)`: the `rebuild` callable `step_memory` uses while
    asleep. It only spills to the SSD.
  - `sleep.SleepRefused(CacheRebuildRejected)`: the engine is in a known state.
    `sleep.WakeFailed(RuntimeError)`: the state is unknown. `sleep.WAKE_MARGIN_BYTES = 512 MiB`.

- [ ] **Step 1: Write the failing tests.** The harness reuses `FakeDiskEngine` from
  `tests/moe/test_disk_banks.py`, which gives a real `OffloadMoeCache`, a real `ExpertDiskCopy`
  and the real `_move_layer`.

`tests/engine/test_engine_sleep.py`:

```python
"""Engine sleep and wake on the CPU harness (review focus 1, 2 and 4).

A real OffloadMoeCache, a real ExpertDiskCopy and the real Engine._move_layer: the only fakes
are the KV / GDN pools, the graph runner and the memory probe, which reports "free VRAM" from
what the harness holds on the "card" so released/needed byte counts are exact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.engine.sleep import WAKE_MARGIN_BYTES, SleepRefused, WakeFailed
from tests.moe.test_disk_banks import FakeDiskEngine

EXPERT = (32 * 16 + 16 * 32) * 2  # bf16 bytes of one expert row across both banks
LAYER = 64 * EXPERT
PAGE, SLOT = 100, 1000  # "bytes" the harness charges per KV page and per GDN slot


class FakeKV:
    needs_rebind_on_rebuild = False

    def __init__(self):
        self.rebuilds: list[int] = []

    def rebuild_from_config(self, config, num_pages, *, num_swa_pages=None):
        self.rebuilds.append(num_pages)

    def attach_page_table(self, table):
        self.table = table


class FakeLinear:
    def __init__(self, n):
        self._n, self.rebuilds = n, []

    @property
    def num_slots(self):
        return self._n

    def rebuild(self, n):
        self.rebuilds.append(n)
        self._n = n


class FakeGraphs:
    def __init__(self, bs=(1,)):
        self.graph_bs_list, self.destroyed = list(bs), 0

    def destroy_cuda_graphs(self):
        self.destroyed += 1


class SleepEngine(FakeDiskEngine):
    TOTAL = 1 << 30

    def __init__(self, tmp_path, *, owned=(0, 2), cache_size=256, overlap=False):
        super().__init__(num_layers=4, num_experts=64, cache_size=cache_size, owned_layers=owned,
                         model_dir=tmp_path / "model", disk_dir=tmp_path / "disk", overlap=overlap)
        for lid in range(4):
            self.expert_disk_copy.write_layer(
                lid, {n: self.moe_offload_cache.bank_sources[n][lid] for n in self.bank_schema})
        self.original = {lid: {n: self.moe_offload_cache.bank_sources[n][lid].clone()
                               for n in self.bank_schema} for lid in range(4)}
        self.config.spec_decode = SimpleNamespace(enabled=False, graph_widths=())
        self.kv_cache = FakeKV()
        self._pool_cls = SimpleNamespace(min_kv_tokens=lambda config: config.page_size)
        self.linear_state_pool = FakeLinear(9)
        self.spec_state_ladder = None
        self.graph_runner = FakeGraphs()
        self.attn_backend = SimpleNamespace(reset_capture=lambda: None)
        self.spec_graph_runner = self.spec_draft = self.mtp_shadow_observer = None
        self._spec_graph_widths = ()
        self._graphs_deferred = None
        self._ram_parked_layers = []
        self.model = SimpleNamespace(mark_for_rebind=lambda: None)
        self.max_seq_len = 1024
        self.page_table = torch.zeros((5, 1024), dtype=torch.int32)
        self.ctx = SimpleNamespace(page_table=self.page_table)
        self.dummy_req = SimpleNamespace(table_idx=4)
        self.game_bytes = 0
        self.recaptures: list = []
        self.warmups = 0
        self.spec_captures = 0
        self.budget_snapshots = 0
        self.sleep_snapshot = None

    def _sync_get_memory(self):
        cache = self.moe_offload_cache
        used = (cache.cache_size * EXPERT + len(cache.gpu_owned_layer_ids) * LAYER
                + self.num_pages * PAGE + self.linear_state_pool.num_slots * SLOT)
        free = self.TOTAL - used - self.game_bytes
        return free, free

    def _recapture_graphs(self, config, graph_bs, free_min):
        disk = self.moe_offload_cache.has_disk_layers
        self.recaptures.append((list(graph_bs), disk))
        self.graph_runner = FakeGraphs(() if disk else graph_bs)
        self._graphs_deferred = "deferred until no disk layers" if disk else None

    def _warmup_prefill(self, **kwargs):
        self.warmups += 1

    def _capture_spec_graphs_at_boot(self):
        self.spec_captures += 1

    def snapshot_pool_budget(self):
        self.budget_snapshots += 1

    sleep_preflight = Engine.sleep_preflight
    sleep = Engine.sleep
    wake = Engine.wake
    asleep_rebuild = Engine.asleep_rebuild
    _resize_kv_pool = Engine._resize_kv_pool
    _refresh_seq_state = Engine._refresh_seq_state
    _graph_bs_for_recapture = Engine._graph_bs_for_recapture
    _rearm_spec_graphs = Engine._rearm_spec_graphs


def residency(eng):
    return list(eng.moe_offload_cache.layer_residency)


def test_sleep_releases_the_card_and_keeps_the_host_banks(tmp_path):
    eng = SleepEngine(tmp_path)
    pinned = {lid: eng.moe_offload_cache.bank_sources["gate_up"][lid] for lid in (1, 3)}
    rep = eng.sleep()
    cache = eng.moe_offload_cache
    assert rep["asleep"] is True and rep["note"] is None
    assert rep["released_bytes"] == 256 * EXPERT + 2 * LAYER + 99 * PAGE + 8 * SLOT
    assert cache.cache_size == 0 and cache.bank_caches == {}
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"]
    assert eng.num_pages == 1 and eng.kv_cache.rebuilds == [1]
    assert eng.linear_state_pool.num_slots == 1
    assert eng.graph_runner.destroyed == 1
    assert eng.max_seq_len == 16  # one 16-token page
    for lid, bank in pinned.items():
        assert eng.moe_offload_cache.bank_sources["gate_up"][lid] is bank


def test_wake_restores_the_geometry_and_the_owned_banks_byte_for_byte(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    rep = eng.wake()
    cache = eng.moe_offload_cache
    assert rep["asleep"] is False and eng.sleep_snapshot is None
    assert (cache.cache_size, eng.num_pages, eng.linear_state_pool.num_slots) == (256, 100, 9)
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "pinned"]
    for lid in (0, 2):
        for name in eng.bank_schema:
            assert torch.equal(cache.bank_sources[name][lid], eng.original[lid][name])
    assert eng.recaptures == [([1], False)]
    assert eng.warmups == 1 and eng.budget_snapshots == 1
    assert eng._ram_spilled_layers == [] and eng.max_seq_len == 1024


def test_ram_parked_layers_are_still_parked_after_a_wake(tmp_path):
    eng = SleepEngine(tmp_path)
    eng._ram_parked_layers = [2]
    eng.sleep()
    assert eng._ram_parked_layers == []  # the move to the SSD un-parks it...
    eng.wake()
    assert eng._ram_parked_layers == [2]  # ...and the wake puts it back


def test_prefill_overlap_is_off_asleep_and_back_awake(tmp_path):
    eng = SleepEngine(tmp_path, overlap=True)
    eng.sleep()
    assert not eng.moe_offload_cache.prefill_overlap
    eng.wake()
    assert eng.moe_offload_cache.prefill_overlap
    assert len(eng.moe_offload_cache.prefill_bank_buffers) == 2


def test_sleep_is_refused_before_anything_is_freed_without_an_ssd_copy(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.expert_disk_copy.manifest["complete"].pop("2")
    with pytest.raises(SleepRefused, match="no finished copy on the SSD"):
        eng.sleep()
    assert eng.moe_offload_cache.cache_size == 256 and eng.graph_runner.destroyed == 0
    assert eng.sleep_snapshot is None and residency(eng)[2] == "gpu_owned"


def test_sleep_is_refused_while_the_shadow_observer_runs(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.mtp_shadow_observer = object()
    with pytest.raises(SleepRefused, match="shadow"):
        eng.sleep()
    assert eng.graph_runner.destroyed == 0


def test_wake_is_refused_before_touching_anything_while_a_game_holds_the_card(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    eng.game_bytes = eng.sleep_snapshot.free_after - WAKE_MARGIN_BYTES  # a game takes the card
    with pytest.raises(SleepRefused, match="close the game"):
        eng.wake()
    assert eng.sleep_snapshot is not None and eng.moe_offload_cache.cache_size == 0
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"] and eng.recaptures == []
    eng.game_bytes = 0  # the game quits: the next wake works
    assert eng.wake()["asleep"] is False


def test_a_wake_that_fails_midway_goes_back_to_sleep_and_is_refused(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    eng._recapture_graphs = oom
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    assert eng.sleep_snapshot is not None
    assert eng.moe_offload_cache.cache_size == 0 and eng.num_pages == 1
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"]
    del eng._recapture_graphs  # back to the class method
    assert eng.wake()["asleep"] is False
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "pinned"]


def test_a_wake_that_cannot_go_back_to_sleep_raises_wake_failed(tmp_path, monkeypatch):
    eng = SleepEngine(tmp_path)
    eng.sleep()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    eng._recapture_graphs = boom
    monkeypatch.setattr(eng.moe_offload_cache, "release_slots", boom)
    with pytest.raises(WakeFailed):
        eng.wake()


def test_a_second_sleep_or_wake_changes_nothing(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    assert eng.sleep()["note"] == "already asleep" and eng.graph_runner.destroyed == 1
    eng.wake()
    assert eng.wake()["note"] == "already awake" and len(eng.recaptures) == 1


def test_a_ram_squeeze_while_asleep_spills_to_the_ssd_and_the_layer_stays_there(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    rep = eng.step_memory(axis="ram", direction="down", rebuild=eng.asleep_rebuild)
    assert rep["applied"] == "pinned->disk" and rep["layer"] == 3
    assert eng.sleep_snapshot.spilled_while_asleep == [3]
    eng.wake()
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "disk"]
    assert eng.recaptures == [([1], True)]  # graphs deferred while a layer is on the SSD
    assert eng._ram_spilled_layers == [3]  # the governor recalls it later, as today


def test_asleep_rebuild_refuses_anything_but_ssd_spills(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    with pytest.raises(SleepRefused):
        eng.asleep_rebuild(moe_cache_size=128)
    with pytest.raises(SleepRefused):
        eng.asleep_rebuild(layer_moves=[(1, "gpu_owned")])
    assert residency(eng)[1] == "pinned"


def test_the_mtp_draft_head_and_ladder_go_to_sleep_and_come_back(tmp_path, monkeypatch):
    built = []

    class Head:
        def __init__(self, engine, spec):
            built.append((engine, spec))

    class Ladder:
        rebinds = 0

        def rebind(self):
            Ladder.rebinds += 1

    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", Head)
    eng = SleepEngine(tmp_path)
    closed = []
    eng.spec_draft = SimpleNamespace(close=lambda: closed.append(True))
    eng.spec_state_ladder = Ladder()
    eng.config.spec_decode = SimpleNamespace(enabled=True, graph_widths=())
    eng.sleep()
    assert closed == [True] and eng.spec_draft is None
    assert eng.linear_state_pool.num_slots == 2 and Ladder.rebinds == 1  # padding + ladder slot
    eng.wake()
    assert built == [(eng, eng.config.spec_decode)] and isinstance(eng.spec_draft, Head)
    assert eng.spec_captures == 1 and Ladder.rebinds == 2
```

- [ ] **Step 2: Run to see them fail** (`ModuleNotFoundError: freetoken.engine.sleep`).

- [ ] **Step 3: Create `python/freetoken/engine/sleep.py`**

`python/freetoken/engine/sleep.py`:

```python
"""Sleep and wake: give the graphics card back while the model stays in host RAM.

Design: docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md (section 3.2 is the sleep
order, section 3.3 the wake order). Everything here runs at the scheduler's idle safe point, as
rebuild_runtime_cache does, and composes primitives the memory governor has run live since
2026-09-07: ``Engine._move_layer`` (owned layers to and from the SSD expert copy),
``OffloadMoeCache.release_slots`` / ``rebuild``, the KV and GDN pool rebuilds, and graph
teardown and re-capture. Nothing here touches the host expert banks (about 53 GiB pinned), the
dense weights (phase 1 keeps them on the card) or the CUDA context.

Duck-typed on purpose: the functions take the engine and read only the attributes named here,
so tests drive them on the CPU ``FakeDiskEngine`` harness.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import torch

from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.utils import init_logger, mem_GB

logger = init_logger(__name__)

# Wake asks for the bytes sleep released plus this much before it allocates anything. The
# promote guard in rebuild_runtime_cache wants 256 MiB for one layer; a wake also re-creates
# the graph pools and (MTP on) the draft head's scratch, so it asks for twice that.
WAKE_MARGIN_BYTES = 512 << 20


class SleepRefused(CacheRebuildRejected):
    """Sleep or wake refused before anything was freed or allocated, or a failed wake that went
    back to sleep: the engine is in a known state and the scheduler answers "rejected"."""


class WakeFailed(RuntimeError):
    """A wake failed AND going back to sleep failed too: the engine state is unknown and the
    scheduler latches failed (the helper's watchdog then restarts the server)."""


@dataclass
class SleepSnapshot:
    moe_cache_size: int
    owned_layers: tuple[int, ...]
    ram_parked: tuple[int, ...]
    num_pages: int
    linear_slots: int | None  # physical slots, padding sink included
    graph_bs: list[int]
    had_spec_draft: bool
    free_before: int
    free_after: int = 0
    slept_at: float = 0.0
    spilled_while_asleep: list[int] = field(default_factory=list)

    @property
    def released_bytes(self) -> int:
        return max(0, self.free_after - self.free_before)


def _disk_copy(engine):
    cache = engine.moe_offload_cache
    return getattr(engine, "expert_disk_copy", None) or (
        getattr(cache, "expert_disk_copy", None) if cache is not None else None
    )


def _sleep_kv_pages(engine) -> int:
    """The smallest pool the family allows (one page for every family today)."""
    config = engine.config
    min_tokens = int(engine._pool_cls.min_kv_tokens(config))
    return max(1, -(-min_tokens // int(config.page_size)))


def _sleep_linear_slots(engine) -> int:
    # Slot 0 is the padding sink; the MTP ladder takes one more slot in rebind().
    return 2 if getattr(engine, "spec_state_ladder", None) is not None else 1


def _gb(n: int) -> str:
    return f"{n / (1 << 30):.1f} GB"


def _report(snap: SleepSnapshot | None, *, asleep: bool, elapsed: float, free: int, note=None) -> dict:
    return {
        "asleep": asleep,
        "released_bytes": snap.released_bytes if (snap is not None and asleep) else 0,
        "vram_free_bytes": int(free),
        "elapsed_s": round(float(elapsed), 2),
        "note": note,
    }


def check_can_sleep(engine) -> None:
    """Every refusal sleep can make, with nothing freed. The scheduler calls this before it
    parks conversations, so a refused sleep parks nothing either."""
    if getattr(engine, "mtp_shadow_observer", None) is not None:
        raise SleepRefused("sleep is not available while the MTP shadow observer runs (a diagnostic boot)")
    cache = engine.moe_offload_cache
    owned = sorted(engine._gpu_owned_layer_ids) if cache is not None else []
    if owned:
        disk = _disk_copy(engine)
        missing = [l for l in owned if disk is None or not disk.layer_complete(l)]
        if missing:
            raise SleepRefused(
                f"expert layers {missing} have no finished copy on the SSD yet (the first boot "
                "writes it in about four minutes), so sleep would have to keep them in PC memory"
            )


def sleep_engine(engine) -> dict:
    t0 = time.monotonic()
    if getattr(engine, "sleep_snapshot", None) is not None:
        return _report(engine.sleep_snapshot, asleep=True, elapsed=0.0,
                       free=engine.sleep_snapshot.free_after, note="already asleep")
    check_can_sleep(engine)
    cache = engine.moe_offload_cache
    pool = engine.linear_state_pool
    free_before = int(engine._sync_get_memory()[0])
    snap = SleepSnapshot(
        moe_cache_size=int(cache.cache_size) if cache is not None else 0,
        owned_layers=tuple(sorted(engine._gpu_owned_layer_ids)) if cache is not None else (),
        ram_parked=tuple(getattr(engine, "_ram_parked_layers", None) or ()),
        num_pages=int(engine.num_pages),
        linear_slots=int(pool.num_slots) if pool is not None else None,
        graph_bs=list(engine._graph_bs_for_recapture()),
        had_spec_draft=getattr(engine, "spec_draft", None) is not None,
        free_before=free_before,
    )
    # Point of no return for the scheduler's verdict: a failure from here is "after teardown".
    engine.rebuild_teardown_started = True
    release_to_sleep(engine, snap)
    snap.free_after = int(engine._sync_get_memory()[0])
    snap.slept_at = time.monotonic()
    engine.sleep_snapshot = snap
    elapsed = time.monotonic() - t0
    logger.info_rank0(
        f"Asleep in {elapsed:.1f} s: released {mem_GB(snap.released_bytes)} of VRAM, "
        f"{mem_GB(snap.free_after)} free on the card"
    )
    return _report(snap, asleep=True, elapsed=elapsed, free=snap.free_after)


def release_to_sleep(engine, snap: SleepSnapshot) -> None:
    """Free everything sleep frees. Idempotent, so a failed wake can call it to get back to a
    consistent asleep state from anywhere in the wake sequence."""
    progress = engine._report_maintenance_progress
    engine._graphs_deferred = getattr(engine, "_graphs_deferred", None) or "asleep"
    # 1. Graphs first: they bake pool, page-table and slot addresses (rebuild step 1).
    if getattr(engine, "spec_graph_runner", None) is not None:
        engine.spec_graph_runner.destroy()
        engine.spec_graph_runner = None
    engine.attn_backend.reset_capture()
    engine.graph_runner.destroy_cuda_graphs()
    progress("sleep:graphs")
    # 2. The MTP draft head (2.17-2.56 GiB resident, measured 2026-09-02 / 09-07): no host copy
    #    exists, so it is dropped and wake rebuilds it with the boot constructor.
    draft = getattr(engine, "spec_draft", None)
    if draft is not None:
        draft.close()
        engine.spec_draft = None
        progress("sleep:draft")
    # 3. Owned (and RAM-parked) layers go to the SSD copy, never to pinned RAM: 1.32 GiB of
    #    Windows RAM per layer is not there to spend (control-panel acceptance 2026-09-25:
    #    Windows free fell to 2.7 GB during a boot).
    cache = engine.moe_offload_cache
    if cache is not None:
        for layer_id in snap.owned_layers:
            if cache.layer_residency[layer_id] == "gpu_owned":
                engine._move_layer(layer_id, "disk")
                progress("sleep:layer", f"layer {layer_id} -> SSD")
        # 4. The shared slot cache, after the moves (a disk rebind reads the cache geometry).
        if cache.cache_size:
            cache.release_slots()
            progress("sleep:slots")
    # 5. KV to the family's minimum, GDN to its padding slot (+ the ladder's). The scheduler has
    #    already parked every eligible conversation (cache_manager.prepare_rebuild).
    pages = _sleep_kv_pages(engine)
    if engine.num_pages != pages:
        engine._resize_kv_pool(engine.config, pages, None)
    pool = engine.linear_state_pool
    if pool is not None:
        slots = _sleep_linear_slots(engine)
        if pool.num_slots != slots:
            pool.rebuild(slots)
            if getattr(engine, "spec_state_ladder", None) is not None:
                engine.spec_state_ladder.rebind()
    engine._refresh_seq_state(engine.config)
    progress("sleep:pools")
    gc.collect()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None and torch.cuda.is_available() and engine.device.type == "cuda":
        clear()


def wake_engine(engine) -> dict:
    t0 = time.monotonic()
    snap = getattr(engine, "sleep_snapshot", None)
    if snap is None:
        return _report(None, asleep=False, elapsed=0.0, free=0, note="already awake")
    free_now = int(engine._sync_get_memory()[0])
    need = snap.released_bytes + WAKE_MARGIN_BYTES
    if free_now < need:
        raise SleepRefused(
            f"the graphics card has {_gb(free_now)} free and waking needs {_gb(need)}; "
            "close the game or program using it, then try again"
        )
    engine.rebuild_teardown_started = True
    try:
        _restore(engine, snap)
    except Exception as exc:  # noqa: BLE001 - every failure takes the same way back
        logger.error(f"wake failed ({exc!r}); putting the model back to sleep")
        try:
            release_to_sleep(engine, snap)
            engine._sync_get_memory()
        except Exception as exc2:  # noqa: BLE001
            raise WakeFailed(
                f"wake failed ({exc!r}) and going back to sleep failed too ({exc2!r})"
            ) from exc2
        raise SleepRefused(f"wake failed and the model went back to sleep: {exc!r}") from exc
    engine.sleep_snapshot = None
    engine.snapshot_pool_budget()
    free_after = int(engine._sync_get_memory()[0])
    elapsed = time.monotonic() - t0
    logger.info_rank0(f"Awake in {elapsed:.1f} s, {mem_GB(free_after)} free on the card")
    return _report(None, asleep=False, elapsed=elapsed, free=free_after)


def _restore(engine, snap: SleepSnapshot) -> None:
    progress = engine._report_maintenance_progress
    config = engine.config
    # 1. GDN and KV back to their pre-sleep sizes.
    pool = engine.linear_state_pool
    if pool is not None and snap.linear_slots is not None and pool.num_slots != snap.linear_slots:
        pool.rebuild(snap.linear_slots)
        if getattr(engine, "spec_state_ladder", None) is not None:
            engine.spec_state_ladder.rebind()
    if engine.num_pages != snap.num_pages:
        engine._resize_kv_pool(config, snap.num_pages, None)
    engine._refresh_seq_state(config)
    progress("wake:pools")
    cache = engine.moe_offload_cache
    if cache is not None:
        # 2. Slot cache BEFORE the layers come home: resuming prefill overlap (inside
        #    _move_layer) needs cache_size >= 2 * num_experts and live banks.
        if cache.cache_size != snap.moe_cache_size:
            cache.rebuild(snap.moe_cache_size)
            progress("wake:slots")
        # 3. Owned layers SSD -> card. A layer the governor spilled WHILE asleep was pinned,
        #    not owned, so it stays on the SSD until the governor recalls it.
        for layer_id in snap.owned_layers:
            if cache.layer_residency[layer_id] == "disk":
                engine._move_layer(layer_id, "gpu_owned")
                progress("wake:layer", f"layer {layer_id} -> card")
        # _move_layer drops a layer from the RAM-parked list on its way to the SSD.
        engine._ram_parked_layers = [l for l in snap.ram_parked if cache.is_gpu_owned_layer(l)]
    # 4. Graphs, with the boot-resolved sizes (deferred while any layer is on the SSD).
    gc.collect()
    free_min = engine._sync_get_memory()[0]
    progress("wake:capture")
    engine._recapture_graphs(config, snap.graph_bs, free_min)
    # 5. MTP: the draft head, then the verify/draft/ladder graphs captured as at boot.
    spec = getattr(config, "spec_decode", None)
    if snap.had_spec_draft and getattr(engine, "spec_draft", None) is None:
        from .spec_draft import SpecDraftHead

        engine.spec_draft = SpecDraftHead(engine, spec)
        progress("wake:draft")
    engine._rearm_spec_graphs()
    if spec is not None and (getattr(spec, "enabled", False) or getattr(spec, "graph_widths", ())):
        engine._capture_spec_graphs_at_boot()
    progress("wake:captured")
    # 6. Self-check: two short prefills through the fresh pools and kernels, so a sticky CUDA
    #    fault surfaces as a failed wake, not inside Jay's next chat.
    engine._warmup_prefill()
    progress("wake:checked")


def asleep_rebuild(
    engine, *, moe_cache_size=None, num_pages=None, num_mamba_slots=None, num_swa_pages=None,
    layer_moves=None,
) -> None:
    """The ``rebuild`` step_memory runs while the model sleeps: SSD spills and nothing else.

    With the slot cache empty the RAM ladder's park-on-card rung cannot fire (it needs
    num_experts slots to give up), so a RAM-down step reaches the SSD rung and lands here. A
    game squeezing Windows memory then still gets expert layers out of the way; the governor's
    recall brings them back after the wake (design D8)."""
    if any(v is not None for v in (moe_cache_size, num_pages, num_mamba_slots, num_swa_pages)):
        raise SleepRefused("asleep: only expert-layer spills to the SSD run while the model sleeps")
    snap = getattr(engine, "sleep_snapshot", None)
    disk = _disk_copy(engine)
    for layer_id, target in layer_moves or ():
        if target != "disk":
            raise SleepRefused(f"asleep: layer {layer_id} cannot move to {target} until the model wakes")
        if disk is None or not disk.layer_complete(layer_id):
            raise SleepRefused(f"layer {layer_id} has no finished SSD copy")
        engine._move_layer(layer_id, "disk")
        if snap is not None:
            snap.spilled_while_asleep.append(layer_id)
```

- [ ] **Step 4: Add the Engine methods**

```diff
--- a/python/freetoken/engine/engine.py
+++ b/python/freetoken/engine/engine.py
@@ -664,6 +664,30 @@
             # ladder's replay rungs (which need the verify warm-ups' stash to have run).
             self._capture_spec_graphs_at_boot()
 
+    # ---- sleep (engine/sleep.py; docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md) ----
+    # A CLASS default like _gpu_owned_layer_ids: Engine.__new__ stubs in the tests read it.
+    sleep_snapshot = None
+
+    def sleep_preflight(self) -> None:
+        from .sleep import check_can_sleep
+
+        check_can_sleep(self)
+
+    def sleep(self) -> dict:
+        from .sleep import sleep_engine
+
+        return sleep_engine(self)
+
+    def wake(self) -> dict:
+        from .sleep import wake_engine
+
+        return wake_engine(self)
+
+    def asleep_rebuild(self, **kwargs) -> None:
+        from .sleep import asleep_rebuild
+
+        asleep_rebuild(self, **kwargs)
+
     def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
         if config.tp_info.size == 1 or config.use_pynccl:
             torch.distributed.init_process_group(
```

- [ ] **Step 5: Run**

`PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_engine_sleep.py tests/engine/test_memory_step.py tests/moe/test_disk_banks.py -q -p no:cacheprovider`
Expected: `13 passed` for the new file; the others unchanged.

- [ ] **Step 6: Controller commit** — `feat(engine): sleep and wake (release the card, keep the host banks)`

---

### Task 4: The `CacheSleep*` messages and the tokenizer passthrough

**Files:**
- Modify: `python/freetoken/message/backend.py`, `tokenizer.py`, `frontend.py`, `__init__.py`, `python/freetoken/tokenizer/server.py`
- Test: `tests/server/test_sleep_messages.py`

**Interfaces (the wire contract Tasks 5-6 build on):**
- `CacheSleepMsg(request_id: str, action: "sleep"|"wake")` goes api → tokenizer worker, then
  `CacheSleepBackendMsg(request_id, action)` goes to the scheduler.
- `CacheSleepResultMsg(request_id, action, status, asleep=False, released_bytes=0, vram_free_bytes=0, elapsed_s=0.0, error=None)`
  goes scheduler → detokenizer, then `CacheSleepReply` (same fields) goes to the API. `status` is
  one of `ok | rejected | busy | unsupported | failed`. `asleep` is the engine's state
  **after** the operation, whatever its status.

- [ ] **Step 1: Write the failing tests**

`tests/server/test_sleep_messages.py`:

```python
"""Sleep/wake control messages survive the wire and the tokenizer worker's passthrough."""

from __future__ import annotations

from freetoken.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheSleepBackendMsg,
    CacheSleepMsg,
    CacheSleepReply,
    CacheSleepResultMsg,
)
from freetoken.tokenizer.server import _CONTROL_MSG_TYPES, _forward_control_msg


class Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_every_sleep_message_round_trips():
    msg = CacheSleepMsg(request_id="a", action="wake")
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg)) == msg
    backend = CacheSleepBackendMsg(request_id="b", action="sleep")
    assert BaseBackendMsg.decoder(backend.encoder()) == backend
    result = CacheSleepResultMsg(request_id="c", action="sleep", status="ok", asleep=True,
                                 released_bytes=24 << 30, vram_free_bytes=23 << 30, elapsed_s=6.5)
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(result)) == result
    reply = CacheSleepReply(request_id="d", action="wake", status="rejected", asleep=True,
                            error="the graphics card has 2.0 GB free")
    assert BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)) == reply


def test_the_tokenizer_worker_forwards_both_directions_field_by_field():
    assert CacheSleepMsg in _CONTROL_MSG_TYPES and CacheSleepResultMsg in _CONTROL_MSG_TYPES
    backend, frontend = Queue(), Queue()
    assert _forward_control_msg(CacheSleepMsg(request_id="r", action="sleep"), backend, frontend)
    assert backend.items == [CacheSleepBackendMsg(request_id="r", action="sleep")]
    result = CacheSleepResultMsg(request_id="r", action="sleep", status="ok", asleep=True,
                                 released_bytes=7, vram_free_bytes=9, elapsed_s=1.25, error=None)
    assert _forward_control_msg(result, backend, frontend)
    assert frontend.items == [CacheSleepReply(request_id="r", action="sleep", status="ok", asleep=True,
                                              released_bytes=7, vram_free_bytes=9, elapsed_s=1.25)]
```

- [ ] **Step 2: Run to see them fail** (`ImportError: cannot import name 'CacheSleepBackendMsg'`).

- [ ] **Step 3: Implement**

```diff
--- a/python/freetoken/message/backend.py
+++ b/python/freetoken/message/backend.py
@@ -117,6 +117,14 @@
 
 
 @dataclass
+class CacheSleepBackendMsg(BaseBackendMsg):
+    """tokenizer worker -> scheduler: put the engine to sleep (give the card back, keep the
+    host banks) or wake it. See docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md."""
+    request_id: str
+    action: str  # "sleep" | "wake"
+
+
+@dataclass
 class CacheResidencyBackendMsg(BaseBackendMsg):
     """API server -> scheduler: query layer residency report."""
     request_id: str
```

```diff
--- a/python/freetoken/message/tokenizer.py
+++ b/python/freetoken/message/tokenizer.py
@@ -200,6 +200,26 @@
 
 
 @dataclass
+class CacheSleepMsg(BaseTokenizerMsg):
+    # api -> tokenizer worker (passthrough to CacheSleepBackendMsg).
+    request_id: str
+    action: str  # "sleep" | "wake"
+
+
+@dataclass
+class CacheSleepResultMsg(BaseTokenizerMsg):
+    # scheduler -> detokenizer worker (passthrough to CacheSleepReply).
+    request_id: str
+    action: str
+    status: str  # "ok" | "rejected" | "busy" | "unsupported" | "failed"
+    asleep: bool = False  # the engine's state AFTER this operation, whatever its status
+    released_bytes: int = 0
+    vram_free_bytes: int = 0
+    elapsed_s: float = 0.0
+    error: str | None = None
+
+
+@dataclass
 class CacheResidencyMsg(BaseTokenizerMsg):
     request_id: str
 
```

```diff
--- a/python/freetoken/message/frontend.py
+++ b/python/freetoken/message/frontend.py
@@ -96,6 +96,19 @@
 
 
 @dataclass
+class CacheSleepReply(BaseFrontendMsg):
+    # detokenizer worker -> api server: result of a sleep or wake (see CacheSleepResultMsg).
+    request_id: str
+    action: str
+    status: str
+    asleep: bool = False
+    released_bytes: int = 0
+    vram_free_bytes: int = 0
+    elapsed_s: float = 0.0
+    error: str | None = None
+
+
+@dataclass
 class PrefillProgressReply(BaseFrontendMsg):
     processed_tokens: int
     batch_size: int
```

```diff
--- a/python/freetoken/message/__init__.py
+++ b/python/freetoken/message/__init__.py
@@ -4,6 +4,7 @@
     BatchBackendMsg,
     CacheRebuildBackendMsg,
     CacheResidencyBackendMsg,
+    CacheSleepBackendMsg,
     CacheStepBackendMsg,
     ExitMsg,
     PrefixCacheBackendMsg,
@@ -18,6 +19,7 @@
     CacheProgressReply,
     CacheRebuildReply,
     CacheResidencyReply,
+    CacheSleepReply,
     CacheStepReply,
     PrefixCacheReply,
     KVDynamicStatusReply,
@@ -36,6 +38,8 @@
     CacheRebuildResultMsg,
     CacheResidencyMsg,
     CacheResidencyResultMsg,
+    CacheSleepMsg,
+    CacheSleepResultMsg,
     CacheStepMsg,
     CacheStepResultMsg,
     DetokenizeMsg,
@@ -62,6 +66,10 @@
     "CacheProgressReply",
     "CacheRebuildBackendMsg",
     "CacheResidencyBackendMsg",
+    "CacheSleepBackendMsg",
+    "CacheSleepMsg",
+    "CacheSleepReply",
+    "CacheSleepResultMsg",
     "CacheStepBackendMsg",
     "PrefixCacheBackendMsg",
     "RoutingStatsBackendMsg",
```

```diff
--- a/python/freetoken/tokenizer/server.py
+++ b/python/freetoken/tokenizer/server.py
@@ -34,6 +34,10 @@
     CacheResidencyMsg,
     CacheResidencyReply,
     CacheResidencyResultMsg,
+    CacheSleepBackendMsg,
+    CacheSleepMsg,
+    CacheSleepReply,
+    CacheSleepResultMsg,
     CacheStepBackendMsg,
     CacheStepMsg,
     CacheStepReply,
@@ -385,6 +389,8 @@
     CacheRebuildResultMsg,
     CacheResidencyMsg,
     CacheResidencyResultMsg,
+    CacheSleepMsg,
+    CacheSleepResultMsg,
     CacheStepMsg,
     CacheStepResultMsg,
     ErrorReplyMsg,
@@ -515,6 +521,21 @@
                 ram_tight=m.ram_tight,
             )
         )
+    elif isinstance(m, CacheSleepMsg):
+        send_backend.put(CacheSleepBackendMsg(request_id=m.request_id, action=m.action))
+    elif isinstance(m, CacheSleepResultMsg):
+        send_frontend.put(
+            CacheSleepReply(
+                request_id=m.request_id,
+                action=m.action,
+                status=m.status,
+                asleep=m.asleep,
+                released_bytes=m.released_bytes,
+                vram_free_bytes=m.vram_free_bytes,
+                elapsed_s=m.elapsed_s,
+                error=m.error,
+            )
+        )
     elif isinstance(m, CacheResidencyMsg):
         send_backend.put(CacheResidencyBackendMsg(request_id=m.request_id))
     elif isinstance(m, RoutingStatsMsg):
```

- [ ] **Step 4: Run** `tests/server/test_sleep_messages.py tests/server/test_message_wire.py`. Expected: `19 passed`.

- [ ] **Step 5: Controller commit** — `feat(message): sleep and wake control messages`

---

### Task 5: The scheduler: safe point, held chats, auto-wake

**Files:**
- Modify: `python/freetoken/scheduler/scheduler.py`
- Test: `tests/scheduler/test_sleep_scheduler.py`

**Interfaces:**
- Consumes: the Task 3 engine API and the Task 4 messages.
- Produces:
  - A `CacheSleepBackendMsg` is queued through the one operation slot (`_pending_rebuild`),
    executed by `_execute_pending_sleep`, and always answered with exactly one
    `CacheSleepResultMsg` (rule 9).
  - Scheduler-started wakes use `request_id="auto-wake:<n>"`, preceded by
    `MaintenanceBeginMsg(kind="wake")`.
  - While asleep, `run_when_idle`, `_run_kv_dynamic_idle` and `idle_poll_timeout_ms` do
    nothing. A `CacheRebuildBackendMsg` gets "rejected" and a prefix command gets "failed". A
    `CacheStepBackendMsg` runs only as ram/down, with `engine.asleep_rebuild`.
- Rule for "sleep": refused ("busy") while any prefill or decode is runnable or a chat is held.
  `engine.sleep_preflight()` runs **before** conversations are parked.

- [ ] **Step 1: Write the failing tests**

`tests/scheduler/test_sleep_scheduler.py`:

```python
"""The scheduler's side of sleep: the safe point, held chats and auto-wake (review focus 3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sleep import SleepRefused
from freetoken.message import (
    AbortBackendMsg,
    CacheProgressMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    CacheSleepBackendMsg,
    CacheSleepResultMsg,
    CacheStepBackendMsg,
    CacheStepResultMsg,
    ErrorReplyMsg,
    MaintenanceBeginMsg,
    UserMsg,
)
from freetoken.scheduler.scheduler import Scheduler

OK = {"released_bytes": 24 << 30, "vram_free_bytes": 23 << 30, "elapsed_s": 6.5, "note": None}


class FakeEngine:
    def __init__(self, asleep=False):
        self.sleep_snapshot = object() if asleep else None
        self.calls: list[str] = []
        self.num_pages, self.page_table, self.max_seq_len = 100, "table", 6400
        self.rebuild_teardown_started = False
        self.encoder_cache = None
        self.refuse = None
        self.explode = None

    def sleep_preflight(self):
        self.calls.append("preflight")
        if self.refuse:
            raise SleepRefused(self.refuse)

    def sleep(self):
        self.calls.append("sleep")
        self.rebuild_teardown_started = True
        if self.explode:
            raise RuntimeError(self.explode)
        self.sleep_snapshot, self.num_pages = object(), 1
        return {"asleep": True, **OK}

    def wake(self):
        self.calls.append("wake")
        if self.refuse:
            raise SleepRefused(self.refuse)
        self.sleep_snapshot, self.num_pages = None, 100
        return {"asleep": False, **OK}

    def step_memory(self, **kwargs):
        self.calls.append(("step", kwargs["rebuild"]))
        return {"applied": "pinned->disk", "layer": 3, "moe_cache_size": 0}

    def asleep_rebuild(self, **kwargs):
        pass

    def residency_report(self):
        return {}


class FakeCacheManager:
    supports_runtime_rebuild = True
    prefill_chunk_budget = None

    def __init__(self, parks=False):
        self.park_store = object() if parks else None
        self.calls: list = []

    def prepare_rebuild(self):
        self.calls.append("park")

    def rebuild(self, num_pages, page_table):
        self.calls.append(("rebuild", num_pages, page_table))

    def check_integrity(self):
        self.calls.append("check")


def shell(*, asleep=False, prefill=False, decode=False, parks=False):
    s = Scheduler.__new__(Scheduler)
    s.sent = []
    s.send_result = s.sent.extend
    s.engine = FakeEngine(asleep)
    s.cache_manager = FakeCacheManager(parks)
    s.table_manager = SimpleNamespace(rebuild=lambda table: None, token_pool="pool")
    s.prefill_manager = SimpleNamespace(runnable=prefill, pending_list=[], abort_req=lambda uid: None)
    s.decode_manager = SimpleNamespace(runnable=decode, abort_req=lambda uid: None)
    s.config = SimpleNamespace(max_extend_tokens=8192, tp_info=SimpleNamespace(size=1))
    s.device = torch.device("cpu")
    s._pending_rebuild = None
    s._engine_failed = None
    s._kv_dynamic = None
    s._maintenance_request_id = None
    s._maintenance_progress_at = -float("inf")
    s._abort_tombstones = {}
    s._pending_abort_acks = set()
    s._last_data = None
    s._idle_wait_logged = False
    s._sleep_held = []
    s._auto_wake_seq = 0
    s.admitted = []
    s._admit_user_msg = s.admitted.append
    s._queue_for_kv_dynamic = lambda msg: False
    s._log_cache_geometry = lambda event: None
    s._drop_raw_picture = lambda msg: None
    s._release_request_tensors = lambda msg: None
    s._send_kv_dynamic_status = lambda: None
    return s


def sleep_results(s):
    return [m for m in s.sent if isinstance(m, CacheSleepResultMsg)]


def chat(uid=7):
    return UserMsg(uid=uid, input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), sampling_params=SamplingParams())


def run(s, msg):
    s._process_one_msg(msg)
    if s._pending_rebuild is not None:
        s._execute_pending_rebuild()


def test_sleep_is_refused_while_a_chat_runs():
    s = shell(decode=True)
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("busy", False)]
    assert s.engine.calls == []


def test_sleep_parks_conversations_then_sleeps_then_rethreads():
    s = shell(parks=True)
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert s.engine.calls == ["preflight", "sleep"]
    assert s.cache_manager.calls == ["park", ("rebuild", 1, "table"), "check"]
    (reply,) = sleep_results(s)
    assert (reply.status, reply.asleep, reply.released_bytes) == ("ok", True, 24 << 30)


def test_a_refused_sleep_parks_nothing():
    s = shell(parks=True)
    s.engine.refuse = "layers [2] have no finished copy on the SSD yet"
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert s.cache_manager.calls == [] and s.engine.calls == ["preflight"]
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("rejected", False)]


def test_a_sleep_that_fails_after_teardown_latches_failed():
    s = shell()
    s.engine.explode = "CUDA error: an illegal memory access was encountered"
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert [m.status for m in sleep_results(s)] == ["failed"]
    assert s._engine_failed is not None


def test_a_chat_that_reaches_a_sleeping_scheduler_is_held_and_wakes_it():
    s = shell(asleep=True)
    s._process_one_msg(chat(7))
    assert [m.uid for m in s._sleep_held] == [7] and s.admitted == []
    begin = [m for m in s.sent if isinstance(m, MaintenanceBeginMsg)]
    assert [(m.request_id, m.kind) for m in begin] == [("auto-wake:1", "wake")]
    s._process_one_msg(chat(8))  # a second chat joins the same wake
    assert isinstance(s._pending_rebuild, CacheSleepBackendMsg) and s._auto_wake_seq == 1
    s._execute_pending_rebuild()
    assert [m.uid for m in s.admitted] == [7, 8] and s._sleep_held == []
    assert [(m.request_id, m.status, m.asleep) for m in sleep_results(s)] == [("auto-wake:1", "ok", False)]


def test_a_refused_auto_wake_answers_the_held_chats_in_plain_words():
    s = shell(asleep=True)
    s.engine.refuse = "the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game"
    s._process_one_msg(chat(7))
    s._execute_pending_rebuild()
    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert [m.uid for m in errors] == [7] and "close the game" in errors[0].error
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("rejected", True)]
    assert s.admitted == [] and s._sleep_held == []


def test_an_abort_removes_a_held_chat():
    s = shell(asleep=True)
    s._process_one_msg(chat(7))
    s._process_one_msg(AbortBackendMsg(uid=7))
    s._execute_pending_rebuild()
    assert s.admitted == []


def test_a_manual_rebuild_is_refused_while_asleep():
    s = shell(asleep=True)
    s._current_cache_geometry = lambda: {"moe_cache_size": 0, "num_pages": 1, "num_mamba_slots": 0,
                                         "num_swa_pages": 0}
    s._process_one_msg(CacheRebuildBackendMsg(request_id="r", num_pages=10))
    (reply,) = [m for m in s.sent if isinstance(m, CacheRebuildResultMsg)]
    assert reply.status == "rejected" and "asleep" in reply.error


def test_only_a_ram_down_step_runs_while_asleep_and_it_uses_the_ssd_only_rebuild():
    s = shell(asleep=True)
    s._reply_step = lambda request_id, status, result=None, error=None: s.sent.append((status, error))
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="a", axis="vram", direction="down")) == "rejected"
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="b", axis="ram", direction="up")) == "rejected"
    assert s.engine.calls == []
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="c", axis="ram", direction="down")) == "ok"
    assert s.engine.calls == [("step", s.engine.asleep_rebuild)]


def test_a_second_sleep_request_while_one_is_queued_is_busy():
    s = shell()
    s._process_one_msg(CacheSleepBackendMsg(request_id="r1", action="sleep"))
    s._process_one_msg(CacheSleepBackendMsg(request_id="r2", action="sleep"))
    assert [(m.request_id, m.status) for m in sleep_results(s)] == [("r2", "busy")]
```

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement.** One diff, applied hunk by hunk. The new methods sit together just
  before `_current_cache_geometry`.

```diff
--- a/python/freetoken/scheduler/scheduler.py
+++ b/python/freetoken/scheduler/scheduler.py
@@ -30,6 +30,8 @@
     CacheRebuildResultMsg,
     CacheResidencyBackendMsg,
     CacheResidencyResultMsg,
+    CacheSleepBackendMsg,
+    CacheSleepResultMsg,
     CacheStepBackendMsg,
     CacheStepResultMsg,
     DetokenizeMsg,
@@ -213,6 +215,10 @@
         # (see _note_maintenance_progress) correlate to the operation holding the API gate.
         self._maintenance_request_id: str | None = None
         self._maintenance_progress_at = -float("inf")
+        # Chats that reached the scheduler while the engine sleeps (design section 3.4): held
+        # here, never admitted against the one-page sleep pool, and admitted after the wake.
+        self._sleep_held: list[UserMsg] = []
+        self._auto_wake_seq = 0
         self.tokenizer = load_tokenizer(config.model_path)
         self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
         self.toolcall_anchor_id = None
@@ -279,6 +285,8 @@
         (Timer 1). next_park_delay_ms returns None once nothing is pending or parkable, and
         the loop would then block on the queue with no timer at all -- a quiet server would
         never wake for its own shrink (spec rule 4)."""
+        if self._asleep():
+            return None  # nothing to park, shrink or expire until a message arrives
         delays = [
             self.cache_manager.next_park_delay_ms(),
             None
@@ -298,6 +306,10 @@
             # pools a failed teardown left in an unknown state -- and the controller is
             # disabled, so its idle plan is a no-op anyway. Wait for the restart, do nothing.
             return
+        if self._asleep():
+            # Asleep: the pools are one page and the prefix tree is empty, so there is nothing to
+            # park, check or grow (design section 3.2). Messages still wake the loop.
+            return
         if coordinator := getattr(self, "prefix_coordinator", None):
             coordinator.expire()
         if not self._idle_wait_logged:
@@ -1163,7 +1175,7 @@
         """Rule 2/4/8 at an idle safe point: escalate capacity-blocked requests, execute the
         controller's one plan, then drain the held FIFO according to the outcome."""
         c = self._kv_dynamic
-        if c is None or not c.enabled or self._pending_rebuild is not None:
+        if c is None or not c.enabled or self._pending_rebuild is not None or self._asleep():
             # A manual rebuild or governor step already holds the one-operation-at-a-time slot
             # (rule 9): yield and plan at the next idle point, after its geometry re-snapshot.
             return
@@ -1338,6 +1350,10 @@
             for held in controller.disable(reason):
                 self._release_request_tensors(held.msg)
                 uids.append(held.uid)
+        for held_msg in getattr(self, "_sleep_held", None) or ():
+            self._release_request_tensors(held_msg)
+            uids.append(held_msg.uid)
+        self._sleep_held = []
         if uids:
             self.send_result([
                 ErrorReplyMsg(
@@ -1367,7 +1383,10 @@
         elif isinstance(msg, PrefixCacheBackendMsg):
             result = (
                 {"status": "failed", "result": {}, "error": "cache rebuild failed; server needs a restart"}
-                if self._engine_failed is not None else self.prefix_coordinator.command(msg)
+                if self._engine_failed is not None
+                else {"status": "failed", "result": {}, "error": "the model is asleep; wake it first"}
+                if self._asleep()
+                else self.prefix_coordinator.command(msg)
             )
             self.send_result([PrefixCacheResultMsg(request_id=msg.request_id, **result)])
         elif isinstance(msg, UserMsg):
@@ -1396,6 +1415,11 @@
             if msg.mm_items and self.engine.encoder_cache is None:
                 self.send_result([ErrorReplyMsg(uid=msg.uid, error="image input is not supported by this server")])
                 return
+            if self._asleep():
+                # Never admit against the one-page sleep pool: _admit_user_msg's clip would drop
+                # the chat as too long. Hold it and wake (review focus 3).
+                self._hold_for_wake(msg)
+                return
             # Rule 2: with the dynamic KV pool on, the clip and the fit are judged against
             # the CEILING, not today's pool, and a request that needs more room than the pool
             # holds is held here instead of being admitted into a pool that cannot take it.
@@ -1415,6 +1439,9 @@
             kv_dynamic = getattr(self, "_kv_dynamic", None)
             if kv_dynamic is not None:
                 kv_dynamic.on_abort(msg.uid)
+            held = getattr(self, "_sleep_held", None)
+            if held:
+                self._sleep_held = [m for m in held if m.uid != msg.uid]
             tombstones = getattr(self, "_abort_tombstones", None)
             if tombstones is None:
                 tombstones = self._abort_tombstones = {}
@@ -1464,7 +1491,9 @@
                 if is_moe_only
                 else (self._prefill_has_chunked_continuation() or self.decode_manager.runnable)
             )
-            if not self.cache_manager.supports_runtime_rebuild:
+            if self._asleep():
+                self._reply_rebuild(msg.request_id, "rejected", "the model is asleep; wake it first")
+            elif not self.cache_manager.supports_runtime_rebuild:
                 self._reply_rebuild(
                     msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                 )
@@ -1503,6 +1532,16 @@
                     self._reply_step(msg.request_id, "ok", noop)
                 else:
                     self._queue_maintenance(msg)
+        elif isinstance(msg, CacheSleepBackendMsg):
+            if self.config.tp_info.size > 1:
+                self._reply_sleep(msg.request_id, msg.action, "unsupported", error="sleep is unsupported under TP > 1")
+            elif not self.cache_manager.supports_runtime_rebuild:
+                self._reply_sleep(msg.request_id, msg.action, "unsupported",
+                                  error="this model's cache does not support runtime rebuild")
+            elif self._pending_rebuild is not None:
+                self._reply_sleep(msg.request_id, msg.action, "busy", error="another cache operation is queued")
+            else:
+                self._queue_maintenance(msg)
         elif isinstance(msg, CacheResidencyBackendMsg):
             try:
                 rep = self.engine.residency_report()
@@ -1576,6 +1615,8 @@
             error = "server latched failed: cache rebuild failed; server needs a restart"
             if isinstance(msg, CacheRebuildBackendMsg):
                 self._reply_rebuild(msg.request_id, "failed", error=error)
+            elif isinstance(msg, CacheSleepBackendMsg):
+                self._reply_sleep(msg.request_id, msg.action, "failed", error=error)
             else:
                 self._reply_step(msg.request_id, "failed", error=error)
             return
@@ -1775,6 +1816,15 @@
         from freetoken.engine.engine import CacheRebuildRejected
 
         is_idle = not (self.prefill_manager.runnable or self.decode_manager.runnable)
+        rebuild = self.rebuild_cache
+        if self._asleep():
+            # Design D8: while asleep the card belongs to a game. Only the RAM axis's down rung
+            # runs, and engine.asleep_rebuild lets it spill to the SSD and nothing else.
+            if (msg.axis, msg.direction) != ("ram", "down"):
+                self._reply_step(msg.request_id, "rejected",
+                                 error="asleep: only SSD spills run while the model sleeps")
+                return "rejected"
+            rebuild = self.engine.asleep_rebuild
         self.engine.rebuild_teardown_started = False
         try:
             res = self.engine.step_memory(
@@ -1782,7 +1832,7 @@
                 direction=msg.direction,
                 ram_tight=msg.ram_tight,
                 is_idle=is_idle,
-                rebuild=self.rebuild_cache,
+                rebuild=rebuild,
             )
         except CacheRebuildRejected as e:
             logger.warning(f"cache step rejected: {e}")
@@ -1826,7 +1876,8 @@
         # card back bytes the governor deliberately took. Only "ok" changed the geometry:
         # "rejected" retains the old one and "failed" leaves it unknown (and the controller
         # about to be disabled). Outside the finally so the maintenance hook is already gone.
-        if outcome == "ok" and not str(msg.request_id).startswith("auto-kv:"):
+        went_to_sleep = isinstance(msg, CacheSleepBackendMsg) and msg.action == "sleep"
+        if outcome == "ok" and not str(msg.request_id).startswith("auto-kv:") and not went_to_sleep:
             # NOT gated on the controller: validate_rebuild's byte-neutral swap allowance reads
             # pool_budget_bytes whether or not the dynamic pool is on, so a stale budget after a
             # governor step would mis-judge the NEXT operator rebuild too. _send_kv_dynamic_status
@@ -1841,6 +1892,9 @@
                 # uncommitted charges and Timer 1 (external review of 8d566de).
                 self._kv_dynamic.replace_policy(self._build_kv_dynamic_policy(self.config))
             self._send_kv_dynamic_status()
+        # A chat held while this operation ran (or while a sleep that just finished was queued)
+        # needs a wake of its own.
+        self._maybe_auto_wake()
         return outcome
 
     def _execute_pending_operation(self, msg) -> str:
@@ -1854,6 +1908,8 @@
 
         if isinstance(msg, CacheStepBackendMsg):
             return self._execute_pending_step(msg)
+        if isinstance(msg, CacheSleepBackendMsg):
+            return self._execute_pending_sleep(msg)
         requested = {
             "moe_cache_size": msg.moe_cache_size,
             "num_pages": msg.num_pages,
@@ -1926,6 +1982,134 @@
         self._reply_rebuild(msg.request_id, "ok")
         return "ok"
 
+    # ---- sleep (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md) ----
+
+    def _asleep(self) -> bool:
+        return getattr(getattr(self, "engine", None), "sleep_snapshot", None) is not None
+
+    def _reply_sleep(
+        self, request_id: str, action: str, status: str, *, result: dict | None = None,
+        error: str | None = None,
+    ) -> None:
+        res = result or {}
+        self.send_result([
+            CacheSleepResultMsg(
+                request_id=request_id,
+                action=action,
+                status=status,
+                # the engine's state AFTER this operation, whatever its status: the API's
+                # asleep flag follows it (a refused wake leaves the model asleep)
+                asleep=self._asleep(),
+                released_bytes=int(res.get("released_bytes", 0) or 0),
+                vram_free_bytes=int(res.get("vram_free_bytes", 0) or 0),
+                elapsed_s=float(res.get("elapsed_s", 0.0) or 0.0),
+                error=error,
+            )
+        ])
+
+    def _execute_pending_sleep(self, msg: CacheSleepBackendMsg) -> str:
+        """Sleep or wake at the safe point; rule 8's three outcomes, like a rebuild."""
+        from freetoken.engine.engine import CacheRebuildRejected
+
+        engine = self.engine
+        if msg.action == "sleep" and not self._asleep():
+            if (self.prefill_manager.runnable or self.decode_manager.runnable
+                    or getattr(self, "_sleep_held", None)):
+                self._reply_sleep(msg.request_id, "sleep", "busy",
+                                  error="a chat is running; try again when it ends")
+                return "rejected"
+            try:
+                engine.sleep_preflight()  # refuse before a single conversation is parked
+            except CacheRebuildRejected as e:
+                logger.warning(f"sleep refused: {e}")
+                self._reply_sleep(msg.request_id, "sleep", "rejected", error=str(e))
+                return "rejected"
+            if self.device.type == "cuda":
+                torch.cuda.synchronize(self.device)
+            if coordinator := getattr(self, "prefix_coordinator", None):
+                coordinator.before_rebuild()
+            if getattr(self.cache_manager, "park_store", None) is not None:
+                # Park every eligible conversation to RAM before the KV pool goes: chats survive
+                # a sleep and restore on their next turn (cache.py prepare_rebuild).
+                self.cache_manager.prepare_rebuild()
+            self._note_maintenance_progress("sleep:parked", force=True)
+        engine.rebuild_teardown_started = False
+        try:
+            result = engine.sleep() if msg.action == "sleep" else engine.wake()
+        except CacheRebuildRejected as e:
+            # SleepRefused: refused before anything moved, or a failed wake that went back to
+            # sleep. Either way the engine is in a known state.
+            logger.warning(f"{msg.action} refused: {e}")
+            self._reply_sleep(msg.request_id, msg.action, "rejected", error=str(e))
+            if msg.action == "wake":
+                self._refuse_held(f"the model is asleep and could not wake: {e}")
+            return "rejected"
+        except Exception as e:  # noqa: BLE001
+            if not getattr(engine, "rebuild_teardown_started", True):
+                logger.error(f"{msg.action} failed before teardown: {e!r}; engine untouched")
+                self._reply_sleep(msg.request_id, msg.action, "rejected", error=repr(e))
+                return "rejected"
+            logger.error(f"{msg.action} failed after teardown: {e!r}; latching failed")
+            self._reply_sleep(msg.request_id, msg.action, "failed", error=repr(e))
+            self._latch_engine_failed(f"{msg.action} failed: {e!r}")
+            return "failed"
+        if not result.get("note"):  # a real change, not "already asleep/awake"
+            self._rethread_after_sleep()
+            self._log_cache_geometry("Asleep" if msg.action == "sleep" else "Awake")
+        self._reply_sleep(msg.request_id, msg.action, "ok", result=result)
+        if msg.action == "wake":
+            self._admit_held()
+        return "ok"
+
+    def _rethread_after_sleep(self) -> None:
+        """Point the page managers at the pools sleep or wake just re-made: rebuild_cache's
+        tail for a KV + GDN resize (a new prefix tree; parked prefixes restore into it)."""
+        self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
+        self.table_manager.rebuild(self.engine.page_table)
+        self.token_pool = self.table_manager.token_pool
+        if coordinator := getattr(self, "prefix_coordinator", None):
+            coordinator.max_seq_len = self.engine.max_seq_len
+        self.cache_manager.check_integrity()
+        chunk_cap = self.cache_manager.prefill_chunk_budget
+        self.prefill_budget = (
+            min(self.config.max_extend_tokens, chunk_cap) if chunk_cap else self.config.max_extend_tokens
+        )
+
+    def _hold_for_wake(self, msg: UserMsg) -> None:
+        held = getattr(self, "_sleep_held", None)
+        if held is None:
+            held = self._sleep_held = []
+        held.append(msg)
+        self._maybe_auto_wake()
+
+    def _maybe_auto_wake(self) -> None:
+        """Queue a wake for held chats unless one (or any other operation) is already queued."""
+        if (not self._asleep() or not getattr(self, "_sleep_held", None)
+                or self._pending_rebuild is not None or self._engine_failed is not None):
+            return
+        self._auto_wake_seq = getattr(self, "_auto_wake_seq", 0) + 1
+        request_id = f"auto-wake:{self._auto_wake_seq}"
+        # Rule 9, as for auto-kv: the frontend opens its own record on this begin and closes it
+        # on the CacheSleepResultMsg that _execute_pending_sleep always sends.
+        self.send_result([
+            MaintenanceBeginMsg(request_id=request_id, kind="wake", detail="a chat arrived while asleep")
+        ])
+        self._queue_maintenance(CacheSleepBackendMsg(request_id=request_id, action="wake"))
+
+    def _admit_held(self) -> None:
+        held, self._sleep_held = getattr(self, "_sleep_held", None) or [], []
+        for held_msg in held:
+            if not self._queue_for_kv_dynamic(held_msg):
+                self._admit_user_msg(held_msg)
+
+    def _refuse_held(self, error: str) -> None:
+        held, self._sleep_held = getattr(self, "_sleep_held", None) or [], []
+        if not held:
+            return
+        self.send_result([ErrorReplyMsg(uid=m.uid, error=error, code="server_error") for m in held])
+        for held_msg in held:
+            self._drop_raw_picture(held_msg)
+
     def _current_cache_geometry(self) -> dict:
         """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
         the single source for _reply_rebuild's readout. None for a pool this model lacks
```

- [ ] **Step 4: Run** `tests/scheduler/test_sleep_scheduler.py` (expect `10 passed`), then the
  whole `tests/scheduler` against the baseline. Expected: no new failures.

- [ ] **Step 5: Controller commit** — `feat(scheduler): sleep and wake at the safe point, hold chats and auto-wake`

---

### Task 6: The API server: `asleep`, auto-wake at the front door, `/v1/sleep` and `/v1/wake`

**Files:**
- Modify: `python/freetoken/server/api_server.py`, `python/freetoken/server/control_api.py`
- Test: `tests/server/test_sleep_api.py`

**Interfaces:**
- `FrontendManager.asleep: bool`, `sleep_info: dict`, `ensure_awake(timeout) -> str | None`.
- `api_server.WAKE_WAIT_S = 300.0`, `dispatch_sleep(state, *, action, timeout) -> dict`.
- `control_api.public_state(state) -> str` returns `"sleeping"` when `asleep` and the
  maintenance state is `"serving"`. It is used by `/health` (`maintenance`), `/ready` (ready)
  and `/v1/cache/status` (`state`, plus a `sleep` block).
- `POST /v1/sleep` answers 200 ok, 409 busy, 504 timeout, or 503 otherwise.
  `POST /v1/wake` waits out a running operation and then wakes: 200, or 503 with the reason.
- Every chat route already goes through `wait_until_serving` or `new_user`, and both now wake a
  sleeping model. Ten chats at once send one wake.

- [ ] **Step 1: Write the failing tests**

`tests/server/test_sleep_api.py`:

```python
"""The API side of sleep: state words, the chat gate's auto-wake, and the two routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import freetoken.server.api_server as api
from freetoken.message import CacheSleepMsg, CacheSleepReply
from freetoken.server.accounting import AdmissionClosedError
from freetoken.server.api_server import FrontendManager
from freetoken.server.control_api import build_health, is_ready, public_state


def manager(asleep=False) -> FrontendManager:
    config = SimpleNamespace(served_model_name="qwen", kv_park="ram", kv_dtype="fp8")
    m = FrontendManager(config=config, send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving")
    m.asleep = asleep
    return m


def wire_scheduler(m: FrontendManager, *, wake_status="ok", error=None):
    """send_one answers each CacheSleepMsg the way the scheduler would, on the next loop turn."""
    sent = []

    async def send_one(msg):
        sent.append(msg)
        assert isinstance(msg, CacheSleepMsg)
        ok = wake_status == "ok"
        asleep = msg.action == "sleep" if ok else True
        reply = CacheSleepReply(request_id=msg.request_id, action=msg.action,
                                status="ok" if msg.action == "sleep" else wake_status,
                                asleep=asleep, released_bytes=24 << 30, elapsed_s=12.0, error=error)
        asyncio.get_running_loop().call_soon(m._resolve_sleep, reply)

    m.send_one = send_one
    return sent


def test_public_state_health_and_ready_say_sleeping():
    m = manager(asleep=True)
    m.ready_at = 0.0
    assert public_state(m) == "sleeping"
    doc = build_health(m, "1.0")
    assert doc["status"] == "ok" and doc["maintenance"] == "sleeping" and is_ready(doc)
    m.maintenance_state = "rebuilding"  # a wake in flight reads as the rebuild it is
    assert public_state(m) == "rebuilding"


@pytest.mark.anyio
async def test_a_chat_wakes_a_sleeping_model_once_and_is_admitted():
    m = manager(asleep=True)
    sent = wire_scheduler(m)
    uids = await asyncio.gather(m.new_user_async(), m.new_user_async(), m.new_user_async())
    assert sorted(uids) == [0, 1, 2]
    assert [msg.action for msg in sent] == ["wake"]  # three chats, one wake
    assert m.asleep is False and m.maintenance_state == "serving"
    assert m.sleep_info["last_wake_s"] == 12.0


@pytest.mark.anyio
async def test_a_chat_while_a_game_holds_the_card_gets_a_plain_refusal():
    m = manager(asleep=True)
    wire_scheduler(m, wake_status="rejected",
                   error="the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game")
    with pytest.raises(AdmissionClosedError, match="close the game"):
        await m.new_user_async()
    assert m.asleep is True and m.maintenance_state == "serving" and m.stats.active == 0
    reason = await m.wait_until_serving()
    assert "could not wake" in reason


@pytest.mark.anyio
async def test_a_chat_that_waited_out_the_sleep_itself_then_wakes_the_model():
    m = manager()
    sent = wire_scheduler(m)
    sleep = asyncio.ensure_future(api.dispatch_sleep(m, action="sleep"))
    await asyncio.sleep(0)  # the sleep is in flight: the gate is shut
    assert m.maintenance_state == "rebuilding"
    reason = await m.wait_until_serving()
    assert reason is None and (await sleep)["status"] == "ok"
    assert [msg.action for msg in sent] == ["sleep", "wake"] and m.asleep is False


@pytest.mark.anyio
async def test_the_scheduler_auto_wake_reply_clears_the_flag_without_a_waiter():
    m = manager(asleep=True)
    api._open_maintenance(m, "auto-wake:1", "wake")
    m._resolve_sleep(CacheSleepReply(request_id="auto-wake:1", action="wake", status="ok", asleep=False))
    assert m.asleep is False and m.maintenance_state == "serving" and m.rebuild_done.is_set()


def test_the_routes(monkeypatch):
    m = manager()
    monkeypatch.setattr(api, "get_global_state", lambda: m)
    client = TestClient(api.app)

    async def fake_dispatch(state, *, action, timeout=api.WAKE_WAIT_S):
        state.asleep = action == "sleep"
        return {"status": "ok", "action": action, "asleep": state.asleep, "released_bytes": 24 << 30}

    monkeypatch.setattr(api, "dispatch_sleep", fake_dispatch)
    r = client.post("/v1/sleep")
    assert r.status_code == 200 and r.json()["asleep"] is True
    assert client.post("/v1/sleep").json()["note"] == "already asleep"
    assert client.get("/v1/cache/status").json()["state"] == "sleeping"
    r = client.post("/v1/wake")
    assert r.status_code == 200 and r.json()["woke"] is True and m.asleep is False
    m.maintenance_state = "loading"
    assert client.post("/v1/sleep").status_code == 503
```

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement**

```diff
--- a/python/freetoken/server/api_server.py
+++ b/python/freetoken/server/api_server.py
@@ -29,6 +29,8 @@
     CacheRebuildReply,
     CacheResidencyMsg,
     CacheResidencyReply,
+    CacheSleepMsg,
+    CacheSleepReply,
     CacheStepMsg,
     CacheStepReply,
     KVDynamicStatusReply,
@@ -62,7 +64,7 @@
 from .args import ServerArgs
 from .anthropic_api import register_anthropic_routes
 from .accounting import AdmissionClosedError, register_accounting_routes
-from .control_api import register_control_routes
+from .control_api import public_state, register_control_routes
 from .openai_api import register_openai_routes
 from .prefix_api import register_prefix_routes
 from .recent_prompts import register_recent_prompt_routes
@@ -93,6 +95,11 @@
 # operation, and is never subject to this clock. Slightly slow to catch a real wedge beats
 # ever declaring a healthy server dead (it takes the server down until a ten-minute reboot).
 MAINTENANCE_STUCK_S = 600.0
+# How long a chat (or POST /v1/wake) waits for a sleeping engine to wake. The design estimate is
+# 10-33 s (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md section 2.4); 300 s covers
+# a slow SSD and matches the helper's wait for a whole Stop, and is still well under llama-swap's
+# healthCheckTimeout of 600 s.
+WAKE_WAIT_S = 300.0
 # The event-loop task that applies the limit when nobody polls /health or /v1/cache/status.
 MAINTENANCE_WATCH_INTERVAL_S = 5.0
 
@@ -391,6 +398,13 @@
     _frontend_warm_started: bool = False
     # Event set when not rebuilding; cleared when rebuilding starts
     rebuild_done: asyncio.Event = field(default_factory=asyncio.Event)
+    # Sleep (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md): True while the engine
+    # has given the card back. Kept apart from maintenance_state so every existing gate keeps its
+    # meaning; control_api.public_state reports "sleeping" when this is set and the engine is
+    # otherwise serving. Only a CacheSleepReply changes it.
+    asleep: bool = False
+    sleep_info: Dict[str, Any] = field(default_factory=dict)
+    _wake_task: Any = None
     _residency_cache: Dict[str, Any] | None = None
     _residency_time: float = 0.0
 
@@ -467,6 +481,28 @@
             )
         return self._allocate_user()
 
+    async def ensure_awake(self, timeout: float = WAKE_WAIT_S) -> str | None:
+        """Wake a sleeping engine for a chat. None once awake, else the 503 reason. Concurrent
+        callers share one wake task, so ten chats arriving together send one wake."""
+        if not self.asleep:
+            return None
+        task = self._wake_task
+        if task is None or task.done():
+            task = self._wake_task = asyncio.ensure_future(dispatch_sleep(self, action="wake", timeout=timeout))
+        try:
+            result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
+        except asyncio.TimeoutError:
+            return f"server unavailable: waking up took longer than {int(timeout)} s"
+        if result.get("status") == "ok" and not self.asleep:
+            return None
+        return f"server is asleep and could not wake: {result.get('error') or result.get('status')}"
+
+    async def _wake_and_allocate(self, timeout: float) -> int:
+        reason = await self.ensure_awake(max(timeout, WAKE_WAIT_S))
+        if reason is not None:
+            raise AdmissionClosedError(reason)
+        return await self.new_user_async(timeout)
+
     async def wait_until_serving(self, timeout: float = 120.0) -> str | None:
         """The adapters' front-door gate. ``None`` when the engine is serving (after waiting out
         a runtime rebuild, up to ``timeout``); otherwise the 503 reason.
@@ -480,6 +516,12 @@
                 await asyncio.wait_for(self.rebuild_done.wait(), timeout=timeout)
             except asyncio.TimeoutError:
                 return f"server unavailable: cache rebuild timed out after {timeout}s"
+        if self.maintenance_state == "serving" and self.asleep:
+            # A chat to a sleeping model wakes it (design section 2.5); a refused wake (a game
+            # holds the card) is the 503 reason, in words Jay can act on.
+            reason = await self.ensure_awake(max(timeout, WAKE_WAIT_S))
+            if reason is not None:
+                return reason
         if self.maintenance_state == "loading":
             return "model is still loading"
         if self.maintenance_state == "failed":
@@ -495,6 +537,8 @@
             )
         if self.maintenance_state == "rebuilding":
             return self._wait_rebuild_and_allocate(timeout=timeout)
+        if self.asleep:
+            return self._wake_and_allocate(timeout)
         return _AwaitableInt(self._allocate_user())
 
     async def new_user_async(self, timeout: float = 120.0) -> int:
@@ -531,6 +575,9 @@
             if isinstance(msg, CacheStepReply):
                 self._resolve_step(msg)
                 continue
+            if isinstance(msg, CacheSleepReply):
+                self._resolve_sleep(msg)
+                continue
             if isinstance(msg, CacheResidencyReply):
                 self._resolve_residency(msg)
                 continue
@@ -629,6 +676,28 @@
             self._residency_time = 0.0
         _close_maintenance(self, msg.request_id, failed=(msg.status == "failed"))
 
+    def _resolve_sleep(self, msg: CacheSleepReply) -> None:
+        """A sleep or wake reply (API-dispatched or the scheduler's own auto-wake). The asleep
+        flag follows the scheduler's report of the engine's state after the operation, set
+        BEFORE the gate reopens so a waiter released by it sees the right side of sleep."""
+        if msg.status != "failed":
+            self.asleep = bool(msg.asleep)
+            if msg.status == "ok" and msg.action == "sleep":
+                self.sleep_info = {"since": time.time(), "released_bytes": msg.released_bytes,
+                                   "vram_free_bytes": msg.vram_free_bytes, "sleep_s": msg.elapsed_s}
+            elif msg.status == "ok" and msg.action == "wake":
+                self.sleep_info = {"woke_at": time.time(), "last_wake_s": msg.elapsed_s}
+        fut = self.rebuild_futures.pop(msg.request_id, None)
+        if fut is not None and not fut.done():
+            fut.set_result({
+                "status": msg.status, "action": msg.action, "asleep": bool(msg.asleep),
+                "released_bytes": msg.released_bytes, "vram_free_bytes": msg.vram_free_bytes,
+                "elapsed_s": msg.elapsed_s, "error": msg.error,
+            })
+        if not _reply_matches_open_operation(self, msg.request_id):
+            return
+        _close_maintenance(self, msg.request_id, failed=(msg.status == "failed"))
+
     def _note_progress(self, msg: CacheProgressReply) -> None:
         """One unit of work completed on (or ahead of) its operation: restart that record's
         clock. A report for an id with no open record (an operation that already finished, or
@@ -1224,6 +1293,55 @@
     return JSONResponse(result, status_code=200 if result.get("status") == "ok" else 503)
 
 
+async def dispatch_sleep(state: FrontendManager, *, action: str, timeout: float = WAKE_WAIT_S) -> Dict[str, Any]:
+    """Send a sleep or wake to the scheduler under the maintenance gate and await the reply
+    (the same shape as dispatch_step: requests wait while it runs)."""
+    request_id = str(uuid.uuid4())
+    fut = asyncio.get_running_loop().create_future()
+    state.rebuild_futures[request_id] = fut
+    _open_maintenance(state, request_id, action)
+    try:
+        await state.send_one(CacheSleepMsg(request_id=request_id, action=action))
+    except Exception as e:  # noqa: BLE001
+        state.rebuild_futures.pop(request_id, None)
+        _abort_maintenance(state, request_id)
+        return {"status": "failed", "error": f"failed to dispatch {action}: {e!r}"}
+    try:
+        return await asyncio.wait_for(fut, timeout=timeout)
+    except asyncio.TimeoutError:
+        # As dispatch_rebuild: the gate stays shut until the reply lands.
+        state.rebuild_futures.pop(request_id, None)
+        return {"status": "timeout", "request_id": request_id}
+
+
+@app.post("/v1/sleep")
+async def sleep_route(timeout: float = 120.0):
+    """Give the graphics card back and keep the model in PC memory (design section 3.2)."""
+    state = get_global_state()
+    if state.maintenance_state in ("loading", "failed"):
+        return JSONResponse({"status": state.maintenance_state, "error": f"engine is {state.maintenance_state}"},
+                            status_code=503)
+    if state.maintenance_state in ("rebuilding", "stopping"):
+        return JSONResponse({"status": "busy", "error": f"engine is {state.maintenance_state}; try again"},
+                            status_code=409)
+    if state.asleep:
+        return {"status": "ok", "asleep": True, "note": "already asleep", **state.sleep_info}
+    result = await dispatch_sleep(state, action="sleep", timeout=timeout)
+    code = {"ok": 200, "busy": 409, "timeout": 504}.get(result.get("status"), 503)
+    return JSONResponse(result, status_code=code)
+
+
+@app.post("/v1/wake")
+async def wake_route(timeout: float = WAKE_WAIT_S):
+    """Wake a sleeping engine (a chat does this by itself); waits out a running operation."""
+    state = get_global_state()
+    was_asleep = state.asleep
+    reason = await state.wait_until_serving(timeout)
+    if reason is None:
+        return {"status": "ok", "asleep": False, "woke": was_asleep, **state.sleep_info}
+    return JSONResponse({"status": "rejected", "asleep": state.asleep, "error": reason}, status_code=503)
+
+
 @app.get("/v1/cache/residency")
 async def cache_residency(timeout: float = 10.0):
     """Report current layer residency, cache sizes, and model-derived ``layer_bytes``."""
@@ -1494,9 +1612,10 @@
     inference = check_inference() if callable(check_inference) else None
     return {
         "instance_id": getattr(state, "instance_id", None),
-        "state": state.maintenance_state,
+        "state": public_state(state),
         "inference": inference,
         "maintenance": maintenance,
+        "sleep": {"asleep": bool(getattr(state, "asleep", False)), **(getattr(state, "sleep_info", None) or {})},
         "kv_dynamic": getattr(state, "kv_dynamic_status", None),
         "last_rebuild": state.last_rebuild,
         "geometry": cache_geometry(state),
```

```diff
--- a/python/freetoken/server/control_api.py
+++ b/python/freetoken/server/control_api.py
@@ -15,6 +15,16 @@
 from fastapi.responses import JSONResponse
 
 
+def public_state(state: Any) -> str:
+    """maintenance_state as the outside world sees it: "sleeping" while the engine has given
+    the card back and is otherwise serving (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md).
+    The helper's watchdog, governor and adapters read this word from /v1/cache/status."""
+    mstate = getattr(state, "maintenance_state", "serving")
+    if mstate == "serving" and getattr(state, "asleep", False):
+        return "sleeping"
+    return mstate
+
+
 def build_health(state: Any, version: str) -> dict:
     """Full-lifecycle health doc: loading -> ok -> error."""
     instance_id = getattr(state, "instance_id", None)
@@ -28,7 +38,7 @@
     if fatal:
         return {"status": "error", "message": fatal, "instance_id": instance_id, "inference": inference}
 
-    mstate = getattr(state, "maintenance_state", "serving")
+    mstate = public_state(state)
     config = getattr(state, "config", None)
     model = getattr(config, "served_model_name", None)
 
@@ -56,6 +66,8 @@
         "version": version,
         "inference": inference,
     }
+    if mstate == "sleeping":
+        doc["sleep"] = dict(getattr(state, "sleep_info", None) or {})
     if isinstance(maintenance, dict) and maintenance.get("age_s") is not None:
         doc["maintenance_age_s"] = maintenance["age_s"]
         doc["maintenance_phase"] = maintenance.get("phase")
@@ -67,7 +79,9 @@
 # out by the chat routes' gate (a few seconds of slowness), so it still counts as ready. A
 # rebuild stuck past its deadline turns /health (and so /ready) to "error" through
 # check_maintenance, the same verdict the chat gate reaches.
-_READY_MAINTENANCE = ("serving", "rebuilding")
+# A sleeping engine is ready too: a chat wakes it (design section 2.5), and llama-swap must keep
+# treating a sleeping FreeToken as loaded, not start it again.
+_READY_MAINTENANCE = ("serving", "rebuilding", "sleeping")
 
 
 def is_ready(health: dict) -> bool:
```

- [ ] **Step 4: Run** `tests/server/test_sleep_api.py` (expect `6 passed`), then `tests/server tests/tokenizer` against the baseline.

- [ ] **Step 5: Controller commit** — `feat(server): /v1/sleep and /v1/wake; a chat wakes a sleeping model`

---

### Task 7: The helper: proxy, route, watchdog, governor, reclaim

**Files:**
- Modify: `python/freetoken/daemon/settings/process_manager.py`, `app.py`, `watchdog.py`, `governor.py`, `memory_reclaim.py`
- Test: `tests/settings/test_sleep_helper.py`

**Interfaces:**
- `ProcessManager.sleep_server(action, *, timeout=330.0) -> dict` returns the server's JSON plus
  `httpStatus` (0 when nothing answered). It creates no lifecycle job and takes no GPU lock.
- `POST /api/server/sleep|wake` returns the proxied reply with the server's status code (503 when unreachable).
- The crash watchdog adopts and keeps `sleeping` exactly like `serving`.
- The governor handles a `sleeping` tick with `_tick_asleep()`: RAM axis, down only, no VRAM
  probe. The memory-reclaim controller stays enabled while sleeping.
- (Task 9) `create_app` wires `PanelService(freetoken_state=..., freetoken_control=getattr(process_manager, "sleep_server", None))`.
  The `getattr` matters: `tests/settings/test_routes.py` hands in a `RecordingManager` without
  `sleep_server`, and without `getattr` it failed in the prototype.

- [ ] **Step 1: Write the failing tests**

`tests/settings/test_sleep_helper.py`:

```python
"""The settings helper's side of sleep: the proxy, the route, the watchdog and the governor."""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings import governor
from freetoken.daemon.settings import process_manager as pm_module
from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.governor import GIB, GovernorLoop, GovernorPolicy
from freetoken.daemon.settings.process_manager import ProcessManager
from tests.settings.test_crash_watchdog import _wired


class Reply:
    def __init__(self, status, body):
        self.status, self._body = status, json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def manager(tmp_path, readiness=lambda: {"state": "serving"}):
    return ProcessManager(boot_file=tmp_path / "boot.ps1", stop_script=tmp_path / "stop.ps1",
                          log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
                          runner=lambda *a, **k: None, readiness=readiness, sleep=lambda _: None,
                          poll_interval=0)


def test_sleep_server_posts_to_the_model_server_and_reports_the_http_status(tmp_path, monkeypatch):
    seen = []

    def urlopen(request, timeout=None):
        seen.append((request.full_url, request.get_method(), timeout))
        if request.full_url.endswith("/v1/wake"):
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(
                json.dumps({"status": "rejected", "error": "close the game"}).encode()))
        return Reply(200, {"status": "ok", "asleep": True})

    monkeypatch.setattr(pm_module.urllib.request, "urlopen", urlopen)
    pm = manager(tmp_path)
    assert pm.sleep_server("sleep") == {"status": "ok", "asleep": True, "httpStatus": 200}
    assert pm.sleep_server("wake") == {"status": "rejected", "error": "close the game", "httpStatus": 503}
    assert seen[0] == (f"http://127.0.0.1:{pm.port}/v1/sleep", "POST", 330.0)
    with pytest.raises(ValueError):
        pm.sleep_server("nap")


def test_sleep_server_says_unreachable_when_nothing_answers(tmp_path, monkeypatch):
    def refused(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr(pm_module.urllib.request, "urlopen", refused)
    assert manager(tmp_path).sleep_server("sleep")["httpStatus"] == 0


def test_the_server_route_proxies_sleep_and_wake_without_a_job(tmp_path):
    pm = manager(tmp_path)
    pm.sleep_server = lambda action: {"status": "ok" if action == "sleep" else "rejected",
                                      "httpStatus": 200 if action == "sleep" else 503}
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    client = TestClient(create_app(boot_file=boot, process_manager=pm, log_path=tmp_path / "server.log",
                                   static_path=tmp_path / "missing.html"))
    assert client.post("/api/server/sleep").status_code == 200
    assert client.post("/api/server/wake").status_code == 503
    assert pm.current_job() is None
    assert client.post("/api/server/nap").status_code == 422


def test_the_watchdog_treats_a_sleeping_server_as_alive(tmp_path):
    state = {"state": "sleeping"}
    manager_, dog = _wired(tmp_path, lambda: dict(state))
    for _ in range(5):
        dog.tick()
    assert dog.armed is True and dog.misses == 0 and manager_.current_job() is None


def _asleep_loop(monkeypatch, free_ram):
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "sleeping"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB), http_port=2020)
    monkeypatch.setattr(governor, "_is_wsl", lambda: True)
    monkeypatch.setattr(governor, "_read_proc_meminfo_available", lambda: 80 * GIB)
    monkeypatch.setattr(governor, "read_free_windows_ram_bytes", lambda **_kwargs: free_ram)
    monkeypatch.setattr(governor, "read_free_vram_bytes",
                        lambda: pytest.fail("the governor must not look at the card while asleep"))
    executed = []
    monkeypatch.setattr(loop, "_execute_action", lambda action, fv, fr: executed.append(action))
    return loop, executed


def test_asleep_the_governor_only_spills_to_the_ssd_under_ram_pressure(monkeypatch):
    loop, executed = _asleep_loop(monkeypatch, free_ram=1 * GIB)  # a game squeezes Windows RAM
    loop._tick()
    assert [(a.axis, a.direction) for a in executed] == [("ram", "down")]


def test_asleep_the_governor_never_recalls_even_with_ram_to_spare(monkeypatch):
    loop, executed = _asleep_loop(monkeypatch, free_ram=60 * GIB)
    for _ in range(3):
        loop._tick()
    assert executed == []
```

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement**

```diff
--- a/python/freetoken/daemon/settings/process_manager.py
+++ b/python/freetoken/daemon/settings/process_manager.py
@@ -620,6 +620,34 @@
                 return snippet or signature
         return None
 
+    # ---- sleep -----------------------------------------------------------
+
+    def sleep_server(self, action: str, *, timeout: float = 330.0) -> dict[str, Any]:
+        """POST /v1/sleep or /v1/wake on the model server: its reply plus ``httpStatus``.
+
+        No job and no GPU lock: sleep keeps the process, so nothing here can race a Start or
+        Stop the way a lifecycle action would. ``timeout`` sits above the server's own 300 s
+        wake wait (api_server.WAKE_WAIT_S). ``httpStatus`` 0 means the server did not answer.
+        """
+        if action not in {"sleep", "wake"}:
+            raise ValueError("action must be sleep or wake")
+        request = urllib.request.Request(
+            f"http://127.0.0.1:{self.port}/v1/{action}", data=b"", method="POST",
+            headers={"Accept": "application/json"},
+        )
+        try:
+            with urllib.request.urlopen(request, timeout=timeout) as response:
+                body = json.loads(response.read().decode("utf-8") or "{}")
+                return {**body, "httpStatus": int(response.status)}
+        except urllib.error.HTTPError as exc:
+            try:
+                body = json.loads(exc.read().decode("utf-8") or "{}")
+            except Exception:  # noqa: BLE001 - a non-JSON error page
+                body = {}
+            return {"status": "failed", **body, "httpStatus": int(exc.code)}
+        except OSError as exc:
+            return {"status": "unreachable", "error": str(exc), "httpStatus": 0}
+
     # ---- status and log helpers -----------------------------------------
 
     def server_status(self) -> dict[str, Any]:
```

```diff
--- a/python/freetoken/daemon/settings/app.py
+++ b/python/freetoken/daemon/settings/app.py
@@ -543,8 +543,12 @@
 
     @app.post("/api/server/{action}", status_code=202)
     async def server_action(action: str, body: ServerActionBody | None = None):
+        if action in {"sleep", "wake"}:
+            # Sleep keeps the process: no lifecycle job (design section 3.4).
+            result = await run_in_threadpool(process_manager.sleep_server, action)
+            return JSONResponse(result, status_code=int(result.get("httpStatus") or 503))
         if action not in {"start", "stop", "restart"}:
-            raise HTTPException(status_code=422, detail="action must be start, stop, or restart")
+            raise HTTPException(status_code=422, detail="action must be start, stop, restart, sleep or wake")
         body = body or ServerActionBody()
         if body.settings is not None and action == "stop":
             raise HTTPException(status_code=422, detail="settings are accepted only for start or restart")
```

```diff
--- a/python/freetoken/daemon/settings/watchdog.py
+++ b/python/freetoken/daemon/settings/watchdog.py
@@ -155,7 +155,8 @@
         except Exception as exc:  # noqa: BLE001 - unreachable is the signal we watch for
             document = {"state": "unreachable", "error": str(exc)}
         state = document.get("state") if isinstance(document, dict) else "unreachable"
-        if state == "serving":
+        if state in ("serving", "sleeping"):
+            # A sleeping server is alive and holds the model (sleep design section 2.5).
             with self._lock:
                 if not self.armed:
                     logger.info("crash watchdog: adopted a serving model server")
```

```diff
--- a/python/freetoken/daemon/settings/governor.py
+++ b/python/freetoken/daemon/settings/governor.py
@@ -344,6 +344,9 @@
             self._tick_reclaim(enabled=False)
             return
         status = self.process_manager.server_status()
+        if status.get("reachable") and status.get("state") == "sleeping":
+            self._tick_asleep()
+            return
         if not status.get("reachable") or status.get("state") != "serving":
             if status.get("reachable") and status.get("state") == "rebuilding":
                 self._probe_ram()
@@ -413,6 +416,26 @@
                 self.policy.note_step_done(action.axis, time.monotonic())
                 stepped = True
 
+    def _tick_asleep(self) -> None:
+        """While the model sleeps the card belongs to a game: never step VRAM and never recall.
+        Only the RAM axis's down rung runs, and the engine lets it spill to the SSD and nothing
+        else (engine/sleep.py asleep_rebuild). FreeToken keeps about 58-61 GB of Windows RAM
+        while asleep, and on 2026-09-12 Windows memory pressure took the WSL disk down, so a
+        game squeezing RAM must still get expert layers out of its way (sleep design D8)."""
+        self._serving_since = None  # a wake starts a fresh boot-settle window, like a boot
+        free_ram = self._probe_ram()
+        self._tick_reclaim()
+        if free_ram is None:
+            return
+        # "ram" marked exhausted: an up step is neither chosen nor stamped (decide's contract).
+        actions = self.policy.decide(time.monotonic(), 0, free_ram, allow_down=True,
+                                     exhausted_up={"ram"}, axes=("ram",))
+        for action in actions:
+            if action.direction != "down":
+                continue
+            self._execute_action(action, 0, free_ram)
+            self.policy.note_step_done(action.axis, time.monotonic())
+
     def _tick_reclaim(self, *, enabled: bool = True) -> None:
         """Cache probes never call cache/step or alter layer-placement timers."""
         target = (self.policy.ram_cushion + self.policy.ram_rungs_before_up * self.policy.rung_bytes
```

```diff
--- a/python/freetoken/daemon/settings/memory_reclaim.py
+++ b/python/freetoken/daemon/settings/memory_reclaim.py
@@ -437,7 +437,7 @@
                     enabled=(
                         governor.get("enabled", False)
                         and server.get("reachable", False)
-                        and server.get("state") in ("serving", "rebuilding")
+                        and server.get("state") in ("serving", "rebuilding", "sleeping")
                         and os.environ.get("FREETOKEN_CACHE_RECLAIM", "1") != "0"
                     ),
                 )
```

(The `PanelService(...)` wiring in `app.py` needs Task 9's constructor, so it lands in Task 9.)

- [ ] **Step 4: Run** `tests/settings/test_sleep_helper.py` (expect `7 passed`), then
  `tests/settings tests/daemon` against the baseline. `tests/daemon` asserts that torch stays
  out of the daemon's import graph; nothing here imports it.

- [ ] **Step 5: Controller commit** — `feat(helper): sleep and wake routes; watchdog and governor know "sleeping"`

---

### Task 8: `freetoken.sh` adopts a sleeping server

**Files:**
- Modify: `engines/adapters/freetoken.sh`
- Test: `tests/engines/test_adapters_sleep.py`

- [ ] **Step 1: Write the tests**

`tests/engines/test_adapters_sleep.py`:

```python
"""A sleeping FreeToken under llama-swap (review focus 5): adopted by its own model, fully
stopped for any other engine (Jay's rule: switching models while FreeToken sleeps unloads it)."""

from __future__ import annotations

import subprocess
import time

from tests.engines.test_adapters import ADAPTERS, env_for, fake_engine, helper  # noqa: F401 - fixture


def test_freetoken_adopts_its_own_sleeping_server_without_rebooting(helper, tmp_path):
    helper.state, helper.model_path = "sleeping", "/m/B"
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path))
    time.sleep(1.5)
    proc.terminate()
    proc.wait(20)
    assert "POST /api/server/start" not in helper.calls


def test_ninfer_fully_stops_a_sleeping_freetoken_first(helper, tmp_path):
    helper.state = "sleeping"
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "quasar-27b"], env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls and helper.state == "unreachable"
    assert (tmp_path / "argv").exists()
```

- [ ] **Step 2: Run.** Expected, checked in the prototype: the adopt test **fails** on today's
  script (it reboots the sleeping server), and the NInfer test **passes** already (`ninfer.sh`
  stops any FreeToken that is not `unreachable`; the test pins Jay's rule).

- [ ] **Step 3: Implement**

```diff
--- a/engines/adapters/freetoken.sh
+++ b/engines/adapters/freetoken.sh
@@ -146,7 +146,9 @@
   profile_state=$(push_profile) || exit 1
 fi
 adopt=0
-if [ "$state" = serving ] && [ "$job_now" = - ] && ! other_model_loaded; then
+# A sleeping server (it gave the card back, the model is still loaded) counts as running: the
+# first chat wakes it. Rebooting it would throw away the fast wake.
+if { [ "$state" = serving ] || [ "$state" = sleeping ]; } && [ "$job_now" = - ] && ! other_model_loaded; then
   # A running server is kept only if it runs this folder, on this model's profile, with the
   # settings the panel has now; anything else reboots so the new settings take effect.
   if [ -z "$PROFILE" ] || { [ "$(active_profile)" = "$PROFILE" ] && [ "$profile_state" = same ]; }; then
@@ -196,7 +198,7 @@
   else
     gone=0
   fi
-  if [ "$state" = serving ] && other_model_loaded; then
+  if { [ "$state" = serving ] || [ "$state" = sleeping ]; } && other_model_loaded; then
     log "the settings page switched to another model; releasing $MODEL_PATH"
     exit 0
   fi
```

- [ ] **Step 4: Run** `bash -n engines/adapters/freetoken.sh`, then
  `tests/engines/test_adapters_sleep.py tests/engines/test_adapters.py`. Expected: `14 passed`.

- [ ] **Step 5: Controller commit** — `feat(adapters): freetoken.sh adopts a sleeping server`

---

### Task 9: The control panel: Sleep and Wake

**Files:**
- Modify: `python/freetoken/daemon/settings/panel.py`, `static/panel.js`, `static/index.html`
- Test: `tests/settings/test_panel_sleep.py`

**Interfaces:**
- `PanelService(..., freetoken_state=Callable[[], str|None], freetoken_control=Callable[[str], dict])`.
- Rows from `models()` and `now()["switcher"]["running"]` gain `"sleep"`. The value is
  `"asleep"` or `"awake"` for the FreeToken row that llama-swap reports `ready`, and `null`
  for every other row.
- `POST /api/panel/models/{id}/sleep|wake` returns `{"id", "sleep", "result"}`, or a
  `PanelError` with a code of `not_freetoken` (409), `not_loaded` (409), `helper_missing`
  (503), or `<action>_<status>` (409 busy / 504 timeout / 503 otherwise).
- Page: loaded FreeToken rows show **Sleep** (tooltip: "Frees the graphics card for games. The
  model stays in PC memory, so it wakes in about half a minute instead of a full load."), or
  **Wake** when asleep, next to **Unload**. The status reads "Asleep (graphics card free)" with
  an accent-coloured dot.

- [ ] **Step 1: Write the failing tests**

`tests/settings/test_panel_sleep.py`:

```python
"""Sleep and Wake on the control panel (review focus 5): rows, routes and plain words."""

from __future__ import annotations

from tests.settings.test_panel_page import _node
from tests.settings.test_panel_routes import env, seed  # noqa: F401 - the fixture


def _wire(env, state="serving", reply=None):
    calls = []
    env.service._freetoken_state = lambda: state
    env.service._freetoken_control = lambda action: calls.append(action) or (reply or {"status": "ok"})
    return calls


def test_only_the_loaded_freetoken_row_offers_sleep(env):
    seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    _wire(env, state="serving")
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "awake"
    assert rows["qwen3.8-flash-abliterated"]["sleep"] is None and rows["quasar-27b"]["sleep"] is None
    _wire(env, state="sleeping")
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "asleep"
    now = env.client.get("/api/panel/now").json()
    assert now["switcher"]["running"][0]["sleep"] == "asleep"


def test_sleep_and_wake_go_to_the_helper_not_the_switcher(env):
    seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    calls = _wire(env)
    assert env.client.post("/api/panel/models/qwen3.8-flash/sleep").json()["sleep"] == "asleep"
    assert env.client.post("/api/panel/models/qwen3.8-flash/wake").json()["sleep"] == "awake"
    assert calls == ["sleep", "wake"] and env.switcher.calls == []


def test_sleep_refusals_speak_plainly(env):
    seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    _wire(env)
    r = env.client.post("/api/panel/models/quasar-27b/sleep")
    assert r.status_code == 409 and r.json()["code"] == "not_freetoken"
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 409 and r.json()["code"] == "not_loaded"
    env.switcher.states = {"qwen3.8-flash": "ready"}
    _wire(env, reply={"status": "busy"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 409 and "chat is still running" in r.json()["message"]
    _wire(env, reply={"status": "rejected", "error": "the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/wake")
    assert r.status_code == 503 and "close the game" in r.json()["message"]


def test_the_page_words_and_buttons():
    _node(r"""
assert.equal(p.stateWord(p.rowState({state: 'ready', sleep: 'asleep'})), 'Asleep (graphics card free)');
assert.equal(p.stateWord(p.rowState({state: 'ready', sleep: 'awake'})), 'Loaded');
const rows = [
  {id: 'qwen3.8-flash', name: 'Q', state: 'ready', sleep: 'asleep', engineLabel: 'FreeToken', ramNeedGB: 62},
  {id: 'quasar-27b', name: 'QS', state: 'stopped', sleep: null, engineLabel: 'NInfer', ramNeedGB: 18},
];
const html = p.modelsTableHtml(rows, true);
assert.ok(html.includes('data-wake="qwen3.8-flash"') && html.includes('data-unload="qwen3.8-flash"'));
assert.ok(html.includes('dot asleep') && !html.includes('data-sleep='));
const awake = p.modelsTableHtml([{...rows[0], sleep: 'awake'}], true);
assert.ok(awake.includes('data-sleep="qwen3.8-flash"') && awake.includes('Frees the graphics card for games'));
assert.equal(p.sleepButtonHtml(rows[1]), '');
""")
```

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement**

```diff
--- a/python/freetoken/daemon/settings/panel.py
+++ b/python/freetoken/daemon/settings/panel.py
@@ -193,7 +193,8 @@
     def __init__(self, *, store, writer, switcher, profiles, boot_file: Callable[[], BootFile],
                  default_boot: Callable[[], Path], estimate_service, card_probe=None, windows_free_probe=None,
                  holds_path=None, artifact_size=None, spawn=None, clock=time.monotonic, sleep=time.sleep,
-                 restart_wait_s: float = 30.0) -> None:
+                 restart_wait_s: float = 30.0, freetoken_state: Callable[[], Any] | None = None,
+                 freetoken_control: Callable[[str], dict] | None = None) -> None:
         self.store, self.writer, self.switcher, self.profiles = store, writer, switcher, profiles
         self._boot_file, self._default_boot, self.estimate_service = boot_file, default_boot, estimate_service
         self._card_probe = card_probe or _default_card_probe
@@ -202,6 +203,10 @@
         self._artifact_size = artifact_size or os.path.getsize
         self._spawn = spawn or _default_spawn
         self._clock, self._sleep, self.restart_wait_s = clock, sleep, restart_wait_s
+        # Sleep (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md): the FreeToken
+        # server's public state word and its /v1/sleep|/v1/wake, through the helper.
+        self._freetoken_state = freetoken_state
+        self._freetoken_control = freetoken_control
         self._lock = threading.RLock()
         self._stop = threading.Event()
         self._watcher: threading.Thread | None = None
@@ -761,15 +766,31 @@
         text = self.writer.current_text()
         return live is not None and text is not None and live != config_sha256(text)
 
+    def _sleep_word(self) -> str:
+        """"asleep" or "awake" for the loaded FreeToken model (one server, one port)."""
+        try:
+            state = self._freetoken_state() if self._freetoken_state is not None else None
+        except Exception:  # noqa: BLE001 - an unreachable helper probe reads as awake
+            state = None
+        return "asleep" if state == "sleeping" else "awake"
+
+    def _mark_sleep(self, rows: list[dict[str, Any]], engines: Mapping[str, str]) -> None:
+        loaded = [row for row in rows if engines.get(row["id"]) == "freetoken" and row.get("state") == "ready"]
+        word = self._sleep_word() if loaded else None
+        for row in rows:
+            row["sleep"] = word if row in loaded else None
+
     def now(self) -> dict[str, Any]:
         running = self.switcher.running()
         up = running is not None and not is_down(running)
         try:
             doc, _ = self.store.load()
             names, floor = {m["id"]: m["name"] for m in doc["models"]}, doc["system"]["floorGB"]
+            engines = {m["id"]: m["engine"] for m in doc["models"]}
         except RegistryError:
-            names, floor = {}, None
+            names, floor, engines = {}, None, {}
         rows = [{"id": m, "name": names.get(m, m), "state": s} for m, s in sorted((running or {}).items())]
+        self._mark_sleep(rows, engines)
         last = self.last_restart
         if last is not None and self._last_restart_at is not None and self._clock() - self._last_restart_at >= RESTART_SHOWN_S:
             last = None
@@ -794,8 +815,33 @@
                 "state": running.get(model["id"], "stopped") if up else "unknown",
                 "held": model["id"] in holds,
             })
+        self._mark_sleep(rows, {m["id"]: m["engine"] for m in doc["models"]})
         return {"revision": revision, "switcherUp": up, "models": rows}
 
+    def sleep_model(self, model_id: str, action: str) -> dict[str, Any]:
+        """Sleep or wake the loaded FreeToken model. Sleep keeps it loaded (llama-swap still
+        says "ready"), so this goes to the helper, not the switcher."""
+        doc, _ = self.store.load()
+        model = find_model(doc, model_id)
+        if model["engine"] != "freetoken":
+            raise PanelError(409, "not_freetoken", "Only FreeToken models can sleep.")
+        running = self.switcher.running()
+        if running is None or is_down(running) or running.get(model_id) != "ready":
+            raise PanelError(409, "not_loaded", f"{model['name']} is not loaded, so there is nothing to {action}.")
+        if self._freetoken_control is None:
+            raise PanelError(503, "helper_missing", "The settings helper cannot reach FreeToken.")
+        result = self._freetoken_control(action)
+        status = result.get("status")
+        if status == "ok":
+            return {"id": model_id, "sleep": "asleep" if action == "sleep" else "awake", "result": result}
+        if status == "busy":
+            message = "A chat is still running. Try again when it finishes."
+        elif status == "unreachable":
+            message = "FreeToken is not answering."
+        else:
+            message = str(result.get("error") or f"Could not {action} {model['name']}.")
+        raise PanelError({"busy": 409, "timeout": 504}.get(status, 503), f"{action}_{status or 'failed'}", message)
+
     def load(self, model_id: str) -> dict[str, Any]:
         doc, _ = self.store.load()
         find_model(doc, model_id)
@@ -997,6 +1043,14 @@
     async def unload(model_id: str):
         return await call(service.unload, model_id)
 
+    @router.post("/models/{model_id}/sleep")
+    async def sleep_model(model_id: str):
+        return await call(service.sleep_model, model_id, "sleep")
+
+    @router.post("/models/{model_id}/wake")
+    async def wake_model(model_id: str):
+        return await call(service.sleep_model, model_id, "wake")
+
     @router.get("/models/{model_id}/effective")
     async def effective(model_id: str):
         return await call(service.effective, model_id)
```

```diff
--- a/python/freetoken/daemon/settings/static/panel.js
+++ b/python/freetoken/daemon/settings/static/panel.js
@@ -11,8 +11,17 @@
 
 function panelEsc(value) { return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
 function fmtGB(bytes) { const n = Number(bytes); if (bytes == null || bytes === '' || !Number.isFinite(n)) return '—'; return `${(n / PANEL_GB).toFixed(1)} GB`; }
-const STATE_WORDS = { ready: 'Loaded', starting: 'Loading…', stopping: 'Unloading…', stopped: 'Not loaded', shutdown: 'Not loaded', unknown: 'Switcher not running' };
+const STATE_WORDS = { asleep: 'Asleep (graphics card free)', ready: 'Loaded', starting: 'Loading…', stopping: 'Unloading…', stopped: 'Not loaded', shutdown: 'Not loaded', unknown: 'Switcher not running' };
 function stateWord(value) { return STATE_WORDS[value] || String(value || 'unknown'); }
+// A sleeping FreeToken is still "ready" to the switcher; the page says what Jay needs to know.
+function rowState(row) { return row && row.sleep === 'asleep' ? 'asleep' : (row && row.state); }
+const SLEEP_TIP = 'Frees the graphics card for games. The model stays in PC memory, so it wakes in about half a minute instead of a full load.';
+const WAKE_TIP = 'Takes the graphics card back. A chat also wakes it by itself.';
+function sleepButtonHtml(row) {
+  if (row.sleep === 'asleep') return `<button class="button small primary" type="button" data-wake="${panelEsc(row.id)}" title="${panelEsc(WAKE_TIP)}">Wake</button>`;
+  if (row.sleep === 'awake') return `<button class="button small" type="button" data-sleep="${panelEsc(row.id)}" title="${panelEsc(SLEEP_TIP)}">Sleep</button>`;
+  return '';
+}
 function sourceText(source) {
   if (!source) return '';
   if (source.from === 'model') return 'changed for this model';
@@ -68,7 +77,7 @@
 function nowStripHtml(now) {
   if (!now) return '<p class="empty">Checking…</p>';
   const sw = now.switcher || {};
-  const running = !sw.up ? 'Model switcher not running' : ((sw.running || []).length ? sw.running.map((row) => `${panelEsc(row.name || row.id)} <span class="small">${panelEsc(stateWord(row.state))}</span>`).join('<br>') : 'Nothing loaded');
+  const running = !sw.up ? 'Model switcher not running' : ((sw.running || []).length ? sw.running.map((row) => `${panelEsc(row.name || row.id)} <span class="small">${panelEsc(stateWord(rowState(row)))}</span>`).join('<br>') : 'Nothing loaded');
   const card = now.card ? `${fmtGB(now.card.usedBytes)} <span class="muted">/ ${fmtGB(now.card.totalBytes)}</span>` : '—';
   const pct = now.card && now.card.totalBytes ? Math.round(Math.min(100, (now.card.usedBytes / now.card.totalBytes) * 100)) : 0;
   const win = now.windowsFreeBytes == null ? '—' : fmtGB(now.windowsFreeBytes);
@@ -89,8 +98,8 @@
     const loaded = row.state === 'ready' || row.state === 'starting';
     const engine = row.runtimeLabel ? `${panelEsc(row.engineLabel)} <span class="small">(${panelEsc(row.runtimeLabel)})</span>` : panelEsc(row.engineLabel);
     const idle = row.idleMinutes ? `${panelEsc(row.idleMinutes)} min` : 'never';
-    const action = !switcherUp ? '' : loaded ? `<button class="button small danger" type="button" data-unload="${panelEsc(row.id)}">Unload</button>` : `<button class="button small primary" type="button" data-load="${panelEsc(row.id)}">Load</button>`;
-    return `<tr><td><span class="dot ${panelEsc(row.state)}"></span> ${panelEsc(stateWord(row.state))}${row.held ? '<div class="small">old settings until the next load</div>' : ''}</td><td><strong>${panelEsc(row.name)}</strong><div class="small">${panelEsc(row.id)}</div></td><td data-label="Engine">${engine}</td><td data-label="Preset">${panelEsc(row.activePreset || '—')}</td><td data-label="PC memory">${panelEsc(row.ramNeedGB)} GB</td><td data-label="Unload when idle">${idle}${row.idleFromSystem ? ' <span class="small">(System)</span>' : ''}</td><td class="row-actions"><div class="actions">${action}<button class="button small" type="button" data-settings="${panelEsc(row.id)}">Settings</button></div></td></tr>`;
+    const action = !switcherUp ? '' : loaded ? `${sleepButtonHtml(row)}<button class="button small danger" type="button" data-unload="${panelEsc(row.id)}">Unload</button>` : `<button class="button small primary" type="button" data-load="${panelEsc(row.id)}">Load</button>`;
+    return `<tr><td><span class="dot ${panelEsc(rowState(row))}"></span> ${panelEsc(stateWord(rowState(row)))}${row.held ? '<div class="small">old settings until the next load</div>' : ''}</td><td><strong>${panelEsc(row.name)}</strong><div class="small">${panelEsc(row.id)}</div></td><td data-label="Engine">${engine}</td><td data-label="Preset">${panelEsc(row.activePreset || '—')}</td><td data-label="PC memory">${panelEsc(row.ramNeedGB)} GB</td><td data-label="Unload when idle">${idle}${row.idleFromSystem ? ' <span class="small">(System)</span>' : ''}</td><td class="row-actions"><div class="actions">${action}<button class="button small" type="button" data-settings="${panelEsc(row.id)}">Settings</button></div></td></tr>`;
   }).join('');
   return `<table class="models-table"><thead><tr><th>Status</th><th>Model</th><th>Engine</th><th>Preset</th><th>PC memory</th><th>Unload when idle</th><th></th></tr></thead><tbody>${body}</tbody></table>`;
 }
@@ -122,7 +131,7 @@
   }
   return `<div class="panel-head"><div><h2>The model list is damaged</h2><p class="small">${panelEsc(body.message || '')}</p><p class="small">Nothing was changed. Restore the last good backup:</p></div></div>${backups ? `<ul class="backups">${backups}</ul>` : '<p class="empty">No backups were found.</p>'}`;
 }
-if (typeof module !== 'undefined') module.exports = { fmtGB, stateWord, sourceText, dialSourceFor, verdictWords, fitSummary, ramSummary, restartQuestion, nowStripHtml, modelsTableHtml, panelErrorText,
+if (typeof module !== 'undefined') module.exports = { fmtGB, stateWord, rowState, sleepButtonHtml, sleepModel, sourceText, dialSourceFor, verdictWords, fitSummary, ramSummary, restartQuestion, nowStripHtml, modelsTableHtml, panelErrorText,
   loadQuestion, unloadQuestion, registryProblemHtml, panelSave, answerRestart, answerConfirm, startNow, loadModel, unloadModel, panel };
 
 /* ---------- browser side: uses index.html's state, json, $, setNotice and dial renderer ---------- */
@@ -245,6 +254,8 @@
   list.innerHTML = modelsTableHtml(panel.models, body.switcherUp);
   list.querySelectorAll('[data-load]').forEach((button) => button.addEventListener('click', () => loadModel(button.dataset.load)));
   list.querySelectorAll('[data-unload]').forEach((button) => button.addEventListener('click', () => unloadModel(button.dataset.unload)));
+  list.querySelectorAll('[data-sleep]').forEach((button) => button.addEventListener('click', () => sleepModel(button.dataset.sleep, 'sleep')));
+  list.querySelectorAll('[data-wake]').forEach((button) => button.addEventListener('click', () => sleepModel(button.dataset.wake, 'wake')));
   list.querySelectorAll('[data-settings]').forEach((button) => button.addEventListener('click', () => openModel(button.dataset.settings)));
 }
 async function loadModel(id) {
@@ -256,6 +267,13 @@
   if (response.ok) setNotice(`${id} is loaded.`, 'good'); else setNotice(panelErrorText(body, `Could not load ${id}.`), 'bad');
   loadNow(); loadModels();
 }
+async function sleepModel(id, action) {
+  setNotice(action === 'sleep' ? `Putting ${id} to sleep… the graphics card frees up in a few seconds.` : `Waking ${id}… about half a minute.`);
+  const { response, body } = await json(`/api/panel/models/${encodeURIComponent(id)}/${action}`, { method: 'POST' });
+  if (response.ok) setNotice(action === 'sleep' ? `${id} is asleep. The graphics card is free; a chat wakes it.` : `${id} is awake.`, 'good');
+  else setNotice(panelErrorText(body, `Could not ${action} ${id}.`), 'bad');
+  loadNow(); loadModels();
+}
 async function unloadModel(id) {
   if (!(await askConfirm(unloadQuestion(id, panel.models), 'Put it away'))) return;
   setNotice(`Unloading ${id}…`);
```

```diff
--- a/python/freetoken/daemon/settings/static/index.html
+++ b/python/freetoken/daemon/settings/static/index.html
@@ -254,7 +254,7 @@
     .source { color: var(--muted); margin-top: 4px; } .source.changed-here { color: var(--accent); }
     .linkish { border: 0; background: none; color: var(--accent); padding: 0; text-decoration: underline; }
     .backups { list-style: none; padding: 0; display: grid; gap: 6px; }
-    .dot.ready { background: var(--good); } .dot.starting, .dot.stopping { background: var(--warn); animation: pulse 1.2s infinite; } .dot.unknown { background: var(--bad); }
+    .dot.ready { background: var(--good); } .dot.asleep { background: var(--accent); } .dot.starting, .dot.stopping { background: var(--warn); animation: pulse 1.2s infinite; } .dot.unknown { background: var(--bad); }
     @media (max-width: 640px) { #now.status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .models-table td.row-actions .actions { justify-content: flex-start; } .models-table thead { display: none; } .models-table td { display: block; border: 0; padding: 4px 0; } .models-table tr { display: block; border-bottom: 1px solid var(--line); padding: 8px 0; } .models-table td[data-label]::before { content: attr(data-label) ": "; color: var(--muted); font-size: .84rem; } }
 
     @media (max-width: 900px) { .top-right { align-items: flex-start; } }
```

Wire the panel to the helper in `create_app` (the `getattr` is load-bearing, see Task 7):

```diff
--- a/python/freetoken/daemon/settings/app.py
+++ b/python/freetoken/daemon/settings/app.py
@@ -234,6 +234,9 @@
             boot_file=lambda: app.state.boot_file,
             default_boot=lambda: app.state.default_boot_file,
             estimate_service=app.state.estimate_service,
+            freetoken_state=lambda: process_manager.server_status().get("state"),
+            # getattr: route tests hand in recording managers that predate sleep
+            freetoken_control=getattr(process_manager, "sleep_server", None),
         )
     app.state.panel = panel
     app.include_router(create_panel_router(panel))
```

- [ ] **Step 4: Run** `tests/settings/test_panel_sleep.py tests/settings/test_panel_routes.py tests/settings/test_panel_page.py`. Expected: all pass (`4 passed` new).

- [ ] **Step 5: Controller commit** — `feat(panel): Sleep and Wake buttons for the loaded FreeToken model`

---

### Task 10: Docs

**Files:** `README.md` (a new fork subsection after "KV prefix parking"), `CONTEXT.md` (glossary),
`engines/adapters/CONTRACT.md` (one paragraph).

- [ ] **Step 1: README subsection** (insert before `### Still-picture (vision) serving`):

```markdown
### Sleep (free the graphics card, keep the model loaded)

`POST /v1/sleep` gives the card back for a game while the model stays in PC memory: the
GPU-owned expert layers go to the SSD expert copy, the shared slot cache, KV pool, GDN state,
CUDA graphs and the MTP draft head are freed, and conversations are parked to RAM first. The
dense weights stay on the card (phase 1). The next chat, or `POST /v1/wake`, brings it back;
a wake is refused, and the model stays asleep, while another program holds the card.
`/health`, `/ready` and `/v1/cache/status` report `sleeping`. The control panel shows
**Sleep** / **Wake** on the loaded FreeToken model; llama-swap still sees it as loaded, and
loading any other model unloads it fully. While asleep the memory governor only spills expert
layers to the SSD under Windows RAM pressure. Design and measured numbers:
`docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md`,
`docs/research/freetoken-sleep-acceptance-*.md`.
```

- [ ] **Step 2: CONTEXT.md entry** (append):

```markdown
## Sleep

FreeToken giving the graphics card back while the model stays in PC memory. The model is still
"loaded" (the switcher says ready); a chat or Wake brings it back in about half a minute instead
of a two-and-a-half-minute load. Unload is still the way to free the PC memory too.

_Avoid:_ suspend, hibernate, unload
```

- [ ] **Step 3: CONTRACT.md**: add after the `--profile` paragraph: "FreeToken's `sleeping`
  state (the card given back, the model still loaded) counts as running this model: the adapter
  adopts it, and any other adapter stops it fully before starting (`ninfer.sh` treats every
  state but `unreachable` as holding the card)."

- [ ] **Step 4: Controller commit** — `docs: sleep in the README, glossary and adapter contract`

---

### Task 11: The bench script and the GPU test file

**Files:**
- Create: `scripts/bench/sleep_bench.py`, `tests/engine/test_sleep_gpu.py`

- [ ] **Step 1: Create the bench** (stdlib only; runs on the box against the live server)

`scripts/bench/sleep_bench.py`:

```python
#!/usr/bin/env python3
"""Sleep/wake acceptance bench for a live FreeToken server (stdlib only; run on the serving box).

Per cycle: two greedy chats awake (A1 cold, A2 warm), POST /v1/sleep, read the card, then a
greedy chat that wakes the model by itself (B, timed end to end), then sleep and an explicit
POST /v1/wake (timed). Pass when B equals A1 or A2 token for token, every sleep and wake says
ok, and the card reading while asleep is under --max-asleep-gib above --baseline-gib.

    python3 scripts/bench/sleep_bench.py --cycles 3 --baseline-gib 2.1 --out ~/sleep-bench.json

Design: docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md (section 1 criteria).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = (
    "List the first twelve prime numbers, then explain in three sentences why there are "
    "infinitely many primes."
)


def call(base: str, path: str, body: dict | None = None, timeout: float = 600.0) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def card_used_gib() -> float | None:
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return round(int(out.split()[0]) / 1024, 2)
    except Exception:  # noqa: BLE001 - the bench still reports the server's own numbers
        return None


def chat(base: str, model: str, tokens: int) -> tuple[str, float]:
    t0 = time.monotonic()
    status, body = call(base, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0, "stream": False,
    })
    if status != 200:
        raise SystemExit(f"chat failed: HTTP {status} {body}")
    return body["choices"][0]["message"]["content"], round(time.monotonic() - t0, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:2020")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--baseline-gib", type=float, required=True,
                    help="card used with FreeToken stopped (desktop only), measured first")
    ap.add_argument("--max-asleep-gib", type=float, default=7.5)
    ap.add_argument("--max-wake-s", type=float, default=45.0)
    ap.add_argument("--out", default="sleep-bench.json")
    args = ap.parse_args()
    _, health = call(args.server, "/health")
    model = health.get("model") or "default"
    cycles, ok = [], True
    for n in range(args.cycles):
        a1, a1_s = chat(args.server, model, args.tokens)
        a2, a2_s = chat(args.server, model, args.tokens)
        awake_gib = card_used_gib()
        t0 = time.monotonic()
        s_status, s_body = call(args.server, "/v1/sleep", {})
        sleep_s = round(time.monotonic() - t0, 2)
        time.sleep(3)  # let WSL/WDDM hand the freed memory back before reading the card
        asleep_gib = card_used_gib()
        _, status_doc = call(args.server, "/v1/cache/status")
        b, b_s = chat(args.server, model, args.tokens)  # wakes the model by itself
        call(args.server, "/v1/sleep", {})
        t0 = time.monotonic()
        w_status, w_body = call(args.server, "/v1/wake", {})
        wake_s = round(time.monotonic() - t0, 2)
        row = {
            "cycle": n + 1, "same_output": b in (a1, a2), "a1_s": a1_s, "a2_s": a2_s,
            "sleep_http": s_status, "sleep_s": sleep_s, "sleep_reply": s_body,
            "state_asleep": status_doc.get("state"), "card_awake_gib": awake_gib,
            "card_asleep_gib": asleep_gib, "chat_after_sleep_s": b_s,
            "wake_http": w_status, "wake_s": wake_s, "wake_reply": w_body,
        }
        row["pass"] = bool(
            row["same_output"] and s_status == 200 and w_status == 200
            and row["state_asleep"] == "sleeping" and wake_s <= args.max_wake_s
            and (asleep_gib is None or asleep_gib - args.baseline_gib <= args.max_asleep_gib)
        )
        ok &= row["pass"]
        cycles.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_reply")}), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": model, "baseline_gib": args.baseline_gib, "cycles": cycles, "pass": ok}, fh, indent=2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create the GPU tests** (skip without CUDA)

`tests/engine/test_sleep_gpu.py`:

```python
"""Sleep on a real card (the serving box; skips without CUDA). Run by the controller."""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (serving box)")

E, SLOTS = 64, 2048
ROW = 256 * 256 * 2  # one bf16 [256, 256] row


def _cuda_cache() -> OffloadMoeCache:
    cache = OffloadMoeCache(num_layers=4, num_experts=E, cache_size=SLOTS, device=torch.device("cuda"))
    sources = {
        name: [torch.randn(E, 256, 256, dtype=torch.bfloat16).pin_memory() for _ in range(4)]
        for name in ("gate_up", "down")
    }
    cache.set_bank_sources(sources, layer_residency=["pinned"] * 4, gpu_owned_layers=frozenset())
    return cache


def _free() -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def test_release_slots_hands_the_bytes_back_to_the_card():
    cache = _cuda_cache()
    before = _free()
    freed = cache.release_slots()
    assert freed == SLOTS * ROW * 2  # two banks
    # expandable segments unmap on empty_cache; allow 5% for allocator rounding
    assert _free() - before >= 0.95 * freed


def test_a_released_cache_rebuilds_to_the_same_size_and_copy_plan():
    cache = _cuda_cache()
    fused = cache._copy_fused_ok
    before = _free()
    cache.release_slots()
    cache.rebuild(SLOTS)
    assert cache.cache_size == SLOTS and cache.bank_caches["gate_up"].shape == (SLOTS, 256, 256)
    assert cache._copy_fused_ok == fused
    assert abs(_free() - before) <= 64 << 20
```

- [ ] **Step 3: Devbox checks.** `python3 -m py_compile scripts/bench/sleep_bench.py`. The
  pytest run gives `2 skipped`. For a smoke test of the bench, run it against a tiny stdlib fake
  server that answers `/health`, `/v1/chat/completions`, `/v1/sleep`, `/v1/wake` and
  `/v1/cache/status`. The prototype did this and printed one cycle row, then `PASS`.

- [ ] **Step 4: Full devbox suite against the baseline** (Task 0 command). Expected: `comm`
  prints nothing.

- [ ] **Step 5: Controller commit** — `test: sleep GPU checks and the live sleep bench`

---

### Task 12: GPU tests on the box (controller)

- [ ] **Step 1: Push the branch** (`git push origin feat/freetoken-sleep`).
- [ ] **Step 2: Check the box is free** (Global Constraints), then put the box checkout on the
  branch. This changes no running process; the server only picks up the code on its next boot.

```bash
ssh 5090 'wsl -d vllm -e bash -l' <<'EOF'
cd ~/FreeToken && git fetch origin && git status --short && git switch feat/freetoken-sleep && git pull --ff-only
EOF
```

- [ ] **Step 3: Run the GPU tests and the GPU-relevant suites.** The card must have about 1 GB
  free; `test_sleep_gpu` allocates 512 MiB. If FreeToken is serving and the card is full, run
  this in Task 13 while the model is asleep.

```bash
ssh 5090 'wsl -d vllm -e bash -l' <<'EOF'
cd ~/FreeToken && PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_sleep_gpu.py \
  tests/engine/test_engine_sleep.py tests/moe/test_offload_release_slots.py tests/moe/test_disk_banks.py -q
EOF
```
Expected: all pass, with `test_sleep_gpu.py` **not** skipped (2 passed). A free-memory
assertion that fails means WSL is not returning freed VRAM. Record the numbers; that is a
finding for the design's risk table, not something to paper over.

- [ ] **Step 4: Record the output** in the acceptance doc (Task 13).

---

### Task 13: Live acceptance on the box, with screenshots (controller)

The box checkout is on `feat/freetoken-sleep`. **Announce each boot.** At most 3 FreeToken boots
this session. Watch Windows free RAM through each boot, and never evict a model Jay is using.

- [ ] **Step 1: Baselines (no boot).** With FreeToken unloaded (through the panel or
  llama-swap unload, only if nothing is in use): card used is `BASE` GiB,
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` (and Windows' Task Manager
  reading). Also note the cold-boot phases from the last boot's `logs/server-2020.log`
  timestamps: expert load start and end, weights done, `Free memory after initialization`,
  graph capture, MTP graphs, serving.
- [ ] **Step 2: Load the new code (boot 1, MTP off, the daily profile otherwise).**
  `systemctl --user restart freetoken-settings` (helper code changed), then load
  `qwen3.8-flash` through the panel. Record the boot time and the Windows free-RAM minimum.
- [ ] **Step 3: The bench, MTP off:**
  `python3 scripts/bench/sleep_bench.py --cycles 3 --baseline-gib $BASE --out ~/sleep-bench-mtp-off.json`.
  Pass means every cycle has `same_output`, `wake_s ≤ 45` and card used while asleep
  `≤ BASE + 7.5 GiB`. The target is `wake_s ≤ 30`.
- [ ] **Step 4: Screenshots of the panel** with `~/.npm-global/bin/chrome-devtools-axi` against
  `https://5090.tail45ff04.ts.net`, Models tab: (a) awake with the Sleep button, (b) after Sleep:
  "Asleep (graphics card free)", the Wake button, and the Right-now card reading, (c) during Wake
  (the "Waking…" notice), (d) awake again. Save them to `docs/research/img/sleep-*.png`.
- [ ] **Step 5: Auto-wake and refusals:**
  - Sleep, then send a chat through llama-swap (`:2040`) and through tailnet `:12040`. It must
    answer. Record the time to first token.
  - **Card busy:** sleep, then hold VRAM with a throwaway
    `python -c "import torch,time; x=torch.empty(int(27e9),dtype=torch.uint8,device='cuda'); time.sleep(120)"`.
    A chat must get a plain 503 ("close the game…"), `/health` must stay `ok`/`sleeping`, and
    after the hog exits a chat wakes the model.
  - **RAM squeeze while asleep:** sleep, grab 8 GB of Windows RAM (the existing `grab.py` from
    the governor tests, 60 s). The helper log must show `governor: ram down -> pinned->disk` and
    no VRAM step. After the wake that layer stays on the SSD until the governor recalls it.
  - **Switch while asleep:** sleep, then load `quasar-27b` in the panel. FreeToken must unload
    fully (the helper stops it; card and RAM freed), and QUASAR loads through the P2 gate.
    Afterwards restore what Jay had loaded. **Boot 2** if that was FreeToken.
- [ ] **Step 6: MTP on (boot 3, only if the box's daily shape is MTP on, or with Jay's OK):**
  repeat Step 3 as `sleep-bench-mtp-on.json`. It is also a pass if the wake time differs, as
  long as it stays within 45 s.
- [ ] **Step 7: Fault watch.** Over the session, the Windows System log must show no new
  `nvlddmkm` event 153:
  `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=(Get-Date).AddHours(-3)}`.
- [ ] **Step 8: Write `docs/research/freetoken-sleep-acceptance-<date>.md`.** Include a table of
  each check with pass or fail and its evidence, the bench JSON summaries, the card readings,
  boot versus wake times, the RAM minimums, the screenshots, and every deviation. Controller
  commit: `docs: live acceptance of FreeToken sleep`.

---

### Task 14: Codex review (Astra, xhigh) and fixes

- [ ] **Step 1:** Run the `codex-review` skill on `git diff mtp-upstream-merge...feat/freetoken-sleep`
  with the Astra slug at xhigh (look the slug up in the `codex-models` skill). The prompt names
  the five Review Focus items and the design doc.
- [ ] **Step 2:** List every finding in one table: accept or reject, and why. Fix the accepted
  ones test-first (an implementer subagent, no git writes). Re-run the devbox suite against the
  baseline, and the GPU tests on the box if engine code changed.
- [ ] **Step 3:** If the fixes touched engine, scheduler or API code, re-run one bench cycle on the
  box. That needs a server restart, which counts toward the 3-boot budget; if the budget is
  spent, run the cycle in the next session and note it in the acceptance doc.
- [ ] **Step 4:** Second round only if round 1 had accepted findings in engine, scheduler or API
  code. Two rounds at most.
- [ ] **Step 5: Controller commit** — `fix: <what>` per logical fix.

---

### Task 15: PR and merge (controller)

- [ ] **Step 1:** Open the PR against `mtp-upstream-merge` with the title
  `feat: sleep (free the graphics card, keep the model in memory)`. The body carries the
  summary, the measured numbers from the acceptance doc, the test list, and the known
  limitations: phase 2 not done, and Stop during a game can wait up to 120 s for VRAM, as
  today. End the body with the attribution line.
- [ ] **Step 2:** Squash-merge once the acceptance doc says PASS and the review is closed.
  Put the box back on `mtp-upstream-merge`:
  `ssh 5090 'wsl -d vllm -e bash -l' <<< 'cd ~/FreeToken && git switch mtp-upstream-merge && git pull --ff-only'`.
  Restart the helper (`systemctl --user restart freetoken-settings`) and reload whatever Jay had
  loaded. Do not boot FreeToken if it was not loaded.
- [ ] **Step 3:** Update the memory index (a `project-freetoken-sleep` note: the numbers, what is
  live, and phase 2 as the follow-up).

## Self-review against the spec

- Spec 2.1 (inventory): Task 3 releases every freeable item except the dense weights (phase 2);
  Task 12/13 measure what is left.
- Spec 3.1 D1-D12: D1/D2/D3/D4/D5 are in Task 3. D6/D7 are in Tasks 3, 5 and 6. D8 is in Tasks 3,
  5 and 7. D9 is in Task 8 (no llama-swap change). D10: only manual sleep plus auto-wake. D11
  is in Tasks 3 and 5. D12 is in Task 3 (`check_can_sleep`).
- Spec 3.2/3.3 order is `release_to_sleep` / `_restore`, pinned by the Task 3 tests.
- Spec 3.4 surfaces are in Tasks 4-9.
- Spec 5 risks are pinned by the Review Focus tests plus Task 13 Step 5 (card busy, RAM squeeze,
  switch while asleep) and Step 7 (fault watch).
- Spec 6 test plan is Tasks 1-11 (devbox), 12 (box GPU) and 13 (live).
