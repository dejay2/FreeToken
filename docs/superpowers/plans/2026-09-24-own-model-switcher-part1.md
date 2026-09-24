# Own Model Switcher, Part 1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze llama-swap and NInfer into this repo and give llama-swap our own rules:
latest pick wins, wait for memory, clamp engine limits. Add standard engine adapters, one
build script and one service, then switch the RTX 5090 box over from the downloaded binary.

**Architecture:**
- Plain source copies live under `engines/`. Each has a `FROZEN.md` that records the source
  commit and every patch.
- The llama-swap patches are small, commented `FreeToken patch Pn`, and tested next to
  upstream's own tests.
- Engines are started by adapter scripts under `engines/adapters/`, which follow one
  contract. `scripts/engines/build.sh` builds everything; `install-service.sh` points the
  existing `llama-swap` systemd --user unit at the new binary.

**Tech Stack:** Go 1.27.1 (llama-swap), Svelte/Vite UI (Node, built by `make ui`), C++/CUDA
13.x with CMake/Ninja (NInfer), bash plus Python pytest for the adapter tests, systemd --user
in WSL `vllm`.

**Spec:** `docs/superpowers/specs/2026-09-24-own-model-switcher-part1-design.md`

## Global Constraints

- llama-swap is frozen at tag v257, commit `f00d375a927f72d72e74ca86012be210f155c67b` (MIT).
  Keep `LICENSE.md`.
- NInfer candidates:
  - `MirkoCovizzi/ninfer-rtx5090-mobile` at `d4bc75dbc7066109c3d9692ed564e5904a849ba0` (Apache-2.0);
  - `Neroued/ninfer` at `f76e19c0`, the build on the box today (Apache-2.0).
- Go toolchain 1.27.1, as `go.mod` declares.
- Ports stay as they are:
  - llama-swap listens on `127.0.0.1:2040`; tailnet `12020` → `127.0.0.1:2040` is unchanged;
  - FreeToken helper `127.0.0.1:2031`, FreeToken server `127.0.0.1:2020`, NInfer `127.0.0.1:8090`.
- Defaults:
  - `latestWins: true`;
  - memory gate `floorGB: 6`, `waitSeconds: 300`, poll every 5 s, probe cache 2 s;
  - the gate is off unless `memoryGate.probe: windows` is set.
- Every change inside `engines/llama-swap/` carries a `// FreeToken patch Pn:` comment and is
  listed in `engines/llama-swap/FROZEN.md`.
- The live config stays at `~/llama-swap/config.yaml` on the box and is never committed.
- Model IDs, aliases and Pi entries are unchanged: `qwen3.8-flash`, `qwen3.8-flash-abliterated`
  (aliases `Qwen3.8-Flash-Next-NVFP4`, `Qwen3.8-Flash-Next-ABLITERATED-NVFP4`), `fable-27b`,
  `twin-27b`, `quasar-27b`.
- **Box rules:**
  - Ask Jay before any test that loads a model on the box.
  - Check `curl 127.0.0.1:2040/running` first, and never evict a model Jay is using.
  - At most one FreeToken boot per live session; sample Windows free RAM during it.
- **Git rules:**
  - Subagents run no git write commands; the controller commits.
  - Never force-push.
  - Work on branch `feat/own-switcher` off `mtp-upstream-merge`.
- Commits follow `type: subject`, and every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Three quick picks (A, then B, then C) while A loads.** Only C ends up loading. A's and B's
   callers get 409, and no process is left starting. Pinned in Task 3.
2. **The same model requested again while it loads.** It joins the swap and must never cancel
   itself. Pinned in Task 3.
3. **A request that arrives after the target became ready but before `SwapDone` was processed.**
   That swap is not cancelled. Pinned in Task 3.
4. **The memory probe fails, returns garbage or hangs.** The load goes ahead after the probe's
   own 10 s timeout. A superseding pick during the memory wait ends the wait at once. Pinned in
   Task 4.
5. **Non-numeric or missing params under `clampParams`** (for example `"top_k": "40"`, or no
   `top_k`). They are left untouched and the request still succeeds. Pinned in Task 2.

---

### Task 1: Freeze llama-swap into `engines/llama-swap` and prove it builds and tests

**Files:**
- Create: `engines/README.md`
- Create: `engines/llama-swap/` (copy of upstream v257, without `.git`)
- Create: `engines/llama-swap/FROZEN.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a buildable Go module at `engines/llama-swap`, module path
  `github.com/mostlygeek/llama-swap` (unchanged), and a `FROZEN.md` patch list that later
  tasks append to.

- [ ] **Step 1: Install Go 1.27.1 on the devbox (user-local, no sudo)**

```bash
mkdir -p ~/.local && cd ~/.local
curl -sfLO https://go.dev/dl/go1.27.1.linux-amd64.tar.gz
rm -rf ~/.local/go && tar xzf go1.27.1.linux-amd64.tar.gz && rm go1.27.1.linux-amd64.tar.gz
~/.local/go/bin/go version
```
Expected: `go version go1.27.1 linux/amd64`. Every later `go` command in this plan means
`~/.local/go/bin/go`, or add it to `PATH` for the session.

- [ ] **Step 2: Copy the source at the exact commit**

```bash
cd /home/jay/projects/FreeToken
tmp=$(mktemp -d)
git clone -q --branch v257 --depth 1 https://github.com/mostlygeek/llama-swap.git "$tmp/ls"
test "$(git -C "$tmp/ls" rev-parse HEAD)" = f00d375a927f72d72e74ca86012be210f155c67b
mkdir -p engines
rsync -a --exclude .git --exclude ui/node_modules "$tmp/ls/" engines/llama-swap/
du -sh engines/llama-swap
```
Expected: about 13 MB; there is no `.git` inside `engines/llama-swap`.

- [ ] **Step 3: Ignore build outputs**

Append to `.gitignore`:
```gitignore
# frozen engines: build outputs only (sources are tracked)
engines/llama-swap/build/
engines/llama-swap/ui/node_modules/
engines/llama-swap/internal/server/ui_dist/*
!engines/llama-swap/internal/server/ui_dist/.gitkeep
engines/ninfer*/build/
```
Then check whether `internal/server/ui_dist` exists upstream with a placeholder:
`ls engines/llama-swap/internal/server/ui_dist`. If upstream ships files there, keep them
tracked: delete the two `ui_dist` lines above, and the `go:embed all:ui_dist` build keeps
working without a UI build.

- [ ] **Step 4: Run upstream's tests unchanged**

```bash
cd engines/llama-swap && ~/.local/go/bin/go test -short -count=1 ./internal/... 2>&1 | tail -15
```
Expected: every package `ok` (or `[no test files]`). Record any failing package in
`FROZEN.md` under "Known upstream test state" with its first error line. It must then fail the
same way in later tasks, never newly.

- [ ] **Step 5: Build without the UI (quick check)**

```bash
cd engines/llama-swap && ~/.local/go/bin/go build -o build/llama-swap . && ./build/llama-swap --version
```
Expected: prints a version line and exits 0.

- [ ] **Step 6: Write `engines/llama-swap/FROZEN.md`**

```markdown
# llama-swap (frozen copy)

- Source: https://github.com/mostlygeek/llama-swap
- Version: v257, commit f00d375a927f72d72e74ca86012be210f155c67b (2026-09-22)
- Copied: 2026-09-24, without .git and ui/node_modules
- License: MIT (LICENSE.md, kept unchanged)

Nothing here updates by itself. To take something from upstream: diff upstream against the
commit above, copy only the wanted change, re-run the tests, and add a line below.

## Known upstream test state

(none failing) <- replace with the Step 4 result if any package failed

## Our patches

Every changed spot carries a `// FreeToken patch Pn:` comment.

| Patch | What | Files |
|---|---|---|
```

- [ ] **Step 7: Write `engines/README.md`**

```markdown
# Frozen engines

Plain copies of other projects that this repo builds and runs. They change only when we
change them. Each folder has a FROZEN.md with its source commit and our patch list.

| Folder | What | Source |
|---|---|---|
| llama-swap/ | the one-address model switcher (Go + web UI) | mostlygeek/llama-swap v257 |
| adapters/ | start/stop/ready scripts, one per engine (see adapters/CONTRACT.md) | ours |
| config/ | example switcher config; the live one is ~/llama-swap/config.yaml on the box | ours |

Build everything on the serving box with `scripts/engines/build.sh`.
```

- [ ] **Step 8: Commit**

```bash
git checkout -b feat/own-switcher origin/mtp-upstream-merge   # controller only
git add .gitignore engines/README.md engines/llama-swap
git commit -m "build(engines): freeze llama-swap v257 into engines/llama-swap"
```

---

### Task 2: Patch P4, clamp out-of-range request params (`clampParams`)

**Files:**
- Modify: `engines/llama-swap/internal/config/filters.go`
- Modify: `engines/llama-swap/internal/server/filters.go` (`applyFilters`, around line 130)
- Modify: `engines/llama-swap/config-schema.json`: the model `filters` object (near line 507)
  and the peer `filters` object (near line 753)
- Test: `engines/llama-swap/internal/config/filters_test.go`
- Test: `engines/llama-swap/internal/server/filters_test.go`
- Modify: `engines/llama-swap/FROZEN.md`

**Interfaces:**
- Produces: `config.Filters.ClampParams map[string][]float64` (yaml `clampParams`), and
  `func (f Filters) SanitizedClampParams() ([]string, map[string][2]float64)`.
  `applyFilters` applies the clamp after `stripParams` and before `setParams`.

- [ ] **Step 1: Write the failing config test** (append to `internal/config/filters_test.go`)

```go
// FreeToken patch P4: clampParams sanitising.
func TestFilters_SanitizedClampParams(t *testing.T) {
	f := Filters{ClampParams: map[string][]float64{
		"top_k":       {0, 20},
		"temperature": {0, 2},
		"model":       {0, 1}, // protected: dropped
		"bad_len":     {1},    // not a pair: dropped
		"inverted":    {5, 1}, // min > max: dropped
	}}
	keys, bounds := f.SanitizedClampParams()
	want := []string{"temperature", "top_k"}
	if !slices.Equal(keys, want) {
		t.Fatalf("keys=%v want %v", keys, want)
	}
	if bounds["top_k"] != [2]float64{0, 20} {
		t.Errorf("top_k bounds=%v", bounds["top_k"])
	}
}
```
If `slices` is not imported in that test file yet, add `"slices"` to its imports.

- [ ] **Step 2: Run it to see it fail**

Run: `cd engines/llama-swap && ~/.local/go/bin/go test ./internal/config/ -run TestFilters_SanitizedClampParams`
Expected: FAIL to compile (`unknown field ClampParams`).

- [ ] **Step 3: Implement in `internal/config/filters.go`**

Add to the `Filters` struct, after `SetParamsByID`:
```go
	// FreeToken patch P4: ClampParams pulls numeric request parameters into an
	// inclusive [min, max] range instead of letting the upstream reject them
	// (NInfer answers 400 "top_k must be in [0,20]"). Non-numeric or absent
	// params are left alone. Applied after StripParams, before SetParams.
	ClampParams map[string][]float64 `yaml:"clampParams"`
```
Add the method at the end of the file:
```go
// FreeToken patch P4: SanitizedClampParams returns the clamp keys in sorted
// order and their [min, max] bounds, dropping protected params, entries that
// are not exactly two numbers, and inverted ranges.
func (f Filters) SanitizedClampParams() ([]string, map[string][2]float64) {
	if len(f.ClampParams) == 0 {
		return nil, nil
	}
	bounds := make(map[string][2]float64, len(f.ClampParams))
	keys := make([]string, 0, len(f.ClampParams))
	for key, pair := range f.ClampParams {
		key = strings.TrimSpace(key)
		if key == "" || slices.Contains(ProtectedParams, key) || len(pair) != 2 || pair[0] > pair[1] {
			continue
		}
		bounds[key] = [2]float64{pair[0], pair[1]}
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		return nil, nil
	}
	sort.Strings(keys)
	return keys, bounds
}
```

- [ ] **Step 4: Run the config test**

Run: `go test ./internal/config/ -run TestFilters_SanitizedClampParams -v`
Expected: PASS.

- [ ] **Step 5: Write the failing server test** (append to `internal/server/filters_test.go`)

```go
// FreeToken patch P4: clamp numbers into range; leave everything else alone.
func TestApplyFilters_ClampParams(t *testing.T) {
	f := config.Filters{ClampParams: map[string][]float64{
		"top_k": {0, 20}, "temperature": {0, 2}, "top_p": {0, 1},
	}}
	cases := []struct{ name, in, want string }{
		{"above max int", `{"model":"m","top_k":40}`, `{"model":"m","top_k":20}`},
		{"below min float", `{"model":"m","temperature":-0.5}`, `{"model":"m","temperature":0}`},
		{"in range untouched", `{"model":"m","top_p":0.95}`, `{"model":"m","top_p":0.95}`},
		{"absent untouched", `{"model":"m"}`, `{"model":"m"}`},
		{"string untouched", `{"model":"m","top_k":"40"}`, `{"model":"m","top_k":"40"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyFilters([]byte(tc.in), "m", "", f)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tc.want {
				t.Errorf("got %s want %s", got, tc.want)
			}
		})
	}
}
```

- [ ] **Step 6: Run it to see it fail**

Run: `go test ./internal/server/ -run TestApplyFilters_ClampParams -v`
Expected: FAIL (`top_k` stays 40).

- [ ] **Step 7: Implement in `applyFilters`**, right after the `SanitizedStripParams` loop

```go
	// FreeToken patch P4: clamp numeric params into their configured range.
	clampKeys, clampBounds := f.SanitizedClampParams()
	for _, key := range clampKeys {
		v := gjson.GetBytes(body, key)
		if v.Type != gjson.Number {
			continue
		}
		b := clampBounds[key]
		x := v.Float()
		var bound float64
		switch {
		case x < b[0]:
			bound = b[0]
		case x > b[1]:
			bound = b[1]
		default:
			continue
		}
		var out any = bound
		if bound == math.Trunc(bound) {
			out = int64(bound)
		}
		if body, err = sjson.SetBytes(body, key, out); err != nil {
			return nil, fmt.Errorf("error clamping parameter %s in request", key)
		}
	}
```
Add `"math"` to the imports of `internal/server/filters.go`.

- [ ] **Step 8: Add `clampParams` to `config-schema.json`**, in both `filters` objects, next to
  `stripParams`:

```json
"clampParams": {
    "type": "object",
    "description": "FreeToken patch P4: inclusive [min, max] per numeric request parameter; out-of-range values are clamped.",
    "additionalProperties": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2
    }
},
```

- [ ] **Step 9: Run the package tests**

Run: `go test -short -count=1 ./internal/config/ ./internal/server/`
Expected: `ok` for both, including `TestConfig_ExampleMatchesSchema`.

- [ ] **Step 10: Record the patch and commit**

Append to the `FROZEN.md` table:
`| P4 | clampParams filter: clamp numeric params into [min,max] | internal/config/filters.go, internal/server/filters.go, config-schema.json |`
```bash
git add engines/llama-swap
git commit -m "feat(engines): llama-swap P4 clampParams keeps engine param limits"
```

---

### Task 3: Patch P1, latest pick wins

**Files:**
- Modify: `engines/llama-swap/internal/config/config.go:249` (`FifoConfig`)
- Modify: `engines/llama-swap/internal/router/scheduler/scheduler.go` (`Effects` interface)
- Modify: `engines/llama-swap/internal/router/scheduler/fifo.go` (`OnRequest` between steps 3
  and 4, plus a new `supersede`)
- Modify: `engines/llama-swap/internal/router/base.go` (`StartSwap`, `doSwap`, new
  `CancelSwap`, `UnloadTimeout`, and the struct fields)
- Create: `engines/llama-swap/internal/swaputil/superseded.go`
- Modify: `engines/llama-swap/config-schema.json` (the `fifo` object near line 943)
- Test: `engines/llama-swap/internal/router/scheduler/fifo_test.go`
- Test: every other `Effects` fake. Find them with
  `grep -rn "func (.*) StopProcesses(" engines/llama-swap/internal`.

**Interfaces:**
- Consumes: nothing from Task 2.
- Produces:
  - `config.FifoConfig.LatestWins *bool` (yaml `latestWins`), and
    `func (c FifoConfig) LatestWinsEnabled() bool`, true when unset.
  - `scheduler.Effects` gains `CancelSwap(modelID string)` and
    `UnloadTimeout(modelID string) time.Duration`.
  - `swaputil.SupersededError{Model, By string}`, an HTTPError with status 409 and code
    `model_superseded`.
  - `baseRouter.doSwap(ctx context.Context, modelID string, toStop []string)`: Task 4 hooks
    the memory gate into this signature.

- [ ] **Step 1: Extend the test fake** in `fifo_test.go`. Add fields and methods to `fakeEffects`:

```go
	cancelled []string // FreeToken patch P1: CancelSwap calls
```
```go
func (f *fakeEffects) CancelSwap(modelID string) { f.cancelled = append(f.cancelled, modelID) }

func (f *fakeEffects) UnloadTimeout(string) time.Duration { return 7 * time.Second }
```
Add the same two methods, with empty or constant bodies, to every other `Effects` fake the
grep in **Files** finds.

- [ ] **Step 2: Write the failing tests** (append to `fifo_test.go`)

```go
// FreeToken patch P1: helpers — a group-style planner where every model evicts every other.
func exclusivePlanner(models ...string) *stubPlanner {
	ev := map[string][]string{}
	for _, m := range models {
		for _, o := range models {
			if o != m {
				ev[m] = append(ev[m], o)
			}
		}
	}
	return &stubPlanner{evict: ev}
}

func supersededFor(eff *fakeEffects, model string) int {
	n := 0
	for _, g := range eff.grants {
		var se swaputil.SupersededError
		if g.model == model && errors.As(g.err, &se) {
			n++
		}
	}
	return n
}

func TestFIFO_LatestWins_CancelsStartingSwap(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	s := newFIFO(exclusivePlanner("a", "b"), eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateStarting
	s.OnRequest(req("b"))

	if !slices.Equal(eff.cancelled, []string{"a"}) {
		t.Fatalf("cancelled=%v want [a]", eff.cancelled)
	}
	if len(eff.stops) != 1 || !slices.Equal(eff.stops[0].ids, []string{"a"}) || eff.stops[0].timeout != 7*time.Second {
		t.Fatalf("stops=%v want one stop of [a] with the unload timeout", eff.stops)
	}
	if got := supersededFor(eff, "a"); got != 1 {
		t.Errorf("a waiters superseded=%d want 1", got)
	}
	if got := eff.startsFor("b"); got != 1 {
		t.Errorf("StartSwap(b)=%d want 1", got)
	}
	// A late SwapDone for the cancelled swap must not grant anything.
	s.OnSwapDone(SwapDone{ModelID: "a", Err: context.Canceled})
	if got := eff.served("a"); got != 0 {
		t.Errorf("served(a)=%d want 0", got)
	}
}

func TestFIFO_LatestWins_ThreeQuickPicksOnlyLastLoads(t *testing.T) {
	eff := newFakeEffects()
	for _, m := range []string{"a", "b", "c"} {
		eff.states[m] = process.StateStopped
	}
	s := newFIFO(exclusivePlanner("a", "b", "c"), eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateStarting
	s.OnRequest(req("b"))
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStarting
	s.OnRequest(req("c"))

	if !slices.Equal(eff.cancelled, []string{"a", "b"}) {
		t.Fatalf("cancelled=%v want [a b]", eff.cancelled)
	}
	if supersededFor(eff, "a") != 1 || supersededFor(eff, "b") != 1 {
		t.Errorf("superseded a=%d b=%d want 1 each", supersededFor(eff, "a"), supersededFor(eff, "b"))
	}
	if eff.startsFor("c") != 1 {
		t.Errorf("StartSwap(c)=%d want 1", eff.startsFor("c"))
	}
}

func TestFIFO_LatestWins_SameModelJoinsNotCancels(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(exclusivePlanner("a", "b"), eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateStarting
	s.OnRequest(req("a"))

	if len(eff.cancelled) != 0 || len(eff.stops) != 0 {
		t.Fatalf("same-model request cancelled=%v stops=%v; want none", eff.cancelled, eff.stops)
	}
	if eff.startsFor("a") != 1 {
		t.Errorf("StartSwap(a)=%d want 1", eff.startsFor("a"))
	}
}

func TestFIFO_LatestWins_ReadyTargetNotCancelled(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	s := newFIFO(exclusivePlanner("a", "b"), eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateReady // ready, SwapDone not processed yet
	s.OnRequest(req("b"))

	if len(eff.cancelled) != 0 {
		t.Fatalf("cancelled=%v; a ready target must not be cancelled", eff.cancelled)
	}
	if eff.startsFor("b") != 0 {
		t.Errorf("b must queue behind the finishing swap, StartSwap(b)=%d", eff.startsFor("b"))
	}
}

func TestFIFO_LatestWins_QueuedRequestsForVictimAreSuperseded(t *testing.T) {
	eff := newFakeEffects()
	for _, m := range []string{"a", "b", "c"} {
		eff.states[m] = process.StateStopped
	}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), exclusivePlanner("a", "b", "c"),
		config.FifoConfig{LatestWins: boolPtr(false)}, nil, eff)
	s.OnRequest(req("a"))
	eff.states["a"] = process.StateStarting
	s.OnRequest(req("b")) // latestWins off: queued behind a
	s.cfg.LatestWins = boolPtr(true)
	s.OnRequest(req("c"))

	if supersededFor(eff, "a") != 1 || supersededFor(eff, "b") != 1 {
		t.Errorf("superseded a=%d b=%d want 1 each", supersededFor(eff, "a"), supersededFor(eff, "b"))
	}
	if eff.startsFor("b") != 0 {
		t.Errorf("queued b must not start later, StartSwap(b)=%d", eff.startsFor("b"))
	}
}

func TestFIFO_LatestWinsOff_KeepsUpstreamQueueing(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	s := NewFIFO("test", logmon.NewWriter(io.Discard), exclusivePlanner("a", "b"),
		config.FifoConfig{LatestWins: boolPtr(false)}, nil, eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateStarting
	s.OnRequest(req("b"))

	if len(eff.cancelled) != 0 || eff.startsFor("b") != 0 {
		t.Fatalf("latestWins=false: cancelled=%v StartSwap(b)=%d; want none", eff.cancelled, eff.startsFor("b"))
	}
}

func boolPtr(b bool) *bool { return &b }
```
Add `"slices"` to the test imports if it is missing. `errors`, `context`, `io`, `config`,
`logmon`, `process` and `swaputil` are already imported.

- [ ] **Step 3: Run them to see them fail**

Run: `go test ./internal/router/scheduler/ -run LatestWins -v`
Expected: FAIL to compile (`unknown field LatestWins`, `undefined: swaputil.SupersededError`).

- [ ] **Step 4: Add the config field** in `internal/config/config.go`

```go
type FifoConfig struct {
	Priority map[string]int `yaml:"priority"` // model ID -> priority, default 0
	// FreeToken patch P1: when true (the default), a request for a different
	// model cancels an in-flight swap whose target has not become ready.
	LatestWins *bool `yaml:"latestWins"`
}

// FreeToken patch P1: LatestWinsEnabled reports the effective latestWins value.
func (c FifoConfig) LatestWinsEnabled() bool { return c.LatestWins == nil || *c.LatestWins }
```

- [ ] **Step 5: Add the error type** in `internal/swaputil/superseded.go`

```go
package swaputil

import (
	"fmt"
	"net/http"
)

// FreeToken patch P1: SupersededError answers callers whose model load was
// cancelled because a request for another model arrived ("latest wins").
type SupersededError struct {
	Model string // the model whose load was cancelled
	By    string // the model requested instead
}

func (e SupersededError) Error() string {
	return fmt.Sprintf("load of %s cancelled: model %s was requested instead", e.Model, e.By)
}

func (e SupersededError) StatusCode() int { return http.StatusConflict }

func (e SupersededError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return h
}

func (e SupersededError) Body() []byte {
	return NewErrorEnvelope(e.StatusCode(), e.Error(), "model_superseded").JSON()
}
```

- [ ] **Step 6: Extend `Effects`** in `scheduler.go`, after `StopProcesses`:

```go
	// FreeToken patch P1: CancelSwap aborts the swap goroutine for modelID so
	// it never starts its target after a superseding stop. No-op if none runs.
	CancelSwap(modelID string)
	// FreeToken patch P1: UnloadTimeout is modelID's configured graceful stop timeout.
	UnloadTimeout(modelID string) time.Duration
```

- [ ] **Step 7: Implement `supersede` in `fifo.go`**, and call it in `OnRequest` right after
  step (3):

```go
	// (3b) FreeToken patch P1 — latest wins: cancel colliding in-flight swaps
	// for other models whose target is not ready yet, then decide afresh.
	if s.cfg.LatestWinsEnabled() && s.supersede(req.Model, evict) {
		running = s.runningSet(req.Model)
		evict = s.planner.EvictionFor(req.Model, running)
	}
```
```go
// FreeToken patch P1: supersede cancels every in-flight swap for another model
// that collides with target and whose process is not ready yet: its waiters
// get SupersededError, as do queued requests for models whose load would evict
// target; the swap goroutine is cancelled and its process stopped (blocking,
// as OnUnload does). Reports whether any swap was cancelled.
func (s *FIFO) supersede(target string, evict []string) bool {
	var victims []string
	for id, sw := range s.active {
		if id == target {
			continue
		}
		if !containsString(evict, id) && !containsString(sw.evict, target) && !slicesOverlap(evict, sw.evict) {
			continue
		}
		if st, ok := s.effects.ModelState(id); ok && st == process.StateReady {
			continue
		}
		victims = append(victims, id)
	}
	if len(victims) == 0 {
		return false
	}
	sort.Strings(victims)
	victimSet := make(map[string]bool, len(victims))
	for _, id := range victims {
		victimSet[id] = true
		sw := s.active[id]
		delete(s.active, id)
		s.effects.CancelSwap(id)
		for _, w := range sw.waiters {
			s.grantError(w, swaputil.SupersededError{Model: id, By: target})
		}
		s.logger.Infof("%s: latest wins: cancelled load of %s for %s", s.name, id, target)
	}
	// Older picks still queued lose too: any queued request for another model
	// whose load would evict target (the planner is pure, so asking is safe).
	if len(s.queued) > 0 {
		kept := s.queued[:0]
		for _, w := range s.queued {
			if victimSet[w.Model] || (w.Model != target && containsString(s.planner.EvictionFor(w.Model, []string{target}), target)) {
				s.grantError(w, swaputil.SupersededError{Model: w.Model, By: target})
				continue
			}
			kept = append(kept, w)
		}
		s.queued = kept
	}
	for _, id := range victims {
		s.effects.StopProcesses(s.effects.UnloadTimeout(id), []string{id})
	}
	return true
}
```
Add `"sort"` to the imports of `fifo.go` if it is missing.

- [ ] **Step 8: Run the scheduler tests**

Run: `go test -count=1 ./internal/router/scheduler/ -v -run 'LatestWins|FIFO' 2>&1 | tail -30`
Expected: all PASS, including upstream's `TestFIFO_*` (latestWins defaults on, and none of
them have a starting target colliding with a different model). If an upstream test expected
queueing behind a *starting* swap, change only that test to build its FIFO with
`config.FifoConfig{LatestWins: boolPtr(false)}`, and note it in `FROZEN.md`.

- [ ] **Step 9: Implement the router side** in `internal/router/base.go`

Add to the `baseRouter` struct:
```go
	// FreeToken patch P1: per-model cancel handles for running swap goroutines.
	swapMu      sync.Mutex
	swapCancels map[string]*swapHandle
```
Add the type next to it:
```go
// FreeToken patch P1: swapHandle identifies one swap goroutine's cancel func.
type swapHandle struct{ cancel context.CancelFunc }
```
In `newBaseRouter`, initialise `swapCancels: make(map[string]*swapHandle)`. Replace
`StartSwap` and add `CancelSwap` and `UnloadTimeout`:
```go
// StartSwap implements scheduler.Effects, launching the swap goroutine.
func (b *baseRouter) StartSwap(modelID string, evict []string) {
	// FreeToken patch P1: each swap gets a cancellable context.
	ctx, cancel := context.WithCancel(b.shutdownCtx)
	h := &swapHandle{cancel: cancel}
	b.swapMu.Lock()
	if old := b.swapCancels[modelID]; old != nil {
		old.cancel()
	}
	b.swapCancels[modelID] = h
	b.swapMu.Unlock()
	go func() {
		defer func() {
			b.swapMu.Lock()
			if b.swapCancels[modelID] == h {
				delete(b.swapCancels, modelID)
			}
			b.swapMu.Unlock()
			cancel()
		}()
		b.doSwap(ctx, modelID, evict)
	}()
}

// FreeToken patch P1: CancelSwap implements scheduler.Effects.
func (b *baseRouter) CancelSwap(modelID string) {
	b.swapMu.Lock()
	h := b.swapCancels[modelID]
	delete(b.swapCancels, modelID)
	b.swapMu.Unlock()
	if h != nil {
		h.cancel()
	}
}

// FreeToken patch P1: UnloadTimeout implements scheduler.Effects.
func (b *baseRouter) UnloadTimeout(modelID string) time.Duration { return b.unloadTimeout(modelID) }
```
Change `doSwap` to take the context. Stop starting once cancelled, and start through the swap
context:
```go
func (b *baseRouter) doSwap(ctx context.Context, modelID string, toStop []string) {
	timeout := b.healthCheckTimeout()
	// ... the existing parallel stop of toStop, unchanged ...
	wg.Wait()

	var err error
	// FreeToken patch P1: a superseded swap must not start its target.
	if err = ctx.Err(); err == nil {
		target := b.processes[modelID]
		err = target.EnsureReady(ctx, timeout)
	}
	if err != nil && b.shutdownCtx.Err() == nil && ctx.Err() == nil {
		b.logger.Warnf("%s: starting %s failed: %v", b.name, modelID, err)
	}

	select {
	case b.swapDoneCh <- scheduler.SwapDone{ModelID: modelID, Err: err}:
	case <-b.shutdownCtx.Done():
	}
}
```
Keep upstream's comment block about `EnsureReady` above the `EnsureReady` line.

- [ ] **Step 10: Add `latestWins` to the schema** in the `fifo` object's `properties`, next to
  `priority`:

```json
"latestWins": {
    "type": "boolean",
    "default": true,
    "description": "FreeToken patch P1: a request for a different model cancels an in-flight load that is not ready yet."
}
```

- [ ] **Step 11: Run the full suite**

Run: `go test -short -count=1 ./internal/... 2>&1 | tail -20`
Expected: every package `ok`, with the same known-upstream state as Task 1 Step 4.

- [ ] **Step 12: Record and commit**

`FROZEN.md` row:
`| P1 | latest wins: a new pick cancels a colliding not-ready swap (409 model_superseded) | internal/config/config.go, internal/router/scheduler/{scheduler.go,fifo.go}, internal/router/base.go, internal/swaputil/superseded.go, config-schema.json |`
```bash
git add engines/llama-swap
git commit -m "feat(engines): llama-swap P1 latest pick cancels a half-finished load"
```

---

### Task 4: Patch P2, wait for memory before loading

**Files:**
- Create: `engines/llama-swap/internal/memgate/memgate.go`
- Create: `engines/llama-swap/internal/memgate/windows_probe.go`
- Test: `engines/llama-swap/internal/memgate/memgate_test.go`
- Modify: `engines/llama-swap/internal/config/config.go` (a new `MemoryGate` field on `Config`)
- Modify: `engines/llama-swap/internal/config/model_config.go` (a new `RamNeedGB` field)
- Modify: `engines/llama-swap/internal/router/base.go` (a gate field, built in
  `newBaseRouter`, called in `doSwap`)
- Modify: `engines/llama-swap/config-schema.json` (top-level `memoryGate`, model `ramNeedGB`)

**Interfaces:**
- Consumes: `doSwap(ctx, modelID, toStop)` from Task 3.
- Produces:
  - `memgate.Gate{Probe, FloorGB, Wait, Poll, Logf}` and
    `func (g *Gate) WaitForRoom(ctx context.Context, model string, needGB float64) error`;
  - `memgate.NotEnoughMemoryError` (HTTP 503, code `not_enough_memory`);
  - `memgate.WindowsProbe(run func(context.Context) ([]byte, error), ttl time.Duration) Probe`;
  - `config.Config.MemoryGate config.MemoryGateConfig{Probe string; FloorGB float64; WaitSeconds int}`;
  - `config.ModelConfig.RamNeedGB float64`.

- [ ] **Step 1: Write the failing gate tests** (`internal/memgate/memgate_test.go`)

```go
package memgate

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"
)

func seq(values ...float64) Probe {
	i := 0
	return func(context.Context) (float64, error) {
		v := values[min(i, len(values)-1)]
		i++
		return v, nil
	}
}

func fastGate(p Probe) *Gate {
	return &Gate{Probe: p, FloorGB: 6, Wait: 50 * time.Millisecond, Poll: time.Millisecond}
}

func TestWaitForRoom_EnoughNow(t *testing.T) {
	if err := fastGate(seq(40)).WaitForRoom(context.Background(), "m", 18); err != nil {
		t.Fatal(err)
	}
}

func TestWaitForRoom_WaitsThenLoads(t *testing.T) {
	if err := fastGate(seq(10, 12, 30)).WaitForRoom(context.Background(), "m", 18); err != nil {
		t.Fatalf("want room after waiting, got %v", err)
	}
}

func TestWaitForRoom_TimesOutWith503(t *testing.T) {
	err := fastGate(seq(10)).WaitForRoom(context.Background(), "big", 60)
	var nem NotEnoughMemoryError
	if !errors.As(err, &nem) {
		t.Fatalf("want NotEnoughMemoryError, got %v", err)
	}
	if nem.StatusCode() != http.StatusServiceUnavailable || nem.Model != "big" || nem.NeedGB != 60 || nem.FreeGB != 10 {
		t.Errorf("unexpected error %+v", nem)
	}
}

func TestWaitForRoom_ProbeErrorLoadsAnyway(t *testing.T) {
	g := fastGate(func(context.Context) (float64, error) { return 0, errors.New("powershell missing") })
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatalf("a broken probe must not block loads, got %v", err)
	}
}

func TestWaitForRoom_CancelEndsWaitAtOnce(t *testing.T) {
	g := &Gate{Probe: seq(1), FloorGB: 6, Wait: time.Hour, Poll: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- g.WaitForRoom(ctx, "m", 18) }()
	time.Sleep(10 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("wait did not end after cancel")
	}
}

func TestWaitForRoom_DisabledWhenNoNeedOrNilGate(t *testing.T) {
	var g *Gate
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatal(err)
	}
	if err := fastGate(seq(0)).WaitForRoom(context.Background(), "m", 0); err != nil {
		t.Fatal(err)
	}
}

func TestWindowsProbe_ParsesKBAndCaches(t *testing.T) {
	calls := 0
	p := WindowsProbe(func(context.Context) ([]byte, error) {
		calls++
		return []byte("  8388608\r\n"), nil // 8 GiB in KB
	}, time.Minute)
	for i := 0; i < 3; i++ {
		v, err := p(context.Background())
		if err != nil || v != 8 {
			t.Fatalf("v=%v err=%v want 8", v, err)
		}
	}
	if calls != 1 {
		t.Errorf("calls=%d want 1 (cached)", calls)
	}
}

func TestWindowsProbe_GarbageIsAnError(t *testing.T) {
	p := WindowsProbe(func(context.Context) ([]byte, error) { return []byte("Access denied"), nil }, time.Minute)
	if _, err := p(context.Background()); err == nil {
		t.Fatal("want error for unparsable output")
	}
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `go test ./internal/memgate/`
Expected: FAIL (package has no non-test Go files, or `undefined: Gate`).

- [ ] **Step 3: Implement `internal/memgate/memgate.go`**

```go
// Package memgate is FreeToken patch P2: before a model loads, wait until the
// PC has room for it, so a load never squeezes Windows into swapping (measured
// 2026-09-24: a FreeToken boot took Windows free RAM to 0-2 GB).
package memgate

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// Probe reports free host memory in GB.
type Probe func(ctx context.Context) (float64, error)

// Gate holds a load until free - need >= FloorGB, for at most Wait.
type Gate struct {
	Probe   Probe
	FloorGB float64
	Wait    time.Duration
	Poll    time.Duration
	Logf    func(format string, args ...any)
}

// WaitForRoom returns nil when the model may load, NotEnoughMemoryError after
// Wait, or ctx.Err() when the load was cancelled. A failing probe never blocks.
func (g *Gate) WaitForRoom(ctx context.Context, model string, needGB float64) error {
	if g == nil || g.Probe == nil || needGB <= 0 {
		return nil
	}
	deadline := time.Now().Add(g.Wait)
	for {
		free, err := g.Probe(ctx)
		if err != nil {
			g.logf("memory gate: probe failed, loading %s anyway: %v", model, err)
			return nil
		}
		if free-needGB >= g.FloorGB {
			return nil
		}
		if !time.Now().Before(deadline) {
			return NotEnoughMemoryError{Model: model, NeedGB: needGB, FreeGB: free, FloorGB: g.FloorGB}
		}
		g.logf("memory gate: %s needs %.0f GB, %.1f GB free (floor %.0f GB); waiting", model, needGB, free, g.FloorGB)
		t := time.NewTimer(g.Poll)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}

func (g *Gate) logf(format string, args ...any) {
	if g.Logf != nil {
		g.Logf(format, args...)
	}
}

// NotEnoughMemoryError is the 503 answer when the wait runs out.
type NotEnoughMemoryError struct {
	Model   string
	NeedGB  float64
	FreeGB  float64
	FloorGB float64
}

func (e NotEnoughMemoryError) Error() string {
	return fmt.Sprintf("not enough free memory to load %s (need %.0f GB plus a %.0f GB cushion, Windows has %.1f GB free); close something and try again",
		e.Model, e.NeedGB, e.FloorGB, e.FreeGB)
}

func (e NotEnoughMemoryError) StatusCode() int { return http.StatusServiceUnavailable }

func (e NotEnoughMemoryError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	h.Set("Retry-After", "30")
	return h
}

func (e NotEnoughMemoryError) Body() []byte {
	return swaputil.NewErrorEnvelope(e.StatusCode(), e.Error(), "not_enough_memory").JSON()
}
```

- [ ] **Step 4: Implement `internal/memgate/windows_probe.go`**

```go
package memgate

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// powershellPath finds powershell.exe from WSL. systemd --user units may not
// carry /mnt/c on PATH, so fall back to the fixed System32 location.
func powershellPath() string {
	if p, err := exec.LookPath("powershell.exe"); err == nil {
		return p
	}
	return "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
}

// RunWindowsFreeKB asks Windows for FreePhysicalMemory (KB), with a 10 s cap.
func RunWindowsFreeKB(ctx context.Context) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if _, err := os.Stat(powershellPath()); err != nil {
		return nil, fmt.Errorf("powershell.exe not reachable: %w", err)
	}
	return exec.CommandContext(ctx, powershellPath(), "-NoProfile", "-Command",
		"(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory").Output()
}

// WindowsProbe turns run's KB output into GB and caches it for ttl.
func WindowsProbe(run func(context.Context) ([]byte, error), ttl time.Duration) Probe {
	var mu sync.Mutex
	var at time.Time
	var cached float64
	return func(ctx context.Context) (float64, error) {
		mu.Lock()
		defer mu.Unlock()
		if !at.IsZero() && time.Since(at) < ttl {
			return cached, nil
		}
		out, err := run(ctx)
		if err != nil {
			return 0, err
		}
		kb, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
		if err != nil {
			return 0, fmt.Errorf("unexpected FreePhysicalMemory output %q", strings.TrimSpace(string(out)))
		}
		cached, at = kb/1024/1024, time.Now()
		return cached, nil
	}
}
```

- [ ] **Step 5: Run the gate tests**

Run: `go test -count=1 ./internal/memgate/ -v`
Expected: all PASS.

- [ ] **Step 6: Add the config fields**

In `internal/config/config.go`, add to `Config`, next to `UnloadTimeout`:
```go
	// FreeToken patch P2: wait for free memory before loading a model.
	MemoryGate MemoryGateConfig `yaml:"memoryGate"`
```
And the type:
```go
// FreeToken patch P2: MemoryGateConfig. Probe "windows" enables the gate
// (WSL: asks Windows for free RAM); "" or "none" disables it.
type MemoryGateConfig struct {
	Probe       string  `yaml:"probe"`
	FloorGB     float64 `yaml:"floorGB"`     // default 6
	WaitSeconds int     `yaml:"waitSeconds"` // default 300
}
```
In `internal/config/model_config.go`, add to `ModelConfig` after `UnloadAfter`:
```go
	// FreeToken patch P2: host RAM (GB) this model needs to load; 0 = never wait.
	RamNeedGB float64 `yaml:"ramNeedGB"`
```

- [ ] **Step 7: Wire the gate into the router** (`internal/router/base.go`)

Add the struct field `memGate *memgate.Gate // FreeToken patch P2`. At the end of
`newBaseRouter`, before it returns:
```go
	// FreeToken patch P2: build the memory gate from config.
	if mg := conf.MemoryGate; mg.Probe == "windows" {
		floor, wait := mg.FloorGB, mg.WaitSeconds
		if floor <= 0 {
			floor = 6
		}
		if wait <= 0 {
			wait = 300
		}
		b.memGate = &memgate.Gate{
			Probe:   memgate.WindowsProbe(memgate.RunWindowsFreeKB, 2*time.Second),
			FloorGB: floor,
			Wait:    time.Duration(wait) * time.Second,
			Poll:    5 * time.Second,
			Logf:    b.logger.Infof,
		}
	}
```
Use whatever the constructor's config and receiver variables are actually called; read the
first 30 lines of `newBaseRouter`. In `doSwap`, between `wg.Wait()` and the P1 `ctx.Err()`
check:
```go
	// FreeToken patch P2: evicted models are stopped; now wait for room.
	if mc, ok := b.config.Models[modelID]; ok && ctx.Err() == nil {
		if gateErr := b.memGate.WaitForRoom(ctx, modelID, mc.RamNeedGB); gateErr != nil {
			select {
			case b.swapDoneCh <- scheduler.SwapDone{ModelID: modelID, Err: gateErr}:
			case <-b.shutdownCtx.Done():
			}
			return
		}
	}
```
`WaitForRoom` is nil-receiver safe, so a disabled gate costs nothing.

- [ ] **Step 8: Add to the schema.** At the top level:

```json
"memoryGate": {
    "type": "object",
    "description": "FreeToken patch P2: wait for free host memory before loading a model.",
    "properties": {
        "probe": {"type": "string", "enum": ["", "none", "windows"], "default": "none"},
        "floorGB": {"type": "number", "default": 6},
        "waitSeconds": {"type": "integer", "default": 300}
    },
    "additionalProperties": false
},
```
In the model properties, next to `unloadTimeout`:
```json
"ramNeedGB": {"type": "number", "description": "FreeToken patch P2: host RAM this model needs to load (GB); 0 disables the wait."},
```

- [ ] **Step 9: Run the full suite**

Run: `go test -short -count=1 ./internal/... 2>&1 | tail -20`
Expected: every package `ok`.

- [ ] **Step 10: Record and commit**

`FROZEN.md` row:
`| P2 | memory gate: wait for Windows free RAM - ramNeedGB >= floorGB before loading (503 not_enough_memory) | internal/memgate/*, internal/config/{config.go,model_config.go}, internal/router/base.go, config-schema.json |`
```bash
git add engines/llama-swap
git commit -m "feat(engines): llama-swap P2 waits for free memory before loading"
```

---

### Task 5: Engine adapters and their contract

**Files:**
- Create: `engines/adapters/CONTRACT.md`
- Move: `scripts/llama-swap/freetoken.sh` → `engines/adapters/freetoken.sh` (`git mv`, content
  unchanged)
- Create: `engines/adapters/ninfer.sh`, replacing `scripts/llama-swap/other-engine.sh`
- Delete: `scripts/llama-swap/other-engine.sh`
- Move: `scripts/llama-swap/config.example.yaml` → `engines/config/config.example.yaml`, then
  rewrite it (Task 7)
- Test: `tests/engines/test_adapters.py`, with `tests/engines/__init__.py` (empty)

**Interfaces:**
- Consumes: the FreeToken helper API (`/api/status`, `/api/settings`,
  `/api/server/{start,stop}`, `/api/server/jobs/{id}`) and the server's `/ready?model=`, both
  already in the repo.
- Produces:
  - `engines/adapters/ninfer.sh <runtime> <artifact> <model-id> [ninfer-serve flags...]`;
  - `engines/adapters/freetoken.sh <model folder>`, unchanged.
  - Both honour the env overrides `FREETOKEN_HELPER` (default `http://127.0.0.1:2031`) and
    `FREETOKEN_SERVER` (default `http://127.0.0.1:2020`); `ninfer.sh` also honours
    `NINFER_PROCESS_NAME` (default `ninfer-serve`).

- [ ] **Step 1: Write `engines/adapters/CONTRACT.md`**

```markdown
# Engine adapter contract

An adapter is the program llama-swap runs as a model's `cmd`. Every adapter must:

1. **Make room.** Stop anything else holding the GPU through that engine's proper stop path.
   FreeToken is stopped through its settings helper (`POST /api/server/stop`), which also
   disarms its crash watchdog. If the card cannot be freed, exit non-zero without starting.
2. **Start** the engine on its fixed local port and stay in the foreground while it runs,
   either by exec'ing the engine or by monitoring it.
3. **Be ready only when chats are answered.** The model's `checkEndpoint` must answer 200 only
   then. FreeToken uses `/ready?model=<folder>`; NInfer uses `/health`, which it opens only
   after "engine ready".
4. **Stop fully on SIGTERM.** Exit once the card is released; exit non-zero if the stop failed.

Adding an engine means writing one adapter against this contract, adding a build step to
`scripts/engines/build.sh` if needed, and adding config entries.
```

- [ ] **Step 2: Write the failing adapter tests** (`tests/engines/test_adapters.py`)

```python
"""Adapter contract tests with a fake FreeToken helper/server and fake engines.

No GPU, no real engines: the helper is a tiny HTTP server in a thread whose state the
tests script, and the "engine" is a shell script that records its argv and sleeps.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = REPO / "engines" / "adapters"


class FakeHelper:
    """Scriptable stand-in for the settings helper (:2031) and FreeToken server (:2020)."""

    def __init__(self):
        self.state = "unreachable"   # server.state in /api/status
        self.job = None              # currentJob id or None
        self.armed = False
        self.model_path = "/m/A"
        self.stop_result = "stopped"  # stage the stop job ends in
        self.start_result = "serving"
        self.calls: list[str] = []
        self.jobs: dict[str, str] = {}
        helper = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                helper.calls.append("GET " + self.path)
                if self.path == "/api/status":
                    return self._send(200, {
                        "server": {"state": helper.state},
                        "currentJob": {"jobId": helper.job} if helper.job else None,
                        "autoRestart": {"armed": helper.armed, "enabled": True, "gave_up": False},
                    })
                if self.path == "/api/settings":
                    return self._send(200, {"settings": {"ModelPath": helper.model_path}})
                if self.path.startswith("/api/server/jobs/"):
                    jid = self.path.rsplit("/", 1)[1]
                    if jid not in helper.jobs:
                        return self._send(404, {"detail": "not found"})
                    return self._send(200, {"jobId": jid, "stage": helper.jobs[jid]})
                if self.path.startswith("/ready"):
                    want = self.path.split("model=", 1)[1] if "model=" in self.path else ""
                    if helper.state == "serving" and os.path.basename(helper.model_path) != want:
                        return self._send(503, {"status": "not_ready", "reason": "another model is loaded"})
                    return self._send(200 if helper.state == "serving" else 503, {})
                return self._send(404, {})

            def do_PUT(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                helper.calls.append("PUT " + self.path)
                helper.model_path = body["settings"]["ModelPath"]
                return self._send(200, {"status": "saved", "settings": body["settings"]})

            def do_POST(self):
                helper.calls.append("POST " + self.path)
                jid = f"job-{len(helper.jobs) + 1}"
                if self.path == "/api/server/stop":
                    helper.jobs[jid] = helper.stop_result
                    if helper.stop_result == "stopped":
                        helper.state, helper.job, helper.armed = "unreachable", None, False
                    return self._send(202, {"jobId": jid})
                if self.path == "/api/server/start":
                    helper.jobs[jid] = helper.start_result
                    if helper.start_result == "serving":
                        helper.state = "serving"
                    return self._send(202, {"jobId": jid})
                return self._send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def helper():
    h = FakeHelper()
    yield h
    h.close()


def env_for(helper, tmp_path):
    env = dict(os.environ)
    env.update(FREETOKEN_HELPER=helper.url, FREETOKEN_SERVER=helper.url,
               NINFER_PROCESS_NAME=f"fake-ninfer-{os.getpid()}", NINFER_STOP_POLLS="3")
    return env


def fake_engine(tmp_path) -> Path:
    p = tmp_path / "engine.sh"
    p.write_text('#!/usr/bin/env bash\necho "$@" > "$(dirname "$0")/argv"\nexec sleep 30\n')
    p.chmod(0o755)
    return p


def test_ninfer_execs_the_runtime_when_the_card_is_free(helper, tmp_path):
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "quasar-27b", "--port", "8090"],
                            env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert (tmp_path / "argv").read_text().split() == ["/a.ninfer", "--model-id", "quasar-27b", "--port", "8090"]
    assert "POST /api/server/stop" not in helper.calls


def test_ninfer_stops_freetoken_first(helper, tmp_path):
    helper.state = "serving"
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls
    assert (tmp_path / "argv").exists()


def test_ninfer_refuses_when_freetoken_will_not_stop(helper, tmp_path):
    helper.state, helper.stop_result = "serving", "failed"
    eng = fake_engine(tmp_path)
    r = subprocess.run([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert not (tmp_path / "argv").exists()
    assert "refusing" in r.stderr


def test_ninfer_stops_freetoken_when_its_watchdog_is_live(helper, tmp_path):
    helper.armed = True  # crashed server about to be restarted
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path))
    time.sleep(1.5)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls


def test_freetoken_boots_then_stops_on_sigterm(helper, tmp_path):
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path),
                            stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if helper.state == "serving":
            break
        time.sleep(0.1)
    assert helper.model_path == "/m/B" and "POST /api/server/start" in helper.calls
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(20) == 0
    assert helper.calls.count("POST /api/server/stop") == 1


def test_freetoken_sigterm_exits_nonzero_when_stop_fails(helper, tmp_path):
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path))
    for _ in range(100):
        if helper.state == "serving":
            break
        time.sleep(0.1)
    helper.stop_result = "failed"
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(20) != 0


def test_freetoken_adopts_the_same_model_without_rebooting(helper, tmp_path):
    helper.state, helper.model_path = "serving", "/m/B"
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path))
    time.sleep(1.5)
    proc.terminate()
    proc.wait(20)
    assert "POST /api/server/start" not in helper.calls
```

- [ ] **Step 3: Move the files and run the tests to see the ninfer ones fail**

```bash
mkdir -p engines/adapters engines/config tests/engines && touch tests/engines/__init__.py
git mv scripts/llama-swap/freetoken.sh engines/adapters/freetoken.sh
git mv scripts/llama-swap/config.example.yaml engines/config/config.example.yaml
PYTHONPATH=python .venv/bin/python -m pytest tests/engines -q 2>&1 | tail -5
```
Expected: the `freetoken` tests PASS (the script is unchanged). The `ninfer` tests FAIL
because `ninfer.sh` does not exist.

- [ ] **Step 4: Write `engines/adapters/ninfer.sh`** (then `chmod +x`)

```bash
#!/usr/bin/env bash
# Engine adapter for NInfer (see engines/adapters/CONTRACT.md):
#
#   ninfer.sh <runtime ninfer-serve> <artifact.ninfer> <model-id> [ninfer-serve flags...]
#
# Makes room on the card, then exec's the runtime so llama-swap's SIGTERM reaches it
# directly. FreeToken is stopped through its settings helper (which disarms its crash
# watchdog); a stray NInfer started outside llama-swap is stopped by exact process name.
set -uo pipefail
RUNTIME="${1:?usage: ninfer.sh <runtime> <artifact> <model-id> [flags...]}"
ARTIFACT="${2:?artifact}"
MODEL_ID="${3:?model-id}"
shift 3
HELPER="${FREETOKEN_HELPER:-http://127.0.0.1:2031}"
SERVER="${FREETOKEN_SERVER:-http://127.0.0.1:2020}"
PROC="${NINFER_PROCESS_NAME:-ninfer-serve}"

log() { printf '[ninfer.sh] %s\n' "$*" >&2; }

# "state has_job watchdog_live"; "helper-down - False" when the helper does not answer.
snapshot() {
  curl -s --max-time 5 "$HELPER/api/status" | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: print('helper-down - False'); sys.exit(0)
w=d.get('autoRestart') or {}
print(d['server']['state'], bool(d.get('currentJob')), bool(w.get('armed') and w.get('enabled', True) and not w.get('gave_up')))"
}

read -r state has_job live <<<"$(snapshot)"
if [ "$state" = helper-down ]; then
  if curl -s --max-time 3 -o /dev/null "$SERVER/health"; then
    log "FreeToken answers on $SERVER but the helper is down; refusing"
    exit 1
  fi
elif [ "$state" != unreachable ] || [ "$has_job" = True ] || [ "$live" = True ]; then
  log "stopping FreeToken (state: $state) to free the card"
  curl -s --max-time 10 -X POST "$HELPER/api/server/stop" >/dev/null
  for _ in $(seq 1 "${NINFER_STOP_POLLS:-90}"); do
    read -r state has_job _ <<<"$(snapshot)"
    [ "$state" = unreachable ] && [ "$has_job" = False ] && break
    sleep 2
  done
  if [ "$state" != unreachable ] || [ "$has_job" != False ]; then
    log "FreeToken did not stop (state: $state); refusing to start"
    exit 1
  fi
fi

if pgrep -x "$PROC" >/dev/null; then
  log "stopping a $PROC started outside llama-swap"
  pkill -TERM -x "$PROC"
  for _ in $(seq 1 30); do pgrep -x "$PROC" >/dev/null || break; sleep 1; done
  pkill -KILL -x "$PROC" 2>/dev/null
  sleep 2
  pgrep -x "$PROC" >/dev/null && { log "$PROC would not stop; refusing"; exit 1; }
fi

exec "$RUNTIME" "$ARTIFACT" --model-id "$MODEL_ID" "$@"
```
`NINFER_STOP_POLLS` (default 90 polls × 2 s) lets the tests give up after 3 polls.

- [ ] **Step 5: Run the adapter tests**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/engines -q`
Expected: 7 passed.

- [ ] **Step 6: Remove the old wrapper and commit**

```bash
git rm -q scripts/llama-swap/other-engine.sh
rmdir scripts/llama-swap 2>/dev/null || true
git add engines/adapters engines/config tests/engines
git commit -m "feat(engines): adapter contract, ninfer.sh adapter and adapter tests"
```

---

### Task 6: Freeze NInfer (one copy or two)

**Files:**
- Create: `engines/ninfer/` (and possibly `engines/ninfer-quasar/`), each with a `FROZEN.md`
- Modify: `engines/README.md` (table row or rows)

**Interfaces:**
- Produces: `engines/ninfer*/` source trees that `build.sh` (Task 7) builds to
  `engines/ninfer*/build/apps/ninfer-serve`.

- [ ] **Step 1: Ask Jay, then run the one-runtime check on the box** (light: NInfer loads in
  about 20 s)

First confirm nothing is loaded: `curl -s 127.0.0.1:2040/running` must show
`{"running":[]}`; if not, ask Jay. Then, inside WSL `vllm`, for each of Fable and Twin:
```bash
cd ~/ninfer-work
./ninfer-mobile/build/apps/ninfer-serve models/fable_27b_nvfp4.ninfer --host 127.0.0.1 --port 8091 \
  --model-id fable-27b --max-context 150000 --kv-capacity 200000 --max-concurrency 4 \
  --kv-dtype fp8 --host-state-slots 8 --host-kv-mib 8192 --spec mtp --draft-tokens 4 \
  --lm-head-draft --preserve-thinking --vision > /tmp/mobile-fable.log 2>&1 &
until curl -sf 127.0.0.1:8091/health >/dev/null; do sleep 1; done
python3 - <<'EOF'
import json,time,urllib.request
body=json.dumps({"model":"fable-27b","messages":[{"role":"user","content":"Write a Python LRU cache class with get/put and O(1) operations, plus three pytest tests."}],"max_tokens":700,"temperature":0}).encode()
t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions",body,{"content-type":"application/json"}),timeout=600))
print("mobile", d["usage"]["completion_tokens"]/(time.time()-t), "tok/s")
EOF
pkill -x ninfer-serve
```
Then repeat with `./ninfer/build/apps/ninfer-serve` (today's upstream build) on the same
prompt. Do Twin the same way with `models/twin_nvfp4.ninfer`.

- [ ] **Step 2: Decide.** Freeze only the mobile runtime if both artifacts load and answer and
  each mobile tok/s is at least 95% of upstream's. Otherwise freeze both. Write the numbers
  into `engines/README.md`.

- [ ] **Step 3: Copy the chosen source(s)**

```bash
tmp=$(mktemp -d)
git clone -q https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile.git "$tmp/nm"
git -C "$tmp/nm" checkout -q d4bc75dbc7066109c3d9692ed564e5904a849ba0
git -C "$tmp/nm" submodule status   # must print nothing; if it lists submodules, record them in FROZEN.md and let build.sh fetch them at the listed commits
du -sh "$tmp/nm" && find "$tmp/nm" -type f -size +5M -not -path '*/.git/*'
rsync -a --exclude .git --exclude build "$tmp/nm/" engines/ninfer/
```
Leave out any file over 5 MB that is a test fixture or model sample, and list it in
`FROZEN.md` under "Left out". For a second copy, do the same for `Neroued/ninfer` at
`f76e19c0` into `engines/ninfer-upstream/`, and keep the mobile copy at `engines/ninfer/`,
because QUASAR needs it.

- [ ] **Step 4: Write `engines/ninfer/FROZEN.md`**

```markdown
# NInfer (frozen copy)

- Source: https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile (a fork of Neroued/ninfer that
  adds the QUASAR binding; includes upstream through ce7dee50)
- Commit: d4bc75dbc7066109c3d9692ed564e5904a849ba0
- Copied: 2026-09-24, without .git and build/
- License: Apache-2.0 (LICENSE kept)
- Runs: quasar-27b (DFlash2 K7), fable-27b and twin-27b (see engines/README.md for the check)

## Left out

(none) <- or list files over 5 MB and why

## Our patches

| Patch | What | Files |
|---|---|---|
```

- [ ] **Step 5: Build it once on the box to prove the copy is complete** (Task 7's script does
  this later; here run it directly):

```bash
cd ~/FreeToken && git pull -q && cd engines/ninfer && export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 20 2>&1 | tail -2
```
Expected: `Linking CXX executable apps/ninfer-serve` and exit 0. The box must first have this
branch: the controller pushes `feat/own-switcher`, and the box checks it out only for this
step (`git fetch && git checkout feat/own-switcher`). No model loads here.

- [ ] **Step 6: Commit**

```bash
git add engines/ninfer* engines/README.md
git commit -m "build(engines): freeze NInfer (QUASAR-capable runtime) into engines/ninfer"
```

---

### Task 7: Build script, service installer, example config

**Files:**
- Create: `scripts/engines/build.sh`
- Create: `scripts/engines/install-service.sh`
- Modify: `engines/config/config.example.yaml` (rewrite)

**Interfaces:**
- Consumes: `engines/llama-swap` (Tasks 1-4), `engines/adapters` (Task 5), `engines/ninfer*`
  (Task 6).
- Produces:
  - `~/.local/share/freetoken-engines/bin/llama-swap` (UI embedded);
  - `engines/ninfer*/build/apps/ninfer-serve`;
  - the systemd --user unit `llama-swap.service`, running the built binary.

- [ ] **Step 1: Write `scripts/engines/build.sh`**

```bash
#!/usr/bin/env bash
# Build the frozen engines on the serving box (WSL). Safe to re-run.
#   scripts/engines/build.sh            # everything
#   scripts/engines/build.sh llama-swap # one part: llama-swap | ninfer
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${FREETOKEN_ENGINES_BIN:-$HOME/.local/share/freetoken-engines/bin}"
GO_VERSION=1.27.1
NODE_VERSION=24.9.0
mkdir -p "$OUT" "$HOME/.local"

need_go() {
  if ! "$HOME/.local/go/bin/go" version 2>/dev/null | grep -q "go$GO_VERSION "; then
    echo "== installing Go $GO_VERSION"
    local tmp; tmp=$(mktemp -d)
    curl -sfL "https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz" | tar xz -C "$tmp"
    rm -rf "$HOME/.local/go" && mv "$tmp/go" "$HOME/.local/go" && rm -rf "$tmp"
  fi
  export PATH="$HOME/.local/go/bin:$PATH"
}

need_node() {
  local dir="$HOME/.local/node-v$NODE_VERSION-linux-x64"
  if [ ! -x "$dir/bin/node" ]; then
    echo "== installing Node $NODE_VERSION"
    curl -sfL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" | tar xJ -C "$HOME/.local"
  fi
  export PATH="$dir/bin:$PATH"
}

build_llama_swap() {
  need_go; need_node
  cd "$REPO/engines/llama-swap"
  echo "== llama-swap UI"
  (cd ui && npm ci --no-audit --no-fund && npm run build)
  echo "== llama-swap tests"
  go test -short -count=1 ./internal/...
  echo "== llama-swap binary"
  go build -tags embed_ui -ldflags "-X main.version=frozen-v257-freetoken -X main.commit=$(git -C "$REPO" rev-parse --short HEAD)" -o "$OUT/llama-swap.new" .
  mv "$OUT/llama-swap.new" "$OUT/llama-swap"
  "$OUT/llama-swap" --version
}

build_ninfer() {
  export PATH=/usr/local/cuda/bin:$PATH
  for d in "$REPO"/engines/ninfer*; do
    [ -f "$d/CMakeLists.txt" ] || continue
    echo "== $(basename "$d")"
    cmake -S "$d" -B "$d/build" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$d/build" -j "$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))"
  done
}

case "${1:-all}" in
  llama-swap) build_llama_swap ;;
  ninfer) build_ninfer ;;
  all) build_llama_swap; build_ninfer ;;
  *) echo "usage: $0 [all|llama-swap|ninfer]" >&2; exit 2 ;;
esac
```
Check that Node 24.9.0 exists at nodejs.org/dist. If it does not, use the newest 24.x listed
there, and make sure `engines/llama-swap/ui/package.json` has no `engines` constraint against
it.

- [ ] **Step 2: Write `scripts/engines/install-service.sh`**

```bash
#!/usr/bin/env bash
# Point the llama-swap systemd --user unit at the frozen build. Keeps the old unit as a backup.
set -euo pipefail
BIN="${FREETOKEN_ENGINES_BIN:-$HOME/.local/share/freetoken-engines/bin}/llama-swap"
CONF="$HOME/llama-swap/config.yaml"
UNIT="$HOME/.config/systemd/user/llama-swap.service"
[ -x "$BIN" ] || { echo "build first: scripts/engines/build.sh" >&2; exit 1; }
[ -f "$CONF" ] || { echo "missing $CONF (copy engines/config/config.example.yaml and fill in paths)" >&2; exit 1; }
mkdir -p "$(dirname "$UNIT")"
[ -f "$UNIT" ] && cp "$UNIT" "$UNIT.bak-$(date +%Y%m%d-%H%M%S)"
cat > "$UNIT" <<EOF
[Unit]
Description=llama-swap (FreeToken frozen build): one address for all local models (port 2040)
After=freetoken-settings.service

[Service]
WorkingDirectory=%h/llama-swap
ExecStart=$BIN --config $CONF --listen 127.0.0.1:2040 --watch-config
Restart=on-failure
KillMode=mixed
TimeoutStopSec=200

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable llama-swap >/dev/null
echo "installed; apply with: systemctl --user restart llama-swap"
```

- [ ] **Step 3: Rewrite `engines/config/config.example.yaml`**, matching the live config's
  models:

```yaml
# llama-swap (FreeToken frozen build) config. Copy to ~/llama-swap/config.yaml and fix paths.
# Patches used here: P1 latestWins, P2 memoryGate/ramNeedGB, P4 clampParams (engines/llama-swap/FROZEN.md).
# ttl per model: 0 = keep loaded, N = unload after N idle seconds.

healthCheckTimeout: 600
logLevel: info
sendLoadingState: false
globalTTL: 0

memoryGate:
  probe: windows      # asks Windows for free RAM from WSL; "none" turns the wait off
  floorGB: 6          # keep at least this much free for Windows after the load
  waitSeconds: 300    # then answer 503 with the numbers

routing:
  scheduler:
    settings:
      fifo:
        latestWins: true   # a new pick cancels a half-finished load

macros:
  ft: "${env.HOME}/FreeToken/engines/adapters/freetoken.sh"
  ninfer: "${env.HOME}/FreeToken/engines/adapters/ninfer.sh ${env.HOME}/FreeToken/engines/ninfer/build/apps/ninfer-serve"
  ninfer_common: >-
    --host 127.0.0.1 --port 8090
    --max-context 150000 --kv-capacity 200000
    --max-concurrency 4 --max-pending-requests 16 --pending-timeout-ms 300000
    --host-state-slots 8 --host-kv-mib 8192
    --lm-head-draft --preserve-thinking --vision

models:
  "qwen3.8-flash":
    name: "Qwen3.8 Flash Next NVFP4 (FreeToken)"
    cmd: ${ft} ${env.HOME}/models/Qwen3.8-Flash-Next-NVFP4
    proxy: http://127.0.0.1:2020
    checkEndpoint: /ready?model=Qwen3.8-Flash-Next-NVFP4
    unloadTimeout: 180
    ramNeedGB: 60        # measured in Task 8; update from the measurement
    ttl: 0
    aliases: ["Qwen3.8-Flash-Next-NVFP4"]

  "qwen3.8-flash-abliterated":
    name: "Qwen3.8 Flash Next ABLITERATED NVFP4 (FreeToken, uncensored)"
    cmd: ${ft} ${env.HOME}/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4
    proxy: http://127.0.0.1:2020
    checkEndpoint: /ready?model=Qwen3.8-Flash-Next-ABLITERATED-NVFP4
    unloadTimeout: 180
    ramNeedGB: 60
    ttl: 0
    aliases: ["Qwen3.8-Flash-Next-ABLITERATED-NVFP4"]

  "quasar-27b":
    name: "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"
    cmd: ${ninfer} ${env.HOME}/ninfer-work/models/quasar_27b_nvfp4.ninfer quasar-27b ${ninfer_common} --kv-dtype int8 --spec dflash2 --draft-tokens 7
    proxy: http://127.0.0.1:8090
    useModelName: "quasar-27b"
    unloadTimeout: 60
    ramNeedGB: 18
    ttl: 0
    env: ["PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"]
    filters:
      clampParams: {top_k: [0, 20], temperature: [0, 2], top_p: [0, 1], min_p: [0, 1], presence_penalty: [-2, 2], frequency_penalty: [-2, 2]}

  "fable-27b":
    name: "Fable 27B NVFP4 (NInfer)"
    cmd: ${ninfer} ${env.HOME}/ninfer-work/models/fable_27b_nvfp4.ninfer fable-27b ${ninfer_common} --kv-dtype fp8 --spec mtp --draft-tokens 4
    proxy: http://127.0.0.1:8090
    useModelName: "fable-27b"
    unloadTimeout: 60
    ramNeedGB: 18
    ttl: 0
    env: ["PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"]
    filters:
      clampParams: {top_k: [0, 20], temperature: [0, 2], top_p: [0, 1], min_p: [0, 1], presence_penalty: [-2, 2], frequency_penalty: [-2, 2]}

  "twin-27b":
    name: "Twin 27B NVFP4 (NInfer)"
    cmd: ${ninfer} ${env.HOME}/ninfer-work/models/twin_nvfp4.ninfer twin-27b ${ninfer_common} --kv-dtype fp8 --spec mtp --draft-tokens 4
    proxy: http://127.0.0.1:8090
    useModelName: "twin-27b"
    unloadTimeout: 60
    ramNeedGB: 18
    ttl: 0
    env: ["PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"]
    filters:
      clampParams: {top_k: [0, 20], temperature: [0, 2], top_p: [0, 1], min_p: [0, 1], presence_penalty: [-2, 2], frequency_penalty: [-2, 2]}
```
If Task 6 kept two NInfer copies, point Fable and Twin at
`engines/ninfer-upstream/build/apps/ninfer-serve` through a second macro, `ninfer_upstream`.

- [ ] **Step 4: Smoke-test the example config on the devbox** (no engines start: listing
  models never runs a `cmd`)

```bash
bash scripts/engines/build.sh llama-swap   # needs Node + Go; on the devbox this also runs the Go tests
H=$(mktemp -d); mkdir -p "$H/llama-swap"
HOME=$H ~/.local/share/freetoken-engines/bin/llama-swap --config engines/config/config.example.yaml --listen 127.0.0.1:12999 &
sleep 2; curl -s 127.0.0.1:12999/v1/models | python3 -c "import json,sys; print(sorted(m['id'] for m in json.load(sys.stdin)['data']))"
curl -s -o /dev/null -w "ui %{http_code}\n" 127.0.0.1:12999/ui/
kill %1
```
Expected: `['fable-27b', 'quasar-27b', 'qwen3.8-flash', 'qwen3.8-flash-abliterated', 'twin-27b']`
and `ui 200`. If the config fails to load, fix the example; never loosen the schema.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/engines/*.sh
git add scripts/engines engines/config
git commit -m "build(engines): build script, service installer and example config"
```

---

### Task 8: Live acceptance and switch-over on the box

**Files:**
- Modify: `~/llama-swap/config.yaml` on the box (untracked). Keep a backup first.
- Modify: `engines/config/config.example.yaml` (the measured `ramNeedGB`)
- Modify: `docs/research/`: create `own-switcher-acceptance-2026-09-XX.md` with the measured
  results, dated the day the run happens.

**Interfaces:**
- Consumes: everything above.
- Produces: the box running the frozen build on `127.0.0.1:2040` (tailnet 12020), with the old
  binary kept for rollback.

- [ ] **Step 1: Ask Jay for the live window.** Say: one FreeToken load, about 20 minutes, and
  Windows memory watched. Do not start without a yes.

- [ ] **Step 2: Build on the box, and install without switching yet**

```bash
cd ~/FreeToken && git fetch -q && git checkout -q feat/own-switcher && git pull -q
bash scripts/engines/build.sh
cp ~/llama-swap/config.yaml ~/llama-swap/config.yaml.bak-before-frozen
```
Then edit `~/llama-swap/config.yaml` to match `engines/config/config.example.yaml`: adapter
macros, `memoryGate`, `latestWins`, `clampParams`, `ramNeedGB`, and model paths as on the box.

- [ ] **Step 3: Switch the unit and check it serves**

```bash
bash scripts/engines/install-service.sh && systemctl --user restart llama-swap && sleep 3
curl -s 127.0.0.1:2040/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
~/.local/share/freetoken-engines/bin/llama-swap --version
```
Expected: the five model IDs, and a version that contains `frozen-v257-freetoken`.

- [ ] **Step 4: Run the acceptance list** (spec, Testing section). Record every result:
  1. QUASAR answers through `http://100.106.5.124:12020/v1` (from the devbox).
  2. Latest wins: request `qwen3.8-flash`. After 10 s, request `quasar-27b`. The first request
     gets 409 `model_superseded`, QUASAR answers, `/running` shows only QUASAR, and the helper
     state is `unreachable`.
  3. QUASAR → Fable → QUASAR swaps work. Run `qbench.py` (the 2026-09-24 benchmark: 700-token
     greedy answers). QUASAR must reach at least 95% of 293-320 tok/s code and 199-209 chat.
  4. Memory gate: set `ramNeedGB: 500` on `fable-27b` and `waitSeconds: 15`, request Fable, and
     expect a 503 `not_enough_memory` after about 15 s. Then restore both values.
  5. `top_k: 40` to QUASAR succeeds (clamped).
  6. Idle unload: `ttl: 60` on Fable. It unloads after about a minute, the card returns to
     baseline, and `pgrep -x ninfer-serve` finds nothing. Then restore `ttl: 0`.
  7. The one FreeToken boot (from item 2, or a separate load if item 2 was cancelled early):
     let it reach serving. Sample Windows free RAM every 3 s, and record the host RAM it uses
     (`free -g` "used" plus shared) as `ramNeedGB`, rounded up.
  8. PC-restart behaviour: only with Jay's OK. Run `wsl --shutdown` from Windows, start WSL,
     and check that `systemctl --user is-active llama-swap` is active with `/running` empty.
     Without an OK, check `systemctl --user is-enabled llama-swap` and that the unit has no
     preload.

- [ ] **Step 5: Roll back if anything in Step 4 failed and cannot be fixed in the session**

```bash
cp ~/.config/systemd/user/llama-swap.service.bak-* ~/.config/systemd/user/llama-swap.service  # newest backup
cp ~/llama-swap/config.yaml.bak-before-frozen ~/llama-swap/config.yaml
systemctl --user daemon-reload && systemctl --user restart llama-swap
```

- [ ] **Step 6: Record and commit** the measured `ramNeedGB` in the example config, and the
  results doc.

```bash
git add engines/config/config.example.yaml docs/research/own-switcher-acceptance-*.md
git commit -m "docs(engines): live acceptance results for the frozen switcher"
```

---

### Task 9: Reviews, PR, merge

- [ ] **Step 1: Open the PR** from `feat/own-switcher` to `mtp-upstream-merge`. Title:
  `feat: own model switcher part 1 (frozen llama-swap + NInfer, latest wins, memory gate)`.
  The body lists the patches, the test results and the acceptance numbers, and ends with the
  Claude Code attribution line.
- [ ] **Step 2: Review round 1.** One reviewer on the most capable model, read-only and with no
  git writes. It reads the spec, this plan and the diff, excluding the vendored upstream files
  except the patched ones: `git diff origin/mtp-upstream-merge...feat/own-switcher -- engines/llama-swap/internal/router engines/llama-swap/internal/memgate engines/llama-swap/internal/config engines/llama-swap/internal/server/filters.go engines/llama-swap/internal/swaputil/superseded.go engines/llama-swap/config-schema.json engines/adapters scripts/engines tests/engines`.
  Fix every must-fix.
- [ ] **Step 3: Review round 2** on the fixes. Fix any remaining must-fix, at two rounds at most.
- [ ] **Step 4: Ask Jay to merge.** Merge only on his yes, when both reviews pass and the
  acceptance list passed. Squash-merge. Then the box goes back to `mtp-upstream-merge`
  (`git checkout mtp-upstream-merge && git pull`), which is the same code.
- [ ] **Step 5: Update memory** (`project-model-switcher-and-llamacpp.md`) with the merge
  commit, the box state and the rollback path.
