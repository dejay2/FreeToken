package router

// FreeToken patch P5: a config reload reconfigures the local router in place.
// Models whose entry did not change keep their process (and their in-flight
// requests); changed and removed models are stopped.

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
)

func p5Conf(models map[string]config.ModelConfig, groups map[string][]string) config.Config {
	gs := make(map[string]config.GroupConfig, len(groups))
	for id, members := range groups {
		gs[id] = config.GroupConfig{Swap: true, Exclusive: true, Members: members}
	}
	return config.Config{HealthCheckTimeout: 5, UnloadTimeout: 5, Routing: groupRouting(gs), Models: models}
}

type fakeFactory struct {
	mu   sync.Mutex
	made map[string]*fakeProcess
}

func (f *fakeFactory) build(_ context.Context, id string, _ config.ModelConfig) (process.Process, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.made == nil {
		f.made = map[string]*fakeProcess{}
	}
	p := newFakeProcess(id)
	f.made[id] = p
	return p, nil
}

func newReconfigGroup(t *testing.T, conf config.Config, procs map[string]process.Process) (*Group, *fakeFactory) {
	t.Helper()
	g := newTestGroup(t, conf, procs)
	ff := &fakeFactory{}
	g.factory = ff.build
	g.plannerFor = groupPlannerFor
	return g, ff
}

func procOf(g *Group, id string) process.Process {
	_, procs := g.snapshot()
	return procs[id]
}

func TestReconfigure_UnchangedLoadedModelSurvives(t *testing.T) {
	a, b := newFakeProcess("a"), newFakeProcess("b")
	a.setState(process.StateReady)
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b"}}, map[string][]string{"g": {"a", "b"}})
	g, ff := newReconfigGroup(t, conf, map[string]process.Process{"a": a, "b": b})

	next := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b --new"}}, map[string][]string{"g": {"a", "b"}})
	plan, err := g.PrepareReconfigure(next)
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	if !slices.Equal(plan.Kept, []string{"a"}) || !slices.Equal(plan.Stopped, []string{"b"}) || !slices.Equal(plan.Added, []string{"b"}) {
		t.Fatalf("plan kept=%v stopped=%v added=%v", plan.Kept, plan.Stopped, plan.Added)
	}
	plan.Commit()

	if a.stopCalls.Load() != 0 || a.State() != process.StateReady {
		t.Fatalf("a was touched: stops=%d state=%s", a.stopCalls.Load(), a.State())
	}
	if procOf(g, "a") != process.Process(a) {
		t.Fatalf("a's process was replaced")
	}
	if procOf(g, "b") != process.Process(ff.made["b"]) {
		t.Fatalf("b was not rebuilt from its new entry")
	}
	if b.stopCalls.Load() == 0 {
		t.Errorf("the old b process was never stopped")
	}
	w, done := serveAsync(g, "a")
	waitSignal(t, done, "request to a")
	if w.Code != http.StatusOK || w.Body.String() != "ok:a" {
		t.Fatalf("a after reload: code=%d body=%q", w.Code, w.Body.String())
	}
}

func TestReconfigure_RequestInFlightOnKeptModelFinishes(t *testing.T) {
	a, b := newFakeProcess("a"), newFakeProcess("b")
	a.setState(process.StateReady)
	a.serveBlock = make(chan struct{})
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b"}}, map[string][]string{"g": {"a", "b"}})
	g, ff := newReconfigGroup(t, conf, map[string]process.Process{"a": a, "b": b})

	w, done := serveAsync(g, "a")
	waitSignal(t, a.serveStarted, "a serving")
	plan, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b2"}}, map[string][]string{"g": {"a", "b"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	plan.Commit()
	close(a.serveBlock)
	waitSignal(t, done, "a request")
	if w.Code != http.StatusOK || a.stoppedWhileServing.Load() {
		t.Fatalf("a's request: code=%d stoppedWhileServing=%v", w.Code, a.stoppedWhileServing.Load())
	}
	// The in-flight count survived the reload: once a drains, a pick of b evicts a.
	w2, done2 := serveAsync(g, "b")
	waitSignal(t, ff.made["b"].runStarted, "new b start")
	ff.made["b"].markReady()
	waitSignal(t, done2, "b request")
	if w2.Code != http.StatusOK || a.State() != process.StateStopped {
		t.Fatalf("b: code=%d, a state=%s", w2.Code, a.State())
	}
}

func TestReconfigure_ChangedLoadedModelIsStopped(t *testing.T) {
	a := newFakeProcess("a")
	a.setState(process.StateReady)
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}}, map[string][]string{"g": {"a"}})
	g, ff := newReconfigGroup(t, conf, map[string]process.Process{"a": a})
	plan, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a --faster"}}, map[string][]string{"g": {"a"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	plan.Commit()
	if a.stopCalls.Load() == 0 || a.State() != process.StateStopped {
		t.Fatalf("old a: stops=%d state=%s", a.stopCalls.Load(), a.State())
	}
	if got := ff.made["a"].State(); got != process.StateStopped {
		t.Fatalf("new a must wait for its next load, state=%s", got)
	}
}

func TestReconfigure_RemovedModelIsStopped(t *testing.T) {
	a, b := newFakeProcess("a"), newFakeProcess("b")
	b.setState(process.StateReady)
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b"}}, map[string][]string{"g": {"a", "b"}})
	g, _ := newReconfigGroup(t, conf, map[string]process.Process{"a": a, "b": b})
	plan, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}}, map[string][]string{"g": {"a"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	plan.Commit()
	if b.stopCalls.Load() == 0 {
		t.Fatalf("removed b was not stopped")
	}
	if g.Handles("b") {
		t.Fatalf("the router still handles removed b")
	}
	if _, ok := g.RunningModels()["b"]; ok {
		t.Fatalf("RunningModels still lists b")
	}
	if a.stopCalls.Load() != 0 {
		t.Fatalf("unchanged a was stopped")
	}
}

func TestReconfigure_InvalidConfigKeepsEverything(t *testing.T) {
	a, b := newFakeProcess("a"), newFakeProcess("b")
	a.setState(process.StateReady)
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b"}}, map[string][]string{"g": {"a", "b"}})
	g, ff := newReconfigGroup(t, conf, map[string]process.Process{"a": a, "b": b})
	bad := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b2"}}, map[string][]string{"g": {"a", "b"}, "h": {"b"}})
	if _, err := g.PrepareReconfigure(bad); err == nil {
		t.Fatalf("a model in two groups must be refused")
	}
	if a.stopCalls.Load() != 0 || a.State() != process.StateReady || procOf(g, "b") != process.Process(b) || len(ff.made) != 0 {
		t.Fatalf("a refused reload changed the router")
	}
}

func TestReconfigure_AbortLeavesTheTableAlone(t *testing.T) {
	a := newFakeProcess("a")
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}}, map[string][]string{"g": {"a"}})
	g, _ := newReconfigGroup(t, conf, map[string]process.Process{"a": a})
	plan, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a2"}}, map[string][]string{"g": {"a"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	plan.Abort()
	plan.Commit() // a no-op after Abort
	if procOf(g, "a") != process.Process(a) || a.stopCalls.Load() != 0 {
		t.Fatalf("Abort changed the router")
	}
}

func TestReconfigure_UnsupportedWithoutAPlannerFactory(t *testing.T) {
	g := newTestBase(t, map[string]process.Process{"a": newFakeProcess("a")}, &stubPlanner{})
	if _, err := g.PrepareReconfigure(config.Config{}); !errors.Is(err, ErrReconfigureUnsupported) {
		t.Fatalf("err=%v want ErrReconfigureUnsupported", err)
	}
}

// FreeToken patch P5: two plans prepared from the same table; the first
// commit wins and the second is refused as stale, so its processes are
// thrown away and nothing of the first plan is lost or cancelled.
func TestReconfigure_StalePlanIsRefused(t *testing.T) {
	a, b := newFakeProcess("a"), newFakeProcess("b")
	a.setState(process.StateReady)
	conf := p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b"}}, map[string][]string{"g": {"a", "b"}})
	g := newTestGroup(t, conf, map[string]process.Process{"a": a, "b": b})
	type built struct {
		ctx context.Context
		p   *fakeProcess
	}
	var mu sync.Mutex
	var made []built
	g.factory = func(ctx context.Context, id string, _ config.ModelConfig) (process.Process, error) {
		mu.Lock()
		defer mu.Unlock()
		p := newFakeProcess(id)
		made = append(made, built{ctx, p})
		return p, nil
	}
	g.plannerFor = groupPlannerFor

	plan1, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b1"}}, map[string][]string{"g": {"a", "b"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure 1: %v", err)
	}
	plan2, err := g.PrepareReconfigure(p5Conf(map[string]config.ModelConfig{"a": {Cmd: "run-a"}, "b": {Cmd: "run-b2"}}, map[string][]string{"g": {"a", "b"}}))
	if err != nil {
		t.Fatalf("PrepareReconfigure 2: %v", err)
	}
	if len(made) != 2 {
		t.Fatalf("made %d processes, want 2", len(made))
	}
	b1, b2 := made[0], made[1]

	if err := plan1.Commit(); err != nil {
		t.Fatalf("plan1.Commit: %v", err)
	}
	if err := plan2.Commit(); !errors.Is(err, ErrStaleReconfigure) {
		t.Fatalf("plan2.Commit = %v, want ErrStaleReconfigure", err)
	}

	if procOf(g, "b") != process.Process(b1.p) || procOf(g, "a") != process.Process(a) {
		t.Fatalf("the table does not hold plan1's processes")
	}
	if b1.ctx.Err() != nil {
		t.Fatalf("plan1's b was cancelled by the refused plan")
	}
	if b2.ctx.Err() == nil {
		t.Fatalf("the refused plan's b was leaked (its context is still live)")
	}
	g.stateMu.RLock()
	_, hasCancel := g.procCancels["b"]
	g.stateMu.RUnlock()
	if !hasCancel {
		t.Fatalf("plan1's b lost its cancel handle")
	}
	if a.stopCalls.Load() != 0 {
		t.Fatalf("a was stopped")
	}
	w, done := serveAsync(g, "b")
	waitSignal(t, b1.p.runStarted, "plan1's b start")
	b1.p.markReady()
	waitSignal(t, done, "b request")
	if w.Code != http.StatusOK || b2.p.runCalls.Load() != 0 {
		t.Fatalf("b: code=%d, refused b runs=%d", w.Code, b2.p.runCalls.Load())
	}
}

// FreeToken patch P5: the config body the control panel generates (macros,
// ${env.HOME}, env lists, clampParams, aliases), loaded from YAML. Item 9 of
// the 2026-09-25 tidy: a reload after a change to the comment header only
// must keep every model. The comparison is reflect.DeepEqual on the parsed
// ModelConfig, which carries nothing from comments, so it does.
const p5GeneratedBody = `
healthCheckTimeout: 600
memoryGate:
  probe: windows
  floorGB: 6
  waitSeconds: 300
routing:
  scheduler:
    settings:
      fifo:
        latestWins: true
macros:
  ft: "${env.HOME}/FreeToken/engines/adapters/freetoken.sh"
  ninfer: "${env.HOME}/FreeToken/engines/adapters/ninfer.sh ${env.HOME}/ninfer-serve"
models:
  # --- model qwen ---
  "qwen":
    name: "Qwen (FreeToken)"
    cmd: "${ft} --profile model-qwen ${env.HOME}/models/Qwen"
    proxy: "http://127.0.0.1:2020"
    checkEndpoint: "/ready?model=Qwen"
    unloadTimeout: 180
    ramNeedGB: 62
    ttl: 0
    aliases: ["Qwen-NVFP4"]
  # --- model quasar ---
  "quasar":
    name: "Quasar (NInfer)"
    cmd: "${ninfer} ${env.HOME}/models/quasar quasar --host 127.0.0.1 --port 8090 --max-concurrency 4"
    proxy: "http://127.0.0.1:8090"
    checkEndpoint: "/health"
    useModelName: "quasar"
    unloadTimeout: 30
    ramNeedGB: 20.5
    ttl: 600
    aliases: []
    env: ["CUDA_VISIBLE_DEVICES=0"]
    filters:
      clampParams: {"temperature": [0, 2], "top_k": [0, 20], "top_p": [0, 1]}
  # --- model ported ---
  "ported":
    cmd: "llama-server --port ${PORT} -m ${env.HOME}/m.gguf"
`

func loadYAML(t *testing.T, text string) config.Config {
	t.Helper()
	cfg, err := config.LoadConfigFromReader(strings.NewReader(text))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return cfg
}

func TestReconfigure_HeaderOnlyChangeKeepsEveryModel(t *testing.T) {
	t.Setenv("HOME", "/home/test")
	before := loadYAML(t, "# generated by the control panel from registry.json; edits here are overwritten\n"+p5GeneratedBody)
	after := loadYAML(t, "# generated by the control panel from registry.json; edits here are overwritten\n# saved 2026-09-25 by a newer panel\n"+p5GeneratedBody)

	g, err := NewGroup(before, logmon.NewWriter(io.Discard), logmon.NewWriter(io.Discard))
	if err != nil {
		t.Fatalf("NewGroup: %v", err)
	}
	t.Cleanup(func() { _ = g.Shutdown(time.Second) })
	plan, err := g.PrepareReconfigure(after)
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	defer plan.Abort()
	if want := []string{"ported", "quasar", "qwen"}; !slices.Equal(plan.Kept, want) || len(plan.Stopped) != 0 || len(plan.Added) != 0 {
		t.Fatalf("kept=%v stopped=%v added=%v; want every model kept", plan.Kept, plan.Stopped, plan.Added)
	}
}

// FreeToken patch P5, item 10: ${PORT} is allocated from startPort in sorted
// model-ID order at every load, so an unchanged model set gives every model
// the same port and a ${PORT} model is kept. Adding a model that sorts before
// it moves its port; that entry then really differs (its cmd and proxy name
// another port, and the old port goes to the new model), so it is restarted.
func TestReconfigure_PortMacroUnchangedModelIsKept(t *testing.T) {
	const body = `
models:
  "b-ported":
    cmd: "llama-server --port ${PORT}"
  "c-fixed":
    cmd: "llama-server --port 9000"
    proxy: "http://127.0.0.1:9000"
`
	before := loadYAML(t, body)
	g, err := NewGroup(before, logmon.NewWriter(io.Discard), logmon.NewWriter(io.Discard))
	if err != nil {
		t.Fatalf("NewGroup: %v", err)
	}
	t.Cleanup(func() { _ = g.Shutdown(time.Second) })

	plan, err := g.PrepareReconfigure(loadYAML(t, "# touched\n"+body))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	if !slices.Equal(plan.Kept, []string{"b-ported", "c-fixed"}) || len(plan.Stopped) != 0 {
		t.Fatalf("unchanged set: kept=%v stopped=%v; want both kept", plan.Kept, plan.Stopped)
	}
	plan.Abort()

	plan, err = g.PrepareReconfigure(loadYAML(t, body+`  "a-new":
    cmd: "llama-server --port ${PORT}"
`))
	if err != nil {
		t.Fatalf("PrepareReconfigure: %v", err)
	}
	defer plan.Abort()
	if !slices.Equal(plan.Kept, []string{"c-fixed"}) || !slices.Equal(plan.Stopped, []string{"b-ported"}) {
		t.Fatalf("port shift: kept=%v stopped=%v; want c-fixed kept, b-ported restarted", plan.Kept, plan.Stopped)
	}
}
