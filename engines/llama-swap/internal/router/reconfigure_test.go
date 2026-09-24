package router

// FreeToken patch P5: a config reload reconfigures the local router in place.
// Models whose entry did not change keep their process (and their in-flight
// requests); changed and removed models are stopped.

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"sync"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
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
