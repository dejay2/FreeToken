package router

// FreeToken patch P2: router-level tests for the memory gate.

import (
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// TestBase_MemGate_NotEnoughMemoryReturns503: a model configured with a
// ramNeedGB the gate's probe never satisfies gets NotEnoughMemoryError (HTTP
// 503) back once the gate's wait budget runs out, and its process is never
// started — the gate sits ahead of EnsureReady in doSwap.
func TestBase_MemGate_NotEnoughMemoryReturns503(t *testing.T) {
	a := newFakeProcess("a")

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a"}},
		}),
		Models: map[string]config.ModelConfig{
			"a": {RamNeedGB: 60},
		},
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a})
	g.memGate = &memgate.Gate{
		Probe:   func(context.Context) (float64, error) { return 10, nil }, // always short
		FloorGB: 6,
		Wait:    30 * time.Millisecond,
		Poll:    5 * time.Millisecond,
	}

	w, done := serveAsync(g, "a")
	waitSignal(t, done, "a request")

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("a: code=%d want 503 body=%q", w.Code, w.Body.String())
	}
	if got := a.runCalls.Load(); got != 0 {
		t.Errorf("a.runCalls=%d want 0: a gated model must never start", got)
	}
	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state=%s want stopped", got)
	}
}

// TestBase_MemGate_SupersedeEndsWaitPromptly: a is mid-wait in the memory
// gate (probe never has room) when b is requested. Latest wins must cancel
// a's swap the same way it cancels one parked in EnsureReady: the wait ends
// at once via ctx, a's caller gets the 409 superseded error (not the gate's
// 503), a's process never starts, and b proceeds normally.
func TestBase_MemGate_SupersedeEndsWaitPromptly(t *testing.T) {
	a := newFakeProcess("a")
	pb := newFakeProcess("b")

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a", "b"}},
		}),
		Models: map[string]config.ModelConfig{
			"a": {RamNeedGB: 60},
		},
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a, "b": pb})

	probeCalled := make(chan struct{})
	var once sync.Once
	g.memGate = &memgate.Gate{
		Probe: func(context.Context) (float64, error) {
			once.Do(func() { close(probeCalled) })
			return 1, nil // never enough
		},
		FloorGB: 6,
		// Long enough that only the supersede's ctx cancel — not the wait
		// budget or the poll tick — can end the wait within this test.
		Wait: time.Hour,
		Poll: time.Hour,
	}

	w1, done1 := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, probeCalled, "memory gate probe")

	w2, done2 := serveAsync(g, "b") // supersedes a's swap
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, done1, "a request")

	if w1.Code != http.StatusConflict {
		t.Fatalf("a: code=%d want 409 body=%q", w1.Code, w1.Body.String())
	}
	if got := a.runCalls.Load(); got != 0 {
		t.Errorf("a.runCalls=%d want 0: a gated model must never start", got)
	}
	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state=%s want stopped", got)
	}

	waitSignal(t, pb.runStarted, "b start")
	pb.markReady()
	select {
	case <-done2:
	case <-time.After(time.Second):
		t.Fatal("b request did not complete")
	}
	if w2.Code != http.StatusOK {
		t.Fatalf("b: code=%d want 200 body=%q", w2.Code, w2.Body.String())
	}
}
