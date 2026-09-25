package router

// FreeToken patch P2: router-level tests for the memory gate.

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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
	if got := w.Header().Get("Retry-After"); got != "30" {
		t.Errorf("Retry-After=%q want 30", got)
	}
	if body := w.Body.String(); !strings.Contains(body, "not_enough_memory") {
		t.Errorf("body %q lacks not_enough_memory", body)
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

// gatedProbe returns a probe that has no room until room is set, and a
// channel closed on its first call.
func gatedProbe(room *atomic.Bool) (memgate.Probe, chan struct{}) {
	called := make(chan struct{})
	var once sync.Once
	return func(context.Context) (float64, error) {
		once.Do(func() { close(called) })
		if room.Load() {
			return 100, nil
		}
		return 1, nil
	}, called
}

// TestBase_MemGate_UnloadDuringWaitNeverStarts: final review 2026-09-24. a is
// parked in the memory gate when it is unloaded. OnUnload must cancel the
// swap, so when room appears later a never starts.
func TestBase_MemGate_UnloadDuringWaitNeverStarts(t *testing.T) {
	a := newFakeProcess("a")
	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a"}},
		}),
		Models: map[string]config.ModelConfig{"a": {RamNeedGB: 60}},
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a})
	var room atomic.Bool
	probe, probeCalled := gatedProbe(&room)
	g.memGate = &memgate.Gate{Probe: probe, FloorGB: 6, Wait: time.Hour, Poll: 5 * time.Millisecond}

	w1, done1 := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, probeCalled, "memory gate probe")

	g.Unload(time.Second, "a")
	waitSignal(t, done1, "a request")
	if w1.Code == http.StatusOK {
		t.Fatalf("a: code=200 after unload, body=%q", w1.Body.String())
	}

	room.Store(true)
	time.Sleep(100 * time.Millisecond) // 20 poll ticks: an uncancelled wait would start a
	if got := a.runCalls.Load(); got != 0 {
		t.Errorf("a.runCalls=%d want 0: an unloaded gated swap must never start", got)
	}
	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state=%s want stopped", got)
	}
}

// TestBase_MemGate_UnloadThenPickOtherOnlyOtherRuns: the reviewer's second
// repro. a is parked in the gate, a is unloaded, b (same exclusive group) is
// picked and becomes ready, then room appears: only b may be running.
func TestBase_MemGate_UnloadThenPickOtherOnlyOtherRuns(t *testing.T) {
	a := newFakeProcess("a")
	pb := newFakeProcess("b")
	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a", "b"}},
		}),
		Models: map[string]config.ModelConfig{"a": {RamNeedGB: 60}},
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a, "b": pb})
	var room atomic.Bool
	probe, probeCalled := gatedProbe(&room)
	g.memGate = &memgate.Gate{Probe: probe, FloorGB: 6, Wait: time.Hour, Poll: 5 * time.Millisecond}

	_, done1 := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, probeCalled, "memory gate probe")

	g.Unload(time.Second, "a")
	waitSignal(t, done1, "a request")

	w2, done2 := serveAsync(g, "b")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, pb.runStarted, "b start")
	pb.markReady()
	waitSignal(t, done2, "b request")
	if w2.Code != http.StatusOK {
		t.Fatalf("b: code=%d want 200 body=%q", w2.Code, w2.Body.String())
	}

	room.Store(true)
	time.Sleep(100 * time.Millisecond)
	if got := a.runCalls.Load(); got != 0 {
		t.Errorf("a.runCalls=%d want 0: a must not start beside b", got)
	}
	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state=%s want stopped", got)
	}
	if got := pb.State(); got != process.StateReady {
		t.Errorf("b state=%s want ready", got)
	}
}

// FreeToken patch P2: a request that fails in or after the memory gate (503
// from the gate, a failed start, or a client that gives up while parked in
// the gate) must hand back its concurrency reservation. With
// concurrencyLimit 1 a leaked reservation turns the next request into an
// instant 429, so the second request below reaching 200 proves the release.
func TestBase_MemGate_ReservationReleased(t *testing.T) {
	newGated := func(t *testing.T, wait time.Duration) (*Group, *fakeProcess, *atomic.Bool, chan struct{}) {
		a := newFakeProcess("a")
		conf := config.Config{
			HealthCheckTimeout: 5,
			Routing: groupRouting(map[string]config.GroupConfig{
				"g": {Swap: true, Members: []string{"a"}},
			}),
			Models: map[string]config.ModelConfig{"a": {RamNeedGB: 60, ConcurrencyLimit: 1}},
		}
		g := newTestGroup(t, conf, map[string]process.Process{"a": a})
		room := &atomic.Bool{}
		probe, called := gatedProbe(room)
		g.memGate = &memgate.Gate{Probe: probe, FloorGB: 6, Wait: wait, Poll: 5 * time.Millisecond}
		return g, a, room, called
	}
	secondSucceeds := func(t *testing.T, g *Group, a *fakeProcess, room *atomic.Bool) {
		t.Helper()
		room.Store(true)
		w, done := serveAsync(g, "a")
		select {
		case <-a.runStarted:
			a.markReady()
		case <-done: // answered without a start: the 429 of a leaked reservation
		case <-time.After(5 * time.Second):
			t.Fatal("second request neither started a nor answered")
		}
		waitSignal(t, done, "second request")
		if w.Code != http.StatusOK {
			t.Fatalf("second request: code=%d want 200 (a leaked reservation answers 429) body=%q", w.Code, w.Body.String())
		}
	}

	t.Run("gate refuses with 503", func(t *testing.T) {
		g, a, room, _ := newGated(t, 30*time.Millisecond)
		w, done := serveAsync(g, "a")
		waitProcessed(t, g.testProcessed, 1)
		waitSignal(t, done, "first request")
		if w.Code != http.StatusServiceUnavailable {
			t.Fatalf("first request: code=%d want 503", w.Code)
		}
		waitProcessed(t, g.testProcessed, 1) // its SwapDone
		secondSucceeds(t, g, a, room)
	})

	t.Run("start fails after the gate", func(t *testing.T) {
		g, a, room, _ := newGated(t, time.Hour)
		room.Store(true)
		a.mu.Lock()
		a.ensureErr = errors.New("boom")
		a.mu.Unlock()
		w, done := serveAsync(g, "a")
		waitProcessed(t, g.testProcessed, 1)
		waitSignal(t, done, "first request")
		if w.Code == http.StatusOK {
			t.Fatalf("first request: code=200, want the start error")
		}
		waitProcessed(t, g.testProcessed, 1) // its SwapDone
		a.mu.Lock()
		a.ensureErr = nil
		a.mu.Unlock()
		secondSucceeds(t, g, a, room)
	})

	t.Run("client cancels while parked in the gate", func(t *testing.T) {
		g, a, room, called := newGated(t, time.Hour)
		ctx, cancel := context.WithCancel(t.Context())
		done := make(chan struct{})
		go func() {
			g.ServeHTTP(httptest.NewRecorder(), newRequestCtx(ctx, "a"))
			close(done)
		}()
		waitProcessed(t, g.testProcessed, 1)
		waitSignal(t, called, "memory gate probe")
		cancel()
		waitSignal(t, done, "cancelled request")
		waitProcessed(t, g.testProcessed, 1) // OnCancel
		// The parked swap is still in flight; the second request joins it.
		secondSucceeds(t, g, a, room)
	})
}
