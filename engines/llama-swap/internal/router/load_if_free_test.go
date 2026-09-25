package router

// FreeToken patch P8: LoadIfFree never preempts another app's load.

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

func wantNotFree(t *testing.T, err error, other string) {
	t.Helper()
	var nf scheduler.NotFreeError
	if !errors.As(err, &nf) || nf.Other != other {
		t.Fatalf("LoadIfFree = %v, want NotFreeError naming %s", err, other)
	}
}

func TestLoadIfFree_RefusesWhileAnotherModelLoads(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	w, done := serveAsync(g, "b") // another app's pick of b is loading
	waitSignal(t, b.runStarted, "b start")

	wantNotFree(t, g.LoadIfFree(context.Background(), "a"), "b")
	if a.runCalls.Load() != 0 || b.stopCalls.Load() != 0 {
		t.Fatalf("a runs=%d b stops=%d: an if-free load preempted b", a.runCalls.Load(), b.stopCalls.Load())
	}
	b.markReady()
	waitSignal(t, done, "b request")
	if w.Code != http.StatusOK {
		t.Fatalf("b code=%d body=%q", w.Code, w.Body.String())
	}
}

// A load parked in the memory gate has no process state yet (absent from
// /running); it must still count.
func TestLoadIfFree_RefusesWhileAnotherModelWaitsInTheMemoryGate(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {RamNeedGB: 60}})
	var room atomic.Bool
	probe, probeCalled := gatedProbe(&room)
	g.memGate = &memgate.Gate{Probe: probe, FloorGB: 6, Wait: time.Hour, Poll: 5 * time.Millisecond}
	w, done := serveAsync(g, "b")
	waitSignal(t, probeCalled, "memory gate probe")
	if b.State() != process.StateStopped {
		t.Fatalf("b state=%s, want stopped while gated", b.State())
	}

	wantNotFree(t, g.LoadIfFree(context.Background(), "a"), "b")
	if a.runCalls.Load() != 0 {
		t.Fatalf("a was started beside a gated b")
	}
	room.Store(true)
	waitSignal(t, b.runStarted, "b start")
	b.markReady()
	waitSignal(t, done, "b request")
	if w.Code != http.StatusOK {
		t.Fatalf("b code=%d body=%q: the gated load was cancelled", w.Code, w.Body.String())
	}
}

func TestLoadIfFree_RefusesWhileAnotherModelIsReady(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	b.setState(process.StateReady)
	wantNotFree(t, g.LoadIfFree(context.Background(), "a"), "b")
	if a.runCalls.Load() != 0 || b.stopCalls.Load() != 0 {
		t.Fatalf("a runs=%d b stops=%d", a.runCalls.Load(), b.stopCalls.Load())
	}
}

func TestLoadIfFree_LoadsOnAnEmptyCardAndHandsTheGrantBack(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	errc := make(chan error, 1)
	go func() { errc <- g.LoadIfFree(context.Background(), "a") }()
	waitSignal(t, a.runStarted, "a start")
	a.markReady()
	if err := <-errc; err != nil {
		t.Fatalf("LoadIfFree(a) = %v", err)
	}
	// The target itself being ready is not "something else".
	if err := g.LoadIfFree(context.Background(), "a"); err != nil {
		t.Fatalf("LoadIfFree(a) when a is ready = %v", err)
	}
	if err := g.UnloadIfIdle("a"); err != nil {
		t.Fatalf("UnloadIfIdle(a) after the loads = %v: a grant was not handed back", err)
	}
	_ = b
}

// FreeToken patch P8: after the only caller of a load cancels it, the swap
// holds nobody, so an if-idle unload stops it; a load another app joined is
// still refused as busy.
func TestUnloadIfIdle_StopsALoadWhoseCallerCancelled(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- g.LoadIfFree(ctx, "a") }()
	waitSignal(t, a.runStarted, "a start")
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Fatalf("LoadIfFree after cancel = %v", err)
	}
	var err error
	for i := 0; i < 100; i++ { // the cancel reaches the run loop on its own
		if err = g.UnloadIfIdle("a"); err == nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err != nil || a.stopCalls.Load() == 0 {
		t.Fatalf("UnloadIfIdle(a) = %v, stops=%d: an orphaned load was kept", err, a.stopCalls.Load())
	}
}

func TestUnloadIfIdle_KeepsALoadAnotherAppJoined(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- g.LoadIfFree(ctx, "a") }()
	waitSignal(t, a.runStarted, "a start")
	w, done := serveAsync(g, "a") // another app joins the load
	waitProcessed(t, g.testProcessed, 2)
	cancel()
	<-errc
	var busy BusyError
	if err := g.UnloadIfIdle("a"); !errors.As(err, &busy) || a.stopCalls.Load() != 0 {
		t.Fatalf("UnloadIfIdle(a) = %v stops=%d: the other app's load was cancelled", err, a.stopCalls.Load())
	}
	a.markReady()
	waitSignal(t, done, "other app's request")
	if w.Code != http.StatusOK {
		t.Fatalf("other app code=%d", w.Code)
	}
}
