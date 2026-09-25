package router

// FreeToken patch P8 (round 2): a cancelled load nobody joined is aborted, and
// a chat request can ask for if-free admission with X-FreeToken-If-Free: 1.

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

// A load parked in the memory gate whose only caller cancels it must not stay
// behind as a swap: it would refuse every later if-free load (P8) and boot its
// model once room appeared.
func TestLoad_CancelledWhileGatedIsAborted(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {RamNeedGB: 60}})
	var room atomic.Bool
	probe, probeCalled := gatedProbe(&room)
	g.memGate = &memgate.Gate{Probe: probe, FloorGB: 6, Wait: time.Hour, Poll: 5 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- g.LoadIfFree(ctx, "b") }()
	waitSignal(t, probeCalled, "memory gate probe")
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Fatalf("LoadIfFree(b) after cancel = %v", err)
	}

	// The cancel is handled on the run loop before this load is admitted.
	errA := make(chan error, 1)
	go func() { errA <- g.LoadIfFree(context.Background(), "a") }()
	select {
	case <-a.runStarted:
	case err := <-errA:
		t.Fatalf("LoadIfFree(a) = %v: the cancelled gated load of b was kept", err)
	case <-time.After(2 * time.Second):
		t.Fatal("a never started")
	}
	a.markReady()
	if err := <-errA; err != nil {
		t.Fatalf("LoadIfFree(a) = %v", err)
	}
	room.Store(true)
	time.Sleep(50 * time.Millisecond)
	if b.runCalls.Load() != 0 {
		t.Fatalf("the cancelled load of b booted once room appeared")
	}
}

// A load another app joined is not aborted by the load caller's cancel.
func TestLoad_CancelKeepsASwapAnotherRequestJoined(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- g.LoadIfFree(ctx, "a") }()
	waitSignal(t, a.runStarted, "a start")
	w, done := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 2)
	cancel()
	<-errc
	a.markReady()
	waitSignal(t, done, "other app's request")
	if w.Code != http.StatusOK || a.stopCalls.Load() != 0 {
		t.Fatalf("code=%d stops=%d: the joined load was aborted", w.Code, a.stopCalls.Load())
	}
}

func ifFreeRequest(model string) *http.Request {
	r := newRequest(model)
	r.Header.Set(scheduler.IfFreeHeader, "1")
	return r
}

func TestChatIfFree_RefusedWhileAnotherModelLoads(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	wb, done := serveAsync(g, "b") // another app's pick of b is loading
	waitSignal(t, b.runStarted, "b start")

	w, refused := serveWithin(g, ifFreeRequest("a"))
	if !refused {
		t.Fatal("the if-free chat was not refused at once: it superseded b or queued")
	}
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), `"code":"busy"`) {
		t.Fatalf("if-free chat code=%d body=%q, want 409 busy", w.Code, w.Body.String())
	}
	if a.runCalls.Load() != 0 || b.stopCalls.Load() != 0 {
		t.Fatalf("a runs=%d b stops=%d: an if-free chat preempted b", a.runCalls.Load(), b.stopCalls.Load())
	}
	b.markReady()
	waitSignal(t, done, "b request")
	if wb.Code != http.StatusOK {
		t.Fatalf("b code=%d", wb.Code)
	}
}

func TestChatIfFree_ServesOnAFreeCardAndBesideItsOwnModel(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	a.markReady()
	w := httptest.NewRecorder()
	g.ServeHTTP(w, ifFreeRequest("a"))
	if w.Code != http.StatusOK {
		t.Fatalf("if-free chat on a ready target code=%d body=%q", w.Code, w.Body.String())
	}
}

// serveWithin serves r and reports whether it returned within a second.
func serveWithin(h http.Handler, r *http.Request) (*httptest.ResponseRecorder, bool) {
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(w, r)
		close(done)
	}()
	select {
	case <-done:
		return w, true
	case <-time.After(time.Second):
		return w, false
	}
}
