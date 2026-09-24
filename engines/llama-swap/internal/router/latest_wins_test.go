package router

// FreeToken patch P1: router-level tests for "latest wins".

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// serveAsync runs one request against r in a goroutine and returns its
// recorder and a channel closed when ServeHTTP returns.
func serveAsync(r http.Handler, model string) (*httptest.ResponseRecorder, chan struct{}) {
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		r.ServeHTTP(w, newRequest(model))
		close(done)
	}()
	return w, done
}

// TestBase_LatestWins_StaleSwapDoneDoesNotFailRepick: A, then B, then A again,
// while every swap is parked stopping a slow model x. When x finally stops,
// the cancelled A and B swaps must not report SwapDone: OnSwapDone matches by
// model ID, so a stale SwapDone{a, context.Canceled} would fail the re-picked
// A request and drop its live swap, leaving a loading with nothing tracking it.
func TestBase_LatestWins_StaleSwapDoneDoesNotFailRepick(t *testing.T) {
	a := newFakeProcess("a")
	pb := newFakeProcess("b")
	x := newFakeProcess("x")
	x.markReady()
	x.stopBlock = make(chan struct{})

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a", "b", "x"}},
		}),
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a, "b": pb, "x": x})

	w1, done1 := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, x.stopStarted, "x stop (a's swap)")

	w2, done2 := serveAsync(g, "b") // supersedes a
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, done1, "first a request")
	if w1.Code != http.StatusConflict {
		t.Fatalf("first a: code=%d want 409 body=%q", w1.Code, w1.Body.String())
	}

	w3, done3 := serveAsync(g, "a") // supersedes b
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, done2, "b request")
	if w2.Code != http.StatusConflict {
		t.Fatalf("b: code=%d want 409 body=%q", w2.Code, w2.Body.String())
	}

	close(x.stopBlock) // every parked swap goroutine proceeds
	waitSignal(t, a.runStarted, "re-picked a start")

	// Leave room for a stale SwapDone from the cancelled a swap to arrive.
	select {
	case <-done3:
		t.Fatalf("re-picked a finished before a was ready: code=%d body=%q", w3.Code, w3.Body.String())
	case <-time.After(200 * time.Millisecond):
	}

	a.markReady()
	select {
	case <-done3:
	case <-time.After(time.Second):
		t.Fatal("re-picked a request did not complete")
	}
	if w3.Code != http.StatusOK || w3.Body.String() != "ok:a" {
		t.Fatalf("re-picked a: code=%d body=%q want 200 ok:a", w3.Code, w3.Body.String())
	}
	if got := a.runCalls.Load(); got != 1 {
		t.Errorf("a.runCalls=%d want 1", got)
	}
	if got := pb.runCalls.Load(); got != 0 {
		t.Errorf("b.runCalls=%d want 0 (b was superseded before starting)", got)
	}
}

// TestBase_LatestWins_CancelledSwapNeverStartsTarget: a's swap goroutine has
// passed doSwap's ctx check and is inside EnsureReady when b supersedes it.
// The cancel and a's stop both finish before EnsureReady takes its start
// decision; a must stay stopped. The fake mirrors ProcessCommand's run loop,
// which refuses a start whose ctx is already cancelled (see
// TestProcessCommand_StartRequestWithCancelledCtxIsRefused).
func TestBase_LatestWins_CancelledSwapNeverStartsTarget(t *testing.T) {
	a := newFakeProcess("a")
	a.ensureGate = make(chan struct{})
	a.ensureExit = make(chan struct{})
	pb := newFakeProcess("b")

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing: groupRouting(map[string]config.GroupConfig{
			"g": {Swap: true, Members: []string{"a", "b"}},
		}),
	}
	g := newTestGroup(t, conf, map[string]process.Process{"a": a, "b": pb})

	w1, done1 := serveAsync(g, "a")
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, a.ensureAsked, "a EnsureReady")

	w2, done2 := serveAsync(g, "b") // cancels a's swap and stops a
	waitProcessed(t, g.testProcessed, 1)
	waitSignal(t, done1, "a request")
	if w1.Code != http.StatusConflict {
		t.Fatalf("a: code=%d want 409 body=%q", w1.Code, w1.Body.String())
	}

	close(a.ensureGate)
	waitSignal(t, a.ensureExit, "a EnsureReady return")
	if got := a.runCalls.Load(); got != 0 {
		t.Fatalf("a.runCalls=%d want 0: a superseded swap started its target", got)
	}
	if got := a.State(); got != process.StateStopped {
		t.Fatalf("a state=%s want stopped", got)
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
