package router

// FreeToken patch P7: unload only if idle.

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

func TestUnloadIfIdle_RefusesWhileARequestIsServed(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	a.setState(process.StateReady)
	a.serveBlock = make(chan struct{})
	w, done := serveAsync(g, "a")
	waitSignal(t, a.serveStarted, "a serve")

	var busy BusyError
	if err := g.UnloadIfIdle("a"); !errors.As(err, &busy) || busy.Model != "a" {
		t.Fatalf("UnloadIfIdle(a) while serving = %v, want BusyError", err)
	}
	if a.stopCalls.Load() != 0 || a.State() != process.StateReady {
		t.Fatalf("a was stopped (%d stops, state %v) while answering", a.stopCalls.Load(), a.State())
	}

	close(a.serveBlock)
	waitSignal(t, done, "a request")
	if w.Code != http.StatusOK {
		t.Fatalf("the served request failed: %d", w.Code)
	}
	// trackedServe's ServeDone is received by the run loop before ServeHTTP
	// returns, so the next unload sees the model idle.
	if err := g.UnloadIfIdle("a"); err != nil {
		t.Fatalf("UnloadIfIdle(a) once idle = %v", err)
	}
	if a.stopCalls.Load() == 0 || a.stoppedWhileServing.Load() {
		t.Fatalf("stops=%d stoppedWhileServing=%v", a.stopCalls.Load(), a.stoppedWhileServing.Load())
	}
}

func TestUnloadIfIdle_RefusesWhileARequestWaitsForTheModelToLoad(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	_, done := serveAsync(g, "a") // a is stopped: this request starts a swap and waits on it
	waitSignal(t, a.runStarted, "a start")

	var busy BusyError
	if err := g.UnloadIfIdle("a"); !errors.As(err, &busy) {
		t.Fatalf("UnloadIfIdle(a) while a request waits for it = %v, want BusyError", err)
	}
	if a.stopCalls.Load() != 0 {
		t.Fatalf("a was stopped under a waiting request")
	}
	a.markReady()
	waitSignal(t, done, "a request")
}

func TestUnloadIfIdle_IdleModelIsStoppedAndOthersUntouched(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	a.setState(process.StateReady)
	if err := g.UnloadIfIdle("a"); err != nil {
		t.Fatalf("UnloadIfIdle(a) = %v", err)
	}
	if a.stopCalls.Load() == 0 || b.stopCalls.Load() != 0 {
		t.Fatalf("a stops=%d b stops=%d", a.stopCalls.Load(), b.stopCalls.Load())
	}
}

func TestUnloadIfIdle_BusyErrorBody(t *testing.T) {
	e := BusyError{Model: "a"}
	if e.StatusCode() != http.StatusConflict {
		t.Fatalf("status %d", e.StatusCode())
	}
	if body := string(e.Body()); !strings.Contains(body, `"code":"busy"`) {
		t.Fatalf("body %s", body)
	}
}
