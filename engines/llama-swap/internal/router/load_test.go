package router

// FreeToken patch P6: Load goes through the scheduler like a chat request.

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

func twoModelGroup(t *testing.T, models map[string]config.ModelConfig) (*Group, *fakeProcess, *fakeProcess) {
	t.Helper()
	a, b := newFakeProcess("a"), newFakeProcess("b")
	conf := p5Conf(models, map[string][]string{"g": {"a", "b"}})
	return newTestGroup(t, conf, map[string]process.Process{"a": a, "b": b}), a, b
}

func TestLoad_StartsTheModelAndHandsTheGrantBack(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	errc := make(chan error, 1)
	go func() { errc <- g.Load(context.Background(), "a") }()
	waitSignal(t, a.runStarted, "a start")
	a.markReady()
	if err := <-errc; err != nil {
		t.Fatalf("Load(a) = %v", err)
	}
	// No in-flight count was left behind: a pick of b evicts a straight away.
	w, done := serveAsync(g, "b")
	waitSignal(t, b.runStarted, "b start")
	b.markReady()
	waitSignal(t, done, "b request")
	if w.Code != http.StatusOK || a.stopCalls.Load() == 0 {
		t.Fatalf("b code=%d, a stops=%d: Load leaked an in-flight count", w.Code, a.stopCalls.Load())
	}
}

func TestLoad_ReadyModelReturnsAtOnce(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	a.setState(process.StateReady)
	if err := g.Load(context.Background(), "a"); err != nil || a.runCalls.Load() != 0 {
		t.Fatalf("err=%v runCalls=%d", err, a.runCalls.Load())
	}
}

func TestLoad_SupersededByANewerPick(t *testing.T) {
	g, a, b := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	errc := make(chan error, 1)
	go func() { errc <- g.Load(context.Background(), "a") }()
	waitSignal(t, a.runStarted, "a start")
	_, done := serveAsync(g, "b")
	var se swaputil.SupersededError
	select {
	case err := <-errc:
		if !errors.As(err, &se) {
			t.Fatalf("Load(a) = %v, want SupersededError", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Load(a) did not return after b was picked")
	}
	waitSignal(t, b.runStarted, "b start")
	b.markReady()
	waitSignal(t, done, "b request")
}

func TestLoad_NotEnoughMemory(t *testing.T) {
	g, a, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {RamNeedGB: 60}, "b": {}})
	g.memGate = &memgate.Gate{
		Probe:   func(context.Context) (float64, error) { return 10, nil },
		FloorGB: 6,
		Wait:    30 * time.Millisecond,
		Poll:    5 * time.Millisecond,
	}
	err := g.Load(context.Background(), "a")
	var ne memgate.NotEnoughMemoryError
	if !errors.As(err, &ne) || a.runCalls.Load() != 0 {
		t.Fatalf("err=%v runCalls=%d", err, a.runCalls.Load())
	}
}

func TestLoad_UnknownModel(t *testing.T) {
	g, _, _ := twoModelGroup(t, map[string]config.ModelConfig{"a": {}, "b": {}})
	if err := g.Load(context.Background(), "nope"); !errors.Is(err, scheduler.ErrModelNotFound) {
		t.Fatalf("err=%v", err)
	}
}
