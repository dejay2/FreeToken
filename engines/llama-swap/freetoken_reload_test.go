package main

// FreeToken patch P5: a reload asked for while one runs is not dropped; it
// causes exactly one more reload once the running one finishes.

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestReloadCoalescer_RequestDuringAReloadRunsOnceMore(t *testing.T) {
	var c reloadCoalescer
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	fn := func() {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
	}
	firstDone := make(chan struct{})
	go func() { c.run(fn); close(firstDone) }()
	<-started

	// Three requests land mid-reload: they return at once and fold into one.
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); c.run(fn) }()
	}
	wg.Wait()
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls=%d while the first reload is still running, want 1", got)
	}
	close(release)
	select {
	case <-firstDone:
	case <-time.After(5 * time.Second):
		t.Fatal("the running reload never finished")
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls=%d, want 2 (the running reload plus exactly one more)", got)
	}

	// Idle again: the next request runs straight away.
	c.run(fn)
	if got := calls.Load(); got != 3 {
		t.Fatalf("calls=%d after an idle request, want 3", got)
	}
}

// FreeToken patch P5 (fix round 1): stop drops a pending reload, waits for
// the running one, and makes every later request a no-op.
func TestReloadCoalescer_StopWaitsDropsAndRefuses(t *testing.T) {
	var c reloadCoalescer
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	fn := func() {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
	}
	firstDone := make(chan struct{})
	go func() { c.run(fn); close(firstDone) }()
	<-started
	c.run(fn) // pending, folded into one more reload

	stopped := make(chan struct{})
	go func() { c.stop(); close(stopped) }()
	select {
	case <-stopped:
		t.Fatal("stop returned while a reload was still running")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("stop never returned")
	}
	<-firstDone
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls=%d, want 1: the pending reload must be dropped by stop", got)
	}
	c.run(fn)
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls=%d: a reload after stop must not run", got)
	}
	c.stop() // idle stop returns at once
}
