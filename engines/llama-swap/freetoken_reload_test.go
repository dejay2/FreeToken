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
