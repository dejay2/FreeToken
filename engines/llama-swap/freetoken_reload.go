package main

// FreeToken patch P5: coalescing reload guard.

import "sync"

// reloadCoalescer runs one reload at a time. Upstream dropped a reload asked
// for while another was running; the control panel writes the config and
// then waits for GET /api/config/hash to show it, so a save landing
// mid-reload must not be lost. A request during a reload now marks one more
// reload, which the running caller performs when it finishes: any number of
// requests during one reload fold into exactly one extra reload. It is also
// the single path to server.Rebuild, which keeps router reconfigures
// (PrepareReconfigure + Commit) strictly sequential.
type reloadCoalescer struct {
	mu      sync.Mutex
	running bool
	pending bool
}

// run calls fn, or, when a reload is already running, records one more and
// returns at once.
func (c *reloadCoalescer) run(fn func()) {
	c.mu.Lock()
	if c.running {
		c.pending = true
		c.mu.Unlock()
		return
	}
	c.running = true
	c.mu.Unlock()
	for {
		fn()
		c.mu.Lock()
		if !c.pending {
			c.running = false
			c.mu.Unlock()
			return
		}
		c.pending = false
		c.mu.Unlock()
	}
}
