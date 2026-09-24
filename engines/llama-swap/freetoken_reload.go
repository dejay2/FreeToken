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
//
// stop is called by the SIGINT/SIGTERM path before it reads the active
// Server: a reload that commits after shutdown picked its Server would hand
// the kept local router to a Server nobody shuts down, and the model
// processes (Setpgid, no Pdeathsig) would outlive llama-swap. Reviewer
// reproduced that with a SIGTERM 2.5 s into a reload (fix round 1).
type reloadCoalescer struct {
	mu      sync.Mutex
	running bool
	pending bool
	closed  bool
	done    chan struct{} // closed when the running reload loop ends
}

// run calls fn, or, when a reload is already running, records one more and
// returns at once. After stop it does nothing.
func (c *reloadCoalescer) run(fn func()) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	if c.running {
		c.pending = true
		c.mu.Unlock()
		return
	}
	c.running = true
	done := make(chan struct{})
	c.done = done
	c.mu.Unlock()
	defer close(done)
	for {
		fn()
		c.mu.Lock()
		if c.closed || !c.pending {
			c.running, c.pending = false, false
			c.mu.Unlock()
			return
		}
		c.pending = false
		c.mu.Unlock()
	}
}

// stop refuses every later reload, drops a pending one and waits for the
// running one (if any) to finish.
func (c *reloadCoalescer) stop() {
	c.mu.Lock()
	c.closed = true
	c.pending = false
	running, done := c.running, c.done
	c.mu.Unlock()
	if running {
		<-done
	}
}
