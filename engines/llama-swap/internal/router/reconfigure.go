package router

// FreeToken patch P5: selective reload. A config reload reconfigures the local
// router in place instead of replacing it: processes whose model entry did
// not change are kept, with their requests in flight; changed and removed
// models are stopped through the same OnUnload path an unload takes. See the
// decision note in docs/superpowers/plans/2026-09-24-control-panel-stage-a.md
// (Task 6) for why processes are not adopted into a new router.

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

// ErrReconfigureUnsupported sends the caller down upstream's full rebuild.
var ErrReconfigureUnsupported = errors.New("router cannot be reconfigured in place")

// ErrStaleReconfigure refuses a plan prepared from a table that another
// commit has replaced since: applying it would drop that commit's processes.
var ErrStaleReconfigure = errors.New("reload plan is stale: the router was reconfigured after it was prepared")

// ProcessFactory builds a stopped process for one model entry.
type ProcessFactory func(ctx context.Context, id string, mc config.ModelConfig) (process.Process, error)

// Reconfigurer is implemented by local routers that can take a new config
// without stopping the models whose entry did not change.
type Reconfigurer interface {
	PrepareReconfigure(newCfg config.Config) (*ReconfigPlan, error)
}

// ReconfigPlan is a prepared reload: Commit applies it, Abort throws away the
// processes it built. Only the first of the two calls has any effect.
type ReconfigPlan struct {
	Kept, Stopped, Added []string
	CommitFn, AbortFn    func()
	once                 sync.Once
	err                  error // set by CommitFn when the commit was refused
}

// Commit applies the plan. A non-nil error means it was refused (stale, or
// the router has shut down); the plan's own processes are then thrown away
// and the router is left as it was.
func (p *ReconfigPlan) Commit() error {
	p.once.Do(func() {
		if p.CommitFn != nil {
			p.CommitFn()
		}
	})
	return p.err
}
func (p *ReconfigPlan) Abort() {
	p.once.Do(func() {
		if p.AbortFn != nil {
			p.AbortFn()
		}
	})
}

type reconfigState struct {
	cfg         config.Config
	processes   map[string]process.Process
	cancels     map[string]context.CancelFunc
	planner     scheduler.Swapper
	kept        []string
	stopped     []string
	stopTimeout time.Duration
	generation  uint64 // the table generation this state was prepared from
}

type reconfigReq struct {
	state *reconfigState
	done  chan struct{}
	err   error // set by applyReconfig before done is closed
}

// processFactory builds real upstream processes.
func processFactory(upstreamlog, proxylog *logmon.Monitor) ProcessFactory {
	return func(ctx context.Context, id string, mc config.ModelConfig) (process.Process, error) {
		p, err := process.New(ctx, id, mc, logmon.NewWriter(upstreamlog), proxylog)
		if err != nil {
			return nil, err
		}
		return p, nil
	}
}

// newProcess gives every process its own child context of procCtx, so a
// reload can retire one process without touching the others.
func (b *baseRouter) newProcess(id string, mc config.ModelConfig) (process.Process, context.CancelFunc, error) {
	ctx, cancel := context.WithCancel(b.procCtx)
	p, err := b.factory(ctx, id, mc)
	if err != nil {
		cancel()
		return nil, nil, err
	}
	return p, cancel, nil
}

// snapshot returns the current config and process table. Reconfigure replaces
// both (it never mutates them), so callers may use them without the lock.
func (b *baseRouter) snapshot() (config.Config, map[string]process.Process) {
	b.stateMu.RLock()
	defer b.stateMu.RUnlock()
	return b.config, b.processes
}

// newMemGate is FreeToken patch P2's gate construction, moved here so a reload
// can rebuild it. Nil when memoryGate.probe is not "windows".
func newMemGate(conf config.Config, logger *logmon.Monitor) *memgate.Gate {
	mg := conf.MemoryGate
	if mg.Probe != "windows" {
		return nil
	}
	// FreeToken patch P2: unset floorGB is 6, an explicit 0 is no cushion.
	floor, wait := mg.EffectiveFloorGB(), mg.WaitSeconds
	if wait <= 0 {
		wait = 300
	}
	g := &memgate.Gate{
		Probe:   memgate.WindowsProbe(memgate.RunWindowsFreeKB, 2*time.Second),
		FloorGB: floor,
		Wait:    time.Duration(wait) * time.Second,
		Poll:    5 * time.Second,
		Logf:    logger.Infof,
		Warnf:   logger.Warnf,
	}
	if mg.HelperURL != "" {
		g.Bypass = memgate.HelperBypass(mg.HelperURL, nil)
	}
	return g
}

// PrepareReconfigure builds processes for new and changed entries without
// touching anything running. An error leaves the router exactly as it was.
func (b *baseRouter) PrepareReconfigure(newCfg config.Config) (*ReconfigPlan, error) {
	if b.plannerFor == nil || b.factory == nil {
		return nil, ErrReconfigureUnsupported
	}
	planner, members, err := b.plannerFor(newCfg)
	if err != nil {
		return nil, err
	}
	b.stateMu.RLock()
	oldCfg, oldProcs, gen := b.config, b.processes, b.generation
	b.stateMu.RUnlock()
	st := &reconfigState{
		generation: gen,
		cfg:        newCfg,
		processes:  make(map[string]process.Process, len(members)),
		cancels:    make(map[string]context.CancelFunc),
		planner:    planner,
	}
	abort := func() {
		for _, cancel := range st.cancels {
			cancel()
		}
	}
	var added []string
	for _, id := range members {
		mc, ok := newCfg.Models[id]
		if !ok {
			abort()
			return nil, fmt.Errorf("no model config for %q", id)
		}
		if p, had := oldProcs[id]; had && reflect.DeepEqual(oldCfg.Models[id], mc) {
			st.processes[id] = p
			st.kept = append(st.kept, id)
			continue
		}
		p, cancel, err := b.newProcess(id, mc)
		if err != nil {
			abort()
			return nil, fmt.Errorf("creating process for %q: %w", id, err)
		}
		st.processes[id] = p
		st.cancels[id] = cancel
		added = append(added, id)
	}
	for id := range oldProcs {
		if slices.Contains(st.kept, id) {
			continue
		}
		st.stopped = append(st.stopped, id)
		t := time.Duration(oldCfg.Models[id].UnloadTimeout) * time.Second
		if t <= 0 {
			t = time.Duration(oldCfg.UnloadTimeout) * time.Second
		}
		if t > st.stopTimeout {
			st.stopTimeout = t
		}
	}
	sort.Strings(st.kept)
	sort.Strings(st.stopped)
	sort.Strings(added)
	plan := &ReconfigPlan{
		Kept:    st.kept,
		Stopped: st.stopped,
		Added:   added,
		AbortFn: abort,
	}
	plan.CommitFn = func() {
		req := &reconfigReq{state: st, done: make(chan struct{})}
		select {
		case b.reconfigCh <- req:
			<-req.done
			if req.err != nil {
				abort()
				plan.err = req.err
			}
		case <-b.runDone:
			abort()
			plan.err = fmt.Errorf("%s has shut down", b.name)
		}
	}
	return plan, nil
}

// applyReconfig runs in the run loop, so it is ordered with every scheduler
// event: requests that arrive meanwhile are handled against the new table.
func (b *baseRouter) applyReconfig(req *reconfigReq) {
	st := req.state
	// The run loop is the only writer of generation, so it reads it unlocked.
	if st.generation != b.generation {
		req.err = ErrStaleReconfigure
		b.logger.Warnf("%s: reload refused: %v", b.name, req.err)
		close(req.done)
		return
	}
	if len(st.stopped) > 0 {
		// Waiters and queued requests for these models get an error, their
		// swaps are cancelled (P1), and the processes are stopped.
		b.schedule.OnUnload(st.stopped, st.stopTimeout)
	}
	b.stateMu.Lock()
	oldCancels := b.procCancels
	b.config = st.cfg
	b.processes = st.processes
	b.memGate = newMemGate(st.cfg, b.logger)
	cancels := make(map[string]context.CancelFunc, len(st.processes))
	for _, id := range st.kept {
		if cancel, ok := oldCancels[id]; ok {
			cancels[id] = cancel
		}
	}
	for id, cancel := range st.cancels {
		cancels[id] = cancel
	}
	b.procCancels = cancels
	b.generation++
	b.stateMu.Unlock()
	for _, id := range st.stopped {
		if cancel, ok := oldCancels[id]; ok {
			cancel()
		}
	}
	b.schedule.OnReconfigure(st.cfg, st.planner)
	b.logger.Infof("%s: reload kept %v, stopped %v", b.name, st.kept, st.stopped)
	close(req.done)
}
