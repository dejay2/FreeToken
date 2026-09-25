package router

// FreeToken patch P6: load a model on request (POST /api/models/load/{model}).

import (
	"context"
	"fmt"

	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

// Loader is implemented by local routers that can load a model without a chat request.
type Loader interface {
	Load(ctx context.Context, modelID string) error
}

// Load asks the scheduler for modelID exactly as a chat request would, so
// latest-wins (P1) and the memory gate (P2) apply, and returns once the model
// is ready or its load failed. The grant it receives is handed straight back
// with a ServeDone, as if a request had been served and finished.
func (b *baseRouter) Load(ctx context.Context, modelID string) error {
	return b.load(ctx, modelID, false)
}

// FreeToken patch P8: FreeLoader is implemented by local routers that can load
// a model only when nothing else is on the card or on its way there.
type FreeLoader interface {
	LoadIfFree(ctx context.Context, modelID string) error
}

// LoadIfFree is Load without preemption: the scheduler refuses it with
// scheduler.NotFreeError (409 busy) at admission, on the run loop, when any
// other model is running, loading, waiting in the memory gate, queued or
// holding requests. A refused load cancels nothing (Load would supersede a
// colliding not-ready swap, P1).
func (b *baseRouter) LoadIfFree(ctx context.Context, modelID string) error {
	return b.load(ctx, modelID, true)
}

func (b *baseRouter) load(ctx context.Context, modelID string, ifFree bool) error {
	if b.shuttingDown.Load() {
		return fmt.Errorf("%s is shutting down", b.name)
	}
	// No "already ready" shortcut: a ready model goes through the scheduler
	// like a chat request (its own fast path grants at once), so a load
	// supersedes a colliding swap exactly as a chat request would (P1).
	if _, procs := b.snapshot(); procs[modelID] == nil {
		return scheduler.ErrModelNotFound
	}
	hr := scheduler.HandlerReq{
		Model:      modelID,
		Ctx:        ctx,
		Admit:      make(chan error, 1),
		Respond:    make(chan scheduler.HandlerResp),
		PositionCh: make(chan int, 1),
		IfFree:     ifFree, // FreeToken patch P8
	}
	shutdownErr := fmt.Errorf("%s is shutting down", b.name)
	select {
	case b.handlerCh <- hr:
	case <-ctx.Done():
		return ctx.Err()
	case <-b.shutdownCtx.Done():
		return shutdownErr
	}
	cancel := func() {
		select {
		case b.cancelCh <- hr:
		case <-b.shutdownCtx.Done():
		}
	}
	select {
	case err := <-hr.Admit:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		cancel()
		return ctx.Err()
	case <-b.shutdownCtx.Done():
		return shutdownErr
	}
	select {
	case resp := <-hr.Respond:
		if resp.Err != nil {
			return resp.Err
		}
		select {
		case b.serveDoneCh <- scheduler.ServeDoneEvent{ModelID: modelID}:
		case <-b.shutdownCtx.Done():
		}
		return nil
	case <-ctx.Done():
		cancel()
		return ctx.Err()
	case <-b.shutdownCtx.Done():
		return shutdownErr
	}
}
