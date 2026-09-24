package router

// FreeToken patch P6: load a model on request (POST /api/models/load/{model}).

import (
	"context"
	"fmt"

	"github.com/mostlygeek/llama-swap/internal/process"
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
	if b.shuttingDown.Load() {
		return fmt.Errorf("%s is shutting down", b.name)
	}
	_, procs := b.snapshot()
	p, ok := procs[modelID]
	if !ok {
		return scheduler.ErrModelNotFound
	}
	if p.State() == process.StateReady {
		return nil
	}
	hr := scheduler.HandlerReq{
		Model:      modelID,
		Ctx:        ctx,
		Admit:      make(chan error, 1),
		Respond:    make(chan scheduler.HandlerResp),
		PositionCh: make(chan int, 1),
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
