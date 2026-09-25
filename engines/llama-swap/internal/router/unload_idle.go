package router

// FreeToken patch P7: unload only if idle (POST /api/models/unload/{model}?ifIdle=1).
//
// Upstream Unload stops a process even while it is answering: in-flight
// requests are killed (see Unload). The control panel's Test tab puts models
// away between its steps and must never kill another app's answer, so it asks
// for the unload only if the model is idle.
//
// What is atomic: the idle check and the stop run in one step of the router's
// run loop, which is also the only place a request is admitted (OnRequest),
// counted in flight (grantHandler) or joins a swap. So no request the
// scheduler can see is admitted between "idle" and the stop.
//
// What is not: a request that has entered ServeHTTP but not yet reached the
// run loop (it is parsing its body or blocked sending on handlerCh while the
// stop runs) is not seen. It is not killed either: it reaches the scheduler
// after the stop and loads the model again like any request for a stopped
// model. Ignored websocket connections (routing.ignoreWebsockets) bypass the
// scheduler and are not seen at all, as for every upstream eviction.

import (
	"fmt"
	"net/http"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// IdleUnloader is implemented by local routers that can refuse an unload
// while the model is in use.
type IdleUnloader interface {
	UnloadIfIdle(modelID string) error
}

// BusyError answers an if-idle unload of a model that is still in use
// (409, code "busy").
type BusyError struct{ Model string }

func (e BusyError) Error() string {
	return fmt.Sprintf("model %s is answering a request, so it was not unloaded", e.Model)
}

func (e BusyError) StatusCode() int { return http.StatusConflict }

func (e BusyError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return h
}

func (e BusyError) Body() []byte {
	return swaputil.NewErrorEnvelope(e.StatusCode(), e.Error(), "busy").JSON()
}

// busyReporter is implemented by schedulers that can tell whether they hold a
// request for a model (FIFO does).
type busyReporter interface {
	Busy(modelID string) bool
}

// anyBusy runs on the run loop. A scheduler that cannot tell counts as busy,
// so an if-idle unload never kills anything.
func (b *baseRouter) anyBusy(targets []string) bool {
	br, ok := b.schedule.(busyReporter)
	if !ok {
		return true
	}
	for _, id := range targets {
		if br.Busy(id) {
			return true
		}
	}
	return false
}

// UnloadIfIdle stops modelID with its configured unloadTimeout, unless the
// scheduler holds a request for it, in which case nothing is stopped and a
// BusyError is returned. Blocks until the process has stopped.
func (b *baseRouter) UnloadIfIdle(modelID string) error {
	busy := false
	req := unloadReq{targets: []string{modelID}, timeout: b.unloadTimeout(modelID), ifIdle: true,
		busy: &busy, respond: make(chan struct{})}
	select {
	case b.unloadCh <- req:
	case <-b.runDone:
		return fmt.Errorf("%s is shutting down", b.name)
	}
	<-req.respond
	if busy {
		return BusyError{Model: modelID}
	}
	return nil
}
