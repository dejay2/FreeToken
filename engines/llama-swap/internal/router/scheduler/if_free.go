package scheduler

// FreeToken patch P8: load only if the card is free (POST /api/models/load/{model}?ifFree=1).

import (
	"fmt"
	"net/http"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// IfFreeHeader on a chat request asks for the same non-preempting admission
// as ?ifFree=1 on a load (FreeToken patch P8, round 2): 409 code "busy" instead
// of superseding when another model is running, loading, waiting in the memory
// gate, queued or holding requests. The value must be "1".
const IfFreeHeader = "X-FreeToken-If-Free"

// NotFreeError refuses an if-free request because another model is running,
// loading, waiting in the memory gate, queued or answering (409, code "busy").
// Nothing was admitted and nothing was cancelled.
type NotFreeError struct {
	Model string
	Other string
}

func (e NotFreeError) Error() string {
	return fmt.Sprintf("model %s was not loaded because %s is loading or in use", e.Model, e.Other)
}

func (e NotFreeError) StatusCode() int { return http.StatusConflict }

func (e NotFreeError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return h
}

func (e NotFreeError) Body() []byte {
	return swaputil.NewErrorEnvelope(e.StatusCode(), e.Error(), "busy").JSON()
}
