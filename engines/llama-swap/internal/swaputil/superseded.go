package swaputil

import (
	"fmt"
	"net/http"
)

// FreeToken patch P1: SupersededError answers callers whose model load was
// cancelled because a request for another model arrived ("latest wins").
type SupersededError struct {
	Model string // the model whose load was cancelled
	By    string // the model requested instead
}

func (e SupersededError) Error() string {
	return fmt.Sprintf("load of %s cancelled: model %s was requested instead", e.Model, e.By)
}

func (e SupersededError) StatusCode() int { return http.StatusConflict }

func (e SupersededError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return h
}

func (e SupersededError) Body() []byte {
	return NewErrorEnvelope(e.StatusCode(), e.Error(), "model_superseded").JSON()
}
