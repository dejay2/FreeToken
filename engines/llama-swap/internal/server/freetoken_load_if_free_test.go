package server

// FreeToken patch P8: POST /api/models/load/{model}?ifFree=1.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
)

type freeRouter struct {
	*reconfigRouter
	notFree   bool
	freeCalls []string
}

func (r *freeRouter) LoadIfFree(_ context.Context, model string) error {
	r.freeCalls = append(r.freeCalls, model)
	if r.notFree {
		return scheduler.NotFreeError{Model: model, Other: "m2"}
	}
	return nil
}

func TestServer_LoadIfFree(t *testing.T) {
	cases := []struct {
		name    string
		notFree bool
		code    int
		body    string
	}{
		{"free card loads", false, http.StatusOK, `"state":"ready"`},
		{"busy card is refused", true, http.StatusConflict, `"code":"busy"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			local := &freeRouter{reconfigRouter: &reconfigRouter{stubRouter: newStubRouter([]string{"m1"}, "")}, notFree: tc.notFree}
			s := newTestServer(local, newStubRouter(nil, ""))
			s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}
			w := httptest.NewRecorder()
			s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/load/m1?ifFree=1", nil))
			if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.body) {
				t.Fatalf("code=%d body=%q want %d containing %q", w.Code, w.Body.String(), tc.code, tc.body)
			}
			if len(local.freeCalls) != 1 || len(local.loaded) != 0 {
				t.Fatalf("if-free calls %v, plain loads %v", local.freeCalls, local.loaded)
			}
		})
	}
}

func TestServer_LoadIfFreeWithoutSupportIsRefused(t *testing.T) {
	local := &reconfigRouter{stubRouter: newStubRouter([]string{"m1"}, "")}
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/load/m1?ifFree=1", nil))
	if w.Code != http.StatusNotImplemented || len(local.loaded) != 0 {
		t.Fatalf("code=%d loads=%v: a router that can't check must not load", w.Code, local.loaded)
	}
}
