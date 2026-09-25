package server

// FreeToken patch P7: POST /api/models/unload/{model}?ifIdle=1.

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/router"
)

type idleRouter struct {
	*stubRouter
	busy      bool
	idleCalls []string
}

func (r *idleRouter) UnloadIfIdle(model string) error {
	r.idleCalls = append(r.idleCalls, model)
	if r.busy {
		return router.BusyError{Model: model}
	}
	return nil
}

func TestServer_UnloadIfIdle(t *testing.T) {
	cases := []struct {
		name string
		busy bool
		code int
		body string
	}{
		{"idle model is unloaded", false, http.StatusOK, "OK"},
		{"busy model is refused", true, http.StatusConflict, `"code":"busy"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			local := &idleRouter{stubRouter: newStubRouter([]string{"m1"}, ""), busy: tc.busy}
			s := newTestServer(local, newStubRouter(nil, ""))
			s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}
			w := httptest.NewRecorder()
			s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/unload/m1?ifIdle=1", nil))
			if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.body) {
				t.Fatalf("code=%d body=%q want %d containing %q", w.Code, w.Body.String(), tc.code, tc.body)
			}
			if len(local.idleCalls) != 1 || local.unloadCalls.Load() != 0 {
				t.Fatalf("idle calls %v, plain unloads %d", local.idleCalls, local.unloadCalls.Load())
			}
		})
	}
}

func TestServer_UnloadWithoutIfIdleIsUnchanged(t *testing.T) {
	local := &idleRouter{stubRouter: newStubRouter([]string{"m1"}, ""), busy: true}
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/unload/m1", nil))
	if w.Code != http.StatusOK || len(local.idleCalls) != 0 || local.unloadCalls.Load() != 1 {
		t.Fatalf("code=%d idle=%v unloads=%d", w.Code, local.idleCalls, local.unloadCalls.Load())
	}
}

func TestServer_UnloadIfIdleWithoutSupportIsRefused(t *testing.T) {
	local := newStubRouter([]string{"m1"}, "")
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/unload/m1?ifIdle=1", nil))
	if w.Code != http.StatusNotImplemented || local.unloadCalls.Load() != 0 {
		t.Fatalf("code=%d unloads=%d: a router that can't check must not unload", w.Code, local.unloadCalls.Load())
	}
}
