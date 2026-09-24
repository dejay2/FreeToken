package server

// FreeToken patches P5/P6: Rebuild keeps the local router; config hash; load endpoint.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/memgate"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/store/sqlite"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

type reconfigRouter struct {
	*stubRouter
	prepareErr error
	commits    atomic.Int32
	aborts     atomic.Int32
	loadErr    error
	loaded     []string
}

func (r *reconfigRouter) PrepareReconfigure(config.Config) (*router.ReconfigPlan, error) {
	if r.prepareErr != nil {
		return nil, r.prepareErr
	}
	return &router.ReconfigPlan{
		Kept:     []string{"a"},
		CommitFn: func() { r.commits.Add(1) },
		AbortFn:  func() { r.aborts.Add(1) },
	}, nil
}

func (r *reconfigRouter) Load(_ context.Context, model string) error {
	r.loaded = append(r.loaded, model)
	return r.loadErr
}

func groupCfg() config.Config {
	cfg := config.Config{HealthCheckTimeout: 15}
	cfg.Routing.Router.Use = "group"
	return cfg
}

func TestRebuild_KeepsTheLocalRouterWhenItCanReconfigure(t *testing.T) {
	local := &reconfigRouter{stubRouter: newStubRouter([]string{"a"}, "")}
	prev := newTestServerWithConfig(groupCfg(), local, newStubRouter(nil, ""))
	st, err := sqlite.New("")
	if err != nil {
		t.Fatalf("sqlite.New: %v", err)
	}
	defer st.Close()
	next, kept, err := Rebuild(prev, groupCfg(), st, BuildInfo{}, nil, nil)
	if err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	if !kept || next.local != router.LocalRouter(local) {
		t.Fatalf("kept=%v local=%T: the new server must take over the reconfigured router", kept, next.local)
	}
	if local.commits.Load() != 1 || local.aborts.Load() != 0 {
		t.Fatalf("commits=%d aborts=%d", local.commits.Load(), local.aborts.Load())
	}
	if err := prev.ShutdownExceptLocal(time.Second); err != nil {
		t.Fatalf("ShutdownExceptLocal: %v", err)
	}
	if local.shutdownCalls.Load() != 0 {
		t.Fatalf("the kept router was shut down with the old server")
	}
	_ = next.ShutdownExceptLocal(time.Second)
}

func TestRebuild_PrepareErrorKeepsTheOldServer(t *testing.T) {
	local := &reconfigRouter{stubRouter: newStubRouter([]string{"a"}, ""), prepareErr: errors.New("model in two groups")}
	prev := newTestServerWithConfig(groupCfg(), local, newStubRouter(nil, ""))
	st, _ := sqlite.New("")
	defer st.Close()
	next, _, err := Rebuild(prev, groupCfg(), st, BuildInfo{}, nil, nil)
	if err == nil || next != nil {
		t.Fatalf("next=%v err=%v: a refused reload must return an error and no server", next, err)
	}
	if local.commits.Load() != 0 || local.shutdownCalls.Load() != 0 {
		t.Fatalf("a refused reload touched the running router")
	}
}

func TestRebuild_UnsupportedFallsBackToAFullRebuild(t *testing.T) {
	local := &reconfigRouter{stubRouter: newStubRouter(nil, ""), prepareErr: router.ErrReconfigureUnsupported}
	prev := newTestServerWithConfig(groupCfg(), local, newStubRouter(nil, ""))
	st, _ := sqlite.New("")
	defer st.Close()
	next, kept, err := Rebuild(prev, groupCfg(), st, BuildInfo{}, nil, nil)
	if err != nil || kept {
		t.Fatalf("kept=%v err=%v want a full rebuild", kept, err)
	}
	if _, ok := next.local.(*router.Group); !ok {
		t.Fatalf("local=%T want a new *router.Group", next.local)
	}
	_ = next.Shutdown(time.Second)
}

func TestRebuild_RealGroupRouterIsKept(t *testing.T) {
	st, _ := sqlite.New("")
	defer st.Close()
	prev, err := New(groupCfg(), discardLog(), discardLog(), discardLog(), nil, st, BuildInfo{}, nil, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	next, kept, err := Rebuild(prev, groupCfg(), st, BuildInfo{}, nil, nil)
	if err != nil || !kept || next.local != prev.local {
		t.Fatalf("kept=%v err=%v: NewGroup must wire the reload factory", kept, err)
	}
	_ = prev.ShutdownExceptLocal(time.Second)
	_ = next.Shutdown(time.Second)
}

func TestServer_ConfigHash(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.SetConfigHash("abc123")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/config/hash", nil))
	var body map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil || body["sha256"] != "abc123" {
		t.Fatalf("code=%d body=%q", w.Code, w.Body.String())
	}
}

func TestServer_HandleAPILoadModel(t *testing.T) {
	cases := []struct {
		name    string
		path    string
		loadErr error
		code    int
		body    string
	}{
		{"loads", "/api/models/load/m1", nil, http.StatusOK, `"state":"ready"`},
		{"alias resolves", "/api/models/load/alias1", nil, http.StatusOK, `"model":"m1"`},
		{"superseded", "/api/models/load/m1", swaputil.SupersededError{Model: "m1", By: "m2"}, http.StatusConflict, "model_superseded"},
		{"not enough memory", "/api/models/load/m1", memgate.NotEnoughMemoryError{Model: "m1", NeedGB: 60, FreeGB: 10, FloorGB: 6}, http.StatusServiceUnavailable, "not_enough_memory"},
		{"unknown", "/api/models/load/nope", nil, http.StatusNotFound, "model not found"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			local := &reconfigRouter{stubRouter: newStubRouter([]string{"m1"}, ""), loadErr: tc.loadErr}
			s := newTestServer(local, newStubRouter(nil, ""))
			// Loaded from YAML so the alias index (private to config) is built.
			cfg, err := config.LoadConfigFromReader(strings.NewReader("models:\n  m1:\n    cmd: echo\n    proxy: http://127.0.0.1:9\n    aliases: [alias1]\n"))
			if err != nil {
				t.Fatalf("LoadConfigFromReader: %v", err)
			}
			s.cfg = cfg
			w := httptest.NewRecorder()
			s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, tc.path, nil))
			if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.body) {
				t.Fatalf("code=%d body=%q want %d containing %q", w.Code, w.Body.String(), tc.code, tc.body)
			}
		})
	}
}

func TestServer_LoadWithoutALoaderIs404(t *testing.T) {
	s := newTestServer(newStubRouter([]string{"m1"}, ""), newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}} // RealModelName finds real ids without the alias index
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/models/load/m1", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("code=%d want 404", w.Code)
	}
}

func discardLog() *logmon.Monitor { return logmon.NewWriter(io.Discard) }

// FreeToken patch P5: a refused commit (here: the router has shut down)
// returns an error and no server, so the caller keeps prev.
func TestRebuild_RefusedCommitReturnsAnError(t *testing.T) {
	st, _ := sqlite.New("")
	defer st.Close()
	prev, err := New(groupCfg(), discardLog(), discardLog(), discardLog(), nil, st, BuildInfo{}, nil, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := prev.Shutdown(time.Second); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	next, kept, err := Rebuild(prev, groupCfg(), st, BuildInfo{}, nil, nil)
	if err == nil || next != nil || kept {
		t.Fatalf("next=%v kept=%v err=%v: a refused commit must return an error", next, kept, err)
	}
}

// FreeToken patch P5 (fix round 1): the exit shutdown also stops a local
// router the Server had handed on, and stops an owned one exactly once.
func TestServer_ShutdownWithLocal(t *testing.T) {
	handedOn := &reconfigRouter{stubRouter: newStubRouter([]string{"a"}, "")}
	prev := newTestServerWithConfig(groupCfg(), handedOn, newStubRouter(nil, ""))
	if err := prev.ShutdownExceptLocal(time.Second); err != nil {
		t.Fatalf("ShutdownExceptLocal: %v", err)
	}
	if err := prev.ShutdownWithLocal(time.Second); err != nil {
		t.Fatalf("ShutdownWithLocal: %v", err)
	}
	if got := handedOn.shutdownCalls.Load(); got != 1 {
		t.Fatalf("handed-on router shutdownCalls=%d want 1", got)
	}

	owned := newStubRouter([]string{"a"}, "")
	s := newTestServer(owned, newStubRouter(nil, ""))
	if err := s.ShutdownWithLocal(time.Second); err != nil {
		t.Fatalf("ShutdownWithLocal: %v", err)
	}
	if got := owned.shutdownCalls.Load(); got != 1 {
		t.Fatalf("owned router shutdownCalls=%d want 1", got)
	}
}
