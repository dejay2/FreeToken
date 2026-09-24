package server

// FreeToken patches P5 (selective reload, config hash) and P6 (load endpoint).

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/docagent"
	"github.com/mostlygeek/llama-swap/internal/hw"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/store"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// FreeToken patch P5: Rebuild builds the Server for a reloaded config. When
// prev's local router can be reconfigured in place (same router kind), the
// new Server keeps it: models whose entry did not change keep running, and
// changed or removed ones are stopped. Otherwise it builds everything anew,
// as upstream does. keptLocal tells the caller to retire prev with
// ShutdownExceptLocal. An error leaves prev untouched and serving.
func Rebuild(prev *Server, cfg config.Config, st store.Store, build BuildInfo, hardware *hw.HardwareSnapshot, refs *docagent.Docs) (*Server, bool, error) {
	if rc, ok := prev.local.(router.Reconfigurer); ok && prev.cfg.Routing.Router.Use == cfg.Routing.Router.Use {
		plan, err := rc.PrepareReconfigure(cfg)
		switch {
		case errors.Is(err, router.ErrReconfigureUnsupported):
			// fall through to the full rebuild below
		case err != nil:
			return nil, false, err
		default:
			next, err := newWithLocal(cfg, prev.local, prev.muxlog, prev.proxylog, prev.upstreamlog, prev.perf, st, build, hardware, refs)
			if err != nil {
				plan.Abort()
				return nil, false, err
			}
			if err := plan.Commit(); err != nil {
				// Refused (a stale plan, or the router shut down): the plan's
				// processes are gone and the router is as it was. Retire next
				// without touching the router it shares with prev.
				_ = next.ShutdownExceptLocal(time.Second)
				return nil, false, err
			}
			prev.proxylog.Infof("reload: kept %v, stopped %v, rebuilt %v", plan.Kept, plan.Stopped, plan.Added)
			return next, true, nil
		}
	}
	next, err := New(cfg, prev.muxlog, prev.proxylog, prev.upstreamlog, prev.perf, st, build, hardware, refs)
	return next, false, err
}

// FreeToken patch P5: ShutdownExceptLocal retires a Server whose local router
// now belongs to its replacement.
func (s *Server) ShutdownExceptLocal(timeout time.Duration) error {
	s.keepLocal.Store(true)
	return s.Shutdown(timeout)
}

// FreeToken patch P5: SetConfigHash records the sha256 of the config file
// this Server was built from; GET /api/config/hash reports it so the control
// panel knows when its new file is live.
func (s *Server) SetConfigHash(hash string) { s.configHash.Store(&hash) }

func (s *Server) handleAPIConfigHash(w http.ResponseWriter, r *http.Request) {
	hash := ""
	if p := s.configHash.Load(); p != nil {
		hash = *p
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"sha256": hash})
}

// FreeToken patch P6: handleAPILoadModel loads a model without a chat request.
// It goes through the router's scheduler exactly like a chat request, so
// latest-wins (P1) and the memory gate (P2) apply, and answers once the model
// is ready (200) or its load failed (409 model_superseded, 503
// not_enough_memory, or the start error).
func (s *Server) handleAPILoadModel(w http.ResponseWriter, r *http.Request) {
	requested := strings.TrimPrefix(r.PathValue("model"), "/")
	realName, found := s.cfg.RealModelName(requested)
	if !found {
		swaputil.SendResponse(w, r, http.StatusNotFound, "model not found")
		return
	}
	loader, ok := s.local.(router.Loader)
	if !ok || !s.local.Handles(realName) {
		swaputil.SendResponse(w, r, http.StatusNotFound, "no local server found for requested model")
		return
	}
	if err := loader.Load(r.Context(), realName); err != nil {
		swaputil.SendError(w, r, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"model": realName, "state": "ready"})
}
