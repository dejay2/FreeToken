// Package memgate is FreeToken patch P2: before a model loads, wait until the
// PC has room for it, so a load never squeezes Windows into swapping (measured
// 2026-09-24: a FreeToken boot took Windows free RAM to 0-2 GB).
package memgate

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// Probe reports free host memory in GB.
type Probe func(ctx context.Context) (float64, error)

// Gate holds a load until free - need >= FloorGB, for at most Wait.
type Gate struct {
	Probe   Probe
	FloorGB float64
	Wait    time.Duration
	Poll    time.Duration
	Logf    func(format string, args ...any)
	// Warnf logs problems (probe failure, bypass); falls back to Logf.
	Warnf func(format string, args ...any)
	// Bypass, when set, is asked once per gated load before any probe. true
	// skips the wait: a FreeToken that llama-swap did not start is holding
	// the RAM, and the adapter's make-room step will stop it after the gate,
	// so waiting would only end in a 503 (final review 2026-09-24). The
	// string is detail for the log line.
	Bypass func(ctx context.Context) (bool, string)
}

// WaitForRoom returns nil when the model may load, NotEnoughMemoryError after
// Wait, or ctx.Err() when the load was cancelled. A failing probe never blocks.
func (g *Gate) WaitForRoom(ctx context.Context, model string, needGB float64) error {
	if g == nil || g.Probe == nil || needGB <= 0 {
		return nil
	}
	if g.Bypass != nil {
		if skip, detail := g.Bypass(ctx); skip {
			g.warnf("memory gate: FreeToken (not started by llama-swap) is up; skipping the wait, the adapter will stop it (%s, loading %s)", detail, model)
			return nil
		}
	}
	deadline := time.Now().Add(g.Wait)
	for {
		free, err := g.Probe(ctx)
		if err != nil {
			// A cancelled swap is not a probe fault: report the cancel.
			if cerr := ctx.Err(); cerr != nil {
				return cerr
			}
			g.warnf("memory gate: probe failed, loading %s anyway: %v", model, err)
			return nil
		}
		if free-needGB >= g.FloorGB {
			return nil
		}
		if !time.Now().Before(deadline) {
			return NotEnoughMemoryError{Model: model, NeedGB: needGB, FreeGB: free, FloorGB: g.FloorGB}
		}
		g.logf("memory gate: %s needs %.0f GB, %.1f GB free (floor %.0f GB); waiting", model, needGB, free, g.FloorGB)
		t := time.NewTimer(g.Poll)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}

func (g *Gate) logf(format string, args ...any) {
	if g.Logf != nil {
		g.Logf(format, args...)
	}
}

func (g *Gate) warnf(format string, args ...any) {
	if g.Warnf != nil {
		g.Warnf(format, args...)
		return
	}
	g.logf(format, args...)
}

// HelperBypass builds a Gate.Bypass that asks the FreeToken settings helper
// (GET <helperURL>/api/status, 2 s cap) whether its server is up or a job is
// running. Any failure answers false, so the gate behaves as before.
func HelperBypass(helperURL string, client *http.Client) func(context.Context) (bool, string) {
	if client == nil {
		client = &http.Client{}
	}
	url := strings.TrimRight(helperURL, "/") + "/api/status"
	return func(ctx context.Context) (bool, string) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return false, ""
		}
		resp, err := client.Do(req)
		if err != nil {
			return false, ""
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return false, ""
		}
		var st struct {
			Server struct {
				State string `json:"state"`
			} `json:"server"`
			CurrentJob json.RawMessage `json:"currentJob"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
			return false, ""
		}
		job := strings.TrimSpace(string(st.CurrentJob))
		hasJob := job != "" && job != "null"
		if st.Server.State == "" && !hasJob {
			return false, "" // not a helper status answer
		}
		if st.Server.State != "unreachable" || hasJob {
			return true, fmt.Sprintf("helper server state %q, job running %v", st.Server.State, hasJob)
		}
		return false, ""
	}
}

// NotEnoughMemoryError is the 503 answer when the wait runs out.
type NotEnoughMemoryError struct {
	Model   string
	NeedGB  float64
	FreeGB  float64
	FloorGB float64
}

func (e NotEnoughMemoryError) Error() string {
	return fmt.Sprintf("not enough free memory to load %s (need %.0f GB plus a %.0f GB cushion, Windows has %.1f GB free); close something and try again",
		e.Model, e.NeedGB, e.FloorGB, e.FreeGB)
}

func (e NotEnoughMemoryError) StatusCode() int { return http.StatusServiceUnavailable }

func (e NotEnoughMemoryError) Header() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	h.Set("Retry-After", "30")
	return h
}

func (e NotEnoughMemoryError) Body() []byte {
	return swaputil.NewErrorEnvelope(e.StatusCode(), e.Error(), "not_enough_memory").JSON()
}
