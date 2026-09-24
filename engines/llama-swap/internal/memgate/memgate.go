// Package memgate is FreeToken patch P2: before a model loads, wait until the
// PC has room for it, so a load never squeezes Windows into swapping (measured
// 2026-09-24: a FreeToken boot took Windows free RAM to 0-2 GB).
package memgate

import (
	"context"
	"fmt"
	"net/http"
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
}

// WaitForRoom returns nil when the model may load, NotEnoughMemoryError after
// Wait, or ctx.Err() when the load was cancelled. A failing probe never blocks.
func (g *Gate) WaitForRoom(ctx context.Context, model string, needGB float64) error {
	if g == nil || g.Probe == nil || needGB <= 0 {
		return nil
	}
	deadline := time.Now().Add(g.Wait)
	for {
		free, err := g.Probe(ctx)
		if err != nil {
			g.logf("memory gate: probe failed, loading %s anyway: %v", model, err)
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
