package memgate

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func seq(values ...float64) Probe {
	i := 0
	return func(context.Context) (float64, error) {
		v := values[min(i, len(values)-1)]
		i++
		return v, nil
	}
}

func fastGate(p Probe) *Gate {
	return &Gate{Probe: p, FloorGB: 6, Wait: 50 * time.Millisecond, Poll: time.Millisecond}
}

func TestWaitForRoom_EnoughNow(t *testing.T) {
	if err := fastGate(seq(40)).WaitForRoom(context.Background(), "m", 18); err != nil {
		t.Fatal(err)
	}
}

func TestWaitForRoom_WaitsThenLoads(t *testing.T) {
	if err := fastGate(seq(10, 12, 30)).WaitForRoom(context.Background(), "m", 18); err != nil {
		t.Fatalf("want room after waiting, got %v", err)
	}
}

func TestWaitForRoom_TimesOutWith503(t *testing.T) {
	err := fastGate(seq(10)).WaitForRoom(context.Background(), "big", 60)
	var nem NotEnoughMemoryError
	if !errors.As(err, &nem) {
		t.Fatalf("want NotEnoughMemoryError, got %v", err)
	}
	if nem.StatusCode() != http.StatusServiceUnavailable || nem.Model != "big" || nem.NeedGB != 60 || nem.FreeGB != 10 {
		t.Errorf("unexpected error %+v", nem)
	}
}

func TestWaitForRoom_ProbeErrorLoadsAnyway(t *testing.T) {
	g := fastGate(func(context.Context) (float64, error) { return 0, errors.New("powershell missing") })
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatalf("a broken probe must not block loads, got %v", err)
	}
}

func TestWaitForRoom_CancelEndsWaitAtOnce(t *testing.T) {
	g := &Gate{Probe: seq(1), FloorGB: 6, Wait: time.Hour, Poll: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- g.WaitForRoom(ctx, "m", 18) }()
	time.Sleep(10 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("wait did not end after cancel")
	}
}

func TestWaitForRoom_DisabledWhenNoNeedOrNilGate(t *testing.T) {
	var g *Gate
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatal(err)
	}
	if err := fastGate(seq(0)).WaitForRoom(context.Background(), "m", 0); err != nil {
		t.Fatal(err)
	}
}

func TestWindowsProbe_ParsesKBAndCaches(t *testing.T) {
	calls := 0
	p := WindowsProbe(func(context.Context) ([]byte, error) {
		calls++
		return []byte("  8388608\r\n"), nil // 8 GiB in KB
	}, time.Minute)
	for i := 0; i < 3; i++ {
		v, err := p(context.Background())
		if err != nil || v != 8 {
			t.Fatalf("v=%v err=%v want 8", v, err)
		}
	}
	if calls != 1 {
		t.Errorf("calls=%d want 1 (cached)", calls)
	}
}

func TestWindowsProbe_GarbageIsAnError(t *testing.T) {
	p := WindowsProbe(func(context.Context) ([]byte, error) { return []byte("Access denied"), nil }, time.Minute)
	if _, err := p(context.Background()); err == nil {
		t.Fatal("want error for unparsable output")
	}
}

// FreeToken patch P2: a probe error caused by the swap's own cancel is the
// cancel, not a reason to load anyway.
func TestWaitForRoom_ProbeErrorAfterCancelReturnsCtxErr(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var warned bool
	g := fastGate(func(context.Context) (float64, error) {
		cancel()
		return 0, errors.New("signal: killed")
	})
	g.Warnf = func(string, ...any) { warned = true }
	if err := g.WaitForRoom(ctx, "m", 60); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
	if warned {
		t.Error("a cancelled probe must not log 'loading anyway'")
	}
}

func TestWaitForRoom_ProbeErrorLogsWarn(t *testing.T) {
	var warn, info int
	g := fastGate(func(context.Context) (float64, error) { return 0, errors.New("powershell missing") })
	g.Logf = func(string, ...any) { info++ }
	g.Warnf = func(string, ...any) { warn++ }
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatal(err)
	}
	if warn != 1 || info != 0 {
		t.Errorf("warn=%d info=%d, want the probe failure at Warn only", warn, info)
	}
}

func TestWaitForRoom_BypassSkipsWithoutProbing(t *testing.T) {
	probed := false
	var msg string
	g := fastGate(func(context.Context) (float64, error) { probed = true; return 0, nil })
	g.Bypass = func(context.Context) (bool, string) { return true, "state ready" }
	g.Warnf = func(f string, a ...any) { msg = fmt.Sprintf(f, a...) }
	if err := g.WaitForRoom(context.Background(), "m", 60); err != nil {
		t.Fatalf("bypass must let the load through, got %v", err)
	}
	if probed {
		t.Error("bypass true must not probe")
	}
	if !strings.Contains(msg, "FreeToken (not started by llama-swap) is up; skipping the wait, the adapter will stop it") {
		t.Errorf("unexpected warn %q", msg)
	}
}

func TestWaitForRoom_BypassFalseKeepsNormalPath(t *testing.T) {
	g := fastGate(seq(10))
	g.Bypass = func(context.Context) (bool, string) { return false, "" }
	var nem NotEnoughMemoryError
	if err := g.WaitForRoom(context.Background(), "big", 60); !errors.As(err, &nem) {
		t.Fatalf("want NotEnoughMemoryError, got %v", err)
	}
}

func TestHelperBypass(t *testing.T) {
	cases := []struct {
		name string
		body string
		code int
		want bool
	}{
		{"server up", `{"server":{"state":"serving"},"currentJob":null}`, 200, true},
		{"job running", `{"server":{"state":"unreachable"},"currentJob":{"kind":"start"}}`, 200, true},
		{"nothing up", `{"server":{"state":"unreachable"},"currentJob":null}`, 200, false},
		{"helper error", `oops`, 500, false},
		{"bad json", `not json`, 200, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/status" {
					http.NotFound(w, r)
					return
				}
				w.WriteHeader(c.code)
				_, _ = w.Write([]byte(c.body))
			}))
			defer srv.Close()
			if got, _ := HelperBypass(srv.URL+"/", nil)(context.Background()); got != c.want {
				t.Errorf("got %v want %v", got, c.want)
			}
		})
	}
}

func TestHelperBypass_UnreachableHelperIsNoBypass(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	if got, _ := HelperBypass(url, nil)(context.Background()); got {
		t.Error("a helper that does not answer must not bypass")
	}
}

func TestHelperBypass_SlowHelperTimesOut(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)
	start := time.Now()
	if got, _ := HelperBypass(srv.URL, nil)(context.Background()); got {
		t.Error("a hung helper must not bypass")
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("bypass took %v, want the 2 s cap", d)
	}
}

// FreeToken patch P2: the probe command must carry a WaitDelay so a
// descendant holding stdout after the timeout kill cannot block Output().
func TestWindowsFreeCmd_HasWaitDelay(t *testing.T) {
	cmd := windowsFreeCmd(context.Background(), "/fake/powershell.exe")
	if cmd.WaitDelay != 2*time.Second {
		t.Errorf("WaitDelay=%v want 2s", cmd.WaitDelay)
	}
	if cmd.Path != "/fake/powershell.exe" || len(cmd.Args) != 4 || cmd.Args[1] != "-NoProfile" {
		t.Errorf("unexpected command %q %q", cmd.Path, cmd.Args)
	}
}
