package memgate

import (
	"context"
	"errors"
	"net/http"
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
