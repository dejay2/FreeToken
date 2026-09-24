package memgate

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// powershellPath finds powershell.exe from WSL. systemd --user units may not
// carry /mnt/c on PATH, so fall back to the fixed System32 location.
func powershellPath() string {
	if p, err := exec.LookPath("powershell.exe"); err == nil {
		return p
	}
	return "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
}

// RunWindowsFreeKB asks Windows for FreePhysicalMemory (KB), with a 10 s cap.
func RunWindowsFreeKB(ctx context.Context) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if _, err := os.Stat(powershellPath()); err != nil {
		return nil, fmt.Errorf("powershell.exe not reachable: %w", err)
	}
	return exec.CommandContext(ctx, powershellPath(), "-NoProfile", "-Command",
		"(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory").Output()
}

// WindowsProbe turns run's KB output into GB and caches it for ttl.
func WindowsProbe(run func(context.Context) ([]byte, error), ttl time.Duration) Probe {
	var mu sync.Mutex
	var at time.Time
	var cached float64
	return func(ctx context.Context) (float64, error) {
		mu.Lock()
		defer mu.Unlock()
		if !at.IsZero() && time.Since(at) < ttl {
			return cached, nil
		}
		out, err := run(ctx)
		if err != nil {
			return 0, err
		}
		kb, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
		if err != nil {
			return 0, fmt.Errorf("unexpected FreePhysicalMemory output %q", strings.TrimSpace(string(out)))
		}
		cached, at = kb/1024/1024, time.Now()
		return cached, nil
	}
}
