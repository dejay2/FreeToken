package config

// FreeToken patch P2: memoryGate config parsing.

import (
	"strings"
	"testing"
)

func TestLoadConfig_MemoryGateHelperURL(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
memoryGate:
  probe: windows
  floorGB: 6
  waitSeconds: 300
  helperURL: "http://127.0.0.1:2031"
models:
  m:
    cmd: echo hi
    proxy: http://127.0.0.1:9999
    ramNeedGB: 58
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MemoryGate.HelperURL != "http://127.0.0.1:2031" || cfg.MemoryGate.Probe != "windows" {
		t.Errorf("unexpected memoryGate %+v", cfg.MemoryGate)
	}
	if cfg.Models["m"].RamNeedGB != 58 {
		t.Errorf("ramNeedGB=%v want 58", cfg.Models["m"].RamNeedGB)
	}
}
