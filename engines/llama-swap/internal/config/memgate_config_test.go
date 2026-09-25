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

// FreeToken patch P2: floorGB unset -> 6, explicit 0 -> 0 (no cushion), negative refused.
func TestLoadConfig_MemoryGateFloorGB(t *testing.T) {
	load := func(floor string) (Config, error) {
		return LoadConfigFromReader(strings.NewReader("memoryGate:\n  probe: windows\n" + floor + "models:\n  m:\n    cmd: echo hi\n    proxy: http://127.0.0.1:9999\n"))
	}
	cfg, err := load("")
	if err != nil || cfg.MemoryGate.FloorGB != nil || cfg.MemoryGate.EffectiveFloorGB() != 6 {
		t.Fatalf("unset: err=%v floor=%v effective=%v", err, cfg.MemoryGate.FloorGB, cfg.MemoryGate.EffectiveFloorGB())
	}
	cfg, err = load("  floorGB: 0\n")
	if err != nil || cfg.MemoryGate.FloorGB == nil || cfg.MemoryGate.EffectiveFloorGB() != 0 {
		t.Fatalf("explicit 0: err=%v effective=%v", err, cfg.MemoryGate.EffectiveFloorGB())
	}
	cfg, err = load("  floorGB: 2.5\n")
	if err != nil || cfg.MemoryGate.EffectiveFloorGB() != 2.5 {
		t.Fatalf("2.5: err=%v effective=%v", err, cfg.MemoryGate.EffectiveFloorGB())
	}
	for _, bad := range []string{"  floorGB: -1\n", "  floorGB: .nan\n"} {
		if _, err := load(bad); err == nil || !strings.Contains(err.Error(), "floorGB") {
			t.Errorf("%q: err=%v, want a floorGB error", bad, err)
		}
	}
}
