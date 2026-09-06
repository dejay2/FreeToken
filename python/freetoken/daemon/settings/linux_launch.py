"""Linux / WSL launch and stop for the settings helper.

The helper's profiles and boot file are PowerShell-shaped documents (``-ModelPath 'x'`` lines
and ``$env:FREETOKEN_*`` assignments) that ``boot_parser.BootFile`` turns into a settings dict.
On Windows the helper runs them through ``scripts/start-freetoken-windows.ps1``; on Linux this
module maps the same settings onto an ``ft serve`` command with the same rules (what the model
is, from its ``config.json`` alone), and stops a server by walking ``/proc`` the way the Windows
stop script walks the process table.

Torch-free by construction (stdlib only): the helper must never import torch, and
``tests/settings/test_settings_import_safety.py`` enforces it.

WSL specifics, measured on the RTX 5090 box on 2026-09-05 (``docs/research`` and the batch
log carry the numbers):

* The engine's default WSL pin budget is 40 % of RAM (``engine._pin_budget_bytes``); on an
  88 GB VM that is 34 GB and moves 23 of Qwen3.8's 48 expert layers to the CPU (35 tok/s).
  ``cudaHostAlloc`` pinned 60 GiB at 52 GB/s and ``cudaHostRegister`` reached 64 GiB before
  failing, so with born-pinned banks (``FREETOKEN_BANK_CUDA_ALLOC=1``) and a budget of 85 % of
  the VM's RAM every bank pins and decode runs at 81.5 tok/s, 10 % above native Windows.
* ``/usr/lib/wsl/lib`` (nvidia-smi) is not on PATH under systemd, and flashinfer's JIT needs
  the compute capability; both are supplied here.
* The dxg bridge fails with EMFILE under the default 1,024 file handles and ``mlock`` needs a
  raised memlock limit for CPU-resident layers; :func:`raise_limits` lifts the soft limits to
  the hard ones (a systemd unit sets the hard ones).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dials import DIALS, DIAL_BY_NAME, ENV_DIALS, canonical_value

# Dials that only mean something on Windows (the Desktop venv, the local picture packages).
WINDOWS_ONLY_DIALS = frozenset({"DesktopPython", "VisionPackagesPath"})
MTP_ENV = ("FREETOKEN_MTP_SPECULATE", "FREETOKEN_MTP_RESIDENT", "FREETOKEN_MTP_SHADOW", "FREETOKEN_MTP_SPEC_GRAPH")
MTP_ENTRY_ENV = ("FREETOKEN_MTP_SPECULATE", "FREETOKEN_MTP_SHADOW")
WSL_PIN_BUDGET_FRACTION = 0.85


def is_wsl() -> bool:
    try:
        return "microsoft" in os.uname().release.lower()
    except AttributeError:  # Windows has no os.uname
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get(settings: dict[str, Any], name: str) -> Any:
    if name in settings:
        return settings[name]
    dial = DIAL_BY_NAME.get(name)
    return dial.default if dial is not None else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- model facts


@dataclass
class ModelFacts:
    architecture: str = ""
    model_type: str = ""
    has_ple: bool = False
    has_vision: bool = False
    has_mtp: bool = False
    is_moe: bool = False
    expert_count: int = 0
    max_context: int = 0
    parking_supported: bool = False


def read_model_facts(model_path: str | os.PathLike[str]) -> ModelFacts:
    """What the model is, from ``config.json`` alone (mirrors the Windows launcher)."""
    root = Path(model_path)
    config_path = root / "config.json"
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in the model directory: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config") or config

    def field_(name: str) -> Any:
        value = text.get(name) if isinstance(text, dict) else None
        if value is None:
            value = config.get(name)
        return value

    facts = ModelFacts()
    arch = config.get("architectures") or []
    facts.architecture = str(arch[0]) if arch else ""
    facts.model_type = str(field_("model_type") or "")
    ple_ids = field_("ple_layer_ids")
    facts.has_ple = bool(ple_ids)
    facts.has_vision = config.get("vision_config") is not None
    mtp_layers = field_("mtp_num_hidden_layers")
    if mtp_layers is None:
        mtp_layers = field_("num_nextn_predict_layers")
    facts.has_mtp = _int(mtp_layers) > 0
    facts.max_context = _int(field_("max_position_embeddings"))
    num_layers = field_("num_hidden_layers")
    if num_layers is None:
        num_layers = field_("n_layer")
    first_dense = _int(field_("first_k_dense_replace"))
    moe_layers = max(0, _int(num_layers) - first_dense) if num_layers is not None else 0
    experts = None
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        experts = field_(key)
        if experts is not None:
            break
    facts.is_moe = _int(experts) > 0
    facts.expert_count = moe_layers * _int(experts) if facts.is_moe and moe_layers > 0 else 0
    facts.parking_supported = facts.model_type in {"qwen4_exp_text", "qwen4_exp"}
    return facts


# ----------------------------------------------------------------------------- the plan


@dataclass
class LaunchPlan:
    argv: list[str]
    env: dict[str, str]
    notes: list[str] = field(default_factory=list)
    model_path: str = ""
    port: int = 2020

    def command_line(self) -> str:
        return " ".join(self.argv)


def _wsl_memory_gib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return 0.0


def _compute_capability() -> str | None:
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0].strip() if out else None


def wsl_environment(base: dict[str, str] | None = None, *, wsl: bool | None = None) -> dict[str, str]:
    """Environment additions for WSL (no-op elsewhere). Existing values are never overridden."""
    env = dict(os.environ if base is None else base)
    if not (is_wsl() if wsl is None else wsl):
        return env
    wsl_lib = "/usr/lib/wsl/lib"
    if os.path.isdir(wsl_lib) and wsl_lib not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = env.get("PATH", "") + os.pathsep + wsl_lib
    env.setdefault("FREETOKEN_BANK_CUDA_ALLOC", "1")
    if "FREETOKEN_PIN_BUDGET_GB" not in env:
        total = _wsl_memory_gib()
        if total > 0:
            env["FREETOKEN_PIN_BUDGET_GB"] = str(int(total * WSL_PIN_BUDGET_FRACTION))
    if "TVM_FFI_CUDA_ARCH_LIST" not in env:
        cap = _compute_capability()
        if cap:
            env["TVM_FFI_CUDA_ARCH_LIST"] = cap
    return env


def build_launch(
    settings: dict[str, Any],
    *,
    python: str | None = None,
    base_env: dict[str, str] | None = None,
    facts: ModelFacts | None = None,
    wsl: bool | None = None,
) -> LaunchPlan:
    """Map a settings snapshot onto an ``ft serve`` command, with the Windows launcher's rules."""
    model_path = str(_get(settings, "ModelPath") or "").strip()
    model_path = os.path.expanduser(os.path.expandvars(model_path))
    if not model_path:
        raise ValueError("ModelPath is empty")
    if facts is None:
        facts = read_model_facts(model_path)
    notes: list[str] = []
    env = dict(os.environ if base_env is None else base_env)
    for name in ENV_DIALS:
        dial = DIAL_BY_NAME[name]
        value = _get(settings, name)
        if dial.control == "toggle":
            env[name] = "1" if _truthy(value) else "0"
            continue
        # An env amount (the guess depth) goes through as its number, not a 1/0.
        try:
            env[name] = str(canonical_value(dial, value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} {value!r} is not a valid value: {exc}") from exc

    port = _int(_get(settings, "Port"), 2020)
    context = _int(_get(settings, "ContextTokens"))
    if context <= 0:
        context = facts.max_context or 32768
        notes.append(f"Context tokens not given; using the model's own limit {context}")
    elif facts.max_context and context > facts.max_context:
        raise ValueError(
            f"ContextTokens {context} is longer than this model can read ({facts.max_context}, config.json max_position_embeddings)"
        )

    # Only the two entry switches turn the trick on; RESIDENT and SPEC_GRAPH just shape it
    # (SPEC_GRAPH defaults to 1, so counting it would put every boot on the mmap table).
    mtp_on = any(env.get(name) == "1" for name in MTP_ENTRY_ENV)
    if not facts.has_mtp:
        for name in MTP_ENV:
            if env.get(name) == "1" and name in MTP_ENTRY_ENV:
                notes.append(f"{name} forced off: this model ships no MTP head")
            env[name] = "0"
        mtp_on = False

    ple_setting = str(_get(settings, "PleBackend") or "auto").strip().lower()
    if ple_setting == "auto":
        # Linux-native io_uring row store unless MTP is on (its capture path never enters the
        # disk backend), then the demand-paged mmap table the Windows launcher uses.
        ple = ("mmap" if mtp_on else "disk") if facts.has_ple else "off"
    elif ple_setting == "off":
        ple = "off"
    elif not facts.has_ple:
        notes.append(f"PleBackend {ple_setting} ignored: this model has no PLE table")
        ple = "off"
    else:
        ple = ple_setting
    if ple == "disk" and mtp_on:
        notes.append("PLE backend switched to mmap: the disk backend does not support MTP")
        ple = "mmap"

    kv_park = str(_get(settings, "KVPark") or "off")
    if kv_park != "off" and not facts.parking_supported:
        notes.append(f"KV parking '{kv_park}' forced off: the engine parks chats only for the Qwen3.8-Flash-Next (qwen4_exp) memory layout")
        kv_park = "off"

    vision = _truthy(_get(settings, "EnableVision"))
    if vision and not facts.has_vision:
        notes.append("Picture input switched off: this model has no picture tower (no vision_config)")
        vision = False
    if vision:
        execution = str(_get(settings, "VisionExecution") or "layer-stream")
        weights = str(_get(settings, "VisionWeights") or "ram")
        if weights == "mmap" and execution != "layer-stream":
            raise ValueError(f"VisionWeights mmap needs VisionExecution layer-stream (got '{execution}')")
        env["FREETOKEN_LOAD_VISION"] = "1"
        env["FREETOKEN_VISION_EXECUTION"] = execution
        env["FREETOKEN_VISION_WEIGHTS"] = weights
    else:
        env["FREETOKEN_LOAD_VISION"] = "0"
        env["FREETOKEN_VISION_EXECUTION"] = "gpu"
        env["FREETOKEN_VISION_WEIGHTS"] = "ram"
    dense_quant = str(_get(settings, "DenseQuant") or "").strip()
    env["FREETOKEN_DENSE_QUANT"] = dense_quant or "none"
    env["FREETOKEN_EMBED_HOST"] = "1" if _truthy(_get(settings, "EmbedHost")) else "0"

    moe_cache = _int(_get(settings, "MoECacheSize"))
    owned = str(_get(settings, "GpuOwnedLayers") or "").strip()
    if not owned and env.get("FREETOKEN_MOE_GPU_OWNED_LAYERS"):
        owned = env["FREETOKEN_MOE_GPU_OWNED_LAYERS"]
    if not facts.is_moe and (moe_cache > 0 or owned):
        notes.append("Expert-slot settings ignored: this model has no routed experts")
        moe_cache, owned = 0, ""
    if facts.is_moe and facts.expert_count > 0 and moe_cache > facts.expert_count:
        notes.append(f"MoE cache slots clamped from {moe_cache} to {facts.expert_count}: this model has only {facts.expert_count} expert pieces")
        moe_cache = facts.expert_count

    env = wsl_environment(env, wsl=wsl)

    argv = [python or sys.executable, "-m", "freetoken.cli", "serve", "--model", model_path, "--host", "127.0.0.1", "--port", str(port)]
    if ple != "off":
        argv += ["--ple-backend", ple]
    if facts.is_moe:
        argv += ["--moe-backend", "offload"]
        argv += ["--moe-cache-size", str(moe_cache)] if moe_cache > 0 else ["--moe-cache-auto"]
    argv += ["--max-running-requests", str(_int(_get(settings, "MaxRunningRequests"), 1)), "--kv-reserve-tokens", str(context)]
    if facts.is_moe:
        argv += ["--expert-load", str(_get(settings, "ExpertLoad") or "auto")]
    if str(_get(settings, "KVDtype") or "bf16") == "fp8":
        argv += ["--kv-dtype", "fp8"]
    argv += ["--kv-park", kv_park]
    if kv_park != "off":
        argv += [
            "--kv-park-idle-ms", str(_int(_get(settings, "KVParkIdleMs"))),
            "--kv-park-min-tokens", str(_int(_get(settings, "KVParkMinTokens"), 8192)),
            "--kv-park-ram-gib", str(_get(settings, "KVParkRAMGiB")),
            "--kv-park-ssd-dir", str(_get(settings, "KVParkSSDDir")),
            "--kv-park-ssd-gib", str(_get(settings, "KVParkSSDGiB")),
            "--kv-park-window-mib", str(_int(_get(settings, "KVParkWindowMiB"), 256)),
        ]
    if _truthy(_get(settings, "EnableCacheReport")):
        argv.append("--enable-cache-report")
    if _truthy(_get(settings, "CollectRoutingStats")) and facts.is_moe:
        argv.append("--moe-collect-decode-freq")
    if facts.is_moe and owned:
        argv += ["--moe-gpu-owned-layers", owned]
    reserve = _int(_get(settings, "MoEVramReserveBytes"), -1)
    headroom = _int(_get(settings, "MoECacheHeadroomBytes"), -1)
    if facts.is_moe and reserve >= 0:
        argv += ["--moe-vram-reserve-bytes", str(reserve)]
    if facts.is_moe and headroom >= 0:
        argv += ["--moe-cache-headroom-bytes", str(headroom)]
    graph_bs = _int(_get(settings, "CudaGraphMaxBS"), -1)
    if graph_bs >= 0:
        argv += ["--cuda-graph-max-bs", str(graph_bs)]
    kv_tokens = _int(_get(settings, "KVCacheTokens"))
    if kv_tokens > 0:
        argv += ["--num-tokens", str(kv_tokens)]
    return LaunchPlan(argv=argv, env=env, notes=notes, model_path=model_path, port=port)


# ----------------------------------------------------------------------------- limits


def raise_limits() -> list[str]:
    """Lift the soft nofile/memlock limits to the hard ones (best effort)."""
    notes: list[str] = []
    try:
        import resource
    except ImportError:  # Windows
        return notes
    for name in ("RLIMIT_NOFILE", "RLIMIT_MEMLOCK"):
        res = getattr(resource, name, None)
        if res is None:
            continue
        try:
            soft, hard = resource.getrlimit(res)
            if hard != soft:
                resource.setrlimit(res, (hard, hard))
                notes.append(f"{name}: {soft} -> {hard}")
        except (OSError, ValueError) as exc:
            notes.append(f"{name}: could not raise ({exc})")
    return notes


# ----------------------------------------------------------------------------- stop


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _proc_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _all_pids() -> list[int]:
    try:
        return sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    except OSError:
        return []


def find_server_pids(port: int = 0) -> set[int]:
    """The ``ft serve`` processes for ``port`` (0 = every port), their descendants, and any
    orphaned spawn children that still hold the ZMQ side ports."""
    pids = _all_pids()
    lines = {pid: _proc_cmdline(pid) for pid in pids}
    ppids = {pid: _proc_ppid(pid) for pid in pids}
    roots: set[int] = set()
    for pid, line in lines.items():
        if not (("freetoken.cli serve" in line or "/ft serve" in line or line.startswith("ft serve")) and "python" in line):
            continue
        if port and f"--port {port}" not in line and f"--port={port}" not in line:
            continue
        roots.add(pid)
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppids.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    for pid, line in lines.items():
        if "multiprocessing.spawn" in line and "freetoken" in line and ppids.get(pid) in (1, None) and pid not in selected:
            selected.add(pid)
    selected.discard(os.getpid())
    return selected


def _listeners(ports: set[int]) -> set[int]:
    found: set[int] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="utf-8") as fh:
                next(fh)
                for line in fh:
                    parts = line.split()
                    if len(parts) < 4 or parts[3] != "0A":  # LISTEN
                        continue
                    port = int(parts[1].rsplit(":", 1)[1], 16)
                    if port in ports:
                        found.add(port)
        except (OSError, ValueError, StopIteration):
            continue
    return found


def _vram_used_mb() -> int | None:
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def stop_servers(
    port: int = 0,
    *,
    timeout: float = 120.0,
    vram_free_threshold_mb: int = 3072,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> dict[str, Any]:
    """Terminate the server on ``port`` and wait until the card and the ports are free."""
    pids = find_server_pids(port)
    report: dict[str, Any] = {"killed": sorted(pids), "port": port}
    term = signal.SIGTERM
    kill = getattr(signal, "SIGKILL", term)  # Windows has no SIGKILL; this path only runs on Linux
    for sig in (term, kill):
        for pid in list(pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pids.discard(pid)
            except PermissionError:
                pass
        deadline = monotonic() + (10.0 if sig == term else 5.0)
        while pids and monotonic() < deadline:
            sleep(0.5)
            pids = {pid for pid in pids if os.path.exists(f"/proc/{pid}")}
        if not pids:
            break
    report["remaining"] = sorted(pids)
    watched = set(range(port, port + 10)) if port else set()
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        busy_ports = _listeners(watched) if watched else set()
        used = _vram_used_mb()
        card_busy = used is not None and used >= vram_free_threshold_mb
        if not busy_ports and not card_busy:
            report["vram_used_mb"] = used
            report["ok"] = True
            return report
        sleep(1.0)
    report["vram_used_mb"] = _vram_used_mb()
    report["busy_ports"] = sorted(_listeners(watched)) if watched else []
    report["ok"] = False
    return report


# ----------------------------------------------------------------------------- CLI


def parse_launcher_args(argv: list[str]) -> dict[str, Any]:
    """``-ModelPath x -Port 2020 -EnableVision`` (the PowerShell launcher's spelling) -> settings."""
    settings: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-") or token.startswith("--"):
            raise ValueError(f"unexpected argument {token!r}: expected -Name value")
        name = token[1:]
        dial = DIAL_BY_NAME.get(name)
        if dial is not None and dial.control == "toggle" and dial.source != "env":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                settings[name] = _truthy(argv[i + 1]); i += 2
            else:
                settings[name] = True; i += 1
            continue
        if i + 1 >= len(argv):
            raise ValueError(f"-{name} needs a value")
        settings[name] = argv[i + 1]
        i += 2
    return settings


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    dry = "-DryRun" in args
    args = [a for a in args if a != "-DryRun"]
    settings = parse_launcher_args(args)
    plan = build_launch(settings)
    for note in plan.notes:
        print(f"  Note: {note}", file=sys.stderr)
    print(f"Starting FreeToken (Linux): {plan.command_line()}", file=sys.stderr)
    if dry:
        return 0
    for note in raise_limits():
        print(f"  limits: {note}", file=sys.stderr)
    os.execve(plan.argv[0], plan.argv, plan.env)
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
