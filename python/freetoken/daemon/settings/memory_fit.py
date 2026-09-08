"""Torch-free coordinator for the settings-page memory fit check.

The settings helper owns validation, launch normalization, resource probes, and the subprocess
boundary.  Engine imports deliberately live in :mod:`freetoken.engine.memory_plan`; this module
must remain safe to import in the daemon process on a machine without CUDA libraries.
"""

from __future__ import annotations

import csv
import datetime as _datetime
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .boot_parser import BootFile, BootParseError
from .dials import DIALS, DIAL_BY_NAME, adapt_dial, canonical_value, validate_settings
from .model_info import ModelInfo, read_model


PROTOCOL_VERSION = 1
RESOURCE_STALE_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 45.0


class EstimateUnavailable(RuntimeError):
    """An estimate could not produce a trustworthy result."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


class SettingsValidationError(ValueError):
    """The requested effective settings failed the catalogue/model validator."""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("settings validation failed")
        self.errors = errors


@dataclass(frozen=True)
class MachineSnapshot:
    """The small, safe machine document passed to the child and returned to the page."""

    ram_free_bytes: int
    ram_total_bytes: int
    vram_free_bytes: int
    vram_total_bytes: int
    ram_source: str = "/proc/meminfo:MemAvailable"
    vram_source: str = "nvidia-smi"
    gpu_uuid: str | None = None
    gpu_name: str | None = None
    cgroup_limited: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ram_free_bytes": self.ram_free_bytes,
            "ram_total_bytes": self.ram_total_bytes,
            "vram_free_bytes": self.vram_free_bytes,
            "vram_total_bytes": self.vram_total_bytes,
            "ram_source": self.ram_source,
            "vram_source": self.vram_source,
            "gpu_uuid": self.gpu_uuid,
            "gpu_name": self.gpu_name,
            "cgroup_limited": self.cgroup_limited,
        }


def _finite_cgroup_value(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _unescape_mountinfo_path(value: str) -> str:
    return value.replace(r"\040", " ").replace(r"\011", "\t").replace(r"\134", "\\")


def _cgroup_v2_paths() -> tuple[Path, ...]:
    """Return the current cgroup-v2 directory followed by its constraining ancestors."""
    relative = ""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, separator, path = line.partition("::")
            if separator and hierarchy == "0":
                relative = path.strip()
                break
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        pass

    mounts: list[tuple[Path, Path]] = []
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            before, separator, after = line.partition(" - ")
            if not separator or not after.split() or after.split()[0] != "cgroup2":
                continue
            fields = before.split()
            if len(fields) >= 5:
                mounts.append((
                    Path(_unescape_mountinfo_path(fields[3])),
                    Path(_unescape_mountinfo_path(fields[4])),
                ))
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        pass
    if not mounts:
        mounts.append((Path("/"), Path("/sys/fs/cgroup")))

    paths: list[Path] = []
    seen: set[Path] = set()
    for root, mount in mounts:
        # mountinfo field 4 is the filesystem subtree exposed at field 5. A service under
        # /user.slice must not have that prefix appended again to a /user.slice mount.
        try:
            suffix = Path(relative or "/").relative_to(root)
        except ValueError:
            continue
        leaf = mount / suffix
        while True:
            if leaf not in seen:
                paths.append(leaf)
                seen.add(leaf)
            if leaf == mount or leaf.parent == leaf:
                break
            leaf = leaf.parent
    return tuple(paths)


def _read_ram_snapshot() -> tuple[int, int, str, bool]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                name, separator, rest = line.partition(":")
                if not separator:
                    continue
                fields = rest.split()
                if fields and fields[0].isdigit():
                    values[name] = int(fields[0]) * 1024
    except (OSError, UnicodeDecodeError) as exc:
        raise EstimateUnavailable("probe_failed", f"could not read physical RAM: {exc}") from exc

    total = values.get("MemTotal")
    free = values.get("MemAvailable")
    if not total or free is None or free < 0:
        raise EstimateUnavailable(
            "probe_failed", "MemTotal and a valid MemAvailable are required for the RAM probe"
        )

    source = "/proc/meminfo:MemAvailable"
    cgroup_limited = False
    for cgroup_path in _cgroup_v2_paths():
        cgroup_max = _finite_cgroup_value(cgroup_path / "memory.max")
        if cgroup_max is None:
            continue
        cgroup_current = _finite_cgroup_value(cgroup_path / "memory.current")
        total = min(total, cgroup_max)
        if cgroup_current is None:
            free = min(free, cgroup_max)
        else:
            free = min(free, max(0, cgroup_max - cgroup_current))
        cgroup_limited = True
    if cgroup_limited:
        source += ";cgroup-v2:effective-limit"
    return max(0, free), max(0, total), source, cgroup_limited


def _nvidia_smi_path(environ: Mapping[str, str]) -> str | None:
    path = shutil.which("nvidia-smi", path=environ.get("PATH"))
    if path:
        return path
    fallback = "/usr/lib/wsl/lib/nvidia-smi"
    return fallback if os.path.isfile(fallback) else None


def _read_vram_snapshot(environ: Mapping[str, str]) -> tuple[int, int, str | None, str | None, str]:
    executable = _nvidia_smi_path(environ)
    if executable is None:
        raise EstimateUnavailable("probe_failed", "nvidia-smi is not available")
    command = [
        executable,
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
            env=dict(environ),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EstimateUnavailable("probe_failed", f"nvidia-smi failed: {type(exc).__name__}") from exc
    if result.returncode != 0:
        raise EstimateUnavailable("probe_failed", "nvidia-smi returned a failure status")

    rows: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "total": int(row[3].strip()) * 1024 * 1024,
                    "free": int(row[4].strip()) * 1024 * 1024,
                    "used": int(row[5].strip()) * 1024 * 1024,
                }
            )
        except ValueError:
            continue
    if not rows:
        raise EstimateUnavailable("probe_failed", "nvidia-smi returned no usable GPU row")

    visible = (environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if visible:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if len(entries) != 1:
            raise EstimateUnavailable(
                "probe_failed", "the serving environment exposes more than one CUDA device"
            )
        token = entries[0]
        matches = [
            row
            for row in rows
            if token == row["uuid"] or (token.isdigit() and int(token) == row["index"])
        ]
        if len(matches) != 1:
            raise EstimateUnavailable(
                "probe_failed", "CUDA_VISIBLE_DEVICES does not identify one reported GPU"
            )
        selected = matches[0]
    elif len(rows) == 1:
        selected = rows[0]
    else:
        raise EstimateUnavailable(
            "probe_failed", "multiple GPUs are visible without an unambiguous serving selection"
        )
    return selected["free"], selected["total"], selected["uuid"], selected["name"], "nvidia-smi"


def probe_machine(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read one physical RAM and selected-GPU snapshot without importing torch."""
    env = dict(os.environ if environ is None else environ)
    ram_free, ram_total, ram_source, cgroup_limited = _read_ram_snapshot()
    vram_free, vram_total, gpu_uuid, gpu_name, vram_source = _read_vram_snapshot(env)
    return MachineSnapshot(
        ram_free_bytes=ram_free,
        ram_total_bytes=ram_total,
        vram_free_bytes=vram_free,
        vram_total_bytes=vram_total,
        ram_source=ram_source,
        vram_source=vram_source,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        cgroup_limited=cgroup_limited,
    ).as_dict()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _normalize_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept injected test snapshots while keeping the child protocol field names stable."""
    if not isinstance(value, Mapping):
        raise EstimateUnavailable("probe_failed", "machine probe did not return an object")
    ram = value.get("ram") if isinstance(value.get("ram"), Mapping) else {}
    vram = value.get("vram") if isinstance(value.get("vram"), Mapping) else {}
    result = {
        "ram_free_bytes": _as_int(value.get("ram_free_bytes", ram.get("free_bytes"))),
        "ram_total_bytes": _as_int(value.get("ram_total_bytes", ram.get("total_bytes"))),
        "vram_free_bytes": _as_int(value.get("vram_free_bytes", vram.get("free_bytes"))),
        "vram_total_bytes": _as_int(value.get("vram_total_bytes", vram.get("total_bytes"))),
        "ram_source": str(value.get("ram_source") or "/proc/meminfo:MemAvailable"),
        "vram_source": str(value.get("vram_source") or "nvidia-smi"),
        "gpu_uuid": value.get("gpu_uuid"),
        "gpu_name": value.get("gpu_name"),
        "cgroup_limited": bool(value.get("cgroup_limited", False)),
    }
    if result["ram_total_bytes"] <= 0 or result["vram_total_bytes"] <= 0:
        raise EstimateUnavailable("probe_failed", "machine probe returned an invalid capacity")
    result["ram_free_bytes"] = max(0, min(result["ram_free_bytes"], result["ram_total_bytes"]))
    result["vram_free_bytes"] = max(0, min(result["vram_free_bytes"], result["vram_total_bytes"]))
    return result


def _resource_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return any(
        abs(_as_int(after.get(f"{resource}_free_bytes")) - _as_int(before.get(f"{resource}_free_bytes")))
        > RESOURCE_STALE_BYTES
        for resource in ("ram", "vram")
    )


def _diagnostic(text: Any, *, redact: tuple[str, ...] = ()) -> str:
    line = next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")
    for secret in redact:
        if secret:
            line = line.replace(secret, "<redacted>")
    if "FREETOKEN_MTP_PRIVATE_ROOT" in line:
        line = line.split("FREETOKEN_MTP_PRIVATE_ROOT", 1)[0].rstrip(" ,;:")
    return line[:512] or "planner child failed without a diagnostic"


def _unavailable(code: str, message: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "status": "unavailable",
        "fits": None,
        "fits_now": None,
        "fits_empty": None,
        "suggestion": None,
        "error": {"code": code, "message": _diagnostic(message)},
    }


def prepare_settings(
    settings: Mapping[str, Any],
    *,
    boot_file: BootFile,
    require_model_metadata: bool = True,
) -> dict[str, Any]:
    """Merge, validate, and canonicalize an estimate or accepted lifecycle snapshot."""
    if not isinstance(settings, Mapping):
        raise SettingsValidationError([{"field": "settings", "message": "must be an object"}])
    try:
        saved = boot_file.load()
    except BootParseError:
        raise
    requested = dict(settings)
    effective = dict(saved)
    effective.update(requested)
    model_path = effective.get("ModelPath", "")
    model = read_model(model_path if isinstance(model_path, str) else "")
    # Match PUT /api/settings: a partial patch is checked against the requested model, while
    # inherited values are allowed to come from the active boot. When the request changes the
    # model, the complete merged snapshot is checked as well so an old context/slot value cannot
    # be carried into a model with a smaller ceiling.
    errors = validate_settings(requested, model, context=saved)
    if not errors and isinstance(requested.get("ModelPath"), str) and requested["ModelPath"] != saved.get("ModelPath", ""):
        errors = validate_settings(effective, model)
    if errors:
        raise SettingsValidationError(errors)
    if require_model_metadata and not model.found:
        raise EstimateUnavailable("unsupported_geometry", model.error or "the model folder is unavailable")
    if require_model_metadata:
        folder = Path(os.path.expandvars(os.path.expanduser(str(model_path).strip())))
        if not folder.is_dir() or not (folder / "config.json").is_file():
            raise EstimateUnavailable("unsupported_geometry", "the requested model directory is not local")

    canonical: dict[str, Any] = {}
    for dial in DIALS:
        value = effective.get(dial.name, dial.default)
        stored_as = adapt_dial(dial, model).get("storedAs")
        try:
            canonical[dial.name] = canonical_value(dial, value, stored_as)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SettingsValidationError(
                [{"field": dial.name, "message": f"Value for {dial.name} is invalid: {exc}"}]
            ) from exc
    return canonical


class MemoryFitService:
    """Coordinate one serialized estimate, with seams for fake probes and child runners."""

    def __init__(
        self,
        *,
        snapshot: Callable[..., Mapping[str, Any]] | None = None,
        runner: Callable[..., Any] | None = None,
        launch_builder: Callable[..., Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        root: str | os.PathLike[str] | None = None,
        release_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._release_probe = release_probe
        self._runner = runner or subprocess.run
        self._launch_builder = launch_builder
        self.timeout = float(timeout)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
        self._busy = threading.Lock()

    def _take_snapshot(self, environ: Mapping[str, str] | None) -> dict[str, Any]:
        if self._snapshot is None:
            return _normalize_snapshot(probe_machine(environ))
        try:
            try:
                value = self._snapshot(environ)
            except TypeError:
                value = self._snapshot()
        except EstimateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - probe failures are protocol failures
            raise EstimateUnavailable("probe_failed", f"machine probe failed: {type(exc).__name__}") from exc
        return _normalize_snapshot(value)

    def _build_launch(self, settings: dict[str, Any], environ: Mapping[str, str]) -> Any:
        builder = self._launch_builder
        if builder is None:
            from .linux_launch import build_launch

            builder = build_launch
        try:
            return builder(settings, base_env=dict(environ))
        except Exception as exc:  # noqa: BLE001 - launch normalization is child-boundary input
            raise EstimateUnavailable("planner_failed", _diagnostic(exc)) from exc

    def _run_child(
        self,
        settings: dict[str, Any],
        machine: dict[str, Any],
        environ: Mapping[str, str],
    ) -> dict[str, Any]:
        plan = self._build_launch(settings, environ)
        argv = list(getattr(plan, "argv", ()) or ())
        if not argv:
            raise EstimateUnavailable("planner_failed", "launch normalization returned no interpreter")
        child_request = {
            "version": PROTOCOL_VERSION,
            "settings": settings,
            "argv": argv[4:],
            "machine": machine,
        }
        child_env = dict(getattr(plan, "env", None) or environ)

        def failure_message(message: str, stderr: Any = "") -> str:
            redact = (
                str(settings.get("ModelPath", "")),
                str(child_env.get("FREETOKEN_MTP_PRIVATE_ROOT", "")),
            )
            summary = _diagnostic(message, redact=redact)
            # The last nonblank line carries the exception, not the startup warning. Redact
            # before truncating, using the effective launch env rather than only the parent.
            last = next((line for line in reversed(str(stderr or "").splitlines()) if line.strip()), "")
            if not last:
                return summary
            return f"{summary[:160]}; stderr: {_diagnostic(last, redact=redact)[:340]}"

        try:
            completed = self._runner(
                [argv[0], str(Path(__file__).resolve().parents[2] / "engine" / "memory_plan.py")],
                input=json.dumps(child_request, separators=(",", ":")),
                env=child_env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                cwd=str(self.root),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EstimateUnavailable("planner_timeout", "the metadata planner exceeded its time limit") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise EstimateUnavailable("planner_failed", f"planner process failed: {type(exc).__name__}") from exc
        stderr = getattr(completed, "stderr", "")
        if getattr(completed, "returncode", 0) not in (0, None):
            raise EstimateUnavailable(
                "planner_failed",
                failure_message(f"planner exited with status {completed.returncode}", stderr),
            )
        stdout = str(getattr(completed, "stdout", "") or "").strip()
        if len(stdout.encode("utf-8", "replace")) > 4 * 1024 * 1024:
            raise EstimateUnavailable("planner_failed", "planner returned an oversized response")
        try:
            decoder = json.JSONDecoder()
            value, end = decoder.raw_decode(stdout)
            if stdout[end:].strip() or not isinstance(value, dict):
                raise ValueError("child stdout contains more than one JSON object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EstimateUnavailable(
                "planner_failed", failure_message("planner returned malformed JSON", stderr)
            ) from exc
        if value.get("version") != PROTOCOL_VERSION or value.get("status") != "ok":
            error = value.get("error") if isinstance(value.get("error"), Mapping) else {}
            raise EstimateUnavailable(
                str(error.get("code") or "planner_failed"),
                failure_message(error.get("message") or "planner did not return an ok result", stderr),
            )
        return value

    @staticmethod
    def _phase(raw: Mapping[str, Any], capacity: int) -> dict[str, int]:
        need = max(0, _as_int(raw.get("need_bytes")))
        resident = max(0, _as_int(raw.get("resident_bytes")))
        peak = max(0, _as_int(raw.get("boot_peak_bytes", need)))
        return {
            "need_bytes": need,
            "resident_bytes": resident,
            "boot_peak_bytes": peak,
            "shortfall_bytes": max(0, need - capacity),
        }

    @staticmethod
    def _validate_planner_contract(planner: Mapping[str, Any]) -> None:
        """Reject incomplete successful child documents before any defaults are applied."""
        if not isinstance(planner, Mapping):
            raise EstimateUnavailable("planner_failed", "planner result is not an object")
        if (
            isinstance(planner.get("version"), bool)
            or not isinstance(planner.get("version"), int)
            or planner.get("version") != PROTOCOL_VERSION
            or planner.get("status") != "ok"
        ):
            raise EstimateUnavailable("planner_failed", "planner result has an invalid status")

        def required_mapping(value: Any, name: str) -> Mapping[str, Any]:
            if not isinstance(value, Mapping):
                raise EstimateUnavailable("planner_failed", f"planner result is missing complete {name}")
            return value

        def integer(value: Any, name: str, *, positive: bool = False) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise EstimateUnavailable("planner_failed", f"planner result has invalid {name}")
            if value < (1 if positive else 0):
                raise EstimateUnavailable("planner_failed", f"planner result has invalid {name}")
            return value

        def boolean(value: Any, name: str) -> None:
            if not isinstance(value, bool):
                raise EstimateUnavailable("planner_failed", f"planner result has invalid {name}")

        fits_now = planner.get("fits_now")
        fits_empty = planner.get("fits_empty")
        boolean(fits_now, "fits_now")
        boolean(fits_empty, "fits_empty")
        boolean(planner.get("fits"), "fits")
        if planner["fits"] != fits_now:
            raise EstimateUnavailable("planner_failed", "planner result has inconsistent fits alias")

        for field in ("sampled_at", "effective_settings", "machine", "resources", "geometry", "pinning", "components", "issues", "assumptions", "suggestion"):
            if field not in planner:
                raise EstimateUnavailable("planner_failed", f"planner result is missing complete {field}")
        if planner["sampled_at"] is not None and not isinstance(planner["sampled_at"], str):
            raise EstimateUnavailable("planner_failed", "planner result has invalid sampled_at")
        if not isinstance(planner["effective_settings"], Mapping):
            raise EstimateUnavailable("planner_failed", "planner result has invalid effective_settings")
        if not isinstance(planner["machine"], Mapping):
            raise EstimateUnavailable("planner_failed", "planner result has invalid machine")

        resources = required_mapping(planner.get("resources"), "resources")
        for resource in ("ram", "vram"):
            row = required_mapping(resources.get(resource), f"resources.{resource}")
            integer(row.get("free_bytes"), f"resources.{resource}.free_bytes")
            integer(row.get("total_bytes"), f"resources.{resource}.total_bytes", positive=True)
            if row["free_bytes"] > row["total_bytes"]:
                raise EstimateUnavailable("planner_failed", f"planner result has invalid resources.{resource}")
            for scenario in ("now", "empty"):
                phase = required_mapping(row.get(scenario), f"resources.{resource}.{scenario}")
                for field in ("need_bytes", "resident_bytes", "boot_peak_bytes", "shortfall_bytes"):
                    integer(phase.get(field), f"resources.{resource}.{scenario}.{field}")

        issues_value = planner.get("issues")
        if not isinstance(issues_value, list):
            raise EstimateUnavailable("planner_failed", "planner result is missing complete issues")
        issue_scopes: set[str] = set()
        for index, issue in enumerate(issues_value):
            issue_map = required_mapping(issue, f"issues[{index}]")
            if not isinstance(issue_map.get("code"), str) or not issue_map["code"].strip():
                raise EstimateUnavailable("planner_failed", f"planner result has invalid issues[{index}].code")
            scope = issue_map.get("scope", "both")
            if scope not in {"now", "empty", "both"}:
                raise EstimateUnavailable("planner_failed", f"planner result has invalid issues[{index}].scope")
            if not isinstance(issue_map.get("message"), str) or not issue_map["message"].strip():
                raise EstimateUnavailable("planner_failed", f"planner result has invalid issues[{index}].message")
            issue_scopes.add(str(scope))

        geometry = required_mapping(planner.get("geometry"), "geometry")
        for scenario in ("now", "empty"):
            value = geometry.get(scenario)
            if value is None:
                if scenario not in issue_scopes and "both" not in issue_scopes:
                    raise EstimateUnavailable(
                        "planner_failed", f"planner result is missing complete geometry.{scenario}"
                    )
                continue
            row = required_mapping(value, f"geometry.{scenario}")
            for field in ("total_slots", "lru_slots", "num_pages", "page_size", "usable_kv_tokens"):
                integer(row.get(field), f"geometry.{scenario}.{field}")
            if row["num_pages"] <= 1 or row["page_size"] <= 0:
                raise EstimateUnavailable("planner_failed", f"planner result has invalid geometry.{scenario}")
            owned = row.get("owned_layers")
            if not isinstance(owned, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in owned
            ):
                raise EstimateUnavailable(
                    "planner_failed", f"planner result has invalid geometry.{scenario}.owned_layers"
                )
            boolean(row.get("prefill_overlap"), f"geometry.{scenario}.prefill_overlap")

        pinning = required_mapping(planner.get("pinning"), "pinning")
        for field in ("need_bytes", "cap_bytes", "shortfall_bytes", "cpu_layers"):
            if field not in pinning:
                raise EstimateUnavailable("planner_failed", f"planner result is missing complete pinning.{field}")
        integer(pinning.get("need_bytes"), "pinning.need_bytes")
        integer(pinning.get("shortfall_bytes"), "pinning.shortfall_bytes")
        cap = pinning.get("cap_bytes")
        if cap is not None:
            integer(cap, "pinning.cap_bytes")
        cpu_layers = pinning.get("cpu_layers")
        if not isinstance(cpu_layers, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in cpu_layers
        ):
            raise EstimateUnavailable("planner_failed", "planner result has invalid pinning.cpu_layers")

        components = planner.get("components")
        if not isinstance(components, list):
            raise EstimateUnavailable("planner_failed", "planner result is missing complete components")
        allowed_kinds = {"allocation", "policy", "reclaimable", "allowance"}
        for index, component in enumerate(components):
            item = required_mapping(component, f"components[{index}]")
            for field in ("name", "resource", "phase", "kind", "source", "scenario"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise EstimateUnavailable(
                        "planner_failed", f"planner result has invalid components[{index}].{field}"
                    )
            if item["resource"] not in {"ram", "vram"} or item["kind"] not in allowed_kinds:
                raise EstimateUnavailable("planner_failed", f"planner result has invalid components[{index}]")
            if item["scenario"] not in {"now", "empty", "both"}:
                raise EstimateUnavailable("planner_failed", f"planner result has invalid components[{index}].scenario")
            integer(item.get("bytes"), f"components[{index}].bytes")

        assumptions = planner.get("assumptions")
        if not isinstance(assumptions, list) or any(not isinstance(item, str) for item in assumptions):
            raise EstimateUnavailable("planner_failed", "planner result is missing complete assumptions")

        effective = planner.get("effective")
        if not isinstance(effective, Mapping):
            raise EstimateUnavailable("planner_failed", "planner result is missing complete effective settings")
        if "page_size" in effective:
            integer(effective["page_size"], "effective.page_size", positive=True)
        if "pin_budget_bytes" in effective and effective["pin_budget_bytes"] is not None:
            integer(effective["pin_budget_bytes"], "effective.pin_budget_bytes")
        if "bank_cuda_alloc" in effective:
            boolean(effective["bank_cuda_alloc"], "effective.bank_cuda_alloc")
        for field in ("attention_backend", "cache_type", "moe_backend", "ple_backend"):
            if field in effective and not isinstance(effective[field], str):
                raise EstimateUnavailable("planner_failed", f"planner result has invalid effective.{field}")

        suggestion = planner.get("suggestion")
        if suggestion is not None:
            suggestion_map = required_mapping(suggestion, "suggestion")
            if suggestion_map.get("target") not in {"now", "empty"}:
                raise EstimateUnavailable("planner_failed", "planner result has invalid suggestion.target")
            if not isinstance(suggestion_map.get("settings"), Mapping):
                raise EstimateUnavailable("planner_failed", "planner result has invalid suggestion.settings")
            boolean(suggestion_map.get("fits"), "suggestion.fits")
            boolean(suggestion_map.get("fits_now"), "suggestion.fits_now")
            boolean(suggestion_map.get("fits_empty"), "suggestion.fits_empty")
            if suggestion_map["fits"] != suggestion_map["fits_now"]:
                raise EstimateUnavailable("planner_failed", "planner result has inconsistent suggestion fits alias")
            changes = suggestion_map.get("changes")
            if not isinstance(changes, list):
                raise EstimateUnavailable("planner_failed", "planner result has invalid suggestion.changes")
            for index, change in enumerate(changes):
                change_map = required_mapping(change, f"suggestion.changes[{index}]")
                for field in ("name", "reason"):
                    if not isinstance(change_map.get(field), str) or not change_map[field].strip():
                        raise EstimateUnavailable(
                            "planner_failed", f"planner result has invalid suggestion.changes[{index}].{field}"
                        )

    @classmethod
    def _assemble(
        cls,
        planner: Mapping[str, Any],
        settings: dict[str, Any],
        before: Mapping[str, Any],
        sampled_at: str,
    ) -> dict[str, Any]:
        cls._validate_planner_contract(planner)
        resources = planner.get("resources")
        if not isinstance(resources, Mapping):
            raise EstimateUnavailable("planner_failed", "planner result has no resources object")
        output = dict(planner)
        output["version"] = PROTOCOL_VERSION
        output["status"] = "ok"
        output["sampled_at"] = sampled_at
        output["effective_settings"] = dict(settings)
        output["machine"] = {
            "ram_source": str(before.get("ram_source") or "/proc/meminfo:MemAvailable"),
            "gpu_uuid": before.get("gpu_uuid"),
            "gpu_name": before.get("gpu_name"),
            "vram_source": str(before.get("vram_source") or "nvidia-smi"),
            "cgroup_limited": bool(before.get("cgroup_limited", False)),
        }
        normalized_resources: dict[str, Any] = {}
        for resource in ("ram", "vram"):
            raw_resource = resources.get(resource)
            if not isinstance(raw_resource, Mapping):
                raise EstimateUnavailable("planner_failed", f"planner result has no {resource} resource")
            free = _as_int(before.get(f"{resource}_free_bytes"))
            total = _as_int(before.get(f"{resource}_total_bytes"))
            normalized_resources[resource] = {
                "free_bytes": free,
                "total_bytes": total,
                "now": cls._phase(
                    raw_resource.get("now") if isinstance(raw_resource.get("now"), Mapping) else {}, free
                ),
                "empty": cls._phase(
                    raw_resource.get("empty") if isinstance(raw_resource.get("empty"), Mapping) else {}, total
                ),
            }
        output["resources"] = normalized_resources
        output.setdefault("effective", {})
        output.setdefault("geometry", {"now": None, "empty": None})
        output.setdefault("pinning", {"need_bytes": 0, "cap_bytes": None, "shortfall_bytes": 0, "cpu_layers": []})
        output.setdefault("components", [])
        output.setdefault("issues", [])
        output.setdefault("assumptions", [])

        issues = output["issues"] if isinstance(output["issues"], list) else []
        pinning = output["pinning"] if isinstance(output["pinning"], Mapping) else {}
        pin_shortfall = max(0, _as_int(pinning.get("shortfall_bytes")))
        geometry = output.get("geometry") if isinstance(output.get("geometry"), Mapping) else {}

        def fits_scenario(name: str) -> bool:
            if any(normalized_resources[res][name]["shortfall_bytes"] > 0 for res in ("ram", "vram")):
                return False
            if pin_shortfall:
                return False
            if geometry.get(name) is None:
                return False
            for issue in issues:
                if not isinstance(issue, Mapping):
                    return False
                scope = str(issue.get("scope") or "both")
                if scope in (name, "both"):
                    return False
            return True

        output["fits_now"] = fits_scenario("now")
        output["fits_empty"] = fits_scenario("empty")
        output["fits"] = output["fits_now"]
        suggestion = output.get("suggestion")
        if suggestion is not None and not isinstance(suggestion, Mapping):
            output["suggestion"] = None
        return output

    def _release_for(self, action: str | None) -> dict[str, int] | None:
        """What a restart frees before its boot: the running server's own RAM and VRAM.

        A restart stops the old server first, so judging it against the machine as it is now
        (the old server still holding ~70 GB of pinned banks and most of the card) says
        "does not fit" for every restart, and the planner then burns its whole time budget
        searching a thousand candidates that fit "right now" (measured 2026-09-08: 32-82 s, past
        the 45 s limit, so the page's Restart never started anything). Only a restart adds the
        release back; Start and a plain estimate keep the honest "now".
        """
        if action != "restart" or self._release_probe is None:
            return None
        try:
            raw = self._release_probe() or {}
        except Exception:  # noqa: BLE001 - an unreadable release is just no release
            return None
        release = {
            "ram_bytes": max(0, _as_int(raw.get("ram_bytes"))),
            "vram_bytes": max(0, _as_int(raw.get("vram_bytes"))),
        }
        return release if release["ram_bytes"] or release["vram_bytes"] else None

    @staticmethod
    def _with_release(snapshot: Mapping[str, Any], release: Mapping[str, int] | None) -> dict[str, Any]:
        if not release:
            return dict(snapshot)
        adjusted = dict(snapshot)
        for resource in ("ram", "vram"):
            total = _as_int(snapshot.get(f"{resource}_total_bytes"))
            free = _as_int(snapshot.get(f"{resource}_free_bytes")) + _as_int(release.get(f"{resource}_bytes"))
            adjusted[f"{resource}_free_bytes"] = min(total, free) if total > 0 else free
        return adjusted

    def estimate_settings(
        self,
        settings: Mapping[str, Any],
        *,
        boot_file: BootFile,
        environ: Mapping[str, str] | None = None,
        action: str | None = None,
    ) -> dict[str, Any]:
        if not self._busy.acquire(blocking=False):
            raise EstimateUnavailable("estimate_busy", "another memory estimate is already running")
        try:
            canonical = prepare_settings(settings, boot_file=boot_file)
            env = dict(os.environ if environ is None else environ)
            release = self._release_for(action)
            before = self._with_release(self._take_snapshot(env), release)
            for attempt in range(2):
                planner = self._run_child(canonical, before, env)
                after = self._with_release(self._take_snapshot(env), release)
                if _resource_changed(before, after):
                    if attempt == 0:
                        before = after
                        continue
                    raise EstimateUnavailable(
                        "stale_resources", "RAM or VRAM changed while the estimate was running"
                    )
                now = _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds").replace(
                    "+00:00", "Z"
                )
                output = self._assemble(planner, canonical, before, now)
                output["release"] = dict(release) if release else None
                return output
            raise EstimateUnavailable("stale_resources", "resource probes remained unstable")
        finally:
            self._busy.release()


def estimate_settings(
    settings: dict,
    *,
    boot_file: BootFile,
    environ: dict[str, str] | None = None,
    action: str | None = None,
) -> dict:
    """Estimate one canonical launch through the serving-v Python subprocess."""
    return MemoryFitService().estimate_settings(settings, boot_file=boot_file, environ=environ, action=action)


__all__ = [
    "EstimateUnavailable",
    "MachineSnapshot",
    "MemoryFitService",
    "PROTOCOL_VERSION",
    "SettingsValidationError",
    "estimate_settings",
    "prepare_settings",
    "probe_machine",
]
