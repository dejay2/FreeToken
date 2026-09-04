"""Serial safetensors loader for the fixed K=2/mul1 GLM EXL3 expert banks.

The proof keeps the routed experts compressed in per-layer host banks for streaming
layers.  A layer marked GPU_OWNED by the ambient residency plan instead receives the
same nine compressed banks as device tensors, so its experts never consume host-bank
memory.  Each selected expert is reconstructed by the card-side operation later; this
module only validates and places the nine stored bank components.  The loader deliberately
reads one safetensors tensor at a time on Windows through ``DirectShard`` so the
operating-system file cache does not retain a second copy beside the roughly 71.29 GiB
of streaming banks.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from tqdm import tqdm

from freetoken.models.loader import drop_page_cache as _drop_page_cache
from freetoken.utils import download_hf_weight

# The cache and the reconstruct-first operation both consume this order.  The scalar
# ``mul1`` marker is validated from the checkpoint but is intentionally not a bank: the
# proof fixes one codebook and one K for every routed projection.
EXL3_BANK_NAMES = (
    "gate_trellis",
    "gate_suh",
    "gate_svh",
    "up_trellis",
    "up_suh",
    "up_svh",
    "down_trellis",
    "down_suh",
    "down_svh",
)
_EXL3_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_EXL3_COMPONENTS = ("trellis", "suh", "svh", "mul1")
_EXL3_BANK_COMPONENTS = ("trellis", "suh", "svh")
_EXL3_K = 2

# This is deliberately broad after ``experts.<id>``.  Unknown projections and component
# names must fail loudly instead of being mistaken for unrelated model tensors; layer 45
# is filtered first so the trailing MTP expert set remains deliberately ignored.
_EXL3_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>[^.]+)\.(?P<kind>[^.]+)$"
)

# Safetensors header dtype spellings used by the checkpoint.  Header validation happens
# before any large bank allocation; the data pass checks the resulting torch tensor too.
_ST_DTYPE = {
    "I16": torch.int16,
    "F16": torch.float16,
    "I32": torch.int32,
}


@dataclass(frozen=True)
class Exl3ExpertRecord:
    """One validated routed-expert component in a checkpoint shard."""

    name: str
    shard: str
    layer: int
    bank_layer: int
    expert: int
    proj: str
    kind: str

    @property
    def identity(self) -> tuple[int, int, str, str]:
        return self.bank_layer, self.expert, self.proj, self.kind


class _PlainBank:
    """CPU-only bank wrapper used by dummy loads when no CUDA pinning exists."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    @property
    def fill(self) -> torch.Tensor:
        return self.tensor

    def pin(self) -> None:
        pass


# --------------------------------------------------------------------------------------
# Checkpoint metadata and geometry
# --------------------------------------------------------------------------------------


def _read_safetensors_header(path: str) -> dict:
    """Read only a shard header, without mapping or touching its tensor payload."""
    with open(path, "rb") as fh:
        raw_size = fh.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"EXL3 shard {path!r} has no complete header length")
        header_size = struct.unpack("<Q", raw_size)[0]
        raw_header = fh.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(f"EXL3 shard {path!r} has a truncated header")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"EXL3 shard {path!r} has invalid safetensors metadata") from exc
    if not isinstance(header, dict):
        raise ValueError(f"EXL3 shard {path!r} header is not an object")
    return header


def _weight_map(folder: str) -> dict[str, str]:
    """Return logical tensor names mapped to root-relative shard names."""
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("EXL3 model.safetensors.index.json has no weight_map object")
        return {str(name): str(shard) for name, shard in weight_map.items()}

    # The downloaded GLM branch has an index, but accepting a single-file checkpoint keeps
    # this provider useful for small fixtures and follows the generic loader's behavior.
    result: dict[str, str] = {}
    shards = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not shards:
        raise ValueError(f"no safetensors shards found in EXL3 model directory {folder!r}")
    for path in shards:
        shard = os.path.basename(path)
        for name in _read_safetensors_header(path):
            if name == "__metadata__":
                continue
            if name in result:
                raise ValueError(f"duplicate safetensors tensor name {name!r}")
            result[name] = shard
    return result


def _config_geometry(config) -> tuple[int, int, int, int, int]:
    """Return ``(num_bank_layers, experts, hidden, intermediate, first_sparse_layer)``."""
    first = int(getattr(config, "first_k_dense_replace", 0))
    num_layers = getattr(config, "num_layers", None)
    num_bank_layers = getattr(config, "num_moe_layers", None)
    if num_bank_layers is None:
        if num_layers is None:
            raise ValueError("EXL3 loader needs config.num_layers or config.num_moe_layers")
        num_bank_layers = int(num_layers) - first
    else:
        num_bank_layers = int(num_bank_layers)
    if num_layers is not None and int(num_layers) != first + num_bank_layers:
        raise ValueError(
            "EXL3 config has inconsistent num_layers/first_k_dense_replace/num_moe_layers: "
            f"{num_layers}, {first}, {num_bank_layers}"
        )

    experts = int(getattr(config, "num_experts"))
    hidden = int(getattr(config, "hidden_size"))
    intermediate = int(getattr(config, "moe_intermediate_size"))
    if num_bank_layers <= 0 or experts <= 0 or hidden <= 0 or intermediate <= 0:
        raise ValueError(
            "EXL3 config dimensions must be positive: "
            f"layers={num_bank_layers}, experts={experts}, hidden={hidden}, intermediate={intermediate}"
        )
    # The reconstruction extension's H128 transforms require these dimensions to be
    # multiples of 128.  GLM uses H=4096/I=2048; fixtures use small 128-multiple shapes.
    if hidden % 128 or intermediate % 128:
        raise ValueError(
            "EXL3 reconstruction requires hidden_size and moe_intermediate_size divisible "
            f"by 128, got hidden_size={hidden}, moe_intermediate_size={intermediate}"
        )
    return num_bank_layers, experts, hidden, intermediate, first


def _expected_shape(proj: str, kind: str, hidden: int, intermediate: int) -> tuple[int, ...]:
    if proj in ("gate_proj", "up_proj"):
        trellis = (hidden // 16, intermediate // 16, 16 * _EXL3_K)
        input_size, output_size = hidden, intermediate
    elif proj == "down_proj":
        trellis = (intermediate // 16, hidden // 16, 16 * _EXL3_K)
        input_size, output_size = intermediate, hidden
    else:
        raise ValueError(f"unsupported EXL3 expert projection {proj!r}")
    if kind == "trellis":
        return trellis
    if kind == "suh":
        return (input_size,)
    if kind == "svh":
        return (output_size,)
    if kind == "mul1":
        return ()
    raise ValueError(f"unsupported EXL3 component {kind!r}")


def _expected_dtype(kind: str) -> str:
    if kind == "trellis":
        return "I16"
    if kind in ("suh", "svh"):
        return "F16"
    if kind == "mul1":
        return "I32"
    raise ValueError(f"unsupported EXL3 component {kind!r}")


def _collect_records(
    weight_map: dict[str, str], config
) -> tuple[dict[tuple[int, int, str, str], Exl3ExpertRecord], tuple[int, int, int, int, int]]:
    """Collect and validate routed names before allocating any bank storage."""
    num_layers, experts, hidden, intermediate, first = _config_geometry(config)
    records: dict[tuple[int, int, str, str], Exl3ExpertRecord] = {}
    components: dict[tuple[int, int, str], set[str]] = {}
    for name, shard in weight_map.items():
        match = _EXL3_KEY_RE.fullmatch(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        # The main GLM language stack is layers 0..44; layer 45 is MTP and is ignored.
        # Dense prefix layers are likewise outside the routed-bank contract.
        if layer < first or layer >= first + num_layers:
            continue
        expert = int(match.group("expert"))
        if expert >= experts:
            raise ValueError(
                f"EXL3 expert id {expert} at checkpoint layer {layer} is outside [0, {experts})"
            )
        proj = match.group("proj")
        kind = match.group("kind")
        if proj not in _EXL3_PROJECTIONS:
            raise ValueError(f"unsupported EXL3 expert projection {proj!r} in {name!r}")
        if kind == "mcg":
            raise ValueError(
                f"EXL3 checkpoint uses unsupported mcg codebook in {name!r}; "
                "the proof accepts the mul1 codebook only"
            )
        if kind not in _EXL3_COMPONENTS:
            raise ValueError(f"unsupported EXL3 expert component {kind!r} in {name!r}")
        bank_layer = layer - first
        identity = (bank_layer, expert, proj, kind)
        if identity in records:
            previous = records[identity]
            raise ValueError(
                "duplicate EXL3 component completion for "
                f"layer={layer}, expert={expert}, projection={proj}, component={kind} "
                f"({previous.name!r} and {name!r})"
            )
        records[identity] = Exl3ExpertRecord(
            name=name,
            shard=shard,
            layer=layer,
            bank_layer=bank_layer,
            expert=expert,
            proj=proj,
            kind=kind,
        )
        components.setdefault((bank_layer, expert, proj), set()).add(kind)

    expected = set(_EXL3_COMPONENTS)
    for bank_layer in range(num_layers):
        layer = first + bank_layer
        for expert in range(experts):
            for proj in _EXL3_PROJECTIONS:
                got = components.get((bank_layer, expert, proj), set())
                if got != expected:
                    missing = sorted(expected - got)
                    extra = sorted(got - expected)
                    detail = []
                    if missing:
                        detail.append(f"missing {missing}")
                    if extra:
                        detail.append(f"unexpected {extra}")
                    raise ValueError(
                        "incomplete EXL3 expert components for "
                        f"layer={layer}, expert={expert}, projection={proj}: "
                        + "; ".join(detail)
                    )

    expected_records = num_layers * experts * len(_EXL3_PROJECTIONS) * len(_EXL3_COMPONENTS)
    if len(records) != expected_records:
        raise ValueError(
            f"EXL3 checkpoint has {len(records)} routed components; "
            f"expected {expected_records} for {num_layers} layers x {experts} experts"
        )
    return records, (num_layers, experts, hidden, intermediate, first)


def _validate_headers(
    folder: str,
    records: dict[tuple[int, int, str, str], Exl3ExpertRecord],
    geometry: tuple[int, int, int, int, int],
) -> dict[str, list[Exl3ExpertRecord]]:
    """Validate all component headers before allocating the roughly 71 GiB bank set."""
    _num_layers, _experts, hidden, intermediate, _first = geometry
    by_shard: dict[str, list[Exl3ExpertRecord]] = {}
    for record in records.values():
        by_shard.setdefault(record.shard, []).append(record)

    for shard in sorted(by_shard):
        path = os.path.join(folder, shard)
        if not os.path.isfile(path):
            raise ValueError(f"EXL3 shard {path!r} named by the index does not exist")
        header = _read_safetensors_header(path)
        for record in by_shard[shard]:
            meta = header.get(record.name)
            if not isinstance(meta, dict):
                raise ValueError(
                    f"EXL3 index names {record.name!r}, but shard {shard!r} has no such tensor"
                )
            shape = tuple(meta.get("shape", ()))
            expected_shape = _expected_shape(record.proj, record.kind, hidden, intermediate)
            if record.kind == "trellis":
                last = shape[-1] if shape else 0
                if not last or last % 16:
                    raise ValueError(
                        f"EXL3 {record.name!r} has trellis shape {shape}; cannot derive an integer K"
                    )
                k = last // 16
                if k != _EXL3_K:
                    raise ValueError(
                        f"EXL3 {record.name!r} has K={k}; the proof accepts K=2 only"
                    )
            if shape != expected_shape:
                raise ValueError(
                    f"wrong EXL3 shape for {record.name!r}: got {shape}, "
                    f"expected {expected_shape}"
                )
            dtype = str(meta.get("dtype", ""))
            expected_dtype = _expected_dtype(record.kind)
            if dtype != expected_dtype:
                raise ValueError(
                    f"wrong EXL3 dtype for {record.name!r}: got {dtype!r}, "
                    f"expected {expected_dtype!r}"
                )
    return by_shard


def _open_shard(path: str, *, whole: bool = False):
    """Open a shard for per-tensor reads, using the shared Windows direct reader.

    Unlike the other model loaders, EXL3 cannot accept a cached-read fallback: its pinned
    banks already consume roughly 71.29 GiB, so a second Windows file-cache copy can exhaust
    host memory before the model is ready. ``DirectShard`` therefore receives its strict
    mode here, while the generic reader keeps its existing fallback for other formats.
    """
    if whole:
        raise ValueError("the EXL3 proof loader only supports per-tensor shard reads")
    from freetoken.moe import win_io

    if os.name == "nt":
        if not win_io.enabled():
            raise RuntimeError(
                "EXL3 Windows loading requires unbuffered shard reads; "
                "unset FREETOKEN_WIN_UNBUFFERED_IO=0 or set it to 1"
            )
        from freetoken.models.weight import DirectShard

        return DirectShard(path, whole=False, unbuffered_only=True)
    if win_io.enabled():
        from freetoken.models.weight import DirectShard

        return DirectShard(path, whole=False, unbuffered_only=True)
    return safetensors.safe_open(path, framework="pt", device="cpu")


def _preflight_windows_reads(folder: str, by_shard: dict[str, list[Exl3ExpertRecord]]) -> None:
    """Prove every EXL3 shard opens unbuffered before allocating the large host banks."""
    if os.name != "nt":
        return
    from freetoken.moe import win_io

    if not win_io.enabled():
        raise RuntimeError(
            "EXL3 Windows loading requires unbuffered shard reads; "
            "FREETOKEN_WIN_UNBUFFERED_IO=0 is not allowed for this format"
        )
    for shard in sorted(by_shard):
        path = os.path.join(folder, shard)
        try:
            with _open_shard(path, whole=False):
                pass
        except Exception as exc:
            raise RuntimeError(
                f"EXL3 shard {path!r} could not be opened with Windows unbuffered reads"
            ) from exc


# --------------------------------------------------------------------------------------
# Bank allocation and data placement
# --------------------------------------------------------------------------------------


def _bank_specs(experts: int, hidden: int, intermediate: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return the nine fixed [expert, ...] bank shapes in cache registration order."""
    trellis_gate_up = (experts, hidden // 16, intermediate // 16, 16 * _EXL3_K)
    trellis_down = (experts, intermediate // 16, hidden // 16, 16 * _EXL3_K)
    return {
        "gate_trellis": (trellis_gate_up, torch.int16),
        "gate_suh": ((experts, hidden), torch.float16),
        "gate_svh": ((experts, intermediate), torch.float16),
        "up_trellis": (trellis_gate_up, torch.int16),
        "up_suh": ((experts, hidden), torch.float16),
        "up_svh": ((experts, intermediate), torch.float16),
        "down_trellis": (trellis_down, torch.int16),
        "down_suh": ((experts, intermediate), torch.float16),
        "down_svh": ((experts, hidden), torch.float16),
    }


def _alloc_banks(
    num_layers: int, experts: int, hidden: int, intermediate: int
) -> dict[str, list]:
    from freetoken.moe.host_banks import alloc_layer_banks

    # alloc_layer_banks reads the engine's ambient residency plan: streaming layers get
    # HostBank storage, while GPU_OWNED layers get GpuOwnedBank device tensors and are filled
    # synchronously by this placement loop (the inference-mode thread rule matters here).
    return alloc_layer_banks(_bank_specs(experts, hidden, intermediate), num_layers)


def _bank_name(proj: str, kind: str) -> str:
    if kind not in _EXL3_BANK_COMPONENTS:
        raise ValueError(f"EXL3 marker {kind!r} is not a bank component")
    prefix = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[proj]
    return f"{prefix}_{kind}"


def _validate_loaded_tensor(record: Exl3ExpertRecord, tensor: torch.Tensor, hidden: int, intermediate: int) -> None:
    expected_shape = _expected_shape(record.proj, record.kind, hidden, intermediate)
    expected_dtype = _ST_DTYPE[_expected_dtype(record.kind)]
    if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
        raise ValueError(
            f"EXL3 tensor {record.name!r} arrived as shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}; expected shape={expected_shape}, dtype={expected_dtype}"
        )


def _place_records(
    folder: str,
    by_shard: dict[str, list[Exl3ExpertRecord]],
    banks: dict[str, list],
    geometry: tuple[int, int, int, int, int],
    sink,
    drop_page_cache: Callable[[str], None],
    primary: bool,
) -> tuple[int, int]:
    num_layers, experts, hidden, intermediate, _first = geometry
    from freetoken.moe.host_banks import LayerCompletionTracker

    # Each routed expert contributes nine bank rows; the marker records are checked and
    # consumed but do not count toward a bank-layer completion.
    tracker = LayerCompletionTracker(experts * len(EXL3_BANK_NAMES), banks, sink)
    placed_records = 0
    placed_rows = 0
    completed: set[tuple[int, int, str, str]] = set()
    for shard in tqdm(
        sorted(by_shard), desc="Loading EXL3 experts", disable=not primary
    ):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        try:
            # whole=False is intentional even for a large shard: every tensor is copied
            # directly into its final bank and no transient multi-GiB shard is retained.
            with _open_shard(path, whole=False) as reader:
                for record in sorted(by_shard[shard], key=lambda item: item.name):
                    if record.identity in completed:
                        raise ValueError(
                            "duplicate EXL3 component completion for "
                            f"layer={record.layer}, expert={record.expert}, "
                            f"projection={record.proj}, component={record.kind}"
                        )
                    tensor = reader.get_tensor(record.name)
                    try:
                        _validate_loaded_tensor(record, tensor, hidden, intermediate)
                        completed.add(record.identity)
                        placed_records += 1
                        if record.kind == "mul1":
                            # The metadata pass already established the scalar marker and kind.
                            # Reading it here proves the name exists in the chosen shard without
                            # retaining it in a bank.
                            continue
                        name = _bank_name(record.proj, record.kind)
                        banks[name][record.bank_layer].fill[record.expert].copy_(tensor)
                        tracker.note(record.bank_layer)
                        placed_rows += 1
                    finally:
                        # DirectShard keeps one mmap per requested tensor alive until its
                        # context exits. The copy above is final, so release that tiny
                        # transient immediately instead of accumulating a whole shard.
                        del tensor
                        alive = getattr(reader, "_alive", None)
                        if alive is not None:
                            alive.clear()
        finally:
            drop_page_cache(path)

    expected_records = num_layers * experts * len(_EXL3_PROJECTIONS) * len(_EXL3_COMPONENTS)
    expected_rows = num_layers * experts * len(EXL3_BANK_NAMES)
    if placed_records != expected_records:
        raise ValueError(
            f"EXL3 loader placed {placed_records} tensors; expected {expected_records}"
        )
    if placed_rows != expected_rows:
        raise ValueError(f"EXL3 loader placed {placed_rows} bank rows; expected {expected_rows}")
    return placed_records, placed_rows


def load_exl3_expert_source_banks(
    model_path: str,
    config,
    *,
    drop_page_cache: Callable[[str], None] = _drop_page_cache,
    primary: bool = True,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Load routed GLM EXL3 experts into nine per-layer source banks.

    Streaming layers use pinned host banks; GPU_OWNED layers use resident device banks from
    the ambient residency plan and are filled on this placement thread. Validation is split
    from placement: all names, component sets, dimensions, K values, dtypes and exact counts
    are checked from safetensors headers before the first bank is allocated. Main sparse
    layers only are mapped to bank layers; trailing MTP layer 45 and all unrelated tensors
    are ignored. ``layer_sink`` receives completed layers as ``{bank_name: HostBank}``,
    matching the NVFP4 loader's converter seam.
    """
    folder = download_hf_weight(model_path)
    weight_map = _weight_map(folder)
    records, geometry = _collect_records(weight_map, config)
    by_shard = _validate_headers(folder, records, geometry)
    # A direct-open failure must stop before the ~71.29 GiB bank allocation; otherwise the
    # generic DirectShard fallback could leave a cached second copy on the Windows standby list.
    _preflight_windows_reads(folder, by_shard)
    num_layers, experts, hidden, intermediate, _first = geometry
    banks = _alloc_banks(num_layers, experts, hidden, intermediate)

    if layer_sink is not None:
        _place_records(
            folder, by_shard, banks, geometry, layer_sink, drop_page_cache, primary
        )
    else:
        from freetoken.moe.host_banks import PinPipeline

        with PinPipeline() as pins:
            _place_records(
                folder, by_shard, banks, geometry, pins, drop_page_cache, primary
            )
    return {name: [bank.tensor for bank in banks[name]] for name in EXL3_BANK_NAMES}


# Provider wording in the batch plan uses the shorter ``sources`` name; keep both spellings
# as one implementation so callers cannot drift to different loader behavior.
load_exl3_expert_sources = load_exl3_expert_source_banks


def dummy_exl3_expert_sources(config) -> dict[str, list[torch.Tensor]]:
    """Fabricate zero EXL3 source banks for model-shape tests and dummy boots."""
    num_layers, experts, hidden, intermediate, _first = _config_geometry(config)
    specs = _bank_specs(experts, hidden, intermediate)
    if not torch.cuda.is_available():
        return {
            name: [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
            for name, (shape, dtype) in specs.items()
        }

    banks = _alloc_banks(num_layers, experts, hidden, intermediate)
    for name in EXL3_BANK_NAMES:
        for bank in banks[name]:
            bank.fill.zero_()
    from freetoken.moe.host_banks import PinPipeline

    with PinPipeline() as pins:
        for layer_id in range(num_layers):
            pins(layer_id, {name: banks[name][layer_id] for name in EXL3_BANK_NAMES})
    return {name: [bank.tensor for bank in banks[name]] for name in EXL3_BANK_NAMES}


__all__ = [
    "EXL3_BANK_NAMES",
    "Exl3ExpertRecord",
    "dummy_exl3_expert_sources",
    "load_exl3_expert_source_banks",
    "load_exl3_expert_sources",
]
