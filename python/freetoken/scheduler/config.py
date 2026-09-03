from __future__ import annotations

from dataclasses import dataclass, field, replace

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def pin_kv_park_model_path_value(model_path: str, mode: str) -> str:
    """Resolve the one complete checkpoint snapshot used by every parking process."""
    if mode == "off":
        return model_path
    from freetoken.utils.hf import download_hf_snapshot

    return download_hf_snapshot(model_path)


def pin_kv_park_model_path(config: "SchedulerConfig") -> "SchedulerConfig":
    """Pin one Hub snapshot before Engine loads any bytes used by persistent KV."""
    resolved = pin_kv_park_model_path_value(config.model_path, config.kv_park)
    if resolved == config.model_path:
        return config
    # A fresh frozen config also drops any cached Hub-derived model_config from the mutable id.
    # Engine weights, PLE/GDN sibling state, tokenizer, and ParkStore fingerprint then all read the
    # same immutable snapshot directory even if the Hub branch advances while this process boots.
    return replace(config, model_path=resolved)


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
