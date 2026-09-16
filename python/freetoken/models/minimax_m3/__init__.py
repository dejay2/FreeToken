from .config import VisionConfig, parse_config, parse_vision_config
from .model import MiniMaxM3ForCausalLM, MiniMaxM3ForConditionalGeneration
from .weight import (
    iter_weights,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)

__all__ = [
    "MiniMaxM3ForConditionalGeneration",
    "MiniMaxM3VisionModel",
    "VisionConfig",
    "parse_vision_config",
    "MiniMaxM3ForCausalLM",
    "parse_config",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]

from .vision import MiniMaxM3VisionModel
