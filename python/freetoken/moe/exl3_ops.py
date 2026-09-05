"""EXL3 routed-expert operation names shared by the config, cache, engine and CLI.

Kept torch-free so ``engine/config.py`` and ``server/args.py`` can name the default without
importing a kernel module. The default is measured, not assumed: see the comment on
``EngineConfig.exl3_expert_op``.
"""

from __future__ import annotations

EXL3_EXPERT_OPS: tuple[str, ...] = ("reconstruct", "mgemm")
DEFAULT_EXL3_EXPERT_OP = "mgemm"
