"""The dense EXL3 workspace is allocated where the boot can still account for it.

``prepare_exl3_dense_workspace`` must run after the weights land (it sizes itself from the
built Exl3Linear tree) and before the post-weights free-VRAM snapshot that the MoE cache and
KV budgets are solved from, and therefore before any CUDA graph captures its addresses. The
engine boot is one long ``__init__`` that cannot run without a GPU, so this pins the order
in its source, the same way test_moe_gpu_owned_layers pins its single call site.
"""

from __future__ import annotations

import inspect


def test_workspace_is_prepared_after_weights_and_before_the_budget_snapshot():
    from freetoken.engine.engine import Engine

    source = inspect.getsource(Engine.__init__)
    install = source.index("self._install_model_weights(config)")
    prepare = source.index("prepare_exl3_dense_workspace(self.model, self.device)")
    snapshot = source.index("post_weights_free = self._sync_get_memory()[0]")
    kv = source.index("_startup_kv_budget(")
    graphs = source.index("GraphRunner(")
    assert install < prepare < snapshot < kv < graphs
    assert source.count("prepare_exl3_dense_workspace(") == 1


def test_workspace_hook_is_keyed_on_exl3():
    # an NVFP4/bf16 boot must never import kernel/exl3.py (it loads exllamav3_ext at import)
    from freetoken.engine.engine import Engine

    source = inspect.getsource(Engine.__init__)
    guard = source.index('getattr(config.model_config, "linear_storage", "bf16") == "exl3"')
    assert guard < source.index("from freetoken.kernel.exl3_linear import prepare_exl3_dense_workspace")
