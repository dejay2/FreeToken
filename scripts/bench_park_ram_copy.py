"""CPU microbench: KV-park RAM save / continuation / restore, per-view path vs page-major gather.

Shipping Qwen3.8-Flash-Next geometry on one rank: 12 QSA layers, 2 KV heads x 256 FP8 K/V,
per-token FP32 scales, a 12-layer x 128-dim BF16 index at ratio 4, page size 64 -> 60 views
and 847,872 bytes per page; a 74-view, 115,642,376-byte GDN state slot. The default entry is
782 pages (50,048 tokens, 742 MiB with state), the live 2026-09-23 case.

    PYTHONPATH=python python scripts/bench_park_ram_copy.py [--pages 782] [--extra 8]

This runs on the CPU pool, so it measures dispatch and memcpy cost, not PCIe or WSL; the
per-view path's per-call overhead is what the page-major path removes on the GPU too.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import torch

import freetoken.kvcache.mha_pool as mha_pool
from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.park_store import ParkStore
from freetoken.kvcache.qsa_pool import QSAKVCache

PAGE = 64


class _StatePool:
    """74 views, 115,642,376 bytes: 36 conv + 36 recurrent + 2 PLE sibling tensors."""

    padding_slot = 0

    def __init__(self, slots: int = 2) -> None:
        self.conv = torch.zeros(36, slots, 65_536, dtype=torch.uint8)
        self.rec = torch.zeros(36, slots, 3_145_728, dtype=torch.uint8)
        self.ple = torch.zeros(2, slots, 18_436, dtype=torch.uint8)

    def slot_byte_views(self, slot: int):
        return (
            *(self.conv[i, slot] for i in range(36)),
            *(self.rec[i, slot] for i in range(36)),
            *(self.ple[i, slot] for i in range(2)),
        )

    def bytes_per_slot(self) -> int:
        return sum(v.numel() for v in self.slot_byte_views(0))


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def wrap(self, store: ParkStore, name: str) -> None:
        real = getattr(store, name)

        def timed(*args, **kwargs):
            mark = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - mark) * 1e3

        setattr(store, name, timed)

    @contextmanager
    def stage(self, name: str):
        mark = time.perf_counter()
        yield
        self.ms[name] = (time.perf_counter() - mark) * 1e3


def _run(kv, state, pages: int, extra: int, paged: bool) -> dict[str, float]:
    store = ParkStore(
        mode="ram",
        page_size=PAGE,
        kv_pool=kv,
        state_pool=state,
        fingerprint="bench",
        min_tokens=PAGE,
        ram_budget_bytes=8 << 30,
        ssd_dir="/nonexistent",
        disk_budget_bytes=0,
        pinned_window_bytes=256 << 20,
    )
    if not paged:
        store._page_source = lambda *_a, **_k: None
    timer = _Timer()
    for name in (
        "_entry_views", "_page_source", "_copy_to_ram", "_ram_prefix_matches",
        "_split_views", "_restore_ram",
    ):
        timer.wrap(store, name)
    total = pages + extra
    order = torch.randperm(total, generator=torch.Generator().manual_seed(0))
    bases = (order * PAGE).to(torch.int32)
    tokens = torch.arange(total * PAGE, dtype=torch.int32)
    with timer.stage("save_root_total"):
        assert store.save(tokens[: pages * PAGE], bases[:pages], 0)
    root = dict(timer.ms)
    timer.ms.clear()
    with timer.stage("save_continuation_total"):
        assert store.save(tokens, bases, 0)
    cont = dict(timer.ms)
    timer.ms.clear()
    entry = store.lookup(tokens)
    assert entry is not None and entry.parent_key is not None
    target = torch.flip(bases, [0])
    with timer.stage("restore_total"):
        store.restore(entry, target, 1)
    rest = dict(timer.ms)
    store.close()
    out = {f"root.{k}": v for k, v in root.items()}
    out.update({f"cont.{k}": v for k, v in cont.items()})
    out.update({f"restore.{k}": v for k, v in rest.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=782)
    parser.add_argument("--extra", type=int, default=8, help="continuation pages")
    args = parser.parse_args()
    mha_pool.get_tp_info = lambda: DistributedInfo(rank=0, size=1)
    kv = QSAKVCache(
        num_kv_heads=2, num_layers=12, head_dim=256, num_pages=args.pages + args.extra + 1,
        page_size=PAGE, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn,
        device=torch.device("cpu"), index_head_dim=128, num_index_layers=12, index_ratio=4,
        num_req_slots=2, layer_ids=tuple(range(12)),
    )
    page_bytes = sum(v.numel() * v.element_size() for v in kv.page_byte_views(0))
    state = _StatePool()
    print(
        f"pages={args.pages}+{args.extra} views/page={len(kv.page_byte_views(0))} "
        f"page_bytes={page_bytes} entry={(args.pages * page_bytes + state.bytes_per_slot()) / 2**20:.0f} MiB "
        f"threads={torch.get_num_threads()}"
    )
    old = _run(kv, state, args.pages, args.extra, paged=False)
    new = _run(kv, state, args.pages, args.extra, paged=True)
    keys = sorted(set(old) | set(new))
    print(f"{'stage':40s} {'per-view ms':>12s} {'page-major ms':>14s}")
    for key in keys:
        a, b = old.get(key), new.get(key)
        fa = f"{a:12.1f}" if a is not None else f"{'-':>12s}"
        fb = f"{b:14.1f}" if b is not None else f"{'-':>14s}"
        print(f"{key:40s} {fa} {fb}")


if __name__ == "__main__":
    main()
