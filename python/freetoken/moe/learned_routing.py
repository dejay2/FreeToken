"""Learned MoE routing statistics: the per-(layer, expert) decode histogram saved beside the
checkpoint, and the layer ranking derived from it.

The engine arms the device-side histogram (``OffloadMoeCache.collect_decode_freq``) whenever
routing learning is on, the scheduler flushes it here on a timer and at shutdown, and the next
boot ranks the MoE layers by how *broad* their routing was so ``--moe-gpu-owned-layers auto:N``
keeps the N hungriest layers on the card for the way this machine actually uses the model.

Why breadth and not the realized LRU miss rate: a GPU-owned layer never misses (its
``miss_rate`` is null by construction), so a miss-rate ranking could never re-evaluate the
layers it already chose. The histogram counts owned layers too. Measured on the four Qwen3.8
captures in ``docs/research/routing-skew-2026-09-02`` (48 layers x 512 experts, 6750 slots),
ranking by "experts needed to cover 90 % of a layer's routes" is a close proxy for the
miss-rate-derived ``GPU_OWNED_LAYER_RANK``: on the pooled histogram (what the recorder
accumulates) it agrees on five of the top six (layer 5, ninth in the measured order, replaces
layer 22, sixth), seven of the top eight, Spearman 0.98 over all 48 layers; averaging the
score per workload instead gives the measured top six exactly. Entropy, working-set size and
"oracle miss at the slot budget" score no better, and this one needs no slot budget.

The file is plain JSON, one per checkpoint directory, torch-free to read. Counts stored from
earlier sessions are halved on each boot that records new routes (``PRIOR_DECAY``; an idle boot
leaves the file alone) so the ranking follows recent use instead of the first week's workload
forever.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from freetoken.utils import init_logger

logger = init_logger(__name__)

ROUTING_STATS_FILENAME = "freetoken-routing-stats.json"
SCHEMA_VERSION = 1
# Routes per layer (steps x top_k) before a learned ranking replaces the fixed list: 4,000
# routes is 500 decode steps at top_k 8 -- one or two short answers. Below that the histogram
# is a handful of warm-up tokens and the fixed measured order is the better guess.
MIN_LEARNED_ROUTES = 4_000
# Fraction the stored prior keeps at each boot.
PRIOR_DECAY = 0.5
# The scheduler flushes at most this often, checked once per loop iteration; the loops block
# while idle, so a killed process loses what was routed since the last flush (at most one
# minute of an active session, but everything since the last active minute of an idle one).
FLUSH_INTERVAL_S = 60.0


@dataclass
class RoutingStats:
    num_layers: int
    num_experts: int
    freq: list[list[int]]
    boots: int = 0
    updated: str = ""
    top_k: int | None = None
    routes: int = 0  # max over layers of the row sum -- the learned-ranking guard reads this

    @classmethod
    def empty(
        cls, num_layers: int, num_experts: int, *, top_k: int | None = None
    ) -> "RoutingStats":
        return cls(
            num_layers, num_experts, [[0] * num_experts for _ in range(num_layers)], top_k=top_k
        )


def routing_stats_path(model_path: str | os.PathLike) -> Path:
    return Path(model_path) / ROUTING_STATS_FILENAME


def rank_layers_by_breadth(freq: list[list[int]], *, mass: float = 0.9) -> list[int]:
    """MoE layer ids, hungriest first: by the number of experts that carry ``mass`` of the
    layer's routes (more = broader routing = higher streaming miss rate), ties by the larger
    working set, then by the lower layer id. A layer with no routes sorts last."""
    keyed = []
    for layer, row in enumerate(freq):
        total = sum(row)
        if total <= 0:
            keyed.append((0, 0, layer))
            continue
        needed = 0
        covered = 0
        for count in sorted(row, reverse=True):
            covered += count
            needed += 1
            if covered >= mass * total:
                break
        working_set = sum(1 for c in row if c > 0)
        keyed.append((needed, working_set, layer))
    keyed.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [layer for _, _, layer in keyed]


def cold_experts(freq: list[list[int]]) -> list[list[int]]:
    """Per layer, the expert ids that were never routed to in the recorded window."""
    return [[e for e, c in enumerate(row) if c == 0] for row in freq]


def load_routing_stats(
    model_path: str | os.PathLike, *, num_layers: int, num_experts: int
) -> RoutingStats | None:
    """The saved stats for this checkpoint, or None (missing, unreadable, or another shape).

    Never raises: a bad file only costs the learned ranking, never a boot."""
    path = routing_stats_path(model_path)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"schema {doc.get('schema')!r}, expected {SCHEMA_VERSION}")
        freq = doc["decode_freq"]
        if len(freq) != num_layers or any(len(row) != num_experts for row in freq):
            width = len(freq[0]) if freq else 0
            raise ValueError(f"shape {len(freq)}x{width}, expected {num_layers}x{num_experts}")
        freq = [[int(c) for c in row] for row in freq]
        return RoutingStats(
            num_layers=num_layers,
            num_experts=num_experts,
            freq=freq,
            boots=int(doc.get("boots", 0)),
            updated=str(doc.get("updated", "")),
            top_k=doc.get("top_k"),
            routes=max((sum(row) for row in freq), default=0),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning_rank0("ignoring routing stats at %s: %s", path, exc)
        return None


def learned_layer_rank(
    model_path: str | os.PathLike, *, num_layers: int, num_experts: int
) -> tuple[list[int] | None, str]:
    """(ranking, reason). The ranking is None when there are no usable stats yet."""
    stats = load_routing_stats(model_path, num_layers=num_layers, num_experts=num_experts)
    path = routing_stats_path(model_path)
    if stats is None:
        return None, f"no routing stats at {path}"
    if stats.routes < MIN_LEARNED_ROUTES:
        return None, (
            f"routing stats at {path} hold {stats.routes} routes per layer, below the "
            f"{MIN_LEARNED_ROUTES} needed for a learned ranking"
        )
    return rank_layers_by_breadth(stats.freq), (
        f"routing stats at {path}: {stats.routes} routes per layer over {stats.boots} boots, "
        f"updated {stats.updated or 'never'}"
    )


class RoutingStatsRecorder:
    """Merges the live histogram into the saved file.

    ``note`` takes the cache's cumulative histogram; the recorder keeps the last snapshot it
    merged so only the new counts are added, and treats any cell that went DOWN as a
    ``reset_stats`` window boundary (``GET /v1/cache/routing?reset=true``), after which the
    whole current histogram is new."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int | None = None,
        prior: RoutingStats | None = None,
        decay: float = PRIOR_DECAY,
    ) -> None:
        self.path = routing_stats_path(model_path)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.decay = decay
        if prior is None:
            self.total = RoutingStats.empty(num_layers, num_experts, top_k=top_k)
        else:
            self.total = RoutingStats(
                num_layers,
                num_experts,
                [list(row) for row in prior.freq],
                boots=prior.boots,
                top_k=top_k if top_k is not None else prior.top_k,
            )
        # The prior is decayed and the boot counted at the FIRST merge that brings new routes,
        # not at construction: a boot that serves nothing (a restart for a test, a crashed
        # boot) must leave the file exactly as it was. R2 measured three idle restarts taking a
        # 20,000-route file to 2,500 and `auto` silently reverting to the fixed order.
        self._started = False
        self._snapshot: list[list[int]] = [[0] * num_experts for _ in range(num_layers)]
        self.dirty = False
        self.saves = 0
        self.last_error: str | None = None

    def note(self, histogram: list[list[int]]) -> int:
        """Merge the cache's cumulative histogram; returns the number of new routes merged."""
        if len(histogram) != self.num_layers or any(
            len(row) != self.num_experts for row in histogram
        ):
            raise ValueError("routing histogram shape does not match the recorder")
        reset = any(
            cur < old
            for row_cur, row_old in zip(histogram, self._snapshot)
            for cur, old in zip(row_cur, row_old)
        )
        deltas: list[list[int]] = []
        merged = 0
        for layer, row in enumerate(histogram):
            snap = self._snapshot[layer]
            delta_row = [cur if reset else cur - snap[expert] for expert, cur in enumerate(row)]
            merged += sum(delta_row)
            deltas.append(delta_row)
        if merged <= 0:
            # Nothing to merge: leave the snapshot where it is so no delta is ever dropped.
            return 0
        self._snapshot = [list(row) for row in histogram]
        if not self._started:
            self._started = True
            self.total.boots += 1
            for row in self.total.freq:
                for expert, count in enumerate(row):
                    row[expert] = int(count * self.decay)
        for tot, delta_row in zip(self.total.freq, deltas):
            for expert, delta in enumerate(delta_row):
                if delta:
                    tot[expert] += delta
        self.dirty = True
        return merged

    def save(self) -> bool:
        """Write the file atomically. False (and a one-line warning, once) when the checkpoint
        directory is not writable -- learning is an optimisation, never a boot failure."""
        if not self.dirty:
            return False
        self.total.routes = max((sum(row) for row in self.total.freq), default=0)
        self.total.updated = time.strftime("%Y-%m-%dT%H:%M:%S")
        doc = {
            "schema": SCHEMA_VERSION,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.total.top_k,
            "boots": self.total.boots,
            "routes": self.total.routes,
            "updated": self.total.updated,
            "layer_rank": rank_layers_by_breadth(self.total.freq),
            "cold_experts": cold_experts(self.total.freq),
            "decode_freq": self.total.freq,
        }
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            if self.last_error is None:
                logger.warning_rank0("cannot save routing stats to %s: %s", self.path, exc)
            self.last_error = str(exc)
            try:
                tmp.unlink()
            except OSError:
                pass
            return False
        self.dirty = False
        self.saves += 1
        self.last_error = None
        return True
