"""Train per-layer expert predictors on the FREETOKEN_MOE_PREDICT_DUMP tensors.

    python scripts/diag/train_predictor.py <dump-dir> [--epochs 40] [--layers 10,11]

``analyze_predict.py`` measures the ceiling of the cheapest predictor (layer L+1's own
router run early). This measures the cheapest predictor that could actually be AFFORDED at
layer L: a small head on layer L's router input that names layer L+1's experts directly, so
the fetch can start a layer ahead.

Two heads per layer, both trained with BCE against the multi-hot top-10 target:

    linear   H -> num_experts
    mlp      H -> 1024 -> num_experts

Split 80/20 by token POSITION (not at random): a random split leaks, because neighbouring
tokens of one prompt are near-duplicates. CPU only -- the GPU belongs to the server.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import torch


def load_layer(paths: list):
    """-> (x [N,H] float32, targets {"l1","l2"} [N,E] multi-hot, positions [N], num_experts)."""
    xs, l1s, l2s, positions = [], [], [], []
    num_experts = 0
    for path in paths:
        # weights_only: the dumps are tensors, ints and lists of ints, and nothing here
        # should ever unpickle arbitrary objects out of a diagnostics directory.
        blob = torch.load(path, map_location="cpu", weights_only=True)
        xs.append(blob["x"].to(torch.float32))
        l1s.append(blob["top10_l1"].to(torch.int64))
        l2s.append(blob["top10_l2"].to(torch.int64))
        positions.extend(blob["positions"])
        num_experts = max(num_experts, int(blob["num_experts"]))
    x = torch.cat(xs)
    ids = {"l1": torch.cat(l1s), "l2": torch.cat(l2s)}
    targets = {}
    for name, rows in ids.items():
        multi = torch.zeros(rows.shape[0], num_experts)
        multi.scatter_(1, rows.clamp(0, num_experts - 1), 1.0)
        targets[name] = multi
    return x, ids, targets, torch.tensor(positions, dtype=torch.int64), num_experts


def build(kind: str, hidden: int, num_experts: int) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(hidden, num_experts)
    return torch.nn.Sequential(
        torch.nn.Linear(hidden, 1024), torch.nn.GELU(), torch.nn.Linear(1024, num_experts)
    )


def recall_at(logits: torch.Tensor, ids: torch.Tensor, k: int) -> float:
    """Mean fraction of the true top-10 that appears in the head's own top-k."""
    predicted = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    hits = (predicted.unsqueeze(1) == ids.unsqueeze(2)).any(dim=2).float()
    return float(hits.mean())


def train_one(x_tr, y_tr, x_te, ids_te, kind, num_experts, epochs, batch, lr, ks):
    torch.manual_seed(0)
    model = build(kind, x_tr.shape[1], num_experts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = x_tr.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            index = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tr[index]), y_tr[index])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(x_te)
        return {k: recall_at(logits, ids_te, k) for k in ks}, float(loss)


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="the FREETOKEN_MOE_PREDICT_DUMP directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ks", default="10,15,20")
    parser.add_argument("--targets", default="l1,l2", help="l1 = layer L+1, l2 = layer L+2")
    parser.add_argument("--layers", default="", help="comma-separated layer ids (default: all)")
    args = parser.parse_args(argv)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    wanted = {int(v) for v in args.layers.split(",") if v.strip()}
    targets = [t for t in args.targets.split(",") if t.strip()]

    by_layer: dict = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.path, "x_layer*_*.pt"))):
        layer_id = int(os.path.basename(path).split("_")[1][5:])
        if not wanted or layer_id in wanted:
            by_layer[layer_id].append(path)
    if not by_layer:
        print(f"no x_layer*.pt dumps under {args.path}")
        return 2

    print(f"{'layer':>6} {'target':>7} {'kind':>7} {'train':>7} {'test':>6} " +
          " ".join(f"{'r@' + str(k):>7}" for k in ks) + f" {'sec':>6}")
    for layer_id in sorted(by_layer):
        x, ids, multi, positions, num_experts = load_layer(by_layer[layer_id])
        # 80/20 by position: the last fifth of each dumped context is held out.
        order = torch.argsort(positions, stable=True)
        cut = int(0.8 * order.numel())
        train_index, test_index = order[:cut], order[cut:]
        if train_index.numel() < 2 or test_index.numel() < 1:
            print(f"{layer_id:>6}  too few tokens ({order.numel()})")
            continue
        for target in targets:
            for kind in ("linear", "mlp"):
                started = time.perf_counter()
                scores, _ = train_one(
                    x[train_index], multi[target][train_index],
                    x[test_index], ids[target][test_index],
                    kind, num_experts, args.epochs, args.batch, args.lr, ks,
                )
                cells = " ".join(f"{scores[k]:>7.3f}" for k in ks)
                print(
                    f"{layer_id:>6} {target:>7} {kind:>7} {train_index.numel():>7} "
                    f"{test_index.numel():>6} {cells} {time.perf_counter() - started:>6.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
