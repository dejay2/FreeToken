"""Recall of the routing-predictor study written by FREETOKEN_MOE_PREDICT_LOG.

    python scripts/diag/analyze_predict.py <dir-or-file.jsonl>

For every layer L the log holds this layer's ACTUAL top-10 and, scored on the same router
input, the top-20 of layers L+1 and L+2's routers. This script joins layer L's prediction to
layer L+1's actual decision -- the same tokens, the same forward -- and reports how much of
L+1's real routing L already knew:

    recall@k = |actual_top10(L+1) INTERSECT pred_next(L)[:k]| / 10

The token-loyalty baseline is the same quantity with L's OWN top-10 in place of the
prediction. Expert ids do not mean the same thing across layers, so it is not a rival
predictor -- it is the sanity floor that says how much of any recall is just "these two
layers happen to route to similarly numbered experts".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_records(path: str) -> list:
    files = (
        sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    )
    records = []
    for name in files:
        with open(name, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _key(record) -> tuple:
    """What identifies "the same tokens in the same forward" across layers."""
    return (record.get("phase"), tuple(record.get("positions") or ()), len(record["actual_top10"]))


def _recall(actual_rows, predicted_rows, k: int) -> tuple:
    """-> (summed recall, token count) of ``actual_rows`` inside the first k predictions."""
    total = 0.0
    tokens = 0
    for actual, predicted in zip(actual_rows, predicted_rows):
        actual = [e for e in actual if e >= 0]
        if not actual:
            continue
        head = set(predicted[:k])
        total += sum(1 for e in actual if e in head) / len(actual)
        tokens += 1
    return total, tokens


def analyze(records: list, ks: list) -> int:
    by_key: dict = defaultdict(dict)
    for record in records:
        by_key[_key(record)][int(record["layer"])] = record

    # layer -> stat -> [sum, tokens]
    stats: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for layers in by_key.values():
        for layer_id, record in layers.items():
            nxt = layers.get(layer_id + 1)
            nxt2 = layers.get(layer_id + 2)
            if nxt is not None and record.get("pred_next_top20"):
                for k in ks:
                    total, tokens = _recall(
                        nxt["actual_top10"], record["pred_next_top20"], k
                    )
                    slot = stats[layer_id][f"next@{k}"]
                    slot[0] += total
                    slot[1] += tokens
                total, tokens = _recall(nxt["actual_top10"], record["actual_top10"], 10)
                slot = stats[layer_id]["loyalty"]
                slot[0] += total
                slot[1] += tokens
            if nxt2 is not None and record.get("pred_next2_top20"):
                for k in ks:
                    total, tokens = _recall(
                        nxt2["actual_top10"], record["pred_next2_top20"], k
                    )
                    slot = stats[layer_id][f"next2@{k}"]
                    slot[0] += total
                    slot[1] += tokens

    if not stats:
        print("no (layer, layer+1) pairs in the log -- nothing to score")
        return 1

    columns = [f"next@{k}" for k in ks] + [f"next2@{k}" for k in ks] + ["loyalty"]
    header = f"{'layer':>6} {'tokens':>8} " + " ".join(f"{c:>10}" for c in columns)
    print(header)
    print("-" * len(header))
    overall: dict = defaultdict(lambda: [0.0, 0])
    for layer_id in sorted(stats):
        row = stats[layer_id]
        tokens = max((row[c][1] for c in columns if c in row), default=0)
        cells = []
        for column in columns:
            total, count = row.get(column, [0.0, 0])
            cells.append(f"{total / count:>10.3f}" if count else f"{'-':>10}")
            overall[column][0] += total
            overall[column][1] += count
        print(f"{layer_id:>6} {tokens:>8} " + " ".join(cells))
    print("-" * len(header))
    cells = []
    for column in columns:
        total, count = overall[column]
        cells.append(f"{total / count:>10.3f}" if count else f"{'-':>10}")
    print(f"{'ALL':>6} {'':>8} " + " ".join(cells))
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="the predict-log directory, or one .jsonl file")
    parser.add_argument("--ks", default="10,15,20", help="recall cut-offs (default 10,15,20)")
    args = parser.parse_args(argv)
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    records = load_records(args.path)
    if not records:
        print(f"no records under {args.path}")
        return 2
    print(f"records: {len(records)}")
    return analyze(records, ks)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
