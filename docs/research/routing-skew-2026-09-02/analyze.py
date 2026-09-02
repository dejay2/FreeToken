"""Offline skew analysis over the per-workload decode_freq JSONs.

Usage: python analyze_routing.py <dir>
Prints a markdown-ready report; also writes analysis.json next to the inputs.
"""

import json
import math
import sys
from pathlib import Path

D = Path(sys.argv[1])
NAMES = ["code", "prose", "chat8k", "toolcall"]
KS = [42, 70, 94]
TOPSETS = [70, 140]

data = {}
for n in NAMES:
    p = D / f"{n}.json"
    if p.exists():
        data[n] = json.loads(p.read_text(encoding="utf-8"))
union = json.loads((D / "union.json").read_text(encoding="utf-8")) if (D / "union.json").exists() else None

names = list(data)
L = data[names[0]]["num_layers"]
E = data[names[0]]["num_experts"]


def freq(doc):
    return doc["decode_freq"]


def layer_stats(row):
    tot = sum(row)
    if tot == 0:
        return None
    order = sorted(range(len(row)), key=lambda i: -row[i])
    cum = 0
    n90 = 0
    for i in order:
        cum += row[i]
        n90 += 1
        if cum / tot >= 0.9:
            break
    ws = sum(1 for v in row if v > 0)
    ent = -sum((v / tot) * math.log(v / tot) for v in row if v > 0)
    return {
        "total": tot,
        "experts_for_90pct": n90,
        "working_set": ws,
        "norm_entropy": ent / math.log(len(row)),
        "order": order,
    }


per_workload = {}
for n in names:
    rows = freq(data[n])
    stats = [layer_stats(r) for r in rows]
    valid = [s for s in stats if s]
    per_workload[n] = {
        "stats": stats,
        "layers_with_traffic": len(valid),
        "total_counts": sum(s["total"] for s in valid),
        "e90_mean": sum(s["experts_for_90pct"] for s in valid) / len(valid),
        "e90_min": min(s["experts_for_90pct"] for s in valid),
        "e90_max": max(s["experts_for_90pct"] for s in valid),
        "ws_mean": sum(s["working_set"] for s in valid) / len(valid),
        "ws_max": max(s["working_set"] for s in valid),
        "ent_mean": sum(s["norm_entropy"] for s in valid) / len(valid),
        "ent_min": min(s["norm_entropy"] for s in valid),
        "ent_max": max(s["norm_entropy"] for s in valid),
    }

# pairwise Jaccard of per-layer top-K expert sets
jaccard = {}
for K in TOPSETS:
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            vals = []
            for L_i in range(L):
                sa, sb = per_workload[a]["stats"][L_i], per_workload[b]["stats"][L_i]
                if not sa or not sb:
                    continue
                A = set(sa["order"][:K])
                B = set(sb["order"][:K])
                vals.append(len(A & B) / len(A | B))
            if vals:
                pairs[f"{a}|{b}"] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
    jaccard[K] = pairs

# static hot set: pick top-K per layer from a reference, measure coverage on each workload
def hot_hit(ref_doc, K):
    ref = freq(ref_doc)
    out = {}
    for n in names:
        rows = freq(data[n])
        num, den = 0.0, 0.0
        per_layer = []
        for L_i in range(L):
            row = rows[L_i]
            tot = sum(row)
            if tot == 0:
                continue
            r = ref[L_i]
            hot = sorted(range(E), key=lambda i: -r[i])[:K]
            hit = sum(row[i] for i in hot)
            per_layer.append(hit / tot)
            num += hit
            den += tot
        out[n] = {
            "overall": num / den if den else 0.0,
            "layer_mean": sum(per_layer) / len(per_layer),
            "layer_min": min(per_layer),
        }
    return out


refs = {"union": union} if union else {}
refs["self"] = None  # each workload against its own top-K (the optimistic bound)

hot = {}
for K in KS:
    hot[K] = {}
    if union:
        hot[K]["ref=union"] = hot_hit(union, K)
    hot[K]["ref=self"] = {}
    for n in names:
        hot[K]["ref=self"][n] = hot_hit(data[n], K)[n]
    hot[K]["ref=code"] = hot_hit(data["code"], K) if "code" in data else {}

result = {
    "num_layers": L,
    "num_experts": E,
    "cache_size": data[names[0]].get("cache_size"),
    "slots_per_layer": (data[names[0]].get("cache_size") or 0) / L,
    "per_workload": {
        n: {k: v for k, v in per_workload[n].items() if k != "stats"} for n in names
    },
    "server_summary": {n: data[n].get("summary", {}) for n in names},
    "generation": {n: data[n].get("generation", {}).get("completion_tokens") for n in names},
    "jaccard": jaccard,
    "static_hot_set": hot,
    "per_layer_e90": {n: [s["experts_for_90pct"] if s else None for s in per_workload[n]["stats"]] for n in names},
    "per_layer_ws": {n: [s["working_set"] if s else None for s in per_workload[n]["stats"]] for n in names},
    "per_layer_entropy": {n: [round(s["norm_entropy"], 4) if s else None for s in per_workload[n]["stats"]] for n in names},
}
if union:
    us = [layer_stats(r) for r in freq(union)]
    uv = [s for s in us if s]
    result["union"] = {
        "total_counts": sum(s["total"] for s in uv),
        "e90_mean": sum(s["experts_for_90pct"] for s in uv) / len(uv),
        "e90_max": max(s["experts_for_90pct"] for s in uv),
        "ws_mean": sum(s["working_set"] for s in uv) / len(uv),
        "ws_max": max(s["working_set"] for s in uv),
        "ent_mean": sum(s["norm_entropy"] for s in uv) / len(uv),
    }

(D / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
