#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""kv_scales.py OUT.json STATS_FOLDER [--margin M] [--fp8-max F]: the per-layer FP8 K/V scale file of
VLLM_HPU_QSA_KV_FP8=2 from a calibration server's VLLM_HPU_QSA_KV_STATS files (one per tensor-parallel rank):
per QSA layer the largest K and V amax over the ranks, and the fraction of elements below 2**-6 for the record.
Stdlib only."""
import argparse
import glob
import json
import os
import sys


def merge(folder: str) -> dict:
    layers = {}
    for path in sorted(glob.glob(os.path.join(folder, "kv_amax_rank*.json"))):
        for layer, rec in (json.load(open(path)).get("layers") or {}).items():
            cur = layers.setdefault(layer, {"k_amax": 0.0, "v_amax": 0.0, "k_small": 0.0, "v_small": 0.0, "ranks": 0})
            cur["k_amax"] = max(cur["k_amax"], float(rec.get("k_amax", 0.0)))
            cur["v_amax"] = max(cur["v_amax"], float(rec.get("v_amax", 0.0)))
            cur["k_small"] = max(cur["k_small"], float(rec.get("k_small", 0.0)))
            cur["v_small"] = max(cur["v_small"], float(rec.get("v_small", 0.0)))
            cur["ranks"] += 1
    return layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("folder")
    ap.add_argument("--margin", type=float, default=1.0, help="scale = amax * margin / fp8_max (above 1: headroom)")
    ap.add_argument("--fp8-max", type=float, default=240.0)
    args = ap.parse_args()
    layers = merge(args.folder)
    if not layers:
        print("kv_scales: no kv_amax_rank*.json in %s" % args.folder)
        return 2
    json.dump({"layers": layers, "fp8_max": args.fp8_max, "margin": args.margin, "source": args.folder},
              open(args.out, "w"), indent=1)
    for layer in sorted(layers, key=int):
        r = layers[layer]
        print("kv_scales: layer %3s K amax %.4g (below 2^-6 %.3f) V amax %.4g (below 2^-6 %.3f), %d rank(s)" % (
            layer, r["k_amax"], r["k_small"], r["v_amax"], r["v_small"], r["ranks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
