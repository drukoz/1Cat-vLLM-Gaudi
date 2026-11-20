# SPDX-License-Identifier: Apache-2.0
"""Count and list the collectives of a Flash-Next step at TP<n> + EP<n> on gloo CPU ranks.

Runs ``tests/unit_tests/models/flash_next/collective_ranks.collective_rank`` (the 8-head tiny model, the
plugin's MoE path, the plugin's ``HpuCommunicator`` over gloo) on ``--tp`` ranks and prints, for one prompt
chunk, one decode step, one speculative verification step and the drafter's forwards of one step: every
collective in order, the call site that issued it, the module that was running, and its payload at the real
geometry (48 layers, hidden 2560, 16 x 160 PLE rows, the same 248,320-token vocabulary) for 1 row, for 32
rows and for the 32 x (1 + k) rows of a verification step at 32 streams.

    python tools/flash_next/collective_count.py --ckpt <tiny TP8 checkpoint dir> [--tp 8] [--k 3]
        [--env VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 ...] [--json out.json]

The checkpoint is the one ``tests/unit_tests/models/flash_next/cpu_model.build_tiny_checkpoint`` writes from
``tiny_tp8_config.json``. The tiny model has 4 layers; ``--real-layers`` (default 48) scales the collectives a
target layer issues (the drafter has one layer in both models), everything else is counted as recorded.

Bytes: "sent" is what one rank puts into a collective (its input tensor), "held" what every rank holds when
the collective returns (the same for an all-reduce, ``world`` times the input for an all-gather).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

REPO = Path(__file__).resolve().parents[2]
for path in (str(REPO), str(REPO / "spikes" / "cpu-tp")):
    if path not in sys.path:
        sys.path.insert(0, path)

REAL_HIDDEN = 2560
REAL_PLE_ROW = 160  # values of one n-gram table row (16 rows per token)
DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "int8": 1, "float8_e4m3fn": 1, "int32": 4, "int64": 8, "float16": 2}
STREAMS = 32
PER_LAYER_KINDS = ("mixer", "moe", "hc_gather")


def classify(entry: dict) -> str:
    """What a recorded collective carries.

    ``embedding`` (token rows), ``ple_table`` (n-gram table rows), ``embedding_ple`` (both in one all-reduce,
    ``VLLM_HPU_PLE_TABLE_RIDER``), ``mixer`` (GDN / attention output), ``moe``
    (MoE block output), ``hc_gather`` (a block output gathered for the fused combine), ``mtp_fc`` (the draft's
    entry linears), ``logits`` (full vocabulary shards), ``top`` (reduced logits), ``other``.
    """
    module, site, caller = entry["module"], entry["site"], entry["caller"]
    where = site + " " + caller
    if "_ple_table_rider" in where:
        return "embedding_ple"
    if "vocab_parallel.py" in where and "_gathered_logits" not in where:
        return "top"
    if entry["method"].startswith("all_gather") and ("logits_processor" in where or "_gathered_logits" in where):
        return "logits"
    if "ngram_embedding" in module or "ple" in module.split(".")[-1:]:
        return "ple_table"
    if module.endswith("embed_tokens"):
        return "embedding"
    if module.endswith((".fc_embedding", ".fc_hidden")):
        return "mtp_fc"
    if "hyper_connection" in module:
        return "hc_gather"
    if module.endswith((".out_proj", ".o_proj")):
        return "mixer"
    if ".mlp" in module:
        return "moe"
    return "other"


def real_columns(entry: dict, kind: str, hidden: int) -> int:
    """Elements per row at the real geometry (the row count is the first dimension)."""
    shape = entry["shape"]
    per_row = 1
    for size in shape[1:]:
        per_row *= size
    if kind in ("logits", "top"):
        return per_row  # the vocabulary is the real one
    if kind == "ple_table" and len(shape) == 3:
        return shape[1] * REAL_PLE_ROW
    return per_row * REAL_HIDDEN // hidden  # every other payload is a multiple of the hidden size


def summarize(results: list[dict], real_layers: int, k: int) -> dict:
    """Per phase: the collectives of rank 0 with their payloads, the per-rank agreement and the real-model count."""
    layout = results[0]["layout"]
    hidden, layers, world = layout["hidden"], layout["layers"], layout["tp"]
    phases: dict[str, dict] = {}
    names: list[str] = []
    for entry in results[0]["entries"]:
        if entry["phase"] not in names:
            names.append(entry["phase"])
    for phase in names:
        per_rank = [[e for e in r["entries"] if e["phase"] == phase] for r in results]
        keys = [[(e["method"], e["shape"], e["dtype"], e["site"]) for e in rank] for rank in per_rank]
        same_order = all(rank == keys[0] for rank in keys)
        rows_out = []
        for index, entry in enumerate(per_rank[0]):
            kind = classify(entry)
            digests = [rank[index]["digest"] for rank in per_rank] if same_order else []
            sent = real_columns(entry, kind, hidden) * DTYPE_BYTES.get(entry["dtype"], 2)
            gather = entry["method"].startswith("all_gather")
            rows_out.append({
                "index": index,
                "method": entry["method"],
                "kind": kind,
                "module": entry["module"],
                "site": entry["site"],
                "caller": entry["caller"],
                "tiny_shape": list(entry["shape"]),
                "dtype": entry["dtype"],
                "ranks_with_payload": sum(1 for rank in per_rank if rank[index]["nonzero"]) if same_order else -1,
                "replicated": bool(digests) and len(set(digests)) == 1,
                "sent_bytes_per_row": sent,
                "held_bytes_per_row": sent * (world if gather else 1),
                "real_multiplier": real_layers // layers if kind in PER_LAYER_KINDS and phase != "draft" else 1,
            })
        counts: dict[str, int] = {}
        real_counts: dict[str, int] = {}
        for row in rows_out:
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
            real_counts[row["kind"]] = real_counts.get(row["kind"], 0) + row["real_multiplier"]
        phases[phase] = {
            "same_order_on_every_rank": same_order,
            "tiny_count": len(rows_out),
            "tiny_counts": counts,
            "real_counts": real_counts,
            "real_count": sum(real_counts.values()),
            "collectives": rows_out,
        }
    return {"layout": layout, "world": world, "k": k, "real_layers": real_layers, "phases": phases}


def step_bytes(phase: dict, rows: int, field: str = "held_bytes_per_row") -> dict:
    """Bytes of the collectives of a real-model phase run with ``rows`` rows, by kind and in total."""
    by_kind: dict[str, int] = {}
    for row in phase["collectives"]:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + row[field] * row["real_multiplier"] * rows
    by_kind["total"] = sum(by_kind.values())
    return by_kind


def fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def render(summary: dict) -> str:
    k, world = summary["k"], summary["world"]
    verify_rows = STREAMS * (1 + k)
    lines = [f"TP{world} + EP{summary['layout']['ep_size']}, k = {k}; real model: {summary['real_layers']} layers, "
             f"hidden {REAL_HIDDEN}", f"layout of rank 0: {summary['layout']}", ""]
    for name, phase in summary["phases"].items():
        lines.append(f"== {name}: {phase['tiny_count']} collectives on the tiny model, {phase['real_count']} on the "
                     f"real model {phase['real_counts']}; same order on every rank: "
                     f"{phase['same_order_on_every_rank']}")
        lines.append("  #  method           kind       tiny shape        dtype     ranks!=0 repl  sent/row   held/row   "
                     "module <- site")
        for row in phase["collectives"]:
            lines.append(f"  {row['index']:<2} {row['method']:<16} {row['kind']:<10} "
                         f"{str(tuple(row['tiny_shape'])):<17} {row['dtype']:<9} "
                         f"{row['ranks_with_payload']:<8} {'yes' if row['replicated'] else 'no':<5}"
                         f"{fmt_bytes(row['sent_bytes_per_row']):<10} {fmt_bytes(row['held_bytes_per_row']):<10} "
                         f"{row['module']} <- {row['site']}")
        widths = [(1, "1 row"), (STREAMS, f"{STREAMS} rows")]
        if name == "verify":
            widths = [(1 + k, f"1 x (1 + {k}) rows"), (verify_rows, f"{STREAMS} x (1 + {k}) = {verify_rows} rows")]
        for rows, label in widths:
            held = step_bytes(phase, rows)
            lines.append(f"  real model, {label}: held per rank " + ", ".join(f"{kind} {fmt_bytes(v)}"
                                                                              for kind, v in held.items()))
        lines.append("")
    return "\n".join(lines)


def run(ckpt: str, tp: int, k: int, env: dict, phases=None, decode_rows: int = 1, verify_requests: int = 1,
        timeout_s: float = 900.0, **extra) -> list[dict]:
    import harness
    from tests.unit_tests.models.flash_next.collective_ranks import DEFAULT_PHASES, collective_rank

    opts = {"env": env, "k": k, "phases": tuple(phases or DEFAULT_PHASES), "decode_rows": decode_rows,
            "verify_requests": verify_requests, **extra}
    return harness.run_ranks(tp, collective_rank, ckpt, tp, opts, timeout_s=timeout_s)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--real-layers", type=int, default=48)
    parser.add_argument("--decode-rows", type=int, default=1)
    parser.add_argument("--verify-requests", type=int, default=1)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    env = dict(item.split("=", 1) for item in args.env)
    results = run(args.ckpt, args.tp, args.k, env, decode_rows=args.decode_rows, verify_requests=args.verify_requests)
    summary = summarize(results, args.real_layers, args.k)
    summary["env"] = env
    print(render(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
