# SPDX-License-Identifier: Apache-2.0
"""Preflight check: does every block-FP8 weight of the Flash-Next checkpoint split into whole blocks per rank?

In ``Qwen/Qwen3.8-Flash-Next-FP8`` only the routed-expert weights are block-FP8 (``weight_block_size``
[128, 128]); attention, GDN, shared experts, the MoE gate, embeddings, PLE, the vision tower and the MTP
head's non-expert layers are BF16 in ``modules_to_not_convert``, so only the expert splits need checking.

The rule is vLLM's ``validate_fp8_block_shape`` (at the pinned vLLM commit,
``vllm/model_executor/layers/quantization/utils/fp8_utils.py``), quoted:

* "Required by row parallel": when ``tp_size > 1`` and
  ``input_size // input_size_per_partition == tp_size`` and
  ``input_size_per_partition % block_k != 0``, raise ``ValueError("Weight input_size_per_partition = ...
  is not divisible by weight quantization block_k = ...")``.
* "Required by column parallel or enabling merged weights": with
  ``is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size`` or
  ``is_merged_gemm = len(output_partition_sizes) > 1``, check every ``output_partition_size`` and raise
  ``ValueError("Weight output_partition_size = ... is not divisible by weight quantization block_n = ...")``
  when ``output_partition_size % block_n != 0``; for a merged GEMM that is not TP-split the last partition
  is exempt ("we allow the last matrix to not be a multiple of block size").

How FusedMoE splits the experts: without ``--enable-expert-parallel`` every expert's intermediate dimension
is split by TP -- ``w13`` (gate+up, column-parallel) as two 640-wide halves of ``640 / tp`` each, ``w2``
(down, row-parallel) on an input partition of ``640 / tp``; the hidden dimension (2560 = 20 blocks) is
never split for experts. With expert parallelism each rank holds ``num_experts / tp`` whole experts, the
intermediate is not split (per rank 640 = 5 blocks), and ``num_experts % tp`` must be 0.

Usage: ``python tools/flash_next/check_fp8_layout.py --config CONFIG_JSON --tp N [--ep]``.
Exit 0 when every row is OK, 1 when any row fails, 2 for bad input. Standard library only, Python 3.10.
"""

from __future__ import annotations

import argparse
import json
import sys


def _text_fields(config: dict) -> dict:
    """The layer fields live under ``text_config`` when present, else at the top level."""
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _block_size(config: dict) -> tuple[int, int]:
    """Read ``quantization_config.weight_block_size``; raise ValueError on anything that is not block FP8."""
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError("config has no quantization_config")
    method = quant.get("quant_method")
    if method != "fp8":
        raise ValueError(f"quant_method is {method!r}, expected 'fp8'")
    block = quant.get("weight_block_size")
    if not isinstance(block, (list, tuple)) or len(block) != 2:
        raise ValueError("quantization_config has no weight_block_size [block_n, block_k]")
    return int(block[0]), int(block[1])


def check_layout(config: dict, tp: int, ep: bool) -> list[dict]:
    """Check the block-FP8 expert splits of the checkpoint config for one TP/EP setting.

    Returns one row per weight, ``{"module", "dim", "size", "per_rank", "block", "ok"}``, in this order:

    * ``experts.w13`` (dim ``intermediate``): gate+up, column-parallel, size 2 x 640 = 1280; split as two
      640-wide halves, so ``per_rank`` of one half is ``640 / tp`` without EP and 640 with EP.
    * ``experts.w2`` (dim ``intermediate``): down, row-parallel, size 640, input partition ``640 / tp``
      without EP and 640 with EP.
    * ``experts.hidden`` (dim ``hidden``): the hidden dimension, 2560, never split for experts.
    * ``experts.count`` (dim ``experts``, block 1), only with EP: ``num_experts`` split whole, so
      ``per_rank = num_experts // tp`` must be exact.

    ``ok`` is ``per_rank % block == 0``, and for a split dimension the division by ``tp`` must be exact.
    Raises ValueError/KeyError/TypeError when the config is not a block-FP8 Flash-Next config.
    """
    block_n, block_k = _block_size(config)
    text = _text_fields(config)
    hidden = int(text["hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    num_experts = int(text["num_experts"])

    per_rank_intermediate = intermediate if ep else intermediate // tp
    split_exact = ep or intermediate % tp == 0

    rows = [
        {
            "module": "experts.w13",
            "dim": "intermediate",
            "size": 2 * intermediate,
            "per_rank": per_rank_intermediate,
            "block": block_n,
            "ok": split_exact and per_rank_intermediate % block_n == 0,
        },
        {
            "module": "experts.w2",
            "dim": "intermediate",
            "size": intermediate,
            "per_rank": per_rank_intermediate,
            "block": block_k,
            "ok": split_exact and per_rank_intermediate % block_k == 0,
        },
        {
            "module": "experts.hidden",
            "dim": "hidden",
            "size": hidden,
            "per_rank": hidden,
            "block": block_n,
            "ok": hidden % block_n == 0,
        },
    ]
    if ep:
        rows.append({
            "module": "experts.count",
            "dim": "experts",
            "size": num_experts,
            "per_rank": num_experts // tp,
            "block": 1,
            "ok": num_experts % tp == 0,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_fp8_layout.py",
        description="Check that the block-FP8 expert weights of the Flash-Next checkpoint split into whole "
        "quantization blocks on every rank for --tp and --ep.",
    )
    parser.add_argument("--config", required=True, help="path to the checkpoint config.json")
    parser.add_argument("--tp", type=int, required=True, help="tensor-parallel size")
    parser.add_argument("--ep", action="store_true", help="expert parallelism is enabled")
    args = parser.parse_args(argv)

    if args.tp < 1:
        print(f"error: --tp must be >= 1, got {args.tp}", file=sys.stderr)
        return 2
    try:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read config {args.config}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(config, dict):
        print(f"error: config {args.config} is not a JSON object", file=sys.stderr)
        return 2
    try:
        rows = check_layout(config, args.tp, args.ep)
    except KeyError as exc:
        print(f"error: config {args.config} is missing field {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: config {args.config}: {exc}", file=sys.stderr)
        return 2

    failed = 0
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        failed += 0 if row["ok"] else 1
        print(f"{row['module']} {row['dim']} {row['size']} {row['per_rank']} {row['block']} {status}")
    ep_flag = 1 if args.ep else 0
    if failed:
        print(f"fp8 layout tp={args.tp} ep={ep_flag}: FAIL ({failed} rows)")
        return 1
    print(f"fp8 layout tp={args.tp} ep={ep_flag}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
