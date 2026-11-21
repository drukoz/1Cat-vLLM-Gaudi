#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe: one Flash-Next FP8 MoE block (one rank of TP8 + EP8) on the torch path against the MoE glue.

    PYTHONPATH=<plugin tree> python3 probe_moe_block.py --out DIR                    timing and parity
    PYTHONPATH=<plugin tree> GRAPH_VISUALIZATION=1 python3 probe_moe_block.py --out DIR --static VARIANT
    python3 -I probe_moe_block.py --parse DIR                                         node counts, offline

Variants (the block as vllm_gaudi/ops/hpu_fused_moe.py runs it, one rank: router 512 x 2560 and shared gate
1 x 2560 replicated, shared expert 80 wide, 64 local FP8 experts of 512, top 10, H 2560, I 640):
  torch   VLLM_HPU_MOE_GLUE=0: router GEMM, shared-gate GEMM, softmax + top-k + renormalize + casts, gate_up GEMM,
          SiLU * up, down GEMM, dynamic_quant (the plugin's), mixture_of_experts, sigmoid * shared + routed
  glue    VLLM_HPU_MOE_GLUE=1: merged gate GEMM [513], moe_route, int64 cast, gate_up GEMM, moe_silu_mul, down GEMM,
          moe_row_quant, mixture_of_experts, moe_combine
  merged  glue with the gate_up GEMM merged into the gate GEMM [673]: moe_silu_mul reads its block at column 513
          (only when the installed op takes first_column)
Per variant and row count (1, 32, 128): parity against torch (the same experts on rows without a tie at the
cut, max abs diff of the block output), and the device time per block: K blocks per lazy graph, each on its own
input, timed as issue / mark_step / synchronize. The static arm dumps the compiled post-graph of one block and
counts its physical nodes (the method of the fork's tools/audit_deepseek_v41_physical_nodes.py).
Last line: "VERDICT moe-block: ..." with the microseconds per block of each variant.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

E, K, H, I, LOCAL, D = 512, 10, 2560, 640, 64, 80
ROWS = (1, 32, 128)
VARIANTS = ("torch", "glue", "merged")


def say(text):
    print(f"PROBE moe-block {text}", flush=True)


def find_kernel_libraries():
    import flashinfer_gaudi

    folders = [Path(flashinfer_gaudi.__file__).resolve().parent / "lib"]
    folders += [Path(entry) / "flashinfer_gaudi" / "lib" for entry in sys.path if entry]
    for folder in folders:
        kernel = folder / "libhc_fused_gaudi2.so"
        ops = sorted(glob.glob(str(folder / "hc_fused_ops*.so")))
        if kernel.is_file() and ops:
            os.environ.setdefault("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel))
            os.environ.setdefault("HC_FUSED_GAUDI_NATIVE_LIBRARY", ops[0])
            gdn = folder / "libgdn_fused_gaudi2.so"
            if gdn.is_file():
                os.environ.setdefault("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(gdn))
            return ""
    return "libhc_fused_gaudi2.so and hc_fused_ops*.so not found in " + ", ".join(str(f) for f in folders[:3])


class Block:

    def __init__(self, torch, dev, seed=1234):
        g = torch.Generator().manual_seed(seed)
        self.torch, self.dev, self.g = torch, dev, g
        bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(bf16).to(dev)

        self.gate = randn(E, H, scale=H**-0.5 * 3)
        self.shared_gate = randn(1, H, scale=H**-0.5)
        self.gate_up = randn(2 * D, H, scale=H**-0.5)
        self.down = randn(H, D, scale=D**-0.5)
        self.merged = torch.cat([self.gate, self.shared_gate], dim=0).contiguous()
        self.merged2 = torch.cat([self.gate, self.shared_gate, self.gate_up], dim=0).contiguous()
        self.w12 = tuple((torch.randn(2 * I, H, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev)
                         for _ in range(LOCAL))
        self.w3 = tuple((torch.randn(H, I, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(LOCAL))
        self.s12 = tuple(torch.full((2 * I, ), 2e-4, dtype=torch.float32).to(dev) for _ in range(LOCAL))
        self.s3 = tuple(torch.full((H, ), 2e-4, dtype=torch.float32).to(dev) for _ in range(LOCAL))

    def x(self, rows):
        return (self.torch.randn(rows, H, generator=self.g) * 1.5).to(self.torch.bfloat16).to(self.dev)

    def vendor(self, x8, scale, ids, weights, dtype):
        return self.torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids.to(self.torch.int64),
                                                     router_weights=weights.to(dtype), w12=self.w12, w3=self.w3,
                                                     d_scale_hidden_states=scale, d_scale_w12=self.s12,
                                                     d_scale_w3=self.s3, permuted_weights=True, activation="silu",
                                                     experts_min=0, experts_max=LOCAL - 1)


def run_torch(blk, x, ref, plugin_ops):
    F = blk.torch.nn.functional

    class Layer:
        top_k = K

    router_logits = F.linear(x, blk.gate)
    shared = F.linear(ref.moe_silu_mul_torch(F.linear(x, blk.gate_up)), blk.down)
    expert_gate = F.linear(x, blk.shared_gate)
    topk_ids, topk_weights = ref.moe_route_torch(x, router_logits, Layer)
    x8, scale = plugin_ops.dynamic_quant(x)
    routed = blk.vendor(x8, scale, topk_ids, topk_weights, x.dtype)
    return ref.moe_combine_torch(shared, expert_gate, routed), topk_ids


def run_glue(blk, x, ops, merged=False):
    F = blk.torch.nn.functional
    if merged:
        gates = F.linear(x, blk.merged2)
        act = ops.moe_silu_mul(gates, E + 1)
    else:
        gates = F.linear(x, blk.merged)
        act = ops.moe_silu_mul(F.linear(x, blk.gate_up))
    ids, weights = ops.moe_route(gates, E, K)
    shared = F.linear(act, blk.down)
    x8, scale = ops.moe_row_quant(x, 240.0)
    routed = blk.vendor(x8, scale, ids, weights, x.dtype)
    return ops.moe_combine(shared, gates, routed, E), ids


DEVICE = os.environ.get("PROBE_DEVICE", "hpu")


class _CpuCore:
    """PROBE_DEVICE=cpu: the probe's own logic on a CPU with the tests' stand-ins (times mean nothing)."""

    @staticmethod
    def mark_step():
        pass


def sync(torch):
    if DEVICE == "hpu":
        torch.hpu.synchronize()


def prepare():
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)
    os.environ.setdefault("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")
    for flag in ("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "VLLM_HPU_DYNAMIC_QUANT_CGUID",
                 "VLLM_HPU_CGUID_DYNAMIC_QUANT"):
        os.environ[flag] = "0"
    import torch

    if DEVICE == "cpu":
        from tests.unit_tests.hc_fused import moe_ops as ops
        from tests.unit_tests.models.flash_next import moe_block

        moe_block.install_vendor_ops(fp8_max=240.0)
        import vllm_gaudi.extension.ops as plugin_ops

        plugin_ops.FP8_MAX = 240.0  # a dry run stands for a Gaudi2 card: its fp8 range (CPU torch has 448)
        from tests.unit_tests.hc_fused import moe_reference as ref

        merged_ok = "first_column" in str(torch.ops.flashinfer_gaudi.moe_silu_mul.default._schema)
        return torch, _CpuCore, ops, merged_ok, "", ref, plugin_ops
    import habana_frameworks.torch.core as htcore

    problem = find_kernel_libraries()
    ops = None
    merged_ok = False
    if not problem:
        from flashinfer_gaudi import _native

        if _native.load_moe_glue_extension():
            from flashinfer_gaudi import moe_glue as ops

            schema = str(torch.ops.flashinfer_gaudi.moe_silu_mul.default._schema)
            merged_ok = "first_column" in schema
        else:
            problem = "; ".join(_native.moe_glue_extension_errors()) or "load_moe_glue_extension returned False"
    from tests.unit_tests.hc_fused import moe_reference as ref
    import vllm_gaudi.extension.ops as plugin_ops

    return torch, htcore, ops, merged_ok, problem, ref, plugin_ops


def variants_for(ops, merged_ok):
    out = ["torch"]
    if ops is not None:
        out.append("glue")
        if merged_ok:
            out.append("merged")
    return out


def call(name, blk, x, ops, ref, plugin_ops):
    if name == "torch":
        return run_torch(blk, x, ref, plugin_ops)
    return run_glue(blk, x, ops, merged=(name == "merged"))


def timing(out, iters, blocks_per_graph):
    torch, htcore, ops, merged_ok, problem, ref, plugin_ops = prepare()
    say("kernels: " + ("loaded" if ops is not None else f"NOT loaded ({problem})") +
        f"; merged gate_up (moe_silu_mul first_column): {merged_ok}")
    blk = Block(torch, DEVICE)
    htcore.mark_step()
    sync(torch)
    names = variants_for(ops, merged_ok)
    results = {"variants": names, "kernel_problem": problem}
    for rows in ROWS:
        # parity on one input
        x = blk.x(rows)
        htcore.mark_step()
        outs = {}
        for name in names:
            y, ids = call(name, blk, x, ops, ref, plugin_ops)
            htcore.mark_step()
            outs[name] = (y.cpu(), ids.cpu().long())
        base, base_ids = outs["torch"]
        logits = torch.nn.functional.linear(x, blk.gate).float().cpu()
        ordered = torch.sort(logits, dim=1, descending=True).values
        clear = ordered[:, K - 1] != ordered[:, K]
        for name in names[1:]:
            y, ids = outs[name]
            same_sets = (torch.sort(ids, dim=1).values == torch.sort(base_ids, dim=1).values).all(dim=1)
            same_rows = same_sets & clear
            diff = (y.float() - base.float()).abs()
            bit_rows = (y.view(torch.int16) == base.view(torch.int16)).all(dim=1)
            results[f"parity_rows{rows}_{name}"] = dict(
                same_sets_clear=int(same_sets[clear].sum()), clear_rows=int(clear.sum()),
                bit_identical_rows=int(bit_rows.sum()), max_abs_diff=float(diff.max()),
                max_abs_diff_same_set_rows=float(diff[same_rows].max()) if bool(same_rows.any()) else None)
            say(f"parity rows {rows:4d} {name:6s}: same experts on {int(same_sets[clear].sum())} of {int(clear.sum())} "
                f"rows without a tie at the cut; bit-identical rows {int(bit_rows.sum())} of {rows}; max abs diff "
                f"{float(diff.max()):.3e} (rows of the same experts: "
                f"{results[f'parity_rows{rows}_{name}']['max_abs_diff_same_set_rows']})")
        # timing: blocks_per_graph blocks, each on its own input
        inputs = [blk.x(rows) for _ in range(blocks_per_graph)]
        htcore.mark_step()
        sync(torch)
        for name in names:
            samples = []
            for it in range(iters + 2):
                t0 = time.perf_counter()
                ys = [call(name, blk, xi, ops, ref, plugin_ops)[0] for xi in inputs]
                t1 = time.perf_counter()
                htcore.mark_step()
                t2 = time.perf_counter()
                sync(torch)
                t3 = time.perf_counter()
                if it >= 2:
                    samples.append((t1 - t0, t2 - t1, t3 - t2))
                del ys
            samples.sort(key=lambda s: s[1] + s[2])
            issue, mark, wait = samples[len(samples) // 2]
            per_block = (mark + wait) / blocks_per_graph * 1e6
            results[f"time_rows{rows}_{name}"] = dict(device_bound_us_per_block=per_block,
                                                      issue_us_per_block=issue / blocks_per_graph * 1e6,
                                                      mark_ms=mark * 1e3, wait_ms=wait * 1e3,
                                                      blocks_per_graph=blocks_per_graph)
            say(f"time rows {rows:4d} {name:6s}: {per_block:8.1f} us per block on the device (bound), host issue "
                f"{issue / blocks_per_graph * 1e6:7.1f} us per block (K={blocks_per_graph})")
    (out / "block.json").write_text(json.dumps(results, indent=1) + "\n")
    parts = []
    for rows in ROWS:
        cells = [f"{name} {results[f'time_rows{rows}_{name}']['device_bound_us_per_block']:.1f}" for name in names]
        parts.append(f"{rows} rows: " + ", ".join(cells) + " us")
    print("VERDICT moe-block: " + "; ".join(parts) + " (device-bound time per block; x48 layers per step)", flush=True)
    return results


def static(out, variant, rows):
    graphs = out / "graphs" / variant
    graphs.mkdir(parents=True, exist_ok=True)
    os.environ["PT_HPU_GRAPH_DUMP_PREFIX"] = str(graphs)
    torch, htcore, ops, merged_ok, problem, ref, plugin_ops = prepare()
    if variant != "torch" and (ops is None or (variant == "merged" and not merged_ok)):
        say(f"static {variant}: not available ({problem or 'no first_column'})")
        return
    blk = Block(torch, DEVICE)
    x = blk.x(rows)
    htcore.mark_step()
    sync(torch)
    marker = out / "graphs" / f"{variant}.before"
    marker.write_text("\n".join(sorted(str(p) for p in graphs.rglob("*"))))
    y, _ = call(variant, blk, x, ops, ref, plugin_ops)
    htcore.mark_step()
    sync(torch)
    say(f"static {variant}: block of {rows} rows run, output {tuple(y.shape)}")


LOGICAL = {"Placeholder", "OutputTensor", "Reshape", "StaticReshape", "Slice", "Split", "Concatenate", "Squeeze",
           "ExpandDims", "Identity", "Flatten", "Broadcast", "LogicalTranspose", "LogicalBroadcast",
           "ReinterpretCast", "TransposedShape", "Reduction", "StridedView"}


def parse(out):
    result = {}
    for variant_dir in sorted(p for p in (out / "graphs").iterdir() if p.is_dir()):
        before = set()
        marker = out / "graphs" / f"{variant_dir.name}.before"
        if marker.is_file():
            before = set(marker.read_text().split("\n"))
        graphs = []
        for path in sorted(variant_dir.rglob("*PostGraph-symbol.pbtxt")):
            raw = path.read_text(errors="replace")
            ops = {}
            for block in re.split(r"(?m)^node \{", raw)[1:]:
                op = re.search(r'\n  op: "([^"]+)"', block)
                if op and op.group(1) not in LOGICAL:
                    ops[op.group(1)] = ops.get(op.group(1), 0) + 1
            graphs.append(dict(graph=path.name, after_setup=str(path) not in before, physical_nodes=sum(ops.values()),
                               ops=dict(sorted(ops.items(), key=lambda kv: -kv[1]))))
        block_graphs = [g for g in graphs if g["after_setup"] and any("moe" in op.lower() or "mixture" in op.lower()
                                                                     for op in g["ops"])]
        result[variant_dir.name] = dict(graphs=graphs, block_graphs=block_graphs)
        for g in block_graphs or graphs:
            say(f"static {variant_dir.name}: {g['graph']}: {g['physical_nodes']} physical nodes "
                f"({'the block' if g in block_graphs else 'not identified as the block'}): {json.dumps(g['ops'])}")
        if not graphs:
            say(f"static {variant_dir.name}: no *PostGraph-symbol.pbtxt (files: "
                f"{sorted(str(p.name) for p in variant_dir.rglob('*'))[:12]})")
    (out / "nodes.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("moe_block"))
    ap.add_argument("--static", choices=VARIANTS)
    ap.add_argument("--static-rows", type=int, default=1)
    ap.add_argument("--parse", type=Path)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=16)
    a = ap.parse_args()
    if a.parse:
        parse(a.parse)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    if a.static:
        static(a.out, a.static, a.static_rows)
        return 0
    timing(a.out, a.iters, a.blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
