#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Card probe: why did 0 of 16 rows of route_real_r16 select "the card's torch.topk set"?

    python probe_moe_route.py        (inside the node's container, ONE card, about a minute)
    farm job kernel-fixes moe-route-probe any:1 15m -- bash tools/flash_next/kernel_probes/with_kernels.sh \
        python tools/flash_next/kernel_probes/probe_moe_route.py

The open card failure of the MoE glue (Gaudi2, 2026-11-20, three runs, both accumulation modes):
test_hpu_op_matches_the_torch_path_on_hpu[route_real_r16] printed

    MOEHPU route_real_r16 ids: 0 of 16 rows select the card's torch.topk set (15 rows do not tie at the cut)

while the kernel alone returned exactly its rule for the same input in the same runs.  What was killed on a
CPU before this probe (tests/unit_tests/hc_fused/test_moe_route_card_models.py):

  arithmetic   on the 15 rows the 10th and 11th logits are 0.0156 to 1.5 apart; a softmax would need a
               relative error of 0.8 % to move one row and of 64 % to move all 15 (bf16 and fp16 softmax: none);
  tie, order   the comparison sorts the ids, and one row (row 0, experts 132 and 450) ties at the cut.

What is left, and what this probe separates, each variant in a graph of its own unless it says "shared":

  H0 production  softmax + top-k of an own logits tensor with int64 ids, as a block runs it, is not CPU torch's
                 (then: is it the card's softmax, or the card's top-k of host-made probabilities?);
  H3 int32       narrowing top-k's int64 ids to int32 on the device (the test's reference does, production not);
  H4 view        routing on gates[:, :experts], a view of the [T, experts + 1] tensor the kernel reads;
  H5 shared      the reference and the kernel in one graph: which of the two sides then changes;
  H6 none        nothing of the above reproduces the line.

The kernel and the production routing are compared directly as well, because that is what decides whether
VLLM_HPU_MOE_GLUE=1 routes like VLLM_HPU_MOE_GLUE=0 on a card.

Read the last line, "VERDICT route: ...".  Everything above it is detail; the ids of every variant are also
written to PROBE_OUT (default probe_moe_route.json in the current folder) so that nothing needs a second run.

Self-contained: torch and habana_frameworks; flashinfer_gaudi (the installed plugin) for the kernel.
PROBE_DEVICE=cpu runs the probe's own logic on a CPU without a kernel (selftest_probe_moe_route.py injects
every fault there and requires the verdict to name it).
"""

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PT_HPU_LAZY_MODE", "1")  # the mode of the servers and of the card test
os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")  # custom ops race with the bridge's accumulation thread
os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)  # a recipe cache blocks the kernel library registration
os.environ.setdefault("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")  # as the card test ran

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEVICE = os.environ.get("PROBE_DEVICE", "hpu")
OUT = os.environ.get("PROBE_OUT") or os.path.join(os.environ.get("FARM_OUT", "."), "probe_moe_route.json")
BF16 = torch.bfloat16
STARTED = time.monotonic()
# sha256 of the bf16 bytes of the input as the repository's tests/unit_tests/hc_fused/moe_case.py builds it
INPUT_SHA = {"route_real_r16": "445e6e9121c78acd"}
RAW: dict = {}  # geometry -> variant -> ids (lists), written to OUT
SHARED = False  # True while a variant that shares a graph with the kernel is being built (self-test faults)


def say(text: str) -> None:
    print(f"PROBE route [{time.monotonic() - STARTED:5.1f}s] {text}", flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_gates(rows: int, experts: int, columns: int, seed: int) -> torch.Tensor:
    """moe_case.make_inputs for a route case of the flavor "random": bf16 [rows, columns]."""
    g = torch.Generator().manual_seed(seed)
    gates = torch.randn(rows, columns, generator=g) * 3.0
    scale = torch.exp(0.7 * torch.randn(rows, 1, generator=g))
    gates[:, :experts] = torch.randn(rows, experts, generator=g) * 2.0 * scale
    return gates.to(BF16)


def make_input(name: str) -> torch.Tensor:
    assert name == "route_real_r16"
    return make_gates(16, 512, 513, 6)


def sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The device side, one function per construct (the self-test replaces them to inject a fault)
# ---------------------------------------------------------------------------
def dev_view(gates_device: torch.Tensor, experts: int) -> torch.Tensor:
    """The slice the card test's reference routes on (moe_reference.moe_route_reference)."""
    return gates_device[:, :experts]


def dev_softmax(router_logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(router_logits, dim=1, dtype=torch.float32)


def dev_topk(probabilities: torch.Tensor, top_k: int):
    return torch.topk(probabilities, top_k, dim=-1)


def dev_narrow(ids: torch.Tensor) -> torch.Tensor:
    return ids.to(torch.int32)


def torch_route(x, router_logits, top_k):
    """The lines of HPUUnquantizedFusedMoEMethod.apply_monolithic (vllm_gaudi/ops/hpu_fused_moe.py)."""
    topk_weights = dev_softmax(router_logits)
    topk_weights, topk_ids = dev_topk(topk_weights, top_k)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(x.dtype)
    return topk_ids, topk_weights


def sets(ids: torch.Tensor) -> torch.Tensor:
    return torch.sort(ids.to(torch.int64), dim=1).values


def same_rows(ids: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    if tuple(ids.shape) != tuple(want.shape):
        return torch.zeros(want.shape[0], dtype=torch.bool)
    return (sets(ids) == sets(want)).all(dim=1)


class Geometry:

    def __init__(self, name, rows, experts, top_k, columns, seed):
        self.name, self.rows, self.experts, self.top_k, self.columns = name, rows, experts, top_k, columns
        self.gates = make_gates(rows, experts, columns, seed)
        self.logits = self.gates[:, :experts].contiguous()
        self.p = F.softmax(self.logits, dim=1, dtype=torch.float32)
        self.cpu_ids, self.cpu_weights = self._cpu_route()
        ordered = torch.sort(self.p, dim=1, descending=True).values
        self.clear = ordered[:, top_k - 1] != ordered[:, top_k] if experts > top_k else torch.ones(rows, dtype=torch.bool)
        self.rule_ids = torch.sort(self.logits.float(), dim=1, descending=True, stable=True).indices[:, :top_k]
        T, E, K = rows, experts, top_k
        flat = self.gates.reshape(-1)
        # ways a routing can go wrong, as CPU computations
        self.models = {
            "the view read as a contiguous [T, E] block of gates":
            torch.topk(F.softmax(flat[:T * E].reshape(T, E), dim=1, dtype=torch.float32), K).indices,
            "softmax over all columns of gates (the view ignored)":
            torch.topk(F.softmax(self.gates, dim=1, dtype=torch.float32), K).indices,
            "softmax over the tokens (dim 0)":
            torch.topk(F.softmax(self.logits, dim=0, dtype=torch.float32), K).indices,
            "the smallest probabilities":
            torch.topk(self.p, K, largest=False).indices,
            "top-k of the bf16 logits without the softmax":
            torch.topk(self.logits, K).indices,
            "every second id of the right table (an 8-byte read of 4-byte ids)":
            torch.cat([self.cpu_ids.reshape(-1)[::2], torch.full((T * K - (T * K + 1) // 2, ), -1)]).reshape(T, K),
            "the right ids stored as 8-byte values and read as 4-byte ones":
            torch.stack([self.cpu_ids.reshape(-1), torch.zeros(T * K, dtype=torch.int64)], dim=1).reshape(-1)[:T * K]
            .reshape(T, K),
            "the right table read in [K, T] order":
            self.cpu_ids.t().reshape(T, K),
        }

    def _cpu_route(self):
        """CPU torch's routing of the logits, with the plain functions (never the replaceable device side)."""
        weights, ids = torch.topk(self.p, self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(BF16)
        return ids, weights

    def describe(self, label: str, ids: torch.Tensor) -> tuple[bool, str]:
        """(every clear row has CPU torch's set, one line)."""
        ids = ids.to(torch.int64)
        cpu = same_rows(ids, self.cpu_ids)
        rule = same_rows(ids, self.rule_ids)
        clear_total = int(self.clear.sum())
        ok = bool(cpu[self.clear].all())
        line = (f"{self.name} | {label}: {int(cpu.sum())} of {self.rows} rows have CPU torch.topk's set "
                f"({int(cpu[self.clear].sum())} of the {clear_total} rows that do not tie at the cut), "
                f"{int(rule.sum())} of {self.rows} the kernel rule's set")
        if not ok:
            in_range = bool(((ids >= 0) & (ids < self.experts)).all())
            fits = [f"{name}: {int(same_rows(ids, model).sum())} of {self.rows} rows"
                    for name, model in self.models.items() if int(same_rows(ids, model).sum()) > 0]
            exact = [name for name, model in self.models.items()
                     if tuple(model.shape) == tuple(ids.shape) and torch.equal(model.to(torch.int64), ids)]
            line += (f"\n      ids in [0, {self.experts}): {in_range}; shape {tuple(ids.shape)}; "
                     f"fits: {'; '.join(fits) if fits else 'none of the models'}"
                     f"{'; exactly: ' + exact[0] if exact else ''}"
                     f"\n      got  row 0 {ids[0].tolist()}  row 1 {ids[1].tolist()}"
                     f"\n      want row 0 {self.cpu_ids[0].tolist()}  row 1 {self.cpu_ids[1].tolist()}")
        return ok, line


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------
def find_kernel_libraries() -> str:
    """Point the loader at the installed libraries when the package on the path has none; '' or the problem."""
    try:
        import flashinfer_gaudi
    except Exception as exc:  # noqa: BLE001
        return f"flashinfer_gaudi does not import ({type(exc).__name__}: {exc})"
    import glob
    from pathlib import Path

    folders = [Path(flashinfer_gaudi.__file__).resolve().parent / "lib"]
    folders += [Path(entry) / "flashinfer_gaudi" / "lib" for entry in sys.path if entry]
    for folder in folders:
        kernel = folder / "libhc_fused_gaudi2.so"
        ops = sorted(glob.glob(str(folder / "hc_fused_ops*.so")))
        if kernel.is_file() and ops:
            os.environ.setdefault("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel))
            os.environ.setdefault("HC_FUSED_GAUDI_NATIVE_LIBRARY", ops[0])
            return ""
    if os.environ.get("HC_FUSED_GAUDI_KERNEL_LIBRARY") and os.environ.get("HC_FUSED_GAUDI_NATIVE_LIBRARY"):
        return ""
    return "libhc_fused_gaudi2.so and hc_fused_ops*.so were not found in " + ", ".join(str(f) for f in folders[:3])


def load_kernel():
    """(the moe_route op or None, why not).  Must run before the first op on the device."""
    if DEVICE != "hpu":
        return None, "PROBE_DEVICE is not hpu"
    problem = find_kernel_libraries()
    if problem:
        return None, problem
    try:
        from flashinfer_gaudi import _native

        if not _native.load_moe_glue_extension():
            return None, "; ".join(_native.moe_glue_extension_errors()) or "load_moe_glue_extension returned False"
        return torch.ops.flashinfer_gaudi.moe_route, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def run(geometry: Geometry, kernel) -> dict:
    """{variant: True (clear rows right) | False | None (did not run)}; prints one line per variant."""
    global SHARED
    g, K, E = geometry, geometry.top_k, geometry.experts
    results: dict = {}
    raw = RAW.setdefault(g.name, {})

    def variant(key, label, compute):
        try:
            ids = compute()
            raw[key] = ids.to(torch.int64).tolist()
            ok, line = g.describe(label, ids)
        except Exception as exc:  # noqa: BLE001
            ok, line = None, f"{g.name} | {label}: raised {type(exc).__name__}: {str(exc)[:300]}"
            raw[key] = f"raised {type(exc).__name__}: {str(exc)[:300]}"
        results[key] = ok
        say(line)

    def own():
        return g.logits.clone().to(DEVICE)

    def view_reference(gates_device, narrow):
        """The test's reference: moe_reference.moe_route_reference(gates, experts, top_k)."""
        ids, weights = torch_route(gates_device, dev_view(gates_device, E), K)
        return (dev_narrow(ids) if narrow else ids), weights

    # ---- the torch path alone: production, and the two things the test's reference adds, one at a time
    variant("own_int64", "torch path, logits a tensor of their own, int64 ids (production)",
            lambda: torch_route(own(), own(), K)[0].cpu())
    variant("own_int32", "torch path, logits a tensor of their own, ids narrowed to int32 on the device",
            lambda: dev_narrow(torch_route(own(), own(), K)[0]).cpu())
    variant("view_int64", "torch path, logits a view gates[:, :E], int64 ids",
            lambda: view_reference(g.gates.to(DEVICE), False)[0].cpu())
    variant("view_int32", "torch path, logits a view gates[:, :E], ids narrowed to int32 (the test's reference, alone)",
            lambda: view_reference(g.gates.to(DEVICE), True)[0].cpu())
    variant("devcopy_int64", "torch path, logits a copy of the view made on the device (gates[:, :E].contiguous()), int64",
            lambda: (lambda copy: torch_route(copy, copy, K)[0].cpu())(dev_view(g.gates.to(DEVICE), E).contiguous()))

    # ---- the pieces of the routing, one at a time
    measures: dict = {}
    try:
        gates_device = g.gates.to(DEVICE)
        view = dev_view(gates_device, E)
        measures["view_bits_differ"] = int((view.cpu().view(torch.int16) != g.logits.view(torch.int16)).sum())
        measures["view_cast_differ"] = int((view.float().cpu() != g.logits.float()).sum())
        p_view = dev_softmax(dev_view(g.gates.to(DEVICE), E)).cpu()
        p_own = dev_softmax(own()).cpu()
        measures["softmax_view_vs_own"] = float((p_view - p_own).abs().max())
        measures["softmax_own_vs_cpu"] = float((p_own - g.p).abs().max())
        top = torch.sort(g.p, dim=1, descending=True).values[:, :K + 1]
        top_own = torch.gather(p_own, 1, torch.sort(g.p, dim=1, descending=True).indices[:, :K + 1])
        measures["softmax_own_rel_top"] = float(((top_own - top).abs() / top).max())
        # the card's probabilities ranked on the host: does the card's softmax alone change a set?
        host_rank = torch.topk(p_own, K, dim=-1).indices
        measures["softmax_sets_ok"] = bool(same_rows(host_rank, g.cpu_ids)[g.clear].all())
        ordered = torch.sort(p_own, dim=1, descending=True).values
        measures["card_p_tie_rows"] = int((ordered[:, K - 1] == ordered[:, K]).sum()) if E > K else 0
        # host-made probabilities ranked on the card: does the card's top-k alone change a set?
        values, indices = dev_topk(g.p.clone().to(DEVICE), K)
        values, indices = values.cpu(), indices.cpu()
        raw["topk_of_host_probabilities"] = indices.to(torch.int64).tolist()
        measures["topk_sets_ok"] = bool(same_rows(indices, g.cpu_ids)[g.clear].all())
        measures["topk_dtype"] = str(indices.dtype)
        measures["topk_sorted_descending"] = bool((values[:, 1:] <= values[:, :-1]).all())
        measures["topk_values_exact"] = bool(torch.equal(values, torch.topk(g.p, K, dim=-1).values))
        measures["topk_order_rows_as_cpu"] = int((indices.to(torch.int64) == g.cpu_ids).all(dim=1).sum())
        # the tie rule of the card's top-k: all-equal values, and (real geometry) five candidates for the last slot
        equal_ids = dev_topk(torch.zeros(1, E, dtype=torch.float32).to(DEVICE), K)[1].cpu()[0].tolist()
        measures["topk_all_equal_selects"] = equal_ids
        if E >= 512 and K == 10:
            cut = torch.full((1, E), -4.0)
            cut[0, 100:109] = torch.arange(9, 0, -1).float()
            cut[0, [7, 44, 250, 300, 511]] = 0.5
            cut_p = F.softmax(cut, dim=1, dtype=torch.float32)
            measures["topk_five_at_the_cut_selects"] = dev_topk(cut_p.to(DEVICE), K)[1].cpu()[0].tolist()
        # int64 on the device
        known = g.cpu_ids.clone()
        back = known.to(DEVICE).cpu()
        narrowed = dev_narrow(known.to(DEVICE)).cpu()
        measures["int64_roundtrip_differ"] = int((back != known).sum())
        measures["int64_roundtrip_dtype"] = str(back.dtype)
        measures["narrow_host_table_differ"] = int((narrowed.to(torch.int64) != known).sum())
        # the same top-k indices read as int64 and, in a second graph, narrowed on the device
        indices_device = dev_topk(dev_softmax(own()), K)[1]
        wide = indices_device.cpu()
        narrow_later = dev_narrow(indices_device).cpu()
        measures["narrow_materialized_differ"] = int((narrow_later.to(torch.int64) != wide.to(torch.int64)).sum())
        say(f"{g.name} | pieces: the view read back: {measures['view_bits_differ']} of {g.logits.numel()} elements "
            f"differ; the view cast to fp32: {measures['view_cast_differ']} differ; softmax of the view against softmax "
            f"of an own tensor: max abs diff {measures['softmax_view_vs_own']:.3e}; softmax of an own tensor against "
            f"CPU: max abs diff {measures['softmax_own_vs_cpu']:.3e}, max relative diff over the top {K + 1} of a row "
            f"{measures['softmax_own_rel_top']:.3e}, its sets ranked on the host are CPU's: "
            f"{measures['softmax_sets_ok']}, rows of it that tie at the cut: {measures['card_p_tie_rows']}")
        say(f"{g.name} | pieces: torch.topk on the card of host-made probabilities: CPU's sets on the rows without a "
            f"tie: {measures['topk_sets_ok']}; values sorted descending: {measures['topk_sorted_descending']}; values "
            f"exact: {measures['topk_values_exact']}; rows in CPU's order: {measures['topk_order_rows_as_cpu']} of "
            f"{g.rows}; indices dtype on the host {measures['topk_dtype']}; all-equal values select "
            f"{measures['topk_all_equal_selects']}"
            + (f"; five equal values at the cut (7, 44, 250, 300, 511) select "
               f"{measures['topk_five_at_the_cut_selects']}" if "topk_five_at_the_cut_selects" in measures else ""))
        say(f"{g.name} | pieces: a host int64 table sent to the device and back: {measures['int64_roundtrip_differ']} "
            f"of {known.numel()} differ ({measures['int64_roundtrip_dtype']}); narrowed to int32 on the device: "
            f"{measures['narrow_host_table_differ']} differ; top-k indices read as int64, then narrowed on the device "
            f"in a second graph: {measures['narrow_materialized_differ']} differ")
    except Exception as exc:  # noqa: BLE001
        measures["raised"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        say(f"{g.name} | pieces: raised {measures['raised']}")
    raw["measures"] = measures
    results["measures"] = measures

    if kernel is None:
        return results

    # ---- the kernel alone, and against the production routing (what decides flag 1 against flag 0)
    variant("kernel_alone", "kernel alone", lambda: kernel(g.gates.to(DEVICE), E, K)[0].cpu())
    try:
        ids, weights = kernel(g.gates.to(DEVICE), E, K)
        ids, weights = ids.cpu(), weights.cpu()
        torch_ids, torch_weights = torch_route(own(), own(), K)
        torch_ids, torch_weights = torch_ids.cpu(), torch_weights.cpu()
        same = same_rows(ids, torch_ids)
        worst = float((torch.sort(weights.float(), dim=1).values - torch.sort(torch_weights.float(), dim=1).values)
                      .abs().max())
        rule_exact = bool((ids.to(torch.int64) == g.rule_ids).all())
        results["kernel_vs_production"] = bool(same[g.clear].all())
        results["kernel_rule_exact"] = rule_exact
        say(f"{g.name} | kernel against the production routing on the card, each in its own graph: the same experts on "
            f"{int(same[g.clear].sum())} of the {int(g.clear.sum())} rows that do not tie at the cut ({int(same.sum())} "
            f"of all {g.rows}); max abs diff of the weights {worst:.3e}; the kernel's ids are its rule exactly "
            f"(largest logit first, lowest id among equals): {rule_exact}")
    except Exception as exc:  # noqa: BLE001
        results["kernel_vs_production"] = None
        say(f"{g.name} | kernel against the production routing: raised {type(exc).__name__}: {str(exc)[:300]}")

    # ---- the test's sequence and its neighbours: the reference built, then the kernel, then both read
    def shared(narrow=True, kernel_first=False, read_reference_first=False, own_logits=False, kernel_on_copy=False):
        global SHARED
        SHARED = True
        try:
            gates_device = g.gates.to(DEVICE)
            kernel_input = g.gates.clone().to(DEVICE) if kernel_on_copy else gates_device

            def reference():
                if own_logits:
                    logits_device = own()
                    ids, weights = torch_route(logits_device, logits_device, K)
                    return (dev_narrow(ids) if narrow else ids), weights
                return view_reference(gates_device, narrow)

            if kernel_first:
                got = kernel(kernel_input, E, K)
                want = reference()
            else:
                want = reference()
                got = kernel(kernel_input, E, K)
            if read_reference_first:
                torch_ids = want[0].cpu()
                fused = got[0].cpu()
            else:
                fused = got[0].cpu()
                torch_ids = want[0].cpu()
            return fused, torch_ids
        finally:
            SHARED = False

    for key, label, kwargs in (
        ("shared", "the test's sequence (reference on the view with int32, then the kernel, one graph)", {}),
        ("shared_int64", "the same with int64 ids", dict(narrow=False)),
        ("shared_kernel_first", "the same as the test with the kernel called first", dict(kernel_first=True)),
        ("shared_read_reference_first", "the same as the test with the reference read first",
         dict(read_reference_first=True)),
        ("shared_kernel_on_copy", "the same as the test with the kernel on a second copy of gates (one graph, two "
         "inputs)", dict(kernel_on_copy=True)),
        ("shared_own", "production's two paths side by side (torch path on an own tensor with int64 ids, then the "
         "kernel, one graph)", dict(narrow=False, own_logits=True)),
        ("shared_own_int32", "the same with the ids narrowed to int32", dict(narrow=True, own_logits=True)),
    ):
        try:
            fused, torch_ids = shared(**kwargs)
        except Exception as exc:  # noqa: BLE001
            results[key + "/kernel"] = results[key + "/reference"] = None
            raw[key] = f"raised {type(exc).__name__}: {str(exc)[:300]}"
            say(f"{g.name} | {label}: raised {type(exc).__name__}: {str(exc)[:300]}")
            continue
        variant(key + "/kernel", f"{label}: the kernel's ids", lambda fused=fused: fused)
        variant(key + "/reference", f"{label}: the reference's ids", lambda torch_ids=torch_ids: torch_ids)
        if key == "shared":
            same = same_rows(fused, torch_ids)
            results["test_line_rows"] = int(same.sum())
            say(f"{g.name} | the card test's line would be: ids: {int(same.sum())} of {g.rows} rows select the card's "
                f"torch.topk set ({int(g.clear.sum())} rows do not tie at the cut)")

    # ---- one op of the reference at a time in the kernel's graph: where in the reference does it start?
    SHARED = True
    try:
        gates_device = g.gates.to(DEVICE)
        probabilities = dev_softmax(dev_view(gates_device, E))
        fused = kernel(gates_device, E, K)[0].cpu()
        p_shared = probabilities.cpu()
        kernel_ok = bool(same_rows(fused, g.rule_ids).all())
        results["shared_softmax_only/kernel"] = kernel_ok
        results["shared_softmax_only/probabilities"] = float((p_shared - g.p).abs().max())
        say(f"{g.name} | only the softmax of the view in the kernel's graph: the kernel keeps its rule: {kernel_ok}; "
            f"the probabilities against CPU: max abs diff {results['shared_softmax_only/probabilities']:.3e}")
    except Exception as exc:  # noqa: BLE001
        results["shared_softmax_only/kernel"] = None
        say(f"{g.name} | only the softmax of the view in the kernel's graph: raised {type(exc).__name__}: "
            f"{str(exc)[:300]}")
    finally:
        SHARED = False
    return results


def conclude(name: str, results: dict, kernel_loaded: bool) -> tuple[str, str]:
    """(hypothesis code, sentence) for one geometry."""
    measures = results.get("measures", {})

    def state(key):
        return results.get(key)

    variants = {key: ok for key, ok in results.items() if isinstance(ok, bool) or ok is None}
    variants = {key: ok for key, ok in variants.items() if "/" in key or key in (
        "own_int64", "own_int32", "view_int64", "view_int32", "devcopy_int64", "kernel_alone")}
    wrong = sorted(key for key, ok in variants.items() if ok is False)
    failed = sorted(key for key, ok in variants.items() if ok is None)
    shared_wrong = [key for key in wrong if key.startswith("shared")]
    if state("own_int64") is False:
        if measures.get("softmax_sets_ok") is False:
            part = "the card's fp32 softmax changes the sets (H1)"
        elif measures.get("topk_sets_ok") is False:
            part = "the card's torch.topk of host-made probabilities changes the sets (H2)"
        else:
            part = "softmax and top-k are each right alone and wrong together"
        code = "H0-PRODUCTION"
        text = ("the card's production torch routing (softmax + top-k of its own logits, int64 ids) is NOT CPU "
                f"torch's: {part}. Flag 0 itself does not route as a CPU does")
    elif kernel_loaded and state("kernel_alone") is False:
        code, text = "KERNEL", "the kernel alone is wrong"
    elif state("own_int32") is False:
        code = "H3-INT32"
        text = ("the production torch path and the kernel alone are right; narrowing top-k's int64 ids to int32 on "
                "the device changes them" + ("" if state("view_int64") is not False else
                                              ", and routing on a view of gates changes them too (H4)"))
    elif state("view_int64") is False:
        code = "H4-VIEW"
        cure = {True: "a copy of the view made on the device routes right", False: "a copy of the view made on the "
                "device routes wrong too", None: "the device copy did not run"}[state("devcopy_int64")]
        text = ("the production torch path and the kernel alone are right; softmax + top-k of the view "
                f"gates[:, :experts] gives other ids ({cure})")
    elif state("view_int32") is False:
        code = "H3+H4"
        text = ("the production torch path and the kernel alone are right; the view and the int32 narrowing are each "
                "right alone and wrong together")
    elif shared_wrong:
        sides = sorted({key.split("/")[1] for key in shared_wrong})
        code = "H5-SHARED-" + "+".join(side.upper() for side in sides)
        text = ("every path is right in a graph of its own; inside one graph with the other path the ids change on "
                f"the side of: {', '.join(sides)} (variants: {', '.join(shared_wrong)})")
    elif failed:
        code, text = "INCOMPLETE", "the variants that ran have CPU torch's sets, but some raised"
    else:
        code = "H6-NONE"
        text = ("nothing goes wrong: every variant has CPU torch's sets on the rows without a tie, the test's own "
                f"sequence included (its line would say {results.get('test_line_rows', '?')} rows)")
    if kernel_loaded:
        equivalent = state("kernel_vs_production")
        kernel_shared = [key for key in shared_wrong if key.endswith("/kernel")]
        kernel_ran = [key for key, ok in variants.items() if key.endswith("/kernel") and ok is not None]
        text += ("; KERNEL = PRODUCTION ROUTING on the rows without a tie: " +
                 {True: "YES", False: "NO", None: "not measured"}[equivalent] +
                 ("; THE KERNEL'S ids change inside a shared graph" if kernel_shared else
                  "; the kernel keeps its rule in every shared graph" if kernel_ran else
                  "; the kernel in a shared graph: not measured"))
    if failed:
        text += f" [did not run: {', '.join(failed)}]"
    return code, f"{name}: [{code}] {text} (wrong: {', '.join(wrong) if wrong else 'none'})"


def main() -> int:
    say(f"device {DEVICE}; torch {torch.__version__}; PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')} "
        f"PT_HPU_LAZY_ACC_PAR_MODE={os.environ.get('PT_HPU_LAZY_ACC_PAR_MODE')}")
    if DEVICE == "hpu":
        import habana_frameworks.torch  # noqa: F401
    kernel, kernel_problem = load_kernel()
    say("moe_route kernel: " + ("loaded" if kernel is not None else f"NOT loaded ({kernel_problem})"))
    real = Geometry("real 512/10", 16, 512, 10, 513, 6)
    say(f"real geometry input as in the repository (route_real_r16): "
        f"{'yes' if sha(real.gates) == INPUT_SHA['route_real_r16'] else 'NO'}; rows that tie at the cut: "
        f"{(~real.clear).nonzero().flatten().tolist()}")
    geometries = (real, Geometry("router only 512/10", 16, 512, 10, 512, 6), Geometry("tiny 16/2", 16, 16, 2, 17, 2))
    conclusions = []
    for geometry in geometries:
        conclusions.append(conclude(geometry.name, run(geometry, kernel), kernel is not None))
    try:
        with open(OUT, "w") as handle:
            json.dump({"device": DEVICE, "torch": torch.__version__, "kernel_loaded": kernel is not None,
                       "kernel_problem": kernel_problem, "codes": [code for code, _ in conclusions],
                       "conclusions": [text for _, text in conclusions], "raw": RAW,
                       "cpu": {g.name: {"cpu_ids": g.cpu_ids.tolist(), "rule_ids": g.rule_ids.tolist(),
                                        "clear": g.clear.tolist()} for g in geometries}}, handle)
        say(f"ids of every variant written to {os.path.abspath(OUT)}")
    except OSError as exc:
        say(f"could not write {OUT}: {exc}")
    kernel_text = "" if kernel is not None else f" Kernel NOT LOADED ({kernel_problem}): torch-path variants only."
    print("VERDICT route: " + " || ".join(text for _, text in conclusions) + "." + kernel_text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
