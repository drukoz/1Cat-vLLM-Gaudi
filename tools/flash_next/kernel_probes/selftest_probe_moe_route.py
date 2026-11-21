#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Self-test of probe_moe_route.py on a CPU: every way the card can go wrong is injected, and the probe's verdict
must name it.

    python selftest_probe_moe_route.py            runs every fault in a process of its own; exit 0 when all are named
    python selftest_probe_moe_route.py <fault>    runs the probe once with that fault (its whole output)

The "device" is the CPU and the kernel is a stand-in that returns the kernel's rule (the largest logits in
descending order, the lowest expert id first among equals).  A fault replaces one device-side function of the
probe, which is what a card that goes wrong in that construct does to it.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probe_moe_route.py"  # tools/flash_next/kernel_probes

# fault -> the hypothesis code the verdict of the real geometry must carry
EXPECTED = {
    "none": "[H6-NONE]",
    "tie_rule_highest_id": "[H6-NONE]",  # another tie rule of top-k is no failure on the rows without a tie
    "view_misread": "[H4-VIEW]",
    "view_misread_in_shared_graph_only": "[H5-SHARED-REFERENCE]",
    "narrow_reads_8_bytes": "[H3-INT32]",
    "narrow_wrong_after_view_only": "[H3+H4]",
    "softmax_over_tokens": "[H0-PRODUCTION]",
    "topk_smallest": "[H0-PRODUCTION]",
    "kernel_wrong": "[KERNEL]",
    "kernel_wrong_in_shared_graph_only": "[H5-SHARED-KERNEL]",
    "both_wrong_in_shared_graph_only": "[H5-SHARED-KERNEL+REFERENCE]",
    "kernel_raises": "[INCOMPLETE]",
}
# what else the verdict must say for some faults
ALSO = {
    "softmax_over_tokens": "the card's fp32 softmax changes the sets (H1)",
    "topk_smallest": "the card's torch.topk of host-made probabilities changes the sets (H2)",
    "view_misread": "a copy of the view made on the device routes right",
    "kernel_wrong_in_shared_graph_only": "THE KERNEL'S ids change inside a shared graph",
    "none": "KERNEL = PRODUCTION ROUTING on the rows without a tie: YES; the kernel keeps its rule in every shared",
    "kernel_wrong": "KERNEL = PRODUCTION ROUTING on the rows without a tie: NO",
}
# the line the card test would print, for the faults that are candidates for the card's "0 of 16"
TEST_LINE = {
    "none": "16 of 16 rows select",
    "view_misread": "1 of 16 rows select",
    "tie_rule_highest_id": "15 of 16 rows select",
    "narrow_reads_8_bytes": "0 of 16 rows select",
}


def run_one(fault: str) -> int:
    os.environ["PROBE_DEVICE"] = "cpu"
    os.environ.setdefault("PROBE_OUT", os.devnull)
    import torch
    import torch.nn.functional as F

    spec = importlib.util.spec_from_file_location("probe", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    def rule(gates, experts, top_k):
        ids = torch.sort(gates[:, :experts].float(), dim=1, descending=True, stable=True).indices[:, :top_k]
        return ids.to(torch.int32), torch.zeros(ids.shape, dtype=torch.bfloat16)

    def shifted(gates, experts, top_k):
        ids, weights = rule(gates, experts, top_k)
        return (ids + 1) % experts, weights

    kernel = rule
    if fault == "kernel_wrong":
        kernel = shifted
    if fault in ("kernel_wrong_in_shared_graph_only", "both_wrong_in_shared_graph_only"):

        def kernel(gates, experts, top_k):  # noqa: F811
            return (shifted if probe.SHARED else rule)(gates, experts, top_k)

    if fault == "kernel_raises":

        def kernel(gates, experts, top_k):  # noqa: F811
            raise RuntimeError("Empty tensor optional Uniqueid: 1 Strided Params Has Value: 0")

    probe.load_kernel = lambda: (kernel, "")

    plain_softmax, plain_narrow = probe.dev_softmax, probe.dev_narrow

    def strided(tensor) -> bool:
        return tensor._base is not None and not tensor.is_contiguous()

    def misreading_softmax(router_logits):
        """A strided view read as the first rows * columns elements of its base."""
        if strided(router_logits):
            rows, experts = router_logits.shape
            router_logits = router_logits._base.reshape(-1)[:rows * experts].reshape(rows, experts)
        return plain_softmax(router_logits)

    if fault == "view_misread":
        probe.dev_softmax = misreading_softmax
    if fault in ("view_misread_in_shared_graph_only", "both_wrong_in_shared_graph_only"):
        probe.dev_softmax = lambda router_logits: (misreading_softmax if probe.SHARED else plain_softmax)(router_logits)
    if fault == "narrow_reads_8_bytes":

        def dev_narrow(ids):
            flat = ids.reshape(-1)
            every_second = flat[::2]
            padded = torch.cat([every_second, torch.full((flat.numel() - every_second.numel(), ), -1,
                                                         dtype=flat.dtype)])
            return padded.reshape(ids.shape).to(torch.int32)

        probe.dev_narrow = dev_narrow
    if fault == "narrow_wrong_after_view_only":
        state = {"view": False}

        def dev_softmax(router_logits):
            state["view"] = strided(router_logits)
            return plain_softmax(router_logits)

        def dev_narrow(ids):  # noqa: F811
            wrong, state["view"] = state["view"], False
            return (plain_narrow(ids) + 1) % 4 if wrong else plain_narrow(ids)

        probe.dev_softmax, probe.dev_narrow = dev_softmax, dev_narrow
    if fault == "softmax_over_tokens":
        probe.dev_softmax = lambda router_logits: F.softmax(router_logits, dim=0, dtype=torch.float32)
    if fault == "topk_smallest":
        probe.dev_topk = lambda probabilities, top_k: torch.topk(probabilities, top_k, dim=-1, largest=False)
    if fault == "tie_rule_highest_id":

        def dev_topk(probabilities, top_k):
            # a stable descending sort of the reversed row takes the highest index among equal values
            ordered = torch.sort(probabilities.flip(1), dim=1, descending=True, stable=True)
            return ordered.values[:, :top_k], probabilities.shape[1] - 1 - ordered.indices[:, :top_k]

        probe.dev_topk = dev_topk
    return probe.main()


def run_all() -> int:
    failures = 0
    for fault, code in EXPECTED.items():
        proc = subprocess.run([sys.executable, __file__, fault], capture_output=True, text=True)
        verdict = [line for line in proc.stdout.splitlines() if line.startswith("VERDICT route: ")]
        real = verdict[0].split(" || ")[0] if verdict else ""
        test_line = [line for line in proc.stdout.splitlines() if "real 512/10 | the card test's line would be" in line]
        ok = proc.returncode == 0 and code in real and ALSO.get(fault, "") in real
        if fault in TEST_LINE:
            ok = ok and bool(test_line) and TEST_LINE[fault] in test_line[0]
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {fault:36s} wants {code:30s} got: {real[15:260] or proc.stderr[-400:]!r}"
              + (f" | test line: {test_line[0].split('ids: ')[1][:22]}" if test_line else ""))
    print(f"{len(EXPECTED) - failures} of {len(EXPECTED)} faults are named by the verdict")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_one(sys.argv[1]) if len(sys.argv) > 1 else run_all())
