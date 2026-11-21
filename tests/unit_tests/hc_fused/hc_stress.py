# SPDX-License-Identifier: Apache-2.0
"""A fused hyper-connection op called behind a queue of lazy ops: the body of the card test's queue stress.

In a model every fused op is called while the ops in front of it are still queued in the lazy bridge. The
card test's other cases call an op on fresh tensors, with one clone queued in front of it, and so say little
about that situation: on 2026-11-20, with the bridge's accumulation thread on, they failed at random (6 and 7
of 90, a different set each run) because the op was built before the producer of its input. The loader now
refuses that mode (``flashinfer_gaudi._native.lazy_accumulation_reason``); this body is the check that the
mode it requires holds under a queue, round after round.

It lives here, not in ``test_hpu_op.py``, so that ``test_hc_ops.py`` can run it on CPU with the oracle as the
kernel: the card is then the first run of the kernels behind a queue, not of this code.
"""

from __future__ import annotations

import torch

from tests.unit_tests.hc_fused import hc_ops
from tests.unit_tests.hc_fused.hc_case import INPUT_ORDER, OP_NAME, OUTPUT_ORDER, HcCase, compare, make_inputs, scalars

# One case per kernel of the decode loop, at the real geometry and a decode batch of 16 rows.
STRESS_CASES = ("norm_real_r16", "silu_split_real_r16", "gate_mix_real_r16", "combine_real_r16",
                "combine_norm_real_r16")
QUEUED_PAIRS = 48  # 96 lazy ops in front of the op, about one hyper-connection block's worth
ROUNDS = 8
_SCALE, _SHIFT = 1.0009765625, 0.25


def run_behind_a_queue(case: HcCase, device: str, rounds: int = ROUNDS, pairs: int = QUEUED_PAIRS) -> list[str]:
    """Call the op of `case` `rounds` times, each time right after queuing `pairs` multiply-add pairs.

    Returns the failure messages, [] for a pass: ``hc_case.compare`` of every round's outputs against the CPU
    oracle, and the queued ops' own result against the same arithmetic on the host. Whatever the device
    raises is not caught.
    """
    inputs = make_inputs(case)
    on_device = {name: tensor.to(device) for name, tensor in inputs.items()}
    filler_host = torch.full((4, 128), 0.5)
    filler = filler_host.to(device)
    filler_want = filler_host
    for _ in range(pairs):
        filler_want = filler_want * _SCALE + _SHIFT
    messages: list[str] = []
    for index in range(rounds):
        queued = filler
        for _ in range(pairs):  # ops the fused op does not depend on, queued just before it
            queued = queued * _SCALE + _SHIFT
        # read at call time, so a test can replace an op
        out = getattr(hc_ops, OP_NAME[case.kind])(*[on_device[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
        out = out if isinstance(out, tuple) else (out, )
        got = {name: tensor.cpu() for name, tensor in zip(OUTPUT_ORDER[case.kind], out)}
        queued_host = queued.cpu()
        messages += ["round %d: %s" % (index, message) for message in compare(case, inputs, got)]
        if not torch.allclose(queued_host, filler_want, rtol=1e-4, atol=0.0):
            messages.append("round %d: the queued ops returned %r, expected %r" %
                            (index, float(queued_host.flatten()[0]), float(filler_want.flatten()[0])))
    return messages
