#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On a card: every in-graph arg max of the plugin against the gathered path's, at the LM head's shapes.

The gathered path compares the LM head's logits as bf16: at TP > 1 its all-gather materializes the matmul's bf16
output before anything converts it. An arg max that runs in the matmul's own lazy graph may compare other values:
the card's graph compiler fuses a ``.to(torch.float32)`` that follows the GEMM into it, and the GEMM then hands out
its fp32 accumulator (engine-step's jobs 00254 / 00255: ``linear -> .to(fp32) -> argmax`` differs from the
materialized form on 2 of 100 random rows of a Flash-Next shard). A near-tie that is a tie in bf16 then resolves
another way than in the gathered path, in the CPU reference, or in the draft that proposed the token. The CPU
engine cannot show this; this script runs every registered helper and the reference on the card:

* reference: ``linear(h, W)``, ``mark_step``, the lowest index of the maximum (the card's ``argmax`` rule on ties,
  job 00252);
* each helper: the same ``linear`` and the helper in one lazy graph, its int ids read back.

A new in-graph arg max (a verify, draft or sampling path) belongs in ``helpers()``. Shapes: Flash-Next's hidden 2560
and the tiny model's 128, one TP8 shard (31040 rows) and the whole vocabulary (248320), rows 1, 3, 96 and 400; the
logits are random with planted near-ties (two columns 2**-9 apart, a tie in bf16). Prints one ``ARGMAX-CHECK`` line
per helper (rows differing from the reference, of how many) and a verdict; exit 1 when a helper differs
(``--report-only``: exit 0). Usage: ``python3 tools/flash_next/rental/argmax_precision_check.py [--report-only]``
in a farm container on one card (or on the CPU, where every helper agrees by construction).
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch

os.environ.setdefault("PT_HPU_LAZY_MODE", "1")  # as the servers run: the fusion is a lazy graph's
try:  # a card
    import habana_frameworks.torch.core as htcore
    DEV = "hpu"
except ImportError:  # the CPU: a check of the script itself
    htcore = None
    DEV = "cpu"


def _step():
    if htcore is not None:
        htcore.mark_step()


# The fused form itself (``linear -> .to(fp32) -> argmax``: vLLM's sampler and the gathered ``top_tokens`` at TP1,
# where no all-gather sits between the head and the cast): it differs on a card, which shows the fusion is there;
# it is reported but never fails the check.
CONTROL = "control: argmax(linear.float()), the fused form (sampler at TP1)"


def _owner(weight: torch.Tensor):
    """A one-rank stand-in of a vocabulary-parallel LM head: ``logits_processor`` is the matmul."""
    vocab = weight.shape[0]
    head = SimpleNamespace(tp_size=1, org_vocab_size=vocab, num_added_embeddings=0,
                           shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=vocab,
                                                         added_vocab_start_index=vocab,
                                                         added_vocab_end_index=vocab))
    linear = torch.nn.functional.linear
    owner = SimpleNamespace(lm_head=head, logits_processor=lambda _head, hidden, skip_gather=True: linear(hidden, weight))
    owner.compute_logits = lambda hidden: linear(hidden, weight)
    return owner


def helpers():
    """``{name: fn(hidden, weight) -> int ids [rows]}``: every arg max that can share the head's lazy graph."""
    from vllm_gaudi.distributed import vocab_parallel
    linear = torch.nn.functional.linear
    found = {
        CONTROL:
        lambda h, w: linear(h, w).to(torch.float32).argmax(dim=-1),
        "vocab_parallel_argmax":
        lambda h, w: vocab_parallel.vocab_parallel_argmax(_owner(w), h),
        "vocab_parallel.top_k + first_maximum (candidate greedy rows)":
        lambda h, w: vocab_parallel.first_maximum(*vocab_parallel.top_k(_owner(w), h, 64)).view(-1),
    }
    if hasattr(vocab_parallel, "greedy_ids"):  # VLLM_HPU_VERIFY_LOGITS_IN_GRAPH
        found["vocab_parallel.greedy_ids (VLG)"] = lambda h, w: vocab_parallel.greedy_ids(_owner(w), h)
    try:
        from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps as steps
    except Exception:  # noqa: BLE001 - another tree
        steps = None
    if steps is not None and hasattr(steps, "first_max_ids"):
        found["flash_next_mtp_steps.first_max_ids (draft chain / one-graph loop)"] = \
            lambda h, w: steps.first_max_ids(linear(h, w))
    if steps is not None and hasattr(steps, "draft_argmax"):
        found["flash_next_mtp_steps.draft_argmax (draft loop)"] = lambda h, w: steps.draft_argmax(linear(h, w))
    return found


def _lowest_max(logits: torch.Tensor) -> torch.Tensor:
    best = logits.amax(dim=-1, keepdim=True)
    ids = torch.arange(logits.shape[-1], dtype=torch.float32, device=logits.device)
    return torch.where(logits == best, ids, torch.full_like(ids, float(1 << 24))).amin(dim=-1).to(torch.int64)


def _planted(rows: int, hidden: int, weight: torch.Tensor, gen) -> torch.Tensor:
    """Hidden rows whose two largest logits sit 2**-9 apart (a tie once rounded to bf16) in half of the rows."""
    h = torch.randn(rows, hidden, generator=gen)
    w = weight.float().cpu()
    for r in range(0, rows, 2):
        logits = h[r] @ w.t()
        top2 = torch.topk(logits, 2).indices
        a, b = int(top2[0]), int(top2[1])
        gap = float(logits[a] - logits[b]) - 2.0**-9
        direction = w[a] - w[b]
        norm = float(direction @ direction)
        if norm > 0:
            h[r] = h[r] - direction * (gap / norm)
    return h.to(torch.bfloat16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("VLLM_HPU_VOCAB_PARALLEL_LOGITS", "1")  # the helpers' VP branch (one rank)
    found = helpers()
    differing = {name: [0, 0] for name in found}
    gen = torch.Generator().manual_seed(1234)
    for hidden, vocab in ((2560, 31040), (128, 31040), (128, 248320)):
        weight = (torch.randn(vocab, hidden, generator=gen) * (0.02 if hidden > 1000 else 0.2)).to(torch.bfloat16)
        weight_dev = weight.to(DEV)
        for rows in (1, 3, 96, 400):
            h = _planted(rows, hidden, weight, gen).to(DEV)
            _step()
            reference = torch.nn.functional.linear(h, weight_dev)
            _step()
            reference = _lowest_max(reference.to(torch.float32))
            _step()
            want = reference.cpu()
            for name, fn in found.items():
                got = fn(h, weight_dev)
                _step()
                got = got.cpu().to(torch.int64).view(-1)
                differing[name][0] += int((got != want).sum())
                differing[name][1] += rows
    bad = []
    for name, (n, total) in differing.items():
        print("ARGMAX-CHECK %-66s differs from the gathered path on %d of %d rows" % (name, n, total))
        if n and name != CONTROL:
            bad.append(name)
    print("ARGMAX-CHECK device %s: %s" % (DEV, "every helper equals the gathered path" if not bad else
                                          "DIFFERS: " + "; ".join(bad)))
    return 0 if (not bad or args.report_only) else 1


if __name__ == "__main__":
    sys.exit(main())
