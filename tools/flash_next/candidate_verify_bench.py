#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What sampled verification of an MTP step costs on one card: the exact path against the candidate path (I07).

    python tools/flash_next/candidate_verify_bench.py [--device hpu] [--requests 1,8,32] [--k 3] [--out FILE]

Per request count B (k drafts each, the real vocabulary of 248,320, bf16 logits shaped like a model's) it times,
in lazy mode with a ``mark_step`` and a device wait per call, as the runner runs them:

* ``exact`` (VLLM_HPU_SPEC_SAMPLED_VERIFY=1): the bonus rows through vLLM's top-k / top-p and random sample (what
  the rejection sampler's own sampler pass does for a sampled request), then ``apply_sampling_constraints`` and
  ``sampled_rejection_sample`` over the verification rows;
* ``candidates`` (=2): ``hpu_candidate_verify.candidate_rejection_sample`` with its host inputs.

It also checks on the device what the CPU tests check on the CPU: the candidates' ids are the ids of the largest
logits (against the CPU's ``topk`` of the same logits, as sets), the tokens are valid ids or the placeholder, a
greedy batch (top_k 1) returns the arg max, and the draft acceptance rate of the candidate path matches the exact
path's (both estimate the mean of p(draft)). Results: one JSON line per B, and ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm_gaudi.v1.sample import hpu_candidate_verify as cv  # noqa: E402
from vllm_gaudi.v1.sample import hpu_rejection_sampler as hrs  # noqa: E402

VOCAB = 248320
TOP_K, TOP_P, TEMPERATURE = 20, 0.95, 1.0  # the checkpoint's generation_config


def _device_sync(device):
    if device.type == "hpu":
        import habana_frameworks.torch as htorch

        htorch.core.mark_step()
        torch.hpu.synchronize()


def _metadata(batch, device):
    from types import SimpleNamespace

    return SimpleNamespace(all_greedy=False, all_random=True, generators={},
                           temperature=torch.full((batch, ), TEMPERATURE, device=device),
                           top_k=torch.full((batch, ), TOP_K, dtype=torch.int32, device=device),
                           top_p=torch.full((batch, ), TOP_P, device=device))


def _exact(logits, target, bonus, draft, cumulative, n_draft, md):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch, random_sample

    bonus_logits = logits.index_select(0, bonus).to(torch.float32) / md.temperature.unsqueeze(1)
    bonus_probs = apply_top_k_top_p_pytorch(bonus_logits, md.top_k, md.top_p).softmax(dim=-1)
    bonus_ids = random_sample(bonus_probs, {}).to(torch.int32).unsqueeze(1)
    target_logits = logits.index_select(0, target.clamp(min=0)).to(torch.float32)
    target_logits = hrs.apply_sampling_constraints(target_logits, cumulative, md)
    return hrs.sampled_rejection_sample(draft, n_draft, cumulative, target_logits, bonus_ids, md)


def _time(fn, device, iters):
    for _ in range(3):  # compiles
        out = fn()
        _device_sync(device)
    laps = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        _device_sync(device)
        laps.append(1000.0 * (time.perf_counter() - start))
    laps.sort()
    return out, {"p50_ms": laps[len(laps) // 2], "min_ms": laps[0], "max_ms": laps[-1]}


def bench(batch: int, k: int, device, iters: int, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    rows = batch * (k + 1)
    logits_cpu = (torch.randn((rows, VOCAB), generator=generator) * 2.5).to(torch.bfloat16)
    logits = logits_cpu.to(device)
    target_cpu = torch.tensor([b * (k + 1) + i for b in range(batch) for i in range(k)], dtype=torch.int32)
    bonus_cpu = torch.tensor([b * (k + 1) + k for b in range(batch)], dtype=torch.int32)
    # MTP drafts the arg max of its own head: a draft that is the target's arg max on half of the rows
    argmax = logits_cpu.float().index_select(0, target_cpu).argmax(dim=-1).to(torch.int32)
    draft_cpu = torch.where(torch.arange(batch * k) % 2 == 0, argmax, (argmax + 1) % VOCAB).to(torch.int32)
    target, bonus, draft = target_cpu.to(device), bonus_cpu.to(device), draft_cpu.to(device)
    cumulative = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * k
    n_draft = [k] * batch
    md = _metadata(batch, device)
    noise = torch.Generator().manual_seed(cv.NOISE_SEED)

    def candidates():
        inputs = cv.host_inputs([TEMPERATURE] * batch, [TOP_K] * batch, [TOP_P] * batch, k, noise)
        return cv.candidate_rejection_sample(logits, target, bonus, draft, cumulative, inputs.to(device), -1)

    out_exact, t_exact = _time(lambda: _exact(logits, target, bonus, draft, cumulative, n_draft, md), device, iters)
    out_cand, t_cand = _time(candidates, device, iters)
    record = {"requests": batch, "k": k, "rows": rows, "exact": t_exact, "candidates": t_cand}

    # checks on the device. bf16 logits tie often (88 to 100 % of rows at the 64th value, job 00135), so the id
    # checks are by value: every candidate id holds its candidate's value in the CPU's logits, and the tokens
    # strictly above the k-th value are the CPU's (ties at the k-th value may be cut differently).
    values, ids = cv.candidates(logits, cv.CANDIDATES)
    ids_cpu, values_cpu = ids.cpu(), values.cpu()
    top_cpu = torch.topk(logits_cpu.float(), cv.CANDIDATES, dim=-1)
    held = logits_cpu.float().gather(1, ids_cpu.long().clamp(0, VOCAB - 1))
    above = sum(set(ids_cpu[r][values_cpu[r] > values_cpu[r, -1]].tolist())
                == set(top_cpu.indices[r][top_cpu.values[r] > top_cpu.values[r, -1]].tolist()) for r in range(rows))
    record["ids_hold_their_values"] = bool(torch.equal(held, values_cpu))
    record["ids_above_kth_equal_cpu"] = "%d/%d" % (above, rows)
    record["kth_value_equal"] = bool(torch.equal(values_cpu[:, -1], top_cpu.values[:, -1]))
    for name, out in (("exact", out_exact), ("candidates", out_cand)):
        tokens = out.cpu()
        valid = ((tokens == -1) | ((tokens >= 0) & (tokens < VOCAB))).all().item()
        record[name + "_tokens_valid"] = bool(valid)
    # acceptance of the first draft over many draws: both paths estimate the mean of p(draft)
    accept = {"exact": 0, "candidates": 0}
    draws = 20
    for _ in range(draws):
        accept["exact"] += (_exact(logits, target, bonus, draft, cumulative, n_draft, md).cpu()[:, 0]
                            == draft_cpu.view(batch, k)[:, 0]).sum().item()
        accept["candidates"] += (candidates().cpu()[:, 0] == draft_cpu.view(batch, k)[:, 0]).sum().item()
    record["first_draft_accept_rate"] = {name: count / (draws * batch) for name, count in accept.items()}
    # greedy (top_k 1) over candidates returns the arg max
    greedy_inputs = cv.host_inputs([0.0] * batch, [-1] * batch, [1.0] * batch, k, noise).to(device)
    greedy = cv.candidate_rejection_sample(logits, target, bonus, draft, cumulative, greedy_inputs, -1).cpu()
    first_rows = logits_cpu.float().index_select(0, target_cpu.view(batch, k)[:, 0])
    greedy_logit = first_rows.gather(1, greedy[:, :1].long().clamp(0, VOCAB - 1)).view(-1)
    record["greedy_first_token_holds_the_max"] = bool(torch.equal(greedy_logit, first_rows.max(dim=-1).values))
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="hpu")
    parser.add_argument("--requests", default="1,8,32")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.device == "hpu":
        import habana_frameworks.torch  # noqa: F401 - registers the device
    device = torch.device(args.device)
    records = []
    for batch in [int(word) for word in args.requests.split(",") if word]:
        record = bench(batch, args.k, device, args.iters)
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(records, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
