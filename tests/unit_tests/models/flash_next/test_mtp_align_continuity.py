# SPDX-License-Identifier: Apache-2.0
"""Multi-step align-mode MTP continuity harness on the upstream scheme (M3-T10c-3).

Six multi-step align-mode MTP chains run on the CPU model through the T10c-1 production helpers
(``recurrent_lattice.align_lattice``, ``recurrent_commit.align_precopy_plan`` /
``precopy_slot_pairs`` / ``boundary_commits`` / ``self_commit_requests`` / ``commit_ssm`` /
``commit_conv``) and assert parity with the straight decode chain at every step. One request,
block_size 4, k = 3, a 33-token prompt; per decode step ``ctx`` is the computed-token count
(row 0's position) and ``n`` the accepted count of the previous step:

1. Pre-step copy: ``align_precopy_plan([prev], [ctx], [1 + k], [n], bs)`` plans the state move
   into the running block ``curr = align_running_block(ctx, 1 + k, bs)``; ``precopy_slot_pairs``
   maps it onto the row's slots and ``commit_ssm``/``commit_conv`` run it; ``prev, n =
   new_prev[0], new_acc[0]``.
2. Verification lattice: ``align_lattice(row, ctx, 1 + k, bs, k)`` (load == store) goes into
   the metadata as ``load_indices_tensor = store_indices_tensor = tensor([load])[None]``,
   ``lattice_store_differs = False``, ``num_accepted = n``.
3. After verification (``a`` accepted drafts, ``n = a + 1``): ``boundary_commits([ctx], [n],
   bs, tensor([store]), [row])`` selects the accepted tokens that complete a block, their ssm
   rows commit from ``store[t]`` and their plain conv windows from ``store[0]`` via
   ``commit_ssm``/``commit_conv``; a pair whose dst is ``store[0]`` is a self-commit
   (``self_commit_requests``) and resets the next step's accepted count to 1.
4. Index-stable synthetic block row (the real KVCacheManager keeps it, RUK-2847):
   ``row = [8 + i for i in range(cdiv(ctx + 1 + k, bs) + k)]`` over ``NUM_SLOTS = 40`` slots.

Scenarios (a step tampers draft j with TAMPER, forcing a = j; gen = the reference stream):

* S0 - tampered every step, a = 0: token 0's state alone carries the chain; one pre-step copy,
  one boundary commit (t = 0) when ctx crosses a block end at step 2.
* S1 - step 0 tampers draft 2 (a = 2): token t = 2 completes block 8 at ctx 33 and commits
  into slot 16; the step's pre-step copy of the running block carries the chain on.
* S2 - step 0 accepts all drafts (a = 3): the accepted run crosses the block end; step 1
  pre-copies with bias 3 from the t = 2 commit and continues a = 0.
* S3 - a = 2, then a clean a = 3: the t = 3 commit lands ON the running block (self-commit),
  which must reset the next accepted count to 1 - without the reset step 2 reads a stale PLE
  window (a 4.8e+00 gap).
* S4 - no tampering, 4 steps: every step accepts all drafts, so the running window walks the
  blocks with a bias-3 pre-step copy and a t = 2 boundary commit at every step.
* S5 - a = 1, 3, 2, 3: short and full acceptances interleaved, exercising bias 0, 3 and 2
  pre-step plans and a t = 0 / t = 1 boundary commit.

Per step the harness asserts the pre-step plan, ``a``, the boundary pairs and the self-commit
flag equal the task trace, the committed stream equals ``gen`` over the overlapping length,
and that the step's ``read``/``log0``/``ckpt`` gaps plus the final read gap are <= FP32_TOL.
Plain asserts: a divergence fails the run, the harness skips and weakens nothing.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_commit import (  # noqa: E402
    align_precopy_plan, boundary_commits, commit_conv, commit_ssm, precopy_slot_pairs, self_commit_requests)
from vllm_gaudi.v1.worker.recurrent_lattice import align_lattice  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_draft_model, cpu_model, eos_token_id, fp32_logits, ple_kwargs)
from tests.unit_tests.models.flash_next.test_mtp_end_to_end import (  # noqa: E402
    FP32_TOL, GDN_WINDOW, K, MTP, PLE_WINDOW, SPEC_SLOT, STRAIGHT_SLOT, TAMPER, _draft_tokens,
    _install_cached_attention, _load_prompt, _model_attns, _set_group_idx, _spec_metadata, _spec_state_modules,
    _straight_decode, _target_prefill)
from tests.unit_tests.models.flash_next.test_mtp_parity import fp32_mtp_buffer  # noqa: E402

BS = 4  # the synthetic block size: 4 positions per recurrent state slot
PROMPT_LEN = 33
BLOCK_SLOT0 = 8  # position block i is recurrent state slot BLOCK_SLOT0 + i
NUM_SLOTS = 40  # block slots 8..22 + the straight/spec slots

_SCENARIOS = {
    "S0": {
        "tampers": {0: (0, TAMPER), 1: (0, TAMPER), 2: (0, TAMPER)},
        "plan": [[(0, 0, 8, 9)], [], []],
        "a": [0, 0, 0],
        "pairs": [[], [], [(0, 0, 17, 16)]],
        "self": [False, False, False],
    },
    "S1": {
        "tampers": {0: (2, TAMPER), 1: (0, TAMPER), 2: (0, TAMPER)},
        "plan": [[(0, 0, 8, 9)], [], [(0, 0, 9, 10)]],
        "a": [2, 0, 0],
        "pairs": [[(0, 2, 19, 16)], [], []],
        "self": [False, False, False],
    },
    "S2": {
        "tampers": {1: (0, TAMPER), 2: (0, TAMPER)},
        "plan": [[(0, 0, 8, 9)], [(0, 3, 9, 10)], []],
        "a": [3, 0, 0],
        "pairs": [[(0, 2, 19, 16)], [], []],
        "self": [False, False, False],
    },
    "S3": {
        "tampers": {0: (2, TAMPER)},
        "plan": [[(0, 0, 8, 9)], [], [(0, 0, 9, 10)]],
        "a": [2, 3, 3],
        "pairs": [[(0, 2, 19, 16)], [(0, 3, 20, 17)], [(0, 3, 21, 18)]],
        "self": [False, True, True],
    },
    "S4": {
        "tampers": {},
        "plan": [[(0, 0, 8, 9)], [(0, 3, 9, 10)], [(0, 3, 10, 11)], [(0, 3, 11, 12)]],
        "a": [3, 3, 3, 3],
        "pairs": [[(0, 2, 19, 16)], [(0, 2, 20, 17)], [(0, 2, 21, 18)], [(0, 2, 22, 19)]],
        "self": [False, False, False, False],
    },
    "S5": {
        "tampers": {0: (1, TAMPER), 2: (2, TAMPER)},
        "plan": [[(0, 0, 8, 9)], [], [(0, 3, 9, 10)], [(0, 2, 10, 11)]],
        "a": [1, 3, 2, 3],
        "pairs": [[], [(0, 0, 17, 16)], [(0, 0, 18, 17)], [(0, 1, 20, 18)]],
        "self": [False, False, False, False],
    },
}


def _row(ctx: int) -> list[int]:
    """Rule 4: the index-stable block row of a step with ``ctx`` computed tokens."""
    row = [BLOCK_SLOT0 + i for i in range(-(-(ctx + 1 + K) // BS) + K)]
    assert max(row) < NUM_SLOTS, f"block row {row} needs slots beyond {NUM_SLOTS}"
    return row


def _snap(gdn, ple, slot: int, off: int = 0):
    """The (ssm, gdn conv, ple conv) rows of one state slot, cloned."""
    return ([m.kv_cache[1][slot].clone() for m in gdn],
            [m.kv_cache[0][slot, off:off + GDN_WINDOW].clone() for m in gdn],
            [m.kv_cache[0][slot, off:off + PLE_WINDOW].clone() for m in ple])


def _gap(a, b) -> float:
    return max(float((x - y).abs().max()) for xs, ys in zip(a, b) for x, y in zip(xs, ys))


def _chain(tiny_checkpoint, name: str, tampers, steps: int):
    """``steps`` align-mode MTP steps; the per-step records, the final read gap and ``gen``.

    Each record: ``step``, ``ctx``, ``plan`` (the pre-step plan, block indices), ``a``,
    ``pairs`` (the boundary pairs), ``self_commit``, and the ``read``/``log0``/``ckpt`` gaps.
    The final gap is what a step beyond the last would read, measured before its pre-step
    copy; the straight decode chain at STRAIGHT_SLOT is the reference.
    """
    ids = list(_load_prompt(tiny_checkpoint, PROMPT_LEN)["prompt_ids"])
    gen = list(_load_prompt(tiny_checkpoint, PROMPT_LEN)["generated_ids"])
    cm = cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS, speculative_config=MTP)
    vllm_config, model = cm.__enter__()
    try:
        eos = eos_token_id(vllm_config)
        fp32_mtp_buffer(model, PROMPT_LEN)
        _install_cached_attention(model)
        gdn, ple = _spec_state_modules(model)
        _set_group_idx(model)
        atts = _model_attns(model)
        draft = cpu_draft_model(vllm_config, fp32=True, num_slots=8)
        _install_cached_attention(draft)
        _set_group_idx(draft)
        datts = _model_attns(draft)
        block_slot = BLOCK_SLOT0 + (len(ids) - 1) // BS  # the block of the last prompt token
        out = _target_prefill(model, vllm_config, eos, ids, block_slot, atts)
        seq = list(ids) + [int(fp32_logits(model, out[0])[-1].argmax())]
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts, reset=False)
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)
        print(f"\n{name}: gen={gen[:20]}")
        ckpt, records = {}, []
        n, ctx, prev = 1, len(seq) - 1, None
        for step in range(steps):
            row = _row(ctx)  # rules 1, 2 and 4 through the T10c-1 helpers
            plan, new_prev, acc = align_precopy_plan([prev], [ctx], [1 + K], [n], BS)
            conv_pairs, ssm_pairs, ts = precopy_slot_pairs(plan, [row])
            for m in gdn:
                commit_ssm(m.kv_cache[1], ssm_pairs)
                commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
            for m in ple:
                commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)
            prev, n = new_prev[0], acc[0]
            load, store = align_lattice(row, ctx, 1 + K, BS, K)
            assert load == store, f"{name} step {step}: align_lattice load {load} != store {store}"
            lat = torch.tensor(load, dtype=torch.int64)
            read = _gap(([m.kv_cache[1][lat[n - 1]] for m in gdn],
                         [m.kv_cache[0][lat[0], n - 1:n - 1 + GDN_WINDOW] for m in gdn],
                         [m.kv_cache[0][lat[0], n - 1:n - 1 + PLE_WINDOW] for m in ple]),
                        _snap(gdn, ple, STRAIGHT_SLOT))
            p_last = ctx
            drafts = _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, datts,
                                   multi, K)
            drafts = (list(gen[len(seq) - len(ids):]) + drafts)[:K]  # the drafter runs, verify gets the reference
            if step in tampers:
                drafts[tampers[step][0]] = tampers[step][1]
            md = _spec_metadata(p_last, n)
            md.load_indices_tensor = lat[None, None]  # [groups=1, batch=1, 1 + k]
            md.store_indices_tensor = lat[None, None].clone()
            md.lattice_store_differs = False
            for att in atts:
                att.slot = SPEC_SLOT
            with torch.no_grad(), set_forward_context(md, vllm_config):
                hidden, aux = model(input_ids=torch.tensor([seq[p_last]] + drafts),
                                    positions=torch.arange(p_last, p_last + 1 + K),
                                    **ple_kwargs(1 + K, eos, context=(seq[p_last - 2], seq[p_last - 1])))
            logits = fp32_logits(model, hidden)
            am = logits.argmax(dim=-1).tolist()
            a = 0
            while a < K and am[a] == drafts[a]:
                a += 1
            seq = list(seq[:p_last + 1]) + drafts[:a] + [am[a]]
            multi = aux[0].view(-1, aux[0].shape[-1])[a]
            for att in atts:  # drop the verification rows past the last committed input
                k_, v_ = att.cache[SPEC_SLOT]
                att.cache[SPEC_SLOT] = (k_[:p_last + a + 1], v_[:p_last + a + 1])
            log0 = None
            for pos in range(p_last, p_last + a + 1):  # the straight reference walks the committed rows
                pick = _straight_decode(model, vllm_config, eos, seq, pos, pos + 1, atts)
                if log0 is None:
                    log0 = float((logits[0] - pick).abs().max())
                if (pos + 1) % BS == 0:
                    ckpt[pos] = _snap(gdn, ple, STRAIGHT_SLOT)
            n = a + 1  # rule 3: commit the accepted tokens that complete a block
            pairs = boundary_commits([ctx], [n], BS, lat[None], [row])
            conv_pairs = [(i, t, int(lat[0]), d) for (i, t, _s, d) in pairs]  # conv rows live in column 0
            ts = [t for (_i, t, _s, _d) in pairs]
            for m in gdn:
                commit_ssm(m.kv_cache[1], pairs)
                commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
            for m in ple:
                commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)
            self_commit = 0 in self_commit_requests(pairs, lat[None])
            ck = max([_gap(_snap(gdn, ple, d), ckpt[ctx + t]) for (_i, t, _s, d) in pairs] or [0.0])
            stream = list(seq[len(ids):])
            L = min(len(stream), len(gen))
            print(f"{name} step {step} ctx {ctx} plan {plan} a {a} pairs {pairs} self {self_commit} "
                  f"read {read:.2e} log0 {log0:.2e} ckpt {ck:.2e} stream_ok {stream[:L] == gen[:L]} "
                  f"gen_left {len(gen) - len(stream)}")
            records.append({"step": step, "ctx": ctx, "plan": plan, "a": a, "pairs": pairs,
                            "self_commit": self_commit, "read": read, "log0": log0, "ckpt": ck,
                            "stream": stream})
            ctx += n
            if self_commit:
                n = 1
        row = _row(ctx)  # what the NEXT step reads, measured before its pre-step copy
        plan, _, _ = align_precopy_plan([prev], [ctx], [1 + K], [n], BS)
        if plan:
            conv_pairs, ssm_pairs, ts = precopy_slot_pairs(plan, [row])
            bias, src_c, src_s = ts[0], conv_pairs[0][2], ssm_pairs[0][2]
            final = _gap(([m.kv_cache[1][src_s] for m in gdn],
                          [m.kv_cache[0][src_c, bias:bias + GDN_WINDOW] for m in gdn],
                          [m.kv_cache[0][src_c, bias:bias + PLE_WINDOW] for m in ple]),
                         _snap(gdn, ple, STRAIGHT_SLOT))
        else:
            load, _ = align_lattice(row, ctx, 1 + K, BS, K)
            final = _gap(([m.kv_cache[1][load[n - 1]] for m in gdn],
                          [m.kv_cache[0][load[0], n - 1:n - 1 + GDN_WINDOW] for m in gdn],
                          [m.kv_cache[0][load[0], n - 1:n - 1 + PLE_WINDOW] for m in ple]),
                         _snap(gdn, ple, STRAIGHT_SLOT))
        print(f"{name} final read gap {final:.2e}")
        return records, final, gen
    finally:
        cm.__exit__(None, None, None)


def _check(name: str, records: list[dict], final: float, gen: list[int]) -> None:
    """Assert the trace (plan, a, pairs, self-commit), the greedy stream and all gaps."""
    exp = _SCENARIOS[name]
    assert len(records) == len(exp["a"]), f"{name}: {len(records)} steps, expected {len(exp['a'])}"
    for r in records:
        step = r["step"]
        assert r["plan"] == exp["plan"][step], f"{name} step {step}: plan {r['plan']} != {exp['plan'][step]}"
        assert r["a"] == exp["a"][step], f"{name} step {step}: a {r['a']} != {exp['a'][step]}"
        assert r["pairs"] == exp["pairs"][step], f"{name} step {step}: pairs {r['pairs']} != {exp['pairs'][step]}"
        assert r["self_commit"] == exp["self"][step], (f"{name} step {step}: self_commit {r['self_commit']} "
                                                       f"!= {exp['self'][step]}")
        L = min(len(r["stream"]), len(gen))
        assert r["stream"][:L] == gen[:L], (f"{name} step {step}: stream {r['stream'][:L]} is not the greedy "
                                            f"prefix {gen[:L]}")
    worst = max([r["read"] for r in records] + [r["log0"] for r in records] + [r["ckpt"] for r in records]
                + [final])
    print(f"{name}: worst gap {worst:.2e} (tol {FP32_TOL:.0e})")
    assert worst <= FP32_TOL, f"{name}: worst gap {worst:.2e} > FP32_TOL {FP32_TOL:.0e}"


def test_s0_zero_accepts_carry_token_zero(tiny_checkpoint):
    """S0: a = 0 at every step - token 0's state alone carries the chain, one commit at ctx 35."""
    exp = _SCENARIOS["S0"]
    records, final, gen = _chain(tiny_checkpoint, "S0", exp["tampers"], len(exp["a"]))
    _check("S0", records, final, gen)


def test_s1_block_completion_at_t_2(tiny_checkpoint):
    """S1: step 0 a = 2 completes block 8 at ctx 33; the committed slot carries steps 1 and 2."""
    exp = _SCENARIOS["S1"]
    records, final, gen = _chain(tiny_checkpoint, "S1", exp["tampers"], len(exp["a"]))
    _check("S1", records, final, gen)


def test_s2_accepted_run_overruns_the_block_end(tiny_checkpoint):
    """S2: step 0 accepts all 3 drafts (a = 3); step 1 pre-copies with bias 3 from the t = 2 commit."""
    exp = _SCENARIOS["S2"]
    records, final, gen = _chain(tiny_checkpoint, "S2", exp["tampers"], len(exp["a"]))
    _check("S2", records, final, gen)


def test_s3_self_commit_resets_the_accepted_count(tiny_checkpoint):
    """S3: the clean a = 3 step self-commits t = 3 into the running block; the reset keeps parity."""
    exp = _SCENARIOS["S3"]
    records, final, gen = _chain(tiny_checkpoint, "S3", exp["tampers"], len(exp["a"]))
    _check("S3", records, final, gen)


def test_s4_all_accepted_walks_the_blocks(tiny_checkpoint):
    """S4: no tampering, 4 steps: every step a = 3, a bias-3 pre-step copy and a t = 2 commit."""
    exp = _SCENARIOS["S4"]
    records, final, gen = _chain(tiny_checkpoint, "S4", exp["tampers"], len(exp["a"]))
    _check("S4", records, final, gen)


def test_s5_mixed_acceptance_lengths(tiny_checkpoint):
    """S5: a = 1, 3, 2, 3 - bias 0/3/2 pre-step plans over t = 0 and t = 1 boundary commits."""
    exp = _SCENARIOS["S5"]
    records, final, gen = _chain(tiny_checkpoint, "S5", exp["tampers"], len(exp["a"]))
    _check("S5", records, final, gen)
