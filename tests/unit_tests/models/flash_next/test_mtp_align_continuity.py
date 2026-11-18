# SPDX-License-Identifier: Apache-2.0
"""Multi-step align-mode continuity spike (M3-T10c-B): does align-mode state carry ACROSS steps?

T10b/T10c proved that ONE align-mode verification step - the ``recurrent_lattice.align_lattice``
load/store pair plus the ``recurrent_commit`` boundary copies - equals an interleaved single-token
chain for that step. This spike runs a THREE-step align-mode chain and compares, at every step, the
verification's own row-0 logits and the state the NEXT step will read against a straight decode
chain. Nothing here fixes production code: divergences are recorded, not repaired.

Synthetic block table (block_size 4): position block ``i`` is recurrent state slot ``8 + i``, and
the ``k = 3`` speculative scratch blocks the MambaManager keeps right after the position blocks are
slots ``[2, 3, 4]``, so the block row of a step is ``[8, 9, ...][:n_pos] + [2, 3, 4]`` with
``n_pos = ceil((context_len + 1 + k) / block_size)``. The prompt is prefilled into the block of its
last token (slot ``8 + (33 - 1) // 4 = 16``) and the straight reference chain runs at slot 5; the
attention cache of the chain is the shared ``SPEC_SLOT = 1``.

Scenarios (33-token prompt, 3 steps each, ``gen`` = the reference greedy continuation):

* ``S0`` control - every step tampers ``d_1`` so ``a = 0``: token 0's state moves to ``store[0]``
  only, no boundary commits. Expected: parity, and it is the test's own sanity gate.
* ``S1`` block completion - step 0 tampers ``d_3`` so ``a = 2``: token ``t = 2`` lands on position
  36, a multiple of block_size 4, so the boundary copy moves its state into the block slot the step
  has just completed (the same slot the chain reads next).
* ``S2`` accepted past the block end - step 0 accepts the whole 3-draft oracle (``a = 3``): the
  accepted run crosses the block end, the step commits only ``t = 2``, and the next step's context
  37 reads the block of position 36 (slot 17), which the step's kernel writes only at rows
  ``[0, window)`` of the store lattice's column-0 slot.
* ``S3`` completion then overrun - step 0 tampers ``d_3`` (``a = 2``, context 36, the block slot
  gets token 2's window), step 1 accepts everything (``a = 3``): token ``t = k = 3`` completes the
  next block and commits into the block slot of context 40, the slot step 2 reads.
* ``S3b`` - S3 with step 1's ``a = 3`` *forced* (the acceptance S3 would have had on a healthy
  chain), so the ``t = k = 3`` self-commit into the block slot of context 40 is exercised on its own.

Per step the spike records ``ctx``/``a``/``pairs`` and three gaps: ``log0`` (the verification's row-0
logits vs the straight decode at the same position), ``ssm`` (the next step's ssm read slot
``load_next[n - 1]`` vs slot 5) and ``conv``/``ple`` (the next step's conv read, rows
``[n - 1, n - 1 + window)`` of ``load_next[0]``, vs rows ``[0, window)`` of slot 5). Every assert is
made while parity holds; scenarios that diverge are marked ``xfail`` so the file stays green and the
divergence is recorded.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_commit import boundary_commits, commit_conv, commit_ssm  # noqa: E402
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
STEPS = 3
BLOCK_SLOT0 = 8  # position block i is recurrent slot BLOCK_SLOT0 + i
SCRATCH = [2, 3, 4]  # the k speculative scratch slots, appended after the position blocks
NUM_SLOTS = 24  # block slots 8..22 + the scratch/straight/spec slots


def _block_row(ctx: int, bs: int = BS) -> list[int]:
    """Block row of a step with ``ctx`` computed tokens: the position blocks, then the k scratch slots."""
    n_pos = -(-(ctx + 1 + K) // bs)
    row = [BLOCK_SLOT0 + i for i in range(n_pos)] + list(SCRATCH)
    assert max(row) < NUM_SLOTS, f"block row {row} needs slots beyond {NUM_SLOTS}"
    return row


def _align_context(tiny_checkpoint):
    """Dense fp32 target with ``NUM_SLOTS`` state slots + drafter (``_dense_context`` at 24 slots)."""
    ctx = cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS, speculative_config=MTP)
    vllm_config, model = ctx.__enter__()
    eos = eos_token_id(vllm_config)
    fp32_mtp_buffer(model, PROMPT_LEN)
    _install_cached_attention(model)
    gdn, ple = _spec_state_modules(model)
    _set_group_idx(model)
    atts = _model_attns(model)
    draft = cpu_draft_model(vllm_config, fp32=True, num_slots=8)
    _install_cached_attention(draft)
    _set_group_idx(draft)
    return ctx, model, draft, vllm_config, eos, atts, _model_attns(draft), gdn, ple


def _align_step(target, draft, vllm_config, seq, p_last, n_accepted, eos, atts, draft_atts, multi_last, row, bs,
                tamper=None, oracle=None, force_accept=False):
    """One align-mode MTP step; ``(n, seq, multi_row, a, bonus, logits, load, store)``.

    Same draft/verify/accept body as ``_mtp_step``, with the runner's align-mode metadata: the
    load/store lattices of ``recurrent_lattice.align_lattice`` for the block row of the step and
    ``lattice_store_differs`` from the two lattices. ``force_accept`` commits the whole oracle run
    regardless of the verification's argmax (probe S3b, whose step 1 state is already off-parity).
    """
    drafts = _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts,
                           multi_last, K)
    if oracle is not None:
        drafts = list(oracle[:K])
    if tamper is not None:
        drafts[tamper[0]] = tamper[1]
    rows = [seq[p_last]] + drafts
    load, store = align_lattice(row, p_last, 1 + K, bs, K)
    load_lat = torch.tensor([load], dtype=torch.int64)[None]  # [groups=1, bs=1, 1 + k]
    store_lat = torch.tensor([store], dtype=torch.int64)[None]
    for att in atts:
        att.slot = SPEC_SLOT
    md = _spec_metadata(p_last, n_accepted)
    md.load_indices_tensor = load_lat
    md.store_indices_tensor = store_lat
    md.lattice_store_differs = not torch.equal(load_lat, store_lat)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        hidden, aux = target(input_ids=torch.tensor(rows),
                             positions=torch.arange(p_last, p_last + 1 + K),
                             **ple_kwargs(1 + K, eos, context=(seq[p_last - 2], seq[p_last - 1])))
    logits = fp32_logits(target, hidden)
    argmax = logits.argmax(dim=-1).tolist()
    a = 0
    while a < K and argmax[a] == drafts[a]:
        a += 1
    if force_accept:
        a = K
    bonus = argmax[a]
    seq = list(seq[:p_last + 1]) + drafts[:a] + [bonus]
    for att in atts:  # drop the verification rows past the last committed input (the bonus row is unwritten)
        k, v = att.cache[SPEC_SLOT]
        att.cache[SPEC_SLOT] = (k[:p_last + a + 1], v[:p_last + a + 1])
    return a + 1, seq, aux[0].view(-1, aux[0].shape[-1])[a], a, bonus, logits, load, store


def _align_chain(model, draft, vllm_config, eos, atts, draft_atts, gdn, ple, ids, gen, steps, tamper_steps=None,
                 force_steps=None, bs: int = BS) -> list[dict]:
    """``steps`` align-mode MTP steps, each followed by straight decodes of its committed input rows.

    Returns the per-step record list the scenarios print and assert: ``ctx``, ``a``, ``n``,
    ``pairs``, the next step's ``ssm_slot``/``conv_slot`` and the ``log0``/``ssm``/``conv``/``ple``
    gaps, plus the committed ``stream``.
    """
    block_slot = BLOCK_SLOT0 + (len(ids) - 1) // bs  # the block of the last prompt token
    out = _target_prefill(model, vllm_config, eos, ids, block_slot, atts)
    logits = fp32_logits(model, out[0])
    seq = list(ids) + [int(logits[-1].argmax())]
    p_last = len(seq) - 1
    multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
    _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts, reset=False)  # the chain's attention keys
    _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)  # the reference chain
    assert _block_row(p_last, bs)[(p_last - 1) // bs] == block_slot, "the first step must load the prefill block"

    records, n, ctx = [], 1, p_last
    for step in range(steps):
        row, old_p = _block_row(ctx, bs), p_last
        oracle = list(gen[len(seq) - len(ids):])  # the drafter runs, the verification gets the reference tokens
        n, seq, multi, a, bonus, logits, load, store = _align_step(
            model, draft, vllm_config, seq, p_last, n, eos, atts, draft_atts, multi, row, bs, oracle=oracle,
            tamper=(tamper_steps or {}).get(step), force_accept=step in (force_steps or ()))

        pick = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + 1, atts)
        _straight_decode(model, vllm_config, eos, seq, old_p + 1, old_p + a + 1, atts)
        log0 = float((logits[0] - pick).abs().max())

        store_lat = torch.tensor([store], dtype=torch.int32)  # [batch=1, 1 + k]
        load_lat = torch.tensor([load], dtype=torch.int32)
        pairs = boundary_commits([ctx], [n], bs, store_lat, [row], load_lat)
        conv_pairs = [(i, t, int(store[0]), d) for (i, t, _s, d) in pairs]  # the conv rows live in column 0
        ts = [t for (_i, t, _s, _d) in pairs]
        for m in gdn:
            commit_ssm(m.kv_cache[1], pairs)
            commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
        for m in ple:
            commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)

        next_ctx = ctx + n  # the state the NEXT step reads (the last step's is what a 4th step would read)
        next_row = _block_row(next_ctx, bs)
        next_load, _next_store = align_lattice(next_row, next_ctx, 1 + K, bs, K)
        ssm_slot, conv_slot = next_load[n - 1], next_load[0]
        ssm_gap = max(float((m.kv_cache[1][ssm_slot] - m.kv_cache[1][STRAIGHT_SLOT]).abs().max()) for m in gdn)
        conv_gap = max(
            float((m.kv_cache[0][conv_slot, n - 1:n - 1 + GDN_WINDOW] -
                   m.kv_cache[0][STRAIGHT_SLOT, 0:GDN_WINDOW]).abs().max()) for m in gdn)
        ple_gap = max(
            float((m.kv_cache[0][conv_slot, n - 1:n - 1 + PLE_WINDOW] -
                   m.kv_cache[0][STRAIGHT_SLOT, 0:PLE_WINDOW]).abs().max()) for m in ple)
        records.append({
            "step": step,
            "ctx": ctx,
            "a": a,
            "n": n,
            "pairs": pairs,
            "ssm_slot": ssm_slot,
            "conv_slot": conv_slot,
            "log0": log0,
            "ssm": ssm_gap,
            "conv": conv_gap,
            "ple": ple_gap,
            "stream": seq[len(ids):],
        })
        p_last, ctx = len(seq) - 1, next_ctx
    return records


def _print_table(name: str, records: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'step':>4} {'ctx':>4} {'a':>2} {'n':>2} {'pairs':<16} {'ssm slot':>8} {'conv slot':>9} "
          f"{'conv rows':>10} {'log0':>10} {'ssm':>10} {'conv':>10} {'ple':>10}")
    for r in records:
        rows = f"[{r['n'] - 1}:{r['n'] - 1 + GDN_WINDOW})"
        print(f"{r['step']:>4} {r['ctx']:>4} {r['a']:>2} {r['n']:>2} {str(r['pairs']):<16} {r['ssm_slot']:>8} "
              f"{r['conv_slot']:>9} {rows:>10} {r['log0']:>10.3e} {r['ssm']:>10.3e} {r['conv']:>10.3e} "
              f"{r['ple']:>10.3e}")
    bad = next((r for r in records if max(r["log0"], r["ssm"], r["conv"], r["ple"]) > FP32_TOL), None)
    if bad is not None:
        print(f"{name}: first divergence at step {bad['step']} (ctx {bad['ctx']}, a {bad['a']}, pairs "
              f"{bad['pairs']}): log0 {bad['log0']:.3e}; ssm slot {bad['ssm_slot']} {bad['ssm']:.3e}; conv slot "
              f"{bad['conv_slot']} rows [{bad['n'] - 1}:{bad['n'] - 1 + GDN_WINDOW}) {bad['conv']:.3e}; "
              f"ple rows [{bad['n'] - 1}:{bad['n'] - 1 + PLE_WINDOW}) {bad['ple']:.3e}")
    else:
        print(f"{name}: no divergence, worst gap "
              f"{max(max(r['log0'], r['ssm'], r['conv'], r['ple']) for r in records):.3e}")


def _check(name: str, records: list[dict], gen: list[int], expected_a: list[int]) -> None:
    """Print the table, assert the committed stream is greedy and the plan's ``a`` while parity holds."""
    _print_table(name, records)
    for r in records:
        if r["log0"] > FP32_TOL:  # the step's own state is already off: its stream is no longer the greedy one
            continue
        exp = expected_a[r["step"]]  # None: the plan does not pin the acceptance of this step
        assert exp is None or r["a"] == exp, (f"{name} step {r['step']}: planned a={exp} at ctx {r['ctx']}, "
                                              f"got a={r['a']}")
        assert r["stream"] == gen[:len(r["stream"])], (f"{name} step {r['step']}: stream {r['stream']} is not the "
                                                       f"reference prefix {gen[:len(r['stream'])]}")
    worst = max(max(r["log0"], r["ssm"], r["conv"], r["ple"]) for r in records)
    assert worst <= FP32_TOL, (f"{name}: align-mode state does not carry across steps, worst gap {worst:.3e} > "
                               f"{FP32_TOL:.0e}")


def _run(tiny_checkpoint, tamper_steps, force_steps=()):
    ids = list(_load_prompt(tiny_checkpoint, PROMPT_LEN)["prompt_ids"])
    gen = list(_load_prompt(tiny_checkpoint, PROMPT_LEN)["generated_ids"])
    ctx, model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = _align_context(tiny_checkpoint)
    try:
        return ids, gen, _align_chain(model, draft, vllm_config, eos, atts, draft_atts, gdn, ple, ids, gen, STEPS,
                                      tamper_steps, force_steps)
    finally:
        ctx.__exit__(None, None, None)


def test_s0_control_no_boundary_commit(tiny_checkpoint):
    """S0: a = 0 at every step (tampered d_1) - token 0's state alone carries the chain, parity."""
    ids, gen, records = _run(tiny_checkpoint, {s: (0, TAMPER) for s in range(STEPS)})
    assert ids and gen
    _check("S0 control (a = 0, no boundary commits)", records, gen, [0, 0, 0])


@pytest.mark.xfail(strict=False, reason="spike finding: the t = 2 boundary copy into the completed block slot "
                   "clobbers the rows of the chain's own conv window, and the next step reads that window "
                   "(records printed by the test)")
def test_s1_block_completion_carries_to_the_next_step(tiny_checkpoint):
    """S1: step 0 tampers d_3 (a = 2) so token t = 2 completes a block; the chain must read it back."""
    ids, gen, records = _run(tiny_checkpoint, {0: (2, TAMPER), 1: (0, TAMPER), 2: (0, TAMPER)})
    assert ids and gen
    _check("S1 block completion at t = 2 (ctx 33 -> 36)", records, gen, [2, None, None])


@pytest.mark.xfail(strict=False, reason="spike finding: the accepted run crosses the block end, so the next "
                   "step reads the block of its first token - a slot the step wrote only at rows "
                   "[0, window) of the store lattice's column-0 slot (records printed by the test)")
def test_s2_accepted_run_overruns_the_block_end(tiny_checkpoint):
    """S2: step 0 accepts all 3 drafts (a = 3); context 37 reads the block of position 36."""
    ids, gen, records = _run(tiny_checkpoint, {1: (0, TAMPER), 2: (0, TAMPER)})
    assert ids and gen
    _check("S2 accepted run past the block end (ctx 33 -> 37)", records, gen, [3, None, None])


@pytest.mark.xfail(strict=False, reason="spike finding: the t = k = 3 completion commits into the block slot the "
                   "next step reads, and the self-copy shifts the rolling conv window below the next read "
                   "offset (records printed by the test)")
def test_s3_completion_then_overrun(tiny_checkpoint):
    """S3: a = 2 then a = 3 - the step commits t = 3 into the block slot step 2 reads (ctx 36 -> 40)."""
    ids, gen, records = _run(tiny_checkpoint, {0: (2, TAMPER), 2: (0, TAMPER)})
    assert ids and gen
    _check("S3 completion then overrun (ctx 33 -> 36 -> 40)", records, gen, [2, 3, None])


@pytest.mark.xfail(strict=False, reason="spike finding (S3 isolated): forcing the a = 3 commit at the block-aligned "
                   "step shows the t = k = 3 self-commit alone breaking the next step's conv read - the plain "
                   "window of token k lands at rows [0, window) of the block slot while the next step reads "
                   "rows [k, k + window) of it (records printed by the test)")
def test_s3b_forced_overrun_commit(tiny_checkpoint):
    """S3b: S3's step 1 with the a = 3 commit forced, so the t = k = 3 copy into slot 17 is exercised alone.

    S3's own step 1 already reads the window step 0 clobbered, so its acceptance collapses to ``a = 0`` and
    the ``t = k`` completion never runs. Forcing the oracle acceptance (the state a healthy chain would have)
    isolates the second commit: step 2's read of the block slot 17 must then equal the straight chain's.
    """
    ids, gen, records = _run(tiny_checkpoint, {0: (2, TAMPER), 2: (0, TAMPER)}, force_steps={1})
    assert ids and gen
    _check("S3b forced t = k = 3 commit (ctx 33 -> 36 -> 40)", records, gen, [2, 3, None])
