# SPDX-License-Identifier: Apache-2.0
"""End-to-end n-gram (prompt lookup) speculative-decoding tests on CPU (M3b-A).

The runner feeds n-gram drafts as ragged decode rows (``hpu_model_runner.py``): a request
whose draft has ``L`` tokens in a batch whose widest row has ``K`` runs the row
``[last] + drafts + [0] * (K - L)`` at positions ``[p, p+1, .. p+L] + [0] * (K - L)``, and
an empty draft becomes the single placeholder token ``[-1]`` (always rejected, its id
clamped to 0 by ``_MtpTargetOutputs.forward``). These tests drive the tiny Flash-Next
checkpoint with ``speculative_config={"method": "mtp", "num_speculative_tokens": 3}`` (the
state layout depends only on ``num_speculative_tokens``; no drafter is needed) and verify
each ragged row in one ``T = 1 + K = 4``-row forward on the M3 lattice ``[1, 2, 3, 4]``,
exactly the padded shape above, with the same metadata and cache truncation as
``test_mtp_end_to_end._mtp_step``. Oracle drafts (the reference's ``generated_ids``) make
acceptance deterministic; every step's committed stream, logits and GDN/PLE state are
compared with an interleaved single-token chain, as in ``test_accept_paths_state_parity``.

The slot layout, tolerance and helpers are imported from ``test_mtp_end_to_end`` (see its
module docstring for the lattice/window arithmetic); the sparse ring test additionally
compares the QSA key ring of the request slot with the straight chain's over the whole ring
window, which catches a pad token at position 0 overwriting ring row 0. The verification
metadata carries each request's real length in ``query_start_loc_p`` (``1 + L``), as the
runner's decode path does; with the padded length ``T`` instead, the pad rows would write
ring row ``0 % R`` of the request slot.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs)
from tests.unit_tests.models.flash_next.test_mtp_end_to_end import (  # noqa: E402
    FP32_TOL, K, LATTICE, LONG_PROMPT, MTP, SPARSE_MAX_MODEL_LEN, SPEC_SLOT, STRAIGHT_SLOT, T, TAMPER, _dense_context,
    _load_prompt, _model_attns, _set_group_idx, _spec_state_modules, _state_gaps, _straight_decode, _target_prefill)
from tests.unit_tests.models.flash_next.test_mtp_parity import fp32_mtp_buffer  # noqa: E402


def _ragged_metadata(p0, n_accepted, real_len):
    """``_spec_metadata`` with the runner's REAL row length: ``query_start_loc_p`` is the
    cumulative scheduled tokens per request (``1 + L``, ``hpu_model_runner.py``
    ``_create_decode_input_data``), so QSA parks the ``K - L`` pad rows on the ring's scratch row."""
    md = decode_metadata(LATTICE[0], p0, num_tokens=T, query_lens=[real_len])
    lattice = torch.tensor([[LATTICE]], dtype=torch.int64)
    md.load_indices_tensor = lattice
    md.store_indices_tensor = lattice
    md.num_accepted_tokens = torch.tensor([n_accepted], dtype=torch.int32)
    return md


def _ngram_step(target, vllm_config, seq, p_last, n_accepted, eos, atts, drafts):
    """One n-gram verify / accept step; returns ``(n, seq, a, bonus, logits)``.

    ``drafts`` has ``L`` (1..K) tokens, the runner's ragged row: ``[-1]`` is the empty-draft
    placeholder (never accepted). The verification forward runs the padded row
    ``[seq[p_last]] + drafts + [0] * (K - L)`` at positions ``p_last .. p_last + L`` padded
    with ``0``, on the lattice with ``_spec_metadata(p_last, n_accepted)``; the attention
    caches are truncated to the last committed input row (as ``_mtp_step`` does).
    """
    L = len(drafts)
    assert 1 <= L <= K, f"an n-gram draft carries 1..K tokens, got {L}"
    rows = [seq[p_last]] + list(drafts) + [0] * (K - L)
    positions = list(range(p_last, p_last + 1 + L)) + [0] * (K - L)
    for att in atts:
        att.slot = SPEC_SLOT
    ctx = (seq[p_last - 2], seq[p_last - 1])
    with torch.no_grad(), set_forward_context(_ragged_metadata(p_last, n_accepted, 1 + L), vllm_config):
        hidden, _aux = target(input_ids=torch.tensor(rows),
                              positions=torch.tensor(positions),
                              **ple_kwargs(T, eos, context=ctx))
    logits = fp32_logits(target, hidden)
    argmax = logits.argmax(dim=-1).tolist()
    a = 0
    while a < L and argmax[a] == drafts[a]:  # the placeholder (-1) is never accepted
        a += 1
    bonus = argmax[a]
    seq = list(seq[:p_last + 1]) + list(drafts[:a]) + [bonus]
    for att in atts:  # drop the verification rows beyond the last committed input (the bonus row was never written)
        k, v = att.cache[SPEC_SLOT]
        att.cache[SPEC_SLOT] = (k[:p_last + a + 1], v[:p_last + a + 1])
    return a + 1, seq, a, bonus, logits


def _dense_ngram_chain(tiny_checkpoint, lengths, placeholder_steps=(), tamper_steps=None):
    """Prefill the 33-token prompt, run one n-gram step per entry of ``lengths``, interleaving
    straight decodes of the committed stream (the parity chain of ``_chain`` without a drafter).

    Step ``i`` drafts the first ``lengths[i]`` oracle tokens of the reference continuation after
    the last committed token; ``placeholder_steps`` replaces a step's draft with the runner's
    ``[-1]``, ``tamper_steps`` maps a step index to ``(idx, tok)`` draft replacements. Returns
    ``(seq, ids, gen, accepts, row0_gaps, bonus_gaps, state_gaps)``.
    """
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, model, _, vllm_config, eos, atts, _, gdn, ple = _dense_context(tiny_checkpoint)
    try:
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        seq = list(ids) + [int(fp32_logits(model, out[0])[-1].argmax())]
        p_last = len(seq) - 1
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)
        accepts, row0_gaps, bonus_gaps, state_gaps = [], [], [], []
        n = 1
        for step, L in enumerate(lengths):
            old_p = p_last
            if step in placeholder_steps:
                drafts = [-1]
            else:
                drafts = list(gen[len(seq) - len(ids):][:L])
            if tamper_steps and step in tamper_steps:
                idx, tok = tamper_steps[step]
                drafts[idx] = tok
            n, seq, a, bonus, logits = _ngram_step(model, vllm_config, seq, p_last, n, eos, atts, drafts)
            pick = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + 1, atts)
            rest = _straight_decode(model, vllm_config, eos, seq, old_p + 1, old_p + a + 1, atts)
            accepts.append(a)
            row0_gaps.append(float((logits[0] - pick).abs().max()))
            bonus_gaps.append(float((logits[a] - (rest if rest is not None else pick)).abs().max()))
            state_gaps.append(max(_state_gaps(model, gdn, ple, a)))
            p_last = len(seq) - 1
        return seq, ids, gen, accepts, row0_gaps, bonus_gaps, state_gaps
    finally:
        ctx.__exit__(None, None, None)


def _assert_chain_exact(seq, ids, gen, row0, bonus, state):
    """The committed stream is the reference prefix; every step's logits rows and committed
    GDN/PLE state match the interleaved straight chain."""
    stream = seq[len(ids):]
    assert stream == gen[:len(stream)], f"stream {stream} is not the reference prefix"
    for i, (r0, b, s) in enumerate(zip(row0, bonus, state)):
        assert r0 <= FP32_TOL, f"step {i} row-0 logits diverge from the straight chain: {r0:.3e}"
        assert b <= FP32_TOL, f"step {i} bonus logits diverge from the straight chain: {b:.3e}"
        assert s <= FP32_TOL, f"step {i} committed state diverges from the straight chain: {s:.3e}"


def test_ngram_ragged_chain_dense(tiny_checkpoint):
    """Ragged drafts of lengths 3, 1, placeholder and 2 on the dense 33-token prompt: the
    lattice stays exact with padded rows (token 0 at position 0) and a [-1] placeholder."""
    seq, ids, gen, accepts, row0, bonus, state = _dense_ngram_chain(tiny_checkpoint, [3, 1, 1, 2],
                                                                    placeholder_steps={2})
    print(f"n-gram dense chain: accepts={accepts}, row-0 gaps {[f'{g:.3e}' for g in row0]}, "
          f"bonus gaps {[f'{g:.3e}' for g in bonus]}, state gaps {[f'{g:.3e}' for g in state]}")
    assert accepts == [3, 1, 0, 2], f"acceptance schedule off: {accepts}"
    _assert_chain_exact(seq, ids, gen, row0, bonus, state)


def test_ngram_partial_reject_dense(tiny_checkpoint):
    """A tampered 2nd draft token in a 3-draft row rejects at a == 1; the next ragged step
    (length 2, loading idx[a]) rolls the rejection back and keeps parity."""
    seq, ids, gen, accepts, row0, bonus, state = _dense_ngram_chain(tiny_checkpoint, [3, 2],
                                                                    tamper_steps={0: (1, TAMPER)})
    print(f"n-gram partial reject: accepts={accepts}, row-0 gaps {[f'{g:.3e}' for g in row0]}, "
          f"bonus gaps {[f'{g:.3e}' for g in bonus]}, state gaps {[f'{g:.3e}' for g in state]}")
    assert accepts[0] == 1, f"the tampered draft was not the first rejection: {accepts}"
    _assert_chain_exact(seq, ids, gen, row0, bonus, state)


def test_ngram_ragged_sparse_ring(tiny_checkpoint):
    """Two ragged steps (lengths 1, 2) on the 2600-token prompt in SPARSE mode: logits, states
    and the key ring over the straight chain's ring window stay exact.

    The ring window comparison covers EVERY row the straight chain wrote in ``pos % R`` over
    ``P - R + 1 .. P`` (``P`` = last committed input position), so a pad token at position 0
    overwriting ring row 0 in the request's slot shows up as a gap on that row.
    """
    prompt = _load_prompt(tiny_checkpoint, LONG_PROMPT)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE,
                   speculative_config=MTP,
                   num_slots=8) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        fp32_mtp_buffer(model, LONG_PROMPT)
        gdn, ple = _spec_state_modules(model)
        _set_group_idx(model)
        atts = _model_attns(model)

        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        t0 = int(fp32_logits(model, out[0])[-1].argmax())
        assert t0 == gen[0], f"sparse prefill pick {t0} != reference {gen[0]}"
        seq = ids + [t0]
        p_last = len(seq) - 1
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)

        n = 1
        for L in (1, 2):
            old_p = p_last
            drafts = list(gen[len(seq) - len(ids):][:L])
            n, seq, a, bonus, logits = _ngram_step(model, vllm_config, seq, p_last, n, eos, atts, drafts)
            assert a == L, f"the oracle draft of length {L} was not fully accepted: a={a}"
            last = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + a + 1, atts)
            logits_gap = float((logits[a] - last).abs().max())
            ssm, conv, ple_g = _state_gaps(model, gdn, ple, a)
            p_last = len(seq) - 1

        # P = the last committed INPUT position (the step's bonus is sampled output, not an input row).
        P = len(seq) - 2
        qsa = [m for m in model.modules() if getattr(getattr(m, "indexer", None), "key_ring", None) is not None]
        ring_len = qsa[0].indexer.key_ring.shape[1]
        rows = torch.tensor([pos % ring_len for pos in range(max(0, P - ring_len + 1), P + 1)], dtype=torch.long)
        row_g = []
        for r in rows.tolist():
            gaps = [float((m.indexer.key_ring[LATTICE[0], r] - m.indexer.key_ring[STRAIGHT_SLOT, r]).abs().max())
                    for m in qsa]
            row_g.append(max(gaps))
        ring_g = max(row_g)
        print(f"n-gram sparse ring: logits {logits_gap:.3e}, ssm {ssm:.3e}, conv {conv:.3e}, ple {ple_g:.3e}, "
              f"ring {ring_g:.3e} (window rows {rows.tolist()}, R={ring_len}, P={P}, "
              f"per-row gaps {[f'{g:.3e}' for g in row_g]})")
        assert logits_gap <= FP32_TOL, f"bonus logits diverge from the straight chain: {logits_gap:.3e}"
        assert max(ssm, conv, ple_g) <= FP32_TOL, f"committed state diverges: {max(ssm, conv, ple_g):.3e}"
        assert ring_g <= FP32_TOL, f"key ring over the straight window diverged: {ring_g:.3e}"


def test_placeholder_draft_does_not_crash(tiny_checkpoint):
    """The runner schedules an empty n-gram draft as [-1]: the step verifies, rejects the
    placeholder (a == 0) and the logits stay finite."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids = list(prompt["prompt_ids"])
    ctx, model, _, vllm_config, eos, atts, _, gdn, ple = _dense_context(tiny_checkpoint)
    try:
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        seq = list(ids) + [int(fp32_logits(model, out[0])[-1].argmax())]
        p_last = len(seq) - 1
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)
        _n, new_seq, a, bonus, logits = _ngram_step(model, vllm_config, seq, p_last, 1, eos, atts, [-1])
        assert a == 0, f"the placeholder draft was accepted: a={a}"
        assert torch.isfinite(logits).all(), "the placeholder verification produced non-finite logits"
        assert len(new_seq) == len(seq) + 1 and new_seq[:len(seq)] == seq
        pick = _straight_decode(model, vllm_config, eos, new_seq, p_last, p_last + 1, atts)
        gap = float((logits[0] - pick).abs().max())
        print(f"n-gram placeholder step: a={a}, row-0 gap {gap:.3e}, bonus {bonus}")
        assert gap <= FP32_TOL, f"placeholder row-0 logits diverge from the straight chain: {gap:.3e}"
    finally:
        ctx.__exit__(None, None, None)
