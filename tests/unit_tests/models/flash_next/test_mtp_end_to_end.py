# SPDX-License-Identifier: Apache-2.0
"""End-to-end MTP speculative-decoding tests on CPU (M3-T8).

Drives the tiny Flash-Next checkpoint with ``speculative_config={"method": "mtp",
"num_speculative_tokens": 3}``: every step drafts ``K=3`` tokens with the MTP drafter,
verifies them in one 4-row target forward, and commits ``a`` drafts plus the target's
bonus token. Tests cover the dense 33-token greedy stream against the HF reference,
GDN/PLE state parity against an interleaved single-token chain, a forced rejection
(tampered draft) rolling back cleanly, one MTP step on the 2600-token prompt in
``QsaMode.SPARSE`` with indexer side-cache parity, and two mutation probes that break
the chain on purpose.

Slot layout (``num_slots=8``): the verification rows write lattice slots ``[1, 2, 3, 4]``
(state rows of the expanded GDN/PLE caches and the sparse ring slots). The speculative
chain runs at ``lattice[0] = 1``; after committing ``a`` drafts the chain's state lives at
``lattice[a]``. The reference single-token chain runs at slot 5, the drafter at slot 2.
With an MTP config the harness binds the expanded states (GDN conv ``(8, 6, 1280)``, PLE
conv ``(8, 12, 512)``); the kernels use rows ``[0:state_len)`` for prefill, single-token
decode and the speculative window alike, so the committed spec window ``[a : a + n)``
(window 3 for GDN conv, 9 for PLE conv) is compared with the straight chain's ``[0 : n)``.
The SSM state of the committed step is the lattice row ``lattice[a]``, which is the state
after the last committed *input* token - the straight chain decodes the committed stream
up to (and excluding) the bonus. The attention caches (``CachedSdpaAttention`` for dense,
``SparseSdpaAttention`` for sparse) are truncated to ``p_last + a + 1`` rows after each
step: the bonus's K/V row is never written by the verification forward, so keeping it
would leak the rejected rows' keys into the next step.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import next_step_hidden, split_draft_output  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, cpu_draft_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata,
    reset_cpu_states)
from tests.unit_tests.models.flash_next.test_decode_parity import CachedSdpaAttention  # noqa: E402
from tests.unit_tests.models.flash_next.test_mtp_parity import draft_fp32_logits, fp32_mtp_buffer  # noqa: E402

MTP = {"method": "mtp", "num_speculative_tokens": 3}
K = 3
T = 1 + K
LATTICE = [1, 2, 3, 4]
SPEC_SLOT = 1
STRAIGHT_SLOT = 5
DRAFT_SLOT = 2
FP32_TOL = 1e-5  # the M0/M2 fp32 parity tolerance
GDN_WINDOW = 3  # conv history (2 taps) + the token row, per committed input row
PLE_WINDOW = 9  # ngram history (8) + the token row
LONG_PROMPT = 2600
SPARSE_MAX_MODEL_LEN = 4096
TAMPER = 51234  # a valid out-of-context token id used to force a rejection


def _load_prompt(tiny_checkpoint, length: int) -> dict:
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    return next(p for p in reference["prompts"] if len(p["prompt_ids"]) == length)


def _qsa_layers(model) -> list:
    return [m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]


def _model_attns(model) -> list:
    """The cache-bearing attention seam of every QSA layer (mode-dependent)."""
    out = []
    for m in _qsa_layers(model):
        att = m.sparse_attn if m.qsa_mode is QsaMode.SPARSE else m.attn
        if att is not None:
            out.append(att)
    return out


def _install_cached_attention(model) -> None:
    for m in _qsa_layers(model):
        m.attn = CachedSdpaAttention(m)


def _set_dense_draft(draft) -> None:
    """Run the drafter dense (its K/V seam starts at the step's position, not at 0)."""
    for m in _qsa_layers(draft):
        m.qsa_mode = QsaMode.DENSE
        m.sparse_attn = None
    _install_cached_attention(draft)


def _spec_state_modules(model):
    """GDN modules (``kv_cache`` = conv + ssm) and PLE modules (conv only)."""
    gdn, ple = [], []
    for m in model.modules():
        if not isinstance(m, MambaBase):
            continue
        (gdn if len(m.kv_cache) == 2 else ple).append(m)
    return gdn, ple


def _set_group_idx(model) -> None:
    for m in model.modules():
        if isinstance(m, MambaBase):
            m.cache_group_idx = torch.tensor(0, dtype=torch.long)
        if isinstance(m, HpuQwen4ExpQSAAttention):
            m.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)


def _spec_metadata(p0: int, n_accepted: int):
    """Decode metadata for the 4-row verification forward on lattice slots ``[1, 2, 3, 4]``."""
    md = decode_metadata(LATTICE[0], p0, num_tokens=T)
    lattice = torch.tensor([[LATTICE]], dtype=torch.int64)  # [groups, bs, T]
    md.load_indices_tensor = lattice
    md.store_indices_tensor = lattice
    md.num_accepted_tokens = torch.tensor([n_accepted], dtype=torch.int32)
    return md


def _target_prefill(model, vllm_config, eos, ids, slot, atts, reset: bool = True):
    """Prefill ``ids`` into ``slot``; ``reset`` clears global CPU state (only for the first prefill)."""
    if reset:
        reset_cpu_states(model)
    for att in atts:
        att.cache.pop(slot, None)
        att.slot = slot
    with torch.no_grad(), set_forward_context(prefill_metadata(len(ids), slot), vllm_config):
        return model(input_ids=torch.tensor(ids), positions=torch.arange(len(ids)), **ple_kwargs(len(ids), eos))


def _draft_tokens(draft, vllm_config, eos, stream, pos0, atts, hidden, n_draft: int, mutate: bool = False):
    """Draft ``n_draft`` tokens: step 0 feeds the last committed token with ``hidden``, steps 1+ feed
    the drafter's own tokens. With ``mutate`` the hidden fed at step 1 is the step-0 sample row tiled
    over the hyper-connection width (mutation probe a)."""
    reset_cpu_states(draft)
    for att in atts:
        att.reset()
        att.slot = DRAFT_SLOT
    drafts = []
    stream = list(stream)
    hid = hidden.view(1, -1)
    for step in range(n_draft):
        ctx = (stream[step], stream[step + 1])
        with torch.no_grad(), set_forward_context(
                decode_metadata(DRAFT_SLOT, pos0 + step, load_slot=DRAFT_SLOT, store_slot=DRAFT_SLOT), vllm_config):
            ret = draft(input_ids=torch.tensor([stream[-1]]),
                        positions=torch.tensor([pos0 + step]),
                        hidden_states=hid,
                        spec_step_idx=step)
        sample, multi = split_draft_output(ret)
        tok = int(draft_fp32_logits(draft, sample)[-1].argmax())
        drafts.append(tok)
        stream.append(tok)
        hid = next_step_hidden(multi, torch.tensor([0])).view(1, -1)
        if mutate and step == 0:
            hid = sample.repeat(1, hid.shape[-1] // sample.shape[-1])
    return drafts


def _mtp_step(target, draft, vllm_config, seq, p_last, n_accepted, eos, atts, draft_atts, multi_last, tamper=None,
              force_accepted=None):
    """One draft-K / verify / accept step; returns ``(n, seq, multi_row, a, bonus, logits)``.

    ``tamper=(idx, tok)`` replaces draft ``idx`` (forced rejection); ``force_accepted`` overrides
    the acceptance count (mutation probe b).
    """
    drafts = _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts,
                           multi_last, K)
    if tamper is not None:
        drafts[tamper[0]] = tamper[1]
    rows = [seq[p_last]] + drafts
    for att in atts:
        att.slot = SPEC_SLOT
    ctx = (seq[p_last - 2], seq[p_last - 1])
    with torch.no_grad(), set_forward_context(_spec_metadata(p_last, n_accepted), vllm_config):
        hidden, aux = target(input_ids=torch.tensor(rows),
                             positions=torch.arange(p_last, p_last + T),
                             **ple_kwargs(T, eos, context=ctx))
    logits = fp32_logits(target, hidden)
    argmax = logits.argmax(dim=-1).tolist()
    a = 0
    while a < K and argmax[a] == drafts[a]:
        a += 1
    if force_accepted is not None:
        a = force_accepted - 1
    bonus = argmax[a]
    seq = list(seq[:p_last + 1]) + drafts[:a] + [bonus]
    for att in atts:  # drop the verification rows beyond the last committed input (the bonus row was never written)
        k, v = att.cache[SPEC_SLOT]
        att.cache[SPEC_SLOT] = (k[:p_last + a + 1], v[:p_last + a + 1])
    return a + 1, seq, aux[0].view(-1, aux[0].shape[-1])[a], a, bonus, logits


def _straight_decode(model, vllm_config, eos, seq, start, end, atts):
    """Plain single-token decodes of ``seq[start:end]`` at positions ``start..end-1``; last logits row."""
    last = None
    for pos in range(start, end):
        for att in atts:
            att.slot = STRAIGHT_SLOT
        ctx = (seq[pos - 2], seq[pos - 1])
        with torch.no_grad(), set_forward_context(
                decode_metadata(STRAIGHT_SLOT, pos, load_slot=STRAIGHT_SLOT, store_slot=STRAIGHT_SLOT), vllm_config):
            out = model(input_ids=torch.tensor([seq[pos]]),
                        positions=torch.tensor([pos]),
                        **ple_kwargs(1, eos, context=ctx))
        hidden = out[0] if isinstance(out, tuple) else out
        last = fp32_logits(model, hidden)[-1]
    return last


def _state_gaps(model, gdn, ple, a: int):
    """(ssm, gdn-conv, ple-conv) max-abs gaps of the committed spec state vs the straight chain."""
    ssm = max(float((m.kv_cache[1][LATTICE[a]] - m.kv_cache[1][STRAIGHT_SLOT]).abs().max()) for m in gdn)
    conv = max(float((m.kv_cache[0][SPEC_SLOT, a:a + GDN_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:GDN_WINDOW]
                      ).abs().max()) for m in gdn)
    ple_g = max(float((m.kv_cache[0][SPEC_SLOT, a:a + PLE_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:PLE_WINDOW]
                      ).abs().max()) for m in ple)
    return ssm, conv, ple_g


def _dense_context(tiny_checkpoint):
    """Open a dense fp32 target + drafter; returns the objects the chain helpers take."""
    ctx = cpu_model(tiny_checkpoint, fp32=True, num_slots=8, speculative_config=MTP)
    vllm_config, model = ctx.__enter__()
    eos = eos_token_id(vllm_config)
    fp32_mtp_buffer(model, 33)
    _install_cached_attention(model)
    gdn, ple = _spec_state_modules(model)
    _set_group_idx(model)
    atts = _model_attns(model)
    draft = cpu_draft_model(vllm_config, fp32=True, num_slots=8)
    _install_cached_attention(draft)
    _set_group_idx(draft)
    return ctx, model, draft, vllm_config, eos, atts, _model_attns(draft), gdn, ple


def _chain(model, draft, vllm_config, eos, atts, draft_atts, gdn, ple, ids, steps, tamper_step=None, force=None):
    """Prefill, run ``steps`` MTP steps interleaved with straight decodes of the committed stream.

    Returns ``(seq, accepts, gaps, tamper_info)``: ``gaps`` collects the per-step state gaps,
    ``tamper_info = (straight_argmax, bonus, bonus_logits_gap)`` for the tampered step.
    """
    out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
    logits = fp32_logits(model, out[0])
    seq = list(ids) + [int(logits[-1].argmax())]
    p_last = len(seq) - 1
    multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
    _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)
    accepts, gaps, tamper_info = [], [], None
    for step in range(steps):
        old_p = p_last
        kw = {}
        if step == tamper_step:
            kw["tamper"] = (2, TAMPER)  # replace the third draft
            if force is not None:
                kw["force_accepted"] = force
        n, seq, multi, a, bonus, logits = _mtp_step(model, draft, vllm_config, seq, p_last, 1, eos, atts, draft_atts,
                                                    multi, **kw)
        last = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + a + 1, atts)
        accepts.append(a)
        gaps.append(_state_gaps(model, gdn, ple, a))
        if step == tamper_step:
            tamper_info = (int(last.argmax()), bonus, float((logits[0] - last).abs().max()))
        p_last = len(seq) - 1
    return seq, accepts, gaps, tamper_info


def test_dense_fp32_greedy_stream_equals_reference(tiny_checkpoint):
    """6 MTP steps on the 33-token prompt: the committed stream is the HF reference's prefix."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        seq, accepts, _, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids, 6)
    finally:
        ctx.__exit__(None, None, None)
    stream = seq[len(ids):]
    print(f"MTP acceptance counts (k={K}): {accepts}")
    assert len(stream) >= 7, f"expected at least 7 committed tokens, got {len(stream)}"
    assert stream[:len(gen)] == gen[:len(stream)], f"stream {stream} is not the reference prefix {gen}"


def test_state_parity_after_speculative_steps(tiny_checkpoint):
    """After 6 MTP steps the committed GDN/PLE state matches an interleaved single-token chain."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids = list(prompt["prompt_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        _, accepts, gaps, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids, 6)
    finally:
        ctx.__exit__(None, None, None)
    worst = max(max(g) for g in gaps)
    print(f"MTP state parity acceptance counts: {accepts}, worst gap {worst:.3e}")
    assert worst <= FP32_TOL, f"committed spec state diverges from the straight chain: {worst:.3e}"


def test_forced_rejection_rolls_back(tiny_checkpoint):
    """Tampering the third draft rejects the tail; the committed stream and state stay exact."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids = list(prompt["prompt_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        normal, _, _, _ = _chain(*args, ids, 4)
        tampered, accepts, gaps, tamper_info = _chain(*args, ids, 4, tamper_step=2)
    finally:
        ctx.__exit__(None, None, None)
    straight_pick, bonus, bonus_gap = tamper_info
    print(f"MTP forced-rejection acceptance counts: {accepts}, bonus gap {bonus_gap:.3e}")
    assert accepts[2] == 0, f"tampered draft was accepted: {accepts}"
    assert straight_pick == bonus, "the committed bonus is not the target's own pick at the tampered step"
    assert bonus_gap <= FP32_TOL, f"bonus logits diverge from the straight chain: {bonus_gap:.3e}"
    assert tampered[:len(normal)] == normal, "the tampered stream diverged from the clean chain's stream"
    worst = max(max(g) for g in gaps)
    assert worst <= FP32_TOL, f"state diverged from the straight chain after the rollback: {worst:.3e}"


def test_sparse_2600_prompt_step(tiny_checkpoint):
    """One MTP step at position 2600 in SPARSE mode: picks, logits, states and side caches stay exact."""
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
        draft = cpu_draft_model(vllm_config, fp32=True, num_slots=8, max_model_len=SPARSE_MAX_MODEL_LEN)
        _set_dense_draft(draft)
        _set_group_idx(draft)
        draft_atts = _model_attns(draft)

        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        logits = fp32_logits(model, out[0])
        t0 = int(logits[-1].argmax())
        assert t0 == gen[0], f"sparse prefill pick {t0} != reference {gen[0]}"
        seq = ids + [t0]
        p_last = len(seq) - 1
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)

        n, seq, multi, a, bonus, logits = _mtp_step(model, draft, vllm_config, seq, p_last, 1, eos, atts, draft_atts,
                                                    multi)
        assert bonus == gen[1], f"MTP bonus {bonus} != reference pick {gen[1]}"
        last = _straight_decode(model, vllm_config, eos, seq, p_last, p_last + a + 1, atts)
        logits_gap = float((logits[0] - last).abs().max())
        ssm, conv, ple_g = _state_gaps(model, gdn, ple, a)

        side = [(m.indexer.key_ring[LATTICE[a]].clone(), m.indexer.compressed_key_cache[:650].clone())
                for m in model.modules() if getattr(getattr(m, "indexer", None), "key_ring", None) is not None]
        ring_g = comp_g = 0.0
        qsa = [m for m in model.modules() if getattr(getattr(m, "indexer", None), "key_ring", None) is not None]
        for (ring_a, comp_650), m in zip(side, qsa):
            ring_g = max(ring_g,
                         float((ring_a[:a + 1] - m.indexer.key_ring[STRAIGHT_SLOT, :a + 1]).abs().max()))
            comp_g = max(comp_g, float((comp_650 - m.indexer.compressed_key_cache[:650]).abs().max()))
        print(f"MTP sparse step: a={a}, logits {logits_gap:.3e}, ssm {ssm:.3e}, conv {conv:.3e}, "
              f"ple {ple_g:.3e}, ring {ring_g:.3e}, compressed {comp_g:.3e}")
        assert logits_gap <= FP32_TOL
        assert max(ssm, conv, ple_g) <= FP32_TOL
        assert ring_g <= FP32_TOL, f"key ring at the committed position diverged: {ring_g:.3e}"
        assert comp_g == 0.0, f"compressed block rows diverged: {comp_g:.3e}"


def test_mutations_break_the_chain(tiny_checkpoint):
    """(a) Tiling the drafter's step-1 hidden changes its drafts; (b) ``num_accepted=K+1`` breaks parity."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids = list(prompt["prompt_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = rest
        args = (model, draft, vllm_config, eos, atts, draft_atts, gdn, ple)
        # (a) at the prefill state: sample.repeat(1, hc) as the drafter's step-1 hidden changes d_2, d_3
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        t0 = int(fp32_logits(model, out[0])[-1].argmax())
        seq0 = ids + [t0]
        p_last = len(seq0) - 1
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        stream = seq0[max(p_last - 2, 0):p_last + 1]
        plain = _draft_tokens(draft, vllm_config, eos, stream, p_last, draft_atts, multi, K)
        mutated = _draft_tokens(draft, vllm_config, eos, stream, p_last, draft_atts, multi, K, mutate=True)
        print(f"MTP drafter mutation: plain={plain} mutated={mutated}")
        assert plain[0] == mutated[0], "the mutation must not touch the step-0 draft"
        assert plain[1:] != mutated[1:], f"drafter drafts unchanged by its hidden mutation: {plain} vs {mutated}"

        # (b) tampered drafts committed with num_accepted=K+1 diverge from the straight chain
        normal, _, _, _ = _chain(*args, ids, 4)
        forced, accepts_f, gaps_f, _ = _chain(*args, ids, 4, tamper_step=2, force=K + 1)
        ssm_f = max(float((m.kv_cache[1][LATTICE[K]] - m.kv_cache[1][STRAIGHT_SLOT]).abs().max()) for m in gdn)
        print(f"MTP force-accept mutation: accepts={accepts_f}, ssm gap {ssm_f:.3e}")
        assert accepts_f[2] == K, f"force-accepted step did not commit K drafts: {accepts_f}"
        assert forced[:p_last + 1] == normal[:p_last + 1]
        assert forced != normal, "the forced acceptance did not change the committed stream"
        assert ssm_f > 1e-3, f"state parity survived the force-accept mutation: {ssm_f:.3e}"
    finally:
        ctx.__exit__(None, None, None)
