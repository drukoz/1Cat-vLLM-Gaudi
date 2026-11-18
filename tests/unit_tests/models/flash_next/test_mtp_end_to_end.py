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

Three tests feed *oracle drafts*: the drafter still runs, but its drafts are replaced by
HF's own greedy continuation (the reference's ``generated_ids``) before the verification,
so greedy verification accepts exactly the oracle prefix. That drives the lattice load
``idx[num_accepted - 1]`` and the ``num_accepted - 1`` conv-window selection with
``num_accepted > 1`` (the random tiny model's own drafts accept nothing), and fixes the
acceptance schedule ``[3, 1, x, 2]`` of the accept-path parity test and the forced
rejection (a tampered ``d_2`` after a 3-draft commit).

Slot layout (``num_slots=8``): the verification rows write lattice slots ``[1, 2, 3, 4]``
(GDN/PLE SSM state rows and the sparse ring slots). The speculative chain runs at
``lattice[0] = 1``; after committing ``a`` drafts the chain's state lives at
``lattice[a]``, and the next verification loads ``idx[a]`` (its ``num_accepted = a + 1``).
The reference single-token chain runs at slot 5, the drafter at slot 2. With an MTP config
the harness binds the expanded states (GDN conv ``(8, 6, 1280)``, PLE conv ``(8, 12, 512)``);
the kernels use rows ``[0:state_len)`` for prefill, single-token decode and the speculative
window alike, so the committed spec window ``[a : a + n)`` (window 3 for GDN conv, 9 for PLE
conv) is compared with the straight chain's ``[0 : n)``. The SSM state of the committed step
is the lattice row ``lattice[a]``, which is the state after the last committed *input*
token - the straight chain decodes the committed stream up to (and excluding) the bonus.
The GDN/PLE conv and the sparse key ring live in the lattice's column-0 slot (the request's
slot, ``lattice[0]``) and are windowed/positioned per ``num_accepted`` / position, so their
comparisons run at slot ``lattice[0]`` over the committed rows, not at ``lattice[a]``.
The attention caches (``CachedSdpaAttention`` for dense, ``SparseSdpaAttention`` for
sparse) are truncated to ``p_last + a + 1`` rows after each step: the bonus's K/V row is
never written by the verification forward, so keeping it would leak the rejected rows'
keys into the next step.
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
from vllm_gaudi.v1.worker.recurrent_commit import boundary_commits, commit_conv, commit_ssm  # noqa: E402

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
MUT_TOL = 1e-3  # a mutation must break parity by more than this
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
              oracle=None, num_accepted_override=None):
    """One draft-K / verify / accept step; returns ``(n, seq, multi_row, a, bonus, logits)``.

    ``tamper=(idx, tok)`` replaces draft ``idx`` (forced rejection). ``oracle`` (the reference's
    greedy tokens after the last committed token) replaces the drafter's drafts *after* the
    drafter has run, so the greedy verification accepts exactly the oracle prefix and the step
    commits ``a > 0`` drafts; ``tamper`` is applied on top. ``num_accepted_override`` feeds a
    wrong ``num_accepted`` into the verification metadata only (mutation probe b).
    """
    drafts = _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts,
                           multi_last, K)
    if oracle is not None:
        drafts = list(oracle[:K])
    if tamper is not None:
        drafts[tamper[0]] = tamper[1]
    rows = [seq[p_last]] + drafts
    for att in atts:
        att.slot = SPEC_SLOT
    ctx = (seq[p_last - 2], seq[p_last - 1])
    n_acc = n_accepted if num_accepted_override is None else num_accepted_override
    with torch.no_grad(), set_forward_context(_spec_metadata(p_last, n_acc), vllm_config):
        hidden, aux = target(input_ids=torch.tensor(rows),
                             positions=torch.arange(p_last, p_last + T),
                             **ple_kwargs(T, eos, context=ctx))
    logits = fp32_logits(target, hidden)
    argmax = logits.argmax(dim=-1).tolist()
    a = 0
    while a < K and argmax[a] == drafts[a]:
        a += 1
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


def _chain(model, draft, vllm_config, eos, atts, draft_atts, gdn, ple, ids, gen, steps, oracle_steps=(),
           tamper_steps=None, overrides=None):
    """Prefill, run ``steps`` MTP steps interleaved with straight decodes of the committed stream.

    ``num_accepted`` is threaded: the first verification runs with 1 after the prefill, every
    later one with the count the previous step committed (``a + 1``), so the verification loads
    its state from ``idx[num_accepted - 1]``. ``oracle_steps`` lists the steps that replace the
    drafter's drafts with the reference tokens ``gen`` (the accepted prefix is then the oracle's),
    ``tamper_steps`` maps a step index to its tampered draft and ``overrides`` maps a step index
    to a ``num_accepted`` override for its verification metadata (mutation probe b).

    Returns ``(seq, accepts, gaps, log0_gaps, bonus_gaps)``: ``gaps`` collects the per-step state
    gaps at ``lattice[a]``, ``log0_gaps`` the gap of the verification's row-0 logits against the
    straight chain's logits at the same position, ``bonus_gaps`` the same for the bonus row.
    """
    out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
    logits = fp32_logits(model, out[0])
    seq = list(ids) + [int(logits[-1].argmax())]
    p_last = len(seq) - 1
    multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
    _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)
    accepts, gaps, log0_gaps, bonus_gaps = [], [], [], []
    n = 1
    for step in range(steps):
        old_p = p_last
        oracle = list(gen[len(seq) - len(ids):]) if step in oracle_steps else None
        n, seq, multi, a, bonus, logits = _mtp_step(model, draft, vllm_config, seq, p_last, n, eos, atts, draft_atts,
                                                    multi, oracle=oracle, tamper=(tamper_steps or {}).get(step),
                                                    num_accepted_override=(overrides or {}).get(step))
        pick = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + 1, atts)
        rest = _straight_decode(model, vllm_config, eos, seq, old_p + 1, old_p + a + 1, atts)
        accepts.append(a)
        gaps.append(_state_gaps(model, gdn, ple, a))
        log0_gaps.append(float((logits[0] - pick).abs().max()))
        bonus_gaps.append(float((logits[a] - (rest if rest is not None else pick)).abs().max()))
        p_last = len(seq) - 1
    return seq, accepts, gaps, log0_gaps, bonus_gaps


def test_dense_fp32_greedy_stream_equals_reference(tiny_checkpoint):
    """6 MTP steps on the 33-token prompt: the committed stream is the HF reference's prefix."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        seq, accepts, _, _, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids,
                                       gen, 6)
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
        _, accepts, gaps, _, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids,
                                        None, 6)
    finally:
        ctx.__exit__(None, None, None)
    worst = max(max(g) for g in gaps)
    print(f"MTP state parity acceptance counts: {accepts}, worst gap {worst:.3e}")
    assert worst <= FP32_TOL, f"committed spec state diverges from the straight chain: {worst:.3e}"


def test_accept_paths_state_parity(tiny_checkpoint):
    """Oracle drafts commit a=3, a=1 and a=2: parity with num_accepted=4 and 2 feeding the next step.

    Step 0 accepts the whole 3-token oracle (num_accepted 1 -> 4), step 1's tampered ``d_2``
    leaves ``a == 1`` (-> 2), step 2 runs the plain drafter (the random tiny model accepts
    nothing), step 3's tampered ``d_3`` leaves ``a == 2``. Every step's committed state
    (lattice row ``lattice[a]``, conv windows at ``[a : a + window]``) and the verification's
    row-0 logits equal the straight chain's.
    """
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        seq, accepts, gaps, log0, bonus = _chain(*args, ids, gen, 4, oracle_steps={0, 1, 3},
                                                 tamper_steps={1: (1, TAMPER), 3: (2, TAMPER)})
    finally:
        ctx.__exit__(None, None, None)
    state_worst = [max(g) for g in gaps]
    print(f"MTP accept-path acceptance counts: {accepts}, state gaps {[f'{g:.3e}' for g in state_worst]}, "
          f"row-0 logits gaps {[f'{g:.3e}' for g in log0]}, bonus gaps {[f'{g:.3e}' for g in bonus]}")
    assert accepts[0] == 3, f"the oracle step did not commit K drafts: {accepts}"
    assert accepts[1] == 1, f"the tampered step did not commit exactly one draft: {accepts}"
    assert accepts[3] == 2, f"the second tampered step did not commit two drafts: {accepts}"
    stream = seq[len(ids):]
    assert stream == gen[:len(stream)], f"stream {stream} is not the reference prefix"
    for i, (g, l0) in enumerate(zip(state_worst, log0)):
        assert g <= FP32_TOL, f"step {i} committed state diverges from the straight chain: {g:.3e}"
        assert l0 <= FP32_TOL, f"step {i} row-0 logits diverge from the straight chain: {l0:.3e}"
        assert bonus[i] <= FP32_TOL, f"step {i} bonus logits diverge from the straight chain: {bonus[i]:.3e}"


def test_forced_rejection_rolls_back(tiny_checkpoint):
    """A tampered ``d_2`` after a 3-draft commit rejects the tail; the next step rolls back exactly.

    Step 0 (oracle) commits ``a == 3``, step 1 (tampered ``d_2``) commits ``a == 1``; the
    rejection step 2 runs with ``num_accepted = 2`` (loading ``idx[1]``, the state of the last
    accepted draft - its verification rewrites the rejected rows ``idx[2..3]``) and must match
    the straight decode continuation.
    """
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        seq, accepts, gaps, log0, bonus = _chain(*args, ids, gen, 3, oracle_steps={0, 1},
                                                 tamper_steps={1: (1, TAMPER)})
    finally:
        ctx.__exit__(None, None, None)
    print(f"MTP forced-rejection acceptance counts: {accepts}, rollback logits {log0[2]:.3e} / "
          f"bonus {bonus[2]:.3e}, per-step state gaps {[f'{max(g):.3e}' for g in gaps]}")
    assert accepts[0] == 3, f"the oracle step did not commit K drafts: {accepts}"
    assert accepts[1] == 1, f"the tampered draft was not the first rejection: {accepts}"
    assert log0[2] <= FP32_TOL, f"the rollback step's row-0 logits diverge from the straight chain: {log0[2]:.3e}"
    assert bonus[2] <= FP32_TOL, f"the rollback step's bonus diverges from the straight chain: {bonus[2]:.3e}"
    worst = max(max(g) for g in gaps)
    assert worst <= FP32_TOL, f"state diverged from the straight chain after the rollback: {worst:.3e}"
    stream = seq[len(ids):]
    assert stream == gen[:len(stream)], f"the rolled-back stream diverged: {stream}"


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
                                                    multi, oracle=gen[1:1 + K])
        assert a == K, f"the oracle drafts were not fully accepted: a={a}"
        assert bonus == gen[K + 1], f"MTP bonus {bonus} != reference token {gen[K + 1]}"
        last = _straight_decode(model, vllm_config, eos, seq, p_last, p_last + a + 1, atts)
        logits_gap = float((logits[a] - last).abs().max())
        ssm, conv, ple_g = _state_gaps(model, gdn, ple, a)

        # The ssm state of the committed row lives at lattice[a]; the key ring rows of the step are
        # position-addressed (ring[slot, pos % R]) inside the request's slot (the lattice's column
        # 0), so the ring rows the verification wrote at p_last..p_last+a are compared, modulo the
        # ring length R, at slot LATTICE[0].
        qsa = [m for m in model.modules() if getattr(getattr(m, "indexer", None), "key_ring", None) is not None]
        ring_len = qsa[0].indexer.key_ring.shape[1]
        rows = torch.tensor([(p_last + i) % ring_len for i in range(a + 1)], dtype=torch.long)
        side = [(m.indexer.key_ring[LATTICE[0]].index_select(0, rows).clone(),
                 m.indexer.compressed_key_cache[:650].clone()) for m in qsa]
        ring_g = comp_g = 0.0
        for (ring_a, comp_650), m in zip(side, qsa):
            ring_g = max(ring_g, float((ring_a - m.indexer.key_ring[STRAIGHT_SLOT].index_select(0, rows)).abs().max()))
            comp_g = max(comp_g, float((comp_650 - m.indexer.compressed_key_cache[:650]).abs().max()))
        print(f"MTP sparse step: a={a} (lattice slot {LATTICE[a]}), logits {logits_gap:.3e}, ssm {ssm:.3e}, "
              f"conv {conv:.3e}, ple {ple_g:.3e}, ring {ring_g:.3e}, compressed {comp_g:.3e}")
        assert logits_gap <= FP32_TOL
        assert max(ssm, conv, ple_g) <= FP32_TOL
        assert ring_g <= FP32_TOL, f"key ring at the committed positions diverged: {ring_g:.3e}"
        assert comp_g == 0.0, f"compressed block rows diverged: {comp_g:.3e}"


def test_mutations_break_the_chain(tiny_checkpoint):
    """(a) Tiling the drafter's step-1 hidden changes its drafts; (b) ``num_accepted=K+1`` breaks parity."""
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
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

        # (b) after the oracle commit and the tampered rejection (true num_accepted = 2 for the
        # rollback step), feeding num_accepted = K + 1 into that verification loads idx[K], the
        # state of a rejected token: the step's own logits/state must diverge from the straight
        # chain (the comparison is taken in the step that used the override).
        _, accepts_b, gaps_b, log0_b, _ = _chain(*args, ids, gen, 3, oracle_steps={0, 1},
                                                 tamper_steps={1: (1, TAMPER)}, overrides={2: K + 1})
        ssm_b = gaps_b[2][0]
        print(f"MTP num_accepted-override mutation: accepts={accepts_b}, row-0 logits gap {log0_b[2]:.3e}, "
              f"ssm gap at lattice[a] {ssm_b:.3e}")
        assert accepts_b[0] == 3 and accepts_b[1] == 1, f"the base chain of probe (b) is off: {accepts_b}"
        assert log0_b[2] > MUT_TOL or ssm_b > MUT_TOL, (f"the num_accepted={K + 1} override kept parity: "
                                                        f"logits {log0_b[2]:.3e}, ssm {ssm_b:.3e}")
    finally:
        ctx.__exit__(None, None, None)


def test_align_lattice_step_with_boundary_commit(tiny_checkpoint):
    """T8 align variant: a differing load/store lattice plus boundary commits matches the straight chain.

    The chain state is prefilled at slot 6 (the block of the last computed token), the
    verification stores token 0's state at slot 7 (the block of its own position) and the
    candidate checkpoints at scratch slots [2, 3, 4]. With block_size 2 the accepted odd
    tokens close a position block, so their states move into the new block's slot via the
    recurrent_commit helpers; that slot must then equal the interleaved straight chain's.
    """
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = _dense_context(tiny_checkpoint)
    chain_slot, new_slot = 6, 7
    try:
        out = _target_prefill(model, vllm_config, eos, ids, chain_slot, atts)
        _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts, reset=False)  # attention keys
        logits = fp32_logits(model, out[0])
        seq = list(ids) + [int(logits[-1].argmax())]
        p_last = len(seq) - 1
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)

        _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts, multi, K)
        drafts = gen[len(seq) - len(ids):len(seq) - len(ids) + K]  # oracle drafts
        rows = [seq[p_last]] + drafts
        load = torch.tensor([[[chain_slot, 2, 3, 4]]], dtype=torch.int64)
        store = torch.tensor([[[new_slot, 2, 3, 4]]], dtype=torch.int64)
        assert not torch.equal(load, store)
        for att in atts:
            att.slot = SPEC_SLOT
        md = _spec_metadata(p_last, 1)
        md.load_indices_tensor = load
        md.store_indices_tensor = store
        md.lattice_store_differs = True
        with torch.no_grad(), set_forward_context(md, vllm_config):
            hidden, aux = model(input_ids=torch.tensor(rows),
                                positions=torch.arange(p_last, p_last + T),
                                **ple_kwargs(T, eos, context=(seq[p_last - 2], seq[p_last - 1])))
        verify_logits = fp32_logits(model, hidden)
        argmax = verify_logits.argmax(dim=-1).tolist()
        a = 0
        while a < K and argmax[a] == drafts[a]:
            a += 1
        assert a == K, f"the oracle verification committed only {a} of {K} drafts"
        for att in atts:
            k, v = att.cache[SPEC_SLOT]
            att.cache[SPEC_SLOT] = (k[:p_last + a + 1], v[:p_last + a + 1])
        seq = list(seq[:p_last + 1]) + drafts[:a] + [argmax[a]]
        assert seq[len(ids):] == gen[:len(seq) - len(ids)], "the align step stream is not the reference prefix"

        _straight_decode(model, vllm_config, eos, seq, p_last, p_last + a + 1, atts)

        # Boundary commits, block_size 2: the accepted odd tokens complete a position block.
        ctx_len = p_last + 1
        block_rows = [[chain_slot] * 17 + [new_slot] * 15]
        pairs = boundary_commits([ctx_len], [a + 1], 2, store[0], block_rows, load[0])
        assert [(i, t, s, d) for (i, t, s, d) in pairs] == [(0, 1, 2, new_slot), (0, 3, 4, new_slot)]
        conv_pairs = [(i, t, int(store[0][0][0]), d) for (i, t, _s, d) in pairs]
        ts = [t for (_i, t, _s, _d) in pairs]
        for m in gdn:
            commit_ssm(m.kv_cache[1], pairs)
            commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
        for m in ple:
            commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)

        ssm_gap = max(float((m.kv_cache[1][new_slot] - m.kv_cache[1][STRAIGHT_SLOT]).abs().max()) for m in gdn)
        conv_gap = max(float((m.kv_cache[0][new_slot, 0:GDN_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:GDN_WINDOW]
                              ).abs().max()) for m in gdn)
        ple_gap = max(float((m.kv_cache[0][new_slot, 0:PLE_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:PLE_WINDOW]
                             ).abs().max()) for m in ple)
        worst = max(ssm_gap, conv_gap, ple_gap)
        print(f"align lattice step: a={a}, pairs {pairs}, gaps ssm={ssm_gap:.3e} conv={conv_gap:.3e} "
              f"ple={ple_gap:.3e}")
        assert worst <= FP32_TOL, f"committed align state diverges from the straight chain: {worst:.3e}"
    finally:
        ctx.__exit__(None, None, None)


def test_non_aligned_boundary_commit_with_equal_lattices(tiny_checkpoint):
    """Runner-gate regression: a t >= 1 completion on a step whose load == store lattices.

    Prompt 33, block_size 2, K=3: the verification runs right after the prefill with
    context_len 33 (unaligned), and the tampered ``d_3`` leaves ``accepted = 3``, so token
    ``t = 2`` completes a position block. The load and store lattices are equal, so the
    runner's ``lattice_store_differs`` flag is False and the kernels write every state
    into the shared lattice; only the boundary-commit payload (set unconditionally under
    prefix caching, review item 1) moves token 2's checkpoint into the completed block's
    slot. That slot must equal the interleaved straight chain's state - the case the
    flag-gated runner dropped (the align variant above is aligned at every step).
    """
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = _dense_context(tiny_checkpoint)
    block_slot = 7
    try:
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        logits = fp32_logits(model, out[0])
        t0 = int(logits[-1].argmax())
        assert t0 == gen[0], f"prefill pick {t0} != reference {gen[0]}"
        seq = list(ids) + [t0]
        p_last = len(seq) - 1  # 33: the first sampled token, row 0 (33 tokens computed)
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, STRAIGHT_SLOT, atts, reset=False)

        _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts, multi, K)
        drafts = [gen[1], gen[2], TAMPER]  # tampered d_3: the step accepts exactly two drafts
        rows = [seq[p_last]] + drafts
        for att in atts:
            att.slot = SPEC_SLOT
        md = _spec_metadata(p_last, 1)  # load == store == [[1, 2, 3, 4]]: the flag stays False
        assert torch.equal(md.load_indices_tensor, md.store_indices_tensor)
        with torch.no_grad(), set_forward_context(md, vllm_config):
            hidden, _aux = model(input_ids=torch.tensor(rows),
                                 positions=torch.arange(p_last, p_last + T),
                                 **ple_kwargs(T, eos, context=(seq[p_last - 2], seq[p_last - 1])))
        verify_logits = fp32_logits(model, hidden)
        argmax = verify_logits.argmax(dim=-1).tolist()
        a = 0
        while a < K and argmax[a] == drafts[a]:
            a += 1
        assert a == 2, f"expected exactly two accepted drafts, got a={a}"
        bonus = argmax[a]
        assert bonus == gen[3], f"bonus {bonus} != reference token {gen[3]}"
        for att in atts:
            k, v = att.cache[SPEC_SLOT]
            att.cache[SPEC_SLOT] = (k[:p_last + a + 1], v[:p_last + a + 1])
        seq = list(seq[:p_last + 1]) + drafts[:a] + [bonus]
        assert seq[len(ids):] == gen[:len(seq) - len(ids)], "the non-aligned step stream is not the reference prefix"

        # Straight decodes of the committed input rows (gen[0] .. gen[2]) at slot 5.
        _straight_decode(model, vllm_config, eos, seq, p_last, p_last + a + 1, atts)

        # Boundary commits, block_size 2: context_len 33 is unaligned and t = 2 completes a block.
        ctx_len = p_last  # 33 computed tokens (the prompt); row 0 = the first sampled token
        assert ctx_len % 2 == 1
        block_rows = [[SPEC_SLOT] * 17 + [block_slot]]
        pairs = boundary_commits([ctx_len], [a + 1], 2, md.store_indices_tensor[0], block_rows,
                                 md.load_indices_tensor[0])
        assert [(i, t, s, d) for (i, t, s, d) in pairs] == [(0, 2, LATTICE[2], block_slot)]
        conv_pairs = [(i, t, int(md.store_indices_tensor[0][0][0]), d) for (i, t, _s, d) in pairs]
        ts = [t for (_i, t, _s, _d) in pairs]
        for m in gdn:
            commit_ssm(m.kv_cache[1], pairs)
            commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
        for m in ple:
            commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)

        ssm_gap = max(float((m.kv_cache[1][block_slot] - m.kv_cache[1][STRAIGHT_SLOT]).abs().max()) for m in gdn)
        conv_gap = max(float((m.kv_cache[0][block_slot, 0:GDN_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:GDN_WINDOW]
                              ).abs().max()) for m in gdn)
        ple_gap = max(float((m.kv_cache[0][block_slot, 0:PLE_WINDOW] - m.kv_cache[0][STRAIGHT_SLOT, 0:PLE_WINDOW]
                             ).abs().max()) for m in ple)
        worst = max(ssm_gap, conv_gap, ple_gap)
        print(f"non-aligned boundary commit (flag False): a={a}, pairs {pairs}, gaps ssm={ssm_gap:.3e} "
              f"conv={conv_gap:.3e} ple={ple_gap:.3e}")
        assert worst <= FP32_TOL, f"flag-False boundary commit diverges from the straight chain: {worst:.3e}"
    finally:
        ctx.__exit__(None, None, None)
