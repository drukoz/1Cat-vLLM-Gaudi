# SPDX-License-Identifier: Apache-2.0
"""End-to-end MTP speculative-decoding tests on CPU (M3-T8).

Drives the tiny Flash-Next checkpoint with ``speculative_config={"method": "mtp",
"num_speculative_tokens": k}``: every step drafts ``k`` tokens with the MTP drafter,
verifies them in one ``1 + k``-row target forward, and commits ``a`` drafts plus the target's
bonus token. Tests cover the dense 33-token greedy stream against the HF reference,
GDN/PLE state parity against an interleaved single-token chain, every accepted count
``0..k`` (a rejection forced at each draft position), a forced rejection (tampered draft)
rolling back cleanly, two MTP steps on the 2600-token prompt in ``QsaMode.SPARSE`` with
indexer side-cache parity, and two mutation probes that break the chain on purpose.

Every test runs at each size of ``flash_next_spec_sizes.SPEC_SIZES`` (``k`` = 3, the recipe,
4 and 5): the ``spec_k`` fixture first asks the HPU platform's startup validator (see that
module: ``k = 5`` is proven here ahead of the validator's bound). ``StepLayout(k)`` holds the
sizes and state slots of a size; the module constants (``K``, ``T``, ``LATTICE``, ...) are the
recipe's and the examples below use them.

Three tests feed *oracle drafts*: the drafter still runs, but its drafts are replaced by
HF's own greedy continuation (the reference's ``generated_ids``) before the verification,
so greedy verification accepts exactly the oracle prefix. That drives the lattice load
``idx[num_accepted - 1]`` and the ``num_accepted - 1`` conv-window selection with
``num_accepted > 1`` (the random tiny model's own drafts accept nothing), and fixes the
acceptance schedule ``[3, 1, x, 2]`` of the accept-path parity test and the forced
rejection (a tampered ``d_2`` after a 3-draft commit).

Slot layout (``k = 3``, ``num_slots=11``): the verification rows write lattice slots ``[1, 2, 3, 4]``
(GDN/PLE SSM state rows and the sparse ring slots). The speculative chain runs at
``lattice[0] = 1``; after committing ``a`` drafts the chain's state lives at
``lattice[a]``, and the next verification loads ``idx[a]`` (its ``num_accepted = a + 1``).
The reference single-token chain runs at slot 5, the drafter at slot 2 of its own states, and the
align variants use slots 6..9 as position blocks. With an MTP config
the harness binds the expanded states (GDN conv ``(11, 6, 1280)``, PLE conv ``(11, 12, 512)``);
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

Every test runs twice: on the torch GDN verification path (``VLLM_HPU_GDN_CORE_VERIFY=0``) and through
the fused verify op (``1``), and asserts which path ran.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from flashinfer_gaudi.gdn_core_verify import GdnCoreVerifyMode  # noqa: E402
import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402

from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import next_step_hidden, split_draft_output  # noqa: E402
from vllm_gaudi.v1.worker.recurrent_commit import boundary_commits, commit_conv, commit_ssm  # noqa: E402

from tests.unit_tests.flash_next_spec_sizes import RECIPE_K  # noqa: E402
from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, cpu_draft_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata,
    reset_cpu_states)
from tests.unit_tests.models.flash_next.test_decode_parity import CachedSdpaAttention  # noqa: E402
from tests.unit_tests.models.flash_next.test_mtp_parity import draft_fp32_logits, fp32_mtp_buffer  # noqa: E402

SPEC_SLOT = 1
DRAFT_SLOT = 2


class StepLayout:
    """Sizes and state slots of the MTP step that drafts ``k`` tokens (``k = 3``: the module docstring's layout)."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.t = 1 + k  # rows of the verification forward
        self.mtp = {"method": "mtp", "num_speculative_tokens": k}
        self.lattice = list(range(SPEC_SLOT, SPEC_SLOT + 1 + k))  # state slots of the verification rows
        self.straight_slot = 2 + k  # the single-token reference chain
        self.chain_slot = 3 + k  # align variant: the block of the last computed token
        self.block_slots = [4 + k, 5 + k, 6 + k]  # align variants: the blocks the accepted tokens complete
        self.block_slot = self.block_slots[0]
        self.num_slots = 8 + k  # the last row is free: it is the sparse ring's scratch row


K = RECIPE_K
RECIPE = StepLayout(K)
MTP = RECIPE.mtp
T = RECIPE.t
LATTICE = RECIPE.lattice
STRAIGHT_SLOT = RECIPE.straight_slot
REFERENCE_TOKENS = 40  # tokens of the in-process non-speculative stream (the committed HF reference has 16)
FP32_TOL = 1e-5  # the M0/M2 fp32 parity tolerance
MUT_TOL = 1e-3  # a mutation must break parity by more than this
GDN_WINDOW = 3  # conv history (2 taps) + the token row, per committed input row
PLE_WINDOW = 9  # ngram history (8) + the token row
LONG_PROMPT = 2600
SPARSE_MAX_MODEL_LEN = 4096
TAMPER = 51234  # a valid out-of-context token id used to force a rejection


@pytest.fixture(autouse=True, params=[GdnCoreVerifyMode.OFF, GdnCoreVerifyMode.FUSED], ids=["verify_off", "verify_fused"])
def gdn_verify_path(request, monkeypatch):
    """Run each test on the torch GDN verification path and on gdn_core_verify, and check which one ran."""
    mode = request.param
    if "spec_k" in request.fixturenames:  # a failed startup check fails once, in spec_k, not again below
        request.getfixturevalue("spec_k")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", str(int(mode)))
    real = qwen3_5.gdn_core_verify
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(qwen3_5, "gdn_core_verify", counting)
    yield mode
    if mode == GdnCoreVerifyMode.FUSED:
        assert calls, "mode FUSED never called gdn_core_verify: the test did not exercise the fused path"
    else:
        assert not calls, f"mode OFF called gdn_core_verify {len(calls)} times"


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


def _spec_metadata(p0: int, n_accepted: int, lay: StepLayout = RECIPE):
    """Decode metadata for the ``1 + k``-row verification forward on the layout's lattice slots."""
    md = decode_metadata(lay.lattice[0], p0, num_tokens=lay.t)
    lattice = torch.tensor([[lay.lattice]], dtype=torch.int64)  # [groups, bs, T]
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
              oracle=None, num_accepted_override=None, lay: StepLayout = RECIPE):
    """One draft-k / verify / accept step; returns ``(n, seq, multi_row, a, bonus, logits)``.

    ``tamper=(idx, tok)`` replaces draft ``idx`` (forced rejection). ``oracle`` (the reference's
    greedy tokens after the last committed token) replaces the drafter's drafts *after* the
    drafter has run, so the greedy verification accepts exactly the oracle prefix and the step
    commits ``a > 0`` drafts; ``tamper`` is applied on top. ``num_accepted_override`` feeds a
    wrong ``num_accepted`` into the verification metadata only (mutation probe b).
    """
    K, T = lay.k, lay.t
    drafts = _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts,
                           multi_last, K)
    if oracle is not None:
        assert len(oracle) >= K, f"the reference stream has {len(oracle)} tokens left, the step drafts {K}"
        drafts = list(oracle[:K])
    if tamper is not None:
        drafts[tamper[0]] = tamper[1]
    rows = [seq[p_last]] + drafts
    for att in atts:
        att.slot = SPEC_SLOT
    ctx = (seq[p_last - 2], seq[p_last - 1])
    n_acc = n_accepted if num_accepted_override is None else num_accepted_override
    with torch.no_grad(), set_forward_context(_spec_metadata(p_last, n_acc, lay), vllm_config):
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


def _straight_decode(model, vllm_config, eos, seq, start, end, atts, slot: int = STRAIGHT_SLOT):
    """Plain single-token decodes of ``seq[start:end]`` at positions ``start..end-1`` in state slot ``slot``;
    last logits row."""
    last = None
    for pos in range(start, end):
        for att in atts:
            att.slot = slot
        ctx = (seq[pos - 2], seq[pos - 1])
        with torch.no_grad(), set_forward_context(decode_metadata(slot, pos, load_slot=slot, store_slot=slot),
                                                  vllm_config):
            out = model(input_ids=torch.tensor([seq[pos]]),
                        positions=torch.tensor([pos]),
                        **ple_kwargs(1, eos, context=ctx))
        hidden = out[0] if isinstance(out, tuple) else out
        last = fp32_logits(model, hidden)[-1]
    return last


def _state_gaps(model, gdn, ple, a: int, lay: StepLayout = RECIPE):
    """(ssm, gdn-conv, ple-conv) max-abs gaps of the committed spec state vs the straight chain."""
    straight = lay.straight_slot
    ssm = max(float((m.kv_cache[1][lay.lattice[a]] - m.kv_cache[1][straight]).abs().max()) for m in gdn)
    conv = max(float((m.kv_cache[0][SPEC_SLOT, a:a + GDN_WINDOW] - m.kv_cache[0][straight, 0:GDN_WINDOW]
                      ).abs().max()) for m in gdn)
    ple_g = max(float((m.kv_cache[0][SPEC_SLOT, a:a + PLE_WINDOW] - m.kv_cache[0][straight, 0:PLE_WINDOW]
                      ).abs().max()) for m in ple)
    return ssm, conv, ple_g


def _dense_context(tiny_checkpoint, lay: StepLayout = RECIPE):
    """Open a dense fp32 target + drafter; returns the objects the chain helpers take."""
    ctx = cpu_model(tiny_checkpoint, fp32=True, num_slots=lay.num_slots, speculative_config=lay.mtp)
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


def _plain_reference(model, vllm_config, eos, ids, gen, atts, slot: int, count: int = REFERENCE_TOKENS):
    """The non-speculative greedy stream of ``count`` tokens after ``ids``: a prefill and single-token decodes
    in state slot ``slot`` (all CPU state is reset first).

    The committed HF reference ``gen`` has 16 tokens per prompt, fewer than a chain of ``1 + k``-token steps
    commits; this stream continues it with the same model on the plain path and must start with ``gen``.
    """
    out = _target_prefill(model, vllm_config, eos, ids, slot, atts)
    seq = list(ids) + [int(fp32_logits(model, out[0])[-1].argmax())]
    while len(seq) - len(ids) < count:
        pos = len(seq) - 1
        seq.append(int(_straight_decode(model, vllm_config, eos, seq, pos, pos + 1, atts, slot=slot).argmax()))
    stream = seq[len(ids):]
    head = min(len(gen), count)
    assert stream[:head] == list(gen[:head]), f"the plain stream {stream[:head]} is not the HF reference {gen[:head]}"
    return stream


def _chain(model, draft, vllm_config, eos, atts, draft_atts, gdn, ple, ids, gen, steps, oracle_steps=(),
           tamper_steps=None, overrides=None, lay: StepLayout = RECIPE):
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
    straight = lay.straight_slot
    _target_prefill(model, vllm_config, eos, ids, straight, atts, reset=False)
    accepts, gaps, log0_gaps, bonus_gaps = [], [], [], []
    n = 1
    for step in range(steps):
        old_p = p_last
        oracle = list(gen[len(seq) - len(ids):]) if step in oracle_steps else None
        n, seq, multi, a, bonus, logits = _mtp_step(model, draft, vllm_config, seq, p_last, n, eos, atts, draft_atts,
                                                    multi, oracle=oracle, tamper=(tamper_steps or {}).get(step),
                                                    num_accepted_override=(overrides or {}).get(step), lay=lay)
        pick = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + 1, atts, slot=straight)
        rest = _straight_decode(model, vllm_config, eos, seq, old_p + 1, old_p + a + 1, atts, slot=straight)
        accepts.append(a)
        gaps.append(_state_gaps(model, gdn, ple, a, lay))
        log0_gaps.append(float((logits[0] - pick).abs().max()))
        bonus_gaps.append(float((logits[a] - (rest if rest is not None else pick)).abs().max()))
        p_last = len(seq) - 1
    return seq, accepts, gaps, log0_gaps, bonus_gaps


def test_dense_fp32_greedy_stream_equals_reference(tiny_checkpoint, spec_k):
    """6 MTP steps on the 33-token prompt: the committed stream is the HF reference's prefix."""
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        seq, accepts, _, _, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids,
                                       gen, 6, lay=lay)
    finally:
        ctx.__exit__(None, None, None)
    stream = seq[len(ids):]
    print(f"MTP acceptance counts (k={lay.k}): {accepts}")
    assert len(stream) >= 7, f"expected at least 7 committed tokens, got {len(stream)}"
    assert stream[:len(gen)] == gen[:len(stream)], f"stream {stream} is not the reference prefix {gen}"


def test_state_parity_after_speculative_steps(tiny_checkpoint, spec_k):
    """After 6 MTP steps the committed GDN/PLE state matches an interleaved single-token chain."""
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids = list(prompt["prompt_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        _, accepts, gaps, _, _ = _chain(rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7], ids,
                                        None, 6, lay=lay)
    finally:
        ctx.__exit__(None, None, None)
    worst = max(max(g) for g in gaps)
    print(f"MTP state parity acceptance counts: {accepts}, worst gap {worst:.3e}")
    assert worst <= FP32_TOL, f"committed spec state diverges from the straight chain: {worst:.3e}"


def test_accept_paths_state_parity(tiny_checkpoint, spec_k):
    """Oracle drafts commit a=k, a=1 and a=2: parity with num_accepted=k+1 and 2 feeding the next step.

    Step 0 accepts the whole k-token oracle (num_accepted 1 -> k+1), step 1's tampered ``d_2``
    leaves ``a == 1`` (-> 2), step 2 runs the plain drafter (the random tiny model accepts
    nothing), step 3's tampered ``d_3`` leaves ``a == 2``. Every step's committed state
    (lattice row ``lattice[a]``, conv windows at ``[a : a + window]``) and the verification's
    row-0 logits equal the straight chain's.
    """
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        seq, accepts, gaps, log0, bonus = _chain(*args, ids, gen, 4, oracle_steps={0, 1, 3},
                                                 tamper_steps={1: (1, TAMPER), 3: (2, TAMPER)}, lay=lay)
    finally:
        ctx.__exit__(None, None, None)
    state_worst = [max(g) for g in gaps]
    print(f"MTP accept-path acceptance counts: {accepts}, state gaps {[f'{g:.3e}' for g in state_worst]}, "
          f"row-0 logits gaps {[f'{g:.3e}' for g in log0]}, bonus gaps {[f'{g:.3e}' for g in bonus]}")
    assert accepts[0] == lay.k, f"the oracle step did not commit k drafts: {accepts}"
    assert accepts[1] == 1, f"the tampered step did not commit exactly one draft: {accepts}"
    assert accepts[3] == 2, f"the second tampered step did not commit two drafts: {accepts}"
    stream = seq[len(ids):]
    assert stream == gen[:len(stream)], f"stream {stream} is not the reference prefix"
    for i, (g, l0) in enumerate(zip(state_worst, log0)):
        assert g <= FP32_TOL, f"step {i} committed state diverges from the straight chain: {g:.3e}"
        assert l0 <= FP32_TOL, f"step {i} row-0 logits diverge from the straight chain: {l0:.3e}"
        assert bonus[i] <= FP32_TOL, f"step {i} bonus logits diverge from the straight chain: {bonus[i]:.3e}"


def test_every_accepted_count_matches_the_plain_path(tiny_checkpoint, spec_k):
    """Every accepted count ``j`` in ``0..k``: the stream, the committed state and the next steps' reads.

    One three-step chain per ``j``. Step 0 feeds the reference drafts with draft ``j + 1`` replaced by a
    wrong token (``j = k``: none replaced), so the verification rejects at position ``j + 1`` and commits
    exactly ``j`` drafts; its committed state is lattice row ``j``. Step 1 loads that row
    (``num_accepted = j + 1``, every load offset ``0..k``) and accepts all ``k`` reference drafts, over
    the rows the rejected drafts of step 0 wrote. Step 2 loads the last lattice row (``num_accepted =
    k + 1``) with the drafter's own drafts. At every step the committed stream is the non-speculative
    stream's prefix, and the committed GDN/PLE state, the row-0 logits and the bonus logits equal the
    plain single-token chain's after the same tokens.
    """
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        model, _draft, vllm_config, eos, atts = rest[0], rest[1], rest[2], rest[3], rest[4]
        reference = _plain_reference(model, vllm_config, eos, ids, gen, atts, lay.straight_slot)
        for j in range(lay.k + 1):
            seq, accepts, gaps, log0, bonus = _chain(*rest, ids, reference, 3, oracle_steps={0, 1},
                                                     tamper_steps={0: (j, TAMPER)} if j < lay.k else None, lay=lay)
            state_worst = [max(g) for g in gaps]
            print(f"k={lay.k} j={j}: accepts {accepts}, state gaps {[f'{g:.3e}' for g in state_worst]}, row-0 "
                  f"gaps {[f'{g:.3e}' for g in log0]}, bonus gaps {[f'{g:.3e}' for g in bonus]}")
            assert accepts[0] == j, f"j={j}: step 0 committed {accepts[0]} drafts"
            assert accepts[1] == lay.k, f"j={j}: step 1 did not commit all {lay.k} reference drafts: {accepts}"
            stream = seq[len(ids):]
            assert len(stream) >= j + lay.k + 4, f"j={j}: the chain committed only {len(stream)} tokens"
            assert stream == reference[:len(stream)], (f"j={j}: stream {stream} is not the non-speculative "
                                                       f"stream {reference[:len(stream)]}")
            for i, (g, l0, b) in enumerate(zip(state_worst, log0, bonus)):
                assert g <= FP32_TOL, f"j={j} step {i}: committed state diverges from the plain chain: {g:.3e}"
                assert l0 <= FP32_TOL, f"j={j} step {i}: row-0 logits diverge from the plain chain: {l0:.3e}"
                assert b <= FP32_TOL, f"j={j} step {i}: bonus logits diverge from the plain chain: {b:.3e}"
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.parametrize("length", [7, 100])
def test_other_reference_prompts_keep_the_stream_and_the_state(tiny_checkpoint, spec_k, length):
    """The 7- and 100-token reference prompts: all k drafts accepted, the first draft rejected, all k accepted.

    The tests above run the 33-token prompt. Here the same parity holds on two more committed prompts: the
    stream of ``2 k + 4`` tokens is the HF reference's, and every step's committed state, row-0 logits and
    bonus logits equal the plain single-token chain's.
    """
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, length)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        seq, accepts, gaps, log0, bonus = _chain(*rest, ids, gen, 3, oracle_steps={0, 1, 2},
                                                 tamper_steps={1: (0, TAMPER)}, lay=lay)
    finally:
        ctx.__exit__(None, None, None)
    assert accepts == [lay.k, 0, lay.k], f"prompt of {length} tokens: acceptance {accepts}"
    stream = seq[len(ids):]
    assert len(stream) == 2 * lay.k + 4 and stream == gen[:len(stream)], (
        f"prompt of {length} tokens: stream {stream} is not the reference prefix {gen[:len(stream)]}")
    worst = max([max(g) for g in gaps] + log0 + bonus)
    print(f"prompt of {length} tokens, k={lay.k}: accepts {accepts}, worst gap {worst:.3e}")
    assert worst <= FP32_TOL, f"prompt of {length} tokens: a step diverges from the plain chain: {worst:.3e}"


def test_forced_rejection_rolls_back(tiny_checkpoint, spec_k):
    """A tampered ``d_2`` after a k-draft commit rejects the tail; the next step rolls back exactly.

    Step 0 (oracle) commits ``a == k``, step 1 (tampered ``d_2``) commits ``a == 1``; the
    rejection step 2 runs with ``num_accepted = 2`` (loading ``idx[1]``, the state of the last
    accepted draft - its verification rewrites the rejected rows ``idx[2..k]``) and must match
    the straight decode continuation.
    """
    lay = StepLayout(spec_k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        seq, accepts, gaps, log0, bonus = _chain(*args, ids, gen, 3, oracle_steps={0, 1},
                                                 tamper_steps={1: (1, TAMPER)}, lay=lay)
    finally:
        ctx.__exit__(None, None, None)
    print(f"MTP forced-rejection acceptance counts: {accepts}, rollback logits {log0[2]:.3e} / "
          f"bonus {bonus[2]:.3e}, per-step state gaps {[f'{max(g):.3e}' for g in gaps]}")
    assert accepts[0] == lay.k, f"the oracle step did not commit k drafts: {accepts}"
    assert accepts[1] == 1, f"the tampered draft was not the first rejection: {accepts}"
    assert log0[2] <= FP32_TOL, f"the rollback step's row-0 logits diverge from the straight chain: {log0[2]:.3e}"
    assert bonus[2] <= FP32_TOL, f"the rollback step's bonus diverges from the straight chain: {bonus[2]:.3e}"
    worst = max(max(g) for g in gaps)
    assert worst <= FP32_TOL, f"state diverged from the straight chain after the rollback: {worst:.3e}"
    stream = seq[len(ids):]
    assert stream == gen[:len(stream)], f"the rolled-back stream diverged: {stream}"


def test_sparse_2600_prompt_step(tiny_checkpoint, spec_k):
    """Two MTP steps from position 2600 in SPARSE mode: picks, logits, states and side caches stay exact.

    Step 0 runs rows ``2600 .. 2600 + k`` with a wrong third draft: two drafts are committed, and the
    rejected rows have already written their raw keys into the ring and, the row at 2603, the compressed
    row of block 650. Step 1 starts at 2603 and accepts all ``k`` reference drafts: its row 0 completes
    block 650 from the ring rows of 2600..2602, written one step earlier, while its own rows write ``k``
    ring rows ahead in the same write-then-score pass. That holds only while the ring has at least
    ``k + 4`` rows per slot (``ring_rows(4, k)``: 8, 8 and 12 for k = 3, 4, 5). After each step the bonus
    logits, the GDN/PLE state, the ring rows of the committed positions and the compressed rows of every
    block the committed tokens complete equal the plain single-token chain's.
    """
    lay = StepLayout(spec_k)
    K = lay.k
    prompt = _load_prompt(tiny_checkpoint, LONG_PROMPT)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE,
                   speculative_config=lay.mtp,
                   num_slots=lay.num_slots) as (vllm_config, model):
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
        _target_prefill(model, vllm_config, eos, ids, lay.straight_slot, atts, reset=False)

        qsa = [m for m in model.modules() if getattr(getattr(m, "indexer", None), "key_ring", None) is not None]
        ring_len = qsa[0].indexer.key_ring.shape[1]
        ratio = int(qsa[0].indexer.compress_ratio)
        n = 1
        for step, (tamper, want_a) in enumerate((((2, TAMPER), 2), (None, K))):
            old_p = p_last
            n, seq, multi, a, bonus, logits = _mtp_step(model, draft, vllm_config, seq, p_last, n, eos, atts,
                                                        draft_atts, multi, oracle=gen[len(seq) - len(ids):],
                                                        tamper=tamper, lay=lay)
            assert a == want_a, f"step {step}: the verification committed {a} drafts, expected {want_a}"
            p_last = len(seq) - 1
            assert bonus == gen[p_last - len(ids)], f"step {step}: MTP bonus {bonus} != reference token"
            # The ssm state of the committed row lives at lattice[a]; the key ring rows of the step are
            # position-addressed (ring[slot, pos % R]) inside the request's slot (the lattice's column 0),
            # so the ring rows the verification wrote at old_p..old_p+a are compared, modulo the ring
            # length R, at slot lattice[0]. The compressed rows are addressed by position alone, so the
            # plain chain rewrites the rows the verification wrote: they are copied first. Only the blocks
            # the committed tokens complete count (a rejected row's block is rewritten by the next step).
            rows = torch.tensor([(old_p + i) % ring_len for i in range(a + 1)], dtype=torch.long)
            blocks = (old_p + a + 1) // ratio
            side = [(m.indexer.key_ring[lay.lattice[0]].index_select(0, rows).clone(),
                     m.indexer.compressed_key_cache[:blocks].clone()) for m in qsa]
            last = _straight_decode(model, vllm_config, eos, seq, old_p, old_p + a + 1, atts, slot=lay.straight_slot)
            logits_gap = float((logits[a] - last).abs().max())
            ssm, conv, ple_g = _state_gaps(model, gdn, ple, a, lay)
            ring_g = comp_g = 0.0
            for (ring_a, comp_a), m in zip(side, qsa):
                ring_g = max(ring_g, float((ring_a - m.indexer.key_ring[lay.straight_slot].index_select(0, rows)
                                            ).abs().max()))
                comp_g = max(comp_g, float((comp_a - m.indexer.compressed_key_cache[:blocks]).abs().max()))
            print(f"MTP sparse step {step} (k={K}, ring rows {ring_len}): a={a} (lattice slot {lay.lattice[a]}), "
                  f"logits {logits_gap:.3e}, ssm {ssm:.3e}, conv {conv:.3e}, ple {ple_g:.3e}, ring {ring_g:.3e}, "
                  f"compressed rows 0..{blocks - 1} {comp_g:.3e}")
            assert logits_gap <= FP32_TOL, f"step {step}: bonus logits diverge from the plain chain: {logits_gap:.3e}"
            assert max(ssm, conv, ple_g) <= FP32_TOL, f"step {step}: committed state diverges"
            assert ring_g <= FP32_TOL, f"step {step}: key ring at the committed positions diverged: {ring_g:.3e}"
            assert comp_g <= FP32_TOL, (f"step {step}: compressed rows of the blocks completed up to position "
                                        f"{old_p + a} diverged: {comp_g:.3e}")
        assert p_last == LONG_PROMPT + 3 + K + 1 and seq[len(ids):] == gen[:K + 5]


def test_mutations_break_the_chain(tiny_checkpoint, spec_k):
    """(a) Tiling the drafter's step-1 hidden changes its drafts; (b) ``num_accepted=k+1`` breaks parity."""
    lay = StepLayout(spec_k)
    K = lay.k
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = rest
        args = (model, draft, vllm_config, eos, atts, draft_atts, gdn, ple)
        # (a) at the prefill state: sample.repeat(1, hc) as the drafter's step-1 hidden changes d_2 .. d_k
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        t0 = int(fp32_logits(model, out[0])[-1].argmax())
        seq0 = ids + [t0]
        p_last = len(seq0) - 1
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        stream = seq0[max(p_last - 2, 0):p_last + 1]
        plain = _draft_tokens(draft, vllm_config, eos, stream, p_last, draft_atts, multi, K)
        mutated = _draft_tokens(draft, vllm_config, eos, stream, p_last, draft_atts, multi, K, mutate=True)
        print(f"MTP drafter mutation: plain={plain} mutated={mutated}")
        assert len(plain) == len(mutated) == K
        assert plain[0] == mutated[0], "the mutation must not touch the step-0 draft"
        assert plain[1:] != mutated[1:], f"drafter drafts unchanged by its hidden mutation: {plain} vs {mutated}"

        # (b) after the oracle commit and the tampered rejection (true num_accepted = 2 for the
        # rollback step), feeding num_accepted = K + 1 into that verification loads idx[K], the
        # state of a rejected token: the step's own logits/state must diverge from the straight
        # chain (the comparison is taken in the step that used the override).
        _, accepts_b, gaps_b, log0_b, _ = _chain(*args, ids, gen, 3, oracle_steps={0, 1},
                                                 tamper_steps={1: (1, TAMPER)}, overrides={2: K + 1}, lay=lay)
        ssm_b = gaps_b[2][0]
        print(f"MTP num_accepted-override mutation: accepts={accepts_b}, row-0 logits gap {log0_b[2]:.3e}, "
              f"ssm gap at lattice[a] {ssm_b:.3e}")
        assert accepts_b[0] == K and accepts_b[1] == 1, f"the base chain of probe (b) is off: {accepts_b}"
        assert log0_b[2] > MUT_TOL or ssm_b > MUT_TOL, (f"the num_accepted={K + 1} override kept parity: "
                                                        f"logits {log0_b[2]:.3e}, ssm {ssm_b:.3e}")
    finally:
        ctx.__exit__(None, None, None)


def _commit_pairs(model_states, pairs, conv_slot: int) -> None:
    """Run the boundary commits ``pairs`` on the GDN and PLE states (the conv rows live in ``conv_slot``)."""
    gdn, ple = model_states
    conv_pairs = [(i, t, conv_slot, d) for (i, t, _s, d) in pairs]
    ts = [t for (_i, t, _s, _d) in pairs]
    for m in gdn:
        commit_ssm(m.kv_cache[1], pairs)
        commit_conv(m.kv_cache[0], conv_pairs, GDN_WINDOW, t_per_pair=ts)
    for m in ple:
        commit_conv(m.kv_cache[0], conv_pairs, PLE_WINDOW, t_per_pair=ts)


def _slot_gap(gdn, ple, slot: int, straight: int) -> tuple[float, float, float]:
    """(ssm, gdn-conv, ple-conv) gaps of the plain-layout state in ``slot`` against the straight chain's."""
    ssm = max(float((m.kv_cache[1][slot] - m.kv_cache[1][straight]).abs().max()) for m in gdn)
    conv = max(float((m.kv_cache[0][slot, 0:GDN_WINDOW] - m.kv_cache[0][straight, 0:GDN_WINDOW]).abs().max())
               for m in gdn)
    ple_g = max(float((m.kv_cache[0][slot, 0:PLE_WINDOW] - m.kv_cache[0][straight, 0:PLE_WINDOW]).abs().max())
                for m in ple)
    return ssm, conv, ple_g


def _plain_snapshot(gdn, ple, slot: int):
    """The plain-layout state of ``slot``, cloned: (ssm rows, GDN conv windows, PLE conv windows)."""
    return ([m.kv_cache[1][slot].clone() for m in gdn], [m.kv_cache[0][slot, 0:GDN_WINDOW].clone() for m in gdn],
            [m.kv_cache[0][slot, 0:PLE_WINDOW].clone() for m in ple])


def _snapshot_gap(a, b) -> float:
    return max(float((x - y).abs().max()) for xs, ys in zip(a, b) for x, y in zip(xs, ys))


def test_align_lattice_step_with_boundary_commit(tiny_checkpoint, spec_k):
    """T8 align variant: a differing load/store lattice plus boundary commits matches the straight chain.

    The chain state is prefilled at ``chain_slot`` (the block of the last computed token), the
    verification stores token 0's state at ``block_slots[0]`` (the block of its own position) and the
    candidate checkpoints at the scratch slots ``lattice[1:]``. With block_size 2 every second accepted
    token closes a position block, the last accepted one among them; their states move into the slot of
    the block each one completes via the recurrent_commit helpers, and every such slot must equal the
    interleaved straight chain's state after that token. Each block has its own slot, as in a real block
    table (two commits into one slot would be one ``index_copy_`` with a repeated row, whose winner torch
    does not define). k = 3: slots 6 and 7, 8, scratch [2, 3, 4], commits at t = 1 and 3.
    """
    lay = StepLayout(spec_k)
    K, T = lay.k, lay.t
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = _dense_context(tiny_checkpoint, lay)
    chain_slot, new_slot = lay.chain_slot, lay.block_slot
    scratch = lay.lattice[1:]
    try:
        out = _target_prefill(model, vllm_config, eos, ids, chain_slot, atts)
        _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts, reset=False)  # attention keys
        logits = fp32_logits(model, out[0])
        seq = list(ids) + [int(logits[-1].argmax())]
        p_last = len(seq) - 1
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, lay.straight_slot, atts, reset=False)

        _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts, multi, K)
        drafts = gen[len(seq) - len(ids):len(seq) - len(ids) + K]  # oracle drafts
        rows = [seq[p_last]] + drafts
        load = torch.tensor([[[chain_slot] + scratch]], dtype=torch.int64)
        store = torch.tensor([[[new_slot] + scratch]], dtype=torch.int64)
        assert not torch.equal(load, store)
        for att in atts:
            att.slot = SPEC_SLOT
        md = _spec_metadata(p_last, 1, lay)
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

        plain = {}  # the straight chain's state after each committed token
        for t in range(a + 1):
            _straight_decode(model, vllm_config, eos, seq, p_last + t, p_last + t + 1, atts, slot=lay.straight_slot)
            plain[t] = _plain_snapshot(gdn, ple, lay.straight_slot)

        # Boundary commits, block_size 2. The synthetic context length makes the LAST accepted token
        # (t = k) complete a block, so the last block's slot ends at the state the straight chain is in.
        ctx_len = p_last + 1 - (T % 2)
        assert (ctx_len + K + 1) % 2 == 0
        block_rows = [[chain_slot] * 17 + lay.block_slots]
        pairs = boundary_commits([ctx_len], [a + 1], 2, store[0], block_rows, load[0])
        expected = [(0, t, int(store[0][0][t]), block_rows[0][(ctx_len + t) // 2])
                    for t in range(T) if (ctx_len + t + 1) % 2 == 0]
        assert [(i, t, s, d) for (i, t, s, d) in pairs] == expected
        assert pairs[-1][1] == K and len(pairs) == (T + 1) // 2
        assert len({d for (_i, _t, _s, d) in pairs}) == len(pairs), f"two commits share a slot: {pairs}"
        if K == 3:
            assert expected == [(0, 1, 2, 7), (0, 3, 4, 8)]
        _commit_pairs((gdn, ple), pairs, int(store[0][0][0]))

        gaps = {t: _snapshot_gap(_plain_snapshot(gdn, ple, d), plain[t]) for (_i, t, _s, d) in pairs}
        ssm_gap, conv_gap, ple_gap = _slot_gap(gdn, ple, pairs[-1][3], lay.straight_slot)
        worst = max([ssm_gap, conv_gap, ple_gap] + list(gaps.values()))
        print(f"align lattice step: a={a}, pairs {pairs}, last block gaps ssm={ssm_gap:.3e} conv={conv_gap:.3e} "
              f"ple={ple_gap:.3e}, per committed token {({t: f'{g:.3e}' for t, g in gaps.items()})}")
        assert worst <= FP32_TOL, f"committed align state diverges from the straight chain: {worst:.3e}"
    finally:
        ctx.__exit__(None, None, None)


def test_non_aligned_boundary_commit_with_equal_lattices(tiny_checkpoint, spec_k):
    """Runner-gate regression: a t >= 1 completion on a step whose load == store lattices.

    Prompt 33, block_size 2: the verification runs right after the prefill with
    context_len 33 (unaligned), and the tampered ``d_3`` leaves ``accepted = 3``, so token
    ``t = 2`` completes a position block. The load and store lattices are equal, so the
    runner's ``lattice_store_differs`` flag is False and the kernels write every state
    into the shared lattice; only the boundary-commit payload (set unconditionally under
    prefix caching, review item 1) moves token 2's checkpoint into the completed block's
    slot. That slot must equal the interleaved straight chain's state - the case the
    flag-gated runner dropped (the align variant above is aligned at every step).
    """
    lay = StepLayout(spec_k)
    K, T = lay.k, lay.t
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, model, draft, vllm_config, eos, atts, draft_atts, gdn, ple = _dense_context(tiny_checkpoint, lay)
    block_slot = lay.block_slot
    try:
        out = _target_prefill(model, vllm_config, eos, ids, SPEC_SLOT, atts)
        logits = fp32_logits(model, out[0])
        t0 = int(logits[-1].argmax())
        assert t0 == gen[0], f"prefill pick {t0} != reference {gen[0]}"
        seq = list(ids) + [t0]
        p_last = len(seq) - 1  # 33: the first sampled token, row 0 (33 tokens computed)
        multi = out[1][0].view(-1, out[1][0].shape[-1])[len(ids) - 1]
        _target_prefill(model, vllm_config, eos, ids, lay.straight_slot, atts, reset=False)

        _draft_tokens(draft, vllm_config, eos, seq[max(p_last - 2, 0):p_last + 1], p_last, draft_atts, multi, K)
        drafts = gen[1:1 + K]
        drafts[2] = TAMPER  # tampered d_3: the step accepts exactly two drafts
        rows = [seq[p_last]] + drafts
        for att in atts:
            att.slot = SPEC_SLOT
        md = _spec_metadata(p_last, 1, lay)  # load == store == the lattice: the flag stays False
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

        # Straight decodes of the committed input rows (gen[0] .. gen[2]).
        _straight_decode(model, vllm_config, eos, seq, p_last, p_last + a + 1, atts, slot=lay.straight_slot)

        # Boundary commits, block_size 2: context_len 33 is unaligned, so t = 0 completes the block
        # of position 33 (row 16, SPEC_SLOT) and t = 2 completes the block of the last accepted token.
        ctx_len = p_last  # 33 computed tokens (the prompt); row 0 = the first sampled token
        assert ctx_len % 2 == 1
        block_rows = [[SPEC_SLOT] * 17 + [block_slot]]
        pairs = boundary_commits([ctx_len], [a + 1], 2, md.store_indices_tensor[0], block_rows,
                                 md.load_indices_tensor[0])
        assert [(i, t, s, d) for (i, t, s, d) in pairs] == [(0, 0, lay.lattice[0], SPEC_SLOT),
                                                            (0, 2, lay.lattice[2], block_slot)]
        _commit_pairs((gdn, ple), pairs, int(md.store_indices_tensor[0][0][0]))

        ssm_gap, conv_gap, ple_gap = _slot_gap(gdn, ple, block_slot, lay.straight_slot)
        worst = max(ssm_gap, conv_gap, ple_gap)
        print(f"non-aligned boundary commit (flag False): a={a}, pairs {pairs}, gaps ssm={ssm_gap:.3e} "
              f"conv={conv_gap:.3e} ple={ple_gap:.3e}")
        assert worst <= FP32_TOL, f"flag-False boundary commit diverges from the straight chain: {worst:.3e}"
    finally:
        ctx.__exit__(None, None, None)
