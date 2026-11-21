# SPDX-License-Identifier: Apache-2.0
"""TP1 decode-path parity for the Flash-Next model on CPU (M0-T7c).

One prefill followed by single-token decode steps that carry the GDN, PLE and attention
state in their cache slots must reproduce HF's tokens (teacher-forced greedy) and the same
logits as a fresh prefill of the same sequence. :class:`CachedSdpaAttention` stands in for
the paged-attention backend with one K/V cache per slot, so a decode step attends over the
whole sequence and two sequences can interleave without seeing each other.

The negative control (GDN+PLE state wiped right before the last decode step) proves the
parity checks would catch a decode path that drops state. Based on the CTO probe
``spikes/flash-next-parity/decode_probe.py``.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.

Every test runs twice: on the torch GDN decode path (``VLLM_HPU_GDN_CORE_DECODE=0``) and through
the fused op (``1``, the default), and asserts which path ran.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402

from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode  # noqa: E402
import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, dense_reference_prompts, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)

FP32_DECODE_VS_PREFILL_TOL = 1e-5
BF16_DECODE_VS_PREFILL_TOL = 3e-2
LOST_STATE_MIN_GAP = 0.1
INTERLEAVE_TOL = 1e-6


@pytest.fixture(autouse=True, params=[GdnCoreDecodeMode.OFF, GdnCoreDecodeMode.FUSED], ids=["gdn_off", "gdn_fused"])
def gdn_decode_path(request, monkeypatch):
    """Run each test on the torch GDN decode path and on gdn_core_decode, and check which one ran."""
    mode = request.param
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", str(int(mode)))
    real = qwen3_5.gdn_core_decode
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(qwen3_5, "gdn_core_decode", counting)
    yield mode
    if mode == GdnCoreDecodeMode.FUSED:
        assert calls, "mode FUSED never called gdn_core_decode: the test did not exercise the fused path"
    else:
        assert not calls, f"mode OFF called gdn_core_decode {len(calls)} times"


class CachedSdpaAttention(torch.nn.Module):
    """Causal SDPA over all K/V seen for the current slot (set `.slot` before each forward)."""

    def __init__(self, qsa_layer):
        super().__init__()
        self.h, self.kv = qsa_layer.num_heads, qsa_layer.num_kv_heads
        self.d, self.sc = qsa_layer.head_dim, qsa_layer.scaling
        self.slot = 1
        self.cache = {}  # slot -> (K, V), each (tokens, kv_heads * head_dim)

    def reset(self):
        self.cache = {}

    def forward(self, q, k, v):
        n = q.shape[0]
        if self.slot in self.cache:
            k0, v0 = self.cache[self.slot]
            k, v = torch.cat([k0, k]), torch.cat([v0, v])
        self.cache[self.slot] = (k, v)
        L = k.shape[0]
        qh = q.view(n, self.h, self.d).transpose(0, 1)
        kh = k.view(L, self.kv, self.d).transpose(0, 1).repeat_interleave(self.h // self.kv, 0)
        vh = v.view(L, self.kv, self.d).transpose(0, 1).repeat_interleave(self.h // self.kv, 0)
        mask = torch.ones(n, L, dtype=torch.bool).tril(L - n)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=self.sc)
        return out.transpose(0, 1).reshape(n, -1)


def decode_metadata(slot):
    return SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=torch.tensor([slot]),
        store_indices_tensor=torch.tensor([slot]),
        query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32),
        has_initial_states_p=None,
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        dflash_full_query=False,
    )


def install_cached_attention(model):
    """Replace `.attn` on every QSA layer with a fresh :class:`CachedSdpaAttention`."""
    atts = []
    for m in model.modules():
        if isinstance(m, HpuQwen4ExpQSAAttention):
            m.attn = CachedSdpaAttention(m)
            atts.append(m.attn)
    return atts


def prefill(model, vllm_config, seq, slot, eos):
    """Prefill ``seq`` into state slot ``slot``; returns fp32 logits (len(seq), vocab)."""
    n = len(seq)
    for att in model_attns(model):
        att.slot = slot
    with torch.no_grad(), set_forward_context(prefill_metadata(n, slot), vllm_config):
        hidden = model(input_ids=torch.tensor(seq), positions=torch.arange(n), **ple_kwargs(n, eos))
    return fp32_logits(model, hidden)


def decode(model, vllm_config, seq, pos, slot, eos):
    """One decode step of token ``seq[pos]`` with the state in ``slot``; returns (1, vocab)."""
    ctx = (seq[pos - 2] if pos >= 2 else eos, seq[pos - 1] if pos >= 1 else eos)
    for att in model_attns(model):
        att.slot = slot
    with torch.no_grad(), set_forward_context(decode_metadata(slot), vllm_config):
        hidden = model(input_ids=torch.tensor([seq[pos]]),
                       positions=torch.tensor([pos]),
                       query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                       ngram_context=torch.tensor([ctx], dtype=torch.int32))
    return fp32_logits(model, hidden)


def model_attns(model):
    return [m.attn for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]


def load_reference(tiny_checkpoint):
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))


def prompt_of(ref, length):
    for p in dense_reference_prompts(ref):
        if len(p["prompt_ids"]) == length:
            return p
    raise KeyError(f"no prompt of length {length}")


def run_decode_chain(model, vllm_config, atts, prompt, slot, eos):
    """Prefill ``prompt`` into ``slot``, then decode the reference tokens teacher-forced.

    Returns ``(picks, last_decode_logits, full_seq)``: the greedy pick of each of the 16
    steps, the fp32 logits of the final decode step, and the sequence the fresh-prefill
    comparison prefills (prompt plus reference tokens, minus the last one).
    """
    reset_cpu_states(model)
    for att in atts:
        att.reset()
    seq = list(prompt["prompt_ids"])
    gen = list(prompt["generated_ids"])
    lg = prefill(model, vllm_config, seq, slot, eos)[-1]
    picks = []
    for i in range(len(gen)):
        picks.append(int(lg.argmax()))
        tf = gen[i]  # teacher-force HF's token so one wrong pick does not derail later steps
        seq.append(tf)
        if i == len(gen) - 1:
            break
        lg = decode(model, vllm_config, seq, len(seq) - 1, slot, eos)[-1]
    return picks, lg, seq[:-1]


def test_decode_fp32_matches_reference(tiny_checkpoint):
    """fp32: every teacher-forced argmax equals HF's token, and the last decode step equals a
    fresh prefill of the same sequence within 1e-5."""
    ref = load_reference(tiny_checkpoint)
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        for prompt in dense_reference_prompts(ref):
            n = len(prompt["prompt_ids"])
            picks, last_decode, full = run_decode_chain(model, vllm_config, atts, prompt, 1, eos)
            for i, (picked, expected) in enumerate(zip(picks, prompt["generated_ids"])):
                assert picked == expected, (f"prompt of {n} tokens, step {i}: teacher-forced argmax {picked} "
                                            f"!= HF generated_ids[{i}] {expected}")
            reset_cpu_states(model)
            for att in atts:
                att.cache.pop(1, None)
            prefill_logits = prefill(model, vllm_config, full, 1, eos)[-1]
            gap = (prefill_logits - last_decode).abs().max().item()
            assert gap <= FP32_DECODE_VS_PREFILL_TOL, (
                f"prompt of {n} tokens: last decode step vs fresh prefill of the same {len(full)}-token "
                f"sequence: max |delta logit| = {gap:.3e} > {FP32_DECODE_VS_PREFILL_TOL}")


def test_decode_bf16_close_to_prefill(tiny_checkpoint):
    """bf16: same flow; the last decode step is within 3e-2 of a fresh prefill (no argmax assertion)."""
    ref = load_reference(tiny_checkpoint)
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        for prompt in dense_reference_prompts(ref):
            n = len(prompt["prompt_ids"])
            _, last_decode, full = run_decode_chain(model, vllm_config, atts, prompt, 1, eos)
            reset_cpu_states(model)
            for att in atts:
                att.cache.pop(1, None)
            prefill_logits = prefill(model, vllm_config, full, 1, eos)[-1]
            gap = (prefill_logits - last_decode).abs().max().item()
            assert gap <= BF16_DECODE_VS_PREFILL_TOL, (
                f"prompt of {n} tokens, bf16: last decode step vs fresh prefill of the same {len(full)}-token "
                f"sequence: max |delta logit| = {gap:.3e} > {BF16_DECODE_VS_PREFILL_TOL}")


def test_lost_state_is_detected(tiny_checkpoint):
    """Negative control: wiping the GDN+PLE state before the last decode step moves the logits
    by >= 0.1 vs a fresh prefill, so the parity checks above would catch a state-dropping decode."""
    ref = load_reference(tiny_checkpoint)
    prompt = prompt_of(ref, 7)
    seq = list(prompt["prompt_ids"])
    n = len(seq)
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        reset_cpu_states(model)
        for att in atts:
            att.reset()
        fresh = prefill(model, vllm_config, seq, 1, eos)[-1]
        # prefill the first 6 tokens from clean slots, wipe GDN+PLE state (the attention cache
        # keeps its K/V), then decode the 7th token: its recurrent/PLE state is gone
        reset_cpu_states(model)
        for att in atts:
            att.cache.pop(1, None)
        prefill(model, vllm_config, seq[:-1], 1, eos)
        reset_cpu_states(model)
        wiped = decode(model, vllm_config, seq, n - 1, 1, eos)[-1]
        gap = (wiped - fresh).abs().max().item()
        assert gap >= LOST_STATE_MIN_GAP, (
            f"prompt of {n} tokens: wiping the GDN+PLE state before the last decode step moved the logits by "
            f"only {gap:.3e} < {LOST_STATE_MIN_GAP}; the parity checks would not detect a dropped state")


def test_two_slots_interleaved(tiny_checkpoint):
    """fp32: a 7-token prompt in slot 1 and a 33-token prompt in slot 2, prefilled then alternating
    8 decode steps A, B, A, B, ...; every step equals a solo run of that prompt in slot 1 within 1e-6."""
    ref = load_reference(tiny_checkpoint)
    steps = 8
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)

        def solo_decode_steps(prompt):
            """Prefill in slot 1, decode `steps` tokens teacher-forced, return the decode logits."""
            reset_cpu_states(model)
            for att in atts:
                att.reset()
                att.slot = 1
            seq = list(prompt["prompt_ids"])
            prefill(model, vllm_config, seq, 1, eos)
            logits = []
            for i in range(steps):
                seq.append(prompt["generated_ids"][i])
                logits.append(decode(model, vllm_config, seq, len(seq) - 1, 1, eos)[-1])
            return logits

        prompt_a, prompt_b = prompt_of(ref, 7), prompt_of(ref, 33)
        solo_a, solo_b = solo_decode_steps(prompt_a), solo_decode_steps(prompt_b)

        reset_cpu_states(model)
        for att in atts:
            att.reset()
        seq_a, seq_b = list(prompt_a["prompt_ids"]), list(prompt_b["prompt_ids"])
        prefill(model, vllm_config, seq_a, 1, eos)
        prefill(model, vllm_config, seq_b, 2, eos)
        for i in range(steps):
            for slot, prompt, seq, solo_logits in ((1, prompt_a, seq_a, solo_a), (2, prompt_b, seq_b, solo_b)):
                seq.append(prompt["generated_ids"][i])
                lg = decode(model, vllm_config, seq, len(seq) - 1, slot, eos)[-1]
                gap = (lg - solo_logits[i]).abs().max().item()
                assert gap <= INTERLEAVE_TOL, (
                    f"interleaved slot {slot}, prompt of {len(prompt['prompt_ids'])} tokens, step {i}: "
                    f"logits vs solo run in slot 1: max |delta logit| = {gap:.3e} > {INTERLEAVE_TOL}")
