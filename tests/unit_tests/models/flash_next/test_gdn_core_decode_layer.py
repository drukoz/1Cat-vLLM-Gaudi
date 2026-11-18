# SPDX-License-Identifier: Apache-2.0
"""Layer-level tests for the fused GDN core decode (T3a of track RUK-2812).

With ``VLLM_HPU_GDN_CORE_DECODE=1`` every non-speculative decode step of
``HPUGatedDeltaNetAttention`` runs ``torch.ops.flashinfer_gaudi.gdn_core_decode`` instead of the
conv + recurrence + ``self.norm`` chain. On the CPU tiny Flash-Next model the 8 teacher-forced
decode rows must match mode 0 (fp32 to 1e-5, bf16 to 5e-2 with equal argmax), a load != store step
must write only the store slot, and a layer whose op is unavailable must log exactly one warning
and run the torch path. The negative control (the op's output replaced by zeros) proves the parity
checks would catch an op output the layer ignores.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402

from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode  # noqa: E402
import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402
from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.test_chunked_prefill_parity import enable_slot_reuse  # noqa: E402
from tests.unit_tests.models.flash_next.test_decode_parity import (  # noqa: E402
    decode, install_cached_attention, prefill)

FP32_TOL = 1e-5
BF16_TOL = 5e-2
FALLBACK_TOL = 1e-6
ZERO_OUTPUT_MIN_GAP = 0.1

BOS_TOKEN_ID = 248044
PREFILL_LEN = 40
DECODE_STEPS = 8
NUM_TOKENS = PREFILL_LEN + DECODE_STEPS  # 48: tokens 0..39 prefill, positions 40..47 decoded
SLOT = 1
NUM_SLOTS = 8
ATTN_SLOT = 1  # the CachedSdpaAttention K/V key, kept apart from the state slots under test
LOAD_SLOT, STORE_SLOT = 2, 3
GDN_LAYERS = 3


def tokens() -> list[int]:
    """The deterministic 48-token sequence (BOS plus 47 ids drawn with seed 11)."""
    drawn = torch.randint(1000, 20000, (NUM_TOKENS - 1,), generator=torch.Generator().manual_seed(11))
    return [BOS_TOKEN_ID] + drawn.tolist()


def gdn_layers(model) -> list:
    """Every ``HPUGatedDeltaNetAttention`` module of the model (3 on the tiny checkpoint)."""
    return [m for m in model.modules() if isinstance(m, HPUGatedDeltaNetAttention)]


def set_mode(model, mode: GdnCoreDecodeMode) -> None:
    """Force the core-decode mode on every GDN layer and re-arm the one-shot precondition check."""
    for module in gdn_layers(model):
        module.gdn_core_decode_mode = mode
        module._gdn_core_ok = None


class CountingOp:
    """Monkeypatch of ``qwen3_5.gdn_core_decode`` that counts calls and forwards to the real op."""

    def __init__(self, real) -> None:
        self.real = real
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real(*args, **kwargs)


class RecordingLogger:
    """Stand-in for ``qwen3_5.logger`` that records the raw ``warning(*args)`` and ``debug(*args)`` tuples."""

    def __init__(self) -> None:
        self.warnings: list[tuple] = []
        self.debugs: list[tuple] = []

    def warning(self, *args) -> None:
        self.warnings.append(args)

    def debug(self, *args) -> None:
        self.debugs.append(args)


def run(model, vllm_config, eos, atts, slot: int = SLOT) -> torch.Tensor:
    """One run: reset, prefill tokens ``0..39`` into ``slot``, decode positions ``40..47``.

    Returns the 8 decode logit rows, ``(DECODE_STEPS, vocab)`` fp32 (teacher-forced: every step
    feeds the token the reference sequence has at that position).
    """
    seq = tokens()
    reset_cpu_states(model)
    for att in atts:
        att.reset()
    prefill(model, vllm_config, seq[:PREFILL_LEN], slot, eos)
    rows = [decode(model, vllm_config, seq, pos, slot, eos) for pos in range(PREFILL_LEN, NUM_TOKENS)]
    return torch.cat(rows, dim=0)


def prefill_into(model, vllm_config, atts, token_ids, metadata, eos) -> torch.Tensor:
    """Prefill ``token_ids`` under ``metadata`` with the attention stand-in pinned to ``ATTN_SLOT``."""
    for att in atts:
        att.slot = ATTN_SLOT
    n = len(token_ids)
    with torch.no_grad(), set_forward_context(metadata, vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids), positions=torch.arange(n), **ple_kwargs(n, eos))
    return fp32_logits(model, hidden)


def decode_step(model, vllm_config, atts, seq, pos, metadata, eos) -> torch.Tensor:
    """One decode step of ``seq[pos]`` under ``metadata`` (the body of ``test_decode_parity.decode``)."""
    ctx = (seq[pos - 2] if pos >= 2 else eos, seq[pos - 1] if pos >= 1 else eos)
    for att in atts:
        att.slot = ATTN_SLOT
    with torch.no_grad(), set_forward_context(metadata, vllm_config):
        hidden = model(input_ids=torch.tensor([seq[pos]]),
                       positions=torch.tensor([pos]),
                       query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                       ngram_context=torch.tensor([ctx], dtype=torch.int32))
    return fp32_logits(model, hidden)


def slot_state_rows(modules, slot: int) -> list[torch.Tensor]:
    """``kv_cache[k][slot]`` for every state tensor of every module (GDN layers: conv, ssm)."""
    return [module.kv_cache[k][slot] for module in modules for k in range(len(module.kv_cache))]


def test_mode_defaults_fused(tiny_checkpoint, monkeypatch):
    """With the variable unset, every GDN layer starts in GdnCoreDecodeMode.FUSED."""
    monkeypatch.delenv("VLLM_HPU_GDN_CORE_DECODE", raising=False)
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layers = gdn_layers(model)
        assert len(layers) == GDN_LAYERS
        for module in layers:
            assert module.gdn_core_decode_mode == GdnCoreDecodeMode.FUSED


def test_flag_reaches_layers(tiny_checkpoint, monkeypatch):
    """VLLM_HPU_GDN_CORE_DECODE=0 set before the build makes every GDN layer OFF (envs is lazy)."""
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "0")
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layers = gdn_layers(model)
        assert len(layers) == GDN_LAYERS
        for module in layers:
            assert module.gdn_core_decode_mode == GdnCoreDecodeMode.OFF


def test_fused_matches_torch_path_fp32(tiny_checkpoint, monkeypatch):
    """fp32: the fused decode rows match the torch path, the op runs 3*8 times, and the states match."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        layers = gdn_layers(model)
        op = CountingOp(qwen3_5.gdn_core_decode)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode", op)

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        assert op.calls == 0, "the OFF run must not call the fused op"
        conv_ref = [module.kv_cache[0][SLOT].clone() for module in layers]
        ssm_ref = [module.kv_cache[1][SLOT].clone() for module in layers]

        set_mode(model, GdnCoreDecodeMode.FUSED)
        fused_rows = run(model, vllm_config, eos, atts)
        assert op.calls == GDN_LAYERS * DECODE_STEPS, f"expected {GDN_LAYERS * DECODE_STEPS} op calls, got {op.calls}"

        diff = (fused_rows - off_rows).abs().max().item()
        assert diff <= FP32_TOL, f"fp32 decode logits differ by {diff:.3e} > {FP32_TOL:.0e}"
        for idx, (module, conv, ssm) in enumerate(zip(layers, conv_ref, ssm_ref)):
            # Layer 0 sees the input embedding, so its conv state (raw mixed_qkv) must be bit-exact.
            # Layers 1-2 consume layer 0's fused output, whose <=1e-5 drift propagates into their conv inputs.
            if idx == 0:
                assert torch.equal(module.kv_cache[0][SLOT], conv), "layer 0 conv state must be bit-exact"
            else:
                assert torch.allclose(module.kv_cache[0][SLOT], conv, atol=FP32_TOL, rtol=0), (
                    f"layer {idx}: conv state of slot {SLOT} changed beyond {FP32_TOL:.0e}")
            assert torch.allclose(module.kv_cache[1][SLOT], ssm, atol=FP32_TOL, rtol=0), (
                f"layer {idx}: ssm state of slot {SLOT} differs by more than {FP32_TOL:.0e}")


def test_fused_matches_torch_path_bf16(tiny_checkpoint, monkeypatch):
    """bf16: same argmax token on every row and logits within 5e-2 of the torch path."""
    with cpu_model(tiny_checkpoint, fp32=False) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        op = CountingOp(qwen3_5.gdn_core_decode)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode", op)

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        assert op.calls == 0
        set_mode(model, GdnCoreDecodeMode.FUSED)
        fused_rows = run(model, vllm_config, eos, atts)
        assert op.calls == GDN_LAYERS * DECODE_STEPS

        diff = (fused_rows - off_rows).abs().max().item()
        assert diff <= BF16_TOL, f"bf16 decode logits differ by {diff:.3e} > {BF16_TOL:.0e}"
        for row in range(DECODE_STEPS):
            off_pick = int(off_rows[row].argmax())
            fused_pick = int(fused_rows[row].argmax())
            assert off_pick == fused_pick, f"row {row}: argmax {fused_pick} != {off_pick}"


def test_fused_load_ne_store(tiny_checkpoint):
    """A load 2 / store 3 decode step writes only slot 3; slot 2 keeps the post-prefill state."""
    seq = tokens()
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        layers = gdn_layers(model)

        def step(mode: GdnCoreDecodeMode):
            set_mode(model, mode)
            reset_cpu_states(model)
            for att in atts:
                att.reset()
            prefill_into(model, vllm_config, atts, seq[:PREFILL_LEN], prefill_metadata(PREFILL_LEN, LOAD_SLOT), eos)
            after_prefill = [row.clone() for row in slot_state_rows(layers, LOAD_SLOT)]
            rows = decode_step(model, vllm_config, atts,
                               seq,
                               PREFILL_LEN,
                               decode_metadata(STORE_SLOT, PREFILL_LEN, load_slot=LOAD_SLOT, store_slot=STORE_SLOT),
                               eos)
            return rows, after_prefill

        off_rows, _ = step(GdnCoreDecodeMode.OFF)
        off_store = [row.clone() for row in slot_state_rows(layers, STORE_SLOT)]
        fused_rows, after_prefill = step(GdnCoreDecodeMode.FUSED)

        diff = (fused_rows - off_rows).abs().max().item()
        assert diff <= FP32_TOL, f"fp32 load/store-move logits differ by {diff:.3e} > {FP32_TOL:.0e}"
        load_rows = slot_state_rows(layers, LOAD_SLOT)
        store_rows = slot_state_rows(layers, STORE_SLOT)
        for idx, (now, before) in enumerate(zip(load_rows, after_prefill)):
            assert torch.equal(now, before), (f"slot {LOAD_SLOT} state tensor {idx} changed in the fused step")
        for idx, (now, ref) in enumerate(zip(store_rows, off_store)):
            # conv tensors are the even indices (idx // 2 = layer). Only layer 0's conv input is
            # bit-exact; layers 1-2 inherit layer 0's fused-output drift, so they keep atol=1e-5.
            if idx == 0:
                assert torch.equal(now, ref), f"conv state of slot {STORE_SLOT} (tensor {idx}) must be bit-exact"
            elif idx % 2 == 0:
                assert torch.allclose(now, ref, atol=FP32_TOL, rtol=0), (
                    f"conv state of slot {STORE_SLOT} (tensor {idx}) differs from the torch run by more than {FP32_TOL:.0e}")
            else:
                assert torch.allclose(now, ref, atol=FP32_TOL, rtol=0), (
                    f"ssm state of slot {STORE_SLOT} (tensor {idx}) differs from the torch run by more than {FP32_TOL:.0e}")
        for slot in (0, 1, 4, 5, 6, 7):
            for idx, tensor in enumerate(slot_state_rows(layers, slot)):
                assert not bool(tensor.any()), f"slot {slot} state tensor {idx} was written (expected all zeros)"


def test_fallback_when_op_unavailable(tiny_checkpoint, monkeypatch):
    """No kernel for the device: one warning per layer, the torch path, and the OFF-run logits."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        op = CountingOp(qwen3_5.gdn_core_decode)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode", op)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode_available", lambda device: False)
        log = RecordingLogger()
        monkeypatch.setattr(qwen3_5, "logger", log)

        set_mode(model, GdnCoreDecodeMode.FUSED)
        fused_rows = run(model, vllm_config, eos, atts)

        assert op.calls == 0, "the fallback path must not call the fused op"
        assert len(log.warnings) == GDN_LAYERS, (f"expected one warning per layer ({GDN_LAYERS}), "
                                                f"got {len(log.warnings)}")
        for args in log.warnings:
            message = args[0] % args[1:]
            assert "is off for" in message, f"warning {message!r} is missing the layer line"
            assert "no gdn_core_decode kernel for cpu" in message, f"warning {message!r} names the wrong reason"
        for module in gdn_layers(model):
            assert module._gdn_core_ok is False

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        diff = (fused_rows - off_rows).abs().max().item()
        assert diff <= FALLBACK_TOL, f"the fallback run differs from mode OFF by {diff:.3e} > {FALLBACK_TOL:.0e}"


def test_missing_extension_is_logged_at_debug(tiny_checkpoint, monkeypatch):
    """A missing HPU extension (already warned once at startup) is a debug line; other reasons stay warnings."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layer = gdn_layers(model)[0]
        log = RecordingLogger()
        monkeypatch.setattr(qwen3_5, "logger", log)
        layer._log_gdn_core_off(qwen3_5._GDN_CORE_NO_EXTENSION)
        assert log.warnings == [] and len(log.debugs) == 1
        assert "is off for" in log.debugs[0][0] % log.debugs[0][1:]
        layer._log_gdn_core_off("conv1d has a bias")
        assert len(log.warnings) == 1 and len(log.debugs) == 1
        assert "conv1d has a bias" in log.warnings[0][0] % log.warnings[0][1:]


def test_fused_output_is_used(tiny_checkpoint, monkeypatch):
    """Negative control: a fused run whose op output is zeroed moves the logits by more than 0.1."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        real = qwen3_5.gdn_core_decode

        def zeroing_op(*args, **kwargs):
            return torch.zeros_like(real(*args, **kwargs))  # states still update, output is ignored

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode", zeroing_op)
        set_mode(model, GdnCoreDecodeMode.FUSED)
        fused_rows = run(model, vllm_config, eos, atts)

        diff = (fused_rows - off_rows).abs().max().item()
        assert diff > ZERO_OUTPUT_MIN_GAP, (f"zeroing the op output moved the logits by only {diff:.3e}: the layer "
                                            "ignores the fused output")
