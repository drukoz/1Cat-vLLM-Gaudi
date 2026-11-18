# SPDX-License-Identifier: Apache-2.0
"""Layer-level tests for the recurrence-only GDN core decode (T3b of track RUK-2812).

With ``VLLM_HPU_GDN_CORE_DECODE=2`` every non-speculative decode step of
``HPUGatedDeltaNetAttention`` keeps the torch conv update and ``self.norm`` but runs the
recurrence through ``torch.ops.flashinfer_gaudi.gdn_recurrent_decode``. With
``VLLM_HPU_GDN_CORE_DECODE_DEBUG=1`` both op modes check the slot indices before any state is
written. On the CPU tiny Flash-Next model the 8 teacher-forced decode rows must match mode 0
(fp32 to 1e-5, bf16 to 5e-2 with equal argmax), a load != store step must write only the store
slot, and a layer whose op is unavailable must log exactly one warning naming
``gdn_recurrent_decode`` and run the torch path. The negative control (the op's output replaced
by zeros) proves the parity checks would catch an op output the layer ignores.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeDebug, GdnCoreDecodeMode  # noqa: E402
import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.test_chunked_prefill_parity import enable_slot_reuse  # noqa: E402
from tests.unit_tests.models.flash_next.test_decode_parity import install_cached_attention  # noqa: E402
from tests.unit_tests.models.flash_next.test_gdn_core_decode_layer import (  # noqa: E402
    BF16_TOL, DECODE_STEPS, FALLBACK_TOL, FP32_TOL, GDN_LAYERS, LOAD_SLOT, NUM_SLOTS, PREFILL_LEN, SLOT, STORE_SLOT,
    ZERO_OUTPUT_MIN_GAP, CountingOp, RecordingLogger, decode_step, gdn_layers, prefill_into, run, set_mode,
    slot_state_rows, tokens)


def test_debug_defaults_off(tiny_checkpoint, monkeypatch):
    """With the debug variable unset, every GDN layer starts in GdnCoreDecodeDebug.OFF."""
    monkeypatch.delenv("VLLM_HPU_GDN_CORE_DECODE_DEBUG", raising=False)
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layers = gdn_layers(model)
        assert len(layers) == GDN_LAYERS
        for module in layers:
            assert module.gdn_core_decode_debug == GdnCoreDecodeDebug.OFF


def test_recurrence_flags_reach_layers(tiny_checkpoint, monkeypatch):
    """VLLM_HPU_GDN_CORE_DECODE=2 and ..._DEBUG=1 set before the build reach every GDN layer (envs is lazy)."""
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "2")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE_DEBUG", "1")
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layers = gdn_layers(model)
        assert len(layers) == GDN_LAYERS
        for module in layers:
            assert module.gdn_core_decode_mode == GdnCoreDecodeMode.RECURRENCE
            assert module.gdn_core_decode_debug == GdnCoreDecodeDebug.CHECK


def test_recurrence_matches_torch_path_fp32(tiny_checkpoint, monkeypatch):
    """fp32: the recurrence decode rows match the torch path, the op runs 3*8 times, states match."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        layers = gdn_layers(model)
        rec_op = CountingOp(qwen3_5.gdn_recurrent_decode)
        fused_op = CountingOp(qwen3_5.gdn_core_decode)
        gating_op = CountingOp(qwen3_5.hpu_fused_gdn_gating)
        monkeypatch.setattr(qwen3_5, "gdn_recurrent_decode", rec_op)
        monkeypatch.setattr(qwen3_5, "gdn_core_decode", fused_op)
        monkeypatch.setattr(qwen3_5, "hpu_fused_gdn_gating", gating_op)

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        assert rec_op.calls == 0, "the OFF run must not call the recurrence op"
        conv_ref = [module.kv_cache[0][SLOT].clone() for module in layers]
        ssm_ref = [module.kv_cache[1][SLOT].clone() for module in layers]
        gating_op.calls = 0  # count the gating calls of the RECURRENCE run alone

        set_mode(model, GdnCoreDecodeMode.RECURRENCE)
        rec_rows = run(model, vllm_config, eos, atts)
        assert rec_op.calls == GDN_LAYERS * DECODE_STEPS, (f"expected {GDN_LAYERS * DECODE_STEPS} recurrence op "
                                                          f"calls, got {rec_op.calls}")
        assert fused_op.calls == 0, "RECURRENCE must not call the fully fused op"
        assert gating_op.calls == GDN_LAYERS, (f"expected {GDN_LAYERS} gating calls (prefill only), "
                                               f"got {gating_op.calls}")

        diff = (rec_rows - off_rows).abs().max().item()
        assert diff <= FP32_TOL, f"fp32 decode logits differ by {diff:.3e} > {FP32_TOL:.0e}"
        for idx, (module, conv, ssm) in enumerate(zip(layers, conv_ref, ssm_ref)):
            # Layer 0 sees the input embedding, so its conv state (raw mixed_qkv) must be bit-exact.
            # Layers 1-2 consume layer 0's op output, whose <=1e-5 drift propagates into their conv inputs.
            if idx == 0:
                assert torch.equal(module.kv_cache[0][SLOT], conv), "layer 0 conv state must be bit-exact"
            else:
                assert torch.allclose(module.kv_cache[0][SLOT], conv, atol=FP32_TOL, rtol=0), (
                    f"layer {idx}: conv state of slot {SLOT} changed beyond {FP32_TOL:.0e}")
            assert torch.allclose(module.kv_cache[1][SLOT], ssm, atol=FP32_TOL, rtol=0), (
                f"layer {idx}: ssm state of slot {SLOT} differs by more than {FP32_TOL:.0e}")


def test_recurrence_matches_torch_path_bf16(tiny_checkpoint, monkeypatch):
    """bf16: same argmax token on every row and logits within 5e-2 of the torch path."""
    with cpu_model(tiny_checkpoint, fp32=False) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        rec_op = CountingOp(qwen3_5.gdn_recurrent_decode)
        monkeypatch.setattr(qwen3_5, "gdn_recurrent_decode", rec_op)

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        assert rec_op.calls == 0
        set_mode(model, GdnCoreDecodeMode.RECURRENCE)
        rec_rows = run(model, vllm_config, eos, atts)
        assert rec_op.calls == GDN_LAYERS * DECODE_STEPS

        diff = (rec_rows - off_rows).abs().max().item()
        assert diff <= BF16_TOL, f"bf16 decode logits differ by {diff:.3e} > {BF16_TOL:.0e}"
        for row in range(DECODE_STEPS):
            off_pick = int(off_rows[row].argmax())
            rec_pick = int(rec_rows[row].argmax())
            assert off_pick == rec_pick, f"row {row}: argmax {rec_pick} != {off_pick}"


def test_recurrence_load_ne_store(tiny_checkpoint):
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
        rec_rows, after_prefill = step(GdnCoreDecodeMode.RECURRENCE)

        diff = (rec_rows - off_rows).abs().max().item()
        assert diff <= FP32_TOL, f"fp32 load/store-move logits differ by {diff:.3e} > {FP32_TOL:.0e}"
        load_rows = slot_state_rows(layers, LOAD_SLOT)
        store_rows = slot_state_rows(layers, STORE_SLOT)
        for idx, (now, before) in enumerate(zip(load_rows, after_prefill)):
            assert torch.equal(now, before), (f"slot {LOAD_SLOT} state tensor {idx} changed in the recurrence step")
        for idx, (now, ref) in enumerate(zip(store_rows, off_store)):
            # conv tensors are the even indices (idx // 2 = layer). Only layer 0's conv input is
            # bit-exact; layers 1-2 inherit layer 0's op-output drift, so they keep atol=1e-5.
            if idx == 0:
                assert torch.equal(now, ref), f"conv state of slot {STORE_SLOT} (tensor {idx}) must be bit-exact"
            elif idx % 2 == 0:
                assert torch.allclose(now, ref, atol=FP32_TOL, rtol=0), (
                    f"conv state of slot {STORE_SLOT} (tensor {idx}) differs from the torch run "
                    f"by more than {FP32_TOL:.0e}")
            else:
                assert torch.allclose(now, ref, atol=FP32_TOL, rtol=0), (
                    f"ssm state of slot {STORE_SLOT} (tensor {idx}) differs from the torch run "
                    f"by more than {FP32_TOL:.0e}")
        for slot in (0, 1, 4, 5, 6, 7):
            for idx, tensor in enumerate(slot_state_rows(layers, slot)):
                assert not bool(tensor.any()), f"slot {slot} state tensor {idx} was written (expected all zeros)"


def test_recurrence_fallback_when_op_unavailable(tiny_checkpoint, monkeypatch):
    """No recurrence kernel for the device: one warning per layer, the torch path, and the OFF-run logits."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        rec_op = CountingOp(qwen3_5.gdn_recurrent_decode)
        monkeypatch.setattr(qwen3_5, "gdn_recurrent_decode", rec_op)
        monkeypatch.setattr(qwen3_5, "gdn_recurrent_decode_available", lambda device: False)
        log = RecordingLogger()
        monkeypatch.setattr(qwen3_5, "logger", log)

        set_mode(model, GdnCoreDecodeMode.RECURRENCE)
        rec_rows = run(model, vllm_config, eos, atts)

        assert rec_op.calls == 0, "the fallback path must not call the recurrence op"
        assert len(log.warnings) == GDN_LAYERS, (f"expected one warning per layer ({GDN_LAYERS}), "
                                                 f"got {len(log.warnings)}")
        for args in log.warnings:
            message = args[0] % args[1:]
            assert "is off for" in message, f"warning {message!r} is missing the layer line"
            assert "no gdn_recurrent_decode kernel for cpu" in message, f"warning {message!r} names the wrong reason"
        for module in gdn_layers(model):
            assert module._gdn_core_ok is False

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        diff = (rec_rows - off_rows).abs().max().item()
        assert diff <= FALLBACK_TOL, f"the fallback run differs from mode OFF by {diff:.3e} > {FALLBACK_TOL:.0e}"


def test_recurrence_output_is_used(tiny_checkpoint, monkeypatch):
    """Negative control: a recurrence run whose op output is zeroed moves the logits by more than 0.1."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        real = qwen3_5.gdn_recurrent_decode

        def zeroing_op(*args, **kwargs):
            return torch.zeros_like(real(*args, **kwargs))  # states still update, output is ignored

        set_mode(model, GdnCoreDecodeMode.OFF)
        off_rows = run(model, vllm_config, eos, atts)
        monkeypatch.setattr(qwen3_5, "gdn_recurrent_decode", zeroing_op)
        set_mode(model, GdnCoreDecodeMode.RECURRENCE)
        rec_rows = run(model, vllm_config, eos, atts)

        diff = (rec_rows - off_rows).abs().max().item()
        assert diff > ZERO_OUTPUT_MIN_GAP, (f"zeroing the op output moved the logits by only {diff:.3e}: the layer "
                                            "ignores the recurrence output")


@pytest.mark.parametrize("mode", [GdnCoreDecodeMode.FUSED, GdnCoreDecodeMode.RECURRENCE])
def test_debug_check_called_each_step(tiny_checkpoint, monkeypatch, mode):
    """With CHECK the slot check runs once per layer per decode step (24) with (num_slots, num_rows) = (4, 1)."""
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        layers = gdn_layers(model)
        assert layers[0].kv_cache[1].shape[0] == 4, "the default cpu_model has 4 state slots"
        real_check = qwen3_5.check_gdn_core_decode_indices
        calls: list[tuple[int, int]] = []

        def recording_check(load_indices, store_indices, num_slots, num_rows):
            calls.append((num_slots, num_rows))
            real_check(load_indices, store_indices, num_slots, num_rows)

        monkeypatch.setattr(qwen3_5, "check_gdn_core_decode_indices", recording_check)
        set_mode(model, mode)
        for module in layers:
            module.gdn_core_decode_debug = GdnCoreDecodeDebug.OFF
        run(model, vllm_config, eos, atts)
        assert calls == [], f"the debug check ran {len(calls)} times with GdnCoreDecodeDebug.OFF"

        for module in layers:
            module.gdn_core_decode_debug = GdnCoreDecodeDebug.CHECK
        run(model, vllm_config, eos, atts)
        assert len(calls) == GDN_LAYERS * DECODE_STEPS, (f"expected {GDN_LAYERS * DECODE_STEPS} debug checks, "
                                                         f"got {len(calls)}")
        for call in calls:
            assert call == (4, 1), f"debug check {call} is not (4 slots, 1 real row)"


def test_debug_check_counts_real_rows(tiny_checkpoint):
    """num_rows counts only the real decode rows: padding rows sharing a dummy slot pass, real ties raise."""
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        layer = gdn_layers(model)[0]
        layer.gdn_core_decode_debug = GdnCoreDecodeDebug.CHECK

        # (a) two padding rows (indices 2, 3) share dummy slot 5; only 2 rows are real (query_start_loc[-1] = 2).
        layer._check_gdn_core_decode_indices(torch.tensor([1, 2, 5, 5]),
                                             torch.tensor([1, 2, 5, 5]),
                                             torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
                                             4)

        # (b) two real rows both store slot 3 -> ValueError naming the slot.
        with pytest.raises(ValueError, match="both store slot 3"):
            layer._check_gdn_core_decode_indices(torch.tensor([1, 2]),
                                                 torch.tensor([3, 3]),
                                                 torch.tensor([0, 1, 2], dtype=torch.int32),
                                                 2)

        # (c) with no query_start_loc, num_tokens is the row count: the same tie raises.
        with pytest.raises(ValueError, match="both store slot 3"):
            layer._check_gdn_core_decode_indices(torch.tensor([1, 2]),
                                                 torch.tensor([3, 3]),
                                                 None,
                                                 2)

        # (d) with the debug check off, the same call is a no-op.
        layer.gdn_core_decode_debug = GdnCoreDecodeDebug.OFF
        layer._check_gdn_core_decode_indices(torch.tensor([1, 2]),
                                             torch.tensor([3, 3]),
                                             torch.tensor([0, 1, 2], dtype=torch.int32),
                                             2)
