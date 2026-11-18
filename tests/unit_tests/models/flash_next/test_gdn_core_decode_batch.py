# SPDX-License-Identifier: Apache-2.0
"""Layer-level padded-batch and slot-reuse tests for the fused GDN core decode (T3d of RUK-2812).

On the first GDN layer of the tiny Flash-Next model a batched decode step with one in-place row
(load 1 / store 1), one moved row (load 2 / store 3) and two padding rows must give FUSED output
rows 0-1 within 1e-6 of mode OFF, write the store slots 1 and 3 the same way (SSM within 1e-6,
conv bit-exact), leave B's load slot 2 and the unnamed slots 0 and 4 untouched, and - with ``-1``
padding - write nothing at all for the padded rows (the torch path writes the pool's last slot,
so slot 5 is asserted for FUSED only, never across modes). A state slot reused by a new request
after the previous request finished in it must start clean: the new request's decode rows and
post-step states equal the same request run alone on zeroed states.

One ``cpu_model`` per test function, never nested.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402

from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode  # noqa: E402
from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, prefill_metadata)
from tests.unit_tests.models.flash_next.test_chunked_prefill_parity import enable_slot_reuse  # noqa: E402

TOL = 1e-6
MODES = (GdnCoreDecodeMode.FUSED,)  # RECURRENCE joins when the mode-2 layer wiring (RUK-2834) merges

NUM_SLOTS = 6  # slots 0..5: 1 = A, 2 = B's load, 3 = B's store, 5 = the pool's last slot (-1 pad target)


def batch_metadata(load: list[int], store: list[int], query_start_loc: list[int], position: int):
    """A batched decode step's metadata: ``decode_metadata`` with the row/position fields replaced.

    The runner pads a decode batch with zero-length rows at the end, so 2 real rows and 2 padding
    rows have ``query_start_loc = [0, 1, 2, 2, 2]``; ``slot_mapping`` covers one attention slot per
    row (the GDN layer ignores it).
    """
    meta = decode_metadata(1, position)
    meta.load_indices_tensor = torch.tensor(load)
    meta.store_indices_tensor = torch.tensor(store)
    meta.query_start_loc_p = torch.tensor(query_start_loc, dtype=torch.int32)
    meta.slot_mapping = torch.arange(position, position + len(load), dtype=torch.int64)
    return meta


def first_gdn_layer(model) -> HPUGatedDeltaNetAttention:
    """The first ``HPUGatedDeltaNetAttention`` module of the model (3 on the tiny checkpoint)."""
    return [m for m in model.modules() if isinstance(m, HPUGatedDeltaNetAttention)][0]


def reset(layer: HPUGatedDeltaNetAttention, mode: GdnCoreDecodeMode) -> None:
    """Zero the layer's conv/SSM state pools, set the core-decode mode and re-arm the precondition check."""
    for state in layer.kv_cache:
        state.zero_()
    layer.gdn_core_decode_mode = mode
    layer._gdn_core_ok = None


@pytest.mark.parametrize("pad", [-1, 5], ids=["negative", "dummy"])
def test_padded_batch_matches_off(tiny_checkpoint, pad):
    """A 2-real + 2-padding decode step: FUSED matches OFF on real rows/slots and writes nothing for -1."""
    gen = torch.Generator().manual_seed(0)
    a_rows = torch.randn(10, 128, generator=gen)
    b_rows = torch.randn(7, 128, generator=gen)
    step_rows = torch.randn(4, 128, generator=gen)

    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        layer = first_gdn_layer(model)

        def run(mode: GdnCoreDecodeMode):
            reset(layer, mode)
            with torch.no_grad(), set_forward_context(prefill_metadata(10, 1), vllm_config):
                layer(a_rows)
            with torch.no_grad(), set_forward_context(prefill_metadata(7, 2), vllm_config):
                layer(b_rows)
            before = [state.clone() for state in layer.kv_cache]
            meta = batch_metadata([1, 2, pad, pad], [1, 3, pad, pad], [0, 1, 2, 2, 2], 20)
            with torch.no_grad(), set_forward_context(meta, vllm_config):
                out = layer(step_rows)
            return out, [state.clone() for state in layer.kv_cache], before

        off_out, off_after, off_before = run(GdnCoreDecodeMode.OFF)

        # The torch path itself: slot 2 (a pure load slot) untouched, unnamed slots 0 and 4 untouched.
        for k, label in ((0, "conv"), (1, "ssm")):
            assert torch.equal(off_after[k][2], off_before[k][2]), f"OFF pad={pad}: {label} slot 2 changed in the step"
        for slot in (0, 4):
            for k, label in ((0, "conv"), (1, "ssm")):
                assert not bool(off_after[k][slot].any()), f"OFF pad={pad}: {label} slot {slot} was written"

        for mode in MODES:
            out, after, before = run(mode)
            tag = f"{mode.name} pad={pad}"
            for row in (0, 1):
                assert torch.allclose(out[row], off_out[row], atol=TOL, rtol=0), (f"{tag}: output row {row} differs "
                                                                                 f"from OFF by more than {TOL:.0e}")
            for slot in (1, 3):
                assert torch.allclose(after[1][slot], off_after[1][slot], atol=TOL, rtol=0), (
                    f"{tag}: ssm slot {slot} differs from OFF by more than {TOL:.0e}")
                assert torch.equal(after[0][slot], off_after[0][slot]), f"{tag}: conv slot {slot} is not bit-exact"
            for k, label in ((0, "conv"), (1, "ssm")):
                assert torch.equal(after[k][2], before[k][2]), f"{tag}: {label} slot 2 changed in the step"
            for slot in (0, 4):
                for k, label in ((0, "conv"), (1, "ssm")):
                    assert not bool(after[k][slot].any()), f"{tag}: {label} slot {slot} was written"
            if pad == -1:
                # The op writes nothing for -1 rows; OFF writes the pool's last slot, so slot 5 is
                # checked for this mode only, never across modes.
                for k, label in ((0, "conv"), (1, "ssm")):
                    assert not bool(after[k][5].any()), f"{tag}: {label} slot 5 was written by the -1 padding rows"


@pytest.mark.parametrize("prompt_len", [7, 2])
def test_slot_reuse_after_request_finishes(tiny_checkpoint, prompt_len):
    """A slot freed by a finished request starts clean: B in reused slot 1 matches B alone on zeros."""
    gen = torch.Generator().manual_seed(0)
    a_rows = torch.randn(14, 128, generator=gen)
    b_rows = torch.randn(prompt_len + 4, 128, generator=gen)

    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        layer = first_gdn_layer(model)

        def run_b(mode: GdnCoreDecodeMode, after_a: bool):
            """Run B (prefill + 4 decodes in slot 1), optionally after request A finished in slot 1."""
            reset(layer, mode)
            if after_a:
                with torch.no_grad(), set_forward_context(prefill_metadata(10, 1), vllm_config):
                    layer(a_rows[:10])
                for i in range(4):
                    with torch.no_grad(), set_forward_context(decode_metadata(1, 10 + i), vllm_config):
                        layer(a_rows[10 + i:11 + i])
            with torch.no_grad(), set_forward_context(prefill_metadata(prompt_len, 1), vllm_config):
                layer(b_rows[:prompt_len])
            rows = []
            for i in range(4):
                with torch.no_grad(), set_forward_context(decode_metadata(1, prompt_len + i), vllm_config):
                    rows.append(layer(b_rows[prompt_len + i:prompt_len + i + 1]))
            return torch.cat(rows, dim=0), layer.kv_cache[0][1].clone(), layer.kv_cache[1][1].clone()

        ref_rows, ref_conv, ref_ssm = run_b(GdnCoreDecodeMode.OFF, after_a=False)
        for mode in (GdnCoreDecodeMode.OFF, *MODES):
            rows, conv, ssm = run_b(mode, after_a=True)
            tag = f"{mode.name} prompt_len={prompt_len}"
            assert torch.allclose(rows, ref_rows, atol=TOL, rtol=0), (f"{tag}: B decode rows in the reused slot "
                                                                     f"differ from B alone by more than {TOL:.0e}")
            assert torch.allclose(ssm, ref_ssm, atol=TOL, rtol=0), (f"{tag}: ssm slot 1 after B differs from "
                                                                    f"B alone by more than {TOL:.0e}")
            assert torch.equal(conv, ref_conv), f"{tag}: conv slot 1 after B is not bit-exact with B alone"
