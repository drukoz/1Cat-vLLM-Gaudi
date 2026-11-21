# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_KV_FP8: FP8 storage of the sparse QSA K/V cache with bf16 attention math (CPU).

* The sparse decode attention over an E4M3 cache, with the gathered rows dequantized (``k_dequant`` /
  ``v_dequant``), equals the same attention over the dequantized cache in bf16, bit for bit, in both forms
  (repeated K/V and grouped), at 1, 8 and 32 rows, 3 and 6 query heads per KV head, the real width 2,051 and
  head dim 256. Against the unquantized bf16 cache the output moves by the E4M3 rounding only.
* A QSA layer serves ``fp8_inc`` only with the flag (``qsa_kv_cache_dtype_allowed``).
* ``bf16_attention_math`` puts the FP8 matmuls and the FP8 fused SDPA of ``fp8_inc`` back to bf16 and keeps
  the FP8 K/V cache modules.
* The capacity arithmetic at TP8 (the worker's ``attention_budget_fraction`` with a 1-byte K/V dtype).
* The flag is documented and defaults to 0.

Gaudi2's E4M3 has the range 240 (card rule 6); the test clamps there before the CPU cast (``float8_e4m3fn``,
range 448), so every stored value is one Gaudi2 can hold and both round the same.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.qwen4_exp.qsa import qsa_kv_cache_dtype_allowed
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import (bf16_attention_math, sparse_decode_attention,
                                                           sparse_decode_attention_grouped)
from vllm_gaudi.models.qwen4_exp.qsa_side_caches import attention_budget_fraction

REPO = Path(__file__).resolve().parents[4]
FP8 = torch.float8_e4m3fn
GAUDI2_FP8_MAX = 240.0
WIDTH = 2051
HEAD_DIM = 256


def to_fp8(t: torch.Tensor) -> torch.Tensor:
    """What VLLMFP8KVCache stores at scale 1.0 (cast_to_fp8_v2), on the CPU: E4M3 within Gaudi2's range."""
    return t.float().clamp(-GAUDI2_FP8_MAX, GAUDI2_FP8_MAX).to(FP8)


def from_fp8(t: torch.Tensor) -> torch.Tensor:
    """What VLLMFP8KVCache.dequant_output returns at scale 1.0 (cast_from_fp8 to bf16), on the CPU."""
    return t.to(torch.bfloat16)


def case(rows: int, groups: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    slots_total = 8192
    num_kv_heads, num_heads = 1, groups
    key_cache = (torch.randn(slots_total, num_kv_heads, HEAD_DIM, generator=gen) * 2.0).to(torch.bfloat16)
    value_cache = (torch.randn(slots_total, num_kv_heads, HEAD_DIM, generator=gen) * 3.0).to(torch.bfloat16)
    query = (torch.randn(rows, num_heads * HEAD_DIM, generator=gen)).to(torch.bfloat16)
    slots = torch.randint(0, slots_total, (rows, WIDTH), generator=gen)
    valid = torch.rand(rows, WIDTH, generator=gen) > 0.1
    valid[:, 0] = True
    return query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads


@pytest.mark.parametrize("attend", [sparse_decode_attention, sparse_decode_attention_grouped],
                         ids=["repeat", "grouped"])
@pytest.mark.parametrize("rows,groups", [(1, 3), (8, 3), (32, 3), (32, 6)])
def test_fp8_cache_with_dequant_equals_bf16_math_on_the_dequantized_cache(attend, rows, groups):
    query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads = case(rows, groups, seed=rows + groups)
    k8, v8 = to_fp8(key_cache), to_fp8(value_cache)
    scale = HEAD_DIM**-0.5
    got = attend(query, k8, v8, slots, valid, num_heads, num_kv_heads, HEAD_DIM, scale, k_dequant=from_fp8,
                 v_dequant=from_fp8)
    want = attend(query, from_fp8(k8), from_fp8(v8), slots, valid, num_heads, num_kv_heads, HEAD_DIM, scale)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # Against the unquantized cache the output moves by the E4M3 rounding of K and V only (3 mantissa bits:
    # about 3.6 % RMS per element); measured as the relative RMS error of the output.
    exact = attend(query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads, HEAD_DIM, scale)
    err = (got.float() - exact.float()).norm() / exact.float().norm()
    assert float(err) < 0.08, float(err)


def test_without_dequant_the_bf16_path_is_unchanged():
    query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads = case(8, 3, seed=5)
    scale = HEAD_DIM**-0.5
    a = sparse_decode_attention_grouped(query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads, HEAD_DIM,
                                        scale)
    b = sparse_decode_attention_grouped(query, key_cache, value_cache, slots, valid, num_heads, num_kv_heads, HEAD_DIM,
                                        scale, k_dequant=None, v_dequant=None)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_the_layer_serves_fp8_inc_only_with_the_flag():
    for dtype in ("auto", "bfloat16"):
        assert qsa_kv_cache_dtype_allowed(dtype, 0) and qsa_kv_cache_dtype_allowed(dtype, 1)
    assert not qsa_kv_cache_dtype_allowed("fp8_inc", 0)
    assert qsa_kv_cache_dtype_allowed("fp8_inc", 1)
    for dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "float16"):
        assert not qsa_kv_cache_dtype_allowed(dtype, 1), dtype


class _Mod:

    def __init__(self, *args):
        self.args = args


def _stand_ins(fused=True):
    return SimpleNamespace(Matmul=type("Matmul", (_Mod, ), {}), B2BMatmul=type("B2BMatmul", (_Mod, ), {}),
                           ModuleFusedSDPA=type("ModuleFusedSDPA", (_Mod, ), {}),
                           fsdpa=(lambda: "bf16-fsdpa-kernel") if fused else (lambda: None))


def test_bf16_attention_math_puts_the_fp8_math_back_and_keeps_the_fp8_storage():
    k_cache, v_cache = object(), object()
    impl = SimpleNamespace(enable_fp8_attn=True, matmul_qk="FP8Matmul", matmul_av="FP8Matmul",
                           batch2block_matmul="FP8Matmul", block2batch_matmul="FP8Matmul",
                           fused_scaled_dot_product_attention="ModuleFP8FusedSDPA", k_cache=k_cache, v_cache=v_cache)
    mods = _stand_ins()
    assert bf16_attention_math(impl, mods) is True
    assert isinstance(impl.matmul_qk, mods.Matmul) and isinstance(impl.matmul_av, mods.Matmul)
    assert isinstance(impl.batch2block_matmul, mods.B2BMatmul) and isinstance(impl.block2batch_matmul, mods.B2BMatmul)
    assert isinstance(impl.fused_scaled_dot_product_attention, mods.ModuleFusedSDPA)
    assert impl.fused_scaled_dot_product_attention.args == ("bf16-fsdpa-kernel", )
    assert impl.enable_fp8_attn is False
    assert impl.k_cache is k_cache and impl.v_cache is v_cache  # the FP8 storage stays
    # A second call (another layer sharing nothing) and a bf16 impl are left alone.
    assert bf16_attention_math(impl, mods) is False
    plain = SimpleNamespace(enable_fp8_attn=False, matmul_qk="Matmul")
    assert bf16_attention_math(plain, mods) is False and plain.matmul_qk == "Matmul"
    no_fsdpa = SimpleNamespace(enable_fp8_attn=True)
    bf16_attention_math(no_fsdpa, _stand_ins(fused=False))
    assert no_fsdpa.fused_scaled_dot_product_attention is None


def test_tp8_capacity_arithmetic():
    """Flash-Next at TP8: 12 QSA layers, 1 KV head of 256 per rank, the indexer's keys (128, ratio 4) in bf16."""
    layers, kv_heads, head, ratio, dim = 12, 1, 256, 4, 128
    for kv_bytes, per_token in ((2, 13_056), (1, 6_912)):
        kv = 2 * kv_heads * head * kv_bytes
        fraction = attention_budget_fraction(kv_heads, head, kv_bytes, ratio, dim)
        assert fraction == pytest.approx(kv / (kv + dim * 2 / ratio))
        assert layers * (kv + dim * 2 // ratio) == per_token
    budget = int(35.05 * 2**30)  # integration build 2b, TP8 rank 0: bytes left for attention blocks
    assert budget // 13_056 == pytest.approx(2_882_000, rel=0.001)  # the 2,881,536 slots that run allocated
    assert budget // 6_912 > 32 * 131_072  # 5.44M tokens: 32 agents at 131K fit


def test_flag_documented_and_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_QSA_KV_FP8", raising=False)
    import vllm_gaudi.envs as envs
    assert envs.VLLM_HPU_QSA_KV_FP8 == 0
    monkeypatch.setenv("VLLM_HPU_QSA_KV_FP8", "1")
    assert envs.VLLM_HPU_QSA_KV_FP8 == 1
    doc = (REPO / "docs" / "configuration" / "env_variables.md").read_text()
    assert "`VLLM_HPU_QSA_KV_FP8`" in doc and "fp8_inc" in doc
