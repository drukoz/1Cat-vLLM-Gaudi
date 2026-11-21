# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_KV_FP8=2: per-layer static scales of the FP8 K/V cache, and the calibration that writes them.

* ``kv_fp8_scales`` reads a layer's K and V scale (amax * margin / fp8_max) from the calibration file; a missing file
  or layer gives None, and ``set_kv_fp8_scales`` then keeps 1.0.
* ``set_kv_fp8_scales`` gives the layer's K and V caches ``input_scale = 1 / scale`` and ``output_scale = scale``, the
  attributes VLLMFP8KVCache multiplies by when it stores and when it dequantizes.
* The scale helps exactly where scale 1.0 hurts: small-magnitude K/V (below E4M3's smallest normal number 2**-6)
  keep their 3-bit mantissa with the scale; values already in the normal range do not change.
* ``record_kv_stats`` (VLLM_HPU_QSA_KV_STATS) writes the running amax and small-element fraction per layer, and
  ``tools/flash_next/rental/kv_scales.py`` merges the ranks' files into the scale file.
* A QSA layer accepts fp8_inc with the flag at 1 or 2.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.models.qwen4_exp.qsa_hpu_attention as qa
from vllm_gaudi.models.qwen4_exp.qsa import qsa_kv_cache_dtype_allowed

REPO = Path(__file__).resolve().parents[4]
FP8_MAX = 240.0


def roundtrip(x: torch.Tensor, scale: float) -> torch.Tensor:
    """VLLMFP8KVCache on the CPU: store x * (1 / scale) as Gaudi2 E4M3 (range 240), dequantize times scale."""
    return (x.float() * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale


def scales_file(tmp_path, layers, margin=1.0):
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"layers": layers, "fp8_max": FP8_MAX, "margin": margin}))
    return str(path)


def test_scales_from_the_file(tmp_path):
    path = scales_file(tmp_path, {"3": {"k_amax": 12.0, "v_amax": 0.6}}, margin=1.25)
    k, v = qa.kv_fp8_scales(path, 3)
    assert k == pytest.approx(12.0 * 1.25 / FP8_MAX) and v == pytest.approx(0.6 * 1.25 / FP8_MAX)
    assert qa.kv_fp8_scales(path, 7) is None
    assert qa.kv_fp8_scales(str(tmp_path / "missing.json"), 3) is None


def test_set_scales_on_the_caches(tmp_path):
    impl = SimpleNamespace(k_cache=SimpleNamespace(input_scale=1.0, output_scale=1.0),
                           v_cache=SimpleNamespace(input_scale=1.0, output_scale=1.0))
    path = scales_file(tmp_path, {"3": {"k_amax": 24.0, "v_amax": 2.4}})
    assert qa.set_kv_fp8_scales(impl, 3, path) == pytest.approx((0.1, 0.01))
    assert impl.k_cache.input_scale == pytest.approx(10.0) and impl.k_cache.output_scale == pytest.approx(0.1)
    assert impl.v_cache.input_scale == pytest.approx(100.0) and impl.v_cache.output_scale == pytest.approx(0.01)
    other = SimpleNamespace(k_cache=SimpleNamespace(input_scale=1.0, output_scale=1.0),
                            v_cache=SimpleNamespace(input_scale=1.0, output_scale=1.0))
    assert qa.set_kv_fp8_scales(other, 11, path) == (1.0, 1.0)  # not in the file: 1.0
    assert other.k_cache.input_scale == 1.0


@pytest.mark.parametrize("std,gain", [(0.004, 0.5), (0.02, 1.0), (0.1, None), (1.0, None)])
def test_the_scale_fixes_small_values_only(std, gain):
    """Relative RMS error of the round trip, Gaussian K/V of standard deviation ``std``. With the per-layer scale it
    is E4M3's mantissa error (about 2.6 %) at every size; at scale 1.0 it is larger only below about 0.03 (std 0.004:
    14 %, 0.02: 3.3 %, 0.1 and above: the same 2.6 %). So a scale lowers the error only for small-valued layers."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4096, 256, generator=g) * std
    amax = float(x.abs().max())
    e1 = float((roundtrip(x, 1.0) - x).norm() / x.norm())
    es = float((roundtrip(x, amax / FP8_MAX) - x).norm() / x.norm())
    assert es < 0.03  # with the scale: the 3-bit mantissa's error only
    if gain is not None:
        assert es < gain * e1  # small values: subnormal or flushed at scale 1.0
    else:
        assert abs(es - e1) < 0.002  # already normal numbers: no gain


def test_record_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "_KV_STATS", {})
    folder = tmp_path / "stats"
    k = torch.tensor([[0.001, 3.0, -7.5, 0.2]])
    v = torch.tensor([[0.01, -0.5, 0.002, 0.003]])
    for _ in range(16):
        qa.record_kv_stats(str(folder), 3, k, v)
    data = json.load(open(folder / "kv_amax_rank0.json"))
    rec = data["layers"]["3"]
    assert rec["k_amax"] == pytest.approx(7.5) and rec["v_amax"] == pytest.approx(0.5)
    assert rec["k_small"] == pytest.approx(0.25) and rec["v_small"] == pytest.approx(0.75)
    assert rec["calls"] == 16
    (folder / "kv_amax_rank1.json").write_text(json.dumps({"rank": 1, "layers": {"3": {"k_amax": 9.0, "v_amax": 0.1}}}))
    out = tmp_path / "scales.json"
    r = subprocess.run([sys.executable, str(REPO / "tools/flash_next/rental/kv_scales.py"), str(out), str(folder)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    merged = json.load(open(out))
    assert merged["layers"]["3"]["k_amax"] == pytest.approx(9.0) and merged["layers"]["3"]["v_amax"] == pytest.approx(0.5)
    assert qa.kv_fp8_scales(str(out), 3) == pytest.approx((9.0 / FP8_MAX, 0.5 / FP8_MAX))


def test_fp8_inc_allowed_with_the_flag_at_1_or_2():
    assert qsa_kv_cache_dtype_allowed("fp8_inc", 1) and qsa_kv_cache_dtype_allowed("fp8_inc", 2)
    assert not qsa_kv_cache_dtype_allowed("fp8_inc", 0)


def test_flags_documented_and_off_by_default(monkeypatch):
    import vllm_gaudi.envs as envs
    for name in ("VLLM_HPU_QSA_KV_FP8_SCALES", "VLLM_HPU_QSA_KV_STATS"):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) == ""
    doc = (REPO / "docs/configuration/env_variables.md").read_text()
    assert "VLLM_HPU_QSA_KV_FP8_SCALES" in doc and "VLLM_HPU_QSA_KV_STATS" in doc
