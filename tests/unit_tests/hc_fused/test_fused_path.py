# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the fused hyper-connection path against ``HpuGatedResidual`` (needs vLLM).

1. ``fused_path`` (the layer's own fused methods, called directly) returns exactly what the torch
   ``HpuGatedResidual.mix`` / ``combine_and_mix`` / ``combine`` return, in bfloat16, for both
   norm-weight layouts and for the final mixer (``use_combine=False``).  On CPU the fused ops run
   the torch oracle, so this checks the wiring: argument order, the split, which tensor feeds what.
2. A torch dispatch-mode counter over one decode-shaped call counts the torch ops of both paths.
   The numbers are the "ops per layer before/after" of ``docs/design/flash-next/hc-fused.md``.

The modules here are built with the flag at its default (0), so their methods are the torch path.
The flag, the preconditions and the fallback are tested in
``tests/unit_tests/models/flash_next/test_hc_fused_layer.py``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import collections  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

pytest.importorskip("vllm")

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel  # noqa: E402
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig  # noqa: E402

from tests.unit_tests.hc_fused.fused_path import (fused_combine, fused_combine_and_mix, fused_mix,  # noqa: E402
                                                  fused_precondition)
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402

# (hidden_size, hc_lowrank): the tiny test model and the real model.
GEOMETRIES = [(128, 16), (2560, 320)]
GEMM_OPS = ("aten::linear", "aten::addmm", "aten::mm", "aten::matmul", "aten::t")


@pytest.fixture(scope="module")
def world1():
    """World-1 gloo group with a current VllmConfig (the pattern of test_hyperconnection.py)."""
    _, temp_file = tempfile.mkstemp(dir="/tmp")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1,
                                     rank=0,
                                     distributed_init_method=f"file://{temp_file}",
                                     local_rank=0,
                                     backend="gloo")
        initialize_model_parallel(1, 1)
        yield
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture
def vllm_ctx(monkeypatch):
    """vLLM linear layers read get_current_vllm_config() at construction; the layers are built with the flag off."""
    monkeypatch.delenv("VLLM_HPU_HC_FUSED", raising=False)
    with set_current_vllm_config(VllmConfig()):
        yield


def _module(hidden_size: int, hc_lowrank: int, *, per_branch: bool = True, use_combine: bool = True):
    config = HyperConnectionConfig(hc_count=4,
                                   hidden_size=hidden_size,
                                   params_dtype=torch.bfloat16,
                                   hc_lowrank=hc_lowrank,
                                   rms_norm_eps=1e-6,
                                   hc_per_branch_norm=per_branch)
    module = HpuGatedResidual(config, use_combine=use_combine)
    g = torch.Generator().manual_seed(hidden_size + hc_lowrank)
    with torch.no_grad():
        for name, param in module.named_parameters():
            scale = 0.25 if name.startswith("hc_norm") else param.shape[-1]**-0.5
            param.copy_((torch.randn(param.shape, generator=g) * scale).to(param.dtype))
        if use_combine and module.pad_size:
            module.input_mix_weight_down_block_inject.weight[module.lora_rank + module.hc_count:].zero_()
    for layer in module.children():
        if hasattr(layer, "quant_method"):  # the vLLM linear layers build their kernel after loading
            layer.quant_method.process_weights_after_loading(layer)
    return module


def _inputs(module, tokens: int):
    g = torch.Generator().manual_seed(tokens)
    hidden = torch.randn(tokens, module.hyper_hidden_size, generator=g).to(torch.bfloat16)
    block = torch.randn(tokens, module.hidden_size, generator=g).to(torch.bfloat16)
    injection = (torch.randn(tokens, module.hc_count, generator=g) * 2).to(torch.bfloat16)
    return hidden, block, injection


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("hidden_size,hc_lowrank", GEOMETRIES)
@pytest.mark.parametrize("per_branch", [True, False])
@pytest.mark.parametrize("tokens", [1, 4])
def test_fused_path_equals_torch_path(world1, vllm_ctx, hidden_size, hc_lowrank, per_branch, tokens):
    module = _module(hidden_size, hc_lowrank, per_branch=per_branch)
    hidden, block, injection = _inputs(module, tokens)
    assert fused_precondition(module, hidden) is None
    with torch.no_grad():
        for torch_out, fused_out in (
            (module.mix(hidden), fused_mix(module, hidden)),
            (module.combine_and_mix(hidden, block, injection), fused_combine_and_mix(module, hidden, block,
                                                                                     injection)),
        ):
            assert len(torch_out) == len(fused_out) == 3
            for a, b in zip(torch_out, fused_out):
                assert _same(a, b)
        assert _same(module.combine(hidden, block, injection), fused_combine(module, hidden, block, injection))


@pytest.mark.parametrize("hidden_size,hc_lowrank", GEOMETRIES)
def test_fused_final_mixer_equals_torch_path(world1, vllm_ctx, hidden_size, hc_lowrank):
    module = _module(hidden_size, hc_lowrank, use_combine=False)
    hidden, block, injection = _inputs(module, 4)
    with torch.no_grad():
        torch_out = module.combine_and_mix(hidden, block, injection)
        fused_out = fused_combine_and_mix(module, hidden, block, injection)
    assert torch_out[2] is None and fused_out[2] is None
    assert _same(torch_out[0], fused_out[0]) and _same(torch_out[1], fused_out[1])


def test_fused_precondition_names_the_reason(world1, vllm_ctx):
    module = _module(128, 16)
    hidden, _, _ = _inputs(module, 2)
    assert "3-D" in fused_precondition(module, hidden.unsqueeze(0))
    assert "float32" in fused_precondition(module, hidden.float())
    odd = _module(64, 20)  # hidden 64: not a multiple of one TPC vector
    assert "multiple of 128" in fused_precondition(odd, torch.zeros(2, 256, dtype=torch.bfloat16))


class _OpCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def _count(fn) -> collections.Counter:
    with torch.no_grad(), _OpCounter() as counter:
        fn()
    return counter.ops


def op_counts(module, tokens: int = 1) -> dict[str, dict[str, collections.Counter]]:
    """Torch ops of one decode-shaped call of each method, torch path and fused path."""
    hidden, block, injection = _inputs(module, tokens)
    return {
        "mix": {
            "torch": _count(lambda: module.mix(hidden)),
            "fused": _count(lambda: fused_mix(module, hidden)),
        },
        "combine_and_mix": {
            "torch": _count(lambda: module.combine_and_mix(hidden, block, injection)),
            "fused": _count(lambda: fused_combine_and_mix(module, hidden, block, injection)),
        },
        "combine": {
            "torch": _count(lambda: module.combine(hidden, block, injection)),
            "fused": _count(lambda: fused_combine(module, hidden, block, injection)),
        },
    }


def test_op_count_before_and_after(world1, vllm_ctx):
    """One decode-shaped call (1 token, real geometry): the fused path is 3 ops plus the 2 GEMMs."""
    module = _module(2560, 320)
    counts = op_counts(module, tokens=1)
    for method, paths in counts.items():
        for path, ops in paths.items():
            gemm = sum(n for name, n in ops.items() if name in GEMM_OPS)
            print(f"HCOPS {method:16s} {path:5s} total={sum(ops.values()):3d} gemm={gemm} "
                  f"other={sum(ops.values()) - gemm:3d} | " +
                  " ".join(f"{name.split('::')[1]}x{n}" for name, n in sorted(ops.items())))
    fused_cm = counts["combine_and_mix"]["fused"]
    torch_cm = counts["combine_and_mix"]["torch"]
    assert {name: n for name, n in fused_cm.items() if name.startswith("flashinfer_gaudi::")} == {
        "flashinfer_gaudi::hc_combine_norm": 1,
        "flashinfer_gaudi::hc_silu_split": 1,
        "flashinfer_gaudi::hc_gate_mix": 1,
    }
    fused_mix_ops = counts["mix"]["fused"]
    assert {name: n for name, n in fused_mix_ops.items() if name.startswith("flashinfer_gaudi::")} == {
        "flashinfer_gaudi::hc_norm": 1,
        "flashinfer_gaudi::hc_silu_split": 1,
        "flashinfer_gaudi::hc_gate_mix": 1,
    }
    # Every activation is copied before its op (aten::clone): a custom HPU op cannot be given a view
    # (flashinfer_gaudi.hc_fused._owned, found on a card on 2026-11-20). One copy per tensor input.
    assert dict(counts["combine"]["fused"]) == {"flashinfer_gaudi::hc_combine": 1, "aten::clone": 3}
    # The GEMMs are the same ops in both paths; everything else collapses to the three fused ops and
    # the copies of their inputs (combine_norm 3, silu_split 1, gate_mix 2; norm 1 in place of 3).
    gemm = {name: n for name, n in torch_cm.items() if name in GEMM_OPS}
    assert gemm == {name: n for name, n in fused_cm.items() if name in GEMM_OPS}
    assert sum(fused_cm.values()) - sum(gemm.values()) == 3 + 6
    assert sum(fused_mix_ops.values()) - sum(gemm.values()) == 3 + 4
    assert sum(torch_cm.values()) - sum(gemm.values()) >= 40
    # A decoder layer runs combine_and_mix twice (attention and MLP hyper-connection).
    per_layer_before = 2 * (sum(torch_cm.values()) - sum(gemm.values()))
    per_layer_after = 2 * (sum(fused_cm.values()) - sum(gemm.values()))
    print(f"HCOPS per layer (2 x combine_and_mix, GEMMs excluded): before={per_layer_before} after={per_layer_after}")
    assert per_layer_after == 6 + 12  # the fused ops, plus the copies of their activation inputs
