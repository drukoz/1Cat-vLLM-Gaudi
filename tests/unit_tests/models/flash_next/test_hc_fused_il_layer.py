# SPDX-License-Identifier: Apache-2.0
"""CPU tests of ``HpuGatedResidual`` with ``VLLM_HPU_HC_FUSED_IL=1`` (the interleaved hyper-connection kernels).

``VLLM_HPU_HC_FUSED_IL=1`` (with ``VLLM_HPU_HC_FUSED=1``) makes a layer call ``hc_gate_mix_il`` and
``hc_combine_norm_il`` instead of ``hc_gate_mix`` and ``hc_combine_norm`` (docs/design/flash-next/hc-kernels.md).
On CPU the ops get the torch oracle as their kernels (``tests/unit_tests/hc_fused/hc_ops.py``), so the
interleaved path must give the torch path's bits; a difference is a wiring error. What is proven here: the flag,
the once-per-layer decision with its one log line, the fallback to the shipped fused ops when the interleaved ones
are not served, the copies in front of the ops, and the wiring. The kernels themselves: the simulator
(``tests/standalone/hc_fused/test_hc_il_simulation.py``) and the card (``tests/unit_tests/hc_fused/test_hpu_op_il.py``).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import collections  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel  # noqa: E402
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig  # noqa: E402

from flashinfer_gaudi import _native, hc_fused  # noqa: E402
from tests.unit_tests.hc_fused import hc_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from vllm_gaudi import envs as gaudi_envs  # noqa: E402
from vllm_gaudi.models.qwen4_exp import hyperconnection as layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402

BF16 = torch.bfloat16
TINY = (128, 16)  # hidden_size, hc_lowrank of the tiny checkpoint
REAL = (2560, 320)  # of the real model
FUSED_PREFIX = "flashinfer_gaudi::"
IL_FLAG = "VLLM_HPU_HC_FUSED_IL"
IL_OFF = "HC interleaved kernels (VLLM_HPU_HC_FUSED_IL=1) are off for"


@pytest.fixture(scope="module")
def world1():
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
def vllm_ctx():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv(IL_FLAG, "1")


@pytest.fixture
def warnings(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(layer_module.logger, "warning", lambda message, *args: lines.append(message % args))
    return lines


def _module(hidden_size: int = 128, hc_lowrank: int = 16, *, per_branch: bool = True, use_combine: bool = True,
            prefix: str = "layers.0.attn_hyper_connection"):
    config = HyperConnectionConfig(hc_count=4,
                                   hidden_size=hidden_size,
                                   params_dtype=BF16,
                                   hc_lowrank=hc_lowrank,
                                   rms_norm_eps=1e-6,
                                   hc_per_branch_norm=per_branch)
    module = HpuGatedResidual(config, use_combine=use_combine, prefix=prefix)
    g = torch.Generator().manual_seed(hidden_size + hc_lowrank)
    with torch.no_grad():
        for name, param in module.named_parameters():
            scale = 0.25 if name.startswith("hc_norm") else param.shape[-1]**-0.5
            param.copy_((torch.randn(param.shape, generator=g) * scale).to(param.dtype))
        if use_combine and module.pad_size:
            module.input_mix_weight_down_block_inject.weight[module.lora_rank + module.hc_count:].zero_()
    for layer in module.children():
        if hasattr(layer, "quant_method"):
            layer.quant_method.process_weights_after_loading(layer)
    return module


def _inputs(module, tokens: int):
    g = torch.Generator().manual_seed(tokens)
    hidden = torch.randn(tokens, module.hyper_hidden_size, generator=g).to(BF16)
    block = torch.randn(tokens, module.hidden_size, generator=g).to(BF16)
    injection = (torch.randn(tokens, module.hc_count, generator=g) * 2).to(BF16)
    return hidden, block, injection


class _OpCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def _run(fn):
    """(result, {fused op name: calls}, {op name: calls}) of one call."""
    with torch.no_grad(), _OpCounter() as counter:
        result = fn()
    fused = {name[len(FUSED_PREFIX):]: n for name, n in counter.ops.items() if name.startswith(FUSED_PREFIX)}
    return result, fused, counter.ops


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _same_tuple(a, b) -> bool:
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))


def _torch_path(module, method: str, *args):
    module.hc_fused_requested = False
    try:
        result, fused, _ = _run(lambda: getattr(module, method)(*args))
    finally:
        module.hc_fused_requested = True
    assert fused == {}
    return result


# ----- the flag ---------------------------------------------------------------------------------------------------


def test_flag_default_and_declaration(monkeypatch):
    monkeypatch.delenv(IL_FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_HC_FUSED_IL == 0
    monkeypatch.setenv(IL_FLAG, "1")
    assert gaudi_envs.VLLM_HPU_HC_FUSED_IL == 1
    names = list(gaudi_envs.environment_variables)
    assert names.index(IL_FLAG) == names.index("VLLM_HPU_HC_FUSED_COPY_MODE") + 1


@pytest.mark.parametrize("fused,il,requested", [("1", None, False), ("1", "0", False), ("1", "2", False),
                                                ("1", "1", True), ("0", "1", False), (None, "1", False)])
def test_flag_is_read_once_at_construction(world1, vllm_ctx, monkeypatch, fused, il, requested):
    for name, value in (("VLLM_HPU_HC_FUSED", fused), (IL_FLAG, il)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    module = _module()
    assert module.hc_fused_il_requested is requested
    monkeypatch.setenv(IL_FLAG, "0" if requested else "1")  # later changes do not reach the layer
    hidden, block, injection = _inputs(module, 2)
    _, ops, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert ("hc_gate_mix_il" in ops) is requested and ("hc_combine_norm_il" in ops) is requested
    if fused == "1":
        assert ("hc_gate_mix" in ops) is not requested and ("hc_combine_norm" in ops) is not requested


# ----- the interleaved path gives the torch path's bits -----------------------------------------------------------


@pytest.mark.parametrize("geometry", [TINY, REAL], ids=["tiny", "real"])
@pytest.mark.parametrize("per_branch", [True, False], ids=["per_branch_norm", "shared_norm"])
@pytest.mark.parametrize("tokens", [1, 32, 128])
def test_il_path_equals_torch_path(world1, vllm_ctx, flags_on, warnings, geometry, per_branch, tokens):
    module = _module(*geometry, per_branch=per_branch)
    hidden, block, injection = _inputs(module, tokens)

    got, fused, _ = _run(lambda: module.mix(hidden))
    assert fused == {"hc_norm": 1, "hc_silu_split": 1, "hc_gate_mix_il": 1}
    assert _same_tuple(got, _torch_path(module, "mix", hidden))

    got, fused, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert fused == {"hc_combine_norm_il": 1, "hc_silu_split": 1, "hc_gate_mix_il": 1}
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))

    got, fused, _ = _run(lambda: module.combine(hidden, block, injection))
    assert fused == {"hc_combine": 1}  # the combine before a PLE layer has no interleaved kernel
    assert _same(got, _torch_path(module, "combine", hidden, block, injection))
    assert warnings == []


@pytest.mark.parametrize("tokens", [1, 32])
def test_final_mixer(world1, vllm_ctx, flags_on, warnings, tokens):
    module = _module(*REAL, use_combine=False, prefix="hyper_connection_mixer")
    hidden, block, injection = _inputs(module, tokens)
    got, fused, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert fused == {"hc_combine_norm_il": 1, "hc_silu": 1, "hc_gate_mix_il": 1}
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))
    assert warnings == []


@pytest.mark.parametrize("copy_mode", ["0", "1", "2"])
def test_a_layer_chain_gives_the_torch_chain_and_the_copies_of_the_shipped_ops(world1, vllm_ctx, monkeypatch,
                                                                                 warnings, copy_mode):
    """mix -> combine_and_mix -> combine_and_mix -> combine as a decoder layer wires them, in each copy mode: the
    interleaved ops take exactly the copies the shipped ops take, and the chain gives the torch chain's bits."""
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED_COPY_MODE", copy_mode)
    counts = {}
    outputs = {}
    for il in ("0", "1"):
        monkeypatch.setenv(IL_FLAG, il)
        attn = _module(prefix="layers.0.attn_hyper_connection")
        mlp = _module(prefix="layers.0.mlp_hyper_connection")
        hidden, block, _ = _inputs(attn, 8)
        block2 = torch.randn(8, attn.hidden_size, generator=torch.Generator().manual_seed(99)).to(BF16)

        def chain(attn=attn, mlp=mlp, hidden=hidden, block=block, block2=block2):
            h, block_input, injection = attn.mix(hidden)
            h, block_input2, injection2 = mlp.combine_and_mix(h, block, injection)
            h, block_input3, injection3 = attn.combine_and_mix(h, block2, injection2)
            return attn.combine(h, block, injection3), block_input, block_input2, block_input3

        outputs[il], fused, ops = _run(chain)
        counts[il] = ops["aten::clone"]
        if il == "1":
            assert fused == {"hc_norm": 1, "hc_combine_norm_il": 2, "hc_silu_split": 3, "hc_gate_mix_il": 3,
                             "hc_combine": 1}
    assert counts["0"] == counts["1"]
    assert all(_same(a, b) for a, b in zip(outputs["0"], outputs["1"]))
    assert warnings == []


# ----- not served: the shipped fused ops, one line ---------------------------------------------------------------


def test_missing_interleaved_ops_keep_the_shipped_fused_ops(world1, vllm_ctx, flags_on, warnings, monkeypatch):
    monkeypatch.setattr(hc_fused, "hc_fused_il_missing_ops", lambda device: ("hc_gate_mix_il", ))
    module = _module(prefix="layers.3.mlp_hyper_connection")
    hidden, block, injection = _inputs(module, 4)
    for _ in range(2):
        got, fused, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
        assert fused == {"hc_combine_norm": 1, "hc_silu_split": 1, "hc_gate_mix": 1}
        assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))
    assert module._hc_il is False and module._hc_fused_ok is True
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{IL_OFF} layers.3.mlp_hyper_connection: the hc_fused_ops extension has no cpu "
                                  "kernel for hc_gate_mix_il; using the shipped fused kernels")


def test_decision_is_made_once_per_layer(world1, vllm_ctx, flags_on, warnings, monkeypatch):
    module = _module()
    hidden, block, injection = _inputs(module, 2)
    _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert module._hc_il is True
    monkeypatch.setattr(hc_fused, "hc_fused_il_missing_ops", lambda device: ("hc_combine_norm_il", ))
    _, fused, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert "hc_combine_norm_il" in fused
    assert warnings == []


class _NoIlOps:
    """An op module that serves the interleaved ops on every device (the HPU check is the kernel library's)."""

    @staticmethod
    def hc_fused_il_missing_ops(device):
        return ()


@pytest.mark.parametrize("served,reason", [((), None), (("hc_combine_norm_il", ), "serves no TPC kernel for "
                                                                                   "hc_combine_norm_il")])
def test_on_hpu_the_kernel_library_is_asked(world1, vllm_ctx, flags_on, monkeypatch, served, reason):
    monkeypatch.setattr(_native, "hc_fused_il_missing_kernels", lambda: served)
    module = _module()
    got = module._hc_fused_il_reason(_NoIlOps, torch.device("hpu"))
    assert (got is None) if reason is None else (reason in got)
    assert module._hc_fused_il_reason(_NoIlOps, torch.device("cpu")) is None  # CPU: no kernel library


def test_a_library_that_cannot_be_asked_keeps_the_shipped_ops(world1, vllm_ctx, flags_on, monkeypatch):

    def broken():
        raise OSError("no such file")

    monkeypatch.setattr(_native, "hc_fused_il_missing_kernels", broken)
    reason = _module()._hc_fused_il_reason(_NoIlOps, torch.device("hpu"))
    assert reason.startswith("the kernel library cannot be asked for its kernels (OSError: no such file)")


# ----- the loader's question to the kernel library ---------------------------------------------------------------


@pytest.fixture
def fresh_il_state(monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_IL_MISSING", None)
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])


def test_missing_kernels_without_the_library(fresh_il_state, monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", None)
    assert _native.hc_fused_il_missing_kernels() == ("hc_gate_mix_il", "hc_combine_norm_il")


@pytest.mark.parametrize("served,missing", [
    (("hc_fused_gate_mix_bf16_gaudi2", "hc_fused_gate_mix_il_bf16_gaudi2", "hc_fused_combine_norm_il_bf16_gaudi2"),
     ()),
    (("hc_fused_gate_mix_bf16_gaudi2", "hc_fused_combine_norm_il_bf16_gaudi2"), ("hc_gate_mix_il", )),
    ((), ("hc_gate_mix_il", "hc_combine_norm_il")),
])
def test_missing_kernels_from_the_library_guids(fresh_il_state, monkeypatch, served, missing):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", True)
    asked = []
    monkeypatch.setattr(_native, "kernel_library_guids", lambda path: asked.append(path) or served)
    assert _native.hc_fused_il_missing_kernels() == missing
    assert _native.hc_fused_il_missing_kernels() == missing  # asked once, then remembered
    assert len(asked) == 1 and asked[0].endswith(_native.HC_FUSED_KERNEL_DB)
    assert bool(_native._LOAD_ERRORS) is bool(missing)


def test_missing_kernels_when_the_library_cannot_be_asked(fresh_il_state, monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", True)

    def broken(path):
        raise RuntimeError("GetKernelGuids returned 1")

    monkeypatch.setattr(_native, "kernel_library_guids", broken)
    assert _native.hc_fused_il_missing_kernels() == ("hc_gate_mix_il", "hc_combine_norm_il")
    assert "cannot list its kernels" in _native._LOAD_ERRORS[0]


def test_guids_agree_with_the_params_header():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[4] / "csrc" / "hc_fused" / "hc_fused_params.h").read_text()
    assert f'kHcFusedGateMixIlGuid = "{_native.HC_FUSED_IL_GUIDS["hc_gate_mix_il"]}"' in text
    assert f'kHcFusedCombineNormIlGuid = "{_native.HC_FUSED_IL_GUIDS["hc_combine_norm_il"]}"' in text
    assert tuple(_native.HC_FUSED_IL_GUIDS) == hc_fused.IL_OP_NAMES
