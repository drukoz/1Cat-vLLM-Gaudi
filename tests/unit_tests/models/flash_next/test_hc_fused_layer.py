# SPDX-License-Identifier: Apache-2.0
"""CPU tests of ``HpuGatedResidual`` with ``VLLM_HPU_HC_FUSED=1`` (the fused hyper-connection kernels).

On CPU the fused ops have no kernel, so these tests drive the layer through a stand-in:
``tests/unit_tests/hc_fused/hc_ops.py`` registers the torch oracle as the ops' CPU kernels.  The
oracle is a verbatim copy of the layer's torch functions (``test_reference_copy.py``), so the fused
path must return the torch path's bits exactly; a difference is a wiring error (argument order,
which output feeds what, where the combine is rounded).  A second stand-in, ``StrictOps``, replaces
the op module itself: it records every call and refuses what the HPU kernels' validation refuses.

What is proven here: the flag, the once-per-layer and per-call preconditions with their one log
line, the automatic fallback, and the wiring.  What is not: the kernels themselves (simulator tests,
``tests/standalone/hc_fused``) and anything on a card.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import collections  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel  # noqa: E402
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig  # noqa: E402

from flashinfer_gaudi import hc_fused  # noqa: E402
from tests.unit_tests.hc_fused import hc_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.hc_fused import hc_reference  # noqa: E402
from tests.unit_tests.hc_fused.hc_case import case_by_name, make_inputs  # noqa: E402
from vllm_gaudi.models.qwen4_exp import hyperconnection as layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
BF16 = torch.bfloat16
TINY = (128, 16)  # hidden_size, hc_lowrank of the tiny checkpoint
REAL = (2560, 320)  # of the real model
GEMM_OPS = ("aten::linear", "aten::addmm", "aten::mm", "aten::matmul", "aten::t")
FUSED_PREFIX = "flashinfer_gaudi::"
OFF_LINE = "HC fused (VLLM_HPU_HC_FUSED=1) is off for %s: %s; using the torch path"


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
def vllm_ctx():
    """vLLM linear layers read get_current_vllm_config() at construction."""
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")


@pytest.fixture
def warnings(monkeypatch):
    """Every line the layer module logs at warning level, formatted."""
    lines: list[str] = []
    monkeypatch.setattr(layer_module.logger, "warning", lambda message, *args: lines.append(message % args))
    return lines


def _module(hidden_size: int = 128, hc_lowrank: int = 16, *, per_branch: bool = True, use_combine: bool = True,
            hc_count: int = 4, dtype=BF16, prefix: str = "layers.0.attn_hyper_connection", eps: float = 1e-6):
    config = HyperConnectionConfig(hc_count=hc_count,
                                   hidden_size=hidden_size,
                                   params_dtype=dtype,
                                   hc_lowrank=hc_lowrank,
                                   rms_norm_eps=eps,
                                   hc_per_branch_norm=per_branch)
    module = HpuGatedResidual(config, use_combine=use_combine, prefix=prefix)
    g = torch.Generator().manual_seed(hidden_size + hc_lowrank + hc_count)
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


def _inputs(module, tokens: int, dtype=BF16):
    g = torch.Generator().manual_seed(tokens)
    hidden = torch.randn(tokens, module.hyper_hidden_size, generator=g).to(dtype)
    block = torch.randn(tokens, module.hidden_size, generator=g).to(dtype)
    injection = (torch.randn(tokens, module.hc_count, generator=g) * 2).to(dtype)
    return hidden, block, injection


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _same_tuple(a, b) -> bool:
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))


class _OpCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def _run(fn):
    """(result, {fused op name: calls}, number of all dispatched ops, number of GEMM ops) of one call."""
    with torch.no_grad(), _OpCounter() as counter:
        result = fn()
    fused = {name[len(FUSED_PREFIX):]: n for name, n in counter.ops.items() if name.startswith(FUSED_PREFIX)}
    gemm = sum(n for name, n in counter.ops.items() if name in GEMM_OPS)
    return result, fused, sum(counter.ops.values()), gemm


def _torch_path(module, method: str, *args):
    """The same call on the torch path of the same module (the flag is an attribute read per call)."""
    module.hc_fused_requested = False
    try:
        result, fused, _, _ = _run(lambda: getattr(module, method)(*args))
    finally:
        module.hc_fused_requested = True
    assert fused == {}
    return result


# ----- the flag ---------------------------------------------------------------------------------------------------


def test_default_is_the_torch_path(world1, vllm_ctx, monkeypatch, warnings):
    monkeypatch.delenv("VLLM_HPU_HC_FUSED", raising=False)
    module = _module()
    assert module.hc_fused_requested is False
    hidden, block, injection = _inputs(module, 4)
    for call in (lambda: module.mix(hidden), lambda: module.combine_and_mix(hidden, block, injection),
                 lambda: module.combine(hidden, block, injection)):
        _, fused, _, _ = _run(call)
        assert fused == {}
    assert warnings == []


@pytest.mark.parametrize("value,requested", [("0", False), ("1", True), ("2", False)])
def test_flag_is_read_once_at_construction(world1, vllm_ctx, monkeypatch, value, requested):
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", value)
    module = _module()
    assert module.hc_fused_requested is requested
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0" if requested else "1")  # later changes do not reach the layer
    hidden, _, _ = _inputs(module, 2)
    _, fused, _, _ = _run(lambda: module.mix(hidden))
    assert bool(fused) is requested


def test_layer_name_is_the_prefix(world1, vllm_ctx, flag_on):
    assert _module(prefix="layers.7.mlp_hyper_connection").layer_name == "layers.7.mlp_hyper_connection"
    assert _module(prefix="").layer_name == "HpuGatedResidual"


# ----- the fused path gives the torch path's bits ----------------------------------------------------------------


@pytest.mark.parametrize("geometry", [TINY, REAL], ids=["tiny", "real"])
@pytest.mark.parametrize("per_branch", [True, False], ids=["per_branch_norm", "shared_norm"])
@pytest.mark.parametrize("tokens", [1, 32])
def test_fused_path_equals_torch_path(world1, vllm_ctx, flag_on, warnings, geometry, per_branch, tokens):
    module = _module(*geometry, per_branch=per_branch)
    assert module.hc_count == 4 and module.hc_norm.weight.numel() == (4 if per_branch else 1) * geometry[0]
    hidden, block, injection = _inputs(module, tokens)

    got, fused, _, _ = _run(lambda: module.mix(hidden))
    assert fused == {"hc_norm": 1, "hc_silu_split": 1, "hc_gate_mix": 1}
    assert _same_tuple(got, _torch_path(module, "mix", hidden))
    assert got[0] is hidden  # mix returns its input, as the torch path does

    got, fused, _, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert fused == {"hc_combine_norm": 1, "hc_silu_split": 1, "hc_gate_mix": 1}
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))

    got, fused, _, _ = _run(lambda: module.combine(hidden, block, injection))
    assert fused == {"hc_combine": 1}
    assert _same(got, _torch_path(module, "combine", hidden, block, injection))
    assert warnings == []


@pytest.mark.parametrize("per_branch", [True, False], ids=["per_branch_norm", "shared_norm"])
def test_prefill_sized_call(world1, vllm_ctx, flag_on, warnings, per_branch):
    module = _module(*TINY, per_branch=per_branch)
    hidden, block, injection = _inputs(module, 2048)

    got, fused, _, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert fused == {"hc_combine_norm": 1, "hc_silu_split": 1, "hc_gate_mix": 1}
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))
    got, fused, _, _ = _run(lambda: module.mix(hidden))
    assert _same_tuple(got, _torch_path(module, "mix", hidden))
    assert warnings == []


@pytest.mark.parametrize("geometry", [TINY, REAL], ids=["tiny", "real"])
@pytest.mark.parametrize("tokens", [1, 32])
def test_final_mixer(world1, vllm_ctx, flag_on, warnings, geometry, tokens):
    module = _module(*geometry, use_combine=False, prefix="hyper_connection_mixer")
    hidden, block, injection = _inputs(module, tokens)

    got, fused, _, _ = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert fused == {"hc_combine_norm": 1, "hc_silu": 1, "hc_gate_mix": 1}  # hc_silu, not hc_silu_split
    assert got[2] is None
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))
    got, fused, _, _ = _run(lambda: module.mix(hidden))
    assert fused == {"hc_norm": 1, "hc_silu": 1, "hc_gate_mix": 1}
    assert _same_tuple(got, _torch_path(module, "mix", hidden))
    assert warnings == []


def test_a_layer_chain_gives_the_torch_chain(world1, vllm_ctx, flag_on, warnings):
    """mix -> combine_and_mix -> combine_and_mix -> combine, each fed by the one before, as a decoder layer
    wires them: the injection a fused mix returns (a new tensor, not a view) feeds the next fused combine."""
    attn, mlp = _module(prefix="layers.0.attn_hyper_connection"), _module(prefix="layers.0.mlp_hyper_connection")
    hidden, block, _ = _inputs(attn, 8)
    g = torch.Generator().manual_seed(99)
    block2 = torch.randn(8, attn.hidden_size, generator=g).to(BF16)

    def chain():
        h, block_input, injection = attn.mix(hidden)
        h, block_input2, injection2 = mlp.combine_and_mix(h, block, injection)
        h, block_input3, injection3 = attn.combine_and_mix(h, block2, injection2)
        return attn.combine(h, block, injection3), block_input, block_input2, block_input3, injection

    fused_out, fused, _, _ = _run(chain)
    assert fused == {"hc_norm": 1, "hc_combine_norm": 2, "hc_silu_split": 3, "hc_gate_mix": 3, "hc_combine": 1}
    assert fused_out[4].is_contiguous()
    for module in (attn, mlp):
        module.hc_fused_requested = False
    torch_out, fused, _, _ = _run(chain)
    assert fused == {}
    assert _same_tuple(fused_out, torch_out)
    assert warnings == []


def test_op_count_of_one_decode_step_call(world1, vllm_ctx, flag_on):
    """1 token, real geometry: three fused ops and the two GEMMs replace the torch ops of a hyper-connection."""
    module = _module(*REAL)
    hidden, block, injection = _inputs(module, 1)

    _, fused, total, gemm = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert sum(fused.values()) == 3 and total - gemm == 3
    module.hc_fused_requested = False
    _, _, torch_total, torch_gemm = _run(lambda: module.combine_and_mix(hidden, block, injection))
    assert torch_gemm == gemm  # the GEMMs are the same ops in both paths
    assert torch_total - torch_gemm >= 40
    module.hc_fused_requested = True
    _, fused, total, gemm = _run(lambda: module.combine(hidden, block, injection))
    assert fused == {"hc_combine": 1} and total == 1 and gemm == 0


# ----- the wiring, through a stand-in that records and validates -------------------------------------------------


class StrictOps:
    """A stand-in for ``flashinfer_gaudi.hc_fused``: the oracle behind the checks of the HPU validation callback
    (``csrc/hc_fused/pytorch/hpu_hc_fused.cpp``).  A call the card would refuse fails the test here."""

    OP_NAMES = hc_fused.OP_NAMES
    VECTOR_LANES = hc_fused.VECTOR_LANES
    MAX_STREAMS = hc_fused.MAX_STREAMS
    FP32_MIN_NORMAL = hc_fused.FP32_MIN_NORMAL
    FP32_MAX = hc_fused.FP32_MAX

    def __init__(self, combine_norm=hc_reference.hc_combine_norm_reference):
        self.calls: list[tuple] = []
        self.outputs: dict[str, object] = {}
        self._combine_norm = combine_norm

    def hc_fused_missing_ops(self, device):
        return ()

    @staticmethod
    def _matrix(tensor, name):
        assert tensor.dtype == BF16, name
        assert tensor.dim() == 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0, name
        assert tensor.is_contiguous(), name

    @staticmethod
    def _hidden(width, hc_count):
        assert 1 <= hc_count <= 64 and width % hc_count == 0
        hidden = width // hc_count
        assert hidden % 128 == 0
        return hidden

    def _weight(self, weight, first, hidden, width):
        assert weight.dtype == BF16 and weight.dim() == 1 and weight.shape[0] in (hidden, width)
        assert weight.is_contiguous() and weight.device == first.device

    def _done(self, name, args, out):
        self.calls.append((name, ) + args)
        self.outputs[name] = out
        return out

    def hc_norm(self, x, norm_weight, eps, hc_count):
        self._matrix(x, "x")
        self._weight(norm_weight, x, self._hidden(x.shape[1], hc_count), x.shape[1])
        assert isinstance(eps, float) and eps > 0 and isinstance(hc_count, int)
        return self._done("hc_norm", (x, norm_weight, eps, hc_count),
                          hc_reference.hc_norm_reference(x, norm_weight, eps, hc_count))

    def hc_silu(self, down, hc_count):
        self._matrix(down, "down")
        assert isinstance(hc_count, int) and 1 <= hc_count <= 64
        return self._done("hc_silu", (down, hc_count), hc_reference.hc_silu_reference(down, hc_count))

    def hc_silu_split(self, down, hc_count, lora_rank):
        self._matrix(down, "down")
        assert isinstance(hc_count, int) and isinstance(lora_rank, int)
        assert 1 <= hc_count <= 64 and lora_rank >= 1 and lora_rank + hc_count <= down.shape[1]
        return self._done("hc_silu_split", (down, hc_count, lora_rank),
                          hc_reference.hc_silu_split_reference(down, hc_count, lora_rank))

    def hc_gate_mix(self, xn, gate, hc_count):
        self._matrix(xn, "xn")
        self._hidden(xn.shape[1], hc_count)
        self._matrix(gate, "gate")
        assert gate.shape == xn.shape and gate.device == xn.device and isinstance(hc_count, int)
        return self._done("hc_gate_mix", (xn, gate, hc_count), hc_reference.hc_gate_mix_reference(xn, gate, hc_count))

    def _combine(self, residual, block_output, injection, hc_count):
        self._matrix(residual, "residual")
        hidden = self._hidden(residual.shape[1], hc_count)
        self._matrix(block_output, "block_output")
        assert tuple(block_output.shape) == (residual.shape[0], hidden)
        self._matrix(injection, "injection")
        assert tuple(injection.shape) == (residual.shape[0], hc_count)
        assert block_output.device == residual.device and injection.device == residual.device
        assert isinstance(hc_count, int)
        return hidden

    def hc_combine(self, residual, block_output, injection, hc_count):
        self._combine(residual, block_output, injection, hc_count)
        return self._done("hc_combine", (residual, block_output, injection, hc_count),
                          hc_reference.hc_combine_reference(residual, block_output, injection, hc_count))

    def hc_combine_norm(self, residual, block_output, injection, norm_weight, eps, hc_count):
        hidden = self._combine(residual, block_output, injection, hc_count)
        self._weight(norm_weight, residual, hidden, residual.shape[1])
        assert isinstance(eps, float) and eps > 0
        return self._done("hc_combine_norm", (residual, block_output, injection, norm_weight, eps, hc_count),
                          self._combine_norm(residual, block_output, injection, norm_weight, eps, hc_count))


@pytest.fixture
def strict(monkeypatch):
    ops = StrictOps()
    monkeypatch.setattr(layer_module, "_hc_fused_ops", lambda: ops)
    return ops


def test_combine_and_mix_wiring(world1, vllm_ctx, flag_on, strict, warnings):
    module = _module(*TINY)
    hidden, block, injection = _inputs(module, 3)
    with torch.no_grad():
        out_hidden, block_input, out_injection = module.combine_and_mix(hidden, block, injection)

    assert [call[0] for call in strict.calls] == ["hc_combine_norm", "hc_silu_split", "hc_gate_mix"]
    combine_norm, silu_split, gate_mix = strict.calls
    # hc_combine_norm(residual, block_output, injection, norm_weight, eps, hc_count)
    assert combine_norm[1] is hidden and combine_norm[2] is block and combine_norm[3] is injection
    assert combine_norm[4] is module.hc_norm.weight and combine_norm[5:] == (1e-6, 4)
    residual_out, xn = strict.outputs["hc_combine_norm"]
    # hc_silu_split(down, hc_count, lora_rank): down is the merged GEMM of xn, with its padding columns
    with torch.no_grad():
        down = module.input_mix_weight_down_block_inject(xn)
    assert tuple(down.shape) == (3, 16 + 4 + 12) and torch.equal(silu_split[1], down)
    assert silu_split[2:] == (4, 16)
    lora, injection_out = strict.outputs["hc_silu_split"]
    # hc_gate_mix(xn, gate, hc_count): xn is the norm output, gate the up GEMM of the silu output
    with torch.no_grad():
        gate = module.input_mix_weight_up(lora)
    assert gate_mix[1] is xn and torch.equal(gate_mix[2], gate) and gate_mix[3] == 4
    # which output goes where
    assert out_hidden is residual_out  # the combined residual, not the input and not the norm
    assert block_input is strict.outputs["hc_gate_mix"]
    assert out_injection is injection_out
    assert warnings == []


def test_mix_wiring(world1, vllm_ctx, flag_on, strict):
    module = _module(*TINY, per_branch=False)
    hidden, _, _ = _inputs(module, 2)
    with torch.no_grad():
        out_hidden, block_input, out_injection = module.mix(hidden)

    assert [call[0] for call in strict.calls] == ["hc_norm", "hc_silu_split", "hc_gate_mix"]
    norm = strict.calls[0]
    assert norm[1] is hidden and norm[2] is module.hc_norm.weight and norm[3:] == (1e-6, 4)
    assert tuple(norm[2].shape) == (128, )  # the shared norm weight goes in as it is
    xn = strict.outputs["hc_norm"]
    assert strict.calls[2][1] is xn
    assert out_hidden is hidden and block_input is strict.outputs["hc_gate_mix"]
    assert out_injection is strict.outputs["hc_silu_split"][1]


def test_final_mixer_wiring(world1, vllm_ctx, flag_on, strict):
    module = _module(*TINY, use_combine=False)
    hidden, block, injection = _inputs(module, 2)
    with torch.no_grad():
        out_hidden, block_input, out_injection = module.combine_and_mix(hidden, block, injection)

    assert [call[0] for call in strict.calls] == ["hc_combine_norm", "hc_silu", "hc_gate_mix"]
    residual_out, xn = strict.outputs["hc_combine_norm"]
    with torch.no_grad():
        down = module.input_mix_weight_down(xn)
    assert tuple(down.shape) == (2, 16) and torch.equal(strict.calls[1][1], down) and strict.calls[1][2] == 4
    assert out_hidden is residual_out and block_input is strict.outputs["hc_gate_mix"] and out_injection is None


def test_combine_wiring(world1, vllm_ctx, flag_on, strict):
    module = _module(*TINY)
    hidden, block, injection = _inputs(module, 2)
    with torch.no_grad():
        out = module.combine(hidden, block, injection)

    (call, ) = strict.calls
    assert call[0] == "hc_combine" and call[1] is hidden and call[2] is block and call[3] is injection and call[4] == 4
    assert out is strict.outputs["hc_combine"]


def _combine_norm_without_the_rounding(residual, block_output, injection, norm_weight, eps, hc_count):
    """The WRONG hc_combine_norm: the right combine, and the norm of the combine before it was rounded."""
    out, _ = hc_reference.hc_combine_norm_reference(residual, block_output, injection, norm_weight, eps, hc_count)
    return out, hc_reference.hc_combine_norm_unrounded(residual, block_output, injection, norm_weight, eps, hc_count)


@pytest.mark.parametrize("case_name", ["combine_norm_tiny_rounding", "combine_norm_real_rounding_shared"])
def test_the_combine_is_rounded_before_the_norm(world1, vllm_ctx, flag_on, monkeypatch, warnings, case_name):
    """On inputs where the norm of the rounded combine and the norm of the fp32 combine differ, the fused
    path equals the torch path, and a stand-in that normalizes the unrounded combine does not."""
    case = case_by_name(case_name)
    inputs = make_inputs(case)
    module = _module(case.hidden, 320 if case.hidden == 2560 else 16, per_branch=not case.shared_weight)
    with torch.no_grad():
        module.hc_norm.weight.copy_(inputs["norm_weight"])
    args = (inputs["residual"], inputs["block_output"], inputs["injection"])

    got, fused, _, _ = _run(lambda: module.combine_and_mix(*args))
    assert fused["hc_combine_norm"] == 1
    want = _torch_path(module, "combine_and_mix", *args)
    assert _same_tuple(got, want)

    mutant = StrictOps(combine_norm=_combine_norm_without_the_rounding)
    monkeypatch.setattr(layer_module, "_hc_fused_ops", lambda: mutant)
    with torch.no_grad():
        wrong = module.combine_and_mix(*args)
    assert [call[0] for call in mutant.calls] == ["hc_combine_norm", "hc_silu_split", "hc_gate_mix"]
    assert _same(wrong[0], want[0])  # the combine itself is the same
    assert not torch.equal(wrong[1], want[1])  # the block input is not: the rounding point is pinned
    assert warnings == []  # both runs were fused: no fallback hides the difference


def test_no_host_read_on_the_fused_path(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    """The whole fused path on meta tensors (shapes and dtypes, no data): any host read would raise."""
    monkeypatch.setattr(hc_fused, "hc_fused_missing_ops", lambda device: ())
    for use_combine in (True, False):
        module = _module(*TINY, use_combine=use_combine).to("meta")
        hidden, block, injection = (t.to("meta") for t in _inputs(module, 5))
        with torch.no_grad():
            out_hidden, block_input, out_injection = module.combine_and_mix(hidden, block, injection)
            mixed = module.mix(hidden)
            combined = module.combine(hidden, block, injection)
        assert tuple(out_hidden.shape) == (5, 512) and tuple(block_input.shape) == (5, 128)
        assert out_hidden.dtype == block_input.dtype == BF16 and out_hidden.device.type == "meta"
        assert (out_injection is None) is (not use_combine)
        if use_combine:
            assert tuple(out_injection.shape) == (5, 4)
        assert tuple(mixed[1].shape) == (5, 128) and tuple(combined.shape) == (5, 512)
    assert warnings == []


# ----- preconditions: one line per layer, then the torch path ---------------------------------------------------


def _expect_fallback(module, warnings, reason_part: str, method: str, *args):
    """The call runs no fused op, returns the torch path's result and logged exactly one line for the layer."""
    got, fused, _, _ = _run(lambda: getattr(module, method)(*args))
    assert fused == {}
    want = _torch_path(module, method, *args)
    assert _same_tuple(got, want) if isinstance(got, tuple) else _same(got, want)
    assert len(warnings) == 1, warnings
    prefix, _, rest = warnings[0].partition(": ")
    assert prefix == "HC fused (VLLM_HPU_HC_FUSED=1) is off for %s" % module.layer_name
    assert rest.endswith("; using the torch path") and reason_part in rest, warnings[0]
    getattr(module, method)(*args)  # a second call logs nothing more
    assert len(warnings) == 1


def test_extension_not_loaded(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    monkeypatch.setattr(hc_fused, "hc_fused_missing_ops", lambda device: hc_fused.OP_NAMES)
    module = _module(prefix="layers.3.attn_hyper_connection")
    hidden, block, injection = _inputs(module, 2)

    _expect_fallback(module, warnings, "the hc_fused_ops extension is not loaded", "combine_and_mix", hidden, block,
                     injection)
    assert warnings[0] == OFF_LINE % ("layers.3.attn_hyper_connection",
                                      "the hc_fused_ops extension is not loaded (no cpu kernel for hc_norm, hc_silu, "
                                      "hc_silu_split, hc_gate_mix, hc_combine, hc_combine_norm)")
    for call in (lambda: module.mix(hidden), lambda: module.combine(hidden, block, injection)):
        _, fused, _, _ = _run(call)
        assert fused == {}
    assert len(warnings) == 1  # one line per layer, whatever it is called with


def test_one_missing_kernel_is_enough(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    monkeypatch.setattr(hc_fused, "hc_fused_missing_ops", lambda device: ("hc_gate_mix", ))
    module = _module()
    hidden, _, _ = _inputs(module, 2)
    _expect_fallback(module, warnings, "no cpu kernel for hc_gate_mix)", "mix", hidden)


def test_op_module_that_cannot_be_imported(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    def broken():
        raise ImportError("no module named flashinfer_gaudi.hc_fused")

    monkeypatch.setattr(layer_module, "_hc_fused_ops", broken)
    module = _module()
    hidden, _, _ = _inputs(module, 2)
    _expect_fallback(module, warnings, "cannot be imported (ImportError", "mix", hidden)


def test_three_dimensional_input(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden, block, injection = (t.unsqueeze(0) for t in _inputs(module, 3))
    _expect_fallback(module, warnings, "hidden states are 3-D", "combine_and_mix", hidden, block, injection)


def test_float32_layer(world1, vllm_ctx, flag_on, warnings):
    module = _module(dtype=torch.float32)
    hidden, block, injection = _inputs(module, 3, dtype=torch.float32)
    _expect_fallback(module, warnings, "hc_norm.weight is torch.float32", "combine_and_mix", hidden, block, injection)


def test_float32_input_of_a_bf16_layer(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden, block, injection = _inputs(module, 3)
    _expect_fallback(module, warnings, "hidden states are torch.float32", "combine", hidden.float(), block, injection)


def test_hidden_size_that_is_not_whole_vectors(world1, vllm_ctx, flag_on, warnings):
    module = _module(64, 20)
    hidden, _, _ = _inputs(module, 2)
    _expect_fallback(module, warnings, "hidden_size 64 is not a multiple of 128", "mix", hidden)


def test_too_many_streams(world1, vllm_ctx, flag_on, warnings):
    module = _module(128, 16, hc_count=65)
    hidden, _, _ = _inputs(module, 1)
    _expect_fallback(module, warnings, "hc_count 65 is outside [1, 64]", "mix", hidden)


def test_eps_that_is_not_a_normal_fp32(world1, vllm_ctx, flag_on, warnings):
    module = _module(eps=1e-45)
    hidden, _, _ = _inputs(module, 1)
    _expect_fallback(module, warnings, "rms_norm_eps 1e-45 is not a positive normal fp32 value", "mix", hidden)


def test_non_contiguous_hidden_states(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden, _, _ = _inputs(module, 3)
    wide = torch.cat([hidden, hidden], dim=1)[:, :module.hyper_hidden_size]
    assert not wide.is_contiguous()
    _expect_fallback(module, warnings, "hidden states are not contiguous", "mix", wide)


def test_injection_that_is_a_view_of_the_torch_path(world1, vllm_ctx, flag_on, warnings):
    """A torch-path mix returns its injection as a column slice; a fused combine does not take it."""
    module = _module()
    hidden, block, _ = _inputs(module, 3)
    injection = _torch_path(module, "mix", hidden)[2]
    assert not injection.is_contiguous()
    _expect_fallback(module, warnings, "the injection is not contiguous", "combine_and_mix", hidden, block, injection)


def test_block_output_of_another_dtype(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden, block, injection = _inputs(module, 3)
    _expect_fallback(module, warnings, "the block output is torch.float32", "combine", hidden, block.float(),
                     injection)


def test_block_output_of_another_shape(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden, block, injection = _inputs(module, 3)
    got, fused, _, _ = _run(lambda: module.combine(hidden, block[:1], injection))  # broadcasts on the torch path
    assert fused == {} and len(warnings) == 1 and "the block output is torch.bfloat16 (1, 128)" in warnings[0]
    assert _same(got, _torch_path(module, "combine", hidden, block[:1], injection))


def test_no_tokens(world1, vllm_ctx, flag_on, warnings):
    module = _module()
    hidden = torch.empty(0, module.hyper_hidden_size, dtype=BF16)
    _expect_fallback(module, warnings, "hidden states are (0, 512)", "mix", hidden)


def test_a_later_input_the_kernels_do_not_take(world1, vllm_ctx, flag_on, warnings):
    """The layer check passed on the first call; a later 3-D call runs torch and logs the layer's one line."""
    module = _module()
    hidden, block, injection = _inputs(module, 3)
    _, fused, _, _ = _run(lambda: module.mix(hidden))
    assert fused and warnings == []

    _expect_fallback(module, warnings, "hidden states are 3-D", "mix", hidden.unsqueeze(0))
    _, fused, _, _ = _run(lambda: module.mix(hidden))  # and 2-D calls stay fused
    assert fused == {"hc_norm": 1, "hc_silu_split": 1, "hc_gate_mix": 1}
    assert len(warnings) == 1


def test_each_layer_logs_its_own_line(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    monkeypatch.setattr(hc_fused, "hc_fused_missing_ops", lambda device: hc_fused.OP_NAMES)
    layers = [_module(prefix="layers.%d.mlp_hyper_connection" % index) for index in range(3)]
    hidden, _, _ = _inputs(layers[0], 2)
    for _ in range(3):
        for layer in layers:
            layer.mix(hidden)
    assert len(warnings) == 3
    assert [line.split(" is off for ")[1].split(":")[0] for line in warnings] == [
        "layers.0.mlp_hyper_connection", "layers.1.mlp_hyper_connection", "layers.2.mlp_hyper_connection"
    ]


def test_an_op_that_raises_turns_the_layer_to_the_torch_path(world1, vllm_ctx, flag_on, monkeypatch, warnings):
    """No op writes an input, so the torch path can take over a call a fused op refused."""
    ops = StrictOps()
    calls = []

    def refuse(*args):
        calls.append(1)
        raise RuntimeError("HC fused gate must have the shape of xn")

    ops.hc_gate_mix = refuse
    monkeypatch.setattr(layer_module, "_hc_fused_ops", lambda: ops)
    module = _module()
    hidden, block, injection = _inputs(module, 3)

    with torch.no_grad():
        got = module.combine_and_mix(hidden, block, injection)
    assert _same_tuple(got, _torch_path(module, "combine_and_mix", hidden, block, injection))
    assert len(warnings) == 1
    assert "a fused op raised RuntimeError: HC fused gate must have the shape of xn" in warnings[0]
    with torch.no_grad():
        module.combine_and_mix(hidden, block, injection)
        module.mix(hidden)
    assert len(calls) == 1 and len(warnings) == 1  # the layer does not try again


# ----- a process without the stand-in ---------------------------------------------------------------------------

_PLAIN_PROCESS = r"""
import os, sys, tempfile
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm_gaudi.models.qwen4_exp import hyperconnection as layer_module
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual

lines = []
for level in ("warning", "info", "error"):
    setattr(layer_module.logger, level, lambda message, *args, level=level: lines.append(level + ": " + message % args))
config = HyperConnectionConfig(hc_count=4, hidden_size=128, params_dtype=torch.bfloat16, hc_lowrank=16,
                               rms_norm_eps=1e-6, hc_per_branch_norm=True)
_, temp_file = tempfile.mkstemp(dir="/tmp")
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{temp_file}", local_rank=0,
                                 backend="gloo")
    initialize_model_parallel(1, 1)
    layer = HpuGatedResidual(config, prefix="layers.0.attn_hyper_connection")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_((torch.randn(param.shape, generator=g) * 0.05).to(param.dtype))
    for child in layer.children():
        if hasattr(child, "quant_method"):
            child.quant_method.process_weights_after_loading(child)
    hidden = torch.randn(3, 512, generator=g).to(torch.bfloat16)
    block = torch.randn(3, 128, generator=g).to(torch.bfloat16)
    with torch.no_grad():
        h, block_input, injection = layer.mix(hidden)
        h2, block_input2, _ = layer.combine_and_mix(h, block, injection)
        out = layer.combine(h2, block, injection)
print("REQUESTED", layer.hc_fused_requested)
print("OP_MODULE_IMPORTED", "flashinfer_gaudi.hc_fused" in sys.modules)
print("HC_OPS_DEFINED", any(name.startswith("flashinfer_gaudi::hc_") for name in torch._C._dispatch_get_all_op_names()))
print("LINES", len(lines))
for line in lines:
    print("LINE", line)
print("SUM", float(out.float().sum()), float(block_input2.float().sum()))
"""


def _plain_process(flag: str | None) -> dict:
    env = dict(os.environ)
    env.pop("VLLM_HPU_HC_FUSED", None)
    if flag is not None:
        env["VLLM_HPU_HC_FUSED"] = flag
    proc = subprocess.run([sys.executable, "-c", _PLAIN_PROCESS], cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out: dict = {"LINE": []}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "LINE":
            out["LINE"].append(value)
        elif key in ("REQUESTED", "OP_MODULE_IMPORTED", "HC_OPS_DEFINED", "LINES", "SUM"):
            out[key] = value
    return out


def test_with_the_flag_off_nothing_is_imported_registered_or_logged():
    for flag in (None, "0"):
        out = _plain_process(flag)
        assert out["REQUESTED"] == "False"
        assert out["OP_MODULE_IMPORTED"] == "False"
        assert out["HC_OPS_DEFINED"] == "False"
        assert out["LINES"] == "0"


def test_without_a_kernel_the_flag_changes_no_result_and_logs_one_line_per_layer():
    """A process that only has the plugin (no extension, no stand-in): the default path's numbers."""
    off, on = _plain_process("0"), _plain_process("1")
    assert on["REQUESTED"] == "True"
    assert on["SUM"] == off["SUM"]
    assert on["LINE"] == [
        "warning: " + OFF_LINE % ("layers.0.attn_hyper_connection",
                                  "the hc_fused_ops extension is not loaded (no cpu kernel for hc_norm, hc_silu, "
                                  "hc_silu_split, hc_gate_mix, hc_combine, hc_combine_norm)")
    ]
