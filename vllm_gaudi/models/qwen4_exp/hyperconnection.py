# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — HPU pure-torch variant.

Drop-in replacement for upstream
``vllm/models/qwen4_exp/nvidia/hyperconnection.py::GatedResidual``: same
constructor, weight names and ``mix`` / ``combine_and_mix`` / ``combine``
API, but the five HC glue kernels of ``nvidia/ops/hc.py`` are replaced by
plain PyTorch implementations (no Triton, no custom-op registration, no
CUDA checks). The math mirrors the Triton kernel bodies: fp32 inside,
cast back to the input dtype at the end.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout). Both 2-D
``[tokens, HC*HS]`` and 3-D ``[batch, seq, HC*HS]`` inputs are supported.

With ``VLLM_HPU_HC_FUSED=1`` (default 0) the glue between and around the two
GEMMs runs as fused TPC kernels instead (``flashinfer_gaudi.hc_fused``; see
``docs/design/flash-next/hc-fused.md``): one op each for the norm, the silu
(with the injection split), the gated mix and the combine. A layer whose
kernels are not loaded, or whose input they do not take, logs one line and
runs the torch code below, which is unchanged.
"""

import importlib

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.qwen4_exp.common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from vllm_gaudi import envs

logger = init_logger(__name__)

# The fused hyper-connection ops. Importing the module defines their schemas, so it is imported on a layer's
# first call and only when that layer was built with VLLM_HPU_HC_FUSED=1.
_HC_FUSED_OPS_MODULE = "flashinfer_gaudi.hc_fused"


def _hc_fused_ops():
    """The op module of the fused hyper-connection kernels (``flashinfer_gaudi.hc_fused``)."""
    return importlib.import_module(_HC_FUSED_OPS_MODULE)


# ---------------------------------------------------------------------------
# Pure-torch replacements for the five glue kernels in ``nvidia/ops/hc.py``
# ---------------------------------------------------------------------------
def grouped_gemma_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int) -> torch.Tensor:
    """Grouped Gemma RMSNorm (torch replacement of ``grouped_gemma_rmsnorm``).

    Each group of ``x.shape[-1] // num_groups`` elements is normalized
    independently with Gemma's ``(1 + w)`` affine; ``weight`` holds either
    one shared affine per group (``numel == group_dim``) or a distinct
    affine for every element of the flat layout (``numel == DIM``).
    """
    dim = x.shape[-1]
    group_dim = dim // num_groups
    assert dim % num_groups == 0
    assert weight.numel() in (group_dim, dim)

    input_dtype = x.dtype
    xg = x.float().unflatten(-1, (num_groups, group_dim))
    rrms = torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + eps)
    y = xg * rrms
    w = weight.float()
    w = w.view(group_dim) if weight.numel() == group_dim else w.view(num_groups, group_dim)
    # Gemma's (1 + w) affine, written like the kernel to stay an FMA.
    y = y + y * w
    return y.flatten(-2).to(input_dtype)


def hc_silu_torch(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Silu with the ``/ hc_count`` prescale (torch replacement of ``hc_silu``)."""
    input_dtype = x.dtype
    xf = x.float() / hc_count
    return (xf * torch.sigmoid(xf)).to(input_dtype)


def hc_gate_mix_torch(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Sigmoid-gated mean over the HC streams (torch replacement of ``hc_gate_mix``).

    ``x`` and ``gate`` share the flat ``[..., HC*HS]`` layout; the output is
    the single ``[..., HS]`` block input.
    """
    assert x.shape == gate.shape
    dim = gate.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count

    input_dtype = x.dtype
    xg = x.float().unflatten(-1, (hc_count, hc_dim))
    g = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, hc_dim))
    acc = (g * xg).sum(dim=-2) / hc_count
    return acc.to(input_dtype)


def hc_combine_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Inject a block output into the HC streams (torch replacement of ``hc_combine``).

    The injection logits become weights via ``2 * sigmoid(logits / hc_count)``,
    exactly as the combine kernel computes them.
    """
    dim = residual.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count
    assert block_output.shape[-1] == hc_dim
    assert injection_logits.shape[-1] == hc_count

    input_dtype = residual.dtype
    w = (2.0 * torch.sigmoid(injection_logits.float() / hc_count)).unsqueeze(-1)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    block = block_output.float().unsqueeze(-2)
    return (res + block * w).flatten(-2).to(input_dtype)


def hc_combine_norm_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused combine + grouped Gemma RMSNorm (torch replacement of ``hc_combine_norm``).

    Returns the combined multi-stream residual **and** its grouped norm.
    The combine result is rounded to the input dtype before normalization,
    matching the fused kernel's materialized combine -> RMSNorm boundary.
    """
    dim = residual.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count
    assert block_output.shape[-1] == hc_dim
    assert injection_logits.shape[-1] == hc_count
    assert norm_weight.numel() in (hc_dim, dim)

    input_dtype = residual.dtype
    w = (2.0 * torch.sigmoid(injection_logits.float() / hc_count)).unsqueeze(-1)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    block = block_output.float().unsqueeze(-2)
    out = (res + block * w).flatten(-2).to(input_dtype)

    xn = grouped_gemma_rmsnorm_torch(out, norm_weight, eps, hc_count)
    return out, xn


# ---------------------------------------------------------------------------
# Gated-residual variant (pure torch)
# ---------------------------------------------------------------------------
class HpuGatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the input RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch applies through the standard quant_method mechanism. The
    five glue ops are pure PyTorch (see module docstring).
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine
        self.layer_name = prefix or type(self).__name__
        # Fused hyper-connection kernels: the flag is read once, here. 1 = fused; anything else = torch.
        self.hc_fused_requested = envs.VLLM_HPU_HC_FUSED == 1
        self._hc_fused_ok: bool | None = None  # None = the once-per-layer check has not run yet
        self._hc_fused_logged = False  # the layer's one "is off" line was written

        norm_size = (self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size)
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count] + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
            if self.pad_size:
                # The checkpoint never fills the padding rows; zero them so they never hold garbage (or NaN).
                with torch.no_grad():
                    self.input_mix_weight_down_block_inject.weight[self.lora_rank + self.hc_count:].zero_()
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    # -- fused kernels (VLLM_HPU_HC_FUSED=1) --------------------------------
    def _hc_fused_log_off(self, reason: str) -> None:
        """One line per layer, the first time it runs the torch path although the flag is on."""
        if self._hc_fused_logged:
            return
        self._hc_fused_logged = True
        logger.warning("HC fused (VLLM_HPU_HC_FUSED=1) is off for %s: %s; using the torch path", self.layer_name,
                       reason)

    def _hc_fused_layer_reason(self, device: torch.device) -> str | None:
        """Why the fused kernels cannot serve this layer at all, or ``None``: what does not change per call.

        The list is what the ops' validation callback and the kernel glue check (``hpu_hc_fused.cpp``,
        ``hc_fused_gaudi2.cpp``): an input they refuse raises inside the op, so the layer checks first.
        """
        try:
            ops = _hc_fused_ops()
        except Exception as exc:  # noqa: BLE001 - whatever stops the import, this layer runs torch
            return f"{_HC_FUSED_OPS_MODULE} cannot be imported ({type(exc).__name__}: {exc})"
        missing = ops.hc_fused_missing_ops(device)
        if missing:
            # No HPU kernel means no extension, or no kernel database for its GUIDs (the loader registers
            # the kernels only after the database is in GC_KERNEL_PATH).
            return f"the hc_fused_ops extension is not loaded (no {device.type} kernel for {', '.join(missing)})"
        if not 1 <= self.hc_count <= ops.MAX_STREAMS:
            return f"hc_count {self.hc_count} is outside [1, {ops.MAX_STREAMS}]"
        if self.hidden_size % ops.VECTOR_LANES != 0:
            return f"hidden_size {self.hidden_size} is not a multiple of {ops.VECTOR_LANES} (one TPC vector)"
        if self.lora_rank < 1:
            return f"hc_lowrank {self.lora_rank} is below 1"
        eps = float(self.config.rms_norm_eps)
        if not ops.FP32_MIN_NORMAL <= eps <= ops.FP32_MAX:
            return f"rms_norm_eps {eps!r} is not a positive normal fp32 value"
        weight = self.hc_norm.weight
        if (weight.dtype != torch.bfloat16 or weight.dim() != 1
                or weight.numel() not in (self.hidden_size, self.hyper_hidden_size) or not weight.is_contiguous()):
            return (f"hc_norm.weight is {weight.dtype} {tuple(weight.shape)}, the fused ops take a contiguous 1-D "
                    f"bfloat16 weight of {self.hidden_size} or {self.hyper_hidden_size} elements")
        down = "input_mix_weight_down_block_inject" if self.use_combine else "input_mix_weight_down"
        for name in (down, "input_mix_weight_up"):
            projection = getattr(getattr(self, name), "weight", None)
            if projection is None or projection.dtype != torch.bfloat16:
                found = "missing" if projection is None else str(projection.dtype)
                return f"{name}.weight is {found}, the fused ops take the bfloat16 output of the GEMMs"
        return None

    def _hc_fused_call_reason(self, hidden_states: torch.Tensor, combine: bool, block_output,
                              injection) -> str | None:
        """Why the fused kernels cannot take this call's inputs, or ``None``.

        Ranks, dtypes, sizes, memory layout and devices only: nothing here reads tensor data.
        """
        if hidden_states.dim() != 2:
            return f"hidden states are {hidden_states.dim()}-D, the fused ops take [tokens, width]"
        if hidden_states.dtype != torch.bfloat16:
            return f"hidden states are {hidden_states.dtype}, the fused ops take bfloat16"
        tokens = hidden_states.shape[0]
        if tokens < 1 or hidden_states.shape[1] != self.hyper_hidden_size:
            return (f"hidden states are {tuple(hidden_states.shape)}, the fused ops take "
                    f"[tokens >= 1, {self.hyper_hidden_size}]")
        if not hidden_states.is_contiguous():
            return "hidden states are not contiguous"
        device = hidden_states.device
        if self.hc_norm.weight.device != device:
            return f"hc_norm.weight is on {self.hc_norm.weight.device}, the hidden states on {device}"
        if not combine:
            return None
        for name, tensor, columns in (("block output", block_output, self.hidden_size), ("injection", injection,
                                                                                         self.hc_count)):
            if not isinstance(tensor, torch.Tensor):
                return f"the {name} is not a tensor"
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (tokens, columns):
                return (f"the {name} is {tensor.dtype} {tuple(tensor.shape)}, the fused ops take bfloat16 "
                        f"[{tokens}, {columns}]")
            if not tensor.is_contiguous():
                return f"the {name} is not contiguous"
            if tensor.device != device:
                return f"the {name} is on {tensor.device}, the hidden states on {device}"
        return None

    def _hc_fused(self, hidden_states: torch.Tensor, combine: bool = False, block_output=None, injection=None):
        """The fused op module when this call runs the fused kernels, else ``None`` (the torch path).

        The layer check runs once, on the layer's first call; the input check runs on every call. Either one
        failing writes the layer's one log line.
        """
        if not self.hc_fused_requested:
            return None
        if self._hc_fused_ok is None:
            reason = self._hc_fused_layer_reason(hidden_states.device)
            self._hc_fused_ok = reason is None
            if reason is not None:
                self._hc_fused_log_off(reason)
        if not self._hc_fused_ok:
            return None
        reason = self._hc_fused_call_reason(hidden_states, combine, block_output, injection)
        if reason is not None:
            self._hc_fused_log_off(reason)
            return None
        return _hc_fused_ops()

    def _hc_fused_raised(self, exc: Exception) -> None:
        """A fused op raised when it was called (it refused its inputs, or its extension is broken).

        No op writes an input, so the caller runs the torch path on the same inputs; the layer stays on it.
        """
        self._hc_fused_ok = False
        self._hc_fused_log_off(f"a fused op raised {type(exc).__name__}: {exc}")

    def _hc_fused_mix_tail(self, ops, hidden_states: torch.Tensor,
                           xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """GEMM, silu (and the injection split), GEMM, gated mix: the part ``mix`` and ``combine_and_mix`` share."""
        if self.use_combine:
            down_and_injection = self.input_mix_weight_down_block_inject(xn)  # GEMM 1 (matrix engine)
            lora, injection = ops.hc_silu_split(down_and_injection, self.hc_count, self.lora_rank)
        else:
            lora = ops.hc_silu(self.input_mix_weight_down(xn), self.hc_count)
            injection = None
        gate = self.input_mix_weight_up(lora)  # GEMM 2 (matrix engine)
        block_input = ops.hc_gate_mix(xn, gate, self.hc_count)
        return hidden_states, block_input, injection

    def _hc_fused_mix(self, ops,
                      hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = ops.hc_norm(hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        return self._hc_fused_mix_tail(ops, hidden_states, xn)

    def _hc_fused_combine_and_mix(
        self,
        ops,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # One kernel: the combine, rounded to bf16, and the norm of the rounded result.
        hidden_states, xn = ops.hc_combine_norm(hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,
                                                self.config.rms_norm_eps, self.hc_count)
        return self._hc_fused_mix_tail(ops, hidden_states, xn)

    # -- torch path (default), with the fused path in front of it ------------
    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ops = self._hc_fused(hidden_states)
        if ops is not None:
            try:
                return self._hc_fused_mix(ops, hidden_states)
            except Exception as exc:  # noqa: BLE001 - the torch path below takes over
                self._hc_fused_raised(exc)

        xn = grouped_gemma_rmsnorm_torch(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu_torch(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [..., D]
        block_input = hc_gate_mix_torch(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        ops = self._hc_fused(hidden_states, True, prev_block_output, prev_injection)
        if ops is not None:
            try:
                return self._hc_fused_combine_and_mix(ops, hidden_states, prev_block_output, prev_injection)
            except Exception as exc:  # noqa: BLE001 - the torch path below takes over
                self._hc_fused_raised(exc)

        hidden_states, xn = hc_combine_norm_torch(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu_torch(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [..., D]
        block_input = hc_gate_mix_torch(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        ops = self._hc_fused(hidden_states, True, block_output, injection)
        if ops is not None:
            try:
                return ops.hc_combine(hidden_states, block_output, injection, self.hc_count)
            except Exception as exc:  # noqa: BLE001 - the torch path below takes over
                self._hc_fused_raised(exc)

        return hc_combine_torch(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "HpuGatedResidual",
    "grouped_gemma_rmsnorm_torch",
    "hc_combine_norm_torch",
    "hc_combine_torch",
    "hc_gate_mix_torch",
    "hc_silu_torch",
]
