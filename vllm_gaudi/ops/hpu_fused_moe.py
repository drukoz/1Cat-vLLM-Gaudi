from collections.abc import Callable
from enum import Enum
from functools import partial
import os
from typing import Union
import weakref

from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner as MoERunnerBase, )
import torch
import vllm
import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter, )
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter, )
import vllm_gaudi.envs as gaudi_envs
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOp
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.utils import has_quant_config
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

logger = init_logger(__name__)

# Map model activation names to the activation strings the Gaudi fused-MoE op's
# MoeActivationMode_t enum accepts. The synapse MoE kernel enumerates "gelu" (and
# silu/selu) but NOT "gelu_tanh"; the Gaudi gelu TPC kernel is itself the tanh
# approximation (tpc_kernels .../gelu_f16.c uses tanh_f16), so HF's
# gelu_pytorch_tanh / gelu_tanh map to the op's "gelu" (the ~1e-3 exact-vs-tanh
# difference does not affect inference). Without this, Gemma4's MoE aborts at
# warmup_graphs with: Activation "gelu_tanh" not found among MoeActivationMode_t.
_MOE_ACTIVATION_ALIASES = {
    "gelu_tanh": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "gelu_new": "gelu",
    "quick_gelu": "gelu",
    # Non-gated squared-ReLU (Nemotron-H). The model's activation config value is
    # "relu2_no_mul"; the Habana MoeActivationMode_t enum spells it "relu2". The
    # no-gate/no-multiply behaviour is selected separately via is_gated=False
    # (see VllmMixtureOfExpertsOpFP8PerChannel), not by this activation name.
    "relu2_no_mul": "relu2",
}


def _normalize_moe_activation(activation):
    act = activation.value if isinstance(activation, Enum) else activation
    return _MOE_ACTIVATION_ALIASES.get(act, act)


# Activation string used by MiniMax-M3 experts. The Habana fused-MoE kernel
# (torch.ops.hpu.mixture_of_experts) only supports silu/gelu, so this
# activation is computed unfused in PyTorch instead (see _unfused_swigluoai_moe).
_SWIGLU_OAI_ACTIVATION = "swigluoai_uninterleave"

# The unfused SwiGLU-OAI dense expert pass materializes [E_local, T, 2*I]
# intermediates. For large prompt buckets (T up to the max batched-token bucket)
# a single such graph exceeds the Synapse compiler's limits and fails to compile
# with ``synStatus 26 [Generic failure]``. Tiling the token dimension bounds the
# size of every fused graph/op while keeping the math and per-tile shapes static,
# which the HPU graph compiler requires. Tunable via env for large-prompt tuning.
_SWIGLU_OAI_TOKEN_TILE = gaudi_envs.VLLM_MINIMAX_M3_MOE_TOKEN_TILE


def _unfused_swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Unfused expert MLP with SwiGLU-OAI activation for MiniMax-M3 on HPU.

    Replaces a single ``moe_op(x, topk_ids, topk_weights, activation=...)`` fused
    call for the ``swigluoai_uninterleave`` activation (unsupported by the Habana
    MoE kernel). Returns the same per-rank (pre-all-reduce) partial as the fused
    op, so the surrounding MoERunner reduction/combine pipeline is unchanged.

    A dense pass is used (every token through every local expert, masked by the
    routing weight) to keep static shapes, which the HPU graph compiler requires
    -- data-dependent gather/indexing would produce dynamic shapes. It is
    vectorized as a sync-free expert-chunked ``bmm`` over the stacked expert
    weights rather than a per-expert Python loop: the loop issues one matmul per
    expert and serializes the HPU graph (~0.3-1.4 tok/s), while the chunked
    ``bmm`` batches experts with static shapes (~11 tok/s). The SwiGLU-OAI
    activation is computed in fp32 for accuracy (matches
    ``minimax_m3_sparse.swiglu_oai``).

    Args:
        layer: the ``FusedMoE``/``RoutedExperts`` layer. Reads the stacked
            per-rank expert weights ``layer.w13_weight`` [E_local, 2*I, H] and
            ``layer.w2_weight`` [E_local, H, I] (the same tensors that seed
            ``moe_op.w13_list``) and ``layer.moe_config.ep_rank`` /
            ``layer.local_num_experts`` for this rank's expert-id offset.
        x: [T, H] flattened activations.
        topk_ids: [T, K] selected (global) expert ids.
        topk_weights: [T, K] routing weights.
    """
    w13 = layer.w13_weight  # [E_local, 2*I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    # First global expert id owned by this (EP) rank; matches the experts_min
    # that process_weights_after_loading passes to VllmMixtureOfExpertsOp.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    d = w13.shape[1] // 2
    hidden = x.shape[-1]
    tokens = x.shape[0]

    # Per-(token, local-expert) combine weight, computed sync-free via scatter
    # (no host sync, unlike an ``== expert_id`` comparison inside a Python loop).
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Dense experts via expert-chunked bmm (static shapes, no host sync). Only
    # this rank's local experts contribute; cross-rank combine happens outside.
    # The token dimension is tiled so no single fused graph materializes the full
    # [E_local, T, 2*I] intermediate (which overflows the Synapse compiler and
    # fails with synStatus 26 for large prompt buckets); per-tile shapes stay
    # static and tiles are concatenated back into the original [T, H] output.
    chunk = 32
    token_tile = _SWIGLU_OAI_TOKEN_TILE
    if token_tile <= 0 or token_tile >= tokens:
        token_tile = tokens
    out_tiles = []
    for tstart in range(0, tokens, token_tile):
        tend = min(tstart + token_tile, tokens)
        xt = x[tstart:tend]  # [t, H]
        t = xt.shape[0]
        acc = torch.zeros_like(xt)  # [t, H]
        for start in range(0, e_local, chunk):
            end = min(start + chunk, e_local)
            c = end - start
            w13c = w13[start:end].transpose(1, 2)  # [c, H, 2I]
            w2c = w2[start:end].transpose(1, 2)  # [c, I, H]
            xe = xt.unsqueeze(0).expand(c, t, hidden)  # [c, t, H]
            h = torch.bmm(xe, w13c)  # [c, t, 2I]
            # SwiGLU-OAI in fp32 for accuracy: gate=first half, up=second half.
            g = h[..., :d].float()
            u = h[..., d:].float()
            if limit is not None:
                g = g.clamp(max=limit)
                u = u.clamp(min=-limit, max=limit)
            act = (g * torch.sigmoid(alpha * g) * (u + beta)).to(x.dtype)  # [c, t, I]
            y = torch.bmm(act, w2c)  # [c, t, H] per-rank partial
            gate_wc = gate_w[tstart:tend, start:end].t().unsqueeze(-1)  # [c, t, 1]
            acc = acc + (y.float() * gate_wc).sum(0).to(x.dtype)
        out_tiles.append(acc)
    return out_tiles[0] if len(out_tiles) == 1 else torch.cat(out_tiles, dim=0)


# --- Decode-time expert gather (bandwidth reduction) -------------------------
# The dense pass above reads EVERY local expert's weights for every token
# (E_local expert-weight reads) even though each token routes to only top_k
# experts. At small token counts (single-stream / low-concurrency decode) the
# number of DISTINCT local experts the batch touches is at most ``tokens * K``,
# which can be far below E_local. This path gathers only those experts' weights
# via ``index_select`` so only ``G = min(E_local, tokens * K)`` experts stream
# from HBM instead of E_local -- up to ``E_local / G`` fewer expert-weight
# bytes/token, the dominant single-stream decode cost.
#
# Static shapes (HPU graph-compiler requirement): ``G`` is a function of the
# static (bucketed) token count, so every shape is fixed per warmup bucket and
# the decode-bucket graphs compile once during warmup (no per-step recompile).
# Because the number of distinct hit experts is provably ``<= tokens * K <= G``,
# every routed expert is always included -- no token is ever dropped and there
# is NO data-dependent overflow branch. Gathering ids sorted ascending makes the
# per-token expert accumulation order identical to the dense index-order pass
# (padding experts contribute exactly +0.0), so the result is numerically
# identical to ``_unfused_swigluoai_moe`` (bit-exact; validated).
#
# Enabled by default for MiniMax-M3 low-concurrency decode. Set
# VLLM_MINIMAX_M3_MOE_DECODE_GATHER=0 to use the dense expert pass instead.
_MOE_DECODE_GATHER = gaudi_envs.VLLM_MINIMAX_M3_MOE_DECODE_GATHER
# Only gather when the batch is at most this many tokens (bounds gather to the
# low-concurrency decode regime where it wins; above it the dense pass is used).
_MOE_GATHER_MAX_TOKENS = gaudi_envs.VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS


def _gather_swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Expert-gather SwiGLU-OAI MLP: compute only the routed experts.

    Numerically identical to :func:`_unfused_swigluoai_moe` (same per-rank,
    pre-all-reduce partial; same fp32 SwiGLU-OAI math and expert accumulation
    order) but reads only ``G = min(E_local, tokens * K)`` expert weights from
    HBM instead of all ``E_local``. See the module comment above for the
    static-shape / no-drop / bit-exactness argument.
    """
    w13 = layer.w13_weight  # [E_local, 2*I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    d = w13.shape[1] // 2
    hidden = x.shape[-1]
    tokens = x.shape[0]
    k = topk_ids.shape[-1]

    # Static gathered-expert count (>= number of distinct hit experts).
    g = min(e_local, tokens * k)

    # Per-(token, local-expert) combine weight, identical to the dense pass.
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Pick the G local experts to compute (sorted ascending so the per-token
    # accumulation order matches the dense index-order pass -> bit-exact).
    hit = (gate_w > 0).to(torch.float32).sum(0)  # [E_local]
    gather_ids = torch.topk(hit, g, sorted=False).indices  # [G]
    gather_ids, _ = torch.sort(gather_ids)  # ascending ids

    w13_g = w13.index_select(0, gather_ids)  # [G, 2I, H]
    w2_g = w2.index_select(0, gather_ids)  # [G, H, I]
    gate_wg = gate_w.index_select(1, gather_ids)  # [T, G]

    # Expert MLP over the gathered experts (static shapes, no host sync).
    xe = x.unsqueeze(0).expand(g, tokens, hidden)  # [G, T, H]
    h = torch.bmm(xe, w13_g.transpose(1, 2))  # [G, T, 2I]
    gg = h[..., :d].float()
    uu = h[..., d:].float()
    if limit is not None:
        gg = gg.clamp(max=limit)
        uu = uu.clamp(min=-limit, max=limit)
    act = (gg * torch.sigmoid(alpha * gg) * (uu + beta)).to(x.dtype)  # [G, T, I]
    y = torch.bmm(act, w2_g.transpose(1, 2))  # [G, T, H] partial

    # Match the dense pass's accumulation EXACTLY so the result is bit-identical:
    # _unfused_swigluoai_moe sums experts within each 32-expert index chunk in
    # fp32, rounds that partial to bf16, then accumulates across chunks in bf16.
    # Reproduce the same chunk grouping + rounding here (gathered experts are
    # grouped by their original index // chunk; experts absent from a chunk
    # contribute exact +0.0, which never perturbs an fp sum).
    cont = y.float() * gate_wg.t().unsqueeze(-1)  # [G, T, H] fp32
    chunk = 32
    chunk_of = torch.div(gather_ids, chunk, rounding_mode="floor")  # [G]
    acc = x.new_zeros(tokens, hidden)  # [T, H] bf16
    n_chunks = (e_local + chunk - 1) // chunk
    for c in range(n_chunks):
        m = (chunk_of == c).to(cont.dtype)  # [G]
        acc = acc + (cont * m[:, None, None]).sum(0).to(x.dtype)
    return acc


def _swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Dispatch SwiGLU-OAI experts: gather only routed experts at small token
    counts (low-concurrency decode), else the dense pass. Both return the
    identical per-rank partial; the split is purely a bandwidth optimization.
    """
    tokens = x.shape[0]
    k = topk_ids.shape[-1]
    e_local = layer.w13_weight.shape[0]
    if (_MOE_DECODE_GATHER and tokens <= _MOE_GATHER_MAX_TOKENS and tokens * k < e_local):
        return _gather_swigluoai_moe(layer, x, topk_ids, topk_weights, alpha=alpha, beta=beta, limit=limit)
    return _unfused_swigluoai_moe(layer, x, topk_ids, topk_weights, alpha=alpha, beta=beta, limit=limit)


# --- MoE glue (VLLM_HPU_MOE_GLUE=1) ------------------------------------------
# The element-wise ops around the GEMMs and the vendor expert op of one MoE
# block, as four TPC kernels (flashinfer_gaudi/moe_glue.py; kernels in
# csrc/hc_fused; contract: docs/design/flash-next/hc-fused.md, "MoE glue
# kernels"):
#
#   gates  = linear(x, [router rows; shared-gate row])     one GEMM, the weight built at load
#   ids, w = moe_route(gates)                              softmax, top-k, renormalization, casts
#   act    = moe_silu_mul(gate_up(x))                      the shared expert's SiLU-multiply
#            (VLLM_HPU_MOE_GLUE_MERGE_GATE_UP=1: gate_up's rows are in the merged weight too, and
#             act = moe_silu_mul(gates, experts + 1) reads its block in the one GEMM output)
#   x8, s  = moe_row_quant(x)                              FP8 experts: the per-row quantization
#   routed = hpu.mixture_of_experts(...)                   unchanged
#   out    = moe_combine(down(act), gates, routed)         sigmoid(gate) * shared + routed
#
# With the flag at 0 (the default) none of this exists: no state object, no
# merged weight, and patched_fused_moe_forward dispatches the ops it always did.
_MOE_GLUE_BUFFER = "_hpu_moe_glue_gate_weight"
# VLLM_HPU_MOE_GLUE_MERGE_GATE_UP=1: the router rows, the shared-gate row, then this rank's gate_up rows.
_MOE_GLUE_GATE_UP_BUFFER = "_hpu_moe_glue_gate_up_weight"
_MOE_GLUE_OFF_LINE = "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for %s: %s; using the torch path"
_MOE_GLUE_MERGE_OFF_LINE = ("MoE glue (VLLM_HPU_MOE_GLUE_MERGE_GATE_UP=1) keeps the separate gate_up GEMM for %s: "
                            "%s")
# The flags that send dynamic_quant (vllm_gaudi/extension/ops.py) down another branch than the one
# moe_row_quant replaces.
_MOE_GLUE_QUANT_FLAGS = ("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT",
                         "VLLM_HPU_DYNAMIC_QUANT_CGUID")


class _MoeGlueFallback(Exception):
    """A call the fused kernels do not take; the message is the reason the block logs."""


class _MoeGlueState:
    """What one MoE block keeps for its fused glue path. Created with the block, only with the flag at 1."""

    def __init__(self, runner) -> None:
        self.layer_name = str(getattr(runner, "layer_name", type(runner).__name__))
        self._runner = weakref.ref(runner)
        self.ok: bool | None = None  # None: not checked yet. True: fused. False: the torch path for good.
        self.logged = False
        self.prepare_error: str | None = None
        self.ops = None  # the flashinfer_gaudi.moe_glue module, once the layer check passed
        self.experts = 0
        self.top_k = 0
        self.fp8 = False
        self.fp8_max = 0.0
        # The shared expert's gate_up GEMM merged into the gate GEMM: asked for once per block, here; the first
        # column of the gate_up block in the merged output once the layer check accepted it (0: not merged).
        self.merge_gate_up = gaudi_envs.VLLM_HPU_MOE_GLUE_MERGE_GATE_UP == 1
        self.gate_up_first_column = 0
        self.merge_prepare_error: str | None = None

    def runner(self):
        return self._runner()

    def log(self, reason: str) -> None:
        """One warning line per block, whatever the number of reasons and calls."""
        if not self.logged:
            self.logged = True
            logger.warning(_MOE_GLUE_OFF_LINE, self.layer_name, reason)

    def off(self, reason: str) -> None:
        self.ok = False
        self.log(reason)

    # What the monolithic quant methods call instead of their torch code.
    def route(self, router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(topk_weights, topk_ids)``: the ids are int32, the caller widens them for the vendor op."""
        ids, weights = self.ops.moe_route(router_logits, self.experts, int(top_k))
        return weights, ids

    def row_quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x_fp8, x_scale)``, what dynamic_quant returns.

        ``x`` is the block input after the model block's ``hidden_states.view(-1, hidden_dim)`` and
        ``HPUFp8MoEMethod.apply_monolithic``'s ``x.view(-1, H)``: a view object on every call. The lazy bridge
        hands a view to a custom op as a graph input without data ("Empty tensor optional ... Strided Params Has
        Value: 1"), also with the accumulation thread off (card probe probe_view_inputs.py, job 00043,
        2026-11-20: 7 of 7 view cases raised at PT_HPU_LAZY_ACC_PAR_MODE=0), so the op gets a tensor of its own.
        The other three ops get GEMM outputs and the vendor op's output as they are made: on this path the
        monolithic methods return the vendor op's output itself, not their ``output.view(...)``."""
        return self.ops.moe_row_quant(x.clone(memory_format=torch.contiguous_format), self.fp8_max)


def _moe_glue_attach(runner) -> None:
    """Give a new MoE runner its glue state (VLLM_HPU_MOE_GLUE=1 only; the flag is read once, here)."""
    state = _MoeGlueState(runner)
    runner._hpu_moe_glue = state
    # The quant method's process_weights_after_loading gets the routed experts, not the runner.
    runner.routed_experts._hpu_moe_glue = state


def _moe_glue_shared_layer(runner):
    shared = getattr(runner, "_shared_experts", None)
    return getattr(shared, "_layer", None) if shared is not None else None


def _moe_glue_prepare(layer) -> None:
    """At load: build the block's merged gate weight, the router rows followed by the shared-gate row.

    Called at the end of process_weights_after_loading with the routed experts. It does nothing without
    VLLM_HPU_MOE_GLUE=1 (no state was attached) and never raises: what is missing is named by the layer check.
    """
    state = getattr(layer, "_hpu_moe_glue", None)
    if state is None:
        return
    runner = state.runner()
    try:
        gate_weight = runner.gate.weight
        shared_gate_weight = _moe_glue_shared_layer(runner).expert_gate.weight
        merged = torch.cat([gate_weight.detach(), shared_gate_weight.detach()], dim=0).contiguous()
        runner.register_buffer(_MOE_GLUE_BUFFER, merged, persistent=False)
    except Exception as exc:  # noqa: BLE001 - an optional fast path never stops a load
        state.prepare_error = f"{type(exc).__name__}: {exc}"
        return
    if state.merge_gate_up:
        # The same rows, then the shared expert's gate_up rows of this rank (TP-sharded: each rank its own).
        try:
            gate_up = _moe_glue_shared_layer(runner).gate_up_proj
            if not _moe_glue_plain_linear(gate_up):
                raise ValueError("the shared expert's gate_up_proj is not a plain linear without bias")
            merged_gate_up = torch.cat([merged, gate_up.weight.detach()], dim=0).contiguous()
            runner.register_buffer(_MOE_GLUE_GATE_UP_BUFFER, merged_gate_up, persistent=False)
        except Exception as exc:  # noqa: BLE001
            state.merge_prepare_error = f"{type(exc).__name__}: {exc}"


def _moe_glue_library_checked_device(device: torch.device) -> bool:
    """True for a device whose kernels come from libhc_fused_gaudi2.so (the loader knows which it serves)."""
    return device.type == "hpu"


def _moe_glue_plain_linear(layer) -> bool:
    """True for a bias-free linear whose forward is F.linear(x, layer.weight)."""
    return (getattr(layer, "bias", None) is None and not getattr(layer, "skip_bias_add", False)
            and type(getattr(layer, "quant_method", None)).__name__ == "UnquantizedLinearMethod"
            and isinstance(getattr(layer, "weight", None), torch.Tensor) and layer.weight.dim() == 2)


def _moe_glue_layer_reason(runner, device_type: str | None = None) -> str | None:
    """Why this MoE block cannot run the fused glue, or None. Reads the block, never a tensor's data.

    ``device_type`` overrides the device of the merged gate weight (tests).
    """
    config = runner.moe_config
    routed = runner.routed_experts
    if config.dp_size != 1 or getattr(config, "pcp_size", 1) != 1 or config.is_sequence_parallel:
        return "data-parallel, context-parallel or sequence-parallel MoE"
    if runner.routed_input_transform is not None or runner.routed_output_transform is not None:
        return "a routed input or output transform (latent MoE)"
    if runner.routed_scaling_factor != 1.0:
        return f"routed_scaling_factor is {runner.routed_scaling_factor}, not 1"
    if isinstance(runner.router, ZeroExpertRouter) or runner._fse_fuse_gate or runner._fused_output_is_reduced:
        return "a zero expert, fused shared experts or an expert kernel that reduces its own output"
    if routed.use_grouped_topk or getattr(routed, "custom_routing_function", None) is not None:
        return "grouped top-k or custom routing"
    gate = runner.gate
    shared = _moe_glue_shared_layer(runner)
    if gate is None or shared is None:
        return "the MoE runner does not hold the router gate and a shared expert"
    expert_gate = getattr(shared, "expert_gate", None)
    if expert_gate is None or any(getattr(shared, name, None) is None for name in ("gate_up_proj", "down_proj")):
        return "the shared expert has no shared expert gate, gate_up_proj or down_proj"
    if not isinstance(getattr(shared, "act_fn", None), SiluAndMul):
        return "the shared expert's activation is not SiluAndMul"
    if gaudi_envs.VLLM_HPU_EXPLICIT_SIGMOID_SILU:
        return "VLLM_HPU_EXPLICIT_SIGMOID_SILU is set (another SiLU formula)"
    if os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower() != "off":
        return "VLLM_HPU_TRITON_MODE is not off (another SiLU-multiply and quantization path)"
    if not _moe_glue_plain_linear(gate) or not _moe_glue_plain_linear(expert_gate):
        return "the router gate or the shared expert gate is not a plain linear without bias"
    experts, hidden = gate.weight.shape
    top_k = int(routed.top_k)
    if tuple(expert_gate.weight.shape) != (1, hidden):
        return f"the shared expert gate weight is {tuple(expert_gate.weight.shape)}, not (1, {hidden})"

    method = routed.quant_method
    kind = type(method).__dict__.get("_hpu_moe_glue_kind")
    if kind is None:
        return f"the expert method {type(method).__name__} has no fused glue"
    if kind == "bf16" and getattr(method, "model_type", None) == "gpt_oss":
        return "gpt_oss routing (top-k before the softmax)"
    if kind == "fp8":
        op = getattr(routed, "moe_op", None)
        if type(op).__name__ != "VllmMixtureOfExpertsOpFP8PerChannel":
            return f"the FP8 expert op is {type(op).__name__} (block FP8 with VLLM_HPU_FORCE_CHANNEL_FP8=0)"
        if op.w13_input_scale is not None:
            return "the FP8 experts have a static input scale"
        flags = [name for name in _MOE_GLUE_QUANT_FLAGS if getattr(gaudi_envs, name)]
        if hpu_ops.envs.VLLM_HPU_CGUID_DYNAMIC_QUANT:
            flags.append("VLLM_HPU_CGUID_DYNAMIC_QUANT")
        if flags:
            return f"{', '.join(flags)} selects another dynamic quantization than the one the kernel replaces"

    from flashinfer_gaudi.moe_glue import GAUDI2_FP8_MAX, MAX_EXPERTS, MAX_TOP_K, VECTOR_LANES

    if hidden % VECTOR_LANES != 0:
        return f"hidden size {hidden} is not a multiple of {VECTOR_LANES}"
    if not (1 <= top_k <= MAX_TOP_K and top_k <= experts <= MAX_EXPERTS):
        return f"top_k {top_k} of {experts} experts is outside 1..{MAX_TOP_K} of at most {MAX_EXPERTS}"
    merged = getattr(runner, _MOE_GLUE_BUFFER, None)
    if merged is None:
        error = getattr(getattr(runner, "_hpu_moe_glue", None), "prepare_error", None)
        return "the merged gate weight was not built at load" + (f" ({error})" if error else "")
    if merged.dtype != torch.bfloat16 or tuple(merged.shape) != (experts + 1, hidden):
        return f"the merged gate weight is {merged.dtype} {tuple(merged.shape)}, not bfloat16 {(experts + 1, hidden)}"
    device_type = device_type or merged.device.type
    if kind == "fp8" and device_type == "hpu" and float(hpu_ops.FP8_MAX) != GAUDI2_FP8_MAX:
        return f"FP8_MAX is {float(hpu_ops.FP8_MAX)}: the quantization kernel takes {GAUDI2_FP8_MAX} (Gaudi2) only"

    # The kernels: every op this block needs has one for the device, and the kernel library serves them.
    try:
        from flashinfer_gaudi import moe_glue as ops
    except Exception as exc:  # noqa: BLE001
        return f"flashinfer_gaudi.moe_glue does not import ({type(exc).__name__}: {exc})"
    needed = [name for name in ops.OP_NAMES if kind == "fp8" or name != "moe_row_quant"]
    device = torch.device(device_type)
    missing = [name for name in ops.moe_glue_missing_ops(device) if name in needed]
    if missing:
        from flashinfer_gaudi import _native

        errors = "; ".join(getattr(_native, "moe_glue_extension_errors", tuple)())
        return f"no {device_type} kernel for {', '.join(missing)}" + (f" ({errors})" if errors else "")
    if _moe_glue_library_checked_device(device):
        from flashinfer_gaudi import _native

        # The ops can have HPU kernels without the library serving them: the extension registers all four,
        # and a MoE kernel that did not compile is left out of the library. Only the loader knows.
        if _native.moe_glue_extension_loaded() is None:
            return "the kernel library was not asked for its MoE kernels (load_moe_glue_extension was not called)"
        unserved = [name for name in _native.moe_glue_missing_kernels() if name in needed]
        if unserved:
            return f"libhc_fused_gaudi2.so has no kernel for {', '.join(unserved)}"
    return None


def _moe_glue_merge_reason(runner, ops) -> str | None:
    """Why this block keeps the separate gate_up GEMM under VLLM_HPU_MOE_GLUE_MERGE_GATE_UP=1, or None.

    Runs after the layer check passed. Reads the block and the schema, never a tensor's data."""
    state = runner._hpu_moe_glue
    merged = getattr(runner, _MOE_GLUE_GATE_UP_BUFFER, None)
    if merged is None:
        return "the merged gate_up weight was not built at load" + (
            f" ({state.merge_prepare_error})" if state.merge_prepare_error else "")
    experts, hidden = runner.gate.weight.shape
    gate_up = _moe_glue_shared_layer(runner).gate_up_proj.weight
    block = merged.shape[0] - (experts + 1)
    if (merged.dtype != torch.bfloat16 or merged.dim() != 2 or merged.shape[1] != hidden or block < 2 or block % 2
            or tuple(gate_up.shape) != (block, hidden)):
        return (f"the merged gate_up weight is {merged.dtype} {tuple(merged.shape)}, not bfloat16 "
                f"({experts + 1} + an even number of gate_up rows, {hidden})")
    if "first_column" not in str(getattr(ops, "SCHEMAS", {}).get("moe_silu_mul", "")):
        return "moe_silu_mul of this build has no first_column"
    return None


def _moe_glue_accept(state: _MoeGlueState, runner) -> None:
    """The layer check passed: keep what the calls need."""
    from flashinfer_gaudi import moe_glue as ops

    state.ops = ops
    state.experts = int(runner.gate.weight.shape[0])
    state.top_k = int(runner.routed_experts.top_k)
    state.fp8 = type(runner.routed_experts.quant_method).__dict__.get("_hpu_moe_glue_kind") == "fp8"
    state.fp8_max = float(hpu_ops.FP8_MAX)
    state.gate_up_first_column = 0
    if state.merge_gate_up:
        reason = _moe_glue_merge_reason(runner, ops)
        if reason is None:
            state.gate_up_first_column = state.experts + 1
        else:
            logger.warning(_MOE_GLUE_MERGE_OFF_LINE, state.layer_name, reason)
    state.ok = True


def _moe_glue_matrix_reason(tensor, name: str, columns: int | None, device: torch.device) -> str | None:
    """Why `tensor` is not a [tokens, columns] activation the kernels take, or None (shape and dtype only)."""
    if not isinstance(tensor, torch.Tensor):
        return f"{name} are not a tensor"
    if tensor.dtype != torch.bfloat16:
        return f"{name} are {tensor.dtype}, not bfloat16"
    if tensor.dim() != 2 or tensor.shape[0] < 1 or (columns is not None and tensor.shape[1] != columns):
        return f"{name} are {tuple(tensor.shape)}, not [tokens, {columns if columns is not None else 'columns'}]"
    if not tensor.is_contiguous():
        return f"{name} are not contiguous"
    if tensor.device != device:
        return f"{name} are on {tensor.device}, not {device}"
    return None


def _moe_glue_call_reason(state: _MoeGlueState, runner, hidden_states, shared_experts_input) -> str | None:
    """Why this call cannot run the fused glue, or None. Ranks, dtypes, sizes, layout and devices only."""
    merged = getattr(runner, _MOE_GLUE_BUFFER)
    reason = _moe_glue_matrix_reason(hidden_states, "hidden states", merged.shape[1], merged.device)
    if reason is None and shared_experts_input is not hidden_states:
        reason = _moe_glue_matrix_reason(shared_experts_input, "the shared expert's input", merged.shape[1],
                                         merged.device)
        if reason is None and shared_experts_input.shape[0] != hidden_states.shape[0]:
            reason = "the shared expert's input has other rows than the hidden states"
    return reason


def _moe_glue_require(tensor, name: str, columns: int | None, device: torch.device) -> None:
    reason = _moe_glue_matrix_reason(tensor, name, columns, device)
    if reason is not None:
        raise _MoeGlueFallback(reason)


def _moe_glue_forward(runner, state: _MoeGlueState, hidden_states, shared_experts_input, padded: bool):
    """One MoE block through the fused glue: the block's output before its all-reduce, or None.

    None means the torch path: the block is off (one log line, once), or this call is outside what the
    kernels take. Nothing here reads a tensor's data, and no op writes an input, so the torch path then
    runs on the same inputs.
    """
    if state.ok is False:
        return None
    if state.ok is None:
        reason = _moe_glue_layer_reason(runner)
        if reason is not None:
            state.off(reason)
            return None
        _moe_glue_accept(state, runner)
    reason = "the hidden states are padded" if padded else _moe_glue_call_reason(
        state, runner, hidden_states, shared_experts_input)
    if reason is not None:
        state.log(reason)
        return None
    ops = state.ops
    merged = getattr(runner, _MOE_GLUE_BUFFER)
    hidden = merged.shape[1]
    try:
        shared = _moe_glue_shared_layer(runner)
        if state.gate_up_first_column and shared_experts_input is hidden_states:
            # One GEMM for the router logits (columns [0, experts)), the shared-gate logit (column experts) and
            # the shared expert's gate_up output (the columns from experts + 1 on): moe_silu_mul reads it there.
            merged_gate_up = getattr(runner, _MOE_GLUE_GATE_UP_BUFFER)
            gates = torch.nn.functional.linear(hidden_states, merged_gate_up)
            _moe_glue_require(gates, "the merged gate output", merged_gate_up.shape[0], merged.device)
            act = ops.moe_silu_mul(gates, state.gate_up_first_column)
        else:
            # One GEMM for the router logits (columns [0, experts)) and the shared-gate logit (column experts).
            gates = torch.nn.functional.linear(hidden_states, merged)
            _moe_glue_require(gates, "the merged gate output", state.experts + 1, merged.device)
            gate_up, _ = shared.gate_up_proj(shared_experts_input)
            _moe_glue_require(gate_up, "the shared expert's gate_up output", None, merged.device)
            if gate_up.shape[1] % 2 != 0:
                raise _MoeGlueFallback("the shared expert's gate_up output has an odd number of columns")
            act = ops.moe_silu_mul(gate_up)
        shared_output, _ = shared.down_proj(act)
        _moe_glue_require(shared_output, "the shared expert's output", hidden, merged.device)
        routed = runner.routed_experts
        routed_output = routed.quant_method.apply_monolithic(layer=routed, x=hidden_states, router_logits=gates,
                                                             moe_glue=state)
        _moe_glue_require(routed_output, "the routed experts' output", hidden, merged.device)
        return ops.moe_combine(shared_output, gates, routed_output, state.experts)
    except _MoeGlueFallback as exc:
        state.log(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 - a fused op that raises turns the block's fused path off
        state.off(f"a fused op raised {type(exc).__name__}: {exc}")
        return None


def model_has_quant_config() -> bool:
    """Whether the active model runs with a MoE quantization config.

    After upstream PR #41184 the ``layer`` reaching ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer carries ``vllm_config`` (that
    field belonged to the old top-level ``FusedMoE``). The model config must
    therefore be resolved from the global vLLM config instead of off ``layer``.

    This MUST be called at build time (e.g. in ``__init__`` /
    ``process_weights_after_loading``), where the vLLM config context is set:
    the result is static per run, so callers cache it and read the cached flag
    on the forward hot path. ``get_current_vllm_config_or_none`` is used so the
    function degrades to ``False`` instead of raising if no context is active.

    Returns:
        ``True`` when the active model config declares a MoE quant config.
    """
    vllm_config = get_current_vllm_config_or_none()
    model_config = vllm_config.model_config if vllm_config is not None else None
    return model_config is not None and has_quant_config(model_config)


def select_experts_from_routed(layer, hidden_states: torch.Tensor,
                               router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to experts for the grouped/custom-routing monolithic path.

    After upstream PR #41184 the ``layer`` passed to ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer owns a ``.router`` object (the
    router moved onto ``MoERunner``). ``RoutedExperts`` does, however, carry all
    the routing parameters, so we reproduce upstream's behaviour via the
    standalone ``select_experts`` helper. It is imported lazily because
    ``experts.cpu_moe`` pulls CPU custom ops from ``vllm._custom_ops`` at
    module import time.

    Args:
        layer: The ``RoutedExperts`` instance holding the routing parameters.
        hidden_states: Flattened input activations.
        router_logits: Gate logits for the current tokens.

    Returns:
        A ``(topk_weights, topk_ids)`` tuple.
    """
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import select_experts

    # Models hand the runner a placeholder ``router_logits`` (== hidden_states)
    # and expect the runner to overwrite it with the gate output. If that step
    # is skipped the placeholder reaches expert selection and only fails later,
    # deep inside the routing kernels, as an opaque broadcast error.
    num_experts = layer.moe_config.num_logical_experts
    if router_logits.size(-1) != num_experts:
        raise RuntimeError(f"Router logits have {router_logits.size(-1)} columns but the layer routes to "
                           f"{num_experts} experts; the MoE gate was not applied to the hidden states.")

    return select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=layer.top_k,
        use_grouped_topk=layer.use_grouped_topk,
        renormalize=layer.renormalize,
        topk_group=layer.topk_group,
        num_expert_group=layer.num_expert_group,
        custom_routing_function=layer.custom_routing_function,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=layer.routed_scaling_factor,
        e_score_correction_bias=layer.e_score_correction_bias,
    )


@UnquantizedFusedMoEMethod.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    # VLLM_HPU_MOE_GLUE: this class (not a subclass) runs the fused glue with bf16 experts.
    _hpu_moe_glue_kind = "bf16"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if (vllm_config is not None and vllm_config.model_config is not None
                and vllm_config.model_config.hf_config is not None):
            self.model_type = vllm_config.model_config.hf_config.model_type

        # SwiGLU-OAI activation params (MiniMax-M3). Cached here (config context
        # active) and read on the forward hot path when the fused kernel cannot
        # provide the activation. Falls back to the MiniMax-M3 defaults.
        tc = None
        if vllm_config is not None and vllm_config.model_config is not None:
            tc = getattr(vllm_config.model_config, "hf_text_config", None)
            if tc is None:
                tc = vllm_config.model_config.hf_config
        self.swiglu_alpha = float(getattr(tc, "swiglu_alpha", 1.702))
        self.swiglu_beta = float(getattr(tc, "swiglu_beta", 1.0))
        _limit = getattr(tc, "swiglu_limit", 7.0)
        self.swiglu_limit = None if _limit is None else float(_limit)

    def _select_monolithic(self) -> Callable:
        """Overriding base method"""
        return self.apply_monolithic

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts
        has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        bias = has_bias if has_bias is True else None

        is_bf16 = getattr(layer, "w13_weight", None) is not None and layer.w13_weight.dtype == torch.bfloat16

        is_unquantized = not self.has_moe_quant_config

        cache_weight_lists = bool(is_bf16 and is_unquantized)

        # Pass cache flag into moe_op (requires ops.py __init__ signature update)
        layer.moe_op = VllmMixtureOfExpertsOp(layer.global_num_experts, num_experts, experts_min, experts_max, bias,
                                              dispatch_fn)

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

        # Build cache once AFTER weights/bias are set (BF16 + unquantized only)
        if cache_weight_lists and hasattr(layer.moe_op, "_cache_weight_lists"):
            layer.moe_op._cache_weight_lists()

        # VLLM_HPU_MOE_GLUE=1: the merged gate weight of the block (nothing with the flag at 0).
        _moe_glue_prepare(layer)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        moe_glue = kwargs.get("moe_glue")
        if moe_glue is not None:
            # VLLM_HPU_MOE_GLUE=1: softmax, top-k, renormalization and the casts as one fused op.
            topk_weights, topk_ids = moe_glue.route(router_logits, layer.top_k)
        elif layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type == "gpt_oss":
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel compiles for bf16 (x.dtype) router
        # weights and int64 routing tables. The grouped-topk / custom-routing
        # helper returns float32 weights and int32 ids; the regular-topk path
        # above already normalized them, but the grouped path previously left
        # them unconverted -> the bf16 MoE kernel graph received a float32
        # router_weights tensor and failed to compile (synStatus 26). Normalize
        # for every routing path so the kernel inputs are dtype-consistent.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # Sequence-parallel MoE without data parallelism (TP>1 + EP with the
            # allgather_reducescatter backend => use_sequence_parallel_moe). The
            # model block chunks tokens by tp_size before the experts and
            # all-gathers afterwards, expecting `self.experts` to be
            # token-neutral. Upstream MoERunner._maybe_combine (mirrored in
            # patched_fused_moe_forward) reduce-scatters the expert output over
            # the EP group whenever is_sequence_parallel is set, regardless of
            # dp_size. The paired dispatch all-gather is set up via dispatch_fn
            # only for dp_size > 1, so at dp_size == 1 the combine is unpaired
            # and halves the token count -> the block's final reshape fails.
            # All-gather x / topk over the EP group here to restore the
            # dispatch/combine symmetry (dp_metadata is None at dp_size == 1, so
            # dispatch_tensor allocates its own EP-sized output).
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused
            # (dense), or gather-only routed experts at low-conc decode.
            output = _swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if moe_glue is not None and output.shape == input_shape:
            return output  # not output.view(...): moe_combine is a custom op (_MoeGlueState.row_quant)
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)

    def forward_oot(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type is not None and self.model_type in ["gpt_oss"]:
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # See apply_monolithic: the bf16 HPU MoE kernel needs int64 routing
        # tables and x.dtype router weights. Normalize for every routing path
        # (grouped-topk / custom routing returns int32 + float32) so the kernel
        # graph receives dtype-consistent inputs and compiles.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # See apply_monolithic: at dp_size == 1 with sequence-parallel MoE
            # (TP>1 + EP), MoERunner._maybe_combine reduce-scatters the expert
            # output over the EP group but no paired dispatch all-gather runs
            # (dispatch_fn is wired only for dp_size > 1). Restore symmetry by
            # all-gathering the inputs over the EP group so the combine leaves
            # the token count unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        if self.model_type in ["gpt_oss"]:
            gpt_oss_out = layer.moe_op(
                x,
                topk_ids.to(torch.int64),
                topk_weights.to(x.dtype),
                permuted_weights=True,
                activation=_normalize_moe_activation(layer.activation),
            )
            if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
                return gpt_oss_out.view(*(gpt_oss_out.size(0), *input_shape[1:]))
            return gpt_oss_out.view(*input_shape)

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused
            # (dense), or gather-only routed experts at low-conc decode.
            output = _swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


def patched_fused_moe_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Patched forward that avoids graph breaks from ForwardContext lookups
    and dynamo per-layer string guards.

    Instead of calling _forward_impl (which uses _sequence_parallel_context
    and _maybe_dispatch — both of which access ForwardContext and cause
    torch.compile graph breaks), for dp_size==1 we inline the quant-config
    init, gate application, _apply_quant_method and _maybe_combine directly.
    This also bypasses self.layer_name (a per-layer string) so dynamo no
    longer emits per-layer string guards that trigger recompilation.

    After upstream PR #41184 (FusedMoE/MoERunner inversion), `self` IS the
    MoERunner: the expert weights and quant_method live on
    self.routed_experts, quant init moved to
    routed_experts._ensure_moe_quant_config_init(), and _apply_quant_method
    no longer takes a `layer` argument. The post-forward reduction sequence
    mirrors upstream MoERunner.forward so we stay in sync with the shared/
    fused output combination logic.
    """
    hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
    # Upstream _maybe_pad_hidden_states returns a 3-tuple: the (possibly padded)
    # hidden_states plus separate pre-transform and post-transform truncation
    # sizes (either may be None). Mirror MoERunner.forward, which keeps them
    # distinct — pre_xform strips fused_output before the routed-output
    # transform, post_xform strips the final output after all-reduce.
    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (self._maybe_pad_hidden_states(
        shared_experts_input, hidden_states))

    if self.moe_config.dp_size == 1:
        # Bypass _forward_impl entirely for dp_size==1 to eliminate
        # graph breaks from _sequence_parallel_context() (which calls
        # get_forward_context()), skip the no-op _maybe_dispatch(), and
        # avoid double gate / stream-sync calls that _forward_impl
        # would redundantly repeat.
        if self.moe_config.pcp_size > 1:
            raise RuntimeError("dp_size==1 fast path does not support pcp_size > 1")
        # Mirror MoERunner._forward_impl's pre-dispatch setup: the quant config
        # is initialized on routed_experts (which the runner holds directly), so
        # unlike the old FusedMoE-layer-based init we do NOT need the layer here.
        self.routed_experts._ensure_moe_quant_config_init()
        # VLLM_HPU_MOE_GLUE=1: the block's element-wise ops as fused kernels around one merged gate GEMM.
        # The state exists only with the flag at 1; a block (or a call) the kernels do not take returns
        # None here, after one log line, and runs the code below.
        moe_glue = getattr(self, "_hpu_moe_glue", None)
        if moe_glue is not None:
            combined = _moe_glue_forward(self, moe_glue, hidden_states, shared_experts_input,
                                         og_hidden_dim_pre_xform is not None or og_hidden_dim_post_xform is not None)
            if combined is not None:
                return self._maybe_reduce_final_output(combined, None, False)
        # Launch the shared experts the way MoERunner._forward_impl does at the
        # engine pin (vllm 8f816a3f): SharedExperts.maybe_forward_async returns
        # True only on a CUDA-alike platform with an aux stream, so on HPU it is
        # False and the shared experts run synchronously inside
        # _apply_quant_method. The pin has no MoERunner._maybe_sync_shared_experts_stream
        # (that name belongs to the vllm PR #52024 revert, which the pin postdates);
        # calling it raised AttributeError on every MoE forward (rental run 2, RUK-2709).
        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(shared_experts_input)
        # Apply the gate if the runner holds it (mirrors _forward_impl).
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = torch.nn.functional.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)
        # Core MoERunner._apply_quant_method takes no layer argument — it reads
        # everything it needs off the runner (self.routed_experts / self.router).
        # Call it exactly as upstream _forward_impl does at the pin, threading the
        # overlap flag so an async-launched shared expert is awaited, not recomputed.
        shared_output, fused_hidden = self._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
            input_ids=input_ids,
            shared_experts_overlapping=shared_experts_overlapping,
        )
        result = self._maybe_combine(shared_output, fused_hidden)
    else:
        # Upstream PR #41184 dropped MoERunner._trtllm_mxfp4_unpadded_dim(); the
        # TRT-LLM MXFP4 unpadded hidden dim now comes from moe_config and is only
        # non-zero when the quant method produces unpadded output. Mirror
        # MoERunner.forward exactly so we stay in sync with the custom-op signature.
        hidden_dim_unpadded = (self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0)
        result = self._forward_entry(hidden_states, router_logits, shared_experts_input, input_ids,
                                     self._encode_layer_name(), hidden_dim_unpadded)

    # Mirror upstream MoERunner.forward post-_forward_entry pipeline.
    if isinstance(result, tuple):
        shared_output, fused_output = result
    else:
        shared_output, fused_output = None, result

    # Trim padding from fused_output before the routed-output transform, matching
    # upstream MoERunner.forward (latent MoE with shared experts).
    if og_hidden_dim_pre_xform is not None:
        fused_output = fused_output[..., :og_hidden_dim_pre_xform]

    # Mirror upstream MoERunner.forward's two mutually-exclusive all-reduce
    # points, threaded by _fused_output_is_reduced: reduce a latent routed
    # output before its (possibly non-linear) transform, then reduce shared
    # output to match, else all-reduce the combined sum in the final step.
    fused_output_is_reduced = self._fused_output_is_reduced
    fused_output, fused_output_is_reduced = self._maybe_reduce_routed_output_before_transform(
        fused_output, fused_output_is_reduced)

    shared_output = self._maybe_reduce_shared_expert_output(shared_output, fused_output_is_reduced)
    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
    fused_output = self.apply_routed_output_transform(fused_output)

    combined = (shared_output + fused_output) if shared_output is not None else fused_output

    combined = self._maybe_reduce_final_output(combined, og_hidden_dim_post_xform, fused_output_is_reduced)
    return self._maybe_add_zero_expert_output(combined)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    This implementation uses a standard Python loop, which is compatible with
    graph compilation modes that do not support dynamic shapes resulting from
    operations like `torch.where`.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping a global expert index to its local index. Contains -1 for
            experts that are not assigned to the current rank.

    Returns:
        str: A string mapping from local to global index,
        ordered by global index.
            (e.g., "0->5, 1->12, 2->23")
    """
    mappings = []
    # A standard loop over a tensor with a known shape is statically analyzable.
    # `enumerate` provides the global_index (the position in the tensor) and
    # `local_index_tensor` (the value at that position).
    for global_index, local_index_tensor in enumerate(expert_map):
        local_index = local_index_tensor.item()
        # We only build strings for valid experts (those not marked as -1).
        if local_index != -1:
            mappings.append(f"{local_index}->{global_index}")

    return ", ".join(mappings)


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing parameters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    eplb_state: EplbLayerState | None = None,
    # zero expert parameters
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
    # Deepseek V4 vision routing bias parameters (signature parity with the pinned vLLM factory)
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. ZeroExpertRouter - if zero_expert_type is not None
    3. GroupedTopKRouter - if use_grouped_topk is True
    4. CustomRoutingRouter - if custom_routing_function is not None
    5. FusedTopKBiasRouter - if e_score_correction_bias is not None
    6. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights

    Grouped topk arguments:
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)
        shared_expert_weight: Weight of the fused shared-expert slot (upstream
            vLLM d039c171+). Accepted for signature parity; unused on HPU since
            the fused-shared-expert path is not taken (default 1.0 no-op).

    Grouped topk and fused topk bias arguments:
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores

    Custom routing arguments:
        custom_routing_function: Optional custom routing function

    EPLB arguments:
        eplb_state: EPLB (Expert Parallelism Load Balancing) state

    Zero expert arguments:
        zero_expert_type: Type of zero expert (e.g. identity). If not None,
            creates a ZeroExpertRouter.
        num_logical_experts: Number of real (non-zero) experts. Required when
            zero_expert_type is not None.

    Hash Indices Table:
        hash_indices_table: Used to map input_ids to experts, needed for
            Deepseek V4

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    if zero_expert_type is not None:
        assert num_logical_experts is not None, "num_logical_experts is required when zero_expert_type is set"
        assert e_score_correction_bias is not None, "e_score_correction_bias is required when zero_expert_type is set"
        return ZeroExpertRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            num_logical_experts=num_logical_experts,
            zero_expert_type=zero_expert_type,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )

    if use_grouped_topk:
        assert custom_routing_function is None
        if num_expert_group is None or topk_group is None:
            raise ValueError("num_expert_group and topk_group must be provided when use_grouped_topk is True")
        grouped_topk_router = GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        return grouped_topk_router

    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
        )

    assert scoring_func in ["sigmoid", "softmax", "sqrtsoftplus"]

    if e_score_correction_bias is not None or hash_indices_table is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            hash_indices_table=hash_indices_table,
            num_fused_shared_experts=num_fused_shared_experts,
            shared_expert_weight=shared_expert_weight,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )


# Apply patches
_orig_default_moe_runner_forward = MoERunnerBase.forward

# When enabled, bypasses the opaque torch.ops.vllm.moe_forward_shared custom
# op wrapper so that torch.ops.hpu.mixture_of_experts is captured directly in
# compiled Synapse graphs instead of running eagerly.
# Set HPU_FUSED_MOE=0 to disable and fall back to the original path.
_MOE_COMPILE = os.getenv("HPU_FUSED_MOE", "1") == "1"


def _patched_default_moe_runner_forward(self, *args, **kwargs):
    if _MOE_COMPILE:
        return patched_fused_moe_forward(self, *args, **kwargs)
    return _orig_default_moe_runner_forward(self, *args, **kwargs)


MoERunnerBase.forward = _patched_default_moe_runner_forward

_orig_default_moe_runner_init = MoERunnerBase.__init__


def _patched_default_moe_runner_init(self, *args, **kwargs):
    _orig_default_moe_runner_init(self, *args, **kwargs)
    # VLLM_HPU_MOE_GLUE is read once per MoE block, here; exactly 1 means the fused glue. With any other
    # value nothing is attached and the block is what it was.
    if gaudi_envs.VLLM_HPU_MOE_GLUE == 1:
        _moe_glue_attach(self)


MoERunnerBase.__init__ = _patched_default_moe_runner_init

# Note: after upstream PR #41184, FusedMoE is a factory function (not an
# nn.Module subclass), so there is no FusedMoE.__init__ to patch. The
# dp_size==1 fast path in patched_fused_moe_forward operates directly on the
# MoERunner (self.routed_experts / self.gate), so no stashed layer reference
# is required.

vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = get_compressed_expert_map
vllm.model_executor.layers.fused_moe.router.router_factory.create_fused_moe_router = create_fused_moe_router
vllm.model_executor.layers.fused_moe.layer.create_fused_moe_router = create_fused_moe_router

# Enable non-gated (is_act_and_mul=False) MoE on HPU.
#
# vLLM's FusedMoEConfig.__post_init__ ends with a guard that raises
# NotImplementedError for non-gated activations on any platform that is not
# CUDA/XPU/ROCm -- that guard lives in vLLM Python core and has no HPU branch,
# so it fires even though the Habana fused-MoE kernel now handles non-gated
# experts directly. Relax it: run the original __post_init__ and swallow only
# that guard. We identify the guard by CONFIG STATE (a non-gated activation),
# not by the exception message -- the message is prose upstream can reword at
# will, whereas ``is_act_and_mul`` is a stable public property. The guard is the
# final statement of __post_init__, so all other configuration has already
# completed by the time it fires -- swallowing it leaves a fully-initialized
# config. Drop this once vLLM core adds HPU to the supported-platform list.
from vllm.model_executor.layers.fused_moe import config as _vllm_moe_config  # noqa: E402

_orig_moe_config_post_init = _vllm_moe_config.FusedMoEConfig.__post_init__


def _patched_moe_config_post_init(self):
    try:
        _orig_moe_config_post_init(self)
    except NotImplementedError:
        # Swallow ONLY vLLM core's non-gated-activation platform guard, which on
        # HPU (is_cuda_alike()/is_xpu() both False) fires for every non-gated
        # activation. Identify it by config state, not the exception message: a
        # non-gated config (is_act_and_mul False) is the exact and only case the
        # guard targets. Any NotImplementedError from a gated config is unrelated
        # -- re-raise it untouched.
        if self.is_act_and_mul:
            raise


_vllm_moe_config.FusedMoEConfig.__post_init__ = _patched_moe_config_post_init
