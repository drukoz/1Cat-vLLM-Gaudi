import os

import torch
from flashinfer_gaudi.gdn_core_decode import (
    GdnCoreDecodeDebug,
    GdnCoreDecodeMode,
    GdnGateActivation,
    check_gdn_core_decode_indices,
    gdn_core_decode,
    gdn_core_decode_available,
    gdn_recurrent_decode,
    gdn_recurrent_decode_available,
)
from flashinfer_gaudi.gdn_core_verify import (
    GDN_CORE_VERIFY_MAX_NODES,
    GdnCoreVerifyMode,
    gdn_core_verify,
    gdn_core_verify_available,
)
from flashinfer_gaudi.gdn_decode import gated_delta_rule_mtp_packed
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.forward_context import get_forward_context
from vllm_gaudi import envs

import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_fn_token_major,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
    hpu_fused_rmsnorm_gated,
    resolve_hpu_gdn_chunk_size,
    resolve_hpu_gdn_compact_repeated_kkt,
    resolve_hpu_gdn_compact_repeated_local_attn,
    resolve_hpu_gdn_compiled_qk_l2norm,
    resolve_hpu_gdn_fused_rmsnorm_gated,
    resolve_hpu_gdn_fused_state_matmul,
    resolve_hpu_gdn_neumann_iters,
    resolve_hpu_gdn_prefill_state_fp32,
    resolve_hpu_gdn_prefill_state_fp32_max_tokens,
    resolve_hpu_gdn_recursive_solver_base,
    select_hpu_gdn_prefill_state_fp32,
)
from vllm_gaudi.ops.qwen38_native_qk import (
    load_qwen38_native_qk_prep,
    qwen38_native_qk_prep,
    validate_qwen38_native_qk_shape,
)
from vllm_gaudi.ops.flashinfer_gaudi_adapter import (
    can_bind_gdn_active_state,
    maybe_run_gdn_decode_packed,
    maybe_run_gdn_fused_decode_step,
    maybe_run_gdn_prefill,
)

# Import only for enabled runs. The runtime keeps small hybrid decode buckets
# on the vendor graph and selects Triton only where the composite performance
# gate has cleared; strict mode remains the fail-closed A/B path.
_triton_gaudi_mode = os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower()
if _triton_gaudi_mode in ("hybrid", "strict"):
    from vllm_gaudi.ops.triton_gaudi import (
        gdn_decode_conv_split_packed as _triton_gdn_decode_conv_packed,
        gdn_decode_packed as _triton_gdn_decode_packed,
    )
else:
    _triton_gdn_decode_conv_packed = None
    _triton_gdn_decode_packed = None

logger = init_logger(__name__)

# The one fallback reason vllm_gaudi._warn_gdn_core_decode_without_kernel already warns about at startup: debug here.
_GDN_CORE_NO_EXTENSION = "the gdn_fused_ops extension is not loaded"


def _try_triton_gdn_decode_conv_packed(
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor | None,
    conv_weight_t: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the performance-gated split width-4 conv + GDN fast path."""
    if _triton_gdn_decode_conv_packed is None:
        return None
    if _triton_gaudi_mode == "hybrid":
        return None
    if state_indices is None or conv_weight_t is None:
        if _triton_gaudi_mode == "strict":
            missing = "state_indices" if state_indices is None else "transposed conv weight"
            raise RuntimeError(f"Gaudi Triton strict fused GDN decode requires {missing}")
        return None
    return _triton_gdn_decode_conv_packed(
        conv_state,
        ssm_state,
        mixed_qkv.contiguous(),
        gate_a.contiguous(),
        gate_b.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state_indices.contiguous(),
        conv_weight_t,
    )


def _try_triton_gdn_decode_packed(
    ssm_state: torch.Tensor,
    mixed_qkv_conv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the performance-gated packed Qwen3.5 decode candidate.

    Keeping this boundary immediately after causal-conv lets Dynamo place the
    vendor conv and the state-mutating Triton op in one HPU graph. The runtime
    validator owns the canonical TP1/single-token specialization checks.
    """
    if _triton_gdn_decode_packed is None:
        return None
    if _triton_gaudi_mode == "hybrid":
        return None
    if state_indices is None:
        if _triton_gaudi_mode == "strict":
            raise RuntimeError("Gaudi Triton strict GDN decode requires state_indices metadata")
        return None
    return _triton_gdn_decode_packed(
        ssm_state,
        mixed_qkv_conv.contiguous(),
        gate_a.contiguous(),
        gate_b.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state_indices.contiguous(),
    )


def _owned(x: torch.Tensor) -> torch.Tensor:
    """``x`` as a contiguous tensor that is not a view: what a custom HPU op must be given."""
    return x.clone(memory_format=torch.contiguous_format)


def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    state_indices = state_indices.reshape(-1).to(device=ssm_state.device, dtype=torch.long)
    if state_indices.numel() == 1:
        # A real single-request prefill always owns its only state slot. Keep
        # this hot path to one index operation per GDN layer; materializing
        # nonzero(valid) here adds a device/host synchronization.
        safe_si = torch.remainder(state_indices, ssm_state.shape[0])
        ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
        return core_attn_out
    valid = (state_indices >= 0) & (state_indices < ssm_state.shape[0])
    valid_positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_positions.numel() > 0:
        safe_si = state_indices.index_select(0, valid_positions)
        state_rows = final_state.index_select(0, valid_positions).to(device=ssm_state.device, dtype=ssm_state.dtype)
        ssm_state.index_copy_(0, safe_si, state_rows)
    return core_attn_out


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    @property
    def cache_group_idx(self):
        return self.__dict__.get("_hpu_cache_group_idx")

    @cache_group_idx.setter
    def cache_group_idx(self, value):
        # The runner hands a device tensor made from a Python int (hpu_model_runner.py). Its value is kept as a
        # Python int too, read here once at setup and never in a forward: the key of this layer's cache group in
        # the per-step memo of VLLM_HPU_GDN_GLUE_CACHE=1.
        self.__dict__["_hpu_cache_group_idx"] = value
        key = None
        if isinstance(value, int):
            key = value
        elif self.__dict__.get("gdn_glue_cache") and torch.is_tensor(value) and value.numel() == 1:
            key = int(value.item())  # only with the flag: without it nothing reads the device here
        self.__dict__["_hpu_cache_group_key"] = key

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None
        self.compact_state_group_offset = None
        self.compact_state_group_count = None
        self.dflash_conv_round_before_activation = envs.VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION
        self.dflash_full_query_conv = envs.VLLM_HPU_DFLASH2_FULL_QUERY_CONV
        self._hpu_defer_ssm_writeback = False
        self._hpu_pending_ssm_state = None

        self.mamba_chunk_size, _ = resolve_hpu_gdn_chunk_size(self.model_config)
        self.gdn_fused_state_matmul = resolve_hpu_gdn_fused_state_matmul()
        self.gdn_neumann_iters = resolve_hpu_gdn_neumann_iters()
        self.gdn_prefill_state_in_fp32 = resolve_hpu_gdn_prefill_state_fp32()
        self.gdn_prefill_state_fp32_max_tokens = resolve_hpu_gdn_prefill_state_fp32_max_tokens()
        self.gdn_recursive_solver_base = resolve_hpu_gdn_recursive_solver_base()
        self.gdn_compact_repeated_kkt = resolve_hpu_gdn_compact_repeated_kkt()
        self.gdn_compact_repeated_local_attn = (resolve_hpu_gdn_compact_repeated_local_attn())
        self.gdn_compiled_qk_l2norm = resolve_hpu_gdn_compiled_qk_l2norm()
        self.gdn_fused_rmsnorm_gated = resolve_hpu_gdn_fused_rmsnorm_gated()

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size
        self.gdn_native_qk_prep = load_qwen38_native_qk_prep()
        self.gdn_compact_qk_input = gaudi_envs.VLLM_GDN_QWEN38_COMPACT_QK
        self.gdn_native_compact_kkt = (gaudi_envs.VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT)
        self.gdn_compact_qk_factor_gate = (gaudi_envs.VLLM_GDN_COMPACT_QK_FACTOR_GATE)
        if self.gdn_compact_qk_factor_gate and not self.gdn_compact_qk_input:
            raise ValueError("VLLM_GDN_COMPACT_QK_FACTOR_GATE requires compact Q/K.")
        if self.gdn_compact_qk_input and not self.gdn_native_qk_prep:
            raise ValueError("VLLM_GDN_QWEN38_COMPACT_QK requires native Q/K preparation.")
        if self.gdn_native_compact_kkt and not self.gdn_compact_qk_input:
            raise ValueError("VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT requires compact Q/K.")
        self.gdn_token_major_causal_conv1d = (gaudi_envs.VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D)
        self.gdn_native_qk_head_repeat = None
        if self.gdn_native_qk_prep:
            self.gdn_native_qk_head_repeat = validate_qwen38_native_qk_shape(
                tp_size=self.tp_size,
                qkv_width=self.qkv_size,
                key_width=self.key_dim // self.tp_size,
                value_width=self.value_dim // self.tp_size,
                key_head_dim=self.head_k_dim,
                value_head_dim=self.head_v_dim,
                compact_qk=self.gdn_compact_qk_input,
            )

        # VLLM_HPU_GDN_GLUE_CACHE=1 (idea I25): the fused decode and verify ops get the transposed conv weight and
        # dt_bias in fp32 from buffers built at load (the hooks below), their owned int32 state indices once per
        # step and cache group (_gdn_glue_memo), and the dead zeros of the output buffer are not made.
        self.gdn_glue_cache = envs.VLLM_HPU_GDN_GLUE_CACHE == 1

        # The split TPC kernels vector-load one convolution tap across channels.
        # Materialize that tap-major layout once while the checkpoint loader
        # writes the canonical [channels, 1, width] parameter.
        self._triton_conv_weight_ready = False
        self._triton_dt_bias_ready = False
        if _triton_gaudi_mode in ("hybrid", "strict") or self.gdn_glue_cache:
            conv_weight = self.conv1d.weight
            self.register_buffer(
                "_triton_conv_weight_t",
                conv_weight.new_empty((self.conv_kernel_size, conv_weight.size(0))),
                persistent=False,
            )
            self.register_buffer(
                "_triton_dt_bias_f32",
                self.dt_bias.new_empty(self.dt_bias.shape, dtype=torch.float32),
                persistent=False,
            )
            original_weight_loader = conv_weight.weight_loader
            original_dt_bias_loader = self.dt_bias.weight_loader

            def load_conv_weight_and_transpose(param, loaded_weight):
                result = original_weight_loader(param, loaded_weight)
                with torch.no_grad():
                    self._triton_conv_weight_t.copy_(param.view(param.size(0), param.size(2)).transpose(0, 1))
                self._triton_conv_weight_ready = True
                return result

            conv_weight.weight_loader = load_conv_weight_and_transpose

            def load_dt_bias_as_fp32(param, loaded_weight):
                result = original_dt_bias_loader(param, loaded_weight)
                with torch.no_grad():
                    self._triton_dt_bias_f32.copy_(param.float())
                self._triton_dt_bias_ready = True
                return result

            self.dt_bias.weight_loader = load_dt_bias_as_fp32

        # Fused GDN core decode (track RUK-2812). GdnCoreDecodeMode(...) raises on an unknown value.
        self.gdn_core_decode_mode = GdnCoreDecodeMode(envs.VLLM_HPU_GDN_CORE_DECODE)
        self.gdn_core_decode_debug = GdnCoreDecodeDebug(envs.VLLM_HPU_GDN_CORE_DECODE_DEBUG)
        self._gdn_core_gate = {"silu": GdnGateActivation.SILU, "sigmoid": GdnGateActivation.SIGMOID}.get(
            getattr(getattr(self, "norm", None), "activation", None))
        self._gdn_core_ok = None  # None = not checked yet; then True/False, once per layer

        # Fused GDN verification (track phase 2): MTP verification steps and the plain step after one.
        self.gdn_core_verify_mode = GdnCoreVerifyMode(envs.VLLM_HPU_GDN_CORE_VERIFY)
        self._gdn_core_verify_ok = None  # None = not checked yet; then True/False, once per layer

    def _gdn_core_decode_reason(self, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> str | None:
        """Why this layer's core-decode op (FUSED or RECURRENCE) cannot serve it, or ``None`` when it can."""
        fused = self.gdn_core_decode_mode == GdnCoreDecodeMode.FUSED
        if fused:
            # RECURRENCE keeps the torch conv update and self.norm, so only FUSED needs these.
            if self._gdn_core_gate is None:
                return (f"output gate {getattr(getattr(self, 'norm', None), 'activation', None)!r} "
                        "is not silu or sigmoid")
            if self.conv1d.bias is not None:
                return "conv1d has a bias"
            if self.conv_kernel_size != 4:
                return f"conv kernel size {self.conv_kernel_size} is not 4"
        if self.head_k_dim != self.head_v_dim:
            return f"head dims k={self.head_k_dim} v={self.head_v_dim} differ"
        if fused and (conv_state.dim() != 3 or conv_state.shape[1] != 3):
            return f"conv state rows {tuple(conv_state.shape)} are not [S, 3, C]"
        if ssm_state.dtype != torch.float32:
            return f"ssm state dtype {ssm_state.dtype} is not fp32"
        if _triton_gaudi_mode == "strict":
            return "Triton strict mode runs the Triton-only decode graph"
        available = gdn_core_decode_available if fused else gdn_recurrent_decode_available
        op_name = "gdn_core_decode" if fused else "gdn_recurrent_decode"
        if ssm_state.device.type == "hpu":
            if self.head_v_dim != 128:
                return f"the HPU kernel requires head_v_dim 128, got {self.head_v_dim}"
            if not available(ssm_state.device):
                return _GDN_CORE_NO_EXTENSION
        if not available(ssm_state.device):
            return f"no {op_name} kernel for {ssm_state.device.type}"
        return None

    def _log_gdn_core_off(self, reason: str) -> None:
        """Log once per layer why it runs the torch decode path (debug level for a missing HPU extension)."""
        log = logger.debug if reason == _GDN_CORE_NO_EXTENSION else logger.warning
        log("GDN core decode (VLLM_HPU_GDN_CORE_DECODE=%d) is off for %s: %s; using the torch decode path",
            int(self.gdn_core_decode_mode), getattr(self, "prefix", type(self).__name__), reason)

    def _gdn_core_decode_ready(self, load_state_indices, store_state_indices, num_tokens, num_accepted_tokens) -> bool:
        """The per-step gate shared by FUSED and RECURRENCE; runs the precondition check once per layer."""
        if self.gdn_core_decode_mode == GdnCoreDecodeMode.OFF:
            return False
        if num_accepted_tokens is not None or self._hpu_defer_ssm_writeback:
            return False
        if (load_state_indices is None or store_state_indices is None or load_state_indices.dim() != 1
                or store_state_indices.dim() != 1 or load_state_indices.shape[0] != num_tokens
                or store_state_indices.shape[0] != num_tokens):
            return False
        if self._gdn_core_ok is None:
            reason = self._gdn_core_decode_reason(self.kv_cache[0], self.kv_cache[1])
            self._gdn_core_ok = reason is None
            if reason is not None:
                self._log_gdn_core_off(reason)
        return bool(self._gdn_core_ok)

    def _check_gdn_core_decode_indices(self, load_state_indices, store_state_indices, query_start_loc,
                                       num_tokens) -> None:
        """With VLLM_HPU_GDN_CORE_DECODE_DEBUG=1, reject slot patterns the kernels cannot handle (host sync)."""
        if self.gdn_core_decode_debug != GdnCoreDecodeDebug.CHECK:
            return
        # Every real decode row is one token and the runner's padding rows are zero tokens at the end,
        # so the last query_start_loc entry is the real row count (num_decodes counts the padding too).
        num_rows = int(query_start_loc[-1]) if query_start_loc is not None else num_tokens
        check_gdn_core_decode_indices(load_state_indices, store_state_indices, self.kv_cache[1].shape[0], num_rows)

    def _gdn_fused_constants(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(conv_weight_t [W, C], dt_bias fp32)`` for the fused ops.

        With VLLM_HPU_GDN_GLUE_CACHE=1 and both load hooks run, the buffers they filled (no op per step);
        otherwise the per-step copy and cast, as before."""
        if self.gdn_glue_cache and self._triton_conv_weight_ready and self._triton_dt_bias_ready:
            return self._triton_conv_weight_t, self._triton_dt_bias_f32
        weight = self.conv1d.weight
        return weight.view(weight.size(0), weight.size(2)).t().contiguous(), self.dt_bias.float()

    def _gdn_glue_memo(self) -> dict | None:
        """A dict that lives exactly as long as this forward (it hangs on the forward context), or None.

        Only with VLLM_HPU_GDN_GLUE_CACHE=1. Its entries are shared by the GDN layers of one cache group:
        each keeps the source tensor it was made from, and a layer whose source is another tensor makes
        its own."""
        if not self.gdn_glue_cache:
            return None
        try:
            context = get_forward_context()
        except Exception:  # noqa: BLE001 - no forward context, no memo
            return None
        memo = getattr(context, "_hpu_gdn_glue_memo", None)
        if memo is None:
            memo = {}
            context._hpu_gdn_glue_memo = memo
        return memo

    def _gdn_memo_get(self, key, source, make):
        """``make()`` once per forward for ``(key, this layer's cache group)`` and the same ``source`` tensor."""
        memo = self._gdn_glue_memo()
        if memo is None:
            return make()
        key = (key, self.__dict__.get("_hpu_cache_group_key"))
        hit = memo.get(key)
        if hit is not None and hit[0] is source:
            return hit[1]
        value = make()
        memo[key] = (source, value)
        return value

    def _owned_state_indices(self, load_state_indices, store_state_indices):
        """The owned int32 copies the fused decode op takes: one pair per step and cache group with the memo.

        Two tensors even when load and store are one tensor (no prefix caching), as without the memo."""
        load = self._gdn_memo_get("owned_load", load_state_indices,
                                  lambda: _owned(load_state_indices.to(torch.int32)))
        store = self._gdn_memo_get("owned_store", store_state_indices,
                                   lambda: _owned(store_state_indices.to(torch.int32)))
        return load, store

    def _maybe_gdn_core_decode(self, mixed_qkv, z, a, b, load_state_indices, store_state_indices, num_tokens,
                               num_accepted_tokens, query_start_loc):
        """The fused core-decode op's ``[num_tokens, HV, D]`` fp32 output, or ``None`` for the torch path."""
        if self.gdn_core_decode_mode != GdnCoreDecodeMode.FUSED:
            return None
        if not self._gdn_core_decode_ready(load_state_indices, store_state_indices, num_tokens, num_accepted_tokens):
            return None
        self._check_gdn_core_decode_indices(load_state_indices, store_state_indices, query_start_loc, num_tokens)
        # The op indexes slots absolutely, so it always gets the full pools: a narrowed
        # direct-state view would shift every index. The indices are absolute pool slots
        # in every runner mode (compact pools pad with -1, which the op treats as padding).
        conv_pool, ssm_pool = self.kv_cache[0], self.kv_cache[1]
        # The per-step inputs are slices of the projections (split / chunk). contiguous() copies them
        # only when the slice is not contiguous, and a slice of ONE row always is: at decode batch 1 the
        # op got views, and the Habana lazy bridge cannot hand a view to a custom op ("ValidateSyncInput
        # Tensors tensor_data is empty ... Strided Params Has Value: 1", first server run of the kernel,
        # 2026-11-20). owned() always makes a tensor of its own.
        if not self.gdn_glue_cache:
            weight = self.conv1d.weight
            conv_weight_t = weight.view(weight.size(0), weight.size(2)).t().contiguous()
            return gdn_core_decode(conv_pool, ssm_pool, _owned(mixed_qkv),
                                   _owned(z.reshape(num_tokens, -1)), _owned(a), _owned(b),
                                   self.A_log.float(), self.dt_bias.float(), conv_weight_t, self.norm.weight,
                                   _owned(load_state_indices.to(torch.int32)),
                                   _owned(store_state_indices.to(torch.int32)),
                                   self.head_k_dim**-0.5, self.norm.eps, int(self._gdn_core_gate))
        # VLLM_HPU_GDN_GLUE_CACHE=1: the same tensors, the constants from load and the indices once per group.
        conv_weight_t, dt_bias_f32 = self._gdn_fused_constants()
        owned_load, owned_store = self._owned_state_indices(load_state_indices, store_state_indices)
        return gdn_core_decode(conv_pool, ssm_pool, _owned(mixed_qkv),
                               _owned(z.reshape(num_tokens, -1)), _owned(a), _owned(b),
                               self.A_log.float(), dt_bias_f32, conv_weight_t, self.norm.weight,
                               owned_load, owned_store,
                               self.head_k_dim**-0.5, self.norm.eps, int(self._gdn_core_gate))

    def _use_gdn_recurrent_decode(self, load_state_indices, store_state_indices, num_tokens,
                                  num_accepted_tokens) -> bool:
        """True when this decode step's recurrence runs through gdn_recurrent_decode (mode RECURRENCE)."""
        if self.gdn_core_decode_mode != GdnCoreDecodeMode.RECURRENCE:
            return False
        return self._gdn_core_decode_ready(load_state_indices, store_state_indices, num_tokens, num_accepted_tokens)

    def _gdn_core_verify_reason(self, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> str | None:
        """Why this layer's fused verification op cannot serve it, or ``None`` when it can."""
        if self._gdn_core_gate is None:
            return (f"output gate {getattr(getattr(self, 'norm', None), 'activation', None)!r} "
                    "is not silu or sigmoid")
        if self.conv1d.bias is not None:
            return "conv1d has a bias"
        if self.conv_kernel_size != 4:
            return f"conv kernel size {self.conv_kernel_size} is not 4"
        if self.head_k_dim != self.head_v_dim:
            return f"head dims k={self.head_k_dim} v={self.head_v_dim} differ"
        if conv_state.dim() != 3 or conv_state.shape[1] < 3:
            return f"conv state rows {tuple(conv_state.shape)} are not [S, R >= 3, C]"
        if ssm_state.dtype != torch.float32:
            return f"ssm state dtype {ssm_state.dtype} is not fp32"
        if _triton_gaudi_mode == "strict":
            return "Triton strict mode runs the Triton-only decode graph"
        if self.dflash_conv_round_before_activation:
            return "VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION rounds the conv output before SiLU"
        if ssm_state.device.type == "hpu":
            if self.head_v_dim != 128:
                return f"the HPU kernel requires head_v_dim 128, got {self.head_v_dim}"
            if not gdn_core_verify_available(ssm_state.device):
                return _GDN_CORE_NO_EXTENSION
        if not gdn_core_verify_available(ssm_state.device):
            return f"no gdn_core_verify kernel for {ssm_state.device.type}"
        return None

    def _log_gdn_core_verify_off(self, reason: str) -> None:
        """Log once per layer why it runs the torch verification path (debug level for a missing HPU extension)."""
        log = logger.debug if reason == _GDN_CORE_NO_EXTENSION else logger.warning
        log("GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=%d) is off for %s: %s; using the torch verification path",
            int(self.gdn_core_verify_mode), getattr(self, "prefix", type(self).__name__), reason)

    def _maybe_gdn_core_verify(self, mixed_qkv, z, a, b, load_state_indices, store_state_indices, num_tokens,
                               num_accepted_tokens, query_start_loc, lattice_store_differs, direct_gdn_state):
        """The fused verification op's ``[num_tokens, HV, D]`` fp32 output, or ``None`` for the torch path."""
        if self.gdn_core_verify_mode != GdnCoreVerifyMode.FUSED:
            return None
        load, store, qsl = load_state_indices, store_state_indices, query_start_loc
        if num_accepted_tokens is None or load is None or qsl is None or self._hpu_defer_ssm_writeback:
            return None
        conv_pool, ssm_pool = self.kv_cache[0], self.kv_cache[1]
        B = qsl.numel() - 1
        R = conv_pool.shape[1] if conv_pool.dim() == 3 else 0
        if load.dim() == 2:  # verification step: the [B, T] lattice
            T = load.shape[1]
            if load.shape[0] != B or num_tokens != B * T or (store is not None and store.shape != load.shape):
                return None
        elif load.dim() == 1:  # the plain step after a verification
            T = 1
            if (direct_gdn_state or load.shape[0] != B or num_tokens != B
                    or (store is not None and store.shape != load.shape)):
                return None  # direct state takes a narrowed conv view in mode 0, which the op does not reproduce
        else:
            return None
        if not (1 <= T <= GDN_CORE_VERIFY_MAX_NODES and T + 2 <= R):
            return None
        if self._gdn_core_verify_ok is None:
            reason = self._gdn_core_verify_reason(conv_pool, ssm_pool)
            self._gdn_core_verify_ok = reason is None
            if reason is not None:
                self._log_gdn_core_verify_off(reason)
        if not self._gdn_core_verify_ok:
            return None
        if not self.gdn_glue_cache:
            row_indices, ssm_store = self._gdn_core_verify_rows(load, store, num_accepted_tokens, qsl, B, T, R,
                                                                lattice_store_differs)
            weight = self.conv1d.weight
            conv_weight_t = weight.view(weight.size(0), weight.size(2)).t().contiguous()
            # _owned, not contiguous(): a custom HPU op cannot be given a view (see _maybe_gdn_core_decode).
            return gdn_core_verify(conv_pool, ssm_pool, _owned(mixed_qkv), _owned(z.reshape(num_tokens, -1)),
                                   _owned(a), _owned(b), self.A_log.float(), self.dt_bias.float(), conv_weight_t,
                                   self.norm.weight, _owned(row_indices), _owned(ssm_store.to(torch.int32)),
                                   self.head_k_dim**-0.5, self.norm.eps, int(self._gdn_core_gate))
        # VLLM_HPU_GDN_GLUE_CACHE=1: the constants from load, the owned row metadata once per group and step.
        conv_weight_t, dt_bias_f32 = self._gdn_fused_constants()

        def owned_rows():
            rows, lattice = self._gdn_core_verify_rows(load, store, num_accepted_tokens, qsl, B, T, R,
                                                       lattice_store_differs)
            return _owned(rows), _owned(lattice.to(torch.int32))

        row_indices, ssm_store = self._gdn_memo_get(("verify_rows", T, B, R, bool(lattice_store_differs)), load,
                                                    owned_rows)
        return gdn_core_verify(conv_pool, ssm_pool, _owned(mixed_qkv), _owned(z.reshape(num_tokens, -1)),
                               _owned(a), _owned(b), self.A_log.float(), dt_bias_f32, conv_weight_t,
                               self.norm.weight, row_indices, ssm_store,
                               self.head_k_dim**-0.5, self.norm.eps, int(self._gdn_core_gate))

    def _gdn_core_verify_rows(self, load, store, num_accepted_tokens, qsl, B, T, R, lattice_store_differs):
        """The verify op's ``(row_indices [B, 5] int32, the SSM store lattice)``: the same for every layer of a group."""
        acc = num_accepted_tokens[:B].long()
        query_len = (qsl[1:B + 1] - qsl[:B]).long().clamp(0, T)
        conv_offset = acc.clamp(1, R - 2) - 1           # the conv path's clamp
        if load.dim() == 2:
            store_lattice = store if (lattice_store_differs and store is not None) else load
            conv_load, conv_store = load[:, 0], store_lattice[:, 0]
            ssm_load = load.gather(1, (acc.clamp(1, T) - 1).unsqueeze(1)).squeeze(1)   # the SSM path's clamp
            ssm_store = store_lattice
        else:
            base = store if store is not None else load     # conv reads and writes the base row
            conv_load, conv_store, ssm_load = base, base, load
            ssm_store = base.unsqueeze(1)
        row_indices = torch.stack([conv_load.long(), conv_offset, conv_store.long(), ssm_load.long(), query_len],
                                  dim=1).to(torch.int32).contiguous()
        return row_indices, ssm_store

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch rearrange – avoids einops graph breaks on HPU."""
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(1, query.size(0), -1, self.head_k_dim).contiguous()
        key = key.reshape(1, key.size(0), -1, self.head_k_dim).contiguous()
        value = value.reshape(1, value.size(0), -1, self.head_v_dim).contiguous()
        return query, key, value

    def _resolve_state_indices(self, attn_metadata, attribute):
        """Resolve one state-index tensor, handling 2-D cache groups (once per step and group with the memo)."""
        indices = getattr(attn_metadata, attribute, None)
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            source = indices
            indices = self._gdn_memo_get(("row", attribute), source,
                                         lambda: source.index_select(0, cg.view(1)).squeeze(0))
        return indices

    def prepare_decode_state_view(self, attn_metadata, num_tokens: int) -> None:
        """Bind a cache alias before Dynamo sees the decoder group's inputs.

        Selecting the span inside the compiled forward exposes the whole pool
        to functionalization. A cached external view limits mutation outputs
        to the active rows while retaining the scheduler-owned backing cache.
        """
        self._hpu_active_ssm_state = None
        if (attn_metadata is None or bool(getattr(attn_metadata, "is_prompt", False))
                or not bool(getattr(attn_metadata, "direct_gdn_state", False))
                or getattr(self.cache_config, "enable_prefix_caching", False) or _triton_gaudi_mode == "strict"):
            return
        indices = getattr(attn_metadata, "load_indices_tensor", None)
        if indices is None or not self.kv_cache:
            return
        batch = indices.shape[-1]
        if num_tokens != batch or not can_bind_gdn_active_state(batch, self.qkv_size, self.A_log.numel()):
            return
        pool = self.kv_cache[1]
        count, offset = self.compact_state_group_count, self.compact_state_group_offset
        if count is None or count <= 0 or offset is None or not 0 <= offset < count:
            return
        span, remainder = divmod(pool.shape[0] - 2, count)
        if remainder or not 0 < batch <= span:
            return
        key = (count, offset, batch)
        if (getattr(self, "_hpu_active_ssm_source", None) is not pool
                or getattr(self, "_hpu_active_ssm_key", None) != key):
            self._hpu_cached_ssm_view = pool.narrow(0, offset * span + 1, batch)
            self._hpu_active_ssm_source = pool
            self._hpu_active_ssm_key = key
        self._hpu_active_ssm_state = self._hpu_cached_ssm_view

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, None, 0, 0, 0, 0, None, False, None, False, False,
                    False)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        load_state_indices = self._resolve_state_indices(attn_metadata, "load_indices_tensor")
        store_state_indices = self._resolve_state_indices(attn_metadata, "store_indices_tensor")
        num_accepted_tokens = getattr(attn_metadata, "num_accepted_tokens", None)
        if not getattr(self.cache_config, "enable_prefix_caching", False) and num_accepted_tokens is None:
            # The scheduler constructs equivalent load/store tensors in this
            # ordinary decode mode. A DFlash T1 transition instead reads the
            # accepted checkpoint and commits to the canonical base slot.
            # Preserve identity only when those destinations are equivalent.
            store_state_indices = load_state_indices
        elif store_state_indices is None:
            store_state_indices = load_state_indices

        conv_state = self.kv_cache[0]
        active_state = getattr(self, "_hpu_active_ssm_state", None)
        state_is_active_view = (not is_prompt and bool(getattr(attn_metadata, "direct_gdn_state", False))
                                and active_state is not None and active_state.shape[0] == num_tokens)
        ssm_state = active_state if state_is_active_view else self.kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        direct_gdn_state = bool(getattr(attn_metadata, "direct_gdn_state", False))
        dflash_full_query = bool(getattr(attn_metadata, "dflash_full_query", False))
        lattice_store_differs = bool(getattr(attn_metadata, "lattice_store_differs", False))

        if not is_prompt:
            num_decodes = (query_start_loc.numel() - 1 if query_start_loc is not None else
                           (load_state_indices.shape[0] if load_state_indices is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and load_state_indices is not None:
            prefill_num_seqs = int(load_state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            if prefill_num_seqs == 1:
                # Avoid three validity-mask kernels in every GDN layer for
                # the latency-critical one-request prefill path.
                initial_state = ssm_state[load_state_indices].contiguous()
            else:
                safe_load_indices = torch.where(
                    (load_state_indices >= 0) & (load_state_indices < ssm_state.shape[0]),
                    load_state_indices,
                    torch.zeros_like(load_state_indices),
                ).long()
                initial_state = ssm_state.index_select(0, safe_load_indices).contiguous()
            if has_initial_state is not None:
                # A load index of 0 is vLLM's null block: it never holds an
                # initial state, so reading it must yield zeros (#1895).
                keep = has_initial_state.bool().view(-1) & (load_state_indices.reshape(-1) != 0)
                initial_state = torch.where(
                    keep.view(-1, 1, 1, 1),
                    initial_state,
                    torch.zeros_like(initial_state),
                )

        return (is_prompt, conv_state, ssm_state, load_state_indices, store_state_indices, query_start_loc,
                has_initial_state, padding_mask_flat, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
                initial_state, direct_gdn_state, num_accepted_tokens, dflash_full_query, state_is_active_view,
                lattice_store_differs)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.

        Return-based since upstream vLLM #46998 (300e33797f) dropped the
        ``output`` in-place buffer; caller now does
        ``hidden_states = self.linear_attn(hidden_states=...)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, load_state_indices, store_state_indices, query_start_loc, has_initial_state,
         padding_mask_flat, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state,
         direct_gdn_state, num_accepted_tokens, dflash_full_query,
         state_is_active_view, lattice_store_differs) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        if hasattr(self, 'in_proj_qkv'):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
                # Pure-torch flatten instead of einops rearrange (graph breaks)
                query = query.reshape(query.size(0), -1)
                key = key.reshape(key.size(0), -1)
                value = value.reshape(value.size(0), -1)
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights already in [q, k, v, z] and [b, a] order
                mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        core_attn_out = None
        if not self.gdn_glue_cache:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        gdn_core_out = None
        use_gdn_recurrent = False
        if conv_state is not None and not is_prompt:
            gdn_core_out = self._maybe_gdn_core_decode(mixed_qkv, z, a, b, load_state_indices, store_state_indices,
                                                       num_tokens, num_accepted_tokens, query_start_loc)
            if gdn_core_out is None:
                gdn_core_out = self._maybe_gdn_core_verify(mixed_qkv, z, a, b, load_state_indices, store_state_indices,
                                                           num_tokens, num_accepted_tokens, query_start_loc,
                                                           lattice_store_differs, direct_gdn_state)
            use_gdn_recurrent = self._use_gdn_recurrent_decode(load_state_indices, store_state_indices, num_tokens,
                                                               num_accepted_tokens)
            if use_gdn_recurrent:
                self._check_gdn_core_decode_indices(load_state_indices, store_state_indices, query_start_loc,
                                                    num_tokens)

        if core_attn_out is None and gdn_core_out is None:
            # VLLM_HPU_GDN_GLUE_CACHE=1: the buffer only where a path below writes into it (the fused op's
            # output replaces it).
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if conv_state is None:
            # No attn_metadata — skip core attention (profile run)
            pass
        elif is_prompt:
            # === Part 2a: Prefill ====================================
            if (padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens):
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if self.gdn_token_major_causal_conv1d:
                mixed_qkv_conv = hpu_causal_conv1d_fn_token_major(
                    x=mixed_qkv,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=load_state_indices,
                    query_start_loc=query_start_loc,
                    block_size_to_align=mamba_block_size,
                    is_prompt=True,
                )
            else:
                mixed_qkv_conv = hpu_causal_conv1d_fn(
                    x=mixed_qkv.transpose(0, 1),
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=store_state_indices,
                    block_idx_first_scheduled_token=None,
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=load_state_indices,
                    query_start_loc=query_start_loc,
                    block_size_to_align=mamba_block_size,
                    num_computed_tokens=None,
                    metadata=None,
                    is_prompt=True,
                ).transpose(0, 1)

            if token_mask_flat is not None:
                mixed_qkv_conv = mixed_qkv_conv * token_mask_flat

            if self.gdn_native_qk_prep:
                query, key = qwen38_native_qk_prep(mixed_qkv_conv)
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value_offset = 2 * (self.key_dim // self.tp_size)
                value = mixed_qkv_conv.narrow(
                    -1,
                    value_offset,
                    self.value_dim // self.tp_size,
                ).reshape(
                    1,
                    mixed_qkv_conv.size(0),
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                ).contiguous()
                use_qk_l2norm_in_kernel = False
            else:
                query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)
                use_qk_l2norm_in_kernel = True

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            prefill_state_in_fp32 = select_hpu_gdn_prefill_state_fp32(
                self.gdn_prefill_state_in_fp32,
                self.gdn_prefill_state_fp32_max_tokens,
                prefill_seq_len,
            )
            flashinfer_prefill_result = maybe_run_gdn_prefill(
                q=query,
                k=key,
                v=value,
                log_decay=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                chunk_size=self.mamba_chunk_size,
                prefill_num_seqs=prefill_num_seqs,
                prefill_seq_len=prefill_seq_len,
                scale=self.head_k_dim**-0.5,
            )
            if flashinfer_prefill_result is not None:
                core_attn_out_result, final_state = flashinfer_prefill_result
            else:
                core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    chunk_size=self.mamba_chunk_size,
                    prefill_num_seqs=prefill_num_seqs,
                    prefill_seq_len=prefill_seq_len,
                    neumann_iters=self.gdn_neumann_iters,
                    fused_state_matmul=self.gdn_fused_state_matmul,
                    recursive_solver_base=self.gdn_recursive_solver_base,
                    compact_repeated_kkt=self.gdn_compact_repeated_kkt,
                    compile_qk_l2norm=self.gdn_compiled_qk_l2norm,
                    qk_head_repeat_override=self.gdn_native_qk_head_repeat,
                    compact_repeated_local_attn=self.gdn_compact_repeated_local_attn,
                    preserve_compact_qk=self.gdn_compact_qk_input,
                    compact_qk_factor_gate=self.gdn_compact_qk_factor_gate,
                    state_in_fp32=prefill_state_in_fp32,
                )
            assert final_state is not None
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                store_state_indices,
            )

            non_spec_out = core_attn_out_result.squeeze(0)
            core_attn_out[:non_spec_out.shape[0]] = non_spec_out

        elif gdn_core_out is not None:
            # === Part 2c: fused core decode/verify (one op: conv, gates, recurrence, gated RMSNorm; serves
            # gdn_core_decode and gdn_core_verify) ===
            core_attn_out = gdn_core_out.to(hidden_states.dtype)

        else:
            # === Part 2b: Decode =====================================
            has_dflash_state = num_accepted_tokens is not None and load_state_indices is not None
            is_spec_decode = has_dflash_state and load_state_indices.dim() == 2
            is_dflash_transition = has_dflash_state and load_state_indices.dim() == 1
            selected_conv_state = conv_state
            direct_conv_state = False
            if (not is_spec_decode and direct_gdn_state and self.compact_state_group_count is not None
                    and self.compact_state_group_count > 0 and self.compact_state_group_offset is not None):
                group_span = (conv_state.shape[0] - 2) // self.compact_state_group_count
                if num_decodes <= group_span:
                    state_start = self.compact_state_group_offset * group_span + 1
                    selected_conv_state = conv_state.narrow(0, state_start, num_decodes)
                    # A speculative cache reserves one extra convolution slot.
                    # The promoted direct kernel consumes only the canonical
                    # width-1 history; MTP rejection is handled by the explicit
                    # checkpoint rather than by exposing the reserve slot.
                    selected_conv_state = selected_conv_state[:, -(self.conv_kernel_size - 1):, :]
                    direct_conv_state = True
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0),
                self.conv1d.weight.size(2),
            )
            # Strict decode must validate the mutation contract before convolution
            # changes its cache. Other modes preserve the mainline graph tactics.
            if _triton_gaudi_mode == "strict" and (load_state_indices is None
                                                   or load_state_indices is not store_state_indices):
                raise RuntimeError("Triton strict GDN requires identical load/store state-index tensors")
            flashinfer_result = None
            # RECURRENCE keeps the torch conv below, so it skips the fork's fused conv-and-recurrence step.
            if _triton_gaudi_mode != "strict" and not has_dflash_state and not use_gdn_recurrent:
                flashinfer_result = maybe_run_gdn_fused_decode_step(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    conv_state=selected_conv_state,
                    conv_weight=conv_weights,
                    conv_bias=self.conv1d.bias,
                    ssm_state=ssm_state,
                    load_state_indices=load_state_indices,
                    direct_conv_state=direct_conv_state,
                    direct_gdn_state=direct_gdn_state,
                    direct_state_group_count=self.compact_state_group_count,
                    direct_state_group_offset=self.compact_state_group_offset,
                    scale=self.head_k_dim**-0.5,
                    state_is_active_view=state_is_active_view,
                    defer_state_writeback=self._hpu_defer_ssm_writeback,
                )
            if flashinfer_result is None:
                if state_is_active_view and not use_gdn_recurrent:
                    raise RuntimeError("Bound active GDN state requires the fused direct decode adapter")
                if _triton_gaudi_mode != "strict" and not use_gdn_recurrent:
                    # gdn_recurrent_decode computes the gates itself from a, b, A_log and dt_bias.
                    g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
                conv_indices = load_state_indices
                if is_spec_decode:
                    conv_indices = load_state_indices[:num_decodes, 0]
                elif is_dflash_transition and store_state_indices is not None:
                    # Recurrent state loads from the accepted checkpoint,
                    # while convolution history uses the accepted offset in
                    # the request's base row.
                    conv_indices = store_state_indices[:num_decodes]
                elif conv_indices is not None:
                    conv_indices = conv_indices[:num_decodes]
                mixed_qkv_conv = hpu_causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=selected_conv_state,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=None if direct_conv_state else conv_indices,
                    num_accepted_tokens=num_accepted_tokens if has_dflash_state else None,
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=None,
                    query_start_loc=query_start_loc,
                    max_query_len=(load_state_indices.size(1) if is_spec_decode else 1 if is_dflash_transition else -1),
                    validate_data=False,
                    direct_state_layout=direct_conv_state,
                    round_before_activation=self.dflash_conv_round_before_activation,
                    full_query_valid_state=(self.dflash_full_query_conv and is_spec_decode and dflash_full_query
                                            and self.compact_state_group_offset is not None
                                            and not getattr(self.cache_config, "enable_prefix_caching", True)),
                    store_cache_indices=(store_state_indices[:num_decodes] if (store_state_indices is not None
                                                                               and not direct_conv_state
                                                                               and not is_spec_decode
                                                                               and not is_dflash_transition)
                                         else (store_state_indices[:num_decodes, 0] if (
                                             is_spec_decode
                                             and lattice_store_differs
                                             and store_state_indices is not None) else None)),
                )
                if use_gdn_recurrent:
                    # === RECURRENCE: the recurrence alone through gdn_recurrent_decode (full SSM pool, ===
                    # === absolute slots); self.norm runs in Part 3 as in mode 0.                       ===
                    core_attn_out_result = gdn_recurrent_decode(
                        self.kv_cache[1], mixed_qkv_conv.contiguous(), a.contiguous(), b.contiguous(),
                        self.A_log.float(), self.dt_bias.float(), load_state_indices.to(torch.int32),
                        store_state_indices.to(torch.int32), self.head_k_dim**-0.5).unsqueeze(0)
                elif _triton_gaudi_mode == "strict":
                    triton_out = _try_triton_gdn_decode_packed(
                        ssm_state,
                        mixed_qkv_conv,
                        a,
                        b,
                        self.A_log,
                        self._triton_dt_bias_f32 if self._triton_dt_bias_ready else self.dt_bias,
                        load_state_indices,
                    )
                    if triton_out is None:
                        raise RuntimeError("Triton strict GDN did not select a recurrent kernel")
                    flashinfer_result = (triton_out.unsqueeze(0), ssm_state)
                elif is_spec_decode:
                    assert query_start_loc is not None
                    tokens_per_request = load_state_indices.size(1)
                    state_indices = load_state_indices[:num_decodes, :tokens_per_request]
                    query_lens = (query_start_loc[1:num_decodes + 1] - query_start_loc[:num_decodes]).clamp(
                        min=0,
                        max=tokens_per_request,
                    )
                    checkpoint_start = None
                    if (direct_gdn_state and dflash_full_query and self.compact_state_group_offset is not None
                            and self.compact_state_group_count is not None and self.compact_state_group_count > 0):
                        group_span = (ssm_state.shape[0] - 2) // self.compact_state_group_count
                        checkpoint_start = self.compact_state_group_offset * group_span + 1
                    core_attn_out_result, _ = gated_delta_rule_mtp_packed(
                        packed_qkv=mixed_qkv_conv.reshape(num_decodes, tokens_per_request, -1),
                        log_decay=g.reshape(num_decodes, tokens_per_request, -1),
                        beta=beta.reshape(num_decodes, tokens_per_request, -1),
                        state_pool=ssm_state,
                        state_indices=state_indices,
                        num_accepted_tokens=num_accepted_tokens[:num_decodes],
                        query_lengths=query_lens,
                        scale=self.head_k_dim**-0.5,
                        use_qk_l2norm=True,
                        assume_full_query=dflash_full_query,
                        # Compact no-prefix-cache state assigns a different
                        # contiguous slot to every verification token.
                        assume_distinct_checkpoints=(self.compact_state_group_offset is not None
                                                     and not getattr(self.cache_config, "enable_prefix_caching", True)),
                        checkpoint_start=checkpoint_start,
                        # Align mode (M3-T10b): token t's checkpoint lands on the
                        # store lattice so a block-boundary token cannot overwrite
                        # a completed prefix-cache block.
                        store_indices=(store_state_indices[:num_decodes, :tokens_per_request]
                                       if (lattice_store_differs and store_state_indices is not None) else None),
                    )
                    core_attn_out_result = core_attn_out_result.reshape(1, num_tokens, core_attn_out_result.size(-2),
                                                                        core_attn_out_result.size(-1))
                else:
                    flashinfer_result = maybe_run_gdn_decode_packed(
                        mixed_qkv=mixed_qkv_conv,
                        log_decay=g,
                        beta=beta,
                        state_pool=ssm_state,
                        load_state_indices=load_state_indices,
                        store_state_indices=store_state_indices,
                        use_qk_l2norm=True,
                        scale=self.head_k_dim**-0.5,
                        direct_state_layout=direct_gdn_state,
                        direct_state_group_count=self.compact_state_group_count,
                        direct_state_group_offset=self.compact_state_group_offset,
                        allow_indexed_reference=is_dflash_transition,
                    )
                if flashinfer_result is None and not is_spec_decode and not use_gdn_recurrent:
                    query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)
                    core_attn_out_result, _ = hpu_fused_recurrent_gated_delta_rule(
                        q=query,
                        k=key,
                        v=value,
                        g=g,
                        beta=beta,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=(query_start_loc[:num_decodes + 1] if query_start_loc is not None else None),
                        ssm_state_indices=load_state_indices,
                        use_qk_l2norm_in_kernel=True,
                        ssm_store_indices=(store_state_indices[:num_decodes] if store_state_indices is not None
                                           else None),
                    )
            if flashinfer_result is not None:
                core_attn_out_result, next_state = flashinfer_result
                if self._hpu_defer_ssm_writeback:
                    self._hpu_pending_ssm_state = next_state

            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        # === Part 3: Output Projection ===============================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if gdn_core_out is not None:
            pass  # gdn_core_decode already applied the gated RMSNorm
        elif is_prompt and self.gdn_fused_rmsnorm_gated:
            core_attn_out = hpu_fused_rmsnorm_gated(
                core_attn_out,
                z,
                self.norm.weight,
                self.norm.eps,
                self.norm.activation,
            )
        else:
            core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)

        output, _ = self.out_proj(core_attn_out)
        # Restore caller's original layout (2-D flat or 3-D [B, L, H]) so
        # the residual add in the decoder layer stays shape-consistent.
        return output.view(orig_shape)


# Replace the class in the upstream modules so that both Qwen3-Next and
# Qwen3.5 model definitions instantiate HPUGatedDeltaNetAttention.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_module  # noqa: E402
import vllm.model_executor.models.qwen3_next as _qwen3_next_module  # noqa: E402
import vllm.model_executor.models.qwen3_5 as _qwen3_5_module  # noqa: E402

_gdn_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_next_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_5_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
