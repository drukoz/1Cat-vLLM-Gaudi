# SPDX-License-Identifier: Apache-2.0
"""HPU QSA attention for Flash-Next: dense paged attention behind QsaMode.

The upstream ``vllm/models/qwen4_exp/nvidia/qsa.py`` owner runs sparse paged
attention through a FlashAttention implementation, registers itself as the
KV-cache owner and keeps side caches plus a topk buffer.  The HPU class
replaces all of that for M0: dense causal attention through a standard
``Attention`` child (exact while the indexer selects the full prefix), the
same checkpoint parameter names, and the platform KV-cache owner.  Mode
selection is the integer ``QsaMode`` enum read from
``VLLM_HPU_FLASH_NEXT_QSA_MODE`` (default ``0`` = dense); ``SPARSE`` (1) is
the M2 indexer top-k path and raises ``NotImplementedError``.
"""

from __future__ import annotations

import enum

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.qwen4_exp.nvidia import model
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.v1.attention.backend import AttentionBackend

import vllm_gaudi.envs as envs


class QsaMode(enum.IntEnum):
    """QSA attention mode for the Flash-Next Qwen4-Exp layers (next free: 2)."""

    DENSE = 0
    SPARSE = 1


class HpuQsaIndexerWeights(nn.Module):
    """Checkpoint-carrying indexer weights for dense QSA (M0).

    Holds exactly the three checkpoint modules of the upstream ``QSAIndexer``
    (``index_qk_proj``, ``q_layernorm``, ``k_layernorm``) under the indexer
    prefix so ``load_weights`` resolves every indexer tensor.  No side caches,
    no buffers and no forward: top-k selection is M2 (sparse mode).
    """

    def __init__(
        self,
        *,
        config: Qwen4ExpTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.prefix = prefix

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )


class HpuQwen4ExpQSAAttention(Qwen4ExpQSAAttention):
    """HPU QSA owner: dense paged attention behind the ``QsaMode`` flag.

    Does not call the upstream ``__init__``.  It keeps the checkpoint-shape
    construction (config checks, head counts, gated QKV projection, output
    projection, RoPE, QK norms), drops the CUDA-only fused
    qk-norm/rope/gate path, and replaces the FlashAttention implementation,
    the side caches, the topk buffer and the self-registration with a
    standard ``Attention`` child registered as ``f"{prefix}.attn"``.  That
    child is the KV-cache owner: the platform selects ``HPUAttentionBackend``
    on HPU and the CPU backend in tests.  ``get_kv_cache_spec`` is inherited
    from the parent only for API compatibility; the registered ``Attention``
    child is the layer the engine collects.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.qsa_mode = QsaMode(envs.VLLM_HPU_FLASH_NEXT_QSA_MODE)
        if self.qsa_mode is QsaMode.SPARSE:
            raise NotImplementedError("sparse QSA is M2")

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1):
            raise NotImplementedError("Qwen4Exp QSA does not support context parallelism")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # The fused qk-norm + rope + gate kernel is CUDA-only; the HPU
        # forward below is always the eager split path.
        self.use_fused_qk_norm_rope_gate = False

        # HPU tail: a standard Attention child owns the KV cache and
        # registers itself in the static forward context as
        # f"{prefix}.attn"; upstream registers the owner itself.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.indexer = HpuQsaIndexerWeights(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn.get_attn_backend()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors _hpu_qwen3next_attention_forward: the upstream forward
        # assumes 2-D tokens for the attention output while keeping 3-D gate
        # shapes, so everything runs flattened to 2-D and the caller's
        # layout is restored at the end.
        orig_shape = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        gate_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*gate_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*gate_shape, -1)
        gate = gate.reshape(*gate_shape, -1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, attn_output.shape[-1])
        attn_output = attn_output * torch.sigmoid(gate.view(-1, gate.shape[-1]))

        output, _ = self.o_proj(attn_output)
        return output.view(*orig_shape[:-1], output.shape[-1])


__all__ = [
    "HpuQsaIndexerWeights",
    "HpuQwen4ExpQSAAttention",
    "QsaMode",
]
