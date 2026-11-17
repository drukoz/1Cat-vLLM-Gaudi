# SPDX-License-Identifier: Apache-2.0
"""HPU PLE layer for the Qwen4Exp (Flash-Next) model (M0, T4b).

``HpuQwen4ExpPLELayer`` replaces the upstream ``Qwen4ExpPLELayer`` through the
``vllm.models.qwen4_exp.nvidia.model.Qwen4ExpPLELayer`` seam. The upstream
``forward`` cannot run in this fork: the n-gram id custom op's torch branch never
fills its output buffer, ``ple_gate`` is Triton-only, and ``_short_conv`` expects
a per-layer metadata dict while the fork passes one metadata object for the whole
model. This layer calls the pure-torch T4a ops and reads/writes its dilated
short-conv state in the ``kv_cache[0]`` slots of its SHORT_CONV mamba group,
exactly the way the fork's GDN layer reads its SSM state.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer

from vllm_gaudi.models.qwen3_5 import _save_ssm_state
from vllm_gaudi.models.qwen4_exp.ple_ops import ple_gate_torch, ple_short_conv_batched_torch


class HpuQwen4ExpPLELayer(Qwen4ExpPLELayer):
    """PLE layer with torch n-gram ids, torch gate and a batched short conv over slot state."""

    # Bound by the model runner (``maybe_set_mamba_kv_cache_groups_ids``) to the
    # index of the KV-cache group owning this layer's conv state. Kept as a
    # tensor so torch.compile treats it as dynamic (same as the GDN layer).
    cache_group_idx: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.flatten(0, -2)
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        # The custom op's torch branch returns the ids and never fills the
        # preallocated buffer, so call the pure-torch path and use its return.
        ngram_ids = self.ple_embedding.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        embeddings = self.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated, normed = ple_gate_torch(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        out = self._hpu_short_conv(gated, normed)
        return out.view(orig_shape)

    def _state_indices(self, md: object, name: str) -> torch.Tensor | None:
        """Resolve one ``[num_groups, bs(, 1)]`` state-index tensor for this layer's group."""
        indices = getattr(md, name, None)
        if indices is None:
            return None
        if indices.dim() > 1:
            if self.cache_group_idx is None:
                raise RuntimeError("PLE cache_group_idx not set")
            indices = indices.index_select(0, self.cache_group_idx.to(device=indices.device).view(1)).squeeze(0)
        if indices.dim() == 2:
            if indices.shape[-1] != 1:
                raise NotImplementedError("PLE state indices must carry one slot per row")
            indices = indices.squeeze(-1)
        return indices.reshape(-1).long()

    def _hpu_short_conv(self, gated: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """Add the dilated short convolution and persist the new state into the cache slots.

        Profiling (no metadata) and an unbound cache leave ``gated`` untouched, as
        upstream does. Inference reads the initial state of every row from its
        ``load_indices_tensor`` slot (padding rows read zeros, chunked prefill rows only
        when ``has_initial_states_p`` says so and the load slot is not the null block 0)
        and writes the new state to the ``store_indices_tensor`` slots through
        ``_save_ssm_state``, so invalid indices are neither read nor written.
        """
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            return gated
        md = attn_metadata
        if isinstance(md, dict):
            md = md.get(self.prefix)
            if md is None:
                return gated
        if self.kv_cache[0].numel() == 0:
            return gated
        is_prompt = bool(getattr(md, "is_prompt", False))
        # A speculative-decode step carries the ``1 + num_spec`` token padding per sequence,
        # which the batched short conv cannot express. Under an MTP speculative config the
        # target prefill is not a decode step, so only the decode path is deferred to M3.
        if (not is_prompt and self.num_spec_tokens > 0) or getattr(md, "num_accepted_tokens", None) is not None:
            raise NotImplementedError("PLE speculative decode is M3")

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, window, channel].
        view = conv_state.transpose(-1, -2) if is_conv_state_dim_first() else conv_state
        state_len = self.conv_state_len
        if view.shape[1] > state_len:
            view = view[:, -state_len:, :]
        num_slots = view.shape[0]

        load = self._state_indices(md, "load_indices_tensor")
        if load is None:
            return gated
        store = self._state_indices(md, "store_indices_tensor")
        if store is None:
            store = load

        rows = load.numel()
        tokens = normed.shape[0]
        if tokens % rows != 0:
            raise ValueError(f"PLE token count {tokens} is not a multiple of the {rows} state rows")
        seq_len = tokens // rows

        valid = (load >= 0) & (load < num_slots)
        init = view.index_select(0, torch.where(valid, load, torch.zeros_like(load)))
        init = torch.where(valid.view(rows, 1, 1), init, torch.zeros_like(init))
        if is_prompt:
            has_initial = getattr(md, "has_initial_states_p", None)
            if has_initial is None:
                mask = torch.zeros(rows, dtype=torch.bool, device=init.device)
            else:
                mask = has_initial.reshape(-1).to(device=init.device, dtype=torch.bool)
            # Block id 0 is vLLM's null block: it never holds an initial state (#1895).
            mask = mask & (load.to(device=init.device) != 0)
            init = torch.where(mask.view(rows, 1, 1), init, torch.zeros_like(init))

        padding_mask = getattr(md, "padding_mask_flat", None)
        if is_prompt and padding_mask is not None and padding_mask.numel() == tokens:
            lengths = padding_mask.view(rows, seq_len).sum(1).long()
        else:
            lengths = torch.full((rows,), seq_len, dtype=torch.long, device=normed.device)

        out, new_state = ple_short_conv_batched_torch(
            normed.view(rows, seq_len, self.hc_hidden_size),
            self.conv1d.weight,
            init,
            self.short_conv_dilation,
            lengths,
        )
        result = gated + out.reshape(tokens, self.hc_hidden_size)
        # State save in the dynamo-disabled wrapper so a compiled graph keeps the
        # in-place write (the GDN layer uses the same helper for its SSM state).
        return _save_ssm_state(result, new_state, view, store)
