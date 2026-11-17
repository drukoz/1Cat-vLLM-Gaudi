# SPDX-License-Identifier: Apache-2.0
"""HPU PLE layer for the Qwen4Exp (Flash-Next) model (M0, T4b).

``HpuQwen4ExpPLELayer`` replaces the upstream ``Qwen4ExpPLELayer`` through the
``vllm.models.qwen4_exp.nvidia.model.Qwen4ExpPLELayer`` seam. The upstream
``forward`` cannot run in this fork: the n-gram id custom op's torch branch never
fills its output buffer, ``ple_gate`` is Triton-only, and ``_short_conv`` expects
a per-layer metadata dict while the fork passes one metadata object for the whole
model. This layer calls the pure-torch T4a ops and reads/writes its dilated
short-conv state in the ``kv_cache[0]`` slots of its SHORT_CONV mamba group,
exactly the way the fork's GDN layer reads its SSM state. Under MTP
speculative decode (M3) it also runs verification and transition steps from
the state lattice, rolling the conv history back by ``num_accepted_tokens``
and storing ``history + candidates`` the way the fork's GDN conv update does.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer

from vllm_gaudi.models.qwen3_5 import _save_ssm_state
from vllm_gaudi.models.qwen4_exp.ple_ops import (
    ple_gate_torch,
    ple_short_conv_batched_torch,
    ple_short_conv_spec_torch,
)


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

    def _lattice_slot(self, md: object, name: str) -> torch.Tensor | None:
        """Resolve one ``[groups, bs, T]``/``[groups, bs]`` state lattice to the ``[bs]`` column-0 slots.

        The group row is picked as ``_state_indices`` picks it; the conv state of a request
        lives in the column-0 slot of its lattice row (all T rows of a verification step
        share the state row), so column 0 is the slot.
        """
        indices = getattr(md, name, None)
        if indices is None:
            return None
        if indices.dim() > 1:
            if self.cache_group_idx is None:
                raise RuntimeError("PLE cache_group_idx not set")
            indices = indices.index_select(0, self.cache_group_idx.to(device=indices.device).view(1)).squeeze(0)
        if indices.dim() == 2:
            indices = indices[:, 0]
        return indices.reshape(-1).long()

    def _hpu_short_conv(self, gated: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """Add the dilated short convolution and persist the new state into the cache slots.

        Profiling (no metadata) and an unbound cache leave ``gated`` untouched, as
        upstream does. Inference reads the initial state of every row from its
        ``load_indices_tensor`` slot (padding rows read zeros, chunked prefill rows only
        when ``has_initial_states_p`` says so and the load slot is not the null block 0)
        and writes the new state to the ``store_indices_tensor`` slots through
        ``_save_ssm_state``, so invalid indices are neither read nor written. Metadata that
        carries ``num_accepted_tokens`` is an MTP verification or transition step and goes
        to ``_hpu_short_conv_spec``.
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

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, window, channel].
        view = conv_state.transpose(-1, -2) if is_conv_state_dim_first() else conv_state

        num_accepted = getattr(md, "num_accepted_tokens", None)
        if num_accepted is not None:
            return self._hpu_short_conv_spec(md, num_accepted, gated, normed, view)

        state_len = self.conv_state_len
        if view.shape[1] > state_len:
            # The fork's conv convention (hpu_causal_conv1d_fn): an expanded speculative
            # state keeps its committed history in rows [0, state_len), so prefill and
            # plain decode under a speculative config read and write the first rows.
            view = view[:, :state_len, :]
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
        is_prompt = bool(getattr(md, "is_prompt", False))

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

    def _hpu_short_conv_spec(
        self,
        md: object,
        num_accepted: torch.Tensor,
        gated: torch.Tensor,
        normed: torch.Tensor,
        view: torch.Tensor,
    ) -> torch.Tensor:
        """MTP verification (``[groups, bs, T]`` lattice) or transition (``[groups, bs]``, T = 1) step.

        Each request reads the committed history window its ``num_accepted_tokens`` selects
        from its column-0 lattice slot, computes the T outputs over ``[history ; candidates]``
        and stores ``[history[1:] ; candidates]`` back, so the next pass rolls the rejected
        candidate rows out of the history (the convention of ``_hpu_causal_conv1d_spec_update``).
        Padding rows are masked to zero and rows whose load slot is invalid are neither read
        nor written; ``store_indices_tensor`` is identical to the load lattice by contract.
        """
        load = self._lattice_slot(md, "load_indices_tensor")
        if load is None:
            return gated
        rows = load.numel()
        tokens = normed.shape[0]
        if tokens % rows != 0:
            raise ValueError(f"PLE token count {tokens} is not a multiple of the {rows} state rows")
        width = tokens // rows
        num_slots = view.shape[0]

        query_start_loc = getattr(md, "query_start_loc_p", None)
        if query_start_loc is None:
            query_lens = torch.full((rows,), width, dtype=torch.long, device=normed.device)
        else:
            qsl = query_start_loc.to(device=normed.device, dtype=torch.long)
            query_lens = (qsl[1:rows + 1] - qsl[:rows]).clamp(min=0, max=width)

        valid = (load >= 0) & (load < num_slots)
        init = view.index_select(0, torch.where(valid, load, torch.zeros_like(load)))
        init = torch.where(valid.view(rows, 1, 1), init, torch.zeros_like(init))

        out, new_state = ple_short_conv_spec_torch(
            normed.view(rows, width, self.hc_hidden_size),
            self.conv1d.weight,
            init,
            num_accepted.reshape(-1)[:rows],
            query_lens,
            self.short_conv_dilation,
        )
        result = gated + out.reshape(tokens, self.hc_hidden_size)
        # State save in the dynamo-disabled wrapper, as on the non-spec path.
        return _save_ssm_state(result, new_state, view, load)
