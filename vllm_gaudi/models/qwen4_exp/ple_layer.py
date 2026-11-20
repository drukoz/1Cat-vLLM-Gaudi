# SPDX-License-Identifier: Apache-2.0
"""HPU PLE layer for the Qwen4Exp (Flash-Next) model (M0, T4b).

``HpuQwen4ExpNGramEmbedding`` replaces the upstream ``Qwen4ExpNGramEmbedding``
through the ``vllm.models.qwen4_exp.nvidia.ple_layer.Qwen4ExpNGramEmbedding``
seam. The engine registers its int64 ``layer_multipliers`` buffer on the model
device, and the Gaudi lazy bridge cannot hold values above int32, so this class
keeps the hash multipliers on the host, registers the hash tables as int32
buffers and computes the ids with the int32-only ``ple_ngram_ids_int32``.

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

With ``VLLM_HPU_PLE_HOST_NGRAM_IDS=1`` the runner computes the n-gram ids on the
host (``ple_ops.host_ngram_ids``) and the layer receives them ready, in the
``ngram_context`` argument (``HpuQwen4ExpPLELayer._host_ngram_ids``): the id
computation of ``compute_ngram_ids`` then does not run in the graph.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding, Qwen4ExpPLELayer

from vllm_gaudi.models.qwen4_exp.ple_ops import (
    NGRAM_HASH_LIMBS,
    check_ngram_hash_bounds,
    ngram_hash_limbs,
    ngram_mod_weights,
    ple_gate_torch,
    ple_ngram_ids_int32,
    ple_short_conv_batched_torch,
    ple_short_conv_spec_torch,
)


def adapt_fp8_table_for_gaudi(embedding, wrapper=None) -> bool:
    """Give an FP8 n-gram table the Gaudi2 range conversion at load. True when it is in place.

    Gaudi2's FP8 ends at 240, the checkpoint's E4M3FN at 448. On the card the conversion of an FP8
    tensor to bfloat16 turns every code above 240 into inf or NaN, and the engine reads the table
    exactly that way (``embeddings.to(dtype) * weight_scale`` in ``_dequantize_embeddings``): the first
    real-model server answered token 0 at every position (rental 2026-11-20). The FP8 linears and
    experts already load through ``gaudi_weight_wrapper``, which halves the codes and doubles the
    scale; the engine's own quantization method builds the table, so it never got that loader.

    ``embedding`` is the table module (``weight``, and ``weight_scale`` when it is FP8). ``wrapper``
    defaults to ``gaudi_weight_wrapper`` on Gaudi2; Gaudi3 has the full range and is left alone. A
    table of another dtype is left alone and returns False, without importing the HPU-only module.
    """
    weight = getattr(embedding, "weight", None)
    if weight is None or weight.dtype != torch.float8_e4m3fn:
        return False
    if wrapper is None:
        import vllm_gaudi.extension.ops as hpu_ops  # HPU-only module: reached with an FP8 table only

        if not hpu_ops.is_hpu_gaudi2:
            return False
        wrapper = hpu_ops.gaudi_weight_wrapper
    weight.weight_loader = wrapper(weight.weight_loader)
    weight_scale = getattr(embedding, "weight_scale", None)
    if weight_scale is not None:
        weight_scale.weight_loader = wrapper(weight_scale.weight_loader)
    return True


class HpuQwen4ExpNGramEmbedding(Qwen4ExpNGramEmbedding):
    """N-gram embedding with the hash multipliers on the host and int32-only ids.

    The engine builds ``layer_multipliers`` as an int64 buffer on the model device,
    copies the checkpoint's multipliers onto the card in ``load_weights`` and
    multiplies them in the forward hash; the Gaudi lazy bridge narrows int64 device
    tensors to int32, so every one of those values (up to 2**62) fails to build.
    This subclass keeps the buffer as zeros, keeps the real hash tables on the host
    and mirrors them into int32 limb buffers, and computes the ids with
    ``ple_ngram_ids_int32``, which reproduces the engine's int64 hash exactly. Its
    ``_shift_precompute`` replaces the engine's ``torch.cummax`` (no HPU kernel) with a
    doubling max-scan, so the EOS cut of the n-gram history runs inside HPU graphs.
    """

    @classmethod
    def _make_layer_multipliers(cls, **kwargs: object) -> list[int]:
        # The engine registers this result as an int64 buffer on the model device;
        # zeros keep that buffer inside int32 on the HPU lazy bridge. The real
        # multipliers are computed on the host in __init__ and live in the
        # int32 ``hash_multiplier_limbs`` buffer, never in ``layer_multipliers``.
        return [0] * int(kwargs["ngram_size"])  # type: ignore[arg-type]

    @staticmethod
    def _shift_precompute(tokens: torch.Tensor, eos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The engine's ``_shift_precompute`` without ``torch.cummax``, which has no HPU kernel.

        On the Gaudi bridge ``torch.cummax`` runs on the CPU through the fallback path,
        which HPU graph capture rejects (rental run 2: ``cpu fallback is not supported
        during hpu graph capturing`` in prompt warmup). The inclusive running maximum of
        the EOS positions is computed instead by a doubling scan of ``torch.maximum`` over
        lagged copies: ``ceil(log2(L))`` fixed-shape steps, every op with an HPU kernel.
        Same return values as the engine: ``(positions [L], position_in_segment [B, L])``.
        """
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        running = eos_positions
        step = 1
        while step < seq_len:
            lag = torch.full((batch_size, step), -1, dtype=running.dtype, device=running.device)
            running = torch.maximum(running, torch.cat([lag, running[:, :-step]], dim=1))
            step *= 2
        first = torch.full((batch_size, 1), -1, dtype=running.dtype, device=running.device)
        previous_eos = torch.cat([first, running[:, :-1]], dim=1)
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """The engine's ``_shift_apply`` with ``masked_fill`` instead of ``where`` over a ``new_full`` scalar.

        Same result; ``Tensor.new_full`` has no int64 row in the Gaudi operator table while
        ``masked_fill`` has, so the shifted tokens are filled with ``eos_token_id`` directly.
        """
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return shifted.masked_fill(~valid, eos_token_id)

    def __init__(
        self,
        config,
        embedding_dim: int,
        ple_dense_layer_id: int,
        prefix: str,
        layer_name: str,
        quant_config=None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            config,
            embedding_dim,
            ple_dense_layer_id,
            prefix,
            layer_name,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        # Before any weight is loaded: an FP8 table needs the Gaudi2 range at load.
        adapt_fp8_table_for_gaudi(self.ngram_embedding)
        # The real tables, as plain Python ints on the host. The engine class, not
        # self, so the zero override above is skipped.
        multipliers = Qwen4ExpNGramEmbedding._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        sizes, offsets, _ = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "hash_multiplier_limbs",
            torch.zeros(self.ngram_size, NGRAM_HASH_LIMBS, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "hash_mod_weights",
            torch.zeros(self.ngram_heads, NGRAM_HASH_LIMBS, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer("hash_sizes", torch.zeros(self.ngram_heads, dtype=torch.int32), persistent=False)
        self.register_buffer("hash_offsets", torch.zeros(self.ngram_heads, dtype=torch.int32), persistent=False)
        self._set_hash_tables(multipliers, sizes, offsets)

    def _set_hash_tables(self, multipliers, sizes, offsets) -> None:
        """Validate the host tables and refill the int32 limb buffers in place."""
        multipliers = [int(value) for value in multipliers]
        sizes = [int(value) for value in sizes]
        offsets = [int(value) for value in offsets]
        check_ngram_hash_bounds(multipliers, sizes, offsets, self.unigram_vocab_size)
        self.hash_multipliers = tuple(multipliers)
        self.hash_head_sizes = tuple(sizes)
        self.hash_head_offsets = tuple(offsets)
        # In place, so the buffers stay on the device they were built on.
        limbs = torch.tensor([ngram_hash_limbs(m) for m in multipliers], dtype=torch.int32)
        weights = torch.tensor(ngram_mod_weights(sizes), dtype=torch.int32)
        self.hash_multiplier_limbs.copy_(limbs.to(self.hash_multiplier_limbs.device))
        self.hash_mod_weights.copy_(weights.to(self.hash_mod_weights.device))
        self.hash_sizes.copy_(torch.tensor(sizes, dtype=torch.int32).to(self.hash_sizes.device))
        self.hash_offsets.copy_(torch.tensor(offsets, dtype=torch.int32).to(self.hash_offsets.device))

    def load_weights(self, weights):
        """Load the checkpoint, keeping ``layer_multipliers`` off the device."""
        recorded: dict[str, list[int]] = {}
        table_names = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")

        def watch():
            for name, loaded_weight in weights:
                if name == "layer_multipliers":
                    # Must never be copied to the card: check the shape with the
                    # engine's message, record the values, skip the buffer copy.
                    if tuple(loaded_weight.shape) != (self.ngram_size,):
                        raise ValueError(f"Shape mismatch for {name}: expected "
                                         f"{(self.ngram_size,)}, got {tuple(loaded_weight.shape)}")
                    recorded[name] = loaded_weight.reshape(-1).tolist()
                    continue
                if name in table_names:
                    recorded[name] = loaded_weight.reshape(-1).tolist()
                yield name, loaded_weight

        loaded = super().load_weights(watch())
        if recorded:
            self._set_hash_tables(
                recorded.get("layer_multipliers", self.hash_multipliers),
                recorded.get("ngram_heads_vocab_sizes", self.hash_head_sizes),
                recorded.get("ngram_heads_offsets", self.hash_head_offsets),
            )
        if "layer_multipliers" in recorded:
            loaded.add("layer_multipliers")
        return loaded

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The engine's torch branch up to the hash, then the int32-only hash.

        Gathering the tokens before the hash is equal to the engine's order (the
        hash is elementwise) and hashes ``[num_tokens]`` rows instead of the whole
        ``[num_reqs, context + num_tokens]`` context.
        """
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc.index_select(0, request_indices)).clamp(0, packed.shape[1] - 1)
        packed.view(-1).index_copy_(0, request_indices * packed.shape[1] + columns, input_ids.to(packed.dtype))
        ngram_context = ngram_context[:num_reqs].to(device=input_ids.device, dtype=torch.long)

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(context, self.eos_token_id)
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(context, positions_2d, position_in_segment, shift, self.eos_token_id))
        adjusted_columns = columns + self.ngram_size - 1

        tokens = [
            shifted[i].reshape(-1).index_select(0, request_indices * shifted[i].shape[1] + adjusted_columns).to(
                torch.int32) for i in range(self.ngram_size)
        ]
        ids = ple_ngram_ids_int32(tokens, self.hash_multiplier_limbs, self.hash_mod_weights, self.hash_sizes,
                                  self.hash_offsets, self.heads_per_ngram).long()
        if output is not None:
            output.copy_(ids)
            return output
        return ids


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
        rider = self.__dict__.pop("_rider_rows", None)
        if rider is not None:
            # VLLM_HPU_PLE_TABLE_RIDER=1: the model forward looked this call's table rows up and summed them over
            # the ranks inside the token embedding's all-reduce (vllm_gaudi/models/qwen4_exp/__init__.py).
            rider_rows, lookup_was_fp8 = rider
            if rider_rows.shape[0] != input_ids.shape[0] or rider_rows.dtype != hidden_states.dtype:
                raise RuntimeError(f"PLE rider rows {tuple(rider_rows.shape)} {rider_rows.dtype} do not fit this "
                                   f"call ({input_ids.shape[0]} tokens, {hidden_states.dtype})")
            embeddings = self._scale_table_rows(rider_rows.flatten(-2), lookup_was_fp8)
        else:
            ngram_ids = self.ngram_ids(input_ids, query_start_loc, ngram_context)
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

    def ngram_ids(self, input_ids: torch.Tensor, query_start_loc: torch.Tensor,
                  ngram_context: torch.Tensor) -> torch.Tensor:
        """The n-gram ids of this call, ``[num_tokens, heads]``, from ``forward``'s arguments (``input_ids`` flat).

        Exactly the ops ``forward`` dispatched for them before it called this (``test_ple_host_ngram_ids.py`` pins
        that sequence), so nothing is reshaped here.
        """
        if ngram_context.dim() == 3:
            # VLLM_HPU_PLE_HOST_NGRAM_IDS=1: the runner computed the ids on the host and they arrive in the
            # ``ngram_context`` slot (see ``_host_ngram_ids``); none of the id computation ops run.
            return self._host_ngram_ids(ngram_context, input_ids.shape[0])
        # The custom op's torch branch returns the ids and never fills the
        # preallocated buffer, so call the pure-torch path and use its return.
        return self.ple_embedding.compute_ngram_ids(input_ids, query_start_loc, ngram_context)

    def local_table_rows(self, ngram_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
        """This rank's rows of the n-gram table for ``ngram_ids``: ``([num_tokens, heads, head_dim] in dtype,
        whether the lookup returned FP8)``.

        What the table's vocabulary-parallel lookup puts into its all-reduce: the rows of the ids this rank's
        shard owns, zeros for every other id (each id has one owner, so the sum over the ranks is that row
        exactly). FP8 rows are widened to ``dtype`` here, which is exact; ``_scale_table_rows`` applies the
        table's scale after the sum, as ``_dequantize_embeddings`` does after the FP8 all-reduce.
        """
        from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
        from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask

        table = self.ple_embedding.ngram_embedding
        if int(table.tp_size) > 1:
            indices = table.shard_indices
            masked, mask = get_masked_input_and_mask(ngram_ids, indices.org_vocab_start_index,
                                                     indices.org_vocab_end_index, indices.num_org_vocab_padding,
                                                     indices.added_vocab_start_index,
                                                     indices.added_vocab_end_index)
        else:
            masked, mask = ngram_ids, None
        rows = table.quant_method.embedding(table, masked.long())
        lookup_was_fp8 = bool(is_fp8(rows))
        rows = rows.to(dtype)
        if mask is not None:
            rows = rows.masked_fill(mask.unsqueeze(-1), 0)
        return rows, lookup_was_fp8

    def _scale_table_rows(self, rows: torch.Tensor, lookup_was_fp8: bool) -> torch.Tensor:
        """Summed, widened table rows as ``_dequantize_embeddings`` returns them: FP8 lookups times the table's
        global scale, any other lookup unchanged."""
        if not lookup_was_fp8:
            return rows
        weight_scale = self._get_embedding_weight_scale()
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != rows.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return rows * weight_scale.to(rows.dtype)

    def _host_ngram_ids(self, ready_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """This layer's row of the n-gram ids the runner computed on the host: ``[num_tokens, heads]``.

        ``ready_ids`` is ``ple_ops.host_ngram_ids``'s int32 ``[ple_layers, num_tokens, heads]`` (row ``i`` belongs
        to the layer with ``ple_dense_layer_id == i``), sent as the model input ``ple_ngram_ids``. The engine
        hands exactly one per-batch tensor to every PLE layer unchanged, ``ngram_context``, so
        ``_MtpTargetOutputs.forward`` puts the ready ids in that argument and ``forward`` tells them from a
        context (``[num_reqs, ngram_size - 1]``) by their rank: a shape, so the choice is part of the graph's
        input hash and nothing is read from the device. The shape must fit this call exactly; a wrong one is an
        error, never a silently wrong embedding row. ``.long()`` keeps the lookup's input as the device path's.
        """
        expected = (num_tokens, self.ple_embedding.ngram_heads)
        if ready_ids.shape[0] <= self.ple_dense_layer_id or tuple(ready_ids.shape[1:]) != expected:
            raise ValueError(f"host n-gram ids must be [ple_layers > {self.ple_dense_layer_id}, {expected[0]}, "
                             f"{expected[1]}] for this PLE call, got {tuple(ready_ids.shape)}")
        return ready_ids[self.ple_dense_layer_id].long()

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

    @staticmethod
    def _prefill_lengths(md: object, padding_mask: torch.Tensor | None, rows: int, seq_len: int,
                         device: torch.device) -> torch.Tensor:
        """Real token count of every row of a padded prompt bucket: int64 ``[rows]``, exact.

        ``query_start_loc_p`` is the runner's int32 cumulative sum of the real query lengths of the
        bucket's rows (``_form_prefill_batch``; a chunk of a chunked prefill is its own bucket), so the
        lengths are its differences. Metadata without it falls back to the padding mask summed as
        int32. The mask is never summed in its own dtype: the runner builds it in the model dtype, and
        bfloat16 holds integers exactly only up to 256 (1701 ones sum to 1704), which would cut the
        saved conv window at the wrong row.
        """
        query_start_loc = getattr(md, "query_start_loc_p", None)
        if query_start_loc is not None and query_start_loc.numel() == rows + 1:
            qsl = query_start_loc.to(device=device, dtype=torch.long)
            return (qsl[1:] - qsl[:-1]).clamp(min=0, max=seq_len)
        if padding_mask is not None and padding_mask.numel() == rows * seq_len:
            return padding_mask.view(rows, seq_len).to(torch.int32).sum(1, dtype=torch.int32).long()
        return torch.full((rows,), seq_len, dtype=torch.long, device=device)

    @staticmethod
    def _save_conv_state(result: torch.Tensor, new_state: torch.Tensor, view: torch.Tensor,
                         store: torch.Tensor) -> torch.Tensor:
        """Write every row's new conv state with one fixed-shape ``index_copy_``; returns ``result``.

        The GDN helper ``qwen3_5._save_ssm_state`` drops the rows whose index is invalid through
        ``torch.nonzero`` and a Python ``if`` on its result: a host read and a data-dependent shape,
        which an HPU graph cannot replay, on every PLE decode step of a batch of two or more. Here a
        row with an invalid index (outside the tensor: the ``-1`` padding of the compact layout)
        writes the state tensor's LAST row instead, so the write always has ``store.numel()`` rows.

        The last row is the padding row in every layout the runner allocates, never a request's
        slot: ``num_blocks + 1`` rows with ``_MAMBA_PAD_BLOCK_ID = num_blocks`` (the hybrid, the
        naive-sharing and the plain allocation in ``initialize_kv_cache``), ``pool_slots + 2`` rows
        with ``pool_pad_block_id = pool_slots + 1`` (the recurrent state pool). The padding rows of a
        bucket already carry that index and wrote that row before this change; ``qsa_side_caches``
        relies on the same row for the key ring. Several rows may write it in one call: it holds no
        state anyone reads back as a request's.
        """
        store = store.reshape(-1).to(device=view.device, dtype=torch.long)
        valid = (store >= 0) & (store < view.shape[0])
        pad_row = torch.full_like(store, view.shape[0] - 1)
        view.index_copy_(0, torch.where(valid, store, pad_row), new_state.to(device=view.device, dtype=view.dtype))
        return result

    def _hpu_short_conv(self, gated: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """Add the dilated short convolution and persist the new state into the cache slots.

        Profiling (no metadata) and an unbound cache leave ``gated`` untouched, as
        upstream does. Inference reads the initial state of every row from its
        ``load_indices_tensor`` slot (padding rows read zeros, chunked prefill rows only
        when ``has_initial_states_p`` says so and the load slot is not the null block 0)
        and writes the new state to the ``store_indices_tensor`` slots through
        ``_save_conv_state``: an invalid index is not read and writes the padding row, never a
        request's slot. Metadata that carries ``num_accepted_tokens`` is an MTP verification or
        transition step and goes to ``_hpu_short_conv_spec``.
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

        if is_prompt:
            lengths = self._prefill_lengths(md, getattr(md, "padding_mask_flat", None), rows, seq_len, normed.device)
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
        # One fixed-shape in-place write; the helper returns ``result`` so the caller's graph
        # consumes the call (the convention of the GDN layer's ``_save_ssm_state``).
        return self._save_conv_state(result, new_state, view, store)

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
        nor written. The state is written to the ``store_indices_tensor`` column-0 slot when
        align mode marks the lattices as differing (M3-T10b), otherwise to the load slot.
        """
        load = self._lattice_slot(md, "load_indices_tensor")
        if load is None:
            return gated
        store = load
        if bool(getattr(md, "lattice_store_differs", False)):
            store = self._lattice_slot(md, "store_indices_tensor")
            if store is None:
                store = load
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
        # The same fixed-shape write as on the non-spec path.
        return self._save_conv_state(result, new_state, view, store)
