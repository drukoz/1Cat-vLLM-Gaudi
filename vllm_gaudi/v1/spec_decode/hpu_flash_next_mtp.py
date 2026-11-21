# SPDX-License-Identifier: Apache-2.0
"""Flash-Next MTP drafter (M3): the HPU proposer for a ``Qwen4ExpMTP`` draft.

``propose`` and ``prepare_attn_metadata`` are copies of ``HpuEagleProposer``
(``hpu_eagle.py``) with exactly four changes, required by the Flash-Next MTP
step contract (plan v0.4, M3):

1. The draft returns ``(sample, multi')`` (``Qwen4ExpMTP.forward`` keeps the
   pre-final-mixer multi stream for the next step), so after every
   ``self.model(...)`` call the pair is split with ``split_draft_output``
   instead of the base's ``if self.method == "mtp"`` single-tensor branches.
2. The hidden state of step 1 is ``next_step_hidden(multi', last_token_indices)``
   and of every later step the previous step's ``multi'`` (the base's "mtp"
   branch re-feeds ``target_hidden_states``, which is the target's single
   stream, not the draft's multi stream, and is the wrong width).
3. Every model call carries ``spec_step_idx`` (0 for the first, ``token_index
   + 1`` in the loop) so the draft picks its MTP layer per step.
4. The per-step decode metadata carries the per-request kernel ``block_table``
   (the QSA decode attention gathers through ``md.block_table``) and the QSA
   raw-key ring slots ``load/store_indices_tensor`` (the state lattice's
   column 0); the QSA layer and GDN/PLE state read them.

The pure step logic lives in ``flash_next_mtp_steps.py`` (CPU-tested). This
file is HPU-only and is checked on hardware, run sheet stage 3e.
"""

import itertools
from collections.abc import Sequence

import torch
import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import MtpDraftLoop, draft_argmax
from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import (DRAFT_RANK_TOP, DraftRankStats,
                                                             align_draft_rope_with_target, draft_rank_log_enabled,
                                                             flash_next_target_model_name, flat_token_rows,
                                                             next_step_hidden,
                                                             ring_slots_from_lattice, split_draft_output)
from vllm_gaudi.v1.worker.flash_next_inputs import build_block_table
from vllm_gaudi.v1.worker.flash_next_decode_metadata import draft_step_block_table_and_slots
from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer
# The "vllm" logger (the runner's): a logger named after this module has no handler in a server, so its lines
# were lost (the draft's RoPE line, the one-graph line).
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HpuFlashNextMtpProposer(HpuEagleProposer):

    def __init__(self, vllm_config, *args, **kwargs):
        # Before the base class reads the draft's config and before the draft model is built.
        speculative_config = vllm_config.speculative_config
        if align_draft_rope_with_target(vllm_config.model_config, speculative_config.draft_model_config):
            logger.info("Flash-Next MTP: the draft runs with the target's plain RoPE (M-RoPE keys removed from "
                        "the draft's config)")
        super().__init__(vllm_config, *args, **kwargs)
        self._init_draft_rank_log()

    def _init_draft_rank_log(self) -> None:
        """Draft-rank logging (M3b, VLLM_FLASH_NEXT_DRAFT_RANK_LOG=1, default off).

        Factored out of ``__init__`` so the CPU gate test can run exactly
        these two lines on an ``object.__new__`` instance.
        """
        self.draft_rank_stats = DraftRankStats() if draft_rank_log_enabled() else None
        self.last_draft_top_ids = None

    def get_model_name(self, model) -> str:
        """The class name upstream's ``load_model`` looks up to find the target's image token id.

        ``SpecDecodeBaseProposer.load_model`` copies the image token of a multimodal target into the draft's
        config and knows where each listed class keeps it; for a class it does not list it reads
        ``target_model.config.image_token_index``. The Flash-Next target is not listed (upstream serves it with
        another runner) and ``Qwen4ExpConfig`` names the field ``image_token_id``, so the server died in
        ``load_model`` with an ``AttributeError`` after the target's weights were loaded. Qwen3.5 keeps the id
        in the same field and is listed: answer with its name for the Flash-Next multimodal class.
        """
        return flash_next_target_model_name(super().get_model_name(model))

    def propose(
        self,
        # [virtual_batch_size, seq_len]
        target_token_ids,
        # [virtual_batch_size, seq_len]
        target_positions,
        # [virtual_batch_size, seq_len, hidden_size]
        target_hidden_states,
        # [batch_size]
        last_token_indices,
        common_attn_metadata,
        # [num_seq, total_blocks]
        block_table_cpu_tensor,
        model_runner,
    ):
        # For decode, the virtual batch_size is batch size * num_tokens
        # and the seq_len is always 1
        batch_size = last_token_indices.shape[0]

        # A prefill-only call must never leave stale top-4 rows for the runner.
        self.last_draft_top_ids = None
        if self._draft_graph_enabled(common_attn_metadata):
            return self._propose_in_one_graph(target_token_ids, target_positions, target_hidden_states,
                                              last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                                              model_runner)
        draft_top_ids = [] if self.draft_rank_stats is not None else None

        if self.method == "eagle3":
            assert isinstance(self.model.model, Eagle3LlamaForCausalLM)
            target_hidden_states = \
                self.model.model.combine_hidden_states(
                    target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # The draft takes one row per token, [T, hc * H]: a prefill hands [1, seq, hc * H] (one bucket row), and
        # the draft reads the token count from dim 0.
        target_hidden_states = flat_token_rows(target_hidden_states)

        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
            spec_step_idx=0,
        )

        # The Flash-Next draft returns (sample, multi'): sample feeds the
        # LM head, multi' (the pre-final-mixer multi stream) feeds the next
        # draft step.
        last_hidden_states, hidden_states = split_draft_output(ret_hidden_states)
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        sample_hidden_states = torch.index_select(last_hidden_states, 0, last_token_indices)
        draft_token_ids, top_ids = self._draft_tokens(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            if top_ids is not None:
                self.last_draft_top_ids = top_ids.unsqueeze(1)
            return draft_token_ids.view(-1, 1)

        # [num_tokens, 1]
        target_positions = target_positions.view(-1)
        # [batch_size]
        positions = torch.index_select(target_positions, 0, last_token_indices)
        # [batch_size, 1, hc * hidden_size]: the draft's own multi' rows
        hidden_states = next_step_hidden(hidden_states, last_token_indices)

        # The first draft tokens
        if top_ids is not None:
            draft_top_ids.append(top_ids)
        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # VLLM_HPU_MTP_TIMING=1: the runner's step timer takes a lap per draft pass (None: no timer).
        timer = getattr(self, "mtp_step_timer", None)
        if timer is not None:
            timer.lap("pass0")
        # Positions used by prepare_attn_metadata needs to be cpu because
        # compile only mode for warmup will not do any real computations
        positions_cpu = self._draft_positions_cpu(target_positions, last_token_indices, batch_size)
        if timer is not None:
            timer.lap("pass0_wait")

        # The QSA raw-key ring is addressed by the request's GDN state slot:
        # the state lattice's column 0, identical for every draft step.
        ring_slots = None
        if getattr(common_attn_metadata, "load_indices_tensor", None) is not None:
            ring_slots = ring_slots_from_lattice(common_attn_metadata.load_indices_tensor)

        # Decode 1 token each time
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            # [batch_size]
            input_ids = draft_token_ids_list[-1].int()

            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Prepare the attn metadata
            positions_cpu += 1
            attn_metadata = self.prepare_attn_metadata(block_table_cpu_tensor, positions_cpu, model_runner, ring_slots)
            if timer is not None:
                timer.lap("pass%d_meta" % (token_index + 1))

            # [batch_size, 1]
            input_ids = input_ids.view(-1, 1)
            # [batch_size, 1]
            input_positions = clamped_positions.view(-1, 1)
            # [batch_size, 1, hidden_size]
            input_hidden_states = hidden_states.view(-1, 1, hidden_states.shape[-1])
            inputs_embeds = None

            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=input_positions,
                hidden_states=input_hidden_states,
                inputs_embeds=inputs_embeds,
                attn_metadata=attn_metadata,
                spec_step_idx=token_index + 1,
            )
            last_hidden_states, hidden_states = split_draft_output(ret_hidden_states)

            # The shape of the returned hidden_states and last_hidden_states:
            # [batch_size, 1, hidden_size]
            # viewed to: [batch_size, hidden_size]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, top_ids = self._draft_tokens(last_hidden_states[:batch_size])
            if top_ids is not None:
                draft_top_ids.append(top_ids)
            draft_token_ids_list.append(draft_token_ids)
            if timer is not None:
                timer.lap("pass%d" % (token_index + 1))

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if self.draft_rank_stats is not None:
            # [batch_size, num_speculative_tokens, DRAFT_RANK_TOP]
            self.last_draft_top_ids = torch.stack(draft_top_ids, dim=1)
        return draft_token_ids

    def fill_prefill_chunk(self, token_ids, position_ids, target_hidden_states, attn_metadata,
                           last_index: int | Sequence[int], next_token_id: int | Sequence[int]) -> None:
        """The draft's first step over a prefill chunk that completes no prompt (VLLM_HPU_MTP_CHUNK_KV=1): its
        attention layer writes its KV for the chunk's positions, as the target's layers do; nothing is sampled.

        The inputs are those of a prefill propose (``propose`` with the chunk's metadata): the tokens shifted one to
        the left, the prompt's next token at the chunk's last token (``last_index``), the target's multi stream
        of the chunk as the hidden state. A prompt batch of several rows (VLLM_HPU_PREFILL_BATCH) passes one
        ``last_index`` (into the flattened rows) and one ``next_token_id`` per row; a row's other columns past its
        last token are bucket padding, whatever they hold.
        """
        last = [int(last_index)] if isinstance(last_index, int) else [int(i) for i in last_index]
        nxt = [int(next_token_id)] if isinstance(next_token_id, int) else [int(t) for t in next_token_id]
        if len(last) != len(nxt):
            raise ValueError(f"fill_prefill_chunk: {len(last)} last indices for {len(nxt)} next tokens")
        tokens = token_ids.reshape(-1)
        shifted = torch.cat((tokens[1:], tokens[:1]))
        index = torch.tensor(last, dtype=torch.int64, device=shifted.device)
        value = torch.tensor(nxt, dtype=shifted.dtype, device=shifted.device)
        shifted.index_copy_(0, index, value)
        hidden = target_hidden_states.unsqueeze(0) if target_hidden_states.dim() == 2 else target_hidden_states
        # [rows, T] as the chunk's batch has them: the draft's metadata processing takes the batch shape from the ids
        rows_shape = tuple(token_ids.shape) if token_ids.dim() == 2 else (1, -1)
        self.model(
            input_ids=shifted.view(rows_shape),
            positions=position_ids,
            hidden_states=flat_token_rows(hidden),
            inputs_embeds=None,
            attn_metadata=attn_metadata,
            spec_step_idx=0,
        )

    # VLLM_HPU_MTP_DRAFT_LOOP=1: the runner's host books give each draft row's base position (the position of the
    # request's last accepted token, num_tokens_no_spec - 2 after the step's tokens are counted), set right before
    # propose and taken by it, so the draft loop reads no device value back (no sync behind draft step 0).
    host_base_positions = None

    def _draft_positions_cpu(self, target_positions, last_token_indices, batch_size):
        """The base positions of the ``batch_size`` draft rows on the host: the runner's (VLLM_HPU_MTP_DRAFT_LOOP=1)
        when it set them for this call, else read back from the device (a sync behind draft step 0)."""
        host = self.host_base_positions
        self.host_base_positions = None
        if host is not None and int(host.shape[0]) == int(batch_size):
            return host.clone()
        return target_positions.cpu()[last_token_indices.cpu()]

    # VLLM_HPU_MTP_DRAFT_GRAPH=1: the k passes of a decode draft in one HPU graph (MtpDraftLoop). The loop is built
    # at the first such draft and kept; the number of drafts it ran is counted for the CPU engine rehearsal.
    _draft_graph = None
    draft_graph_calls = 0

    def _draft_mode_key(self) -> tuple:
        """The switchable flags a captured draft graph depends on (VLLM_HPU_VOCAB_PARALLEL_LOGITS: which LM head
        and arg max the passes run): part of the graph's key, so a runtime switch captures a graph of its own."""
        from vllm_gaudi.distributed import vocab_parallel

        return (int(vocab_parallel.enabled(self.model)), )

    def _draft_graph_enabled(self, common_attn_metadata) -> bool:
        """True for a decode draft of an MTP drafter with VLLM_HPU_MTP_DRAFT_GRAPH=1 and no draft-rank log."""
        return (int(gaudi_envs.VLLM_HPU_MTP_DRAFT_GRAPH) == 1 and self.method == "mtp"
                and self.draft_rank_stats is None and not bool(getattr(common_attn_metadata, "is_prompt", True)))

    def _draft_graph_loop(self):
        """The ``MtpDraftLoop`` of this drafter, in its own HPU graph in lazy mode (built once).

        Each pass runs the draft adapter's own forward (``type(adapter).forward``: the metadata processing, the
        RoPE tables and the forward context of that pass), not ``adapter.forward``, which ``wrap_in_hpu_graph``
        replaced with its capture and replay: the passes are captured into the loop's graph, not into graphs of
        their own.
        """
        if self._draft_graph is None:
            import habana_frameworks.torch as htorch

            adapter = self.model
            adapter_forward = type(adapter).forward

            def pass_forward(**kwargs):
                return adapter_forward(adapter, **kwargs)

            def draft_tokens(hidden_states):
                return self._draft_tokens(hidden_states)[0]

            loop = MtpDraftLoop(pass_forward, draft_tokens, self.num_speculative_tokens, self.max_model_len)
            if htorch.utils.internal.is_lazy():
                loop = htorch.hpu.wrap_in_hpu_graph(loop, disable_tensor_cache=True)
            self._draft_graph = loop
            logger.info("Flash-Next MTP: the %d draft passes of a decode step run in one HPU graph "
                        "(VLLM_HPU_MTP_DRAFT_GRAPH=1, lazy mode %s)", self.num_speculative_tokens,
                        bool(htorch.utils.internal.is_lazy()))
        return self._draft_graph

    # VLLM_HPU_MTP_DRAFT_CHAIN=1: the draft chained behind the verification (MtpDraftChain), built at its first step.
    _draft_chain = None
    draft_chain_calls = 0

    def _draft_chain_module(self, model_runner):
        """The ``MtpDraftChain`` of this drafter, in its own HPU graph in lazy mode (built once). Its passes run the
        draft adapter's own forward like ``_draft_graph_loop``'s; its ids are int32 from fp32 id tables (card rule
        9: the next step reads them outside this graph); its later passes take the sparse-QSA metadata of
        engine-step's I01 (block table and slots, no paged-attention buffers), built on the device."""
        if self._draft_chain is None:
            import habana_frameworks.torch as htorch
            from vllm_gaudi.distributed import vocab_parallel
            from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import MtpDraftChain, first_max_ids

            adapter = self.model
            adapter_forward = type(adapter).forward
            block_size = int(model_runner.block_size)

            def pass_forward(**kwargs):
                return adapter_forward(adapter, **kwargs)

            def draft_tokens(hidden_states):
                if vocab_parallel.enabled(self.model):
                    return vocab_parallel.vocab_parallel_argmax(self.model, hidden_states)
                return first_max_ids(self.model.compute_logits(hidden_states))

            def make_step_metadata(slot_mapping, block_table, ring_slots):
                return HPUAttentionMetadataV1.make_decode_metadata(
                    block_list=None,
                    block_usage=None,
                    block_groups=None,
                    input_positions=None,
                    slot_mapping=slot_mapping,
                    block_size=block_size,
                    window_block_list=None,
                    window_block_usage=None,
                    window_block_groups=None,
                    chunked_block_list=None,
                    chunked_block_usage=None,
                    chunked_block_groups=None,
                    block_table=block_table,
                    load_indices_tensor=ring_slots,
                    store_indices_tensor=ring_slots,
                )

            chain = MtpDraftChain(pass_forward, draft_tokens, self.num_speculative_tokens, self.max_model_len,
                                  block_size, model_runner._PAD_BLOCK_ID, model_runner._PAD_SLOT_ID,
                                  make_step_metadata)
            if htorch.utils.internal.is_lazy():
                chain = htorch.hpu.wrap_in_hpu_graph(chain, disable_tensor_cache=True)
            self._draft_chain = chain
            logger.info("Flash-Next MTP: the draft is chained behind the verification (VLLM_HPU_MTP_DRAFT_CHAIN=1, "
                        "k = %d, lazy mode %s)", self.num_speculative_tokens, bool(htorch.utils.internal.is_lazy()))
        return self._draft_chain

    def propose_chain(self, sampled_token_ids, cu_num_draft_tokens, real, verify_positions, verify_hidden,
                      common_attn_metadata, table_full, model_runner, verify_ids=None, spec_decode_metadata=None):
        """The chained draft (VLLM_HPU_MTP_DRAFT_CHAIN=1): ``MtpDraftChain`` on the verification's device result.

        ``sampled_token_ids`` the rejection sampler's ``[B, 1 + k]``; ``cu_num_draft_tokens`` the step's cumulative
        draft counts; ``real`` bool ``[B]`` (the rows of real requests); ``verify_positions`` and ``verify_hidden``
        (a tuple of the target's multi-stream outputs) of the verification rows; ``common_attn_metadata`` the
        verification's metadata (pass 0's); ``table_full`` int32 ``[B, max_blocks]`` the requests' attention block
        table. Stage B: ``sampled_token_ids`` None, ``verify_ids`` the verification's per-row greedy ids and
        ``spec_decode_metadata`` the step's: the greedy rejection runs inside the chain's graph (its result is the
        returned ``sampled``). Returns ``MtpChainDrafts`` on the device; nothing is read back."""
        self.last_draft_top_ids = None
        self.host_base_positions = None
        extra = {}
        if verify_ids is not None:
            extra = dict(verify_ids=verify_ids, draft_token_ids=spec_decode_metadata.draft_token_ids,
                         target_logits_indices=spec_decode_metadata.target_logits_indices,
                         bonus_logits_indices=spec_decode_metadata.bonus_logits_indices)
        out = self._draft_chain_module(model_runner)(sampled_token_ids, cu_num_draft_tokens, real, verify_positions,
                                                     tuple(verify_hidden), common_attn_metadata, table_full,
                                                     self._draft_mode_key(), **extra)
        self.draft_chain_calls += 1
        return out

    def _propose_in_one_graph(self, target_token_ids, target_positions, target_hidden_states, last_token_indices,
                              common_attn_metadata, block_table_cpu_tensor, model_runner):
        """``propose`` for a decode draft with VLLM_HPU_MTP_DRAFT_GRAPH=1: the same drafts from one replay.

        The host first takes the base positions (the runner's books with VLLM_HPU_MTP_DRAFT_LOOP=1, else a read of
        the device's) and builds the attention metadata of the passes 1 .. k-1 (``prepare_attn_metadata`` at the
        base positions + j, as the per-pass loop does between its passes); then ``MtpDraftLoop`` runs every pass.
        """
        batch_size = last_token_indices.shape[0]
        k = int(self.num_speculative_tokens)
        target_hidden_states = flat_token_rows(target_hidden_states)
        timer = getattr(self, "mtp_step_timer", None)
        step_metadata = ()
        if k > 1:
            positions_cpu = self._draft_positions_cpu(target_positions.view(-1), last_token_indices, batch_size)
            if timer is not None:
                timer.lap("graph_positions")
            # The QSA raw-key ring slots (the lattice's column 0) are sliced from pass 0's metadata inside the graph
            # (MtpDraftLoop): a view made here would be an unmatched graph input, copied by its own launch.
            step_metadata = tuple(
                self.prepare_attn_metadata(block_table_cpu_tensor, positions_cpu + step, model_runner, None)
                for step in range(1, k))
            if timer is not None:
                timer.lap("graph_meta")
        else:
            self.host_base_positions = None
        draft_token_ids = self._draft_graph_loop()(target_token_ids, target_positions, target_hidden_states,
                                                   last_token_indices, common_attn_metadata, step_metadata,
                                                   self._draft_mode_key())
        self.draft_graph_calls += 1
        if timer is not None:
            timer.lap("graph")
        return draft_token_ids

    def prepare_attn_metadata(
            self,
            # [num_seq, total_blocks]
            block_table_cpu_tensor,
            # CPU tensor: [batch_size]
            positions,
            model_runner,
            ring_slots=None):
        if gaudi_envs.VLLM_HPU_FLASH_NEXT_DECODE_METADATA == 1 and model_runner.flash_next_sparse_decode_metadata():
            return self.sparse_attn_metadata(block_table_cpu_tensor, positions, model_runner, ring_slots)
        # Prepare attn metadata on CPU. (Improve for pure HPU based attn metadata preparation)
        block_size = model_runner.block_size
        batch_size = positions.shape[0]
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        # Note: block_table_cpu_tensor doesn't include the padding
        # which might smaller than the (padded) batch_size
        num_seq = block_table_cpu_tensor.shape[0]

        # Prepare block tables list
        # block_tables_list is a nested list of shape [num_seq, num_blocks]
        # num_blocks should include the slots needed for the current token
        # positions are the context lengths, and we need +1 for num_blocks
        num_blocks = torch.ceil((positions + 1) / block_size).int()
        num_blocks = num_blocks[:num_seq].tolist()
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.append(seq_block_table)
        # Needs to be resolved by defragmenter
        block_tables_list = model_runner.defragmenter.resolve_all(block_tables_list)

        # Compute slot mapping in [batch_size, 1] shape
        clamped_positions = clamped_positions.view(-1, 1)
        block_numbers = clamped_positions // block_size

        # Limit with num_seq because block_table_cpu_tensor is in the shape [num_seq, x]
        block_numbers = block_numbers.to(torch.int64)[:num_seq]
        block_ids = torch.ones((batch_size, 1), dtype=torch.int32) * model_runner._PAD_BLOCK_ID
        block_ids[:num_seq] = block_table_cpu_tensor.gather(dim=1, index=block_numbers)
        # Needs to be resolved by defragmenter
        block_ids.apply_(model_runner.defragmenter.resolve)

        # Calculate the slot mapping and fill with padding
        slot_mapping = block_ids * block_size + clamped_positions % block_size
        dummy_slots = itertools.cycle(range(model_runner._PAD_SLOT_ID, model_runner._PAD_SLOT_ID + block_size))
        slot_mapping[num_seq:].apply_(lambda _, ds=dummy_slots: next(ds))
        # Slot mapping needs to be int64 (long) type
        slot_mapping = slot_mapping.to(torch.int64)

        block_list, block_groups, block_usage = \
            model_runner.get_habana_paged_attn_buffers(
                block_tables_list,
                slot_mapping.tolist(),
                batch_size
            )

        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        slot_mapping_device = async_h2d_copy(slot_mapping, device=self.device)
        # The QSA decode attention gathers through md.block_table; use the
        # defragmenter-resolved rows, like block_list and the target's table
        block_table_device = async_h2d_copy(build_block_table(block_tables_list, batch_size,
                                                              model_runner._block_table_max_blocks,
                                                              model_runner._PAD_BLOCK_ID),
                                            device=self.device)

        common_attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
            block_list=block_list_device,
            block_usage=block_usage_device,
            block_groups=block_groups_device,
            input_positions=None,
            slot_mapping=slot_mapping_device,
            block_size=block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
            block_table=block_table_device,
            load_indices_tensor=ring_slots,
            store_indices_tensor=ring_slots,
        )

        return common_attn_metadata

    def sparse_attn_metadata(self, block_table_cpu_tensor, positions, model_runner, ring_slots=None):
        """``prepare_attn_metadata`` under VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1: the same ``block_table`` and
        ``slot_mapping`` (``flash_next_decode_metadata.draft_step_block_table_and_slots``), built without
        per-row lists, and no paged-attention buffers, which the draft's sparse QSA layer does not read."""
        block_table, slot_mapping = draft_step_block_table_and_slots(
            block_table_cpu_tensor, positions, int(positions.shape[0]), model_runner.block_size, self.max_model_len,
            model_runner._block_table_max_blocks, model_runner._PAD_BLOCK_ID, model_runner._PAD_SLOT_ID)
        return HPUAttentionMetadataV1.make_decode_metadata(
            block_list=None,
            block_usage=None,
            block_groups=None,
            input_positions=None,
            slot_mapping=async_h2d_copy(slot_mapping, device=self.device),
            block_size=model_runner.block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
            block_table=async_h2d_copy(block_table, device=self.device),
            load_indices_tensor=ring_slots,
            store_indices_tensor=ring_slots,
        )

    def _draft_tokens(self, hidden_states):
        """``(draft token of every row [rows], top DRAFT_RANK_TOP ids [rows, top] or None)``.

        ``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` >= 1 and no draft-rank log: every rank reduces its own vocabulary shard
        and only ``[rows, 2]`` fp32 per rank is gathered (``vocab_parallel.vocab_parallel_argmax``, int32 ids).
        Otherwise the gathered logits and their arg max, as before (the rank log needs the logits).
        """
        from vllm_gaudi.distributed import vocab_parallel

        if self.draft_rank_stats is None and vocab_parallel.enabled(self.model):
            return vocab_parallel.vocab_parallel_argmax(self.model, hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        top_ids = None if self.draft_rank_stats is None else logits.topk(DRAFT_RANK_TOP, dim=-1).indices
        # draft_argmax: arg max, or int32 ids from an fp32 id table with VLLM_HPU_MTP_DRAFT_IDS=1 (card rule 9).
        return draft_argmax(logits), top_ids
