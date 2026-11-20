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

import torch
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import (DRAFT_RANK_TOP, DraftRankStats, draft_rank_log_enabled,
                                                             next_step_hidden, ring_slots_from_lattice,
                                                             split_draft_output)
from vllm_gaudi.v1.worker.flash_next_inputs import build_block_table
from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer


class HpuFlashNextMtpProposer(HpuEagleProposer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_draft_rank_log()

    def _init_draft_rank_log(self) -> None:
        """Draft-rank logging (M3b, VLLM_FLASH_NEXT_DRAFT_RANK_LOG=1, default off).

        Factored out of ``__init__`` so the CPU gate test can run exactly
        these two lines on an ``object.__new__`` instance.
        """
        self.draft_rank_stats = DraftRankStats() if draft_rank_log_enabled() else None
        self.last_draft_top_ids = None

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
        draft_top_ids = [] if self.draft_rank_stats is not None else None

        if self.method == "eagle3":
            assert isinstance(self.model.model, Eagle3LlamaForCausalLM)
            target_hidden_states = \
                self.model.model.combine_hidden_states(
                    target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

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
        sample_hidden_states = last_hidden_states[last_token_indices]
        draft_token_ids, top_ids = self._draft_tokens(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            if top_ids is not None:
                self.last_draft_top_ids = top_ids.unsqueeze(1)
            return draft_token_ids.view(-1, 1)

        # [num_tokens, 1]
        target_positions = target_positions.view(-1)
        # [batch_size]
        positions = target_positions[last_token_indices]
        # [batch_size, 1, hc * hidden_size]: the draft's own multi' rows
        hidden_states = next_step_hidden(hidden_states, last_token_indices)

        # The first draft tokens
        if top_ids is not None:
            draft_top_ids.append(top_ids)
        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # Positions used by prepare_attn_metadata needs to be cpu because
        # compile only mode for warmup will not do any real computations
        target_positions_cpu = target_positions.cpu()
        positions_cpu = target_positions_cpu[last_token_indices.cpu()]

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

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if self.draft_rank_stats is not None:
            # [batch_size, num_speculative_tokens, DRAFT_RANK_TOP]
            self.last_draft_top_ids = torch.stack(draft_top_ids, dim=1)
        return draft_token_ids

    def prepare_attn_metadata(
            self,
            # [num_seq, total_blocks]
            block_table_cpu_tensor,
            # CPU tensor: [batch_size]
            positions,
            model_runner,
            ring_slots=None):
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
        return logits.argmax(dim=-1), top_ids
