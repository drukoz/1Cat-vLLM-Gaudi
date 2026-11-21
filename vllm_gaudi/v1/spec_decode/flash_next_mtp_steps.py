# SPDX-License-Identifier: Apache-2.0
"""Pure CPU helpers for the Flash-Next MTP drafter (M3).

``HpuFlashNextMtpProposer`` runs the ``Qwen4ExpMTP`` draft on HPU, so it
cannot be imported (let alone tested) on a machine without one. Every piece
of step logic that needs no HPU lives here: the draft-model detection, the
``(sample, multi)`` split of the draft's output, the per-step hidden state
and the QSA ring slots (the GDN state lattice's column 0). The per-step
kernel ``block_table`` reuses ``flash_next_inputs.build_block_table`` on the
defragmenter-resolved rows, exactly as the target's decode does.

``hpu_flash_next_mtp.py`` stays a thin copy of ``HpuEagleProposer.propose``
that calls these helpers. Imports stay torch/os/vllm-logger only,
so this module is unit-testable on CPU (rule from the plan's conventions).
``DraftRankStats`` (M3b, ``VLLM_FLASH_NEXT_DRAFT_RANK_LOG=1``) histograms the
target token's rank among the drafter's top-4 at the first rejected position.
"""

import copy
import dataclasses
import os
from typing import NamedTuple

import torch
from vllm.logger import init_logger

FLASH_NEXT_MTP_ARCH = "Qwen4ExpMTP"

logger = init_logger(__name__)


def is_flash_next_mtp(speculative_config) -> bool:
    """method == "mtp" and "Qwen4ExpMTP" in draft_model_config.hf_config.architectures."""
    if getattr(speculative_config, "method", None) != "mtp":
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    return FLASH_NEXT_MTP_ARCH in architectures


FLASH_NEXT_MULTIMODAL_TARGET = "Qwen4ExpForConditionalGeneration"
IMAGE_TOKEN_ID_TARGET = "Qwen3_5ForConditionalGeneration"


def flash_next_target_model_name(class_name: str) -> str:
    """The name the proposer's ``load_model`` should look the target up by.

    The Flash-Next multimodal target (``Qwen4ExpForConditionalGeneration`` or the plugin's subclass of it,
    ``HpuQwen4ExpForConditionalGeneration``) keeps its image token in ``config.image_token_id``, like
    ``Qwen3_5ForConditionalGeneration``, which upstream's ``load_model`` lists: that name is returned for it.
    Any other name is returned unchanged.
    """
    if class_name.endswith(FLASH_NEXT_MULTIMODAL_TARGET):
        return IMAGE_TOKEN_ID_TARGET
    return class_name


def split_draft_output(ret) -> tuple[torch.Tensor, torch.Tensor]:
    """(sample, multi) from the draft's return.

    TypeError unless ret is a 2-tuple with multi.shape[-1] % sample.shape[-1] == 0.
    """
    if not isinstance(ret, tuple) or len(ret) != 2:
        raise TypeError("the Flash-Next MTP draft must return a (sample, multi) 2-tuple, "
                        f"got {type(ret).__name__}"
                        + (f" of length {len(ret)}" if isinstance(ret, tuple) else ""))
    sample, multi = ret
    if multi.shape[-1] % sample.shape[-1] != 0:
        raise TypeError(f"draft multi stream {tuple(multi.shape)} is not a whole multiple of the "
                        f"sample hidden size {sample.shape[-1]}")
    return sample, multi


MROPE_KEYS = ("mrope_section", "mrope_interleaved")


def _rope_parameter_dicts(model_config) -> list[dict]:
    """The ``rope_parameters`` dicts of a model config's ``hf_config`` and ``hf_text_config`` (each once)."""
    found: dict[int, dict] = {}
    for config in (getattr(model_config, "hf_config", None), getattr(model_config, "hf_text_config", None)):
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            found[id(rope_parameters)] = rope_parameters
    return list(found.values())


def config_uses_mrope(model_config) -> bool:
    """True when a model config still carries an M-RoPE section (its attention layers build MRotaryEmbedding)."""
    return any(rope_parameters.get("mrope_section") for rope_parameters in _rope_parameter_dicts(model_config))


def align_draft_rope_with_target(target_model_config, draft_model_config) -> bool:
    """Give the MTP draft the rotary embedding the target runs with. Returns True when the draft was changed.

    With ``--language-model-only`` the engine strips the M-RoPE keys from the target's config, so its QSA
    layers build the plain ``RotaryEmbedding`` and the runner sends one position per token. The draft's
    config is built on its own (``SpeculativeConfig`` does not pass ``language_model_only`` on) and keeps
    the keys of a checkpoint with a vision tower: its QSA layer then built ``MRotaryEmbedding`` and read
    the runner's ``[rows, tokens]`` positions as the three M-RoPE rows, which fails on the first draft pass
    ("index 1 is out of bounds for dimension 0 with size 1"). The keys are removed from the draft here,
    before the draft model is built.

    A target that keeps M-RoPE (a server without ``--language-model-only``) is refused: the drafter gathers
    its positions per row and does not carry the three M-RoPE rows.
    """
    if config_uses_mrope(target_model_config):
        raise NotImplementedError("Flash-Next MTP on HPU runs with --language-model-only: the target keeps "
                                  "M-RoPE positions, which the drafter does not carry")
    changed = False
    for rope_parameters in _rope_parameter_dicts(draft_model_config):
        for key in MROPE_KEYS:
            if key in rope_parameters:
                rope_parameters.pop(key)
                changed = True
    return changed


def flat_token_rows(hidden: torch.Tensor) -> torch.Tensor:
    """``hidden`` as ``[T, width]``, one row per token.

    The runner hands the drafter ``[vbs, 1, width]`` after a decode step and ``[1, seq, width]`` after a
    prefill (one bucket row); ``Qwen4ExpMultiTokenPredictor.forward`` reads the token count from dim 0 and
    views the rows as ``[T, hc, H]``, so a prefill's ``[1, seq, width]`` failed there with "shape '[1, 4, H]'
    is invalid".
    """
    return hidden.reshape(-1, hidden.shape[-1])


def ring_slots_from_lattice(indices: torch.Tensor) -> torch.Tensor:
    """[groups, bs, T] -> [groups, bs] (column 0); [groups, bs] unchanged; ValueError otherwise."""
    if indices.dim() == 3:
        return indices[:, :, 0]
    if indices.dim() == 2:
        return indices
    raise ValueError(f"expected a [groups, bs, T] state lattice or [groups, bs] ring slots, "
                     f"got shape {tuple(indices.shape)}")


def next_step_hidden(multi: torch.Tensor, last_token_indices: torch.Tensor) -> torch.Tensor:
    """multi viewed as [vbs, hc*H], rows last_token_indices -> [bs, 1, hc*H]."""
    hidden_size = multi.shape[-1]
    rows = torch.index_select(multi.view(-1, hidden_size), 0, last_token_indices)  # no advanced indexing (rule 3)
    return rows.view(-1, 1, hidden_size)


DRAFT_RANK_LOG_ENV = "VLLM_FLASH_NEXT_DRAFT_RANK_LOG"
DRAFT_RANK_TOP = 4


def draft_rank_log_enabled() -> bool:
    """VLLM_FLASH_NEXT_DRAFT_RANK_LOG is "1" or "true" (case-insensitive, stripped)."""
    return os.environ.get(DRAFT_RANK_LOG_ENV, "0").strip().lower() in ("1", "true")


class DraftRankStats:
    """Histogram of the target token's rank among the drafter's top-4 candidates.

    The runner remembers the drafter's top-4 rows by request id right after the
    decode propose (``remember``), then after the next step's sampling hands
    over the batch's accepted streams (``observe``): for a request with ``k``
    drafts and ``L = len(ids)`` committed tokens (accepted drafts + the target's
    bonus), ``L - 1 < k`` means the draft at index ``j = L - 1`` was rejected
    and ``ids[-1]`` is the target's own token there; ``L == k + 1`` means every
    draft was accepted. Requests without a remembered row (prefill-only drafts,
    first steps) are ignored. Under greedy sampling ``rank1`` stays 0: a target
    token equal to the top-1 draft would have been accepted.
    """

    def __init__(self, log_every: int = 200) -> None:
        self.counts = [0] * (DRAFT_RANK_TOP + 1)  # counts[r]: rank r+1; counts[4]: beyond top-4
        self.all_accepted = 0
        self.log_every = log_every
        self.observations = 0
        self._rows: dict[str, torch.Tensor] = {}

    def remember(self, req_ids: list[str], top_ids: torch.Tensor) -> None:
        """Store top_ids row i (a [k, DRAFT_RANK_TOP] top-4 row, any device) under req_ids[i]."""
        if len(req_ids) != top_ids.shape[0]:
            raise ValueError(f"remember: {len(req_ids)} req_ids for {top_ids.shape[0]} top-4 rows")
        top_ids = top_ids.cpu()
        for i, req_id in enumerate(req_ids):
            self._rows[req_id] = top_ids[i]

    def observe(self, req_ids: list[str], sampled: list[list[int]]) -> None:
        """Count the target token's rank at the first rejected position of each request."""
        for req_id, ids in zip(req_ids, sampled):
            row = self._rows.pop(req_id, None)
            if row is None or not ids:
                continue
            k = row.shape[0]
            length = len(ids)
            if length - 1 < k:
                target = ids[-1]
                rank = DRAFT_RANK_TOP
                for position, token in enumerate(row[length - 1].tolist()):
                    if token == target:
                        rank = position
                        break
                self.counts[rank] += 1
            else:
                self.all_accepted += 1
            self.observations += 1
            if self.observations % self.log_every == 0:
                logger.info(self.summary_line())

    def summary_line(self) -> str:
        return ("Flash-Next draft rank at the first rejected position: "
                f"rank1={self.counts[0]} rank2={self.counts[1]} rank3={self.counts[2]} "
                f"rank4={self.counts[3]} beyond4={self.counts[4]} all_accepted={self.all_accepted}")

class MtpDraftLoop(torch.nn.Module):
    """The k draft passes of a decode propose in one forward (VLLM_HPU_MTP_DRAFT_GRAPH=1).

    ``HpuFlashNextMtpProposer.propose`` runs every draft pass as its own HPU-graph replay, the LM head and the arg
    max lazily between the replays, and builds each later pass's attention metadata on the host between them. At 32
    streams on TP8 the draft costs 16 to 18 ms of a 104 ms step with a device wait of only 0.7 to 2.8 ms at its end
    (card job 00178, the MTP step timer): the host issue is the cost. This module runs pass 0 over the verification
    rows, the LM head and the arg max of each request's last accepted row, then the passes 1 .. k-1 (one row per
    request, the previous pass's ids as int32, its multi-stream hidden, the next position), each with its own
    metadata, and returns the ``[batch, k]`` draft ids: wrapped in one HPU graph, one replay issues all of it.

    The calls and their order are those of the per-pass proposer (tests/unit_tests/worker/test_mtp_draft_graph.py):

    * ``pass_forward(input_ids=, positions=, hidden_states=, inputs_embeds=None, attn_metadata=, spec_step_idx=)``
      returns the draft's ``(sample, multi)``; the proposer gives the draft adapter's own forward (its metadata
      processing, RoPE tables and forward context per pass), not the adapter's graph-wrapped one;
    * ``draft_tokens(hidden [rows, H])`` returns the ids ``[rows]`` (the proposer's LM head and arg max);
    * ``step_metadata[j - 1]`` is pass j's attention metadata, built on the host before the call from the base
      positions + j (the proposer's ``prepare_attn_metadata``).

    The forward reads no device value back (inside a graph capture there is none to read). The pools a pass writes
    (the draft's KV cache, the QSA key ring and compressed keys) are written with ``index_copy_``, which the lazy
    graph orders against the next pass's reads; the module holds no submodule and no parameter.
    """

    def __init__(self, pass_forward, draft_tokens, num_speculative_tokens: int, max_model_len: int) -> None:
        super().__init__()
        if int(num_speculative_tokens) < 1:
            raise ValueError(f"the draft loop needs at least one draft token, got {num_speculative_tokens}")
        self._pass_forward = pass_forward
        self._draft_tokens = draft_tokens
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.max_model_len = int(max_model_len)

    def forward(self, target_token_ids: torch.Tensor, target_positions: torch.Tensor,
                target_hidden_states: torch.Tensor, last_token_indices: torch.Tensor, attn_metadata,
                step_metadata: tuple = (), mode_key: tuple = ()) -> torch.Tensor:
        # mode_key: the switchable flags the passes' code reads at capture (the vocabulary-parallel arg max): a
        # Python value, hashed by value in the HPU graph's key, so a switch captures its own graph.
        k = self.num_speculative_tokens
        if len(step_metadata) != k - 1:
            raise ValueError(f"the draft loop with k = {k} takes the metadata of {k - 1} later passes, "
                             f"got {len(step_metadata)}")
        batch_size = last_token_indices.shape[0]
        ret = self._pass_forward(input_ids=target_token_ids, positions=target_positions,
                                 hidden_states=target_hidden_states, inputs_embeds=None, attn_metadata=attn_metadata,
                                 spec_step_idx=0)
        last_hidden_states, hidden_states = split_draft_output(ret)
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        sample_hidden_states = torch.index_select(last_hidden_states, 0, last_token_indices)
        draft_token_ids = self._draft_tokens(sample_hidden_states)
        if k == 1:
            return draft_token_ids.view(-1, 1)

        positions = torch.index_select(target_positions.view(-1), 0, last_token_indices)
        hidden_states = next_step_hidden(hidden_states, last_token_indices)
        draft_token_ids_list = [draft_token_ids]
        step_metadata = self._with_ring_slots(attn_metadata, step_metadata)
        for token_index in range(k - 1):
            # argmax ids are int64: narrowed here for the next pass (an in-graph consumer, card rule 9)
            input_ids = draft_token_ids_list[-1].int()
            positions = positions + 1
            clamped_positions = torch.where(positions >= self.max_model_len, 0, positions)
            ret = self._pass_forward(input_ids=input_ids.view(-1, 1), positions=clamped_positions.view(-1, 1),
                                     hidden_states=hidden_states.view(-1, 1, hidden_states.shape[-1]),
                                     inputs_embeds=None, attn_metadata=step_metadata[token_index],
                                     spec_step_idx=token_index + 1)
            last_hidden_states, hidden_states = split_draft_output(ret)
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = hidden_states[:batch_size]
            draft_token_ids_list.append(self._draft_tokens(last_hidden_states[:batch_size]))
        return torch.stack(draft_token_ids_list, dim=1)

    @staticmethod
    def _with_ring_slots(attn_metadata, step_metadata: tuple) -> tuple:
        """The later passes' metadata with the QSA raw-key ring slots (the lattice's column 0 of pass 0's
        ``load_indices_tensor``) sliced here, inside the graph, where the host left them out: a view made outside
        the forward would be an unmatched graph input, copied by a launch of its own before the replay."""
        indices = getattr(attn_metadata, "load_indices_tensor", None)
        if indices is None:
            return step_metadata
        ring_slots = None
        filled = []
        for md in step_metadata:
            if dataclasses.is_dataclass(md) and getattr(md, "load_indices_tensor", None) is None:
                if ring_slots is None:
                    ring_slots = ring_slots_from_lattice(indices)
                md = copy.copy(md)
                md.load_indices_tensor = ring_slots
                md.store_indices_tensor = ring_slots
            filled.append(md)
        return tuple(filled)


def first_max_ids(logits: torch.Tensor) -> torch.Tensor:
    """int32 ``[rows]``: the first index of each row's maximum (what ``argmax`` returns) from an fp32 id table.

    No int64 id is made on the device and no arg-max index is narrowed (card rule 9: such ids read outside the graph
    that made them came back wrong on that graph's first run). A row without a maximum (NaN) gets the last id.
    """
    vocab = logits.shape[-1]
    ids = torch.arange(vocab, device=logits.device, dtype=torch.float32)
    is_max = logits == logits.amax(dim=-1, keepdim=True)
    first = torch.where(is_max, ids, torch.full_like(ids, float(vocab - 1))).amin(dim=-1)
    return first.to(torch.int32)


def chain_draft_rows(sampled: torch.Tensor, cu_num_draft_tokens: torch.Tensor):
    """The draft's input rows from the verification's result on the device (VLLM_HPU_MTP_DRAFT_CHAIN).

    ``sampled`` is the rejection sampler's ``[B, S]`` output (``S = 1 + max drafts``; accepted drafts, the bonus,
    then -1), ``cu_num_draft_tokens`` the step's cumulative draft counts ``[B]``. Returns ``(picked, last_row,
    accepted, bonus)``: the verification rows the draft's pass 0 reads (``[B * S]``, a row past a request's committed
    tokens reads row 0), each request's last committed row ``[B]``, its accepted drafts ``[B]`` and its bonus token
    ``[B]``, all int32. The same rows as ``HpuEagleProposer.prepare_inputs`` on the host lists and the runner's clamp
    of its -1 to row 0: a request without drafts (bucket padding, one token left) reads its first row only.
    """
    batch, stride = int(sampled.shape[0]), int(sampled.shape[1])
    device = sampled.device
    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int32)
    num_draft = cu - torch.cat((torch.zeros(1, dtype=torch.int32, device=device), cu[:-1]))
    committed = (sampled >= 0).to(torch.int32).sum(dim=1, dtype=torch.int32)
    committed = torch.where(num_draft > 0, committed, torch.ones_like(committed)).clamp(min=1)
    start = torch.arange(batch, dtype=torch.int32, device=device) * stride
    last_row = start + committed - 1
    cols = torch.arange(stride, dtype=torch.int32, device=device)[None, :]
    picked = torch.where(cols < committed[:, None], start[:, None] + cols, torch.zeros_like(cols)).reshape(-1)
    bonus = torch.gather(sampled.to(torch.int32), 1, (committed - 1).to(torch.int64).view(-1, 1)).view(-1)
    return picked, last_row, committed - 1, bonus


def draft_step_slots(table_full: torch.Tensor, positions: torch.Tensor, real: torch.Tensor, block_size: int,
                     max_model_len: int, pad_block_id: int, pad_slot_id: int):
    """A later draft pass's ``(block_table, slot_mapping)`` on the device: the values of engine-step's host builder
    ``flash_next_decode_metadata.draft_step_block_table_and_slots`` (VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1).

    ``table_full`` int32 ``[B, max_blocks]``: the step's requests' attention-group block table (rows past the
    requests anything); ``positions`` ``[B]`` the pass's positions (not clamped); ``real`` bool ``[B]``: the rows of
    real requests (the first ones). Returns the int32 table (a real row's first ``ceil((p + 1) / block_size)``
    blocks, then pad; other rows all pad) and the int64 slots ``[B, 1]`` (a real row's block at ``clamp(p) //
    block_size`` times ``block_size`` plus ``clamp(p) % block_size``, positions at or past ``max_model_len``
    clamped to 0; the i-th row past the real ones ``pad_slot_id + i % block_size``).
    """
    batch, max_blocks = int(table_full.shape[0]), int(table_full.shape[1])
    device = table_full.device
    positions = positions.reshape(-1).to(torch.int64)
    real = real.reshape(-1).to(torch.bool)
    clamped = torch.where(positions >= max_model_len, torch.zeros_like(positions), positions)
    num_blocks = torch.div(positions + block_size, block_size, rounding_mode="floor")
    cols = torch.arange(max_blocks, dtype=torch.int64, device=device)[None, :]
    keep = (cols < num_blocks[:, None]) & real[:, None]
    pad_block = torch.full_like(table_full, int(pad_block_id))
    table = torch.where(keep, table_full, pad_block).to(torch.int32)
    block_numbers = torch.div(clamped, block_size, rounding_mode="floor").clamp(max=max_blocks - 1).view(-1, 1)
    block_ids = torch.gather(table_full.to(torch.int64), 1, block_numbers)
    slots = block_ids * block_size + (clamped % block_size).view(-1, 1)
    pad_index = torch.arange(batch, dtype=torch.int64, device=device) - real.to(torch.int64).sum()
    pad_slots = (int(pad_slot_id) + torch.remainder(pad_index, block_size)).view(-1, 1)
    return table, torch.where(real.view(-1, 1), slots, pad_slots).to(torch.int64)


class MtpChainDrafts(NamedTuple):
    """What the chained draft leaves on the device (int32, the step's padded decode batch, input-batch order)."""

    accepted: torch.Tensor  # [B] accepted drafts per request (0 on rows without drafts)
    last_row: torch.Tensor  # [B] the last committed verification row, flattened [B * S]
    next_ids: torch.Tensor  # [B, 1 + k] the bonus token, then the next step's k drafts
    drafts: torch.Tensor  # [B, k] the next step's drafts
    sampled: torch.Tensor | None = None  # [B, S] the greedy verification's result when the chain computed it


@dataclasses.dataclass
class MtpChainOut:
    """The chained step's device outputs (``HPUModelRunner._mtp_chain_out``), the interface agreed with async
    scheduling's mode 3: all int32, the step's padded decode batch ``B`` in input-batch order (the decodes are the
    first rows), padded rows with ``accepted`` 0.

    * ``accepted`` ``[B]``: accepted drafts per request;
    * ``next_ids`` ``[B, 1 + k]``: the bonus token, then the next step's k drafts (the next verification's ids);
    * ``sampled`` ``[B, 1 + k]``: the rejection sampler's output (accepted drafts, the bonus, -1 after; its dtype);
    * ``last_row`` ``[B]``: each request's last committed verification row, flattened ``[B * (1 + k)]``;
    * ``num_decodes``, ``k``: the real requests and the drafts per request; ``drafts`` ``[B, k]``.
    """

    accepted: torch.Tensor
    next_ids: torch.Tensor
    sampled: torch.Tensor
    last_row: torch.Tensor
    num_decodes: int
    k: int
    drafts: torch.Tensor


class MtpDraftChain(torch.nn.Module):
    """The draft chained behind the verification (VLLM_HPU_MTP_DRAFT_CHAIN=1): no host round trip between them.

    From the verification's ``sampled`` result on the device (``chain_draft_rows``) it picks each request's
    committed rows and last row, gathers the draft's pass-0 inputs (tokens, positions, the target's multi-stream
    rows), builds the later passes' metadata from the positions (``draft_step_slots`` and
    ``make_step_metadata(slot_mapping, block_table, ring_slots)``), then runs ``MtpDraftLoop``'s passes. The
    runner issues it right after the verification, before it reads anything back. Wrapped in one HPU graph by the
    proposer; it holds no parameter.
    """

    def __init__(self, pass_forward, draft_tokens, num_speculative_tokens: int, max_model_len: int, block_size: int,
                 pad_block_id: int, pad_slot_id: int, make_step_metadata) -> None:
        super().__init__()
        self.__dict__["_loop"] = MtpDraftLoop(pass_forward, draft_tokens, num_speculative_tokens, max_model_len)
        self._make_step_metadata = make_step_metadata
        self.block_size = int(block_size)
        self.max_model_len = int(max_model_len)
        self.pad_block_id = int(pad_block_id)
        self.pad_slot_id = int(pad_slot_id)

    def forward(self, sampled: torch.Tensor | None, cu_num_draft_tokens: torch.Tensor, real: torch.Tensor,
                verify_positions: torch.Tensor, verify_hidden: tuple, attn_metadata, table_full: torch.Tensor,
                mode_key: tuple = (), verify_ids: torch.Tensor | None = None,
                draft_token_ids: torch.Tensor | None = None, target_logits_indices: torch.Tensor | None = None,
                bonus_logits_indices: torch.Tensor | None = None) -> MtpChainDrafts:
        """``sampled`` the verification's result (stage A), or ``None`` with ``verify_ids`` (the target's greedy id
        of every verification row, ``[rows]`` or ``[rows, 1]``, as the verify graph returns them) and the step's
        draft tokens and logits indices (stage B): then the greedy rejection
        (``greedy_rejection_sample_from_tokens``) runs here too and its result is returned as ``sampled``."""
        loop = self.__dict__["_loop"]
        k = loop.num_speculative_tokens
        computed = None
        if verify_ids is not None:
            from vllm_gaudi.v1.sample.hpu_greedy_verify import greedy_rejection_sample_from_tokens

            sampled = computed = greedy_rejection_sample_from_tokens(verify_ids.reshape(-1), draft_token_ids,
                                                                     target_logits_indices, bonus_logits_indices,
                                                                     cu_num_draft_tokens)
        picked, last_row, accepted, bonus = chain_draft_rows(sampled, cu_num_draft_tokens)
        rows = picked.to(torch.int64)
        target_token_ids = torch.index_select(sampled.reshape(-1, 1), 0, rows)
        target_positions = torch.index_select(verify_positions, 0, rows)
        target_hidden_states = torch.cat([torch.index_select(h, 0, rows) for h in verify_hidden], dim=-1)
        target_hidden_states = flat_token_rows(target_hidden_states)
        last_token_indices = last_row.to(torch.int64)
        step_metadata = ()
        if k > 1:
            ring_slots = None
            indices = getattr(attn_metadata, "load_indices_tensor", None)
            if indices is not None:
                ring_slots = ring_slots_from_lattice(indices)
            base = torch.index_select(target_positions.reshape(-1), 0, last_token_indices)
            metadata = []
            for step in range(1, k):
                table, slots = draft_step_slots(table_full, base + step, real, self.block_size, self.max_model_len,
                                                self.pad_block_id, self.pad_slot_id)
                metadata.append(self._make_step_metadata(slots, table, ring_slots))
            step_metadata = tuple(metadata)
        drafts = MtpDraftLoop.forward(loop, target_token_ids, target_positions, target_hidden_states,
                                      last_token_indices, attn_metadata, step_metadata).to(torch.int32)
        next_ids = torch.cat((bonus.view(-1, 1), drafts), dim=1)
        return MtpChainDrafts(accepted, last_row, next_ids, drafts, computed)


def draft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """The draft's token ids ``[rows]`` from its logits ``[rows, vocab]``.

    VLLM_HPU_MTP_DRAFT_IDS=0 (default): ``argmax`` (int64 ids; the draft loop narrows them with ``.int()`` in the
    same lazy graph). 1: int32 ids from an fp32 id table, the first index of the row maximum as ``argmax`` gives it,
    with no int64 id made on the device (card rule 9: ``torch.topk`` ids narrowed to int32 in the same lazy graph
    came back as every second id; for ``argmax`` that is not measured). A row without a maximum (NaN) gets the last
    id, never an id outside the vocabulary.
    """
    from vllm_gaudi import envs as gaudi_envs

    if int(gaudi_envs.VLLM_HPU_MTP_DRAFT_IDS) != 1:
        return logits.argmax(dim=-1)
    return first_max_ids(logits)
