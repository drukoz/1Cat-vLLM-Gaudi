# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU registration for the Qwen4Exp (Flash-Next) architectures.

Import order is load-bearing: ``vllm_gaudi.models.qwen3_5`` replaces
``QwenGatedDeltaNetAttention`` in the upstream GDN module and
``vllm_gaudi.models.qwen3_next`` patches ``Qwen3NextSparseMoeBlock.forward``,
both at their import time. Upstream ``vllm.models.qwen4_exp.nvidia.model``
binds those names by ``from ... import`` at its own import time, so the two
plugin modules must be imported before it. ``vllm.models.qwen4_exp.nvidia.mtp``
imports from ``.model``, so it follows the upstream model import too.
"""

import torch
import vllm_gaudi.models.qwen3_5 as _qwen3_5  # noqa: F401  (must precede the upstream model import)
import vllm_gaudi.models.qwen3_next as _qwen3_next  # noqa: F401  (must precede the upstream model import)
import vllm.models.qwen4_exp.nvidia.model as _upstream_model
import vllm.models.qwen4_exp.nvidia.mtp as _upstream_mtp  # must follow the upstream model import
import vllm.models.qwen4_exp.nvidia.ple_layer as _upstream_ple
from vllm_gaudi.models.qwen4_exp.ple_layer import (  # must follow the upstream model import
    HpuQwen4ExpNGramEmbedding,
    HpuQwen4ExpPLELayer,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.registry import ModelRegistry
from vllm.sequence import IntermediateTensors
from vllm_gaudi import envs
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual
from vllm_gaudi.models.qwen4_exp.mtp import mtp_aux_hidden
from vllm_gaudi.models.qwen4_exp.ple_ops import HOST_NGRAM_IDS_KWARG
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention

FLASH_NEXT_ARCHS = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")

MTP_ARCHS = ("Qwen4ExpMTP",)

logger = init_logger(__name__)


def _token_embedding(*args, **kwargs):
    """The token embedding of a Flash-Next model: vocabulary-sharded, or one full copy per rank.

    ``VLLM_HPU_REPLICATED_EMBEDDING=1`` builds it with ``disable_tp=True``: every rank loads all the rows and
    looks its tokens up alone, so the all-reduce that follows a sharded lookup (one rank holds the row, the
    others send zeros) does not exist. A model with tied embeddings keeps the sharded table: its LM head is
    this module, and the head must stay sharded over the vocabulary.
    """
    if int(envs.VLLM_HPU_REPLICATED_EMBEDDING) == 1 and "disable_tp" not in kwargs:
        text_config = get_current_vllm_config().model_config.hf_text_config
        if getattr(text_config, "tie_word_embeddings", False):
            logger.warning("VLLM_HPU_REPLICATED_EMBEDDING=1 is off: the model ties the LM head to the embedding")
        else:
            kwargs["disable_tp"] = True
    return VocabParallelEmbedding(*args, **kwargs)


def _mtp_column_linear(*args, **kwargs):
    """The MTP draft's ``fc_embedding`` / ``fc_hidden`` (hidden x hidden, ``gather_output=True``).

    Upstream shards their output columns and all-gathers them again on every draft step: two collectives per
    drafted token for two small GEMMs whose input and output are replicated anyway.
    ``VLLM_HPU_MTP_REPLICATED_FC=1`` builds them unsharded (``disable_tp=True``), so each rank computes the
    whole product and no collective follows. Only the linears built with ``gather_output=True`` change.
    """
    if int(envs.VLLM_HPU_MTP_REPLICATED_FC) == 1 and kwargs.get("gather_output") and "disable_tp" not in kwargs:
        kwargs["disable_tp"] = True
    return ColumnParallelLinear(*args, **kwargs)


class _MtpTargetOutputs:
    """Return ``(hidden, [multi])`` from a Flash-Next target forward under ``method == "mtp"``.

    ``multi`` is the pre-final-mixer multi-stream state ``Qwen4ExpModel._mtp_hidden_buffer[:T]``
    (viewed with ``hidden``'s leading shape). The tuple feeds the fork's aux-hidden channel
    (``use_aux_hidden_state_outputs`` in ``v1/worker/hpu_model_runner.py``), which the MTP
    drafter reads on step 0. Without an MTP speculative config, or when the upstream forward
    returns ``IntermediateTensors`` (a non-last PP rank), the return is unchanged.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        sc = vllm_config.speculative_config
        self._hpu_mtp_outputs = sc is not None and getattr(sc, "method", None) == "mtp"
        object.__setattr__(self, "_hpu_mtp_multi", None)
        if self._hpu_mtp_outputs:
            self._hpu_tap_mtp_multi()

    def _hpu_tap_mtp_multi(self) -> None:
        """Keep the multi-stream state the final mixer returns in a forward (``combine_and_mix``'s first output).

        The engine copies that tensor into ``_mtp_hidden_buffer[:T]`` and the aux-hidden output used to be a view
        of the buffer. On the card neither survives an HPU graph: the copy is an in-place write through a slice of
        a persistent tensor, and the view left the graph as a view of a tensor that is no graph output, so the
        first decode step of a speculative server failed in the drafter's gathers with ``Empty tensor optional``
        (job 00021, tiny model, TP8). The aux-hidden output is now the mixer's own output tensor, a real output
        of the forward's graph; on an HPU device the buffer is dropped, so the engine's copy does not run.
        """
        owner = getattr(self, "language_model", self)
        inner = getattr(owner, "model", None)
        mixer = getattr(inner, "hyper_connection_mixer", None)
        if mixer is None:
            return
        combine_and_mix = mixer.combine_and_mix
        holder = self

        def tapped_combine_and_mix(*args, **kwargs):
            out = combine_and_mix(*args, **kwargs)
            object.__setattr__(holder, "_hpu_mtp_multi", out[0])
            return out

        mixer.combine_and_mix = tapped_combine_and_mix
        buffer = getattr(inner, "_mtp_hidden_buffer", None)
        if buffer is not None and buffer.device.type == "hpu":
            inner._mtp_hidden_buffer = None

    def forward(self,
                input_ids: torch.Tensor | None,
                positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs: object) -> torch.Tensor | IntermediateTensors:
        # The engine widens embeddings to the HC streams with repeat(1, hc_count), which needs [T] ids / [T, H] embeds;
        # the HPU runner flattens [bs, seq] batches only when the flatten_input feature is on (RUK-3102).
        if (input_ids is not None and input_ids.dim() != 1) or (inputs_embeds is not None and inputs_embeds.dim() != 2):
            shape = tuple((input_ids if input_ids is not None else inputs_embeds).shape)
            raise ValueError(f"Flash-Next takes flat [num_tokens] input_ids (the flatten_input feature); got {shape}")
        # The runner schedules [-1] for an empty n-gram draft (always rejected); id 0 keeps ids in embedding range.
        if input_ids is not None:
            input_ids = input_ids.clamp(min=0)
        # VLLM_HPU_PLE_HOST_NGRAM_IDS=1: the runner sends the PLE n-gram ids it computed on the host (int32
        # [ple_layers, T, heads]). The engine hands ngram_context to every PLE layer unchanged and forwards no other
        # input, so the ready ids travel in that argument; HpuQwen4ExpPLELayer.forward tells them by their rank.
        ple_ngram_ids = kwargs.pop(HOST_NGRAM_IDS_KWARG, None)
        if ple_ngram_ids is not None:
            kwargs["ngram_context"] = ple_ngram_ids
        riders = []
        if inputs_embeds is None and intermediate_tensors is None and int(envs.VLLM_HPU_PLE_TABLE_RIDER) == 1:
            riders = self._ple_table_rider(input_ids, kwargs)
        try:
            hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        finally:
            for module, name in riders:  # nothing a forward did not consume may reach the next one
                module.__dict__.pop(name, None)
        if self._hpu_mtp_outputs and isinstance(hidden, torch.Tensor):
            multi = self._hpu_mtp_multi
            object.__setattr__(self, "_hpu_mtp_multi", None)
            if multi is not None:
                return hidden, [multi.reshape(*hidden.shape[:-1], -1)]
            return hidden, [mtp_aux_hidden(hidden, self.get_mtp_target_hidden_states())]
        return hidden

    def _ple_table_rider(self, input_ids: torch.Tensor | None, kwargs: dict) -> list:
        """``VLLM_HPU_PLE_TABLE_RIDER=1``: the token embedding and the PLE n-gram table rows in one all-reduce.

        Both lookups are vocabulary-parallel and need only the ids: each rank looks up the rows its shards own
        (zeros for the others), the two partial tensors travel as the columns of one all-reduce, and the summed
        parts are left for the embedding's and the PLE layer's next call (they return them instead of looking
        up and reducing). Every id has one owner, so each sum is that owner's row, bit for bit, and the step
        has one collective fewer. Needs the vocabulary-sharded token embedding (with
        ``VLLM_HPU_REPLICATED_EMBEDDING=1`` there is no embedding all-reduce to ride on: nothing changes).
        Returns the ``(module, attribute)`` pairs it set.
        """
        language_model = getattr(self, "language_model", None)
        text_model = getattr(language_model if language_model is not None else self, "model", None)
        embed = getattr(text_model, "embed_tokens", None)
        if input_ids is None or embed is None or not hasattr(embed, "local_rows") or int(embed.tp_size) <= 1:
            return []
        ple_layers = [layer.ple for layer in getattr(text_model, "layers", [])
                      if isinstance(getattr(layer, "ple", None), HpuQwen4ExpPLELayer)]
        query_start_loc, ngram_context = kwargs.get("query_start_loc"), kwargs.get("ngram_context")
        if not ple_layers or query_start_loc is None or ngram_context is None:
            return []
        from vllm.distributed import tensor_model_parallel_all_reduce

        token_rows = embed.local_rows(input_ids)
        tables = [ple.local_table_rows(ple.ngram_ids(input_ids, query_start_loc, ngram_context), token_rows.dtype)
                  for ple in ple_layers]
        summed = tensor_model_parallel_all_reduce(
            torch.cat([token_rows] + [rows.flatten(1) for rows, _ in tables], dim=-1))
        widths = [token_rows.shape[-1]] + [rows[0].numel() for rows, _ in tables]
        parts = [part.contiguous() for part in summed.split(widths, dim=-1)]
        embed._hpu_reduced_rows = parts[0]
        riders = [(embed, "_hpu_reduced_rows")]
        for ple, (rows, lookup_was_fp8), part in zip(ple_layers, tables, parts[1:]):
            ple._rider_rows = (part.view(rows.shape), lookup_was_fp8)
            riders.append((ple, "_rider_rows"))
        return riders


class HpuQwen4ExpForConditionalGeneration(_MtpTargetOutputs, _upstream_model.Qwen4ExpForConditionalGeneration):
    """HPU specialization of the Flash-Next multimodal model; the MTP target output tuple is added above."""
    # Sparse QSA decode gathers K/V through the per-request block_table (M2).
    requires_block_table = True


class HpuQwen4ExpForCausalLM(_MtpTargetOutputs, _upstream_model.Qwen4ExpForCausalLM):
    """HPU specialization of the Flash-Next text-only model; the MTP target output tuple is added above."""
    # Sparse QSA decode gathers K/V through the per-request block_table (M2).
    requires_block_table = True


def apply_flash_next_patches() -> None:
    """Make the fork's GDN/MoE/HyperConnection/QSA/PLE replacements effective on the Flash-Next model.

    The GDN and MoE replacements are installed by importing ``vllm_gaudi.models.qwen3_5``
    and ``vllm_gaudi.models.qwen3_next`` (see module docstring); this asserts
    the GDN binding reached the upstream model module, and rebinds the
    ``GatedResidual``, ``Qwen4ExpQSAAttention``, ``Qwen4ExpPLELayer`` and
    ``Qwen4ExpNGramEmbedding`` names the upstream decoder layer builds through at
    construction time.
    """
    _upstream_model.GatedResidual = HpuGatedResidual
    _upstream_model.Qwen4ExpQSAAttention = HpuQwen4ExpQSAAttention
    if _upstream_model.QwenGatedDeltaNetAttention is not _qwen3_5.HPUGatedDeltaNetAttention:
        raise RuntimeError("Flash-Next GDN replacement did not take: vllm_gaudi.models.qwen3_5 and "
                           "vllm_gaudi.models.qwen3_next must be imported before "
                           "vllm.models.qwen4_exp.nvidia.model")
    _upstream_model.Qwen4ExpPLELayer = HpuQwen4ExpPLELayer
    # The upstream Qwen4ExpPLELayer.__init__ looks this name up when it builds the embedding.
    _upstream_ple.Qwen4ExpNGramEmbedding = HpuQwen4ExpNGramEmbedding
    # The MTP draft's final mixer is built in nvidia/mtp.py through its own GatedResidual import.
    _upstream_mtp.GatedResidual = HpuGatedResidual
    # Collective cuts (docs/design/flash-next/collectives.md): the constructors read their flag when a model is
    # built. The draft's own embedding is replaced by the target's when the engine shares it; when it is not
    # shared it follows the same flag.
    _upstream_model.VocabParallelEmbedding = _token_embedding
    _upstream_mtp.VocabParallelEmbedding = _token_embedding
    _upstream_mtp.ColumnParallelLinear = _mtp_column_linear


def register() -> None:
    apply_flash_next_patches()
    for arch in FLASH_NEXT_ARCHS:
        ModelRegistry.register_model(arch, f"vllm_gaudi.models.qwen4_exp:Hpu{arch}")
    for arch in MTP_ARCHS:
        ModelRegistry.register_model(arch, f"vllm_gaudi.models.qwen4_exp.mtp:Hpu{arch}")
