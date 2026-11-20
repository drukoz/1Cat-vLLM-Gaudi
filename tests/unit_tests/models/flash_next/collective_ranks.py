# SPDX-License-Identifier: Apache-2.0
"""Rank function for the collective count and the collective-cut parity tests (gloo CPU ranks of ``spikes/cpu-tp``).

Plain module (no pytest import): ``run_ranks`` spawns children that re-import this module by its module
path, and a rank returns numpy arrays and plain Python values only (see ``tp_ranks.py``).

:func:`collective_rank` builds the 8-head tiny Flash-Next model sharded at ``tp`` with expert parallelism
through vLLM's real loader, on the plugin's MoE path (``moe_block.use_plugin_fast_path``), puts the plugin's
``HpuCommunicator`` on the tensor-parallel group (the gloo groups under it; ``habana_frameworks`` is the CPU
stub) and runs the phases of ``opts["phases"]`` with a recorder on every collective the communicator issues:

* ``prefill``: one prompt chunk, and the logits of its last row,
* ``decode``: one single-token decode step of ``opts["decode_rows"]`` rows, and their logits,
* ``verify``: one speculative verification step, ``1 + k`` rows for each of ``opts["verify_requests"]``
  requests, and the logits of every row,
* ``draft``: the ``k`` drafter forwards of one speculative step, each followed by its draft-token choice,
* ``chain``: a prefill and ``opts["chain_steps"]`` greedy single-token decode steps of one reference prompt,
  then one speculative step (``k`` drafts and their verification): the tokens every path picked and the
  bits of the hidden states, for the parity tests.

The target's token choices go through :func:`vllm_gaudi.distributed.vocab_parallel.top_tokens` (what the HPU
runner's greedy region calls) and the drafter's through ``HpuFlashNextMtpProposer._draft_tokens`` (what the
proposer calls), so ``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` changes them here as it does there.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("TMPDIR", "/tmp")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[4]
_HARNESS_DIR = _REPO / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import harness  # noqa: E402,F401  (the caller runs this module's function through harness.run_ranks)

COLLECTIVE_METHODS = ("all_reduce", "all_gather", "all_gather_rows", "reduce_scatter", "gather")
DEFAULT_PHASES = ("prefill", "decode", "verify", "draft")
PREFILL_TOKENS = 48
CHAIN_PROMPT_TOKENS = 33
SPEC_SLOT = 1
DRAFT_SLOT = 2


def _site(skip_files: tuple[str, ...]) -> tuple[str, str]:
    """``(call site, its caller)`` of a collective: the first two frames outside the distributed layer."""
    import traceback

    found = []
    for frame in reversed(traceback.extract_stack()[:-2]):
        name = frame.filename.replace("\\", "/")
        if any(part in name for part in skip_files):
            continue
        for marker in ("/vllm_gaudi/", "/vllm/", "/tests/"):
            if marker in name:
                name = marker.strip("/") + "/" + name.split(marker, 1)[1]
                break
        found.append(f"{name}:{frame.lineno} {frame.name}")
        if len(found) == 2:
            break
    while len(found) < 2:
        found.append("")
    return found[0], found[1]


class CollectiveRecorder:
    """Records every collective the tensor-parallel group's device communicator issues, with its call site.

    One entry per call: ``method``, the input ``shape`` and ``dtype``, the ``site`` (file, line and function
    that asked for it) and its ``caller``, the ``module`` whose forward was running (``""`` outside a model
    forward), ``nonzero`` (elements of this rank's input that are not zero) and ``digest`` (a hash of this
    rank's input bytes: equal on every rank exactly when the payload is replicated).
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.phase = ""
        self._modules: list[str] = []
        self._handles = []
        self._originals = {}
        self._cls = None

    # -- module stack ---------------------------------------------------------------------------
    def hook(self, model, prefix: str) -> None:
        names = {id(m): n for n, m in model.named_modules()}

        def pre(module, args, kwargs):
            self._modules.append(prefix + names.get(id(module), type(module).__name__))

        def post(module, args, kwargs, output):
            self._modules.pop()

        for module in model.modules():
            self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(module.register_forward_hook(post, with_kwargs=True, always_call=True))

    # -- communicator methods -------------------------------------------------------------------
    def install(self) -> None:
        import hashlib

        from vllm.distributed.parallel_state import get_tp_group

        comm = get_tp_group().device_communicator
        if comm is None:  # world size 1: no collective exists
            return
        self._cls = type(comm)
        recorder = self

        def make(name, original):

            def wrapper(comm_self, input_, *args, **kwargs):
                site, caller = _site(("/vllm/distributed/", "/vllm_gaudi/distributed/device_communicators/",
                                      "collective_ranks.py", "/torch/nn/modules/module.py"))
                data = input_.detach()
                as_bytes = (data.float() if data.is_floating_point() else data).contiguous().numpy().tobytes()
                recorder.entries.append({
                    "phase": recorder.phase,
                    "method": name,
                    "shape": tuple(input_.shape),
                    "dtype": str(input_.dtype).replace("torch.", ""),
                    "site": site,
                    "caller": caller,
                    "module": recorder._modules[-1] if recorder._modules else "",
                    "nonzero": int((data != 0).sum()),
                    "digest": hashlib.sha256(as_bytes).hexdigest()[:16],
                })
                return original(comm_self, input_, *args, **kwargs)

            return wrapper

        for name in COLLECTIVE_METHODS:
            original = getattr(self._cls, name, None)
            if original is not None:
                self._originals[name] = original
                setattr(self._cls, name, make(name, original))

    def remove(self) -> None:
        for name, original in self._originals.items():
            setattr(self._cls, name, original)
        for handle in self._handles:
            handle.remove()


def install_hpu_communicator() -> bool:
    """Put the plugin's ``HpuCommunicator`` on the tensor-parallel group of this rank. False at world size 1."""
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_gaudi.distributed.device_communicators.hpu_communicator import HpuCommunicator

    group = get_tp_group()
    if group.device_communicator is None:
        return False
    group.device_communicator = HpuCommunicator(cpu_group=group.cpu_group,
                                                device=group.device,
                                                device_group=group.device_group,
                                                unique_name=group.unique_name)
    return True


def collective_rank(rank: int, ckpt_dir: str, tp: int, opts: dict) -> dict:
    """See the module docstring. ``opts``: ``env`` (flags set before the model is built), ``k`` (drafted
    tokens, default 3), ``phases``, ``fp32`` (default False: bf16, as served), ``decode_rows`` (default 1),
    ``verify_requests`` (default 1), ``chain_steps`` (default 6), ``expert_parallel`` (default True).

    Returns ``{"entries": [recorder entries], "layout": {...}, "chain": {...} | None}``.
    """
    for name, value in (opts.get("env") or {}).items():
        os.environ[name] = str(value)
    k = int(opts.get("k", 3))
    fp32 = bool(opts.get("fp32", False))
    phases = tuple(opts.get("phases", DEFAULT_PHASES))
    decode_rows = int(opts.get("decode_rows", 1))
    verify_requests = int(opts.get("verify_requests", 1))

    from tests.unit_tests.models.flash_next import moe_block

    moe_block.use_plugin_fast_path()

    import json

    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    import vllm_gaudi.models.qwen4_exp as fn
    import vllm_gaudi.ops.hpu_vocab_parallel_embedding  # noqa: F401  the token embedding class of a card
    from tests.unit_tests.models.flash_next import cpu_model as cm
    from tests.unit_tests.models.flash_next.test_decode_parity import CachedSdpaAttention
    from vllm_gaudi.distributed import vocab_parallel
    from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention
    from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import next_step_hidden, split_draft_output
    from vllm_gaudi.v1.spec_decode.hpu_flash_next_mtp import HpuFlashNextMtpProposer

    fn.register()
    kwargs = dict(model=ckpt_dir, tokenizer=ckpt_dir, skip_tokenizer_init=True, dtype="bfloat16",
                  enable_prefix_caching=False, max_model_len=512, max_num_seqs=4, enforce_eager=True,
                  language_model_only=True, tensor_parallel_size=tp, distributed_executor_backend="mp",
                  speculative_config={"method": "mtp", "num_speculative_tokens": k})
    if opts.get("expert_parallel", True):
        kwargs["enable_expert_parallel"] = True
    vc = EngineArgs(**kwargs).create_engine_config()
    reference = json.loads((Path(ckpt_dir) / "reference.json").read_text(encoding="utf-8"))
    recorder = CollectiveRecorder()
    num_slots = 8 + k + verify_requests * (1 + k)
    out: dict = {"chain": None}

    def qsa_layers(m):
        return [x for x in m.modules() if isinstance(x, HpuQwen4ExpQSAAttention)]

    def stateless(m):
        cm.use_sdpa_attention(m)

    def cached(m):
        atts = []
        for layer in qsa_layers(m):
            layer.attn = CachedSdpaAttention(layer)
            atts.append(layer.attn)
        return atts

    def set_group_idx(m):
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        for module in m.modules():
            if isinstance(module, MambaBase):
                module.cache_group_idx = torch.tensor(0, dtype=torch.long)
            if isinstance(module, HpuQwen4ExpQSAAttention):
                module.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)

    with set_current_vllm_config(vc):
        hpu_comm = install_hpu_communicator()
        model = get_model(vllm_config=vc)
        if fp32:
            cm.to_fp32(model)
            # The MTP multi-stream buffer is a plain tensor attribute (model dtype), which to_fp32 does not reach;
            # the draft reads it on its first step.
            for module in model.modules():
                buffer = getattr(module, "_mtp_hidden_buffer", None)
                if isinstance(buffer, torch.Tensor):
                    module._mtp_hidden_buffer = buffer.float()
        cm.bind_cpu_states(model, num_slots, fp32, 512, k)
        stateless(model)
        set_group_idx(model)
        draft = cm.cpu_draft_model(vc, fp32=fp32, num_slots=8)
        set_group_idx(draft)
        eos = cm.eos_token_id(vc)
        inner = model.language_model.model
        # As the engine does for an MTP draft (llm_base_proposer._maybe_share_embeddings / _maybe_share_lm_head):
        # the draft looks tokens up in the target's embedding and projects through the target's LM head.
        draft_embed_tp_size = int(draft.model.embed_tokens.tp_size)
        del draft.model.embed_tokens
        draft.model.embed_tokens = inner.embed_tokens
        del draft.lm_head
        draft.lm_head = model.language_model.lm_head
        runner = inner.layers[0].mlp.experts
        out["layout"] = {
            "tp": tp,
            "hpu_communicator": hpu_comm,
            "layers": len(inner.layers),
            "hidden": int(inner.config.hidden_size),
            "embed_class": type(inner.embed_tokens).__name__,
            "embed_rows": int(inner.embed_tokens.weight.shape[0]),
            "embed_tp_size": int(inner.embed_tokens.tp_size),
            "draft_own_embed_tp_size": draft_embed_tp_size,
            "draft_fc_tp_size": int(draft.model.fc_embedding.tp_size),
            "draft_fc_rows": int(draft.model.fc_embedding.weight.shape[0]),
            "lm_head_rows": int(model.language_model.lm_head.weight.shape[0]),
            "ep_size": int(runner.moe_config.ep_size),
            "ep_rank": int(runner.moe_config.ep_rank),
            "local_experts": int(runner.routed_experts.local_num_experts),
            "shared_gate_up_rows": int(inner.layers[0].mlp.shared_expert.gate_up_proj.weight.shape[0]),
        }
        recorder.hook(model, "target.")
        recorder.hook(draft, "draft.")
        recorder.install()

        def target(ids, positions, metadata, ple):
            with torch.no_grad(), set_forward_context(metadata, vc):
                hidden, aux = model(input_ids=torch.tensor(ids), positions=positions, **ple)
            return hidden, aux[0].view(-1, aux[0].shape[-1])

        def tokens_of(owner, hidden):
            with torch.no_grad():
                return vocab_parallel.top_tokens(owner, hidden).reshape(-1).tolist()

        # The draft token choice of the HPU MTP proposer itself (its method only reads .model and the rank log).
        proposer = object.__new__(HpuFlashNextMtpProposer)
        proposer.model = draft
        proposer.draft_rank_stats = None

        def draft_tokens_of(hidden):
            with torch.no_grad():
                tokens, top_ids = proposer._draft_tokens(hidden)
            assert top_ids is None
            return [int(t) for t in tokens.reshape(-1).tolist()]

        def verify_metadata(position, requests, accepted):
            T = 1 + k
            slots = [SPEC_SLOT + r * T for r in range(requests)]
            md = cm.decode_metadata(slots[0], position, num_tokens=T, slots=slots)
            lattice = torch.tensor([[[s + j for j in range(T)] for s in slots]], dtype=torch.int64)
            md.load_indices_tensor = lattice
            md.store_indices_tensor = lattice.clone()
            md.lattice_store_differs = False
            md.num_accepted_tokens = torch.full((requests, ), accepted, dtype=torch.int32)
            return md

        def draft_steps(stream, position, multi_row, atts):
            """The k draft forwards of one step for one request; the drafted tokens."""
            cm.reset_cpu_states(draft)
            for att in atts:
                att.reset()
                att.slot = DRAFT_SLOT
            hid = multi_row.view(1, -1)
            drafts = []
            last = stream[-1]
            for step in range(k):
                md = cm.decode_metadata(DRAFT_SLOT, position + step, load_slot=DRAFT_SLOT, store_slot=DRAFT_SLOT)
                with torch.no_grad(), set_forward_context(md, vc):
                    ret = draft(input_ids=torch.tensor([last]),
                                positions=torch.tensor([position + step]),
                                hidden_states=hid,
                                spec_step_idx=step)
                sample, multi = split_draft_output(ret)
                last = draft_tokens_of(sample.view(-1, sample.shape[-1])[-1:])[0]
                drafts.append(last)
                hid = next_step_hidden(multi, torch.tensor([0])).view(1, -1)
            return drafts

        if "prefill" in phases:
            recorder.phase = "prefill"
            ids = [1000 + i for i in range(PREFILL_TOKENS)]
            hidden, _ = target(ids, torch.arange(PREFILL_TOKENS), cm.prefill_metadata(PREFILL_TOKENS, SPEC_SLOT),
                               cm.ple_kwargs(PREFILL_TOKENS, eos))
            tokens_of(model.language_model, hidden[-1:])
        if "decode" in phases:
            recorder.phase = "decode"
            rows = decode_rows
            md = cm.decode_metadata(SPEC_SLOT, PREFILL_TOKENS, slots=list(range(SPEC_SLOT, SPEC_SLOT + rows)))
            if rows == 1:
                md = cm.decode_metadata(SPEC_SLOT, PREFILL_TOKENS)
            ple = {"query_start_loc": torch.arange(rows + 1, dtype=torch.int32),
                   "ngram_context": torch.full((rows, 2), eos, dtype=torch.int32)}
            hidden, _ = target([100 + i for i in range(rows)], torch.full((rows, ), PREFILL_TOKENS), md, ple)
            tokens_of(model.language_model, hidden)
        multi_last = None
        if "verify" in phases:
            recorder.phase = "verify"
            T = 1 + k
            rows = verify_requests * T
            md = verify_metadata(PREFILL_TOKENS, verify_requests, 1)
            ple = {"query_start_loc": torch.arange(0, rows + 1, T, dtype=torch.int32),
                   "ngram_context": torch.full((verify_requests, 2), eos, dtype=torch.int32)}
            positions = torch.arange(PREFILL_TOKENS, PREFILL_TOKENS + T).repeat(verify_requests)
            hidden, multi = target([200 + i for i in range(rows)], positions, md, ple)
            tokens_of(model.language_model, hidden)
            multi_last = multi[0]
        if "draft" in phases:
            recorder.phase = "draft"
            if multi_last is None:
                multi_last = torch.zeros(inner.config.hc_count * inner.config.hidden_size,
                                         dtype=torch.float32 if fp32 else torch.bfloat16)
            draft_atts = cached(draft)
            draft_steps([300, 301, 302], PREFILL_TOKENS, multi_last, draft_atts)
            for layer in qsa_layers(draft):
                layer.attn = cm.SdpaAttention(layer)

        if "chain" in phases:
            recorder.phase = "chain"
            prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == CHAIN_PROMPT_TOKENS)
            ids = list(prompt["prompt_ids"])
            cm.reset_cpu_states(model)
            atts = cached(model)
            draft_atts = cached(draft)
            for att in atts:
                att.slot = SPEC_SLOT
            n = len(ids)
            hidden, multi = target(ids, torch.arange(n), cm.prefill_metadata(n, SPEC_SLOT), cm.ple_kwargs(n, eos))
            seq = ids + tokens_of(model.language_model, hidden[-1:])
            hiddens = [hidden[-1:].float().numpy()]
            for _ in range(int(opts.get("chain_steps", 6))):
                pos = len(seq) - 1
                md = cm.decode_metadata(SPEC_SLOT, pos)
                # the single-token path reads num_accepted_tokens only through the lattice; a plain step has none
                hidden, multi = target([seq[pos]], torch.tensor([pos]), md,
                                       cm.ple_kwargs(1, eos, context=(seq[pos - 2], seq[pos - 1])))
                hiddens.append(hidden.float().numpy())
                seq = seq + tokens_of(model.language_model, hidden)
            # one speculative step on top of the chain: k drafts from the last committed row, then their
            # verification in one (1 + k)-row forward on the lattice of slots SPEC_SLOT .. SPEC_SLOT + k
            pos = len(seq) - 1
            md = cm.decode_metadata(SPEC_SLOT, pos)
            hidden, multi = target([seq[pos]], torch.tensor([pos]), md,
                                   cm.ple_kwargs(1, eos, context=(seq[pos - 2], seq[pos - 1])))
            bonus = tokens_of(model.language_model, hidden)
            hiddens.append(hidden.float().numpy())
            drafts = draft_steps(seq[pos - 1:pos + 1] + bonus, pos + 1, multi[0], draft_atts)
            T = 1 + k
            md = verify_metadata(pos + 1, 1, 1)
            hidden, multi = target(bonus + drafts, torch.arange(pos + 1, pos + 1 + T), md,
                                   cm.ple_kwargs(T, eos, context=(seq[pos - 1], seq[pos])))
            verified = tokens_of(model.language_model, hidden)
            out["chain"] = {
                "prompt_tokens": n,
                "tokens": [int(t) for t in seq[n:]] + [int(t) for t in bonus],
                "drafts": [int(t) for t in drafts],
                "verified": [int(t) for t in verified],
                "hidden": np.concatenate(hiddens).astype(np.float32),
                "verify_hidden": hidden.float().numpy().astype(np.float32),
                "top_k": None,
            }
            top_k_rows = int(opts.get("top_k", 0))
            if top_k_rows:
                with torch.no_grad():
                    values, ids = vocab_parallel.top_k(model.language_model, hidden, top_k_rows)
                    # every returned id must hold the returned value in the gathered logits (an all-gather the
                    # recorder files under the phase "check", outside the chain's own collectives)
                    recorder.phase = "check"
                    full = model.language_model.compute_logits(hidden).to(torch.float32)
                    holds = bool((torch.gather(full, -1, ids.long()) == values).all())
                    reference = torch.topk(full, top_k_rows, dim=-1).values
                out["chain"]["top_k"] = {"values": values.float().numpy().astype(np.float32),
                                         "ids": ids.numpy().astype(np.int64),
                                         "ids_hold_values": holds,
                                         "gathered_values": reference.numpy().astype(np.float32)}
        recorder.remove()
    out["entries"] = recorder.entries
    return out
