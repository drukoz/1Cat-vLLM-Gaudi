# SPDX-License-Identifier: Apache-2.0
"""The real vLLM engine with the HPU plugin, on CPU: a rehearsal of a card server on this machine.

``python -m tests.unit_tests.hpu_cpu_engine --ckpt DIR --out FILE`` builds ``vllm.LLM`` on a tiny Flash-Next
checkpoint in one process (TP1, the in-process engine core) with the HPU platform, the HPU worker and the HPU
model runner, generates the prompts and writes what happened to ``FILE``. Everything between the HTTP layer and
the device kernels is the code a card runs: the scheduler and its KV cache manager, ``HPUWorker``,
``HPUModelRunner`` (cache groups, block tables, the decode and verification batches, the state lattices, the
sampler, the drafter call sites), ``HpuFlashNextMtpProposer`` and the model classes with the plugin's ops.
The CPU tests of the model (``models/flash_next/cpu_model.py``) call the layers directly and never ran that
code; the first engine start with MTP on this machine stopped five times before the first draft
(``models/flash_next/test_engine_mtp.py`` keeps those cases).

What stands in for the Gaudi runtime, and what that means for a result:

* ``VLLM_USE_FAKE_HPU=1``: the plugin's own switch (device ``cpu``, gloo). Code that names ``"hpu"`` directly
  is sent to the CPU by :func:`install_cpu_device`.
* ``habana_frameworks`` is ``hpu_cpu_stubs``'s mock with real functions where a value is used: lazy mode is
  reported on (as on a card), ``wrap_in_hpu_graph`` returns the module with ``bypass_hpu_graphs`` swallowed,
  ``FusedRMSNorm`` and ``apply_rotary_pos_emb`` are torch formulas, ``torch.ops.hpu.mixture_of_experts`` and
  ``cast_to_fp8_v2`` are the stand-ins of ``models/flash_next/moe_block.py``. The fused GDN and
  hyper-connection ops run their Python reference kernels. So arithmetic is CPU arithmetic: a stream is
  compared with the stream of the same engine without speculation, never with a card.
* The HPU kernels do not bounds-check an index; the base path sends ``-1`` positions and slots of padding rows
  through them and discards the result. CPU kernels raise. :class:`IndexGuard` wraps negative indices, clamps
  large ones, runs the op and records the place: the record lists every op whose result on a card depends on
  what the card does with such an index.
* One rank. Collectives, the recipe cache, graph compilation and device memory are not rehearsed.

Run it in a process of its own (it replaces torch functions and the platform for the whole process); the
tests start it with :func:`run`.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import functools
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# What a rental server gets (tools/flash_next/rental/serve.py RENTAL_ENV_DEFAULTS), plus the fake device.
ENGINE_ENV = {
    "VLLM_USE_FAKE_HPU": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_PLUGINS": "",
    "TMPDIR": "/tmp",
    "VLLM_HPU_FLASH_NEXT_QSA_MODE": "1",
    "VLLM_HPU_RECURRENT_STATE_POOL": "1",
    "PT_HPU_LAZY_MODE": "1",
    "VLLM_SKIP_WARMUP": "true",
}

_FACTORIES = ("tensor", "zeros", "ones", "full", "empty", "arange", "zeros_like", "ones_like", "full_like",
              "empty_like", "as_tensor", "randn", "rand", "randint", "eye", "linspace", "scalar_tensor")
_NEW_FACTORIES = ("new_zeros", "new_ones", "new_full", "new_empty", "new_tensor")


def _is_hpu(device) -> bool:
    import torch

    if isinstance(device, str):
        return device.split(":")[0] == "hpu"
    return isinstance(device, torch.device) and device.type == "hpu"


def install_cpu_device() -> None:
    """Send every request for device ``"hpu"`` to the CPU (tensor factories, ``Tensor.to``, ``Module.to``)."""
    import torch

    if getattr(torch, "_hpu_cpu_engine_device", False):
        return
    torch._hpu_cpu_engine_device = True

    def redirect(real):

        def call(*args, **kwargs):
            args = tuple("cpu" if _is_hpu(arg) else arg for arg in args)
            if _is_hpu(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return real(*args, **kwargs)

        functools.update_wrapper(call, real, assigned=("__name__", "__doc__"), updated=())
        return call

    for name in _FACTORIES:
        setattr(torch, name, redirect(getattr(torch, name)))
    for name in _NEW_FACTORIES:
        setattr(torch.Tensor, name, redirect(getattr(torch.Tensor, name)))
    torch.Tensor.to = redirect(torch.Tensor.to)
    torch.nn.Module.to = redirect(torch.nn.Module.to)


class RotaryPosEmbeddingMode(enum.Enum):
    BLOCKWISE = 0
    PAIRWISE = 1


def apply_rotary_pos_emb(x, cos, sin, position_ids, offset, mode):
    """The vendor RoPE kernel as ``HPURotaryEmbedding`` calls it: ``cos``/``sin`` already as wide as ``x``."""
    import torch

    assert position_ids is None and offset == 0
    if mode is RotaryPosEmbeddingMode.BLOCKWISE:
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    else:
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class FusedRMSNorm:
    """The vendor RMS norm: ``x * rsqrt(mean(x^2) + eps) * weight`` over the last dim, computed in fp32."""

    @staticmethod
    def apply(x, weight, eps):
        import torch

        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _wrap_in_hpu_graph(module, **_kwargs):
    """``wrap_in_hpu_graph``: the same module, its forward taking (and dropping) ``bypass_hpu_graphs``."""
    forward = module.forward

    def graph_forward(*args, **kwargs):
        kwargs.pop("bypass_hpu_graphs", None)
        return forward(*args, **kwargs)

    module.forward = graph_forward
    return module


def install() -> None:
    """Put the stand-ins in place. Call it before the first ``vllm`` or ``vllm_gaudi`` import."""
    for key, value in ENGINE_ENV.items():
        os.environ.setdefault(key, value)
    from tests.unit_tests import hpu_cpu_stubs  # noqa: F401

    import psutil
    import torch

    install_cpu_device()
    torch.hpu.current_device = lambda: torch.device("cpu")
    torch.hpu.is_available = lambda: False
    torch.hpu.synchronize = lambda *args, **kwargs: None
    torch.hpu.device_count = lambda: 1
    if not hasattr(psutil.Process, "cpu_affinity"):  # macOS; the runner's load_model reads it
        psutil.Process.cpu_affinity = lambda self, *args: list(range(os.cpu_count() or 1))

    # Import the submodules before the plugin does: the import system binds each on its parent, so
    # `htorch.utils.internal.is_lazy()` and `from habana_frameworks.torch.utils.internal import is_lazy` agree.
    import habana_frameworks.torch.hpex.kernels as kernels
    import habana_frameworks.torch.hpex.normalization as normalization
    import habana_frameworks.torch.hpu as hpu
    import habana_frameworks.torch.utils.internal as internal

    internal.is_lazy = lambda: os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    hpu.wrap_in_hpu_graph = _wrap_in_hpu_graph
    kernels.RotaryPosEmbeddingMode = RotaryPosEmbeddingMode
    kernels.apply_rotary_pos_emb = apply_rotary_pos_emb
    normalization.FusedRMSNorm = FusedRMSNorm

    import torch._dynamo

    if "hpu_backend" not in torch._dynamo.list_backends(exclude_tags=None):
        # vLLM decorates functions with torch.compile(backend=<the platform's backend>) at import.
        torch._dynamo.register_backend(name="hpu_backend")(lambda gm, example_inputs, **kwargs: gm.forward)

    import vllm.platforms
    from vllm_gaudi.platform import HpuPlatform

    vllm.platforms._current_platform = HpuPlatform()

    import vllm_gaudi

    vllm_gaudi.register()  # the platform plugin's entry point: compile settings and patches.apply()
    vllm_gaudi.register_utils()
    vllm_gaudi.register_ops()
    # register_models() imports every model family, and the CPU environment's transformers lacks some.
    import vllm_gaudi.models.qwen4_exp as flash_next

    flash_next.register()

    from tests.unit_tests.models.flash_next import moe_block

    moe_block.install_vendor_ops()
    moe_block._keep_checkpoint_weight_layout()


_DIM_INDEX = {  # op -> (position of dim, position of index)
    "aten::index_select": (1, 2),
    "aten::index_copy_": (1, 2),
    "aten::index_copy": (1, 2),
    "aten::index_add_": (1, 2),
    "aten::index_fill_": (1, 2),
    "aten::gather": (1, 2),
    "aten::scatter_": (1, 2),
    "aten::scatter": (1, 2),
    "aten::scatter_add_": (1, 2),
}
_INDEX_LIST = ("aten::index", "aten::index_put_", "aten::index_put", "aten::_unsafe_index")
_SOURCE_ROOTS = ("/vllm_gaudi/", "/flashinfer_gaudi/", "/vllm/")


def _frames(depth: int = 3) -> str:
    frames = [frame for frame in traceback.extract_stack()
              if any(root in frame.filename for root in _SOURCE_ROOTS) and "hpu_cpu_engine" not in frame.filename]
    names = []
    for frame in reversed(frames[-depth:]):
        name = frame.filename
        for root in _SOURCE_ROOTS:
            if root in name:
                name = root.strip("/") + "/" + name.split(root, 1)[1]
                break
        names.append(f"{name}:{frame.lineno}")
    return " <- ".join(names)


def make_index_guard():
    """An :class:`IndexGuard` (built here: ``torch`` is imported after :func:`install` in a rehearsal process)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class IndexGuard(TorchDispatchMode):
        """Wraps negative indices and clamps large ones before the op runs, and records where it had to."""

        def __init__(self) -> None:
            super().__init__()
            self.records: Counter = Counter()  # (op, kind, where) -> count
            self.examples: dict = {}  # the first out-of-range index range at each key
            self._busy = False

        def _fix(self, op: str, index, size: int):
            if index.dtype == torch.bool or index.numel() == 0:
                return index
            lo, hi = int(index.min()), int(index.max())
            if lo >= 0 and hi < size:
                return index
            kind = "+".join(word for word, hit in (("negative", lo < 0), ("too large", hi >= size)) if hit)
            key = (op, kind, _frames())
            self.records[key] += 1
            self.examples.setdefault(key, f"[{lo}, {hi}] of {size}")
            return torch.where(index < 0, index + size, index).clamp(0, size - 1)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if self._busy:
                return func(*args, **kwargs)
            name = func._schema.name
            self._busy = True
            try:
                if name in _DIM_INDEX and len(args) > 2:
                    dim_pos, index_pos = _DIM_INDEX[name]
                    dim, index = args[dim_pos], args[index_pos]
                    if isinstance(index, torch.Tensor) and isinstance(dim, int):
                        fixed = self._fix(name, index, args[0].shape[dim] if args[0].dim() else 1)
                        if fixed is not index:
                            args = list(args)
                            args[index_pos] = fixed
                elif name == "aten::embedding" and len(args) > 1:
                    fixed = self._fix(name, args[1], args[0].shape[0])
                    if fixed is not args[1]:
                        args = list(args)
                        args[1] = fixed
                elif name in _INDEX_LIST and len(args) > 1:
                    indices = list(args[1])
                    changed = False
                    for axis, index in enumerate(indices):
                        if isinstance(index, torch.Tensor) and index.dtype != torch.bool:
                            fixed = self._fix(name, index, args[0].shape[axis])
                            changed |= fixed is not index
                            indices[axis] = fixed
                    if changed:
                        args = list(args)
                        args[1] = indices
            finally:
                self._busy = False
            return func(*args, **kwargs)

        def events(self) -> list[dict]:
            return [{"op": op, "kind": kind, "where": where, "count": count, "example": self.examples[(op, kind, where)]}
                    for (op, kind, where), count in sorted(self.records.items())]

    return IndexGuard()


# Ops the card rules name (BRIEF.md, "Rules the card taught us", rule 3): no HPU kernel, they fail at warmup.
NO_HPU_KERNEL_OPS = ("aten::cummax", "aten::cummin", "aten::index_put", "aten::index_put_", "aten::_index_put_impl_")


def make_op_audit():
    """An :class:`OpAudit`: every dispatched op with the dtypes of its tensor arguments, counted with its first
    site; for an op outside ``aten`` and ``prim`` (a custom op: the fused GDN, hyper-connection and MoE ops) every
    tensor argument that is a view (card rule 1) or int64 (card rule 2)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class OpAudit(TorchDispatchMode):

        def __init__(self) -> None:
            super().__init__()
            self.ops: Counter = Counter()
            self.sites: dict = {}
            self.custom: Counter = Counter()  # (op, finding, where) -> count

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func._schema.name
            tensors = [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)]
            key = (name, ",".join(sorted({str(t.dtype).replace("torch.", "") for t in tensors})))
            self.ops[key] += 1
            if key not in self.sites:
                self.sites[key] = _frames(4)
            if name.split("::")[0] not in ("aten", "prim"):
                for t in tensors:
                    for finding, hit in (("view", t._is_view()), ("int64", t.dtype == torch.int64)):
                        if hit:
                            self.custom[(name, finding, _frames(3))] += 1
            return func(*args, **kwargs)

        def record(self) -> dict:
            return {"ops": [{"op": op, "dtypes": dtypes, "count": count, "where": self.sites[(op, dtypes)]}
                            for (op, dtypes), count in sorted(self.ops.items())],
                    "custom_findings": [{"op": op, "finding": finding, "where": where, "count": count}
                                        for (op, finding, where), count in sorted(self.custom.items())]}

    return OpAudit()


def find_runner(llm):
    """The ``HPUModelRunner`` of an in-process ``vllm.LLM``."""
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    worker = core.model_executor.driver_worker
    worker = getattr(worker, "worker", worker)
    return worker.model_runner


def runner_facts(runner) -> dict:
    """The block sizes and cache groups the engine and the runner agreed on."""
    cache_config = runner.cache_config
    groups = []
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        groups.append({"gid": gid,
                       "spec": type(spec).__name__,
                       "block_size": int(spec.block_size),
                       "num_speculative_blocks": getattr(spec, "num_speculative_blocks", None),
                       "layers": list(group.layer_names),
                       "block_table_shape": list(runner.input_batch.block_table[gid].get_cpu_tensor().shape)})
    return {"block_size": int(runner.block_size),
            "attn_block_size": int(runner.attn_block_size),
            "scheduler_block_size": int(cache_config.block_size),
            "mamba_block_size": getattr(cache_config, "mamba_block_size", None),
            "mamba_cache_mode": getattr(cache_config, "mamba_cache_mode", None),
            "prefix_caching": bool(cache_config.enable_prefix_caching),
            "recurrent_pool_slots": int(runner._recurrent_pool_slots),
            "candidate_verify_steps": int(getattr(runner, "_candidate_verify_steps", 0)),
            # VLLM_HPU_MTP_DRAFT_GRAPH=1: decode drafts whose k passes ran in one HPU graph (MtpDraftLoop)
            "draft_graph_calls": int(getattr(getattr(runner, "drafter", None), "draft_graph_calls", 0)),
            "attention_group": runner._get_attention_group_id_for_hybrid(),
            "num_blocks": int(runner.kv_cache_config.num_blocks),
            "groups": groups}


def _record_steps(runner) -> list[dict]:
    """Record every decode batch the runner builds: rows per request and the accepted counts it loads from."""
    steps: list[dict] = []
    create = runner._create_decode_input_data

    def recording(num_decodes, num_scheduled_tokens, *args, **kwargs):
        data = create(num_decodes, num_scheduled_tokens, *args, **kwargs)
        accepted = getattr(data.attn_metadata, "num_accepted_tokens", None)
        steps.append({"rows": [int(n) for n in num_scheduled_tokens[:num_decodes]],
                      "num_accepted": None if accepted is None else [int(a) for a in accepted[:num_decodes]]})
        return data

    runner._create_decode_input_data = recording
    return steps


def _record_committed(runner) -> list[list[int]]:
    """Record the committed token count of every request of every verification step."""
    committed: list[list[int]] = []
    sampler = getattr(runner, "rejection_sampler", None)
    if sampler is None:
        return committed
    parse = sampler.parse_output

    def recording(*args, **kwargs):
        out = parse(*args, **kwargs)
        rows = out[0] if isinstance(out, tuple) else out
        committed.append([len(row) for row in rows])
        return out

    sampler.parse_output = recording
    return committed


def _record_draft_calls(runner) -> list[dict]:
    """Record, for every ``drafter.propose`` call, whether the block table it got is the attention group's."""
    calls: list[dict] = []
    drafter = getattr(runner, "drafter", None)
    if drafter is None or not hasattr(drafter, "propose"):
        return calls
    propose = drafter.propose

    def recording(target_token_ids, target_positions, target_hidden_states, last_token_indices,
                  common_attn_metadata, block_table_cpu_tensor, model_runner, *args, **kwargs):
        attn_table = runner.input_batch.block_table[runner._get_attention_group_id_for_hybrid()].get_cpu_tensor()
        rows = int(block_table_cpu_tensor.shape[0])
        calls.append({"rows": rows,
                      "decode": not bool(getattr(common_attn_metadata, "is_prompt", True)),
                      "width": int(block_table_cpu_tensor.shape[1]),
                      "attention_width": int(attn_table.shape[1]),
                      "is_attention_table": bool(block_table_cpu_tensor.shape[1] == attn_table.shape[1]
                                                 and block_table_cpu_tensor.data_ptr() >= attn_table.data_ptr()
                                                 and block_table_cpu_tensor.data_ptr()
                                                 < attn_table.data_ptr() + attn_table.numel() *
                                                 attn_table.element_size())})
        graphs_before = int(getattr(drafter, "draft_graph_calls", 0))
        out = propose(target_token_ids, target_positions, target_hidden_states, last_token_indices,
                      common_attn_metadata, block_table_cpu_tensor, model_runner, *args, **kwargs)
        # VLLM_HPU_MTP_DRAFT_GRAPH=1: this draft's k passes ran in one graph (MtpDraftLoop)
        calls[-1]["graph"] = int(getattr(drafter, "draft_graph_calls", 0)) > graphs_before
        return out

    drafter.propose = recording
    fill = getattr(drafter, "fill_prefill_chunk", None)
    if fill is not None:

        def recording_fill(token_ids, position_ids, target_hidden_states, attn_metadata, last_index, next_token_id):
            real = position_ids.reshape(-1)[:int(last_index) + 1]
            calls.append({"fill": True, "first_position": int(real.min()), "last_position": int(real.max()),
                          "next_token_id": int(next_token_id)})
            return fill(token_ids, position_ids, target_hidden_states, attn_metadata, last_index, next_token_id)

        drafter.fill_prefill_chunk = recording_fill
    return calls


def probe_recurrent_pools(runner) -> dict:
    """``--slice-write-probe``: the pools the align-mode state copies write (``runner._recurrent_commit_registry``,
    the GDN conv and ssm pools) become ``SliceWriteProbe`` tensors over the same storage, so a write through a slice
    of a pool is recorded (card job 00133); the copies that ran are counted (``commit_conv`` / ``commit_ssm`` pairs,
    the boundary commit after sampling and the pre-step copy before the forward)."""
    import vllm_gaudi.v1.worker.hpu_model_runner as runner_module
    import vllm_gaudi.v1.worker.recurrent_commit as commit_module
    from tests.unit_tests.models.flash_next.slice_write_probe import SliceWriteProbe

    SliceWriteProbe.violations = []
    record = {"conv_pairs": 0, "ssm_pairs": 0, "violations": SliceWriteProbe.violations}
    registry = getattr(runner, "_recurrent_commit_registry", None) or {}
    for gid, entries in list(registry.items()):
        registry[gid] = [(tensor.as_subclass(SliceWriteProbe), kind, window) for tensor, kind, window in entries]
    record["pools"] = sum(len(entries) for entries in registry.values())

    def counting(key, function):

        def wrapper(state, pairs, *args, **kwargs):
            record[key] += len(pairs)
            return function(state, pairs, *args, **kwargs)

        return wrapper

    conv, ssm = commit_module.commit_conv, commit_module.commit_ssm
    for module in (runner_module, commit_module):
        module.commit_conv = counting("conv_pairs", conv)
        module.commit_ssm = counting("ssm_pairs", ssm)
    return record


def install_oracle_drafts(llm, runner, oracle: dict, k: int) -> list[dict]:
    """Replace the drafter's tokens with the reference stream, so the verification accepts a chosen count.

    The drafter still runs (its state and its caches are written as on a card); only the token ids it returns
    are replaced, in ``take_draft_token_ids``, after the scheduler has counted this step's tokens. For the
    request's ``n``-th verification step the drafts are the next ``k`` tokens of its reference stream with the
    token at position ``accept[n % len(accept)]`` changed (none when that value is ``k``): a greedy verification
    must accept exactly that many drafts, and the generated stream must stay the reference stream. Returns the
    record of every replacement: request, tokens generated so far, the drafter's own tokens, the oracle's.
    """
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    scheduler = core.scheduler
    streams = {tuple(s["prompt_ids"]): list(s["generated_ids"]) for s in oracle["streams"]}
    accept = [int(a) for a in oracle["accept"]]
    vocab = int(oracle.get("vocab_size", 248320))
    seen: Counter = Counter()
    record: list[dict] = []
    take = runner.take_draft_token_ids

    def oracle_take():
        out = take()
        if out is None:
            return out
        drafts = [list(row) for row in out.draft_token_ids]
        for i, req_id in enumerate(out.req_ids):
            request = scheduler.requests.get(req_id)
            if request is None or i >= len(drafts):
                continue
            stream = streams.get(tuple(request.prompt_token_ids))
            if stream is None:
                continue
            n_out = int(request.num_output_tokens)
            if n_out == 0:  # a prefill chunk that completed no prompt: the scheduler drops its drafts
                continue
            step = seen[req_id]
            seen[req_id] += 1
            want = accept[step % len(accept)]
            new = [stream[n_out + j] if n_out + j < len(stream) else 0 for j in range(k)]
            if want < k:
                new[want] = (new[want] + 1) % vocab
            record.append({"req": req_id, "prompt_len": len(request.prompt_token_ids), "n_out": n_out,
                           "step": step, "accept": want, "drafter": [int(t) for t in drafts[i][:k]],
                           "oracle": new})
            drafts[i] = new
        return type(out)(out.req_ids, drafts)

    runner.take_draft_token_ids = oracle_take
    worker = core.model_executor.driver_worker
    worker = getattr(worker, "worker", worker)
    if hasattr(worker, "take_draft_token_ids"):
        worker.take_draft_token_ids = oracle_take
    return record


def write_switch_file(lines: str) -> None:
    """Replace the runtime switch file named by VLLM_HPU_RUNTIME_SWITCH_FILE with ``lines`` (NAME=VALUE pairs
    separated by ';'), with a rename, as an operator's probe does between two rounds."""
    path = os.environ.get("VLLM_HPU_RUNTIME_SWITCH_FILE", "").strip()
    if not path:
        raise SystemExit("--switch-after needs VLLM_HPU_RUNTIME_SWITCH_FILE in the environment")
    staging = path + ".new"
    with open(staging, "w", encoding="utf-8") as handle:
        handle.write("".join(pair.strip() + "\n" for pair in lines.split(";") if pair.strip()))
    os.replace(staging, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.unit_tests.hpu_cpu_engine", description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True, help="a tiny Flash-Next checkpoint folder (with reference.json)")
    parser.add_argument("--out", required=True, help="the JSON record of the run")
    parser.add_argument("--k", type=int, default=0, help="num_speculative_tokens of the MTP drafter (0: none)")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--prompts", default=None,
                        help='JSON {"prompts": [{"prompt_ids": [...]}]} (default: the checkpoint\'s reference.json)')
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="the chunk budget of a step")
    parser.add_argument("--concurrent", action="store_true", help="send every prompt in one generate call")
    parser.add_argument("--oracle", default=None,
                        help='JSON {"streams": [{"prompt_ids", "generated_ids"}], "accept": [a0, a1, ...]}: the '
                        "drafts become the reference stream with the token after a_n accepted ones changed "
                        "(install_oracle_drafts)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0, help="top_k of a sampled run (0: none)")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p of a sampled run")
    parser.add_argument("--seed", type=int, default=None, help="per-request seed of a sampled run")
    parser.add_argument("--logprobs", type=int, default=None, help="record the top-N logprobs of every generated token")
    parser.add_argument("--next-turn", default=None, metavar="IDS_JSON",
                        help="after the prompts, one more request: the first prompt, its generated tokens and these "
                        "token ids (a JSON list): the agent's next turn on this engine's own answer")
    parser.add_argument("--slice-write-probe", action="store_true",
                        help="record writes through a slice of the recurrent pools the align-mode copies write")
    parser.add_argument("--op-audit", action="store_true",
                        help="record every dispatched op and the custom ops' view and int64 arguments (make_op_audit)")
    parser.add_argument("--switch-after", nargs=2, action="append", default=[], metavar=("K", "LINES"),
                        help="after the K-th prompt, replace the runtime switch file (VLLM_HPU_RUNTIME_SWITCH_FILE in the "
                        "environment) with LINES, NAME=VALUE pairs separated by ';' (repeatable): a flag switched in "
                        "the middle of a run, as an operator's A/B on one server does")
    args = parser.parse_args(argv)

    install()
    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.ckpt,
                  tokenizer=args.ckpt,
                  skip_tokenizer_init=True,
                  dtype="bfloat16",
                  max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs,
                  language_model_only=True,
                  trust_remote_code=True,
                  enable_prefix_caching=not args.no_prefix_caching)
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.k:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.k}
    llm = LLM(**kwargs)
    runner = find_runner(llm)
    pool_probe = probe_recurrent_pools(runner) if args.slice_write_probe else None
    steps = _record_steps(runner)
    committed = _record_committed(runner)
    draft_calls = _record_draft_calls(runner) if args.k else []
    oracle_record = None
    if args.oracle:
        oracle = json.loads(Path(args.oracle).read_text(encoding="utf-8"))
        oracle_record = install_oracle_drafts(llm, runner, oracle, args.k)

    source = json.loads(Path(args.prompts or Path(args.ckpt) / "reference.json").read_text(encoding="utf-8"))
    prompts = sorted((list(p["prompt_ids"]) for p in source["prompts"]), key=len)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, detokenize=False,
                            seed=args.seed, logprobs=args.logprobs, top_k=args.top_k, top_p=args.top_p)
    guard = make_index_guard()
    audit = make_op_audit() if args.op_audit else None
    streams = []
    with guard, (audit if audit is not None else contextlib.nullcontext()):
        batches = [prompts] if args.concurrent else [[prompt] for prompt in prompts]
        switches = {int(k): lines for k, lines in args.switch_after}
        for number, batch in enumerate(batches, 1):
            outputs = llm.generate([{"prompt_token_ids": prompt} for prompt in batch], params, use_tqdm=False)
            for prompt, output in zip(batch, outputs):
                stream = {"prompt_ids": prompt, "generated_ids": list(output.outputs[0].token_ids),
                          "num_cached_tokens": getattr(output, "num_cached_tokens", None),
                          "draft_calls_end": len(draft_calls)}
                if args.logprobs:
                    stream["top_logprobs"] = [{str(t): float(lp.logprob) for t, lp in step.items()}
                                              for step in (output.outputs[0].logprobs or [])]
                streams.append(stream)
            if number in switches:
                write_switch_file(switches[number])
        if args.next_turn and streams:
            turn = streams[0]["prompt_ids"] + streams[0]["generated_ids"] + list(json.loads(args.next_turn))
            output = llm.generate([{"prompt_token_ids": turn}], params, use_tqdm=False)[0]
            streams.append({"prompt_ids": turn, "generated_ids": list(output.outputs[0].token_ids),
                            "num_cached_tokens": getattr(output, "num_cached_tokens", None), "next_turn": True})
    record = {"k": args.k,
              "prefix_caching": not args.no_prefix_caching,
              "concurrent": bool(args.concurrent),
              "prompts": streams,
              "decode_steps": steps,
              "committed": committed,
              "draft_calls": draft_calls,
              "oracle": oracle_record,
              "index_events": guard.events(),
              "op_audit": audit.record() if audit is not None else None,
              "slice_write_probe": pool_probe,
              "runner": runner_facts(runner)}
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")
    return 0


def run(ckpt: Path, out: Path, *options: str, env: dict | None = None, timeout_s: float = 1800.0,
        cwd: Path | None = None) -> dict:
    """Run one rehearsal in a process of its own and return its record. ``options`` are CLI options of
    :func:`main`; ``env`` is added to the engine's environment (plugin flags); ``cwd`` is the engine's working
    directory (default the repository root), where files a server writes there land. A run that fails raises
    ``RuntimeError`` with the end of its output."""
    run_env = dict(os.environ)
    run_env.update(ENGINE_ENV)
    run_env.update(env or {})
    inherited = [entry for entry in run_env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    run_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + inherited)
    command = [sys.executable, "-m", "tests.unit_tests.hpu_cpu_engine", "--ckpt", str(ckpt), "--out", str(out), *options]
    log = Path(str(out) + ".log")
    with open(log, "wb") as stream:
        done = subprocess.run(command, cwd=cwd or REPO_ROOT, env=run_env, stdout=stream, stderr=subprocess.STDOUT,
                              timeout=timeout_s, check=False)
    if done.returncode != 0 or not Path(out).is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"the engine rehearsal failed (exit {done.returncode}, log {log}):\n{tail}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
