# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup (n-gram) speculation through the real vLLM engine with the HPU plugin, on CPU.

``python -m tests.unit_tests.models.flash_next.lookup_engine --plan PLAN.json --out OUT.json`` (from the repository
root, ``PYTHONPATH`` = the root) boots ``vllm.LLM`` on a tiny Flash-Next checkpoint in this process (TP1, the
in-process engine core, the HPU platform, worker and model runner) once without speculation and once per
speculative configuration of the plan, runs the plan's scenarios on each, and writes every stream and step record
to ``OUT.json``. ``test_lookup_engine.py`` starts it as a child process: the stand-ins below replace torch
functions and the platform for the whole process.

What stands in for the Gaudi runtime (the same seams as the other CPU engine rehearsals):

* ``VLLM_USE_FAKE_HPU=1`` and ``hpu_cpu_stubs``; code that names ``"hpu"`` directly is sent to the CPU;
  ``wrap_in_hpu_graph`` returns the module; ``FusedRMSNorm`` and ``apply_rotary_pos_emb`` are torch formulas;
  ``torch.ops.hpu.mixture_of_experts`` is ``moe_block``'s stand-in.
* The HPU kernels do not bounds-check an index: the base path sends ``-1`` positions and slots of padding rows
  through them and discards the results. CPU kernels raise, so :class:`IndexGuard` wraps negative indices and
  clamps large ones (and counts where).
* The model runs in bfloat16 (the platform refuses float32). The random tiny model's next-token distribution is
  nearly flat, so two logits are often equal or one bfloat16 step apart, and two correct computations of the same
  position in a different batch shape can pick different tokens there. The baseline therefore records the gap
  between the two largest log-probabilities of every token: a speculative stream must equal the baseline up to its
  first difference, and that difference must sit on a near-tie of the baseline (``test_lookup_engine._compare``).

Plan (JSON): ``{"ckpt": DIR, "tie": float, "max_model_len": int, "max_num_seqs": int, "baseline_prefix_caching": bool,
"scenarios": [...], "configs": [...]}``. A scenario is ``{"name", "requests": [{"id", "prompt_ids" | "turn_of",
"suffix_ids", "max_tokens"}]}``: the requests of a scenario are submitted together (one batch, each request
finishing at its own length); a request with ``turn_of`` continues the conversation of an earlier scenario's
request (its prompt is that request's prompt, plus its baseline output, plus ``suffix_ids``), so it starts with a
prefix-cache hit. A config is ``{"name", "speculative_config", "prefix_caching", "oracle": null | {"schedule": [L,
...], "tamper_every": int}}``: with an oracle the drafter's ``propose`` returns, per request and step, the next
``L`` tokens of the baseline stream (``L = 0`` is the empty draft, which the runner schedules as ``[-1]``), every
``tamper_every``-th step with one draft token replaced by a token the target never picks first. A config with ``env_oracle`` (``{"schedule": [...], "tamper_every": int}``) uses the product's own
rehearsal oracle instead (``flash_next_lookup.LookupOracle``): the child writes the baseline streams to a file and
names it in ``VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE``, as ``tools/flash_next/rental/lookup_tiny.py`` does on a card.
A config may name
the ``scenarios`` it runs and ``mutations`` of the runner (``"no_widen"``: the verification batch is the widest
scheduled row, the width before ``verify_width``), for the check that the comparison catches a broken lattice.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
import traceback
from collections import Counter

os.environ["VLLM_USE_FAKE_HPU"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"
os.environ.setdefault("VLLM_HPU_FLASH_NEXT_QSA_MODE", "1")
os.environ.setdefault("VLLM_HPU_RECURRENT_STATE_POOL", "1")
os.environ.setdefault("VLLM_SKIP_WARMUP", "true")

TAMPER = 51234  # a token id the target's first choice never is in these runs (the tests check every tampered step)
TOP = 5  # log-probabilities the baseline records per token: a tie can join more than two tokens


def _install_runtime_stand_ins():
    """The CPU stand-ins for the Gaudi runtime (module docstring). Returns ``(LLM, SamplingParams)``."""
    from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

    import torch

    factories = ("tensor", "zeros", "ones", "full", "empty", "arange", "zeros_like", "ones_like", "full_like",
                 "empty_like", "as_tensor", "randn", "rand", "randint", "eye", "linspace", "scalar_tensor")

    def is_hpu(device) -> bool:
        if isinstance(device, str):
            return device.split(":")[0] == "hpu"
        return isinstance(device, torch.device) and device.type == "hpu"

    for name in factories:
        real = getattr(torch, name)

        def factory(*args, _real=real, **kwargs):
            if is_hpu(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return _real(*args, **kwargs)

        factory.__name__ = getattr(real, "__name__", name)
        setattr(torch, name, factory)

    def cpu_args(args, kwargs):
        args = tuple("cpu" if is_hpu(a) else a for a in args)
        if is_hpu(kwargs.get("device")):
            kwargs = dict(kwargs, device="cpu")
        return args, kwargs

    real_to = torch.Tensor.to
    torch.Tensor.to = lambda self, *a, **k: real_to(self, *cpu_args(a, k)[0], **cpu_args(a, k)[1])
    real_module_to = torch.nn.Module.to
    torch.nn.Module.to = lambda self, *a, **k: real_module_to(self, *cpu_args(a, k)[0], **cpu_args(a, k)[1])
    for name in ("new_zeros", "new_ones", "new_full", "new_empty", "new_tensor"):
        real_new = getattr(torch.Tensor, name)

        def new(self, *args, _real=real_new, **kwargs):
            if is_hpu(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return _real(self, *args, **kwargs)

        setattr(torch.Tensor, name, new)

    import habana_frameworks.torch  # noqa: F401
    import psutil

    torch.hpu.current_device = lambda: torch.device("cpu")
    torch.hpu.is_available = lambda: False
    torch.hpu.synchronize = lambda *args, **kwargs: None
    torch.hpu.device_count = lambda: 1
    if not hasattr(psutil.Process, "cpu_affinity"):  # macOS
        psutil.Process.cpu_affinity = lambda self, *args: list(range(os.cpu_count() or 1))

    import habana_frameworks.torch.core  # noqa: F401
    import habana_frameworks.torch.hpu as hhpu
    import habana_frameworks.torch.utils.internal as hinternal

    hinternal.is_lazy = lambda: True

    def wrap_in_hpu_graph(module, **kwargs):
        forward = module.forward

        def graph_forward(*args, **kw):
            kw.pop("bypass_hpu_graphs", None)
            return forward(*args, **kw)

        module.forward = graph_forward
        return module

    hhpu.wrap_in_hpu_graph = wrap_in_hpu_graph

    import habana_frameworks.torch.hpex.kernels as hkernels

    class RotaryPosEmbeddingMode(enum.Enum):
        BLOCKWISE = 0
        PAIRWISE = 1

    def apply_rotary_pos_emb(x, cos, sin, position_ids, offset, mode):
        assert position_ids is None and offset == 0
        if mode is RotaryPosEmbeddingMode.BLOCKWISE:
            half = x.shape[-1] // 2
            rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        else:
            rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    hkernels.RotaryPosEmbeddingMode = RotaryPosEmbeddingMode
    hkernels.apply_rotary_pos_emb = apply_rotary_pos_emb

    import habana_frameworks.torch.hpex.normalization as hnorm

    class FusedRMSNorm:

        @staticmethod
        def apply(x, weight, eps):
            xf = x.float()
            out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
            return out.to(x.dtype)

    hnorm.FusedRMSNorm = FusedRMSNorm

    import torch._dynamo

    @torch._dynamo.register_backend(name="hpu_backend")
    def _hpu_backend(gm, example_inputs, **kwargs):
        return gm.forward

    import vllm.platforms
    from vllm_gaudi.platform import HpuPlatform

    vllm.platforms._current_platform = HpuPlatform()

    import vllm_gaudi

    vllm_gaudi.register()
    vllm_gaudi.register_utils()
    vllm_gaudi.register_ops()
    import vllm_gaudi.models.qwen4_exp as flash_next

    flash_next.register()

    from tests.unit_tests.models.flash_next import moe_block

    moe_block.install_vendor_ops()
    moe_block._keep_checkpoint_weight_layout()

    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def _index_guard():
    """A TorchDispatchMode that wraps negative indices and clamps large ones (the HPU kernels' tolerance)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    dim_index = {"aten::index_select": (1, 2), "aten::index_copy_": (1, 2), "aten::index_copy": (1, 2),
                 "aten::index_add_": (1, 2), "aten::index_fill_": (1, 2), "aten::gather": (1, 2),
                 "aten::scatter_": (1, 2), "aten::scatter": (1, 2), "aten::scatter_add_": (1, 2)}
    index_list = ("aten::index", "aten::index_put_", "aten::index_put", "aten::_unsafe_index")

    class IndexGuard(TorchDispatchMode):

        def __init__(self) -> None:
            super().__init__()
            self.records: Counter = Counter()
            self._busy = False

        def _fix(self, op, index, size):
            if index.dtype == torch.bool or index.numel() == 0:
                return index
            lo, hi = int(index.min()), int(index.max())
            if lo >= 0 and hi < size:
                return index
            self.records[op] += 1
            return torch.where(index < 0, index + size, index).clamp(0, size - 1)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if self._busy:
                return func(*args, **kwargs)
            name = func._schema.name
            self._busy = True
            try:
                if name in dim_index and len(args) > 2:
                    dim_pos, index_pos = dim_index[name]
                    dim, index = args[dim_pos], args[index_pos]
                    if isinstance(index, torch.Tensor) and isinstance(dim, int):
                        size = args[0].shape[dim] if args[0].dim() else 1
                        fixed = self._fix(name, index, size)
                        if fixed is not index:
                            args = list(args)
                            args[index_pos] = fixed
                elif name == "aten::embedding" and len(args) > 1:
                    fixed = self._fix(name, args[1], args[0].shape[0])
                    if fixed is not args[1]:
                        args = list(args)
                        args[1] = fixed
                elif name in index_list and len(args) > 1:
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

    return IndexGuard()


def tie_free_length(gaps: list[float], tie: float) -> int:
    """Tokens before the first near-tie: the first index whose top-2 log-probability gap is below ``tie``."""
    for index, gap in enumerate(gaps):
        if gap < tie:
            return index
    return len(gaps)


# Variables the platform hooks and the runner write at start-up (VLLM_COMPACT_GDN = 0 under prefix caching, the
# resolved pool size): restored before every boot, so an engine without prefix caching starts like a fresh server.
_BOOT_ENV = ("VLLM_COMPACT_GDN", "VLLM_HPU_RECURRENT_STATE_POOL", "VLLM_HPU_RECURRENT_STATE_POOL_SLOTS",
             "VLLM_USE_HYBRID_CACHE", "VLLM_USE_NAIVE_MAMBA_CACHE_SHARING")
_ENV_AT_START = {name: os.environ.get(name) for name in _BOOT_ENV}


def _boot(LLM, plan: dict, speculative_config: dict | None, prefix_caching: bool, env: dict | None = None):
    for name, value in _ENV_AT_START.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.environ.pop("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP", None)
    os.environ.pop("VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE", None)
    for name, value in (env or {}).items():
        os.environ[name] = str(value)
    kwargs = dict(model=plan["ckpt"], tokenizer=plan["ckpt"], skip_tokenizer_init=True, dtype="bfloat16",
                  max_model_len=plan.get("max_model_len", 4096), max_num_seqs=plan.get("max_num_seqs", 4),
                  language_model_only=True, trust_remote_code=True, enable_prefix_caching=prefix_caching)
    if speculative_config:
        kwargs["speculative_config"] = speculative_config
    return LLM(**kwargs)


def _runner(llm):
    return llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner


def _prompts(scenario: dict, streams: dict) -> list[list[int]]:
    """Each request's prompt ids; a ``turn_of`` request continues an earlier request with its baseline output."""
    out = []
    for request in scenario["requests"]:
        if "turn_of" in request:
            prev = streams[request["turn_of"]]
            out.append(list(prev["prompt_ids"]) + list(prev["tokens"]) + list(request["suffix_ids"]))
        else:
            out.append(list(request["prompt_ids"]))
    return out


def run_baseline(LLM, SamplingParams, plan: dict) -> dict:
    """The engine without speculation: every scenario's streams and per-token top-2 gaps."""
    llm = _boot(LLM, plan, None, plan.get("baseline_prefix_caching", True))
    streams: dict[str, dict] = {}
    for scenario in plan["scenarios"]:
        prompts = _prompts(scenario, streams)
        params = [SamplingParams(temperature=0, max_tokens=r["max_tokens"], detokenize=False, logprobs=TOP)
                  for r in scenario["requests"]]
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        for request, prompt, out in zip(scenario["requests"], prompts, outs):
            completion = out.outputs[0]
            gaps, top = [], []
            for lp in completion.logprobs or []:
                ranked = sorted(((v.logprob, int(t)) for t, v in lp.items()), reverse=True)
                gaps.append(float(ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else float("inf"))
                # each of the TOP best tokens with its distance to the best log-probability
                top.append([[t, float(ranked[0][0] - v)] for v, t in ranked[:TOP]])
            tokens = list(completion.token_ids)
            usable = tie_free_length(gaps, plan["tie"])
            streams[request["id"]] = {"prompt_ids": prompt, "tokens": tokens, "gaps": gaps, "top": top,
                                      "tie_free": usable}
    del llm
    return streams


def run_config(LLM, SamplingParams, plan: dict, config: dict, base: dict) -> dict:
    """One speculative engine over every scenario; each request runs to its baseline's tie-free length."""
    env = dict(config.get("env") or {})
    env_oracle = config.get("env_oracle")
    if env_oracle is not None:
        # the product's rehearsal oracle: the baseline streams in a file named by the environment
        import tempfile

        fd, oracle_path = tempfile.mkstemp(prefix="lookup-oracle-", suffix=".json", dir="/tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"streams": [list(s["prompt_ids"]) + list(s["tokens"]) for s in base.values()],
                       "schedule": env_oracle["schedule"], "tamper_every": int(env_oracle.get("tamper_every", 0))}, f)
        env["VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE"] = oracle_path
    llm = _boot(LLM, plan, config["speculative_config"], config["prefix_caching"], env)
    runner = _runner(llm)
    record = {"drafter": type(runner.drafter).__name__, "prefix_caching": bool(runner.use_prefix_caching),
              "recurrent_spec": bool(runner._recurrent_spec), "steps": [], "widths": [], "boundary_commits": 0,
              "streams": {}}
    by_prompt = {tuple(s["prompt_ids"]): s for s in base.values()}
    oracle = config.get("oracle")
    state = {"step": 0}

    import vllm_gaudi.v1.worker.hpu_model_runner as runner_module

    real_width = runner_module.verification_width

    mutations = set(config.get("mutations", ()))

    def width_spy(num_scheduled, accepted, recurrent_spec):
        # mutation "no_widen": the batch width before verify_width (the widest scheduled row), for the check that the
        # comparison catches the stale-state load it fixes
        width = (max(int(n) for n in num_scheduled) if "no_widen" in mutations else
                 real_width(num_scheduled, accepted, recurrent_spec))
        record["widths"].append({"scheduled": [int(n) for n in num_scheduled], "accepted": [int(a) for a in accepted],
                                 "width": int(width)})
        return width

    runner_module.verification_width = width_spy

    real_commit = runner._commit_recurrent_boundaries

    def commit_spy(boundary, accepted_counts):
        out = real_commit(boundary, accepted_counts)
        _load, _store, context_lens, num_decodes, block_size = boundary
        accepted = [int(a) for a in accepted_counts[:num_decodes]]
        for ctx, a in zip(context_lens, accepted):
            record["boundary_commits"] += sum(1 for t in range(a) if (int(ctx) + t + 1) % block_size == 0)
        return out

    runner._commit_recurrent_boundaries = commit_spy

    lookup_oracle = config.get("lookup_oracle")
    real_find = None
    if lookup_oracle is not None:
        import vllm_gaudi.v1.spec_decode.flash_next_lookup as lookup_module

        real_find = lookup_module.find_lookup_draft
        schedule = lookup_oracle["schedule"]
        calls = {"n": 0}

        def oracle_find(tokens, min_ngram, max_ngram, k, max_model_len):
            row = [int(t) for t in tokens]
            # the request whose prompt is the longest prefix of the row (a shorter prompt can be a prefix of it)
            owners = [s for s in base.values() if len(row) >= len(s["prompt_ids"])
                      and row[:len(s["prompt_ids"])] == list(s["prompt_ids"])]
            stream = max(owners, key=lambda s: len(s["prompt_ids"])) if owners else None
            length = schedule[calls["n"] % len(schedule)]
            calls["n"] += 1
            if stream is None or length == 0:
                draft = []
            else:
                done = len(row) - len(stream["prompt_ids"])
                draft = list(stream["tokens"][done:done + min(length, k)])
            record["steps"].append({"step": calls["n"] - 1, "row": -1, "done": len(row), "sampled": [],
                                    "draft": draft, "tampered": None, "lookup": True})
            return draft

        lookup_module.find_lookup_draft = oracle_find
        real_spec_merge = runner._spec_lookup.merge if runner._spec_lookup is not None else None
        if real_spec_merge is None:
            raise RuntimeError("the lookup hook is not installed: VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP did not reach the "
                               "runner")

        def merge_spy(mtp_drafts, req_ids, *args, **kwargs):
            drafts = real_spec_merge(mtp_drafts, req_ids, *args, **kwargs)
            record.setdefault("hybrid_rows", []).append({"drafts": drafts, "sources": [
                runner._spec_lookup.last_source.get(r) for r in req_ids]})
            return drafts

        runner._spec_lookup.merge = merge_spy
    elif oracle is not None:
        schedule, tamper_every = oracle["schedule"], int(oracle.get("tamper_every", 0))

        def oracle_propose(num_speculative_tokens, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *a, **kw):
            drafts = []
            step = state["step"]
            for i, sampled in enumerate(sampled_token_ids):
                if not sampled:
                    drafts.append([])
                    continue
                n = int(num_tokens_no_spec[i])
                plen = int(runner.input_batch.num_prompt_tokens[i])
                stream = by_prompt[tuple(int(t) for t in token_ids_cpu[i, :plen])]
                done = n - plen
                length = min(schedule[(step + i) % len(schedule)], num_speculative_tokens)
                draft = list(stream["tokens"][done:done + length])
                tampered = None
                if draft and tamper_every and step % tamper_every == tamper_every - 1:
                    tampered = min(len(draft) - 1, step % len(draft))
                    draft[tampered] = TAMPER
                drafts.append(draft)
                record["steps"].append({"step": step, "row": i, "done": done, "sampled": [int(t) for t in sampled],
                                        "draft": draft, "tampered": tampered})
            state["step"] += 1
            return drafts

        runner.drafter.propose = oracle_propose
    elif env_oracle is not None and config["speculative_config"].get("method") == "ngram":
        if getattr(runner, "_lookup_oracle", None) is None:
            raise RuntimeError("the rehearsal oracle did not reach the runner (VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE)")
        real_ngram = runner.propose_ngram_draft_token_ids

        def ngram_spy(sampled_token_ids):
            drafts = real_ngram(sampled_token_ids)
            for i, draft in enumerate(drafts):
                record["steps"].append({"step": state["step"], "row": i, "sampled": [int(t) for t in sampled_token_ids[i]],
                                        "draft": [int(t) for t in draft if int(t) != -1], "tampered": None})
            state["step"] += 1
            return drafts

        runner.propose_ngram_draft_token_ids = ngram_spy
    elif env_oracle is not None:
        if runner._spec_lookup is None or runner._spec_lookup.oracle is None:
            raise RuntimeError("the lookup hook or its rehearsal oracle is not installed")
        real_env_merge = runner._spec_lookup.merge

        def env_merge_spy(mtp_drafts, req_ids, *args, **kwargs):
            drafts = real_env_merge(mtp_drafts, req_ids, *args, **kwargs)
            record.setdefault("hybrid_rows", []).append({"drafts": drafts, "sources": [
                runner._spec_lookup.last_source.get(r) for r in req_ids]})
            return drafts

        runner._spec_lookup.merge = env_merge_spy
    elif config["speculative_config"].get("method") == "ngram":
        real_propose = runner.drafter.propose

        def propose_spy(*args, **kwargs):
            drafts = real_propose(*args, **kwargs)
            sampled = args[1] if len(args) > 1 else kwargs.get("sampled_token_ids")
            for i, draft in enumerate(drafts):
                record["steps"].append({"step": state["step"], "row": i, "sampled": [int(t) for t in sampled[i]],
                                        "draft": [int(t) for t in draft]})
            state["step"] += 1
            return drafts

        runner.drafter.propose = propose_spy

    for scenario in plan["scenarios"]:
        if config.get("scenarios") is not None and scenario["name"] not in config["scenarios"]:
            continue
        prompts = _prompts(scenario, base)
        params = [SamplingParams(temperature=0, max_tokens=r["max_tokens"], detokenize=False)
                  for r in scenario["requests"]]
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        for request, out in zip(scenario["requests"], outs):
            record["streams"][request["id"]] = list(out.outputs[0].token_ids)
    runner_module.verification_width = real_width
    if real_find is not None:
        lookup_module.find_lookup_draft = real_find
        record["hybrid_counts"] = runner._spec_lookup.counts
    if env_oracle is not None:
        oracle_object = getattr(runner, "_lookup_oracle", None) or getattr(runner._spec_lookup, "oracle", None)
        record["oracle"] = {"drafts": oracle_object.drafts, "tampered": oracle_object.tampered}
        if runner._spec_lookup is not None:
            record["hybrid_counts"] = runner._spec_lookup.counts
        os.unlink(env["VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE"])
    del llm
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(open(args.plan, encoding="utf-8").read())
    LLM, SamplingParams = _install_runtime_stand_ins()
    guard = _index_guard()
    result: dict = {"configs": {}}
    try:
        with guard:
            result["baseline"] = run_baseline(LLM, SamplingParams, plan)
            for config in plan["configs"]:
                try:
                    result["configs"][config["name"]] = run_config(LLM, SamplingParams, plan, config,
                                                                   result["baseline"])
                except Exception:  # noqa: BLE001 - one configuration's failure is that configuration's record
                    result["configs"][config["name"]] = {"error": traceback.format_exc()}
    except Exception:  # noqa: BLE001 - the parent test prints the child's failure
        result["error"] = traceback.format_exc()
    result["index_guard"] = dict(guard.records)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
