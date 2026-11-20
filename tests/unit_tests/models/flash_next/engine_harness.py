# SPDX-License-Identifier: Apache-2.0
"""A whole serving loop of the HPU runner on CPU: the engine's scheduler and step functions, the real
``HPUModelRunner`` and the tiny Flash-Next model.

Plain functions (no pytest import), next to ``cpu_model.py`` and ``hpu_bucket.py``. The other CPU tests call the
model with hand-built metadata; here nothing between the request and the token is hand-built:

* the engine side is vLLM's own ``Scheduler`` (or ``AsyncScheduler`` under async scheduling, which the plugin
  rebinds to ``HPUAsyncScheduler``) with its ``KVCacheManager`` (the plugin's recurrent-pool coordinator), driven
  by ``EngineCore.step`` / ``EngineCore.step_with_batch_queue`` themselves, bound to an object that carries only
  the attributes those two methods read;
* the worker side is a real ``HPUModelRunner`` (``__init__``, ``get_kv_cache_spec``, ``initialize_kv_cache``,
  ``execute_model``, ``sample_tokens`` and everything below them) around the tiny checkpoint in the real
  ``HpuModelAdapter``;
* the executor in between is :class:`LocalExecutor`, the single-process form of vLLM's ``UniProcExecutor``: a
  non-blocking ``sample_tokens`` hands back a future, and an ``AsyncModelRunnerOutput`` is turned into tokens only
  when the engine asks for that future's result. Under async scheduling that is after the next step was
  submitted, so every test of that mode runs with the latest possible host read of the sampled tokens.

What a card has and this Mac has not is replaced at the same seams the other tests use: ``habana_frameworks``
and ``torch.hpu`` (``hpu_cpu_stubs``), the device (``"hpu"`` becomes the CPU in ``async_h2d_copy`` /
``async_h2d_update`` and in the few literal ``device="hpu"`` constructors), and the paged-attention kernels (the
real ``HpuQsaSparseAttention`` wrapper over ``hpu_bucket.FakeHpuAttentionImpl``, bound to the K/V tensors the
runner allocated). The platform hooks a server runs before the worker starts are run here in the same order
(``test_kv_cache_sizing.py`` holds that order against a card log).

:class:`Counters` is the instrumentation of the engine-step track: every host-to-device copy (calls and bytes,
by call site), every host read of a device tensor, and the wall time of the runner's phases.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import os  # noqa: E402

os.environ["TMPDIR"] = "/tmp"  # the gloo init file must not sit on a sandboxed TMPDIR
os.environ.setdefault("VLLM_USE_FAKE_HPU", "1")  # extension.environment: hardware "cpu"

import collections  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402

import vllm_gaudi.utils  # noqa: E402, F401  (installs the recurrent-pool coordinator factory and HPUAsyncScheduler)
import vllm.platforms  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    generate_scheduler_kv_cache_config, get_kv_cache_configs, get_request_block_hasher, init_none_hash,
    resolve_kv_cache_block_sizes, update_kv_cache_capacity)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
from vllm.v1.engine.core import EngineCore  # noqa: E402
from vllm.v1.outputs import AsyncModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

import vllm_gaudi.models.qwen4_exp as flash_next  # noqa: E402
import vllm_gaudi.patches as hpu_patches  # noqa: E402
import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: E402, F401  (register_ops installs it in every server)
import vllm_gaudi.v1.worker.hpu_input_batch as input_batch_module  # noqa: E402
import vllm_gaudi.v1.worker.hpu_model_runner as runner_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import HpuQsaSparseAttention  # noqa: E402
from vllm_gaudi.platform import HpuPlatform, _configure_recurrent_state  # noqa: E402
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionBackendV1  # noqa: E402

from tests.unit_tests.models.flash_next import hpu_bucket  # noqa: E402
from tests.unit_tests.models.flash_next.cpu_model import to_fp32  # noqa: E402

# Every variable a start reads or writes; restored when the engine closes.
ENV_VARS = ("VLLM_HPU_RECURRENT_STATE_POOL", "VLLM_HPU_RECURRENT_STATE_POOL_SLOTS",
            "VLLM_HPU_RECURRENT_STATE_POOL_CACHE_SLOTS", "VLLM_COMPACT_GDN", "VLLM_HPU_FLASH_NEXT_QSA_MODE",
            "VLLM_USE_HYBRID_CACHE", "VLLM_USE_NAIVE_MAMBA_CACHE_SHARING", "VLLM_SKIP_WARMUP",
            "VLLM_HPU_PLE_HOST_NGRAM_IDS", "VLLM_HPU_FUSED_GREEDY_LOGITS", "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
            "VLLM_DEFRAG_WITH_GRAPHS", "VLLM_HPU_FLASH_NEXT_DECODE_METADATA")

DEVICE = torch.device("cpu")
_OPEN = False


class Counters:
    """What crossed the host/device boundary and where the host time went, per labelled step.

    ``h2d`` maps a call site (``file:function:line`` of the caller of ``async_h2d_copy`` / ``async_h2d_update``
    / a literal device constructor) to ``[calls, bytes]``. ``sync`` maps a call site to the number of host
    reads (``.cpu()``, ``.tolist()``, ``.item()``, ``.numpy()`` and truth tests) of tensors that came from the
    device side. ``seconds`` maps a phase name to wall time. Counting is on only between ``start`` and ``stop``.
    """

    def __init__(self) -> None:
        self.active = False
        self.h2d: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        self.sync: dict[str, int] = collections.defaultdict(int)
        self.seconds: dict[str, float] = collections.defaultdict(float)
        self.calls: dict[str, int] = collections.defaultdict(int)
        self.device_ops: dict[str, int] = collections.defaultdict(int)  # torch calls on device tensors, by scope
        self.scope = "runner"  # "model" inside the model forward (the HPU graph), "runner" outside it

    def start(self) -> None:
        self.h2d.clear()
        self.sync.clear()
        self.seconds.clear()
        self.calls.clear()
        self.device_ops.clear()
        self.active = True

    def stop(self) -> None:
        self.active = False

    @staticmethod
    def _site(depth: int = 2) -> str:
        frame = sys._getframe(depth)
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        if frame is None:
            return "?"
        return f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{frame.f_lineno}"

    def copy(self, tensor: torch.Tensor) -> None:
        if self.active:
            entry = self.h2d[self._site(3)]
            entry[0] += 1
            entry[1] += tensor.numel() * tensor.element_size()

    def read(self) -> None:
        if self.active:
            self.sync[self._site(3)] += 1

    @contextlib.contextmanager
    def phase(self, name: str):
        begin = time.perf_counter()
        try:
            yield
        finally:
            if self.active:
                self.seconds[name] += time.perf_counter() - begin
                self.calls[name] += 1

    def op(self) -> None:
        if self.active:
            self.device_ops[self.scope] += 1

    @contextlib.contextmanager
    def scoped(self, scope: str):
        previous, self.scope = self.scope, scope
        try:
            yield
        finally:
            self.scope = previous

    def totals(self) -> dict:
        return {
            "h2d_calls": sum(entry[0] for entry in self.h2d.values()),
            "h2d_bytes": sum(entry[1] for entry in self.h2d.values()),
            "host_reads": sum(self.sync.values()),
            "runner_device_ops": self.device_ops.get("runner", 0),
        }


class DeviceTensor(torch.Tensor):
    """Marks a tensor as living on the device, so a host read of it is seen.

    The mark follows every torch function (a subclass result stays a subclass). ``COUNTERS`` is the active
    :class:`Counters`; the read methods below are the host synchronisation points of a card.
    """

    COUNTERS: Counters | None = None
    _READS = frozenset({"cpu", "tolist", "item", "numpy", "__bool__", "__int__", "__float__", "__index__"})
    # Attribute reads and shape arithmetic: no device work.
    _FREE = frozenset({"__get__", "size", "dim", "numel", "__len__", "__repr__", "element_size", "is_floating_point",
                       "nelement", "ndimension", "__format__", "__hash__", "__reduce_ex__", "__deepcopy__",
                       "as_subclass", "type", "is_contiguous", "data_ptr", "get_device", "requires_grad_"})

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if cls.COUNTERS is not None:
            if name in cls._READS:
                cls.COUNTERS.read()
            elif name not in cls._FREE:
                cls.COUNTERS.op()
        result = super().__torch_function__(func, types, args, kwargs)
        if name in ("cpu", "numpy") and isinstance(result, torch.Tensor):
            return result.as_subclass(torch.Tensor)  # the host copy is a host tensor
        return result


def mark_device(tensor):
    return tensor.as_subclass(DeviceTensor) if isinstance(tensor, torch.Tensor) else tensor


@dataclasses.dataclass
class Options:
    """One server start. ``env`` is applied on top of the rental defaults (sparse QSA, recurrent pool)."""
    max_num_seqs: int = 32
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    prefix_caching: bool = True
    async_scheduling: bool = False
    fp32: bool = True
    kv_cache_bytes: int = 192 << 20
    speculative_config: dict | None = None
    env: dict = dataclasses.field(default_factory=dict)
    count: bool = False  # mark device tensors and count copies / host reads (slower)
    long_prefill_token_threshold: int = 0
    # A server at another model's geometry without its weights: the model's config.json, the tensor-parallel
    # size its per-rank shapes are computed for, and a model stand-in that returns fixed logits (StubModel).
    # Tensors above ``meta_bytes`` are allocated on the meta device (shape and dtype, no storage).
    stub_config: Path | None = None
    tensor_parallel_size: int = 1
    meta_bytes: int = 64 << 20


class LocalExecutor:
    """The engine's executor over one in-process runner: ``UniProcExecutor``'s contract without a worker class."""

    def __init__(self, runner, counters: Counters) -> None:
        self.runner = runner
        self.counters = counters
        self.messages: list[tuple[str, object]] = []  # ("execute_model", scheduler_output), ... when recording
        self.record = False

    @staticmethod
    def _done(value) -> Future:
        future: Future = Future()
        future.set_result(value)
        return future

    def execute_model(self, scheduler_output, non_block: bool = False):
        if self.record:
            self.messages.append(("execute_model", scheduler_output))
        with self.counters.phase("worker.execute_model"):
            output = self.runner.execute_model(scheduler_output)
        return self._done(output) if non_block else output

    def sample_tokens(self, grammar_output, non_block: bool = False):
        if self.record:
            self.messages.append(("sample_tokens", grammar_output))
        with self.counters.phase("worker.sample_tokens"):
            output = self.runner.sample_tokens(grammar_output)
        if not non_block:
            if isinstance(output, AsyncModelRunnerOutput):
                output = output.get_output()
            if self.record:
                self.messages.append(("reply", output))
            return output
        if isinstance(output, AsyncModelRunnerOutput):
            return _AsyncOutputFuture(output, self)
        if self.record:
            self.messages.append(("reply", output))
        return self._done(output)

    def take_draft_token_ids(self):
        return self.runner.take_draft_token_ids()


class _AsyncOutputFuture(Future):
    """``uniproc_executor.AsyncOutputFuture``: the host read happens when the engine asks for the result."""

    def __init__(self, async_output: AsyncModelRunnerOutput, executor: LocalExecutor) -> None:
        super().__init__()
        self.async_output = async_output
        self.executor = executor

    def result(self, timeout=None):
        if not super().done():
            try:
                with self.executor.counters.phase("worker.async_output"):
                    output = self.async_output.get_output()
                if self.executor.record:
                    self.executor.messages.append(("reply", output))
                self.set_result(output)
            except Exception as error:  # noqa: BLE001  (the engine re-raises it from result())
                self.set_exception(error)
        return super().result()


class _BoundAttention:
    """A QSA layer's ``Attention`` child as the sparse wrapper sees it, over the runner's own K/V tensors."""

    def __init__(self, qsa_layer: torch.nn.Module, kv_cache) -> None:
        self.impl = hpu_bucket.FakeHpuAttentionImpl(qsa_layer.num_heads, qsa_layer.num_kv_heads, qsa_layer.head_dim,
                                                    qsa_layer.scaling)
        self.kv_cache = kv_cache


class Engine:
    """An open server: ``add`` requests, ``run`` to completion, read ``tokens``."""

    def __init__(self, vllm_config, runner, scheduler, core, executor, counters, hash_block_size) -> None:
        self.vllm_config = vllm_config
        self.runner = runner
        self.scheduler = scheduler
        self.core = core
        self.executor = executor
        self.counters = counters
        self.tokens: dict[str, list[int]] = {}
        self.finished: dict[str, str] = {}
        self.steps = 0
        self.trace: list[dict] = []  # per engine step: scheduled request ids and token counts
        self._hash_block_size = hash_block_size
        self._hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(self._hash_fn)

    def add(self, request_id: str, prompt_token_ids: list[int], max_tokens: int, **sampling) -> None:
        params = SamplingParams(max_tokens=max_tokens, **({"temperature": 0.0} | sampling))
        params.update_from_generation_config({}, eos_token_id=self.eos_token_id)
        request = Request(request_id=request_id,
                          prompt_token_ids=list(prompt_token_ids),
                          mm_features=None,
                          sampling_params=params,
                          pooling_params=None,
                          block_hasher=get_request_block_hasher(self._hash_block_size, self._hash_fn)
                          if self._hash_block_size else None)
        self.tokens[request_id] = []
        self.scheduler.add_request(request)

    @property
    def eos_token_id(self) -> int:
        eos = self.vllm_config.model_config.hf_text_config.eos_token_id
        return eos[0] if isinstance(eos, (list, tuple)) else eos

    def has_work(self) -> bool:
        return self.scheduler.has_requests() or bool(self.core.batch_queue)

    def step(self) -> bool:
        """One call of the engine's step function; outputs are appended to ``tokens``. True if a model ran."""
        with self.counters.phase("engine.step"):
            outputs, model_executed = self.core.step_fn()
            self.core.post_step(model_executed)
        self.steps += 1
        for engine_outputs in (outputs or {}).values():
            for output in engine_outputs.outputs:
                self.tokens[output.request_id].extend(output.new_token_ids)
                if output.finish_reason is not None:
                    self.finished[output.request_id] = str(output.finish_reason)
        return model_executed

    def run(self, max_steps: int = 100000) -> dict[str, list[int]]:
        for _ in range(max_steps):
            if not self.has_work():
                return self.tokens
            self.step()
        raise RuntimeError(f"the engine did not finish in {max_steps} steps")


def _cpu_h2d_copy(counters: Counters, mark: bool):

    def async_h2d_copy(source, dest_tensor=None, dtype=None, device=None):
        if isinstance(source, torch.Tensor):
            plain = source.as_subclass(torch.Tensor)
            counters.copy(plain)
            if dest_tensor is not None:
                dest_tensor.as_subclass(torch.Tensor).copy_(plain)
                return dest_tensor
            target = plain.clone()
        else:
            if dtype is None:
                raise ValueError("dtype must be specified when source is not a tensor")
            target = torch.tensor(source, dtype=dtype)
            counters.copy(target)
        return mark_device(target) if mark else target

    return async_h2d_copy


def _cpu_tensor_to():
    """``Tensor.to`` with the device ``"hpu"`` sent to the CPU (the verification step moves its sampled tokens
    with ``.to("hpu", non_blocking=True)``)."""
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        if args and (args[0] == "hpu" or (isinstance(args[0], torch.device) and args[0].type == "hpu")):
            args = ("cpu", ) + args[1:]
        if kwargs.get("device") == "hpu":
            kwargs["device"] = "cpu"
        return real_to(self, *args, **kwargs)

    return to


def _cpu_h2d_update(counters: Counters):

    def async_h2d_update(source, dest, indices, device=None):
        idx = torch.tensor(indices, dtype=torch.long)
        vals = source[indices]
        counters.copy(idx)
        counters.copy(vals)
        dest.as_subclass(torch.Tensor).index_copy_(0, idx, vals.as_subclass(torch.Tensor))

    return async_h2d_update


class StubModel(torch.nn.Module):
    """Stands in for ``HpuModelAdapter`` at a geometry whose weights are not here: fixed hidden states and logits.

    The runner's host work does not depend on what the model computes, so a start at the real model's shapes
    (131072-token contexts, 1024 kernel blocks per request, the recurrent pool) can be timed on this Mac. The
    sampled token of a row is ``(input token + 1) % vocab`` over a 64-token vocabulary: never EOS.
    """

    requires_raw_input_tokens = True
    requires_block_table = True
    VOCAB = 64

    def __init__(self, weightless: torch.nn.Module) -> None:
        super().__init__()
        self.model = weightless.model  # .layers (QSA owners) and .config, as the runner reads them
        self._last_ids = None

    def forward(self, input_ids=None, positions=None, **kwargs):
        self._last_ids = input_ids.reshape(-1).as_subclass(torch.Tensor)
        return torch.zeros((self._last_ids.numel(), 4), dtype=torch.float32)

    def _logits(self, logits_indices):
        picked = self._last_ids[logits_indices.as_subclass(torch.Tensor).long().clamp_min(0)]
        return torch.nn.functional.one_hot((picked.long().clamp_min(0) + 1) % self.VOCAB, self.VOCAB).float()

    def select_and_compute_logits(self, hidden_states, logits_indices):
        return hidden_states[logits_indices], self._logits(logits_indices)

    def select_compute_logits_and_greedy(self, hidden_states, logits_indices, stop_ids=None, stop_rows=None,
                                         narrow_ids=True):
        ids = self._logits(logits_indices).argmax(dim=-1)
        return hidden_states[logits_indices], (ids.to(torch.int32) if narrow_ids else ids).unsqueeze(-1)

    def compute_logits(self, hidden_states):
        return torch.zeros((hidden_states.shape[0], self.VOCAB), dtype=torch.float32)


def _engine_config(ckpt_dir: Path, options: Options):
    kwargs = dict(model=str(ckpt_dir),
                  tokenizer=str(ckpt_dir),
                  skip_tokenizer_init=True,
                  dtype="bfloat16",
                  trust_remote_code=True,
                  language_model_only=True,
                  enforce_eager=True,
                  enable_prefix_caching=options.prefix_caching,
                  max_model_len=options.max_model_len,
                  max_num_seqs=options.max_num_seqs,
                  max_num_batched_tokens=options.max_num_batched_tokens,
                  long_prefill_token_threshold=options.long_prefill_token_threshold,
                  tensor_parallel_size=options.tensor_parallel_size,
                  async_scheduling=options.async_scheduling)
    if options.speculative_config is not None:
        kwargs["speculative_config"] = options.speculative_config
    vllm_config = EngineArgs(**kwargs).create_engine_config()
    # The CPU platform of this Mac decides async scheduling by its own rules; a server's decision is the HPU
    # platform's (vllm_gaudi/platform.py), which the tests of that hook cover. Here the caller chooses.
    vllm_config.scheduler_config.async_scheduling = bool(options.async_scheduling)
    return vllm_config


def _install_attention_seams(model: torch.nn.Module) -> int:
    """The real sparse wrapper on every QSA layer, over the runner's K/V tensors; -1 positions clamped."""
    layers = [module for module in model.modules() if isinstance(module, HpuQwen4ExpQSAAttention)]
    for layer in layers:
        kv_cache = layer.attn.kv_cache
        if kv_cache is None or kv_cache[0] is None:
            raise RuntimeError(f"{layer.prefix}: the runner bound no K/V cache")
        layer.sparse_attn = HpuQsaSparseAttention(_BoundAttention(layer, kv_cache))
        rotary = layer.rotary_emb
        kernel = rotary._forward_method

        def forward(positions, query, key=None, _kernel=kernel, **kwargs):
            # The CPU rotary kernel wants int64 and no -1; a padding row's rotation is a don't-care value.
            return _kernel(positions.long().clamp_min(0).flatten(), query, key, **kwargs)

        rotary._forward_method = forward
    return len(layers)


def _mark_outputs(value):
    if isinstance(value, torch.Tensor):
        return mark_device(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_mark_outputs(item) for item in value)
    return value


def _instrument_runner(runner, counters: Counters) -> None:
    """Time the runner's phases and mark what the model returns as device tensors.

    ``worker.forward`` (the model, one HPU graph on a card), ``worker.logits`` and ``worker.sampler`` are device
    work on a card; what is left of ``worker.sample_tokens`` and all of ``worker.execute_model`` is host time.
    """

    def timed(owner, name: str, phase: str, scope: str | None = None, mark: bool = False):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            with counters.phase(phase), counters.scoped(scope or counters.scope):
                result = original(*args, **kwargs)
            return _mark_outputs(result) if mark else result

        setattr(owner, name, wrapper)

    timed(runner.model, "forward", "worker.forward", scope="model", mark=True)
    timed(runner.model, "select_and_compute_logits", "worker.logits", scope="logits", mark=True)
    timed(runner.model, "select_compute_logits_and_greedy", "worker.logits", scope="logits", mark=True)
    timed(runner, "sampler", "worker.sampler", scope="sampler")
    timed(runner, "_update_states", "worker.update_states")
    timed(runner, "_prepare_inputs", "worker.prepare_inputs")
    timed(runner, "_prepare_sampling", "worker.prepare_sampling")
    timed(runner, "_get_prompts_and_decodes", "worker.partition")


@contextlib.contextmanager
def open_engine(ckpt_dir: Path | None, options: Options | None = None):
    """Start a server on the tiny checkpoint and yield its :class:`Engine`. One may be open at a time."""
    global _OPEN
    if _OPEN:
        raise RuntimeError("an engine is already open in this process (close it first: one model-parallel group, "
                           "one set of patched seams)")
    _OPEN = True
    options = options or Options()
    counters = Counters()
    saved_env = {name: os.environ.get(name) for name in set(ENV_VARS) | set(options.env)}
    saved_modules = {name: sys.modules.get(name) for name in (hpu_bucket.OPS_MODULE, hpu_bucket.HPU_ATTN_MODULE)}
    fd, init_file = tempfile.mkstemp(dir="/tmp")
    os.close(fd)
    stack = contextlib.ExitStack()
    try:
        for name in ENV_VARS:
            os.environ.pop(name, None)
        os.environ.update({"VLLM_HPU_FLASH_NEXT_QSA_MODE": "1", "VLLM_HPU_RECURRENT_STATE_POOL": "1",
                           "VLLM_SKIP_WARMUP": "true",
                           # a server in lazy mode does not compile the (disabled) defragmenter's swap either
                           "VLLM_DEFRAG_WITH_GRAPHS": "false"})
        os.environ.update({name: str(value) for name, value in options.env.items()})

        flash_next.register()
        # What the plugin's load_general_plugins hook installs in every process of a server.
        hpu_patches._patch_mamba_bind_kv_cache()
        hpu_patches._patch_free_blocks()
        hpu_patches._patch_get_num_layers_by_block_type()
        if options.stub_config is not None:
            ckpt_dir = Path(tempfile.mkdtemp(dir="/tmp"))
            stack.callback(shutil.rmtree, ckpt_dir, ignore_errors=True)
            shutil.copyfile(options.stub_config, ckpt_dir / "config.json")
        vllm_config = _engine_config(Path(ckpt_dir), options)
        cache_config = vllm_config.cache_config
        # APIServer, HpuPlatform.check_and_update_config: 128-token blocks, the compact-GDN rule and the pool slots.
        cache_config.block_size = 128
        if cache_config.mamba_cache_mode == "align":
            cache_config.mamba_block_size = 128
        _configure_recurrent_state(vllm_config)
        engine_config = copy.deepcopy(vllm_config)  # taken before the worker builds the model into its copy

        stack.enter_context(patch.object(runner_module, "async_h2d_copy", _cpu_h2d_copy(counters, options.count)))
        stack.enter_context(patch.object(input_batch_module, "async_h2d_copy", _cpu_h2d_copy(counters, False)))
        stack.enter_context(patch.object(input_batch_module, "async_h2d_update", _cpu_h2d_update(counters)))
        stack.enter_context(patch.object(runner_module, "is_fake_hpu", lambda: True))
        stack.enter_context(patch.object(torch.Tensor, "to", _cpu_tensor_to()))
        stack.enter_context(patch.object(runner_module, "get_attn_backend", lambda *a, **k: HPUAttentionBackendV1))
        stack.enter_context(patch.object(runner_module.htorch.utils.internal, "is_lazy", lambda: False))
        stack.enter_context(set_current_vllm_config(vllm_config))

        stub = options.stub_config is not None
        # One rank here whatever the config's tensor-parallel size: the runner only asks for its pipeline rank.
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{init_file}",
                                     local_rank=0, backend="gloo")
        initialize_model_parallel(1, 1)

        # Worker: init_device builds the runner, load_model builds the model.
        runner = runner_module.HPUModelRunner(vllm_config, device=DEVICE, is_driver_worker=True)
        if stub:
            from tests.unit_tests.startup import test_kv_cache_sizing as sizing
            model = sizing._build_model(vllm_config)
            for owner in model.model.layers:
                owner.attn.attn_backend = HPUAttentionBackendV1
                owner.attn.get_attn_backend = lambda: HPUAttentionBackendV1
            runner.model = StubModel(model)
            stack.enter_context(
                patch.object(runner_module, "maybe_set_mamba_kv_cache_groups_ids", lambda *args, **kwargs: None))
        else:
            from vllm.model_executor.model_loader import get_model
            model = get_model(vllm_config=vllm_config)
            if options.fp32:
                to_fp32(model)
            for module in model.modules():
                if isinstance(module, HpuQwen4ExpQSAAttention):
                    # On a card the platform selects this backend for the layer's Attention child: 128-token
                    # kernel blocks, so a scheduler block is split into several of them (virtual block splitting).
                    module.attn.attn_backend = HPUAttentionBackendV1
            runner.model = runner_module.HpuModelAdapter(model, vllm_config)
        runner._needs_block_table = runner_module.model_requires_block_table(runner.model)
        runner._block_table_max_blocks = -(-runner.max_model_len // runner.attn_block_size)
        runner.is_mm_optimized = False

        # Worker, after load_model: the block size for the backend and the hybrid page alignment.
        with patch.object(HpuPlatform, "_find_non_ssm_backend", classmethod(lambda cls, cfg: HPUAttentionBackendV1)):
            HpuPlatform.update_block_size_for_backend(vllm_config)
        cache_config.kv_cache_layout = "LBNHC"

        # EngineCore._initialize_kv_caches.
        specs = runner.get_kv_cache_spec()
        with patch.object(vllm.platforms, "_current_platform", HpuPlatform()):
            register_all_kvcache_specs(vllm_config)
        kv_cache_config = get_kv_cache_configs(vllm_config, [specs], [options.kv_cache_bytes])[0]
        kv_cache_config.kv_cache_layout = cache_config.kv_cache_layout
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config([kv_cache_config])

        real_zeros, real_tensor = torch.zeros, torch.tensor

        def cpu_zeros(*args, **kwargs):
            if stub:
                shape = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
                nbytes = torch.empty((), dtype=kwargs.get("dtype") or torch.float32).element_size()
                for dim in (shape if isinstance(shape, (tuple, list, torch.Size)) else (shape, )):
                    nbytes *= int(dim)
                if nbytes > options.meta_bytes:
                    kwargs["device"] = "meta"
            elif options.fp32 and kwargs.get("dtype") == torch.bfloat16:
                kwargs["dtype"] = torch.float32  # the fp32 model keeps fp32 caches, as cpu_model binds them
            return real_zeros(*args, **kwargs)

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == "hpu":
                kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)

        with patch.object(torch, "zeros", cpu_zeros), patch.object(torch, "tensor", cpu_tensor):
            runner.initialize_kv_cache(kv_cache_config)
        if not stub:
            _install_attention_seams(model)
        # Worker.compile_or_warm_up_model with VLLM_SKIP_WARMUP: the buckets and the (disabled) defragmenter.
        runner.warmup_model()
        sys.modules[hpu_bucket.OPS_MODULE] = hpu_bucket.fake_ops_module()
        sys.modules[hpu_bucket.HPU_ATTN_MODULE] = hpu_bucket.fake_hpu_attn_module()

        # The engine core is another process on a server: it has its own copy of the config, and the tail of
        # EngineCore._initialize_kv_caches writes the block count and the scheduler block size into that copy.
        engine_config.cache_config.kv_cache_layout = cache_config.kv_cache_layout
        engine_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        participating = [group.kv_cache_spec.block_size for group in scheduler_kv_cache_config.kv_cache_groups
                         if group.kv_cache_spec.prefix_cacheable]
        engine_config.cache_config.block_size = min(
            participating or [group.kv_cache_spec.block_size for group in scheduler_kv_cache_config.kv_cache_groups])
        update_kv_cache_capacity(engine_config, scheduler_kv_cache_config)
        engine_config.validate_block_size()

        # EngineCore.__init__: the scheduler and the step function.
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(scheduler_kv_cache_config, engine_config)
        scheduler_cls = engine_config.scheduler_config.get_scheduler_cls()
        from vllm.v1.structured_output import StructuredOutputManager
        scheduler = scheduler_cls(vllm_config=engine_config,
                                  kv_cache_config=scheduler_kv_cache_config,
                                  structured_output_manager=StructuredOutputManager(engine_config),
                                  include_finished_set=False,
                                  log_stats=False,
                                  block_size=scheduler_block_size,
                                  hash_block_size=hash_block_size)
        executor = LocalExecutor(runner, counters)
        core = object.__new__(EngineCore)
        core.vllm_config = engine_config
        core.log_stats = False
        core.scheduler = scheduler
        core.model_executor = executor
        core.use_spec_decode = engine_config.speculative_config is not None
        core.check_for_draft_tokens = core.use_spec_decode
        core.batch_queue_size = engine_config.max_concurrent_batches
        core.batch_queue = collections.deque(maxlen=core.batch_queue_size) if core.batch_queue_size > 1 else None
        core.is_ec_consumer = True
        core.is_pooling_model = False
        core.async_scheduling = engine_config.scheduler_config.async_scheduling
        core.aborts_queue = queue.Queue()
        core.step_fn = core.step if core.batch_queue is None else core.step_with_batch_queue

        if options.count:
            DeviceTensor.COUNTERS = counters
            _instrument_runner(runner, counters)
        yield Engine(vllm_config, runner, scheduler, core, executor, counters,
                     hash_block_size if options.prefix_caching else None)
    finally:
        _OPEN = False
        DeviceTensor.COUNTERS = None
        stack.close()
        with contextlib.suppress(Exception):
            destroy_model_parallel()
            destroy_distributed_environment()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(init_file)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
