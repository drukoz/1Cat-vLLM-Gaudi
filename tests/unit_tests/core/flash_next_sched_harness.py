# SPDX-License-Identifier: Apache-2.0
"""The engine's real scheduler over the Flash-Next KV-cache layout of a card start, played step by step on CPU.

Plain module (no pytest fixtures): tests and probes import ``Settings``, ``Sim`` and the token helpers.

What is real and what is a stand-in:

* ``server_start(tp)`` runs the CPU server start of ``tests/unit_tests/startup/test_kv_cache_sizing.py``: the
  model's real ``config.json``, the platform hooks (recurrent pool slots, attention block size), the runner's
  ``get_kv_cache_spec`` and the engine's grouping. It gives the engine config and the scheduler-side
  ``KVCacheConfig`` of the rental layouts: TP8 with 32 sequences (scheduler block 512, 896 pool slots, 8,781
  attention blocks) and TP4 with 16 sequences (block 896).
* ``Sim`` builds ``vllm.v1.core.sched.scheduler.Scheduler`` (async scheduling is off for Flash-Next) the way
  ``EngineCore.__init__`` does (``resolve_kv_cache_block_sizes``, the request block hasher) with the plugin's
  ``HpuRecurrentPoolCoordinator`` installed through the factory patch of ``vllm_gaudi.utils``, as on a server.
* The model runner is a stand-in that does what ``HPUModelRunner`` does with a ``SchedulerOutput``: it keeps
  each request's per-group block-id rows (``_update_states``: the rows of a new request, appended ids for a
  running one), runs the prompt chunks first and the decode rows after them (``sample_tokens``), and for every
  recurrent group reads the state slot ``row[cdiv(context, B) - 1]`` and writes ``row[cdiv(context + n, B) - 1]``
  (``compute_prefix_caching_block_indices``, the real function, and ``prepare_mamba_state_idxs``). Instead of
  tensors the slots hold the identity of the token prefix whose state they contain, so a read of a slot that
  holds another prefix (a stale or shared state) is recorded in ``Sim.violations``. Attention blocks are
  modelled the same way per scheduler block, checked on every prompt chunk.
* Every finished prompt or decode row "samples" one token: ``Sim.outputs[request_id]``.

``Sim.step()`` is one engine step (``EngineCore.step``): schedule, run, ``update_from_output``.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import os  # noqa: E402

os.environ["TMPDIR"] = "/tmp"

import copy  # noqa: E402
import dataclasses  # noqa: E402
import tempfile  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import vllm_gaudi.utils  # noqa: E402, F401  (installs the recurrent-pool coordinator factory, as on a card)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.utils.math_utils import cdiv  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes)
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from tests.unit_tests.startup import test_kv_cache_sizing as sizing  # noqa: E402
from vllm_gaudi.v1.core.recurrent_pool_coordinator import HpuRecurrentPoolCoordinator  # noqa: E402
from vllm_gaudi.v1.core.recurrent_state_pool import ENV_MODE, ENV_SLOTS  # noqa: E402
from vllm_gaudi.v1.worker.hpu_model_runner import compute_prefix_caching_block_indices  # noqa: E402

EOS_TOKEN_ID = 248044  # never produced by the stand-in runner
SHARED_TOKENS = 16384  # the workload's shared system block (tools/flash_next/rental/workload.py)
RENTAL_BUDGET = 8192  # tools/flash_next/rental/serve.py: --max-num-batched-tokens
RENTAL_THRESHOLD = 2048  # --long-prefill-token-threshold
GEOMETRIES = {8: sizing.FN_TP8, 4: sizing.FN_TP4}

_STARTS: dict[int, sizing.Start] = {}


def server_start(tp: int = 8) -> sizing.Start:
    """The CPU server start of the rental layout at ``tp`` (8: 32 sequences, 4: 16 sequences), run once."""
    if tp not in _STARTS:
        model_dir = Path(tempfile.mkdtemp(prefix=f"fn-sched-tp{tp}-", dir="/tmp"))
        _STARTS[tp] = sizing.start(GEOMETRIES[tp], model_dir)
    return _STARTS[tp]


@dataclasses.dataclass(frozen=True)
class Settings:
    """One server's scheduling-relevant settings. The defaults are the rental's variant A (TP8 + EP8)."""
    tp: int = 8
    max_num_batched_tokens: int = RENTAL_BUDGET
    long_prefill_token_threshold: int = RENTAL_THRESHOLD  # 0 = off
    # --prefix-cache-retention-interval: 0 is vLLM's default (only the replay boundary and shared-prefix
    # junctions of a recurrent group are kept), a multiple of the block size keeps one state per interval
    # (the block size itself: every block boundary), None keeps every one as well.
    retention_interval: int | None = 0
    max_num_seqs: int | None = None  # None: the layout's (32 at TP8, 16 at TP4)
    pool_slots: int | None = None  # None: what the platform hook resolved for the layout
    num_attention_blocks: int | None = None  # None: what the server start computed


# -- token helpers -------------------------------------------------------------------------------------------


def shared_block(num_tokens: int = SHARED_TOKENS) -> list[int]:
    """The token ids every agent's prompt starts with."""
    return list(range(1_000, 1_000 + num_tokens))


def own_tokens(agent: int, num_tokens: int, salt: int = 0) -> list[int]:
    """``num_tokens`` ids no other agent (or salt) has."""
    base = 10_000_000 + agent * 1_000_000 + salt * 200_000_000
    return list(range(base, base + num_tokens))


def agent_prompt(agent: int, total_tokens: int, shared_tokens: int = SHARED_TOKENS) -> list[int]:
    """The first-turn prompt of one agent: the shared block, then its own files."""
    return shared_block(shared_tokens) + own_tokens(agent, total_tokens - shared_tokens)


def output_token(request_id: str, index: int) -> int:
    """The ``index``-th token the stand-in runner samples for a request (deterministic, never EOS)."""
    return 500_000_000 + (zlib.crc32(request_id.encode()) % 100_000) * 2_048 + index


# -- records -------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Admission:
    step: int
    prompt_tokens: int
    cached_tokens: int  # request.prefill_stats.num_cached_tokens: the API's usage.prompt_tokens_details.cached_tokens
    shared_prefix_boundary: int
    first_chunk: int
    deferred_steps: int  # schedule() calls in which the request was first in the waiting queue and not admitted


@dataclasses.dataclass
class StepRecord:
    index: int
    prefill: dict[str, int]  # request id -> prompt tokens computed in this step (the runner's prompt batches)
    decode: list[str]  # request ids in the decode batch
    admitted: list[str]
    finished: list[str]
    recurrent_free: int
    attention_free: int

    @property
    def prefill_tokens(self) -> int:
        return sum(self.prefill.values())


@dataclasses.dataclass
class Violation:
    step: int
    kind: str
    request_id: str
    group: int
    slot: int
    detail: str


class Sim:
    """A scheduler plus the runner stand-in. See the module docstring."""

    def __init__(self, settings: Settings = Settings(), check_attention: bool = True) -> None:
        self.settings = settings
        start = server_start(settings.tp)
        self.start = start
        vllm_config = copy.copy(start.vllm_config)
        vllm_config.scheduler_config = copy.copy(start.vllm_config.scheduler_config)
        vllm_config.cache_config = copy.copy(start.vllm_config.cache_config)
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.max_num_batched_tokens = settings.max_num_batched_tokens
        scheduler_config.max_num_scheduled_tokens = None
        scheduler_config.long_prefill_token_threshold = settings.long_prefill_token_threshold
        if settings.max_num_seqs is not None:
            scheduler_config.max_num_seqs = settings.max_num_seqs
        kv_cache_config = dataclasses.replace(start.scheduler_config,
                                              prefix_cache_retention_interval=settings.retention_interval)
        if settings.num_attention_blocks is not None:
            kv_cache_config = dataclasses.replace(kv_cache_config, num_blocks=settings.num_attention_blocks)
        vllm_config.cache_config.prefix_cache_retention_interval = settings.retention_interval
        vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks  # EngineCore._initialize_kv_caches
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.pool_slots = start.pool_slots if settings.pool_slots is None else settings.pool_slots

        # EngineCore.__init__
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        saved = {name: os.environ.get(name) for name in (ENV_MODE, ENV_SLOTS)}
        os.environ[ENV_MODE] = "1"
        os.environ[ENV_SLOTS] = str(self.pool_slots)
        try:
            self.scheduler = Scheduler(vllm_config=vllm_config,
                                       kv_cache_config=kv_cache_config,
                                       structured_output_manager=StructuredOutputManager(vllm_config),
                                       include_finished_set=False,
                                       log_stats=True,
                                       block_size=self.block_size,
                                       hash_block_size=self.hash_block_size)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.coordinator = self.scheduler.kv_cache_manager.coordinator
        assert isinstance(self.coordinator, HpuRecurrentPoolCoordinator), type(self.coordinator)
        hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        self._hasher = get_request_block_hasher(self.hash_block_size, hash_fn)

        groups = kv_cache_config.kv_cache_groups
        self.recurrent_groups = [gid for gid, group in enumerate(groups) if isinstance(group.kv_cache_spec, MambaSpec)]
        attention = [gid for gid, group in enumerate(groups) if isinstance(group.kv_cache_spec, FullAttentionSpec)]
        assert len(attention) == 1, attention  # the runner turns ONE block table into attention slots
        self.attention_group = attention[0]
        self.mamba_block_size = int(vllm_config.cache_config.mamba_block_size)
        self.check_attention = check_attention

        self.requests: dict[str, Request] = {}
        self.outputs: dict[str, list[int]] = {}
        self.admissions: dict[str, Admission] = {}
        self.finished_at: dict[str, int] = {}
        self.steps: list[StepRecord] = []
        self.violations: list[Violation] = []
        self.evicted_recurrent: list[tuple[int, int, int | None]] = []  # (step, block id, tokens of its hash)
        self._rows: dict[str, list[list[int]]] = {}  # the runner's block-id rows, one list per group
        self._recurrent_state: dict[tuple[int, int], object] = {}  # (group, slot) -> prefix identity
        self._attention_block: dict[int, object] = {}  # attention block id -> prefix identity at its fill point
        self._deferred: dict[str, int] = {}
        self._watch_recurrent_evictions()

    # -- requests ------------------------------------------------------------------------------------------

    def add(self, request_id: str, prompt_token_ids: list[int], max_tokens: int = 1024) -> Request:
        """Queue a request, as ``EngineCore.add_request`` does."""
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        params.update_from_generation_config({}, eos_token_id=EOS_TOKEN_ID)
        request = Request(request_id=request_id,
                          prompt_token_ids=list(prompt_token_ids),
                          mm_features=None,
                          sampling_params=params,
                          pooling_params=None,
                          block_hasher=self._hasher)
        self.requests[request_id] = request
        self.outputs[request_id] = []
        self.scheduler.add_request(request)
        return request

    def position(self, request_id: str) -> int:
        """Tokens of the request the scheduler has handed to the runner so far (prefix hits included)."""
        return self.requests[request_id].num_computed_tokens

    def is_finished(self, request_id: str) -> bool:
        return request_id in self.finished_at

    # -- one engine step -----------------------------------------------------------------------------------

    def step(self) -> StepRecord:
        scheduler = self.scheduler
        index = len(self.steps)
        waiting_head = scheduler.waiting.peek_request().request_id if scheduler.waiting else None
        output = scheduler.schedule()
        scheduled = dict(output.num_scheduled_tokens)
        if waiting_head is not None and waiting_head not in scheduled:
            self._deferred[waiting_head] = self._deferred.get(waiting_head, 0) + 1

        admitted = []
        for new in output.scheduled_new_reqs:
            self._rows[new.req_id] = [list(ids) for ids in new.block_ids]
            request = self.requests[new.req_id]
            cached = request.prefill_stats.num_cached_tokens
            assert cached == request.num_computed_tokens - scheduled[new.req_id], (new.req_id, cached)
            self.admissions[new.req_id] = Admission(step=index,
                                                    prompt_tokens=request.num_prompt_tokens,
                                                    cached_tokens=cached,
                                                    shared_prefix_boundary=request.shared_prefix_boundary,
                                                    first_chunk=scheduled[new.req_id],
                                                    deferred_steps=self._deferred.get(new.req_id, 0))
            admitted.append(new.req_id)
        cached_reqs = output.scheduled_cached_reqs
        for position, request_id in enumerate(cached_reqs.req_ids):
            new_block_ids = cached_reqs.new_block_ids[position]
            assert request_id not in cached_reqs.resumed_req_ids, f"{request_id} was preempted"
            if new_block_ids is not None:
                for row, new_ids in zip(self._rows[request_id], new_block_ids):
                    row.extend(new_ids)

        # HPUModelRunner.sample_tokens: every prompt batch first (one request each), then the decode batch.
        prefill: dict[str, int] = {}
        decode: list[str] = []
        for request_id, num_tokens in scheduled.items():
            request = self.requests[request_id]
            context = request.num_computed_tokens - num_tokens  # _update_after_schedule already advanced it
            if context < request.num_prompt_tokens:
                prefill[request_id] = num_tokens
            else:
                assert num_tokens == 1, (request_id, num_tokens)
                decode.append(request_id)
        for request_id in list(prefill) + decode:
            request = self.requests[request_id]
            num_tokens = scheduled[request_id]
            self._run_rows(index, request, request.num_computed_tokens - num_tokens, num_tokens,
                           is_prompt=request_id in prefill)

        request_ids = list(scheduled)
        sampled: list[list[int]] = []
        for request_id in request_ids:
            request = self.requests[request_id]
            if request.is_prefill_chunk:
                sampled.append([])
            else:
                token = output_token(request_id, len(self.outputs[request_id]))
                self.outputs[request_id].append(token)
                sampled.append([token])
        runner_output = ModelRunnerOutput(req_ids=request_ids,
                                          req_id_to_index={request_id: i for i, request_id in enumerate(request_ids)},
                                          sampled_token_ids=sampled,
                                          logprobs=None,
                                          prompt_logprobs_dict={},
                                          pooler_output=[])
        scheduler.update_from_output(output, runner_output)
        finished = [request_id for request_id in request_ids if self.requests[request_id].is_finished()]
        for request_id in finished:
            self.finished_at[request_id] = index
            self._rows.pop(request_id, None)
        record = StepRecord(index=index,
                            prefill=prefill,
                            decode=decode,
                            admitted=admitted,
                            finished=finished,
                            recurrent_free=self.coordinator.recurrent_pool.get_num_free_blocks(),
                            attention_free=self.coordinator.block_pool.get_num_free_blocks())
        self.steps.append(record)
        return record

    def run(self, max_steps: int, until=None) -> int:
        """Step until ``until(sim)`` is true, nothing is left to do, or ``max_steps`` steps; returns the steps run."""
        for count in range(max_steps):
            if until is not None and until(self):
                return count
            if not self.scheduler.has_requests():
                return count
            self.step()
        return max_steps

    def run_until_finished(self, request_ids, max_steps: int) -> int:
        wanted = list(request_ids)
        steps = self.run(max_steps, until=lambda sim: all(sim.is_finished(r) for r in wanted))
        assert all(self.is_finished(r) for r in wanted), [r for r in wanted if not self.is_finished(r)]
        return steps

    # -- the runner stand-in -------------------------------------------------------------------------------

    def prefix_identity(self, request: Request, num_tokens: int):
        """What identifies ``request``'s first ``num_tokens`` tokens: the engine's chained hash of the full hash
        blocks plus the tokens after them. Equal for two requests exactly when the prefixes are equal."""
        full = num_tokens // self.hash_block_size
        tail = tuple(request.all_token_ids[full * self.hash_block_size:num_tokens])
        return (request.block_hashes[full - 1] if full else None, tail)

    def state_slots(self, request_id: str, context: int, num_tokens: int) -> dict[int, tuple[int, int]]:
        """(load slot, store slot) per recurrent group for a batch row, as ``_form_prefill_batch`` and
        ``_create_decode_input_data`` compute them under prefix caching."""
        last_computed, _first, last_scheduled = compute_prefix_caching_block_indices(1, [context], [num_tokens],
                                                                                    self.mamba_block_size)
        rows = self._rows[request_id]
        load, store = int(last_computed[0]), int(last_scheduled[0])
        return {gid: (rows[gid][load], rows[gid][store]) for gid in self.recurrent_groups}

    def _run_rows(self, step: int, request: Request, context: int, num_tokens: int, is_prompt: bool) -> None:
        request_id = request.request_id
        after = self.prefix_identity(request, context + num_tokens)
        before = self.prefix_identity(request, context) if context else None
        managers = self.coordinator.single_type_managers
        for gid, (load_slot, store_slot) in self.state_slots(request_id, context, num_tokens).items():
            if context:  # has_initial_state: the layer reads the load slot
                if load_slot == 0:
                    self._violate(step, "null_state", request_id, gid, load_slot,
                                  f"context {context}: the load slot is the null block")
                elif self._recurrent_state.get((gid, load_slot)) != before:
                    self._violate(step, "stale_state", request_id, gid, load_slot,
                                  f"context {context}: slot does not hold this request's state at {context}")
            owned = managers[gid].req_to_blocks[request_id]
            index = cdiv(context + num_tokens, self.mamba_block_size) - 1
            block = owned[index]
            if block.block_id != store_slot or block.is_null:
                self._violate(step, "foreign_store", request_id, gid, store_slot,
                              f"the scheduler's block at index {index} is {block.block_id}")
            elif block.ref_cnt != 1:
                self._violate(step, "shared_store", request_id, gid, store_slot,
                              f"write into a block with {block.ref_cnt} references")
            self._recurrent_state[(gid, store_slot)] = after
        self._run_attention(step, request, context, num_tokens, is_prompt)

    def _run_attention(self, step: int, request: Request, context: int, num_tokens: int, is_prompt: bool) -> None:
        row = self._rows[request.request_id][self.attention_group]
        size = self.block_size
        if is_prompt and self.check_attention:
            for index in range(cdiv(context, size)):
                expected = self.prefix_identity(request, min((index + 1) * size, context))
                if self._attention_block.get(row[index]) != expected:
                    self._violate(step, "stale_kv", request.request_id, self.attention_group, row[index],
                                  f"context block {index} does not hold this request's tokens")
                    break
        end = context + num_tokens
        for index in range(context // size, cdiv(end, size)):
            self._attention_block[row[index]] = self.prefix_identity(request, min((index + 1) * size, end))

    def _violate(self, step: int, kind: str, request_id: str, group: int, slot: int, detail: str) -> None:
        self.violations.append(Violation(step, kind, request_id, group, slot, detail))

    def _watch_recurrent_evictions(self) -> None:
        pool = self.coordinator.recurrent_pool
        evict = pool._maybe_evict_cached_block

        def watched(block):
            tokens = block.block_hash_num_tokens if block.block_hash is not None else None
            evicted = evict(block)
            if evicted:
                self.evicted_recurrent.append((len(self.steps), block.block_id, tokens))
            return evicted

        pool._maybe_evict_cached_block = watched

    # -- views for tests ------------------------------------------------------------------------------------

    def cached_recurrent_states(self, request: Request) -> list[int]:
        """Token positions at which EVERY recurrent group holds a cached state for ``request``'s prefix."""
        pool = self.coordinator.recurrent_pool
        positions = []
        hashes_per_block = self.mamba_block_size // self.hash_block_size
        for index in range(len(request.block_hashes) // hashes_per_block):
            block_hash = request.block_hashes[(index + 1) * hashes_per_block - 1]
            if pool.get_cached_block(block_hash, self.recurrent_groups) is not None:
                positions.append((index + 1) * self.mamba_block_size)
        return positions

    def prefill_tokens_of(self, request_id: str) -> int:
        """Prompt tokens the runner computed for a request (what was not a prefix hit, and any recomputation)."""
        return sum(record.prefill.get(request_id, 0) for record in self.steps)
