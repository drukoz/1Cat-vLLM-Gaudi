# SPDX-License-Identifier: Apache-2.0
"""The head's side of a sampled step that draws from candidates: every rank's largest logits, their top-k, the
greedy pick (``vllm_gaudi.distributed.vocab_parallel``; the sampler stage is ``v1/sample/candidate_sampling.py``).

The ranks are played in one process: each rank's shard of one logits tensor behind a stand-in LM head, and an
all-gather that returns what the ranks would send (the whole-model run on gloo ranks is
``models/flash_next/test_collective_cuts.py``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()

from vllm_gaudi.distributed import vocab_parallel  # noqa: E402

VOCAB = 200
PADDING = 3


def distinct_logits(rows: int, seed: int) -> torch.Tensor:
    """``[rows, VOCAB]`` bf16, every row a permutation of -50.0, -49.5, ..., 49.5: no two logits of a row are equal."""
    gen = torch.Generator().manual_seed(seed)
    base = (torch.arange(VOCAB, dtype=torch.float32) - VOCAB / 2) * 0.5
    return torch.stack([base[torch.randperm(VOCAB, generator=gen)] for _ in range(rows)]).to(torch.bfloat16)


class Ranks:
    """The shards of ``logits`` on ``world`` ranks, and the collectives between them."""

    def __init__(self, logits: torch.Tensor, world: int, monkeypatch):
        self.logits, self.world = logits, world
        self.per = math.ceil(VOCAB / world)
        self.sent: dict[int, list[torch.Tensor]] = {}  # rank -> what it sent, collective by collective
        self.rank = 0
        self.replay = False
        self.call = 0
        monkeypatch.setattr(vocab_parallel, "get_tensor_model_parallel_world_size", lambda: world)
        monkeypatch.setattr(vocab_parallel, "tensor_model_parallel_all_gather", self.all_gather)

    def owner(self, rank: int):
        start, end = rank * self.per, min((rank + 1) * self.per, VOCAB)
        head = SimpleNamespace(tp_size=self.world, org_vocab_size=VOCAB, num_added_embeddings=0,
                               shard_indices=SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end,
                                                             added_vocab_start_index=VOCAB,
                                                             added_vocab_end_index=VOCAB))
        return SimpleNamespace(lm_head=head, logits_processor=lambda _head, hidden, skip_gather=True: hidden)

    def shard(self, rank: int) -> torch.Tensor:
        """Rank ``rank``'s logits: its real columns, then zero padding columns up to the common width."""
        start, end = rank * self.per, min((rank + 1) * self.per, VOCAB)
        rows = self.logits.shape[0]
        out = torch.zeros(rows, self.per + PADDING, dtype=self.logits.dtype)
        out[:, :end - start] = self.logits[:, start:end]
        return out

    def all_gather(self, tensor, dim=-1):
        self.call += 1
        if not self.replay:  # first pass: record what this rank sends in its n-th collective
            self.sent.setdefault(self.rank, []).append(tensor)
            return torch.cat([tensor] * self.world, dim=dim)
        return torch.cat([self.sent[r][self.call - 1] for r in range(self.world)], dim=dim)

    def run(self, rank: int, fn):
        self.rank, self.call = rank, 0
        return fn(self.owner(rank), self.shard(rank))

    def on_every_rank(self, fn):
        """``fn(owner, shard)`` on every rank with the real exchange: a recording pass, then the answer of each.

        What a rank sends in its n-th collective must not depend on what it received before (true of every
        function here: a rank's part is computed from its own shard)."""
        self.replay, self.sent = False, {}
        for rank in range(self.world):
            self.run(rank, fn)
        self.replay = True
        return [self.run(rank, fn) for rank in range(self.world)]


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("k", [4, 20, 32])
def test_candidates_hold_the_k_largest_logits_of_every_row(world, k, monkeypatch):
    logits = distinct_logits(5, seed=world * 100 + k)
    ranks = Ranks(logits, world, monkeypatch)
    results = ranks.on_every_rank(lambda owner, shard: vocab_parallel.vocab_parallel_candidates(owner, shard, k))
    want_values, want_ids = torch.topk(logits.float(), k, dim=-1)
    for values, ids in results:  # every rank holds the same candidates
        assert values.shape == ids.shape == (5, world * k)
        assert values.dtype == ids.dtype == torch.float32
        torch.testing.assert_close(values, results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(ids, results[0][1], rtol=0, atol=0)
        top, where = torch.topk(values, k, dim=-1)
        torch.testing.assert_close(top, want_values, rtol=0, atol=0)
        assert torch.equal(torch.gather(ids, -1, where).long(), want_ids)
        real = values > float("-inf")  # a candidate is a real token with its own logit, or -inf padding
        assert torch.equal(torch.gather(logits.float(), -1, ids.long())[real], values[real])
        assert (ids[~real] == 0).all()
        assert int(real.sum()) == 5 * sum(min(k, min((r + 1) * ranks.per, VOCAB) - r * ranks.per)
                                          for r in range(world))


@pytest.mark.parametrize("world", [1, 2, 8])
def test_top_k_equals_the_top_k_of_the_gathered_logits(world, monkeypatch):
    logits = distinct_logits(7, seed=world)
    ranks = Ranks(logits, world, monkeypatch)
    for k in (1, 5, 20):
        want_values, want_ids = torch.topk(logits.float(), k, dim=-1)
        for values, ids in ranks.on_every_rank(lambda owner, shard: vocab_parallel.vocab_parallel_topk(owner, shard, k)):
            assert ids.dtype == torch.int32
            torch.testing.assert_close(values, want_values, rtol=0, atol=0)
            assert torch.equal(ids.long(), want_ids)


def test_first_maximum_is_the_lowest_id_of_the_largest_value(monkeypatch):
    logits = distinct_logits(6, seed=3)
    logits[0, 150] = logits[0, 7] = 60.0  # a tie at the maximum, on two different ranks: the lower id wins
    logits[1, 199] = 61.0
    logits[2, 30] = logits[2, 29] = logits[2, 180] = 62.0
    want = [7, 199, 29] + logits.float().argmax(-1)[3:].tolist()
    for world in (1, 4, 8):
        ranks = Ranks(logits, world, monkeypatch)
        # from the unsorted per-rank candidates, and from the sorted top-k: the same pick on every rank
        for values, ids in ranks.on_every_rank(lambda owner, shard: vocab_parallel.vocab_parallel_candidates(owner, shard, 32)):
            assert vocab_parallel.first_maximum(values, ids).view(-1).tolist() == want
        for values, ids in ranks.on_every_rank(lambda owner, shard: vocab_parallel.vocab_parallel_topk(owner, shard, 20)):
            token = vocab_parallel.first_maximum(values, ids)
            assert token.shape == (6, 1) and token.dtype == torch.int32
            assert token.view(-1).tolist() == want
        # and what the arg max region gives for the same rows
        for token in ranks.on_every_rank(vocab_parallel.vocab_parallel_argmax):
            assert token.tolist() == want


def test_first_maximum_keeps_the_dtype_of_the_ids_and_never_casts_int64():
    values = torch.tensor([[5.0, 9.0, 9.0, 2.0], [3.0, 3.0, 3.0, 3.0], [7.0, 1.0, 2.0, 6.0]])
    ids = torch.tensor([[40, 30, 20, 10], [11, 5, 31, 41], [12, 22, 32, 42]])
    for dtype in (torch.int32, torch.int64, torch.float32):
        typed = ids.to(dtype)
        with _IntegerCasts() as watch:
            token = vocab_parallel.first_maximum(values, typed)
        assert token.dtype == dtype and token.shape == (3, 1)
        assert token.view(-1).tolist() == [20, 5, 12]
        assert watch.casts == []


def test_first_maximum_of_a_row_without_a_maximum_is_a_candidate():
    values = torch.tensor([[float("nan")] * 4, [1.0, 4.0, 2.0, 3.0]])
    ids = torch.tensor([[17, 3, 9, 4], [10, 11, 12, 13]], dtype=torch.int32)
    for dtype in (torch.int32, torch.int64):
        assert vocab_parallel.first_maximum(values, ids.to(dtype)).view(-1).tolist() == [17, 11]


def test_top_k_is_one_call_for_the_gathered_and_the_sharded_head(monkeypatch):
    """``top_k``: with the flag at 2 the per-rank candidates, otherwise the gathered logits; the same values and
    ids, int32, either way."""
    logits = distinct_logits(5, seed=17)
    want_values, want_ids = torch.topk(logits.float(), 20, dim=-1)
    ranks = Ranks(logits, 8, monkeypatch)
    monkeypatch.setattr(vocab_parallel, "unsupported_reason", lambda model: None)
    gathered = SimpleNamespace(compute_logits=lambda hidden: logits)
    for flag, source in (("2", "ranks"), ("1", "gathered"), ("0", "gathered")):
        monkeypatch.setenv("VLLM_HPU_VOCAB_PARALLEL_LOGITS", flag)
        if source == "ranks":
            results = ranks.on_every_rank(lambda owner, shard: vocab_parallel.top_k(owner, shard, 20))
            assert ranks.call == 1  # one collective
        else:
            ranks.call = 0
            results = [vocab_parallel.top_k(gathered, None, 20)]
            assert ranks.call == 0  # compute_logits did the gathering
        for values, ids in results:
            assert values.dtype == torch.float32 and ids.dtype == torch.int32
            torch.testing.assert_close(values, want_values, rtol=0, atol=0)
            assert torch.equal(ids.long(), want_ids)


class _IntegerCasts(TorchDispatchMode):
    """Records every dtype cast whose input is an int64 tensor (card rule 9: such a cast in the graph that made
    the ids returned every second id on the card)."""

    def __init__(self):
        super().__init__()
        self.casts = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.__name__ if hasattr(func, "__name__") else str(func)
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if (source is not None and source.dtype == torch.int64 and isinstance(out, torch.Tensor)
                and out.dtype != torch.int64 and out.shape == source.shape
                and any(word in name for word in ("to", "_to_copy", "type", "float", "int"))):
            self.casts.append((name, str(out.dtype)))
        return out


@pytest.mark.parametrize("world", [1, 8])
def test_no_int64_id_is_cast_on_the_way_to_a_token(world, monkeypatch):
    logits = distinct_logits(4, seed=21)
    ranks = Ranks(logits, world, monkeypatch)

    def step(owner, shard):
        token = vocab_parallel.vocab_parallel_argmax(owner, shard)
        values, ids = vocab_parallel.vocab_parallel_candidates(owner, shard, 32)
        top_values, top_ids = vocab_parallel.vocab_parallel_topk(owner, shard, 20)
        return token, top_ids, vocab_parallel.first_maximum(top_values, top_ids), vocab_parallel.first_maximum(values, ids)

    ranks.on_every_rank(step)  # records what every rank sends, outside the watch
    with _IntegerCasts() as watch:
        token, top_ids, first, first_unsorted = ranks.run(0, step)
    assert watch.casts == []
    assert token.dtype == top_ids.dtype == first.dtype == torch.int32
    assert first.view(-1).tolist() == first_unsorted.view(-1).tolist() == token.tolist()


def test_the_watch_sees_the_casts_it_is_there_for():
    x = distinct_logits(3, seed=1).float()
    with _IntegerCasts() as watch:
        x.argmax(dim=-1).to(torch.int32)
        torch.topk(x, 4, dim=-1).indices.to(torch.float32)
        x.argmax(dim=-1).int()
        torch.gather(x, -1, torch.topk(x, 4, dim=-1).indices)  # ids used as they are: no cast
    assert [dtype for _, dtype in watch.casts] == ["torch.int32", "torch.float32", "torch.int32"]


def test_shard_reductions_run_on_the_heads_own_dtype(monkeypatch):
    """The shard's logits reach the reductions in the head's dtype; only reduced values become fp32.

    On the card a ``.to(torch.float32)`` of the whole shard right after the head's matmul is fused into the GEMM,
    which then hands out its fp32 accumulator (job 00254): the arg max would compare other values than the gathered
    path's bf16 logits. So no op may convert a ``[rows, shard]`` tensor to fp32 before the reduction."""
    logits = distinct_logits(rows=3, seed=11)
    ranks = Ranks(logits, 4, monkeypatch)
    owner = ranks.owner(1)
    shard_logits, start, count = vocab_parallel._shard_logits(owner, ranks.shard(1))
    assert shard_logits.dtype == torch.bfloat16 and (start, count) == (ranks.per, ranks.per)
    width = ranks.per + PADDING
    seen = []

    class WholeShardCasts(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if (func.__name__.startswith(("_to_copy", "to.")) and isinstance(out, torch.Tensor)
                    and out.dtype == torch.float32 and out.dim() == 2 and out.shape[-1] >= ranks.per - 1
                    and out.shape[-1] <= width and args and args[0].dtype == torch.bfloat16):
                seen.append(tuple(out.shape))
            return out

    with WholeShardCasts():
        for fn in (vocab_parallel.vocab_parallel_argmax,
                   lambda own, shard: vocab_parallel.vocab_parallel_candidates(own, shard, 8)):
            ranks.on_every_rank(fn)
    assert seen == [], seen
