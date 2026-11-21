# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M1 (VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS): a verification step's inputs made from the previous
verification's (``flash_next_async_spec.VerifyInputAdvance``) equal the ones the runner makes on the host from its
books, step after step, with random acceptance, padded rows and the pad block's slots. The host side here uses the
runner's own helpers (``build_ple_inputs``, ``fill_pad_slots``, ``accepted_counts``)."""

import random

import numpy as np
import pytest
import torch

from vllm_gaudi.v1.worker.flash_next_async_spec import (ChainOut, VerifyInputAdvance, VerifyInputs,
                                                        use_scheduled_drafts, stand_in_chain)
from vllm_gaudi.v1.worker.flash_next_decode_metadata import fill_pad_slots
from vllm_gaudi.v1.worker.flash_next_inputs import build_ple_inputs
from vllm_gaudi.v1.worker.recurrent_spec import accepted_counts

BS = 4  # KV block size
PAD_BLOCK = 0
EOS = 7
NGRAM = 4  # context of 3 tokens


class Books:
    """The host's books of a batch of requests: token history, computed counts, block tables."""

    def __init__(self, rng, num_reqs, k, max_blocks=40):
        self.rng, self.k = rng, k
        self.req_ids = [f"r{i}" for i in range(num_reqs)]
        self.history = [[rng.randrange(10, 99) for _ in range(rng.randrange(1, 9))] for _ in range(num_reqs)]
        self.next_input = [rng.randrange(10, 99) for _ in range(num_reqs)]  # the bonus token of the last step
        self.drafts = [[rng.randrange(10, 99) for _ in range(k)] for _ in range(num_reqs)]
        self.accepted_by_req = {}
        blocks = list(range(1, 1 + num_reqs * max_blocks))
        rng.shuffle(blocks)
        self.table = torch.tensor([blocks[i * max_blocks:(i + 1) * max_blocks] for i in range(num_reqs)],
                                  dtype=torch.int32)

    def host_inputs(self, padded):
        """The runner's host-made inputs of the verification step over the books."""
        n, k = len(self.req_ids), self.k
        width = 1 + k
        ctx = [len(h) for h in self.history]
        positions = torch.zeros((padded, width), dtype=torch.int64)
        ids = torch.zeros((padded, width), dtype=torch.int32)
        for r in range(n):
            positions[r] = torch.arange(ctx[r], ctx[r] + width)
            ids[r] = torch.tensor([self.next_input[r]] + self.drafts[r], dtype=torch.int32)
        block_number = torch.full((padded, width), PAD_BLOCK, dtype=torch.int32)
        block_number[:n] = torch.gather(self.table[:n], 1, positions[:n] // BS).to(torch.int32)
        slots = block_number * BS + positions % BS
        fill_pad_slots(slots, [width] * n + [0] * (padded - n), PAD_BLOCK * BS, BS)
        num_accepted = torch.tensor(accepted_counts(self.req_ids, self.accepted_by_req, padded), dtype=torch.int32)
        tokens = np.zeros((n, max(ctx) + width), dtype=np.int32)
        for r in range(n):
            tokens[r, :ctx[r]] = self.history[r]
        _, context = build_ple_inputs(torch.from_numpy(tokens), np.asarray(ctx), list(range(n)), padded, width, EOS,
                                      NGRAM)
        return VerifyInputs(ids.reshape(-1, 1), positions.to(torch.int32).reshape(-1, 1), slots.reshape(-1, 1),
                            num_accepted, context)

    def verify(self, padded):
        """One verification step: random acceptance; returns the chain's output and updates the books."""
        n, k = len(self.req_ids), self.k
        accepted = [self.rng.randrange(0, k + 1) for _ in range(n)]
        committed = [[self.next_input[r]] + self.drafts[r][:accepted[r]] for r in range(n)]
        bonus = [self.rng.randrange(10, 99) for _ in range(n)]
        new_drafts = [[self.rng.randrange(10, 99) for _ in range(k)] for _ in range(n)]
        acc = torch.zeros(padded, dtype=torch.int32)
        acc[:n] = torch.tensor(accepted, dtype=torch.int32)
        next_ids = torch.zeros((padded, 1 + k), dtype=torch.int32)
        next_ids[:n] = torch.tensor([[bonus[r]] + new_drafts[r] for r in range(n)], dtype=torch.int32)
        for r in range(n):
            self.history[r] += committed[r]
            self.accepted_by_req[self.req_ids[r]] = accepted[r] + 1
        self.next_input, self.drafts = bonus, new_drafts
        return ChainOut(accepted=acc, next_ids=next_ids, num_decodes=n, k=k), committed, bonus


def _assert_equal(made, host):
    for name in ("token_ids", "positions", "slot_mapping", "num_accepted", "ngram_context"):
        assert torch.equal(getattr(made, name).to(torch.int64), getattr(host, name).to(torch.int64)), name


@pytest.mark.parametrize("repeated", [False, True], ids=["table", "repeated_table"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_device_made_inputs_equal_the_host_made_ones_step_after_step(seed, k, repeated):
    rng = random.Random(seed * 10 + k)
    num_reqs = rng.randrange(1, 6)
    padded = num_reqs + rng.randrange(0, 3)
    books = Books(rng, num_reqs, k)
    advance = VerifyInputAdvance()
    step = 1
    advance.record(step, books.req_ids, num_reqs, padded, k, books.host_inputs(padded))
    for _ in range(12):
        chain, _, _ = books.verify(padded)
        advance.set_chain(step, chain)
        step += 1
        if repeated:  # the sparse decode metadata's table: each request's row over its 1 + k rows, pad rows after
            table = torch.full((padded * (1 + k), books.table.shape[1]), PAD_BLOCK, dtype=torch.int32)
            table[:num_reqs * (1 + k)] = books.table.repeat_interleave(1 + k, dim=0)
            made = advance.advance(step, books.req_ids, num_reqs, padded, k, table, BS, PAD_BLOCK * BS, EOS, True,
                                   table_row_stride=1 + k)
        else:
            made = advance.advance(step, books.req_ids, num_reqs, padded, k, books.table, BS, PAD_BLOCK * BS, EOS,
                                   True)
        host = books.host_inputs(padded)
        assert made is not None
        _assert_equal(made, host)
        assert advance.check(made, host) == {}
        advance.record(step, books.req_ids, num_reqs, padded, k, made)
    assert advance.stats.advanced == 12 and advance.stats.mismatched_steps == 0


def test_a_step_that_does_not_continue_the_last_one_keeps_the_host_inputs():
    rng = random.Random(1)
    books = Books(rng, 3, 2)
    advance = VerifyInputAdvance()
    advance.record(1, books.req_ids, 3, 4, 2, books.host_inputs(4))
    chain, _, _ = books.verify(4)
    args = (books.table, BS, PAD_BLOCK * BS, EOS, True)
    advance.set_chain(1, chain)
    assert advance.advance(3, books.req_ids, 3, 4, 2, *args) is None  # a step in between
    assert advance.advance(2, list(reversed(books.req_ids)), 3, 4, 2, *args) is None  # other rows
    assert advance.advance(2, books.req_ids, 3, 8, 2, *args) is None  # another padded batch
    assert advance.advance(2, books.req_ids, 3, 4, 3, *args) is None  # another k
    advance.set_chain(1, None)
    assert advance.advance(2, books.req_ids, 3, 4, 2, *args) is None  # the step left no acceptance
    advance.set_chain(1, chain)
    assert advance.advance(2, books.req_ids, 3, 4, 2, *args) is not None


def test_the_check_counts_what_differs():
    rng = random.Random(2)
    books = Books(rng, 2, 2)
    advance = VerifyInputAdvance()
    host = books.host_inputs(3)
    wrong = VerifyInputs(host.token_ids.clone(), host.positions + 1, host.slot_mapping, host.num_accepted,
                         host.ngram_context)
    assert advance.check(wrong, host) == {"positions": host.positions.numel()}
    assert advance.stats.mismatched_steps == 1 and advance.stats.mismatches["positions"] == 1


def test_the_stand_in_takes_the_host_read_and_the_scheduled_drafts():
    drafts = torch.tensor([[11, 12], [21, 22], [31, 32]], dtype=torch.int32)  # rows past num_decodes are prefills
    out = stand_in_chain([[5, 6, 9], [8]], drafts, padded=4, k=2)
    assert out.stand_in and out.num_decodes == 2
    assert out.accepted.tolist() == [2, 0, 0, 0]
    assert out.next_ids.tolist() == [[9, 11, 12], [8, 21, 22], [0, 0, 0], [0, 0, 0]]
    assert use_scheduled_drafts(out, ["a", "b"], {"a": [41, 42], "b": [51, 52]})
    assert out.next_ids[:2].tolist() == [[9, 41, 42], [8, 51, 52]]
    assert not use_scheduled_drafts(out, ["a", "b"], {"a": [-1, -1], "b": [51, 52]})  # async placeholders
    assert out.next_ids[:2].tolist() == [[9, 41, 42], [8, 51, 52]]
    theirs = ChainOut(accepted=out.accepted, next_ids=out.next_ids.clone(), num_decodes=2, k=2)
    original = theirs.next_ids
    assert use_scheduled_drafts(theirs, ["a", "b"], {"a": [61, 62], "b": [71, 72]})
    assert theirs.next_ids[:2].tolist() == [[9, 61, 62], [8, 71, 72]]
    assert original[:2].tolist() == [[9, 41, 42], [8, 51, 52]]  # the chain's own tensor is not written


def test_trim_verified_keeps_what_parse_output_keeps():
    from vllm_gaudi.v1.worker.flash_next_async_spec import trim_verified
    assert trim_verified([5, 7, 9], 100) == [5, 7, 9]
    assert trim_verified([5, -1, -1], 100) == [5]
    assert trim_verified([5, 120, 3], 100) == [5, 3]  # an id past the vocabulary is dropped, as parse_output does


def test_early_books_seed_once_and_count_every_applied_step():
    from vllm_gaudi.v1.worker.flash_next_async_spec import PendingBooks, SpecEarlyBooks
    books = SpecEarlyBooks()
    first = PendingBooks(1, ("a", "b"), torch.tensor([[4, 5, -1], [6, -1, -1]]), None, {"a": 10, "b": 20})
    second = PendingBooks(2, ("a", "b"), torch.tensor([[7, 8, 9], [1, 2, -1]]), None, {"a": 99})  # a seeded already
    books.push(first)
    books.push(second)
    assert books.true_computed == {"a": 10, "b": 20}
    assert books.apply(books.pending.popleft(), 100) == [[4, 5], [6]]
    assert books.true_computed == {"a": 12, "b": 21}
    assert books.apply(books.pending.popleft(), 100) == [[7, 8, 9], [1, 2]]
    assert books.true_computed == {"a": 15, "b": 23}
