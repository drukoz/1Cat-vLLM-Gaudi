# SPDX-License-Identifier: Apache-2.0
"""The chained MTP draft (VLLM_HPU_MTP_DRAFT_CHAIN): the draft's inputs and the later passes' metadata on the device.

Today the runner reads the verification's result on the host (``parse_output``), builds the draft's inputs from
the host lists (``HpuEagleProposer.prepare_inputs``) and the later passes' metadata from the host books
(``draft_step_block_table_and_slots`` under engine-step's I01), then issues the draft. The chain computes all of it
on the device from the verification's ``sampled_token_ids`` so the draft is issued right behind the verify. These
tests hold the device forms to the host functions they replace, value for value, and the whole chain to the
host-prepared per-pass draft.
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps as steps  # noqa: E402
from vllm_gaudi.v1.worker.flash_next_decode_metadata import draft_step_block_table_and_slots  # noqa: E402

PLACEHOLDER = -1


def host_prepare_inputs(sampled_lists, num_draft_tokens):
    """``HpuEagleProposer.prepare_inputs`` (hpu_eagle.py) on host lists, then the runner's clamp of -1 to row 0."""
    max_num_draft_tokens = max(num_draft_tokens)
    num_picked, last = [], []
    start = 0
    rejected = [n + 1 - len(sampled_lists[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)]
    for i, n in enumerate(num_draft_tokens):
        r = rejected[i]
        step = max_num_draft_tokens + 1
        for j in range(step):
            if j == n - r:
                last.append(start + j)
            if j < n + 1 - r:
                num_picked.append(start + j)
            else:
                num_picked.append(-1)
        start += step
    return torch.tensor(num_picked).clamp(min=0), torch.tensor(last)


def verification_output(accepted, num_draft, k, seed=0):
    """A rejection sampler's output ``[B, 1 + k]`` int32: ``a`` accepted drafts, the bonus, -1 after; a row without
    drafts (bucket padding) has its bonus in column 0 only."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for a, n in zip(accepted, num_draft):
        a = a if n > 0 else 0
        tokens = torch.randint(0, 1000, (a + 1, ), generator=gen).tolist()
        rows.append(tokens + [PLACEHOLDER] * (1 + k - len(tokens)))
    return torch.tensor(rows, dtype=torch.int32)


def cu_draft(num_draft):
    return torch.cumsum(torch.tensor(num_draft, dtype=torch.int32), 0).to(torch.int32)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_chain_rows_equal_the_host_prepare_inputs(k):
    gen = torch.Generator().manual_seed(k)
    for trial in range(20):
        batch = int(torch.randint(1, 9, (1, ), generator=gen))
        real = int(torch.randint(1, batch + 1, (1, ), generator=gen))
        # real requests: k drafts, or fewer near max_tokens (the scheduler cuts them), or none (one token left);
        # padded rows: no drafts
        num_draft = [int(torch.randint(0, k + 1, (1, ), generator=gen)) if trial % 3 == 0 else k for _ in range(real)]
        num_draft += [0] * (batch - real)
        if max(num_draft) == 0:
            num_draft[0] = 1
        stride = max(num_draft)
        accepted = [int(torch.randint(0, n + 1, (1, ), generator=gen)) for n in num_draft]
        sampled = verification_output(accepted, num_draft, stride, seed=trial)
        lists = [[t for t in row if t != PLACEHOLDER] for row in sampled.tolist()]
        want_picked, want_last = host_prepare_inputs(lists, num_draft)
        picked, last_row, accepted_t, bonus = steps.chain_draft_rows(sampled, cu_draft(num_draft))
        assert torch.equal(picked.long(), want_picked.long()), (num_draft, accepted)
        assert torch.equal(last_row.long(), want_last.long()), (num_draft, accepted)
        assert accepted_t.dtype == torch.int32 and last_row.dtype == torch.int32 and bonus.dtype == torch.int32
        assert accepted_t.tolist() == [len(t) - 1 if n > 0 else 0 for t, n in zip(lists, num_draft)]
        assert bonus.tolist() == [t[len(t) - 1] if n > 0 else t[0] for t, n in zip(lists, num_draft)]


@pytest.mark.parametrize("block_size", [128, 512])
def test_draft_step_slots_equal_engine_steps_host_builder(block_size):
    gen = torch.Generator().manual_seed(block_size)
    max_model_len, max_blocks = 8 * block_size, 8
    pad_block, pad_slot = 0, 7
    for trial in range(30):
        batch = int(torch.randint(1, 9, (1, ), generator=gen))
        num_seq = int(torch.randint(1, batch + 1, (1, ), generator=gen))
        table = torch.randint(1, 1000, (num_seq, max_blocks), generator=gen, dtype=torch.int32)
        positions = torch.randint(0, max_model_len - 1, (batch, ), generator=gen, dtype=torch.int64)
        if trial % 5 == 0:
            positions[0] = max_model_len  # clamped to 0 for the slot, as the host does
            positions[0] = max_model_len - 1
        want_table, want_slots = draft_step_block_table_and_slots(table, positions, batch, block_size, max_model_len,
                                                                  max_blocks, pad_block, pad_slot)
        table_full = torch.full((batch, max_blocks), pad_block, dtype=torch.int32)
        table_full[:num_seq] = table
        real = torch.arange(batch) < num_seq
        got_table, got_slots = steps.draft_step_slots(table_full, positions, real, block_size, max_model_len,
                                                      pad_block, pad_slot)
        assert got_table.dtype == torch.int32 and got_slots.dtype == torch.int64
        assert torch.equal(got_table, want_table), trial
        assert torch.equal(got_slots, want_slots), trial


def test_draft_step_slots_clamp_past_max_model_len():
    block_size, max_model_len, max_blocks = 128, 1024, 8
    table = torch.arange(1, 9, dtype=torch.int32).view(1, -1)
    positions = torch.tensor([max_model_len, 5], dtype=torch.int64)
    real = torch.tensor([True, False])
    _, slots = steps.draft_step_slots(torch.cat((table, torch.zeros_like(table))), positions, real, block_size,
                                      max_model_len, 0, 3)
    assert slots.view(-1).tolist() == [1 * block_size + 0, 3]  # row 0 clamps to position 0: block 1, offset 0


class FakeDraft:
    """The draft adapter and the LM head of test_mtp_draft_graph (deterministic in every input)."""

    H, HC, VOCAB = 4, 2, 23

    def __init__(self):
        gen = torch.Generator().manual_seed(3)
        self.embed = torch.randn(self.VOCAB, self.H, generator=gen)
        self.mix = torch.randn(self.H * self.HC, self.H, generator=gen)
        self.spread = torch.randn(self.H, self.H * self.HC, generator=gen)
        self.lm_head = torch.randn(self.H, self.VOCAB, generator=gen)
        self.calls = []

    def pass_forward(self, *, input_ids, positions, hidden_states, inputs_embeds, attn_metadata, spec_step_idx):
        self.calls.append({"ids": input_ids.clone(), "positions": positions.clone(), "md": attn_metadata,
                           "step": spec_step_idx})
        ids = input_ids.reshape(-1).long().clamp(0, self.VOCAB - 1)
        pos = positions.reshape(-1).to(torch.float32)
        tag = attn_metadata["tag"] if isinstance(attn_metadata, dict) else float(
            attn_metadata.slot_mapping.reshape(-1).to(torch.float32).sum() % 7)
        sample = self.embed.index_select(0, ids) + hidden_states.reshape(-1, self.H * self.HC) @ self.mix
        sample = sample + 0.01 * pos[:, None] + tag + 0.1 * spec_step_idx
        multi = torch.tanh(sample @ self.spread)
        return sample.view(-1, 1, self.H), multi.view(-1, 1, self.H * self.HC)

    def draft_tokens(self, hidden):
        return steps.first_max_ids(hidden @ self.lm_head)


def test_the_chain_gives_the_drafts_of_the_host_prepared_loop():
    k, batch, real, block_size, max_model_len, max_blocks = 3, 4, 3, 16, 256, 16
    stride = k + 1
    num_draft = [k, k, 2, 0]
    accepted = [3, 0, 1, 0]
    sampled = verification_output(accepted, num_draft, k, seed=5)
    gen = torch.Generator().manual_seed(9)
    rows = batch * stride
    positions = (torch.tensor([40, 77, 120, 0])[:, None] + torch.arange(stride)[None, :]).reshape(-1)
    hidden = torch.randn(rows, FakeDraft.H * FakeDraft.HC, generator=gen)
    table = torch.randint(1, 500, (real, max_blocks), generator=gen, dtype=torch.int32)
    table_full = torch.cat((table, torch.zeros((batch - real, max_blocks), dtype=torch.int32)))
    real_mask = torch.arange(batch) < real
    verify_md = {"tag": 0.5, "load_indices_tensor": None}

    def make_md(slots, block_table, ring_slots):
        return {"tag": float(slots.reshape(-1).to(torch.float32).sum() % 7), "slots": slots, "table": block_table}

    # the reference: host lists -> prepare_inputs -> the runner's gathers -> host-built I01 metadata -> the loop
    lists = [[t for t in row if t != PLACEHOLDER] for row in sampled.tolist()]
    picked, last = host_prepare_inputs(lists, num_draft)
    tokens = torch.index_select(sampled.reshape(-1, 1), 0, picked)
    target_positions = torch.index_select(positions, 0, picked)
    target_hidden = torch.index_select(hidden, 0, picked)
    base = target_positions.index_select(0, last)
    step_md = []
    for j in range(1, k):
        t, s = draft_step_block_table_and_slots(table, base + j, batch, block_size, max_model_len, max_blocks, 0, 0)
        step_md.append(make_md(s, t, None))
    ref_draft = FakeDraft()
    ref_loop = steps.MtpDraftLoop(ref_draft.pass_forward, ref_draft.draft_tokens, k, max_model_len)
    want = ref_loop(tokens, target_positions.view(-1, 1), target_hidden, last, verify_md, tuple(step_md))

    chain_draft = FakeDraft()
    chain = steps.MtpDraftChain(chain_draft.pass_forward, chain_draft.draft_tokens, k, max_model_len, block_size, 0,
                                0, make_md)
    out = chain(sampled, cu_draft(num_draft), real_mask, positions.view(-1, 1), (hidden, ), verify_md, table_full)
    assert torch.equal(out.drafts, want.to(torch.int32))
    assert out.accepted.tolist() == [3, 0, 1, 0]
    assert out.last_row.tolist() == last.tolist()
    assert torch.equal(out.next_ids[:, 1:], out.drafts)
    assert out.next_ids[:, 0].tolist() == [row[len(row) - 1] for row in lists]
    assert all(t.dtype == torch.int32 for t in (out.accepted, out.last_row, out.next_ids, out.drafts))
    for ref_call, call in zip(ref_draft.calls, chain_draft.calls):
        assert torch.equal(ref_call["ids"].long(), call["ids"].long())
        assert torch.equal(ref_call["positions"].view(-1), call["positions"].view(-1))
        if ref_call["step"]:
            assert torch.equal(ref_call["md"]["slots"], call["md"]["slots"])
            assert torch.equal(ref_call["md"]["table"], call["md"]["table"])


def test_first_max_ids_equal_argmax_as_int32():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(9, 300, generator=gen).to(torch.bfloat16)
    logits[2, 17] = logits[2].max()
    logits[2, 5] = logits[2].max()  # a tie: the first index, as argmax
    ids = steps.first_max_ids(logits)
    assert ids.dtype == torch.int32
    assert torch.equal(ids.long(), logits.argmax(dim=-1))
