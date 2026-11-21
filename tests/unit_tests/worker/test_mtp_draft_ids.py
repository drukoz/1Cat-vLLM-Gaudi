# SPDX-License-Identifier: Apache-2.0
"""The Flash-Next MTP draft's token ids (``draft_argmax``, VLLM_HPU_MTP_DRAFT_IDS).

Card rule 9 (job 00038): ``torch.topk`` ids narrowed to int32 in the same lazy graph came back as every second id
of the true table. Mode 1 makes the draft's ids int32 from an fp32 id table, so no int64 id exists on the device:
these tests hold it to ``argmax`` (the first index of the row maximum) on CPU, and show what the card's reading
would do to the default path's ``argmax -> .int()``.
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps as steps  # noqa: E402


@pytest.fixture
def draft_ids_mode(monkeypatch):
    def set_mode(mode):
        monkeypatch.setenv("VLLM_HPU_MTP_DRAFT_IDS", str(mode))

    return set_mode


@pytest.mark.parametrize("rows,vocab", [(1, 1000), (2, 248320), (32, 248320), (128, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int32_ids_equal_argmax(draft_ids_mode, rows, vocab, dtype):
    torch.manual_seed(rows + vocab)
    logits = torch.randn(rows, vocab).to(dtype)
    top = logits[0].max()
    logits[0, 7] = top
    logits[0, 3] = top  # a tie at the maximum: argmax takes the first index
    draft_ids_mode(0)
    reference = steps.draft_argmax(logits)
    draft_ids_mode(1)
    ids = steps.draft_argmax(logits)
    assert reference.dtype == torch.int64
    assert ids.dtype == torch.int32
    assert ids.shape == (rows, )
    assert torch.equal(reference.to(torch.int32), ids)


def test_default_is_argmax(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_MTP_DRAFT_IDS", raising=False)
    logits = torch.randn(5, 300)
    ids = steps.draft_argmax(logits)
    assert ids.dtype == torch.int64
    assert torch.equal(ids, logits.argmax(dim=-1))


def test_a_row_without_a_maximum_stays_inside_the_vocabulary(draft_ids_mode):
    draft_ids_mode(1)
    logits = torch.full((3, 64), float("nan"))
    ids = steps.draft_argmax(logits)
    assert int(ids.min()) >= 0 and int(ids.max()) <= 63


def card_narrowing_of_int64_ids(ids: torch.Tensor) -> torch.Tensor:
    """What job 00038 measured for ``torch.topk(...).indices.to(torch.int32)`` in one lazy graph: the cast reads
    8-byte elements from ids the bridge holds as 4 bytes, so element i of the result is element 2 * i of the true
    table and the second half is read from beyond it (modelled as -1)."""
    flat = ids.reshape(-1).to(torch.int32)
    out = torch.full_like(flat, -1)
    half = (flat.numel() + 1) // 2
    out[:half] = flat[::2]
    return out.reshape(ids.shape)


def test_the_card_reading_breaks_every_row_but_the_first_and_mode_1_has_no_int64_id(draft_ids_mode):
    """The default path under the card's reading: row 0 keeps its id, the other rows get another row's id or none
    (one stream is sound, more streams lose draft positions 2 to k). Mode 1's ids are int32 already: the draft
    loop's ``.int()`` is the identity, nothing is narrowed."""
    torch.manual_seed(3)
    logits = torch.randn(8, 500)
    draft_ids_mode(0)
    int64_ids = steps.draft_argmax(logits)
    narrowed = card_narrowing_of_int64_ids(int64_ids)
    truth = logits.argmax(dim=-1).to(torch.int32)
    assert narrowed[0] == truth[0]
    assert not torch.equal(narrowed[1:], truth[1:])
    draft_ids_mode(1)
    ids = steps.draft_argmax(logits)
    assert ids.dtype == torch.int32
    assert ids.int() is ids
    assert torch.equal(ids, truth)
