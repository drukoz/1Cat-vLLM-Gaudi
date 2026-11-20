# SPDX-License-Identifier: Apache-2.0
"""The PLE layer in the HPU runner's padded prompt buckets: exact row lengths, and a fixed-shape state write.

**Lengths.** ``HPUModelRunner._form_prefill_batch`` builds ``padding_mask_flat`` in the MODEL dtype
(``torch.zeros(target_bs, target_seq, dtype=self.dtype)``, bfloat16) and the PLE layer took the number of real
tokens of a row as ``padding_mask.view(rows, seq_len).sum(1).long()``. bfloat16 has 8 significant bits: it holds
every integer up to 256 and then only multiples of 2, 4, 8, ..., so the sum of 1701 ones is 1704 (CPU; a
device accumulating in bfloat16 can do worse). ``ple_short_conv_batched_torch`` cuts the saved short-conv window
at ``history[length : length + 9]``, so three padding rows went into the state every later token reads. The
other PLE tests use rows of at most 40 tokens, where the float sum is exact.

The runner already sends the exact lengths: ``query_start_loc_p`` is the int32 cumulative sum of the real
query lengths of the bucket's rows (``make_prefill_metadata(query_start_loc=query_start_loc_p)``), for a whole
prompt and for every chunk of a chunked prefill alike (each chunk is its own prefill batch).

**State write.** ``qwen3_5._save_ssm_state`` writes a multi-row state through ``torch.nonzero`` and a Python
``if`` on its result: a host read and a data-dependent shape inside an HPU graph, on every PLE decode step of a
batch of two or more. The PLE layer writes with one ``index_copy_`` instead; a row whose store index is
invalid goes to the state tensor's last row, the padding row every runner layout reserves (see
``HpuQwen4ExpPLELayer._save_conv_state``).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.models.flash_next.test_ple_layer import (  # noqa: E402,F401  (ple_layer is the fixture)
    CHANNELS, SLOTS, TOL, _eos, _md, _oracle, ple_layer)

REAL, BUCKET = 1701, 2048  # 1701 real tokens in the 2048-token prompt bucket: the bfloat16 sum says 1704
TAIL, TAIL_BUCKET = 300, 384  # a second chunk; 300 is exact in bfloat16, so only the first chunk's state can be wrong
PAD_ROW = SLOTS - 1  # the state tensor's last row, the runner's padding row


def bf16_mask(real: list[int], bucket: int) -> torch.Tensor:
    """The runner's ``padding_mask_flat``: ``[rows * bucket, 1]`` in the model dtype, ones on the real tokens."""
    mask = torch.zeros(len(real), bucket, dtype=torch.bfloat16)
    for row, n in enumerate(real):
        mask[row, :n] = 1.0
    return mask.view(-1, 1)


def cumulative(real: list[int]) -> torch.Tensor:
    """The runner's ``query_start_loc_p``: int32 ``[rows + 1]``, the cumulative REAL lengths."""
    return torch.tensor([0] + list(torch.tensor(real).cumsum(0)), dtype=torch.int32)


def padded_ids(parts: list[torch.Tensor], bucket: int) -> torch.Tensor:
    ids = torch.zeros(len(parts), bucket, dtype=torch.int64)
    for row, part in enumerate(parts):
        ids[row, :part.numel()] = part
    return ids.view(-1)


def ple_inputs(vllm_config, rows: int, bucket: int):
    """The PLE kwargs of a padded batch: every row is one request of ``bucket`` tokens."""
    qsl = torch.arange(rows + 1, dtype=torch.int32) * bucket
    ctx = torch.full((rows, 2), _eos(vllm_config), dtype=torch.int32)
    return qsl, ctx


def test_the_premise_bfloat16_cannot_count_to_1701():
    assert int(bf16_mask([REAL], BUCKET).view(1, BUCKET).sum(1).long()) != REAL
    assert int(bf16_mask([256], BUCKET).view(1, BUCKET).sum(1).long()) == 256


@pytest.mark.parametrize("with_query_start_loc", [True, False], ids=["query_start_loc_p", "mask_only"])
def test_prefill_1701_of_2048_with_a_bfloat16_mask(ple_layer, with_query_start_loc):
    """Output of the real rows and the saved state equal the unpadded oracle.

    ``mask_only`` is metadata without ``query_start_loc_p`` (a caller that sends the mask alone): the length must
    then come from an exact sum of the mask, not a bfloat16 one.
    """
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(11)
    ids = torch.randint(0, 1000, (REAL, ), dtype=torch.int64)
    hidden = torch.randn(BUCKET, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden[:REAL], eos=eos)

    md = _md(True, [1], has_initial=[0], padding=bf16_mask([REAL], BUCKET))
    if with_query_start_loc:
        md.query_start_loc_p = cumulative([REAL])
    qsl, ctx = ple_inputs(vllm_config, 1, BUCKET)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        out = layer(hidden, padded_ids([ids], BUCKET), qsl, ctx)

    torch.testing.assert_close(out[:REAL], ref_out, **TOL)
    gap = (layer.kv_cache[0][1] - ref_state).abs().max().item()
    assert gap <= 1e-5, (f"saved short-conv state of a {REAL}-token row in a {BUCKET}-token bucket differs from "
                         f"the unpadded oracle by {gap:.3e}: the window was cut at the wrong row")


def test_two_padded_rows_keep_their_own_lengths(ple_layer):
    """Rows of 1701 and 300 real tokens in one 2048-token bucket: each state equals its own oracle."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(12)
    ids_a = torch.randint(0, 1000, (REAL, ), dtype=torch.int64)
    ids_b = torch.randint(0, 1000, (TAIL, ), dtype=torch.int64)
    hidden = torch.randn(2 * BUCKET, CHANNELS)
    _, state_a = _oracle(layer, ids_a, hidden[:REAL], eos=eos)
    _, state_b = _oracle(layer, ids_b, hidden[BUCKET:BUCKET + TAIL], eos=eos)

    md = _md(True, [2, 3], has_initial=[0, 0], padding=bf16_mask([REAL, TAIL], BUCKET))
    md.query_start_loc_p = cumulative([REAL, TAIL])
    qsl, ctx = ple_inputs(vllm_config, 2, BUCKET)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        layer(hidden, padded_ids([ids_a, ids_b], BUCKET), qsl, ctx)

    torch.testing.assert_close(layer.kv_cache[0][2], state_a, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][3], state_b, **TOL)


def test_chunked_prefill_in_padded_buckets(ple_layer):
    """Two chunks, each in its own padded bucket (1701 of 2048, then 300 of 384): the state after the second
    chunk equals the oracle over the 2001 tokens, so the first chunk's window was cut at row 1701."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(13)
    ids = torch.randint(0, 1000, (REAL + TAIL, ), dtype=torch.int64)
    hidden = torch.randn(REAL + TAIL, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden, eos=eos)

    def chunk(start: int, length: int, bucket: int, has_initial: int, context) -> torch.Tensor:
        chunk_hidden = torch.randn(bucket, CHANNELS)
        chunk_hidden[:length] = hidden[start:start + length]
        md = _md(True, [1], has_initial=[has_initial], padding=bf16_mask([length], bucket))
        md.query_start_loc_p = cumulative([length])
        qsl, ctx = ple_inputs(vllm_config, 1, bucket)
        if context is not None:
            ctx = context.view(1, 2).to(torch.int32)
        with torch.no_grad(), set_forward_context(md, vllm_config):
            return layer(chunk_hidden, padded_ids([ids[start:start + length]], bucket), qsl, ctx)[:length]

    first = chunk(0, REAL, BUCKET, 0, None)
    second = chunk(REAL, TAIL, TAIL_BUCKET, 1, ids[REAL - 2:REAL])
    torch.testing.assert_close(torch.cat((first, second)), ref_out, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][1], ref_state, **TOL)


def decode_batch(vllm_config, layer, load: list[int]):
    """One single-token decode step of ``len(load)`` rows; returns (output, state before, state after)."""
    eos = _eos(vllm_config)
    rows = len(load)
    layer.kv_cache[0].copy_(torch.randn_like(layer.kv_cache[0]))
    before = layer.kv_cache[0].clone()
    ids = torch.randint(0, 1000, (rows, ), dtype=torch.int64)
    hidden = torch.randn(rows, CHANNELS)
    qsl = torch.arange(rows + 1, dtype=torch.int32)
    ctx = torch.full((rows, 2), eos, dtype=torch.int32)
    with torch.no_grad(), set_forward_context(_md(False, load), vllm_config):
        out = layer(hidden, ids, qsl, ctx)
    return out, before, layer.kv_cache[0].clone()


def forbid(monkeypatch, name: str) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError(f"torch.{name} ran in the PLE forward: a data-dependent shape inside an HPU graph")

    monkeypatch.setattr(torch, name, refuse)


def test_decode_batch_state_write_has_no_data_dependent_shape(ple_layer, monkeypatch):
    """A decode batch of real rows and padding rows writes its states without ``nonzero`` (or a boolean-mask
    index): the write is one fixed-shape ``index_copy_`` whatever the indices hold."""
    vllm_config, layer = ple_layer
    torch.manual_seed(21)
    forbid(monkeypatch, "nonzero")
    forbid(monkeypatch, "masked_select")
    _, before, after = decode_batch(vllm_config, layer, [1, 2, PAD_ROW, PAD_ROW])
    for slot in (1, 2):
        assert not torch.equal(after[slot], before[slot]), f"the real row of slot {slot} wrote no state"
    for slot in (0, 3):
        assert torch.equal(after[slot], before[slot]), f"slot {slot} belongs to no row of the batch"


def test_invalid_store_index_goes_to_the_padding_row(ple_layer, monkeypatch):
    """A row whose index is outside the tensor (``-1``: the compact layout's padding value) is not read and does
    not write a request's row: it writes the state tensor's last row, the padding row."""
    vllm_config, layer = ple_layer
    torch.manual_seed(22)
    forbid(monkeypatch, "nonzero")
    _, before, after = decode_batch(vllm_config, layer, [1, -1])
    assert not torch.equal(after[1], before[1])
    for slot in range(SLOTS - 1):
        if slot != 1:
            assert torch.equal(after[slot], before[slot]), f"the invalid row wrote slot {slot}"


def test_single_row_decode_writes_only_its_slot(ple_layer, monkeypatch):
    vllm_config, layer = ple_layer
    torch.manual_seed(23)
    forbid(monkeypatch, "nonzero")
    _, before, after = decode_batch(vllm_config, layer, [2])
    assert not torch.equal(after[2], before[2])
    for slot in (0, 1, 3, 4):
        assert torch.equal(after[slot], before[slot])
