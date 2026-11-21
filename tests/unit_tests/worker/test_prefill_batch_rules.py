# SPDX-License-Identifier: Apache-2.0
"""The rules of batched prefill across requests (``VLLM_HPU_PREFILL_BATCH``, ``v1/worker/prefill_batch.py``).

Pure functions: the flag, which servers may merge, how a run of equal chunks is cut into forwards without padded
rows, the same-step read-after-write guard and the prompt-graph limit. The runner's grouping over these rules is
exercised on the CPU engine (``tests/unit_tests/models/flash_next/test_engine_prefill_batch.py``).
"""

import pytest

from vllm_gaudi.v1.worker import prefill_batch as pb


@pytest.mark.parametrize("value, rows", [("0", 0), ("1", 0), ("2", 2), ("8", 8)])
def test_the_flag_is_the_most_rows_per_forward(monkeypatch, value, rows):
    monkeypatch.setenv("VLLM_HPU_PREFILL_BATCH", value)
    assert pb.resolve_prefill_batch_rows() == rows


def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_PREFILL_BATCH", raising=False)
    assert pb.resolve_prefill_batch_rows() == 0


def test_a_negative_flag_is_an_error(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_PREFILL_BATCH", "-2")
    with pytest.raises(ValueError):
        pb.resolve_prefill_batch_rows()


def _refusal(**overrides):
    kwargs = dict(model_type="qwen4_exp", num_mamba_like_layers=36, num_gdn=36, use_merged_prefill=False,
                  has_kv_transfer=False, data_parallel_size=1)
    kwargs.update(overrides)
    return pb.prefill_batch_refusal(**kwargs)


def test_flash_next_may_merge():
    assert _refusal() is None


@pytest.mark.parametrize("overrides", [
    dict(model_type="granitemoehybrid"),
    dict(model_type="qwen3_next"),
    dict(num_gdn=0),
    dict(num_mamba_like_layers=0),
    dict(use_merged_prefill=True),
    dict(has_kv_transfer=True),
    dict(data_parallel_size=2),
    dict(prefill_checkpoint=True),
])
def test_other_servers_keep_one_request_per_prompt_forward(overrides):
    assert _refusal(**overrides) is not None


@pytest.mark.parametrize("value, on", [(None, False), ("0", False), ("", False), ("1", True)])
def test_prefill_checkpoint_blocks_merging_until_its_maps_are_per_row(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("VLLM_HPU_PREFILL_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("VLLM_HPU_PREFILL_CHECKPOINT", value)
    assert pb.prefill_checkpoint_on() is on


@pytest.mark.parametrize("count, sizes, groups", [
    (1, [1, 2, 4, 8], [1]),
    (2, [1, 2, 4, 8], [2]),
    (3, [1, 2, 4, 8], [2, 1]),
    (7, [1, 2, 4, 8], [4, 2, 1]),
    (8, [1, 2, 4, 8], [8]),
    (12, [1, 2, 4], [4, 4, 4]),
    (5, [1, 2, 3], [3, 2]),
    (3, [1], [1, 1, 1]),
    (3, [], [1, 1, 1]),
])
def test_a_run_is_cut_into_bucket_sized_forwards_without_padded_rows(count, sizes, groups):
    assert pb.group_sizes(count, sizes) == groups
    assert sum(groups) == count


def test_read_and_written_blocks_of_a_chunk():
    blocks = [11, 12, 13, 14]
    # context 256 of block size 128: blocks 11 and 12 are context, the chunk writes 13 and 14
    assert pb.read_blocks(blocks, 256, 128) == [11, 12]
    assert pb.written_blocks(blocks, 256, 128) == [13, 14]
    # context 300 ends inside block 13: the chunk reads it and continues writing it
    assert pb.read_blocks(blocks, 300, 128) == [11, 12, 13]
    assert pb.written_blocks(blocks, 300, 128) == [13, 14]
    # no context: nothing read
    assert pb.read_blocks(blocks, 0, 128) == []
    assert pb.written_blocks(blocks, 0, 128) == blocks


def test_rows_sharing_a_cached_prefix_may_merge():
    # both read blocks 5 and 6 (a prefix hit computed in an earlier step), each writes its own blocks
    rows = [([5, 6, 21, 22], 256), ([5, 6, 31, 32], 256)]
    assert not pb.reads_what_another_writes(rows, 128)


def test_a_row_reading_what_another_row_writes_in_the_step_must_not_merge():
    # row 1's prefix hit covers blocks 21 and 22, which row 0 computes in this very step
    rows = [([5, 6, 21, 22], 256), ([5, 6, 21, 22, 41], 512)]
    assert pb.reads_what_another_writes(rows, 128)
    # and in the other order
    assert pb.reads_what_another_writes(list(reversed(rows)), 128)


def test_null_and_padding_blocks_are_never_a_conflict():
    rows = [([0, 0, 21], 256), ([0, 0, 31], 256), ([-1, 41], 0)]
    assert not pb.reads_what_another_writes(rows, 128)


@pytest.mark.parametrize("rows, query, ctx, limit, inside", [
    (1, 1024, 15744, 64000, True),
    (3, 1024, 15744, 64000, True),
    (4, 1024, 15744, 64000, False),
    (8, 512, 0, 8192, True),
    (8, 1024, 0, 8192, True),
    (8, 1024, 128, 8192, False),
])
def test_the_prompt_graph_limit_counts_every_row(rows, query, ctx, limit, inside):
    assert pb.within_graph_limit(rows, query, ctx, limit) is inside


def test_shapes_compare_by_bucket_and_logits_class():
    a = pb.ChunkShape(query_bucket=1024, context_bucket=123, has_logits=False)
    assert a == pb.ChunkShape(1024, 123, False)
    assert a != pb.ChunkShape(1024, 123, True)
    assert a != pb.ChunkShape(512, 123, False)
    assert a != pb.ChunkShape(1024, 62, False)
