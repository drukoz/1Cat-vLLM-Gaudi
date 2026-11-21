# SPDX-License-Identifier: Apache-2.0
"""The fake-HPU block mapping of a decode batch, built twice on the same metadata.

The MTP drafter's first step reuses the target's decode metadata, so ``_set_block_mapping`` runs on it a second
time. The fake-HPU branch (CPU ``one_hot`` raises on an out-of-range class) rewrites the padding entries of
``block_groups`` to ``batch_size`` in place; the second pass must treat that value as padding too. The real HPU
branch hands ``block_groups`` to ``one_hot`` unchanged. Run in a process of its own: it loads the CPU stubs.
"""

from types import SimpleNamespace

import torch

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before the first vllm_gaudi import)


def _processor(runner_module):
    processor = object.__new__(runner_module.HPUAttentionMetadataProcessor)
    processor.block_size = 4
    return processor


def _metadata():
    # 5 blocks: requests 0, 1, 1, padding (-1), padding (-1); batch size 2
    return SimpleNamespace(block_usage=torch.tensor([4, 2, 3, 1, 1], dtype=torch.int32),
                           block_groups=torch.tensor([0, 1, 1, -1, -1], dtype=torch.long),
                           block_size=4)


def _replace(metadata, _name, **fields):
    values = dict(vars(metadata))
    values.update(fields)
    return SimpleNamespace(**values)


def test_the_second_pass_over_the_same_metadata_gives_the_same_mapping(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_model_runner as runner_module

    monkeypatch.setattr(runner_module, "is_fake_hpu", lambda: True)
    monkeypatch.setattr(runner_module, "custom_tuple_replace", _replace)
    processor = _processor(runner_module)
    first = processor._set_block_mapping(_metadata(), 2, "cpu", torch.float32)
    expected = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=torch.float32)
    assert torch.equal(first.block_mapping, expected)
    assert first.block_groups.tolist() == [0, 1, 1, 2, 2]  # the padding rewritten to batch_size
    second = processor._set_block_mapping(first, 2, "cpu", torch.float32)
    assert torch.equal(second.block_mapping, expected)
    assert second.block_groups.tolist() == [0, 1, 1, 2, 2]
    assert torch.equal(second.attn_bias, first.attn_bias)
