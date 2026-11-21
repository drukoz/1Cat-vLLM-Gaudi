# SPDX-License-Identifier: Apache-2.0
"""persistent_inputs.PersistentInputs (VLLM_HPU_PERSISTENT_INPUTS): the same values, persistent tensors."""
import os

os.environ.setdefault("VLLM_PLUGINS", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.persistent_inputs import PersistentInputs  # noqa: E402


def test_one_pair_per_input_keeps_its_device_tensor():
    inputs = PersistentInputs("cpu", pin_memory=False, buffers=1)
    a = inputs.upload("positions", torch.tensor([[1], [2], [3]], dtype=torch.int32))
    first = a.data_ptr()
    b = inputs.upload("positions", torch.tensor([[4], [5], [6]], dtype=torch.int32))
    assert b is a and b.data_ptr() == first and b.flatten().tolist() == [4, 5, 6]
    c = inputs.upload("positions", torch.arange(8, dtype=torch.int32))  # another shape: another pair
    assert c is not a and c.tolist() == list(range(8)) and len(inputs) == 2


def test_two_pairs_alternate_per_input():
    inputs = PersistentInputs("cpu", pin_memory=False, buffers=2)
    first = inputs.upload("slots", torch.zeros(4, dtype=torch.int64))
    second = inputs.upload("slots", torch.ones(4, dtype=torch.int64))
    third = inputs.upload("slots", torch.full((4, ), 2, dtype=torch.int64))
    assert first is not second and third is first and second.tolist() == [1, 1, 1, 1] and third.tolist() == [2] * 4
    # another input keeps its own turn
    other = inputs.upload("ids", torch.zeros(4, dtype=torch.int64))
    assert other is not first and other is not second


def test_sources_and_dtypes_as_async_h2d_copy():
    inputs = PersistentInputs("cpu", pin_memory=False)
    from_numpy = inputs.upload("a", np.array([1, 2, 3], dtype=np.int32))
    assert from_numpy.dtype == torch.int32 and from_numpy.tolist() == [1, 2, 3]
    from_list = inputs.upload("b", [4, 5], dtype=torch.int32)
    assert from_list.dtype == torch.int32 and from_list.tolist() == [4, 5]
    with pytest.raises(ValueError):
        inputs.upload("c", [1, 2])  # a list needs a dtype, as async_h2d_copy
    strided = torch.arange(12, dtype=torch.int32).view(3, 4)[:, 1]
    assert inputs.upload("d", strided).tolist() == [1, 5, 9]
