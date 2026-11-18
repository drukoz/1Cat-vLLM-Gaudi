# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the consumers of a one-element recurrent cache (RUK-3106).

The Flash-Next PLE short-conv layer keeps a single recurrent state, so its
MambaSpec cache is a one-element tuple. HpuPlatform.insert_blocks_to_device
and _hpu_transfer_cache_regions used to assume a two-element (conv, ssm)
tuple and raised IndexError / ValueError on the one-element cache.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import pytest
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector import _hpu_transfer_cache_regions
from vllm_gaudi.platform import HpuPlatform


def test_insert_blocks_one_element_tuple():
    src = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    dst = (torch.zeros(4, 3),)
    HpuPlatform.insert_blocks_to_device(src, dst, torch.tensor([2, 0]), torch.tensor([1, 3]))
    assert torch.equal(dst[0][1], src[0][2])
    assert torch.equal(dst[0][3], src[0][0])
    assert torch.equal(dst[0][0], torch.zeros(3))
    assert torch.equal(dst[0][2], torch.zeros(3))


def test_insert_blocks_two_element_tuple_unchanged():
    src = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    dst = (torch.zeros(4, 3), torch.zeros(4, 3))
    HpuPlatform.insert_blocks_to_device(src, dst, torch.tensor([2, 0]), torch.tensor([1, 3]))
    for i in (0, 1):
        assert torch.equal(dst[i][1], src[i][2])
        assert torch.equal(dst[i][3], src[i][0])
        assert torch.equal(dst[i][0], torch.zeros(3))
        assert torch.equal(dst[i][2], torch.zeros(3))


def _cpu_index_copy_fp8_supported() -> bool:
    try:
        zeros = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
        zeros.index_copy_(0, torch.tensor([0]), torch.zeros(1, 2, dtype=torch.float8_e4m3fn))
        return True
    except NotImplementedError:
        return False


@pytest.mark.skipif(not _cpu_index_copy_fp8_supported(),
                    reason="index_copy_cpu not implemented for Float8_e4m3fn on CPU torch")
def test_insert_blocks_fp8_view():
    src = torch.randn(1, 4, 3).to(torch.float8_e4m3fn)
    dst = (torch.zeros(4, 3).to(torch.float8_e4m3fn),)
    HpuPlatform.insert_blocks_to_device(src, dst, torch.tensor([2, 0]), torch.tensor([1, 3]))
    assert torch.equal(dst[0].view(torch.uint8)[1], src.view(torch.uint8)[0][2])
    assert torch.equal(dst[0].view(torch.uint8)[3], src.view(torch.uint8)[0][0])


def test_swap_out_then_insert_roundtrip_one_element():
    dev = torch.randn(4, 3)
    host = torch.zeros(1, 4, 3)
    idx = torch.tensor([1, 3])
    HpuPlatform.swap_out_blocks_to_host((dev,), host, idx, idx)
    back = torch.zeros(4, 3)
    HpuPlatform.insert_blocks_to_device(host, (back,), idx, idx)
    assert torch.equal(back[1], dev[1])
    assert torch.equal(back[3], dev[3])


def test_nixl_regions_single_state_mamba():
    spec = MambaSpec(shapes=((9, 512), ), dtypes=(torch.bfloat16, ), block_size=128)
    t = torch.zeros(4, 9, 512)
    regions = _hpu_transfer_cache_regions(None, (t, ), spec)
    assert len(regions) == 1
    assert regions[0] is t


def test_nixl_regions_two_state_mamba_unchanged():
    spec = MambaSpec(shapes=((9, 512), (4, 64)), dtypes=(torch.bfloat16, torch.bfloat16), block_size=128)
    conv = torch.zeros(4, 9, 512)
    ssm = torch.zeros(4, 4, 64)
    regions = _hpu_transfer_cache_regions(None, (conv, ssm), spec)
    assert len(regions) == 1
    assert regions[0] is conv
