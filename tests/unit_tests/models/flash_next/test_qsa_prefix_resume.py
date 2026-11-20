# SPDX-License-Identifier: Apache-2.0
"""A request resumed from a cached prefix sees the QSA keys a full prefill builds.

Prefix caching resumes a request at a recurrent-block boundary (896 tokens at TP4, 512 at TP8: the runner's
mamba-block-aligned chunks, ``VLLM_HPU_PREFIX_ALIGN_SPLIT``). The sparse-QSA indexer's state at that point is:

* the compressed-key cache, one row per ``compress_ratio`` KV slots, so a shared prefix block carries the rows its
  first writer built (``qsa_side_caches``);
* the raw-key ring of the request's state slot, which holds the raw keys of a compression block that is not yet
  complete. A boundary that is a multiple of ``compress_ratio`` (every 128-token block is) leaves no such block
  open, and the prefill branch of ``HpuQsaIndexer.select`` only writes the ring (the prompt's tail), it never reads
  it.

This test runs one prompt two ways on the same weights, inputs and starting caches:

* full: one prefill of all ``N`` tokens in a fresh state slot, then decode steps;
* resumed: request A prefills the prefix ``[0, P)`` in slot A; request B prefills ``[P, N)`` with
  ``context_len = P`` over the same (shared) KV blocks, loading a state slot whose ring holds another request's
  keys and storing its own slot; then the same decode steps from B's slot;

and checks that the compressed-key rows of ``[0, N)``, the suffix tokens' selections and the selections of the
decode steps that follow (the first of which complete the prompt's last block from the ring) are equal bit for
bit, at ``P`` = one and two recurrent blocks of TP4 (896, 1792) and TP8 (512, 1024), with and without a prompt
tail, in decode mode 0 and mode 1. The top-k is lowered to 32 blocks so that every selection depends on the key
values. A control corrupts the shared prefix rows and expects the selections to change.
"""

import contextlib
import os
import tempfile

import pytest
import torch

from tests.unit_tests.models.flash_next.cpu_model import decode_metadata, prefill_metadata
from tests.unit_tests.models.flash_next.qsa_fused_ranks import (MAX_MODEL_LEN, MB, bind_random_caches,
                                                                build_indexer, rope_table)
from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaFusedMode

BLOCK = 128
RATIO = 4
TOPK_BLOCKS = 32
RING_ROWS = 4  # compress_ratio * ceil(compress_ratio / compress_ratio): a decode-only ring
SLOT_FULL, SLOT_A, SLOT_STALE, SLOT_B = 5, 7, 11, 13
DECODE_STEPS = 6


@pytest.fixture(scope="module", autouse=True)
def cpu_world():
    """A world-1 gloo group and a plain vLLM config: what a standalone indexer's linear layer needs."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel,
                                                 init_distributed_environment, initialize_model_parallel,
                                                 model_parallel_is_initialized)
    if model_parallel_is_initialized():
        yield
        return
    fd, init_file = tempfile.mkstemp(dir="/tmp")
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(1, 0, f"file://{init_file}", 0, "gloo")
            initialize_model_parallel(1)
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(init_file)


def _indexer(seed: int, mode: QsaFusedMode):
    indexer = build_indexer(seed, torch.bfloat16)
    indexer.block_topk = TOPK_BLOCKS
    indexer.output_width = TOPK_BLOCKS * RATIO + RATIO - 1
    gen = torch.Generator().manual_seed(seed + 1)  # the same random history in both indexers
    bind_random_caches(indexer, RING_ROWS, gen, torch.bfloat16)
    indexer.fused_mode = mode
    return indexer


def _run(prefix: int, total: int, mode: QsaFusedMode, *, corrupt_prefix: bool = False):
    seed = 1000 + prefix + total
    gen = torch.Generator().manual_seed(seed + 2)
    hidden = (torch.randn(total + DECODE_STEPS, 128, generator=gen)).to(torch.bfloat16)
    positions = torch.arange(total + DECODE_STEPS, dtype=torch.int64)
    table = rope_table(MAX_MODEL_LEN).to(torch.bfloat16)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731

    full = _indexer(seed, mode)
    resumed = _indexer(seed, mode)
    with torch.no_grad():
        sel_full = full.select(hidden[:total], positions[:total], csa, prefill_metadata(total, SLOT_FULL))
        resumed.select(hidden[:prefix], positions[:prefix], csa, prefill_metadata(prefix, SLOT_A))
        if corrupt_prefix:
            rows = prefix // RATIO
            resumed.compressed_key_cache[:rows] = resumed.compressed_key_cache[:rows].flip(0)
        md_b = prefill_metadata(total - prefix,
                                load_slot=SLOT_STALE,
                                store_slot=SLOT_B,
                                has_initial_state=True,
                                context_len=prefix,
                                block_size=BLOCK)
        sel_b = resumed.select(hidden[prefix:total], positions[prefix:total], csa, md_b)
        dec_full, dec_b = [], []
        for d in range(DECODE_STEPS):
            p = total + d
            h, pos = hidden[p:p + 1], positions[p:p + 1]
            dec_full.append(full.select(h, pos, csa, decode_metadata(SLOT_FULL, p, BLOCK, MB)))
            dec_b.append(resumed.select(h, pos, csa, decode_metadata(SLOT_B, p, BLOCK, MB)))
    rows = -(-(total + DECODE_STEPS) // RATIO)
    return {
        "sel_full": sel_full[prefix:total],
        "sel_b": sel_b,
        "dec_full": torch.cat(dec_full),
        "dec_b": torch.cat(dec_b),
        "cache_full": full.compressed_key_cache[:rows].view(torch.int16),
        "cache_b": resumed.compressed_key_cache[:rows].view(torch.int16),
    }


# (P, suffix tokens): P is one or two recurrent blocks (TP4: 896, TP8: 512); 203 tokens leave a prompt tail of
# 3 raw keys in the ring, 200 none.
CASES = [(896, 203), (1792, 203), (896, 200), (512, 203), (1024, 203), (512, 200)]


@pytest.mark.parametrize("mode", [QsaFusedMode.OFF, QsaFusedMode.TORCH], ids=["mode0", "mode1"])
@pytest.mark.parametrize("prefix,suffix", CASES, ids=[f"P{p}+{s}" for p, s in CASES])
def test_resumed_prefix_sees_the_full_prefill_keys(prefix, suffix, mode):
    r = _run(prefix, prefix + suffix, mode)
    assert r["sel_b"].shape == r["sel_full"].shape
    assert torch.equal(r["cache_b"], r["cache_full"]), "compressed-key rows differ from a full prefill"
    assert torch.equal(r["sel_b"], r["sel_full"]), "the suffix's selections differ from a full prefill"
    assert torch.equal(r["dec_b"], r["dec_full"]), "decode after the resumed prefill differs"
    # The selections are not trivial: fewer blocks are selected than the context holds.
    assert int((r["sel_full"][-1] >= 0).sum()) < prefix + suffix


def test_control_a_wrong_prefix_changes_the_selections():
    r = _run(896, 1099, QsaFusedMode.TORCH, corrupt_prefix=True)
    assert not torch.equal(r["sel_b"], r["sel_full"])


OFFSET_BLOCK = 200  # the physical block of request B's first own (suffix) block


def _phys_slot(p: int, prefix: int) -> int:
    """Request B's KV slot of position ``p``: the prefix in the shared blocks 0.., the rest from OFFSET_BLOCK."""
    if p < prefix:
        return p
    return (OFFSET_BLOCK + (p - prefix) // BLOCK) * BLOCK + (p - prefix) % BLOCK


@pytest.mark.parametrize("mode", [QsaFusedMode.OFF, QsaFusedMode.TORCH], ids=["mode0", "mode1"])
@pytest.mark.parametrize("prefix,suffix", [(896, 203), (512, 203)], ids=["P896+203", "P512+203"])
def test_resumed_request_in_other_blocks(prefix, suffix, mode):
    """As above, with request B's suffix in other physical blocks than the full prefill's (a real prefix hit)."""
    total = prefix + suffix
    seed = 2000 + total
    gen = torch.Generator().manual_seed(seed + 2)
    hidden = torch.randn(total + DECODE_STEPS, 128, generator=gen).to(torch.bfloat16)
    positions = torch.arange(total + DECODE_STEPS, dtype=torch.int64)
    table = rope_table(MAX_MODEL_LEN).to(torch.bfloat16)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    full = _indexer(seed, mode)
    resumed = _indexer(seed, mode)
    pad_block = resumed.compressed_key_cache.shape[0] // (BLOCK // RATIO) - 1
    b_table = list(range(prefix // BLOCK)) + [OFFSET_BLOCK + j for j in range(MB - prefix // BLOCK)]
    b_table = torch.tensor(b_table[:MB], dtype=torch.int32)
    assert int(b_table.max()) < pad_block
    with torch.no_grad():
        sel_full = full.select(hidden[:total], positions[:total], csa, prefill_metadata(total, SLOT_FULL))
        resumed.select(hidden[:prefix], positions[:prefix], csa, prefill_metadata(prefix, SLOT_A))
        md_b = prefill_metadata(total - prefix, load_slot=SLOT_STALE, store_slot=SLOT_B, has_initial_state=True,
                                context_len=prefix, block_size=BLOCK)
        md_b.slot_mapping = torch.tensor([_phys_slot(p, prefix) for p in range(prefix, total)], dtype=torch.int64)
        sel_b = resumed.select(hidden[prefix:total], positions[prefix:total], csa, md_b)
        dec_full, dec_b = [], []
        for d in range(DECODE_STEPS):
            p = total + d
            h, pos = hidden[p:p + 1], positions[p:p + 1]
            dec_full.append(full.select(h, pos, csa, decode_metadata(SLOT_FULL, p, BLOCK, MB)))
            md = decode_metadata(SLOT_B, p, BLOCK, MB)
            md.slot_mapping = torch.tensor([_phys_slot(p, prefix)], dtype=torch.int64)
            md.block_table = b_table[None].contiguous()
            dec_b.append(resumed.select(h, pos, csa, md))
    assert torch.equal(sel_b, sel_full[prefix:total])
    assert torch.equal(torch.cat(dec_b), torch.cat(dec_full))
    groups = -(-(total + DECODE_STEPS) // RATIO)
    for g in range(groups - 1):  # every complete compression block of the prompt and of the decode steps
        row_b = _phys_slot(g * RATIO, prefix) // RATIO
        assert torch.equal(resumed.compressed_key_cache[row_b].view(torch.int16),
                           full.compressed_key_cache[g].view(torch.int16)), f"block {g} (row {row_b})"
