# SPDX-License-Identifier: Apache-2.0
"""CPU tensor-parallel gate: vLLM TP layers on gloo ranks must equal plain PyTorch.

Model under test (identical on every rank, float32, seeded):
    VocabParallelEmbedding(1024, 256)
    -> ColumnParallelLinear(256, 512, bias=False)
    -> F.silu
    -> RowParallelLinear(512, 256, bias=False, input_is_parallel=True, reduce_results=True)

Run: python -m pytest -q test_cpu_tp.py
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from harness import record_collectives, run_ranks

VOCAB = 1024
HIDDEN = 256
INTER = 512


def _full_weights():
    """Full (unsharded) weights, identical on every rank and in the test process."""
    torch.manual_seed(0)
    emb = torch.randn(VOCAB, HIDDEN) * 0.02
    w1 = torch.randn(INTER, HIDDEN) * 0.02
    w2 = torch.randn(HIDDEN, INTER) * 0.02
    return emb, w1, w2


def _input_ids():
    return torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(1))


def _reference():
    """Same math in plain PyTorch with the full weights."""
    emb_w, w1, w2 = _full_weights()
    x = F.embedding(_input_ids(), emb_w)
    x = x @ w1.T
    x = F.silu(x)
    return x @ w2.T


def _forward_once():
    """Build the sharded layers, load full weights via each param's weight_loader,
    run the forward, return the output tensor."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    emb_w, w1, w2 = _full_weights()
    emb = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.float32, prefix="emb")
    col = ColumnParallelLinear(HIDDEN, INTER, bias=False, params_dtype=torch.float32, prefix="col")
    row = RowParallelLinear(
        INTER,
        HIDDEN,
        bias=False,
        input_is_parallel=True,
        reduce_results=True,
        params_dtype=torch.float32,
        prefix="row",
    )
    emb.weight.weight_loader(emb.weight, emb_w)
    col.weight.weight_loader(col.weight, w1)
    row.weight.weight_loader(row.weight, w2)
    # On the CPU platform the unquantized linear packs its weight into a
    # oneDNN `cpu_linear` in process_weights_after_loading(); vLLM's model
    # loader normally calls it, we call it explicitly after loading weights.
    for layer in (emb, col, row):
        layer.quant_method.process_weights_after_loading(layer)

    x = emb(_input_ids())
    x, _ = col(x)
    x = F.silu(x)
    x, _ = row(x)
    return x.detach().clone()


def _rank_forward(rank):
    # Return a numpy array, not a tensor: pickling a torch tensor over a
    # spawn queue uses torch's shm manager, which costs ~30 s (serialized)
    # per rank on macOS. numpy arrays pickle by value. `test_allreduce_smoke`
    # keeps a raw tensor return, which is what proves tensors pass the queue.
    return _forward_once().numpy()


def _rank_forward_recorded(rank):
    with record_collectives() as recorded:
        _forward_once()
    return recorded


def _rank_forward_asymmetric(rank):
    """Rank 0 runs the forward twice, rank 1 once: collective order/count differ."""
    times = 2 if rank == 0 else 1
    with record_collectives() as recorded:
        for _ in range(times):
            _forward_once()
    return recorded


def _rank_allreduce(rank):
    import torch.distributed as dist

    t = torch.ones(4)
    dist.all_reduce(t)
    return t


def _rank_boom(rank):
    if rank == 1:
        raise ValueError("deliberate boom")
    return rank


def test_allreduce_smoke():
    """Step 1 smoke: world 8, every rank all-reduces ones(4) and gets 8*ones."""
    results = run_ranks(8, _rank_allreduce, timeout_s=120)
    expected = torch.full((4,), 8.0)
    for rank, t in enumerate(results):
        assert torch.allclose(t, expected), (rank, t)


def test_run_ranks_reports_rank_failure():
    """A raising rank is reported as a RuntimeError carrying its traceback."""
    with pytest.raises(RuntimeError, match="deliberate boom"):
        run_ranks(2, _rank_boom, timeout_s=60)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_sharded_matches_unsharded(world_size):
    ref = _reference().numpy()
    results = run_ranks(world_size, _rank_forward, timeout_s=120)
    for rank, out in enumerate(results):
        assert out.shape == ref.shape, f"rank {rank} shape {out.shape} != {ref.shape}"
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_collective_order_identical(world_size):
    results = run_ranks(world_size, _rank_forward_recorded, timeout_s=120)
    for rank, rec in enumerate(results):
        assert len(rec) > 0, f"rank {rank} issued no collectives: {rec}"
        assert rec == results[0], f"rank {rank} collective order differs: {rec} != {results[0]}"


def test_detects_mismatch():
    """World 2, rank 0 runs the forward twice and rank 1 once: the order check must
    fail, either as unequal lists or as a TimeoutError (gloo can hang on a
    collective with no partner, which is why the harness has a timeout)."""
    try:
        results = run_ranks(2, _rank_forward_asymmetric, timeout_s=30)
    except TimeoutError:
        return
    assert results[0] != results[1], f"mismatch not detected: {results}"
