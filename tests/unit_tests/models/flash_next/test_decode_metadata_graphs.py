# SPDX-License-Identifier: Apache-2.0
"""Flash-Next sparse-QSA decode metadata (``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``): graphs, copies, refusal.

The companion of ``test_decode_metadata_sparse.py`` (bit-exactness): with the flag on every decode step of one
batch size has one graph signature (the dense path has one per block bucket), a decode step copies three fewer
tensors to the device, and a dense-QSA server refuses the flag.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import random  # noqa: E402

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.test_decode_metadata_sparse import FLAG, _Recorder  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")


def _two_contexts(tiny_checkpoint, flag: int) -> dict:
    """Four short and then four long requests decoding at batch size four: decode signatures by input shape."""
    rng = random.Random(9)
    options = eh.Options(max_num_seqs=4, max_model_len=4096, env={FLAG: str(flag)})
    with eh.open_engine(tiny_checkpoint, options) as engine:
        recorder = _Recorder(engine)
        for length in (40, 700):
            for index in range(4):
                engine.add(f"len{length}-{index}", [rng.randrange(1000, 200000) for _ in range(length + index)],
                           max_tokens=3)
            engine.run()
        signatures: dict = {}
        for shape, signature in recorder.decode_signatures:
            signatures.setdefault(shape, set()).add(signature)
        return signatures


def test_one_decode_graph_per_batch_size(tiny_checkpoint):
    dense = _two_contexts(tiny_checkpoint, 0)
    sparse = _two_contexts(tiny_checkpoint, 1)
    assert set(dense) == set(sparse)
    assert any(len(signatures) > 1 for signatures in dense.values()), "the dense path should split by block bucket"
    assert all(len(signatures) == 1 for signatures in sparse.values()), sparse


@pytest.mark.parametrize("rows", [1, 32])
def test_decode_step_copies_three_fewer_tensors(tiny_checkpoint, rows):
    from tests.unit_tests.models.flash_next.test_engine_step_budget import (BASELINE_H2D_COPIES, BASELINE_HOST_READS,
                                                                            decode_step_counts)

    dense = decode_step_counts(tiny_checkpoint, rows)
    sparse = decode_step_counts(tiny_checkpoint, rows, env={FLAG: "1"})
    assert dense["h2d_copies"] == BASELINE_H2D_COPIES
    assert sparse["h2d_copies"] == BASELINE_H2D_COPIES - 3, sparse["sites"]
    assert sparse["host_reads"] == BASELINE_HOST_READS
    assert sparse["h2d_bytes"] < dense["h2d_bytes"]


def test_flag_is_refused_under_dense_qsa(tiny_checkpoint):
    """Dense QSA reads the paged-attention buffers: with the flag on the runner keeps building them."""
    options = eh.Options(max_num_seqs=2, env={FLAG: "1", "VLLM_HPU_FLASH_NEXT_QSA_MODE": "0"})
    with eh.open_engine(tiny_checkpoint, options) as engine:
        assert engine.runner.flash_next_sparse_decode_metadata() is False
