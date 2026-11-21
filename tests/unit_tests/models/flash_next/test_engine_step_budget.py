# SPDX-License-Identifier: Apache-2.0
"""What one decode step moves across the host/device boundary (``engine_harness.Counters``).

A step of the serving loop is run for real on CPU (the engine's scheduler, the real ``HPUModelRunner``, the tiny
model) with every host-to-device copy and every host read of a device tensor counted. The numbers here are the
budget the engine-step work is measured against: a change that adds a copy or a read to the decode step fails
this file.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import random  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

# One decode step of the unchanged runner: 12 tensors built in _create_decode_input_data and 2 PLE inputs are
# copied to the device, and the sampled tokens are read back once. Neither number depends on the row count.
BASELINE_H2D_COPIES = 14
BASELINE_HOST_READS = 1


def decode_step_counts(tiny_checkpoint, rows: int, env: dict | None = None, async_scheduling: bool = False,
                       steps: int = 6) -> dict:
    """Host-to-device copies and host reads per step over ``steps`` pure decode steps of ``rows`` streams."""
    rng = random.Random(7)
    options = eh.Options(max_num_seqs=32, count=True, env=env or {}, async_scheduling=async_scheduling)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        for index in range(rows):
            engine.add(f"r{index}", [rng.randrange(1000, 200000) for _ in range(40 + 3 * index)], max_tokens=40)
        while min(len(tokens) for tokens in engine.tokens.values()) < 6:
            engine.step()
        engine.counters.start()
        for _ in range(steps):
            engine.step()
        engine.counters.stop()
        totals = engine.counters.totals()
        return {
            "h2d_copies": totals["h2d_calls"] / steps,
            "h2d_bytes": totals["h2d_bytes"] / steps,
            "host_reads": totals["host_reads"] / steps,
            "sites": dict(engine.counters.h2d),
        }


@pytest.mark.parametrize("rows", [1, 32])
def test_decode_step_budget_of_the_unchanged_runner(tiny_checkpoint, rows):
    counts = decode_step_counts(tiny_checkpoint, rows)
    assert counts["h2d_copies"] == BASELINE_H2D_COPIES, counts["sites"]
    assert counts["host_reads"] == BASELINE_HOST_READS
