# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GDN decode bench tool (B1, rental run sheet).

Checks of ``floor_bytes``/``heads``/``slope``, the exact output formats of a small CPU run of
``main``, and the exit code 2 paths.  No timing value is ever asserted: CPU times are a smoke
test only.
"""

from __future__ import annotations

import re

import pytest

from tools.flash_next.gdn_core_decode_bench import floor_bytes, heads, main, slope

RESULT_RE = re.compile(r"^(core|recurrent) tp8 (B=[12]: [0-9.]+ ms/call, floor [0-9.]+ ms, [0-9.]+x floor"
                       r"|slope: -?[0-9.e-]+ ms/seq/layer, floor [0-9.e-]+, -?[0-9.]+x floor \(target <= 2x\))$")


def test_floor_bytes_check_values():
    assert floor_bytes("core", 8, 1) == 808984
    assert floor_bytes("recurrent", 8, 1) == 792088
    assert floor_bytes("core", 2, 1) == 3_235_936


def test_floor_bytes_scales_with_batch():
    for op in ("core", "recurrent"):
        for tp in (2, 8):
            assert floor_bytes(op, tp, 4) == 4 * floor_bytes(op, tp, 1)


def test_heads():
    assert heads(8) == (2, 6, 1280)
    assert heads(2) == (8, 24, 5120)
    with pytest.raises(ValueError):
        heads(3)


def test_slope():
    assert slope([1, 2, 3], [1.0, 3.0, 5.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        slope([4], [1.0])


def test_main_small_cpu(capsys):
    assert main(["--device", "cpu", "--tp", "8", "--batches", "1,2", "--iters", "1", "--warmup", "1",
                 "--repeat", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("device=cpu ops=core,recurrent tp=8 batches=1,2 iters=1 repeat=1 graph=")
    results = [line for line in lines[1:] if line.strip()]
    assert len(results) == 6
    for line in results:
        assert RESULT_RE.match(line), line


def test_main_verify_op_cpu(capsys):
    """``--ops verify``: the fused verification op at T nodes per row (a CPU smoke of the bench path)."""
    assert floor_bytes("verify", 8, 2, 4) == floor_bytes("core", 8, 8)
    assert main(["--device", "cpu", "--ops", "verify", "--tp", "8", "--batches", "1,2", "--iters", "1", "--warmup",
                 "1", "--repeat", "1", "--verify-tokens", "4"]) == 0
    results = [line for line in capsys.readouterr().out.splitlines()[1:] if line.strip()]
    assert len(results) == 3
    assert all(line.startswith("verify(T=4) tp8 ") for line in results), results
    assert main(["--device", "cpu", "--ops", "verify", "--tp", "8", "--batches", "1", "--verify-tokens", "7"]) == 2


def test_main_bad_args():
    base = ["--device", "cpu", "--tp", "8", "--batches", "1,2", "--iters", "1", "--warmup", "1", "--repeat", "1"]
    assert main([*base[:2], "--tp", "3"]) == 2
    assert main([*base, "--ops", "conv"]) == 2
    assert main(["--device", "hpu"]) == 2
