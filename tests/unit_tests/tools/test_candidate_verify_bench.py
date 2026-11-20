# SPDX-License-Identifier: Apache-2.0
"""tools/flash_next/candidate_verify_bench.py on the CPU: both paths run, and the checks it makes on a card hold."""
import json

from tools.flash_next import candidate_verify_bench as bench


def test_the_bench_times_both_paths_and_its_checks_hold_on_the_cpu(tmp_path):
    out = tmp_path / "bench.json"
    assert bench.main(["--device", "cpu", "--requests", "2", "--iters", "1", "--out", str(out)]) == 0
    record = json.loads(out.read_text())[0]
    assert record["requests"] == 2 and record["rows"] == 8
    assert record["exact"]["p50_ms"] > 0 and record["candidates"]["p50_ms"] > 0
    assert record["ids_rows_equal_to_cpu_topk"] == "8/8" and record["kth_value_equal"]
    assert record["exact_tokens_valid"] and record["candidates_tokens_valid"]
    assert record["greedy_first_token_is_argmax"]
