# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M3 (``VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN``) through the real vLLM engine with the HPU plugin on CPU
(``tests/unit_tests/hpu_cpu_engine.py``): prefix caching on (align mode, 256-token recurrent blocks), MTP k = 2 with
the real drafter, mtp-draft-graph's chain, the device-made inputs (M1), generations of 160 tokens that cross a
recurrent block (the 100- and 700-token prompts), so the pre-step copies and the boundary commits run.

* oracle acceptance 2,0,1,2,1,0: ``=1`` checks every step's device-made align mode (running block, lattices, load
  counts, the next load counts after the commits) against the host's without a difference; ``=2`` uses it and gives
  the host's streams;
* async scheduling: mode 3 with ``=2`` serves with prefix caching on (no "acts as 2") and gives the streams of the
  synchronous engine, also for the agent's next turn; no copy writes through a slice of a pool.

On this build the next turn gets no prefix-cache hit on the CPU engine (``test_engine_mtp``'s turn 3 notes the same),
so the commits' slots are checked by the unit tests (``test_flash_next_async_align.py``) and the check mode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_engine

ENV = {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": "1", "VLLM_HPU_MTP_DRAFT_LOOP": "1", "VLLM_HPU_MTP_DRAFT_GRAPH": "1",
       "VLLM_HPU_MTP_DRAFT_CHAIN": "1", "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS": "2",
       "VLLM_HPU_PREFIX_ALIGN_SPLIT": "1", "VLLM_HPU_PREFIX_KEEP_PROMPT_END": "1"}
STATS = re.compile(r"VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN=(\d): (\d+) verification steps made, (\d+) checked, (\d+) "
                   r"with differences")
TOKENS = "160"


@pytest.fixture(scope="module")
def runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_spec_device_align")
    ref = json.loads((Path(tiny_checkpoint) / "reference.json").read_text(encoding="utf-8"))
    prompts = sorted((p for p in ref["prompts"] if len(p["prompt_ids"]) in (100, 700)), key=lambda p: len(p["prompt_ids"]))
    prompts_path = out / "prompts.json"
    prompts_path.write_text(json.dumps({"prompts": [{"prompt_ids": p["prompt_ids"]} for p in prompts]}),
                            encoding="utf-8")
    base = hpu_cpu_engine.run(tiny_checkpoint, out / "k0.json", "--prompts", str(prompts_path), "--max-tokens", TOKENS)
    oracle = out / "oracle.json"
    oracle.write_text(json.dumps({"streams": base["prompts"], "accept": [2, 0, 1, 2, 1, 0]}), encoding="utf-8")
    result = {}
    common = ("--k", "2", "--prompts", str(prompts_path), "--max-tokens", TOKENS)
    for align in ("0", "1", "2"):
        result[f"oracle{align}"] = hpu_cpu_engine.run(tiny_checkpoint, out / f"oracle{align}.json", *common, "--oracle",
                                                      str(oracle), env={**ENV, "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN": align})
    turn = ("--next-turn", json.dumps([11, 12, 13, 14, 15, 16, 17, 18]), "--slice-write-probe")
    for name, sched, align in (("sync0", "0", "0"), ("async3_2", "3", "2")):
        result[name] = hpu_cpu_engine.run(tiny_checkpoint, out / f"{name}.json", *common, *turn,
                                          env={**ENV, "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED": sched,
                                               "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN": align})
    for name, record in result.items():
        record["log"] = Path(str(out / f"{name}.json") + ".log").read_text(encoding="utf-8", errors="replace")
    return result


def _streams(record: dict) -> dict:
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


def test_check_mode_finds_the_device_align_mode_equal_to_the_hosts(runs):
    log = runs["oracle1"]["log"]
    assert "align mode differs" not in log
    mode, steps, checked, differing = (int(v) for v in STATS.findall(log)[-1])
    assert mode == 1 and steps >= 64 and checked >= steps and differing == 0


def test_use_mode_gives_the_streams_of_the_hosts_align_mode(runs):
    assert _streams(runs["oracle2"]) == _streams(runs["oracle0"]) == _streams(runs["oracle1"])
    mode, steps, checked, _ = (int(v) for v in STATS.findall(runs["oracle2"]["log"])[-1])
    assert mode == 2 and steps >= 64 and checked == 0


def test_mode_3_serves_with_prefix_caching_and_gives_the_synchronous_streams(runs):
    early, sync = runs["async3_2"], runs["sync0"]
    assert "acts as 2" not in early["log"]
    assert re.search(r"VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3: \d+ steps issued", early["log"])
    assert _streams(early) == _streams(sync)
    assert len(early["prompts"]) == 3 and early["prompts"][2].get("next_turn")
    assert early["slice_write_probe"]["violations"] == []
