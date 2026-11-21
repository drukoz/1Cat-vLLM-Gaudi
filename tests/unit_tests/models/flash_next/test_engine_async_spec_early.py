# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M2 of MTP under async scheduling (``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3``) through the real vLLM engine with
the HPU plugin on CPU (``tests/unit_tests/hpu_cpu_engine.py``): the real MTP drafter at k = 2, mtp-draft-graph's chain
(``VLLM_HPU_MTP_DRAFT_CHAIN=1``) and the device-made verification inputs (``VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=2``),
prefix caching off.

* A chained step's output is returned before its tokens reach the host, and its books follow one step late; the
  streams equal mode 2's (the same engine whose runner reads every step's tokens first), prompts one at a time and
  all at once (requests joining and finishing make steps that wait for every pending book).
* With prefix caching on, mode 3 acts as mode 2 (its align-mode commits are stage M3) and says so.

On CPU every copy is done when it is issued, so these runs prove the books, not the overlap.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_engine

ENV = {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": "1", "VLLM_HPU_MTP_DRAFT_LOOP": "1", "VLLM_HPU_MTP_DRAFT_GRAPH": "1",
       "VLLM_HPU_MTP_DRAFT_CHAIN": "1", "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS": "2"}
STATS = re.compile(r"VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3: (\d+) steps issued before the previous step's tokens reached the "
                   r"host; books applied (\d+) late, (\d+) after a wait; (\d+) chained steps read first")


@pytest.fixture(scope="module")
def runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_async_spec_early")
    result = {}
    for name, mode, options in (("m2", "2", ()), ("m3", "3", ()), ("m2_conc", "2", ("--concurrent", )),
                                ("m3_conc", "3", ("--concurrent", )), ("m3_prefix", "3", ("--prefix-caching", ))):
        options = ("--k", "2", "--max-tokens", "24") + (options if name == "m3_prefix" else
                                                        ("--no-prefix-caching", ) + options)
        options = tuple(o for o in options if o != "--prefix-caching")  # the engine's default is prefix caching on
        record = hpu_cpu_engine.run(tiny_checkpoint, out / f"{name}.json", *options,
                                    env={**ENV, "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED": mode})
        record["log"] = Path(str(out / f"{name}.json") + ".log").read_text(encoding="utf-8", errors="replace")
        result[name] = record
    return result


def _streams(record: dict) -> dict:
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


@pytest.mark.parametrize("suffix", ["", "_conc"], ids=["one_at_a_time", "all_at_once"])
def test_steps_issued_before_the_last_read_give_mode_2s_streams(runs, suffix):
    early, sync = runs["m3" + suffix], runs["m2" + suffix]
    assert _streams(early) == _streams(sync)
    found = STATS.findall(early["log"])
    assert found, "no step was issued early"
    issued, late, waited, read_first = (int(v) for v in found[-1])
    assert issued >= 32 and late + waited > 0 and read_first == 0
    assert "acts as 2" not in early["log"]
    assert STATS.search(sync["log"]) is None


def test_with_prefix_caching_mode_3_acts_as_mode_2(runs):
    log = runs["m3_prefix"]["log"]
    assert "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3 acts as 2" in log and "prefix caching off" in log
    assert STATS.search(log) is None
