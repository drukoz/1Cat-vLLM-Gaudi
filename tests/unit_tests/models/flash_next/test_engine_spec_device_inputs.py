# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M1 of MTP under async scheduling (``VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS``) through the real vLLM engine
with the HPU plugin on CPU (``tests/unit_tests/hpu_cpu_engine.py``): the real MTP drafter at k = 2, oracle drafts that
make the verification accept every count 0..k, the sparse decode metadata (``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``,
which the device-made inputs need), prefix caching on and off.

* ``=1``: every verification step that continues the previous one gets its ids, positions, KV slots, GDN accepted
  counts and PLE context made on the device from the previous step's, and they equal the host-made ones;
* ``=2``: the model gets the device-made ones, and the streams equal the engine's without the flag;
* off: nothing is made and nothing is logged.

Each engine run is a child process of about 20 s.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_engine

K = 2
ACCEPT = [2, 0, 1, 2, 1, 0]
ENV = {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": "1", "VLLM_HPU_MTP_DRAFT_LOOP": "1"}
FLAG = "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS"
STATS = re.compile(FLAG + r"=(\d): (\d+) verification steps, (\d+) advanced on the device, (\d+) checked, (\d+) with")


@pytest.fixture(scope="module")
def runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_spec_device_inputs")
    base = hpu_cpu_engine.run(tiny_checkpoint, out / "k0.json")
    oracle = out / "oracle.json"
    oracle.write_text(json.dumps({"streams": base["prompts"], "accept": ACCEPT}), encoding="utf-8")
    result = {"base": base}
    for prefix in (True, False):
        options = ("--k", str(K), "--oracle", str(oracle)) + (() if prefix else ("--no-prefix-caching", ))
        for mode in ("0", "1", "2"):
            name = f"m{mode}" + ("" if prefix else "_nopc")
            record = hpu_cpu_engine.run(tiny_checkpoint, out / f"{name}.json", *options, env={**ENV, FLAG: mode})
            record["log"] = Path(str(out / f"{name}.json") + ".log").read_text(encoding="utf-8", errors="replace")
            result[name] = record
    return result


def _streams(record: dict) -> dict:
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


def _last_stats(log: str) -> tuple:
    found = STATS.findall(log)
    assert found, "no stats line"
    return tuple(int(v) for v in found[-1])


@pytest.mark.parametrize("suffix", ["", "_nopc"], ids=["prefix_caching", "no_prefix_caching"])
def test_check_mode_finds_every_device_made_input_equal_to_the_hosts(runs, suffix):
    log = runs["m1" + suffix]["log"]
    assert "device-made inputs differ" not in log
    mode, verify_steps, advanced, checked, differing = _last_stats(log)
    assert mode == 1 and differing == 0
    assert advanced >= 16 and checked == advanced and verify_steps >= advanced
    assert "keeps the host's inputs" not in log  # nothing refused the device-made inputs


@pytest.mark.parametrize("suffix", ["", "_nopc"], ids=["prefix_caching", "no_prefix_caching"])
def test_use_mode_gives_the_streams_of_the_engine_without_the_flag(runs, suffix):
    off, check, use = runs["m0" + suffix], runs["m1" + suffix], runs["m2" + suffix]
    assert _streams(check) == _streams(off)
    assert _streams(use) == _streams(off)
    mode, _, advanced, checked, _ = _last_stats(use["log"])
    assert mode == 2 and advanced >= 16 and checked == 0
    assert use["committed"], "no verification step ran"


def test_the_flag_off_makes_nothing(runs):
    # vLLM's own warning names the variable when it is set at all; the runner's lines read "<FLAG>=<mode>: ..."
    for name in ("m0", "m0_nopc"):
        assert re.search(FLAG + r"=\d: ", runs[name]["log"]) is None, name
