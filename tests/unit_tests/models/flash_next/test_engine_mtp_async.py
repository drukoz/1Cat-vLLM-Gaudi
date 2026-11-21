# SPDX-License-Identifier: Apache-2.0
"""MTP under async scheduling (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2) through the real vLLM engine with the HPU
plugin on CPU (mtp-card-ready's tests/unit_tests/hpu_cpu_engine.py): the real MTP drafter, the scheduler's
placeholder drafts, the runner's draft handoff and count correction, prefix caching on. The streams must equal the
synchronous MTP engine's, one prompt at a time and all prompts at once; the async run must log that async
scheduling is on for MTP."""

from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_engine

K = 3
ASYNC = {"VLLM_HPU_FLASH_NEXT_ASYNC_SCHED": "2"}


@pytest.fixture(scope="module")
def runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_mtp_async")
    result = {}
    for name, options, env in (("sync", ("--k", str(K)), None), ("async", ("--k", str(K)), ASYNC),
                               ("sync_conc", ("--k", str(K), "--concurrent"), None),
                               ("async_conc", ("--k", str(K), "--concurrent"), ASYNC)):
        record = hpu_cpu_engine.run(tiny_checkpoint, out / f"{name}.json", *options, env=env)
        record["log"] = Path(str(out / f"{name}.json") + ".log").read_text(encoding="utf-8", errors="replace")
        result[name] = record
    return result


def _streams(record):
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


@pytest.mark.parametrize("pair", [("sync", "async"), ("sync_conc", "async_conc")])
def test_async_mtp_streams_equal_sync(runs, pair):
    sync, asyn = runs[pair[0]], runs[pair[1]]
    assert _streams(asyn) == _streams(sync)
    assert "Async scheduling on for Flash-Next with MTP" in asyn["log"]
    assert "Async scheduling on for Flash-Next with MTP" not in sync["log"]
    assert asyn["committed"], "no verification step ran"
