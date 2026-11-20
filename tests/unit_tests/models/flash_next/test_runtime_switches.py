# SPDX-License-Identifier: Apache-2.0
"""``VLLM_HPU_RUNTIME_SWITCH_FILE`` (``v1/worker/runtime_switches.py``): an A/B of the per-step flags on one server.

* The file format: ``NAME=VALUE`` lines of the switchable flags only; a change is seen once.
* In the whole serving loop (``engine_harness``): a server started with both flags off takes
  ``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`` and ``VLLM_HPU_FUSED_GREEDY_LOGITS=1`` from the file in the middle of
  its requests and drops them again, each decode step runs the path of the flags in force, and the tokens are
  the HF reference's throughout (the two paths are bit-identical, so a switch may fall on any step).
* A file that names another flag changes nothing and does not stop the server.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402

from vllm_gaudi.v1.worker.runtime_switches import SWITCHABLE, RuntimeSwitchFile, parse_switches  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

ON = "VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1\nVLLM_HPU_FUSED_GREEDY_LOGITS=1\n"
OFF = "VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0\nVLLM_HPU_FUSED_GREEDY_LOGITS=0\n"


def write(path, text: str) -> None:
    """Replace the switch file with a rename, as the probe does."""
    staging = str(path) + ".new"
    with open(staging, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(staging, path)


def test_switch_file_format():
    assert set(SWITCHABLE) == {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA", "VLLM_HPU_FUSED_GREEDY_LOGITS",
                               "VLLM_HPU_CANDIDATE_SAMPLING", "VLLM_HPU_VOCAB_PARALLEL_LOGITS"}
    assert parse_switches("# arm B\n\nVLLM_HPU_FUSED_GREEDY_LOGITS = 1  # in-graph greedy\n") == {
        "VLLM_HPU_FUSED_GREEDY_LOGITS": "1"
    }
    assert parse_switches(ON) == {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": "1", "VLLM_HPU_FUSED_GREEDY_LOGITS": "1"}
    assert parse_switches("") == {}
    for bad in ("VLLM_HPU_HC_FUSED=1", "VLLM_HPU_FUSED_GREEDY_LOGITS", "VLLM_HPU_FUSED_GREEDY_LOGITS=", "junk"):
        with pytest.raises(ValueError):
            parse_switches(bad)


def test_a_change_is_seen_once(tmp_path):
    path = tmp_path / "switches"
    switches = RuntimeSwitchFile(str(path))
    assert switches.poll() is None  # no file: no change
    write(path, ON)
    assert switches.poll() == parse_switches(ON)
    assert switches.poll() is None
    write(path, OFF)
    assert switches.poll() == parse_switches(OFF)
    assert switches.poll() is None
    write(path, "VLLM_HPU_HC_FUSED=1\n")
    with pytest.raises(ValueError):
        switches.poll()
    assert switches.poll() is None  # a refused file is not read again until it changes


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


class _Steps:
    """Per decode step of an engine: (in-graph greedy?, no paged-attention buffers?)."""

    def __init__(self, engine) -> None:
        self.steps: list[tuple[bool, bool]] = []
        runner = engine.runner
        generic = runner._execute_model_generic

        def recording(token_ids, position_ids, attn_metadata, *args, **kwargs):
            if not attn_metadata.is_prompt:
                self.steps.append((bool(kwargs.get("fuse_greedy_sampling", False)), attn_metadata.block_list is None))
            return generic(token_ids, position_ids, attn_metadata, *args, **kwargs)

        runner._execute_model_generic = recording


def test_one_server_takes_both_flags_and_drops_them(tiny_checkpoint, reference, tmp_path):
    prompts = [p for p in reference if len(p["prompt_ids"]) <= 700 and len(p["generated_ids"]) >= 12][:3]
    assert len(prompts) == 3
    path = tmp_path / "switches"
    env = {"VLLM_HPU_RUNTIME_SWITCH_FILE": str(path)}
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=4, env=env)) as engine:
        steps = _Steps(engine)
        for index, prompt in enumerate(prompts):
            engine.add(f"r{index}", prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]))

        def decode_steps(count: int) -> list[tuple[bool, bool]]:
            start = len(steps.steps)
            while len(steps.steps) < start + count and engine.has_work():
                engine.step()
            return steps.steps[start:]

        assert set(decode_steps(3)) == {(False, False)}  # the server starts with both flags off
        write(path, ON)
        assert set(decode_steps(3)) == {(True, True)}
        assert engine.runner.flash_next_sparse_decode_metadata()
        write(path, "VLLM_HPU_HC_FUSED=1\n")  # refused: nothing changes, the server goes on
        assert set(decode_steps(1)) == {(True, True)}
        write(path, "VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0\n")  # one flag only: the other keeps its value
        assert set(decode_steps(2)) == {(True, False)}
        write(path, OFF)
        assert set(decode_steps(2)) == {(False, False)}
        assert not engine.runner.flash_next_sparse_decode_metadata()
        engine.run()
        for index, prompt in enumerate(prompts):
            assert engine.tokens[f"r{index}"] == prompt["generated_ids"]
    assert "VLLM_HPU_RUNTIME_SWITCH_FILE" not in os.environ
