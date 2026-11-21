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

from vllm_gaudi.v1.worker.runtime_switches import (GRAPH_SWITCHABLE, SWITCHABLE, RuntimeSwitchFile,  # noqa: E402
                                                    parse_switches)

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
                               "VLLM_HPU_CANDIDATE_SAMPLING", "VLLM_HPU_VOCAB_PARALLEL_LOGITS",
                               "VLLM_HPU_MTP_DRAFT_GRAPH", "VLLM_HPU_MTP_DRAFT_CHAIN",
                               "VLLM_HPU_MOE_DENSE_MIN_TOKENS", "VLLM_HPU_MOE_DENSE_MAX_TOKENS"}
    assert set(GRAPH_SWITCHABLE) == {"VLLM_HPU_MOE_DENSE_MIN_TOKENS", "VLLM_HPU_MOE_DENSE_MAX_TOKENS"}
    assert parse_switches("VLLM_HPU_MOE_DENSE_MIN_TOKENS=1\nVLLM_HPU_MOE_DENSE_MAX_TOKENS=512\n") == {
        "VLLM_HPU_MOE_DENSE_MIN_TOKENS": "1", "VLLM_HPU_MOE_DENSE_MAX_TOKENS": "512"
    }
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


def test_moe_dense_range_drops_the_captured_graphs(tmp_path, monkeypatch):
    """The MoE dense range is read inside the forward: a file naming it drops the model's and the drafter's captured
    HPU graphs (``wrap_in_hpu_graph``'s ``clear_cache``), so the next steps capture the path in force; a file that
    names only per-step flags keeps them."""
    import types

    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    dropped = []

    class Graphed:  # what wrap_in_hpu_graph leaves on the module it wraps

        def __init__(self, name):
            self.name = name

        def clear_cache(self):
            dropped.append(self.name)

    path = tmp_path / "switches"
    drafter = types.SimpleNamespace(model=Graphed("drafter"), _draft_graph=Graphed("draft loop"), _draft_chain=None)
    runner = types.SimpleNamespace(model=Graphed("model"), drafter=drafter, _runtime_switches=RuntimeSwitchFile(str(path)))
    runner._drop_hpu_graphs = types.MethodType(HPUModelRunner._drop_hpu_graphs, runner)
    for name in GRAPH_SWITCHABLE + ("VLLM_HPU_FUSED_GREEDY_LOGITS", ):
        monkeypatch.setenv(name, "0")  # monkeypatch restores them after the test
    write(path, "VLLM_HPU_MOE_DENSE_MIN_TOKENS=1\nVLLM_HPU_MOE_DENSE_MAX_TOKENS=512\n")
    HPUModelRunner._poll_runtime_switches(runner)
    assert dropped == ["model", "drafter", "draft loop"]  # no chained draft was built
    assert os.environ["VLLM_HPU_MOE_DENSE_MIN_TOKENS"] == "1"
    assert os.environ["VLLM_HPU_MOE_DENSE_MAX_TOKENS"] == "512"
    HPUModelRunner._poll_runtime_switches(runner)  # the file did not change: nothing
    assert dropped == ["model", "drafter", "draft loop"]
    write(path, "VLLM_HPU_FUSED_GREEDY_LOGITS=1\n")
    HPUModelRunner._poll_runtime_switches(runner)  # a flag read outside the captured forward: the graphs stay
    assert dropped == ["model", "drafter", "draft loop"]
    assert os.environ["VLLM_HPU_FUSED_GREEDY_LOGITS"] == "1"
    runner.drafter, runner.model = None, object()  # no drafter, an eager model: nothing to drop
    assert HPUModelRunner._drop_hpu_graphs(runner) == []


def test_moe_route_stats_are_appended_once_per_step(tmp_path, monkeypatch):
    """VLLM_HPU_MOE_ROUTE_STATS=<folder>: the runner appends the MoE ops' route-stats rows to rank<r>.jsonl after
    every step, with the step's token count and kind."""
    import types

    import torch

    from vllm_gaudi.extension import ops as plugin_ops
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    folder = tmp_path / "stats"
    monkeypatch.setenv("VLLM_HPU_MOE_ROUTE_STATS", str(folder))
    monkeypatch.setattr(plugin_ops, "_ROUTE_STATS", {
        "buffer": torch.tensor([[1, 2, 7], [0, 3, 7], [9, 9, 9]], dtype=torch.int32),
        "ops": 2
    })
    step = types.SimpleNamespace(total_num_scheduled_tokens=8, scheduled_spec_decode_tokens={"a": [1, 2, 3]})
    HPUModelRunner._dump_moe_route_stats(object(), step, 0, 2)
    HPUModelRunner._dump_moe_route_stats(object(), step, 0, 2)
    lines = (folder / "rank0.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["ops"] == [[1, 2, 7], [0, 3, 7]]
    assert (record["tokens"], record["spec_tokens"], record["decodes"], record["prefills"]) == (8, 3, 2, 0)


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
