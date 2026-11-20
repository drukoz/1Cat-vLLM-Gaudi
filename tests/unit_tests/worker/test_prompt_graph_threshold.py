# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS: the prompt size above which a prefill forward runs outside HPU graphs.

Upstream (``HPUModelRunner._use_graphs``) replays a prompt forward from an HPU graph only while
``batch x query + context blocks x block size`` stays at or under ``max_cudagraph_capture_size``, which is
``--max-num-batched-tokens`` when no capture size is configured (vLLM caps a configured one at that value).
With the rental settings (8,192 batched tokens, 2,048-token chunks, 128-token bucket blocks) a chunk is
graph-replayed only while its context is 31 blocks or fewer, so about 57 of the 59 chunks of a 120K prompt
are traced op by op on the host. The flag replaces the threshold for prompts; decode is unchanged.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from vllm_gaudi.v1.worker.hpu_model_runner import (  # noqa: E402
    HPUModelRunner, resolve_prompt_graph_max_tokens)

ENV = "VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS"


def _runner(max_capture: int, override: int = 0, enforce_eager: bool = False):
    return SimpleNamespace(model_config=SimpleNamespace(enforce_eager=enforce_eager),
                           max_cudagraph_capture_size=max_capture,
                           prompt_graph_max_tokens=override)


def _meta(is_prompt: bool, seq_len: int, num_blocks: int, block_size: int = 128):
    return SimpleNamespace(is_prompt=is_prompt, block_size=block_size, seq_len=lambda: seq_len,
                           num_blocks=lambda: num_blocks)


@pytest.mark.parametrize("num_blocks,expected", [(0, True), (31, True), (48, True), (62, False), (960, False)])
def test_without_the_flag_a_2048_chunk_replays_only_at_short_context(num_blocks, expected):
    # 2048 + 48 * 128 = 8192 is the last size inside the threshold; the next bucket (62) is outside.
    assert HPUModelRunner._use_graphs(_runner(8192), _meta(True, 2048, num_blocks), 1) is expected


@pytest.mark.parametrize("num_blocks", [0, 62, 484, 960, 1008])
def test_the_flag_replays_every_chunk_of_a_128k_prompt(num_blocks):
    assert HPUModelRunner._use_graphs(_runner(8192, override=140000), _meta(True, 2048, num_blocks), 1) is True


def test_the_flag_also_lowers_the_threshold():
    runner = _runner(8192, override=4096)
    assert HPUModelRunner._use_graphs(runner, _meta(True, 2048, 16), 1) is True  # 2048 + 2048
    assert HPUModelRunner._use_graphs(runner, _meta(True, 2048, 17), 1) is False


def test_the_flag_counts_the_whole_prompt_batch():
    runner = _runner(8192, override=16384)
    assert HPUModelRunner._use_graphs(runner, _meta(True, 8192, 0), 2) is True
    assert HPUModelRunner._use_graphs(runner, _meta(True, 8192, 1), 2) is False


@pytest.mark.parametrize("override", [0, 4096, 140000])
def test_decode_always_replays(override):
    assert HPUModelRunner._use_graphs(_runner(8192, override=override), _meta(False, 1, 2048), 32) is True


def test_enforce_eager_wins_over_the_flag():
    runner = _runner(8192, override=140000, enforce_eager=True)
    assert HPUModelRunner._use_graphs(runner, _meta(True, 2048, 0), 1) is False


def test_a_runner_without_the_attribute_keeps_the_upstream_rule():
    runner = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False), max_cudagraph_capture_size=8192)
    assert HPUModelRunner._use_graphs(runner, _meta(True, 2048, 31), 1) is True
    assert HPUModelRunner._use_graphs(runner, _meta(True, 2048, 62), 1) is False


def test_resolve_reads_the_flag(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_prompt_graph_max_tokens() == 0
    monkeypatch.setenv(ENV, "140000")
    assert resolve_prompt_graph_max_tokens() == 140000
    monkeypatch.setenv(ENV, "0")
    assert resolve_prompt_graph_max_tokens() == 0


@pytest.mark.parametrize("bad", ["-1", "lots"])
def test_resolve_refuses_a_bad_value(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError):
        resolve_prompt_graph_max_tokens()


class _Stop(Exception):
    """Ends ``execute_model`` right after the knob is read (the fake runner has nothing else)."""


def _stop(*_args, **_kwargs):
    raise _Stop


def test_execute_model_takes_the_threshold_from_the_runtime_knob_at_every_step():
    values = iter([0, 140000, 4096])
    runner = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False), max_cudagraph_capture_size=8192,
                             prompt_graph_max_tokens=140000, _prompt_graph_knob=lambda: next(values),
                             run_defragmenter=_stop, _poll_runtime_switches=lambda: None)
    long_chunk = _meta(True, 2048, 960)
    for expected_value, expected_graphs in ((0, False), (140000, True), (4096, False)):
        with pytest.raises(_Stop):
            HPUModelRunner.execute_model(runner, None)
        assert runner.prompt_graph_max_tokens == expected_value
        assert HPUModelRunner._use_graphs(runner, long_chunk, 1) is expected_graphs


def test_execute_model_without_a_knobs_file_keeps_the_servers_threshold():
    # integration build: execute_model first polls engine-step's runtime switches (stubbed here)
    for runner in (SimpleNamespace(prompt_graph_max_tokens=140000, _prompt_graph_knob=None, run_defragmenter=_stop,
                                   _poll_runtime_switches=lambda: None),
                   SimpleNamespace(prompt_graph_max_tokens=140000, run_defragmenter=_stop,
                                   _poll_runtime_switches=lambda: None)):
        with pytest.raises(_Stop):
            HPUModelRunner.execute_model(runner, None)
        assert runner.prompt_graph_max_tokens == 140000
