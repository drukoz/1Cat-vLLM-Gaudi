# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/rental/spec_probe.py: the speculative counters of a server's ``/metrics``.

The exposition texts below were printed by vLLM's own ``SpecDecodingProm`` at the pinned commit
(``8f816a3f``, k = 3, after ``observe_draft(3, 3)``, ``(3, 0)``, ``(3, 1)`` and ``(2, 2)``): once through
the server's multiprocess registry (``vllm/v1/metrics/prometheus.py``) and once through the default
registry, which adds the ``_created`` samples. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

from __future__ import annotations

import pytest

from tools.flash_next.rental import spec_probe

# What a server started with --speculative-config serves on /metrics (the multiprocess registry).
PIN_SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="tiny"} 0.0
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="tiny"} 4.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="tiny"} 11.0
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="tiny"} 6.0
# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="0"} 3.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="1"} 2.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="2"} 1.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="tiny"} 10.0
"""

# The same counters through the default registry: every counter also has a `_created` gauge.
PIN_SAMPLE_WITH_CREATED = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="tiny"} 4.0
# HELP vllm:spec_decode_num_drafts_created Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_created gauge
vllm:spec_decode_num_drafts_created{engine="0",model_name="tiny"} 1.795162164402027e+09
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="tiny"} 11.0
# HELP vllm:spec_decode_num_draft_tokens_created Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_created gauge
vllm:spec_decode_num_draft_tokens_created{engine="0",model_name="tiny"} 1.795162164402039e+09
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="tiny"} 6.0
# HELP vllm:spec_decode_num_accepted_tokens_created Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_created gauge
vllm:spec_decode_num_accepted_tokens_created{engine="0",model_name="tiny"} 1.795162164402046e+09
# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="0"} 3.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="1"} 2.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="2"} 1.0
# HELP vllm:spec_decode_num_accepted_tokens_per_pos_created Accepted tokens per draft position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_created gauge
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="tiny",position="0"} 1.7951621644020538e+09
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="tiny",position="1"} 1.795162164402056e+09
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="tiny",position="2"} 1.795162164402059e+09
"""

# A server without a speculative config: SpecDecodingProm creates no counter at all.
NO_SPECULATION = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="tiny"} 1.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="tiny"} 10.0
"""

COUNTERS = {"drafts": 4, "draft_tokens": 11, "accepted": 6, "accepted_per_pos": [3, 2, 1]}


# --------------------------------------------------------------------------- the counters of one /metrics text


@pytest.mark.parametrize("text", [PIN_SAMPLE, PIN_SAMPLE_WITH_CREATED], ids=["multiprocess", "with_created"])
def test_the_counters_of_the_pins_exposition_text(text):
    assert spec_probe.parse_spec_counters(text) == COUNTERS


def test_the_counter_names_are_the_pins():
    # vllm/v1/spec_decode/metrics.py:229-231 and :255; prometheus_client appends `_total` in the text
    assert spec_probe.SPEC_DRAFTS == "vllm:spec_decode_num_drafts"
    assert spec_probe.SPEC_DRAFT_TOKENS == "vllm:spec_decode_num_draft_tokens"
    assert spec_probe.SPEC_ACCEPTED == "vllm:spec_decode_num_accepted_tokens"
    assert spec_probe.SPEC_ACCEPTED_PER_POS == "vllm:spec_decode_num_accepted_tokens_per_pos"


def test_a_server_without_speculation_has_no_counters():
    assert spec_probe.parse_spec_counters(NO_SPECULATION) is None
    assert spec_probe.parse_spec_counters("") is None


def test_the_counters_of_a_fresh_speculative_server_are_zero_not_absent():
    fresh = PIN_SAMPLE.replace(" 4.0", " 0.0").replace(" 11.0", " 0.0").replace(" 6.0", " 0.0")
    fresh = fresh.replace(" 3.0", " 0.0").replace(" 2.0", " 0.0").replace(" 1.0", " 0.0")
    assert spec_probe.parse_spec_counters(fresh) == {"drafts": 0, "draft_tokens": 0, "accepted": 0,
                                                     "accepted_per_pos": [0, 0, 0]}


def test_the_counters_are_summed_over_engines():
    two = PIN_SAMPLE + "\n".join([
        'vllm:spec_decode_num_drafts_total{engine="1",model_name="tiny"} 2.0',
        'vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="tiny"} 6.0',
        'vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="tiny"} 5.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="tiny",position="0"} 2.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="tiny",position="1"} 2.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="tiny",position="2"} 1.0',
    ]) + "\n"
    assert spec_probe.parse_spec_counters(two) == {"drafts": 6, "draft_tokens": 17, "accepted": 11,
                                                   "accepted_per_pos": [5, 4, 2]}


def test_the_position_label_is_found_wherever_it_stands():
    text = "\n".join([
        'vllm:spec_decode_num_drafts_total{model_name="a b",engine="0"} 5',
        'vllm:spec_decode_num_draft_tokens_total{model_name="a b",engine="0"} 10',
        'vllm:spec_decode_num_accepted_tokens_total{model_name="a b",engine="0"} 4',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1",model_name="a b",engine="0"} 1',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",model_name="a b",engine="0"} 3',
    ])
    assert spec_probe.parse_spec_counters(text) == {"drafts": 5, "draft_tokens": 10, "accepted": 4,
                                                    "accepted_per_pos": [3, 1]}


def test_read_spec_counters_asks_the_servers_metrics_endpoint():
    asked = []

    def get_text(url, timeout_s):
        asked.append((url, timeout_s))
        return PIN_SAMPLE

    assert spec_probe.read_spec_counters("http://127.0.0.1:8010/", get_text=get_text) == COUNTERS
    assert asked == [("http://127.0.0.1:8010/metrics", spec_probe.METRICS_TIMEOUT_S)]
    assert spec_probe.read_spec_counters("http://x", get_text=lambda url, timeout_s: NO_SPECULATION) is None


def test_read_spec_counters_raises_when_the_endpoint_does_not_answer():
    def refuses(url, timeout_s):
        raise OSError("connection refused")

    with pytest.raises(OSError):
        spec_probe.read_spec_counters("http://x", get_text=refuses)


# --------------------------------------------------------------------------- the difference of two readings


def test_the_delta_of_two_readings():
    after = {"drafts": 40, "draft_tokens": 119, "accepted": 60, "accepted_per_pos": [27, 20, 13]}
    assert spec_probe.counters_delta(COUNTERS, after) == {"drafts": 36, "draft_tokens": 108, "accepted": 54,
                                                          "accepted_per_pos": [24, 18, 12]}
    assert spec_probe.counters_delta(None, after) is None  # one reading failed, or no speculation
    assert spec_probe.counters_delta(COUNTERS, None) is None


def test_the_delta_pads_a_shorter_position_list():
    before = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "accepted_per_pos": []}
    after = {"drafts": 2, "draft_tokens": 6, "accepted": 3, "accepted_per_pos": [2, 1, 0]}
    assert spec_probe.counters_delta(before, after)["accepted_per_pos"] == [2, 1, 0]


# --------------------------------------------------------------------------- the acceptance arithmetic


def test_acceptance_length_is_one_plus_accepted_per_draft():
    # vLLM's own definition (SpecDecodingLogging.log, metrics.py:113-114): the bonus token counts
    assert spec_probe.acceptance_length(36, 54) == pytest.approx(2.5)
    assert spec_probe.acceptance_length(28, 37) == pytest.approx(1 + 37 / 28)
    assert spec_probe.acceptance_length(30, 0) == 1.0  # every draft rejected: one token per step
    assert spec_probe.acceptance_length(0, 0) is None  # no step with drafts: nothing to divide by


def test_the_summary_of_a_delta():
    summary = spec_probe.spec_summary({"drafts": 36, "draft_tokens": 108, "accepted": 54,
                                       "accepted_per_pos": [24, 18, 12]})
    assert summary["acceptance_length"] == pytest.approx(2.5)
    assert summary["draft_acceptance_rate"] == pytest.approx(0.5)  # accepted over drafted tokens
    assert summary["position_rates"] == [pytest.approx(24 / 36), pytest.approx(0.5), pytest.approx(12 / 36)]
    assert (summary["drafts"], summary["draft_tokens"], summary["accepted"]) == (36, 108, 54)
    assert spec_probe.spec_summary(None) is None
    empty = spec_probe.spec_summary({"drafts": 0, "draft_tokens": 0, "accepted": 0, "accepted_per_pos": [0, 0, 0]})
    assert empty["acceptance_length"] is None and empty["draft_acceptance_rate"] is None
    assert empty["position_rates"] == [None, None, None]
