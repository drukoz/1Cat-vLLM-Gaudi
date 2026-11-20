# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_STEP_TIMING_COMPONENTS: the step timer's component cuts on the tiny Flash-Next model (CPU).

``install_component_cuts`` finds the components by the attribute names of the upstream decoder layer
(``linear_attn``, ``self_attn``, ``mlp``, ``ple``, the two hyper-connections, the final mixer). This test holds it
to the real module tree: every component is found, every call of a forward is seen by the timer in mode 3, and the
forward returns the same tensor with and without the wrappers.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.extension import step_timing  # noqa: E402
from vllm_gaudi.extension.step_timing import MODE_SPANS, MODE_SYNC, StepTimer  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, ple_kwargs, prefill_metadata, reset_cpu_states)


class _Clock:

    def __init__(self):
        self.ns = 0

    def __call__(self):
        self.ns += 1000  # every clock read is one microsecond later
        return self.ns


class _Runtime:

    def __init__(self):
        self.waits = 0
        self.core = self
        self.hpu = self

    def mark_step(self):
        pass

    def synchronize(self):
        self.waits += 1


@pytest.fixture
def checkpoint(request) -> Path:
    """The tiny checkpoint. ``FLASH_NEXT_TINY_CKPT`` names a prebuilt one (a build takes minutes on a busy
    machine); without it the session's ``tiny_checkpoint`` is built as for every other test."""
    prebuilt = os.environ.get("FLASH_NEXT_TINY_CKPT", "")
    if prebuilt and (Path(prebuilt) / "model.safetensors").exists():
        return Path(prebuilt)
    return request.getfixturevalue("tiny_checkpoint")


def test_component_cuts_see_every_component_of_the_tiny_model(checkpoint, tmp_path):
    seq = [11, 5, 23, 7, 3, 17, 9, 2]
    with cpu_model(checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)

        def prefill():
            reset_cpu_states(model)
            n = len(seq)
            with torch.no_grad(), set_forward_context(prefill_metadata(n, 1), vllm_config):
                return model(input_ids=torch.tensor(seq), positions=torch.arange(n), **ple_kwargs(n, eos))

        expected = prefill()
        layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]
        gdn = sum(1 for layer in layers if getattr(layer, "linear_attn", None) is not None)
        qsa = sum(1 for layer in layers if getattr(layer, "self_attn", None) is not None)
        ple = sum(1 for layer in layers if getattr(layer, "ple", None) is not None)
        assert layers and gdn and qsa and ple and gdn + qsa == len(layers)

        runtime = _Runtime()
        timer = StepTimer("worker", mode=MODE_SYNC, every=1, out_dir=str(tmp_path), clock=_Clock(), wall=lambda: 1.0)
        timer.bind_runtime(runtime, runtime)
        counts = step_timing.install_component_cuts(timer, model)
        assert counts["gdn"] == gdn and counts["qsa"] == qsa and counts["ple"] == ple
        assert counts["moe"] == len(layers) and counts["moe_experts"] == len(layers)
        assert counts["hc"] == 3 * (2 * len(layers) + 1)  # three entry points per hyper-connection, final mixer too
        assert counts["qsa_indexer"] == qsa

        assert torch.equal(prefill(), expected) and runtime.waits == 0  # no step is open: nothing waits

        timer.begin_step()
        timer.tag("prefill", 1)
        got = prefill()
        timer.end_step()
        assert torch.equal(got, expected)
        with open(timer.path, encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        calls = {name: fields["calls"]["p50"] for name, fields in record["components"].items()}
        assert calls["gdn"] == gdn and calls["qsa"] == qsa and calls["ple"] == ple
        assert calls["moe"] == len(layers) and calls["moe_experts"] == len(layers)
        # per layer the mixer's and the MoE's hyper-connection, one more combine in front of a PLE layer that has
        # a pending one, and the final mixer
        assert 2 * len(layers) + 1 <= calls["hc"] <= 2 * len(layers) + 1 + ple
        assert runtime.waits == 2 * sum(v for name, v in calls.items() if name != "other")
        for name in ("hc", "gdn", "qsa", "moe", "moe_experts", "ple"):
            assert record["components"][name]["device"]["p50"] > 0

        timer.set_mode(MODE_SPANS)
        waits = runtime.waits
        timer.begin_step()
        timer.tag("prefill", 1)
        assert torch.equal(prefill(), expected)
        timer.end_step()
        assert runtime.waits == waits  # outside mode 3 the wrappers only pass the call on
        timer.close()
