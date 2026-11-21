# SPDX-License-Identifier: Apache-2.0
"""CPU guard that keeps the dp_size==1 MoE fast path in sync with the pinned vLLM MoERunner.

Rental run 2 (2026-11-19): ``tiny_smoke.sh`` failed in both GDN modes with
``AttributeError: 'MoERunner' object has no attribute '_maybe_sync_shared_experts_stream'``.
The ``dp_size == 1`` branch of ``patched_fused_moe_forward`` mirrored a vLLM state
(2026-08-13) older than the pin (2026-09-04, ``8f816a3f``), where ``MoERunner._forward_impl``
launches the shared experts with ``self._shared_experts.maybe_forward_async(...)`` and
passes ``shared_experts_overlapping=`` to ``_apply_quant_method``. Commit ``ca676dd``
fixed the fast path; these tests are the CPU-side check that would have caught it on
the Mac. Test 1 inventories every ``self.<name>`` the fast path references and fails
when a name is absent from the pinned ``MoERunner``; test 2 drives the fast path with a
stub runner and binds the keywords it passes to ``_apply_quant_method`` against the
pinned signature.

Runs on CPU via ``tests/unit_tests/hpu_cpu_stubs`` (``hpu_fused_moe`` needs
``habana_frameworks``, which the stubs replace with mocks).
"""

import os

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()
os.environ.setdefault("VLLM_PLUGINS", "")

import inspect  # noqa: E402
import re  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # noqa: E402

import vllm_gaudi.ops.hpu_fused_moe as m  # noqa: E402


def test_fast_path_uses_only_pinned_runner_attributes():
    """Every ``self.<name>`` the fast path touches must exist on the pinned MoERunner."""
    # dir() holds methods and properties; instance attributes are set in __init__ or
    # other methods, so add every `self.<name> = ...` assignment in the class sources.
    allowed = set(dir(MoERunner))
    for cls in inspect.getmro(MoERunner):
        module = getattr(cls, "__module__", "")
        if not module.startswith("vllm."):
            continue
        allowed.update(re.findall(r"self\.(\w+)\s*(?::[^=]+)?=", inspect.getsource(cls)))

    referenced = set(re.findall(r"self\.([A-Za-z_]\w*)", inspect.getsource(m.patched_fused_moe_forward)))
    missing = sorted(referenced - allowed)
    assert not missing, (
        f"patched_fused_moe_forward references MoERunner attributes that do not exist at the pinned "
        f"vLLM (8f816a3f), which raises AttributeError on every MoE forward (rental run 2): {missing}")


def test_fast_path_dp1_binds_pinned_apply_quant_method():
    """The dp1 fast path launches shared experts via maybe_forward_async and passes
    only keywords the pinned ``MoERunner._apply_quant_method`` accepts."""
    sig = inspect.signature(MoERunner._apply_quant_method)
    calls = {}

    class Shared:

        def maybe_forward_async(self, x):
            calls["async"] = tuple(x.shape)
            return False

    class Runner:
        moe_config = SimpleNamespace(dp_size=1, pcp_size=1)
        routed_experts = SimpleNamespace(_ensure_moe_quant_config_init=lambda: calls.setdefault("init", True))
        _shared_experts = Shared()
        gate = None
        _fse_fuse_gate = False
        _fused_output_is_reduced = False

        def apply_routed_input_transform(self, h):
            return h, h

        def _maybe_pad_hidden_states(self, s, h):
            return h, None, None

        def _apply_quant_method(self, **kw):
            sig.bind(self, **kw)  # TypeError if a keyword is not in the pinned signature
            calls["kw"] = sorted(kw)
            return None, kw["hidden_states"] * 2

        def _maybe_combine(self, s, f):
            return f

        def _maybe_reduce_routed_output_before_transform(self, f, r):
            return f, r

        def _maybe_reduce_shared_expert_output(self, s, r):
            return s

        def _maybe_apply_routed_scale_to_output(self, s, f):
            return s, f

        def apply_routed_output_transform(self, f):
            return f

        def _maybe_reduce_final_output(self, r, og, red):
            return r

        def _maybe_add_zero_expert_output(self, r):
            return r

    x = torch.ones(3, 4)
    out = m.patched_fused_moe_forward(Runner(), x, torch.zeros(3, 2))

    assert torch.equal(out, x * 2)
    assert calls["async"] == (3, 4) and calls["init"] is True
    assert "shared_experts_overlapping" in calls["kw"]
