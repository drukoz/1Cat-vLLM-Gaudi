# SPDX-License-Identifier: Apache-2.0
"""CPU support for the MoE block tests: the plugin's HPU fast path with stand-ins for the two vendor ops.

The Flash-Next CPU parity tests run the engine's CPU MoE path.  The tests of the MoE glue
(``VLLM_HPU_MOE_GLUE``) need the plugin's own path instead: ``vllm_gaudi.ops.hpu_fused_moe`` (the
``MoERunner.forward`` replacement and the monolithic quant methods), which ends in
``torch.ops.hpu.mixture_of_experts`` and, for FP8 experts, ``torch.ops.hpu.cast_to_fp8_v2``.  Neither
exists without the Gaudi runtime, so :func:`install_vendor_ops` puts CPU functions in their place:

* ``mixture_of_experts``: what the plugin takes the op to compute, from its call sites: for every
  token, the sum over its routed experts that are local to the rank (``experts_min <= id <=
  experts_max``) of ``weight * down(silu(gate) * up)``, in fp32, rounded once to the weights' dtype.
  The experts are summed in id order, so the result does not depend on the order of the routing slots.
  It checks the contract the plugin relies on (int64 ids, weights in the activations' dtype, one weight
  pair per local expert) and records every call.
* ``cast_to_fp8_v2``: ``moe_reference.cast_to_fp8_reference`` with the plugin's ``FP8_MAX``.

Both are the same function with the flag at 0 and at 1, so a difference between the two builds is a
difference in what the glue hands to them.  Import this module before any ``vllm_gaudi`` import.

:func:`use_plugin_fast_path` also keeps the routed experts' weights in the checkpoint layout, as on a
card.  The engine picks the weight layout by platform: on a Gaudi machine (an out-of-tree platform) the
unquantized method leaves ``w13`` ``[E, 2I, H]`` and ``w2`` ``[E, H, I]`` alone, on CPU it prepacks them
for the CPU kernel, and the plugin's op would then be handed prepacked rows.  With the checkpoint
layout the plugin's path with the stand-in computes the model the engine's CPU path computes
(``test_moe_glue_whole_model.py`` checks it against the Hugging Face reference).

A test module that calls :func:`use_plugin_fast_path` must run in its own pytest process (the
repository's convention, ``spikes/mac-cpu-env/run_cpu_tests.sh``): the call replaces ``MoERunner.forward``
and the MoE method of every model built afterwards in the process.

:class:`OpRecorder` records the ops one forward dispatches.  A stand-in runs outside the recorder and is
one entry of the sequence (``hpu.mixture_of_experts``, ``hpu.cast_to_fp8_v2``), as the vendor op is one
op of the real path.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("VLLM_BUILD", "1.23.0")
os.environ.setdefault("TMPDIR", "/tmp")

from tests.unit_tests import hpu_cpu_stubs  # noqa: E402

hpu_cpu_stubs.install()

import hashlib  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes  # noqa: E402

from tests.unit_tests.hc_fused.moe_reference import cast_to_fp8_reference  # noqa: E402

BF16 = torch.bfloat16
VENDOR_CALLS: list[dict] = []  # one entry per mixture_of_experts call: its keyword arguments


class OpRecorder(TorchDispatchMode):
    """Every op one forward dispatches, in order: aten and custom ops by name, a vendor stand-in as one entry."""

    active: "OpRecorder | None" = None

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self.views = 0

    def __enter__(self):
        OpRecorder.active = self
        return super().__enter__()

    def __exit__(self, *exc):
        OpRecorder.active = None
        return super().__exit__(*exc)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(str(func))
        self.views += any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)
        return func(*args, **(kwargs or {}))

    @classmethod
    def note(cls, name: str) -> None:
        """Called by a vendor stand-in: one entry in the sequence of the active recorder."""
        if cls.active is not None:
            cls.active.names.append(name)

    def summary(self) -> tuple[int, int, str]:
        """``(ops, non-view ops, digest of the op-name sequence)``."""
        digest = hashlib.sha256("\n".join(self.names).encode()).hexdigest()[:16]
        return len(self.names), len(self.names) - self.views, digest

    def count(self, prefix: str) -> int:
        return sum(name.startswith(prefix) for name in self.names)


def _dequantized(weights, scales) -> list[torch.Tensor]:
    if scales is None:
        return [w.float() for w in weights]
    return [w.float() * s.float().reshape(-1, 1) for w, s in zip(weights, scales)]


def mixture_of_experts(*args, **kwargs) -> torch.Tensor:
    """CPU stand-in for ``torch.ops.hpu.mixture_of_experts`` (see the module docstring)."""
    assert not args, "the plugin calls the vendor op with keyword arguments only"
    VENDOR_CALLS.append(dict(kwargs))
    OpRecorder.note("hpu.mixture_of_experts")
    with _disable_current_modes():
        return _mixture_of_experts(**kwargs)


def _mixture_of_experts(**kwargs) -> torch.Tensor:
    hidden = kwargs["hidden_states"]
    ids = kwargs["expert_routing_table"]
    weights = kwargs["router_weights"]
    w12, w3 = kwargs["w12"], kwargs["w3"]
    experts_min, experts_max = kwargs["experts_min"], kwargs["experts_max"]
    assert kwargs["permuted_weights"] is True and kwargs["activation"] == "silu"
    assert ids.dtype == torch.int64 and ids.dim() == 2, f"routing table must be int64 [T, K], got {ids.dtype}"
    # the weights have the dtype of the block's activations: bf16, or fp32 in the fp32 debug build of the model
    assert weights.dtype in (BF16, torch.float32) and weights.shape == ids.shape, "router weights must be [T, K]"
    assert len(w12) == len(w3) == experts_max - experts_min + 1, "one weight pair per local expert"
    scale = kwargs.get("d_scale_hidden_states")
    if hidden.dtype == torch.float8_e4m3fn:
        assert scale is not None and scale.dtype == torch.float32 and tuple(scale.shape) == (hidden.shape[0], 1)
        x = hidden.float() * scale
    else:
        assert hidden.dtype == weights.dtype and scale is None
        x = hidden.float()
    if hidden.is_meta:
        return hidden.new_empty(hidden.shape, dtype=weights.dtype)
    gate_up = _dequantized(w12, kwargs.get("d_scale_w12"))
    down = _dequantized(w3, kwargs.get("d_scale_w3"))
    out = torch.zeros(x.shape, dtype=torch.float32)
    for local, (w_gate_up, w_down) in enumerate(zip(gate_up, down)):
        routed = (ids == experts_min + local)
        if not bool(routed.any()):
            continue
        token_weight = (weights.float() * routed).sum(dim=1, keepdim=True)
        projected = x @ w_gate_up.t()
        d = projected.shape[1] // 2
        out += token_weight * ((F.silu(projected[:, :d]) * projected[:, d:]) @ w_down.t())
    return out.to(weights.dtype)


def install_vendor_ops(fp8_max: float | None = None) -> None:
    """Put the CPU stand-ins in ``torch.ops.hpu`` (idempotent).  ``fp8_max``: the clip of the cast stand-in."""

    def cast_to_fp8_v2(data, scale_inv, *_rest, **_kwargs):
        import vllm_gaudi.extension.ops as plugin_ops

        OpRecorder.note("hpu.cast_to_fp8_v2")
        with _disable_current_modes():
            return cast_to_fp8_reference(data, scale_inv, plugin_ops.FP8_MAX if fp8_max is None else fp8_max)

    torch.ops.hpu.mixture_of_experts = mixture_of_experts
    torch.ops.hpu.cast_to_fp8_v2 = cast_to_fp8_v2


_CHECKPOINT_LAYOUT_INSTALLED = False


def _keep_checkpoint_weight_layout() -> None:
    """Make the plugin's unquantized MoE method see the backend a Gaudi machine selects (out-of-tree).

    ``select_unquantized_moe_backend`` answers by platform; on CPU it answers "CPU", and the engine's
    ``process_weights_after_loading`` then prepacks ``w13`` and ``w2`` for the CPU kernel before the plugin
    hands their rows to ``mixture_of_experts``.  With the out-of-tree answer the weights stay as loaded.
    """
    global _CHECKPOINT_LAYOUT_INSTALLED
    if _CHECKPOINT_LAYOUT_INSTALLED:
        return
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend

    import vllm_gaudi.ops.hpu_fused_moe as fused_moe

    original = fused_moe.HPUUnquantizedFusedMoEMethod.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.unquantized_backend = UnquantizedMoeBackend.OOT
        self.experts_cls = None

    fused_moe.HPUUnquantizedFusedMoEMethod.__init__ = init
    _CHECKPOINT_LAYOUT_INSTALLED = True


def use_plugin_fast_path() -> None:
    """Import the plugin's MoE modules, so that models built afterwards run its fast path, and install the stand-ins.

    Call it once per process, before a model is built; the module that does must run in its own process.
    """
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401  registers the monolithic method, replaces MoERunner.forward

    install_vendor_ops()
    _keep_checkpoint_weight_layout()


def moe_blocks(model) -> list:
    """The MoE blocks of a built tiny model, in layer order."""
    return [layer.mlp for layer in model.language_model.model.layers]


def to_fp8_experts(mlp) -> None:
    """Replace the bf16 routed experts of one built MoE block by per-channel FP8 experts.

    This is the state ``HPUFp8MoEMethod.process_weights_after_loading`` leaves the real checkpoint's experts
    in (``VLLM_HPU_FORCE_CHANNEL_FP8`` default): ``VllmMixtureOfExpertsOpFP8PerChannel`` with one FP8 weight
    and one fp32 scale per output channel, dynamic input quantization.  The weights are quantized here from
    the block's bf16 weights (the vendor stand-in dequantizes them again).
    """
    import vllm_gaudi.extension.ops as plugin_ops
    import vllm_gaudi.ops.hpu_fp8 as plugin_fp8

    routed = mlp.experts.routed_experts
    old = routed.moe_op
    local = routed.local_num_experts
    op = plugin_ops.VllmMixtureOfExpertsOpFP8PerChannel(routed.global_num_experts, local, old.experts_min,
                                                        old.experts_max, None)
    for expert in range(local):
        for holder, weight in ((op.w13_list[expert], routed.w13_weight.data[expert]),
                               (op.w2_list[expert], routed.w2_weight.data[expert])):
            scale = (weight.float().abs().amax(dim=1) + 1e-8) / float(plugin_ops.FP8_MAX)
            holder.set_weight((weight.float() / scale.reshape(-1, 1)).to(torch.float8_e4m3fn))
            holder.set_scale_inv_fp8(scale)
    op._cache_weight_lists()
    routed.moe_op = op
    old_method = routed.quant_method
    method = object.__new__(plugin_fp8.HPUFp8MoEMethod)
    method.__dict__.update(old_method.__dict__)
    method.has_moe_quant_config = True
    method.use_dispatch_fn = False
    if method.moe_quant_config is None:
        # The engine builds this on the first forward from the method's own attributes; the FP8 method would
        # read FP8 layer attributes this converted bf16 block does not have. The fast path only needs it set.
        method.moe_quant_config = old_method.get_fused_moe_quant_config(routed)
    if isinstance(old_method, torch.nn.Module):
        routed._modules["quant_method"] = method
    else:
        routed.quant_method = method
