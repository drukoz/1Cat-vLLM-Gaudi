# SPDX-License-Identifier: Apache-2.0
"""CPU dummy-weight tests for the Flash-Next MTP path (M3-T2).

Fast checks (no HF oracle): the draft class is the registered ``HpuQwen4ExpMTP`` with the
HPU final mixer, the draft forward returns ``(sample [T, H], multi [T, hc*H])`` on dummy
weights, and under ``speculative_config.method == "mtp"`` both Flash-Next target subclasses
return ``(hidden, [multi])`` with ``multi == get_mtp_target_hidden_states()[:T]``. The
checkpoint has no MTP weights until T1, so the engine builds with ``load_format="dummy"``.
"""

import inspect
import os

os.environ.setdefault("VLLM_PLUGINS", "")  # register() is called by hand
os.environ["TMPDIR"] = "/tmp"  # the gloo init file must not sit on a sandboxed TMPDIR

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

import vllm_gaudi.models.qwen4_exp as fn  # noqa: E402
import vllm.models.qwen4_exp.nvidia.mtp as upstream_mtp  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402
from vllm_gaudi.models.qwen4_exp.mtp import HpuQwen4ExpMTP, mtp_aux_hidden  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_draft_model, cpu_model, eos_token_id, ple_kwargs, prefill_metadata)

MTP = {"method": "mtp", "num_speculative_tokens": 3}
HIDDEN = 128  # tiny_config.json hidden_size
MULTI = 512  # hc_count (4) * hidden_size (128)
VOCAB = 248320  # tiny_config.json vocab_size
DRAFT_ROWS = 3
TARGET_ROWS = 7


def test_registry_maps_mtp_to_hpu_class():
    """``Qwen4ExpMTP`` is registered to ``HpuQwen4ExpMTP`` (source-inspection style of test_registration)."""
    source = inspect.getsource(fn.register)
    assert "MTP_ARCHS" in source
    assert "vllm_gaudi.models.qwen4_exp.mtp:Hpu{arch}" in source
    assert "Qwen4ExpMTP" in fn.MTP_ARCHS
    fn.register()
    loaded = ModelRegistry._try_load_model_cls("Qwen4ExpMTP")
    assert loaded is HpuQwen4ExpMTP
    assert issubclass(HpuQwen4ExpMTP, upstream_mtp.Qwen4ExpMTP)


def test_mtp_gated_residual_rebound():
    """``register()`` rebinds the draft's final-mixer name to the HPU HyperConnection."""
    fn.register()
    assert upstream_mtp.GatedResidual is HpuGatedResidual


def test_draft_forward_returns_sample_and_multi(tiny_checkpoint):
    """The dummy draft is an ``HpuQwen4ExpMTP`` with the HPU mixer; a 3-row forward is ``(3,128),(3,512)``."""
    with cpu_model(tiny_checkpoint, speculative_config=MTP, load_format="dummy") as (vllm_config, _):
        draft = cpu_draft_model(vllm_config)
        assert type(draft) is HpuQwen4ExpMTP
        assert isinstance(draft.model.hyper_connection_mixer, HpuGatedResidual)

        ids = torch.tensor([5, 6, 7])
        positions = torch.arange(DRAFT_ROWS)
        hidden_states = torch.randn(DRAFT_ROWS, MULTI, dtype=torch.bfloat16)
        with set_forward_context(prefill_metadata(DRAFT_ROWS), vllm_config), torch.no_grad():
            out = draft(input_ids=ids, positions=positions, hidden_states=hidden_states, spec_step_idx=0)

        assert isinstance(out, tuple) and len(out) == 2
        sample, multi = out
        assert tuple(sample.shape) == (DRAFT_ROWS, HIDDEN)
        assert tuple(multi.shape) == (DRAFT_ROWS, MULTI)
        logits = draft.compute_logits(sample)
        assert tuple(logits.shape) == (DRAFT_ROWS, VOCAB)


def test_target_returns_hidden_and_multi(tiny_checkpoint):
    """Under ``method == "mtp"`` the target forward is ``(hidden, [multi])`` with ``multi == buffer[:T]``."""
    with cpu_model(tiny_checkpoint, speculative_config=MTP, load_format="dummy") as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        with set_forward_context(prefill_metadata(TARGET_ROWS), vllm_config), torch.no_grad():
            output = model(input_ids=torch.arange(100, 100 + TARGET_ROWS),
                           positions=torch.arange(TARGET_ROWS),
                           **ple_kwargs(TARGET_ROWS, eos))

        assert isinstance(output, tuple) and len(output) == 2
        hidden, aux = output
        assert isinstance(aux, list) and len(aux) == 1
        multi = aux[0]
        assert tuple(hidden.shape) == (TARGET_ROWS, HIDDEN)
        assert tuple(multi.shape) == (TARGET_ROWS, MULTI)
        assert torch.equal(multi, model.get_mtp_target_hidden_states()[:TARGET_ROWS])


def test_mtp_aux_hidden_guards():
    """``mtp_aux_hidden`` raises ``RuntimeError`` for a ``None`` buffer and for a buffer too small for ``T``."""
    hidden = torch.zeros(TARGET_ROWS, HIDDEN)
    with pytest.raises(RuntimeError):
        mtp_aux_hidden(hidden, None)
    too_small = torch.zeros(TARGET_ROWS - 1, MULTI)
    with pytest.raises(RuntimeError):
        mtp_aux_hidden(hidden, too_small)
    ok = mtp_aux_hidden(hidden, torch.zeros(TARGET_ROWS, MULTI))
    assert tuple(ok.shape) == (TARGET_ROWS, MULTI)


def test_target_plain_tensor_without_speculation(tiny_checkpoint):
    """Without a speculative config the target forward still returns a plain tensor."""
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        with set_forward_context(prefill_metadata(TARGET_ROWS), vllm_config), torch.no_grad():
            output = model(input_ids=torch.arange(100, 100 + TARGET_ROWS),
                           positions=torch.arange(TARGET_ROWS),
                           **ple_kwargs(TARGET_ROWS, eos))
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == (TARGET_ROWS, HIDDEN)
