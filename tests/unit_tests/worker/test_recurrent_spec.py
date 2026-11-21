# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the recurrent-spec lattice helpers (M3-T5a).

The helpers must be method-agnostic: the same lattice rules apply to DFlash2,
MTP and any future speculative method, so no helper may consult
``use_dflash()``.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.models.qwen4_exp.qsa as qsa_module
from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id
from vllm_gaudi.v1.worker.recurrent_spec import (
    accepted_counts,
    load_checkpoint_offsets,
    qsa_modules,
    recurrent_spec_enabled,
    state_slots_per_request,
    verification_state_width,
)


def _spec(num_speculative_tokens, use_dflash):
    return SimpleNamespace(num_speculative_tokens=num_speculative_tokens,
                           use_dflash=lambda: use_dflash)


class _MethodAgnosticConfig:
    """A speculative config whose use_dflash() must never be consulted."""

    def __init__(self, num_speculative_tokens):
        self.num_speculative_tokens = num_speculative_tokens

    def use_dflash(self):
        raise AssertionError("recurrent_spec helpers must not special-case DFlash2")


@pytest.mark.parametrize("method_config", [True, False], ids=["dflash", "mtp"])
def test_recurrent_spec_enabled(method_config):
    assert recurrent_spec_enabled(_spec(3, method_config))
    assert not recurrent_spec_enabled(_spec(0, method_config))
    assert not recurrent_spec_enabled(_spec(None, method_config))
    assert not recurrent_spec_enabled(None)
    assert recurrent_spec_enabled(_MethodAgnosticConfig(3))


@pytest.mark.parametrize("method_config", [True, False], ids=["dflash", "mtp"])
def test_state_slots_per_request(method_config):
    assert state_slots_per_request(_spec(3, method_config)) == 4
    assert state_slots_per_request(_spec(1, method_config)) == 2
    assert state_slots_per_request(_spec(0, method_config)) == 1
    assert state_slots_per_request(None) == 1
    assert state_slots_per_request(_MethodAgnosticConfig(3)) == 4


@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size k the Flash-Next tests run at, after flash_next_spec_sizes' startup check."""
    require_startup(request.param)
    return request.param


def test_one_checkpoint_per_token_at_every_size(spec_k):
    """k drafted tokens: 1 + k state slots per request, a lattice 1 + k wide, checkpoint offsets 0 .. k."""
    t = 1 + spec_k
    assert recurrent_spec_enabled(_MethodAgnosticConfig(spec_k))
    assert state_slots_per_request(_MethodAgnosticConfig(spec_k)) == t
    assert verification_state_width(t, True) == t
    assert verification_state_width(t, False) == 1
    full = {f"r{n}": n for n in range(1, t + 1)}  # every accepted count a step of k drafts can leave
    counts = accepted_counts(list(full), full, t + 2)
    assert counts == list(range(1, t + 1)) + [1, 1]
    assert load_checkpoint_offsets(counts, False) == list(range(t)) + [0, 0]
    assert max(load_checkpoint_offsets(counts, False)) == state_slots_per_request(_MethodAgnosticConfig(spec_k)) - 1
    assert load_checkpoint_offsets(counts, True) is None


def test_verification_state_width():
    assert verification_state_width(4, True) == 4
    assert verification_state_width(1, True) == 1
    assert verification_state_width(4, False) == 1
    assert verification_state_width(1, False) == 1


def test_accepted_counts_defaults_and_padding():
    counts = accepted_counts(["a", "b"], {"a": 3}, 2)
    assert counts == [3, 1]
    counts = accepted_counts(["a", "b"], {}, 4)
    assert counts == [1, 1, 1, 1]
    counts = accepted_counts([], {}, 3)
    assert counts == [1, 1, 1]


def test_accepted_counts_never_truncates():
    counts = accepted_counts(["a", "b", "c"], {"b": 2}, 2)
    assert counts == [1, 2, 1]


def test_accepted_counts_reads_recorded_values():
    counts = accepted_counts(["a", "b", "c"], {"a": 4, "c": 2}, 3)
    assert counts == [4, 1, 2]


def test_load_checkpoint_offsets():
    assert load_checkpoint_offsets([3, 1], True) is None
    assert load_checkpoint_offsets([3, 1], False) == [2, 0]
    assert load_checkpoint_offsets([], False) == []
    assert load_checkpoint_offsets([], True) is None


class _FakeQSA(torch.nn.Module):
    """Stand-in for HpuQwen4ExpQSAAttention.

    ``qsa_modules`` imports the class from ``vllm_gaudi.models.qwen4_exp.qsa``
    at call time, so monkeypatching that attribute swaps the class check;
    constructing the real layer needs a full VllmConfig + distributed setup,
    which is not cheap here.
    """


@pytest.fixture
def fake_qsa_class(monkeypatch):
    monkeypatch.setattr(qsa_module, "HpuQwen4ExpQSAAttention", _FakeQSA)
    return _FakeQSA


def _model_with(children):
    root = torch.nn.Module()
    for name, child in children.items():
        setattr(root, name, child)
    return root


def test_qsa_modules_target_only_and_none_entries(fake_qsa_class):
    target_qsa = fake_qsa_class()
    plain = torch.nn.Linear(2, 2)
    target = _model_with({"qsa": target_qsa, "plain": plain})
    assert qsa_modules(target) == [target_qsa]
    assert qsa_modules(target, None) == [target_qsa]
    assert qsa_modules(None, target) == [target_qsa]
    assert qsa_modules() == []
    assert qsa_modules(None) == []


def test_qsa_modules_target_first_then_drafter(fake_qsa_class):
    target_qsa = fake_qsa_class()
    drafter_qsa = fake_qsa_class()
    target = _model_with({"qsa": target_qsa})
    drafter = _model_with({"draft_qsa": drafter_qsa})
    assert qsa_modules(target, drafter) == [target_qsa, drafter_qsa]
    assert qsa_modules(target) == [target_qsa]
    assert qsa_modules(drafter) == [drafter_qsa]


def test_qsa_modules_finds_nested_and_skips_plain_layers(fake_qsa_class):
    inner_qsa = fake_qsa_class()
    inner = _model_with({"qsa": inner_qsa})
    drafter = _model_with({"inner": inner, "plain": torch.nn.Linear(2, 2)})
    assert qsa_modules(drafter) == [inner_qsa]
    assert qsa_modules(_model_with({"plain": torch.nn.Linear(2, 2)})) == []
