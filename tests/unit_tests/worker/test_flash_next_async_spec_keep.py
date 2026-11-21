# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M1 (``VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS``): what a verification step keeps for the next one.

The next verification step advances from the inputs this one kept (its positions, ids and PLE context). Host-made
inputs can live in buffers that the next step's uploads rewrite before it reads them (engine-step's
``VLLM_HPU_PERSISTENT_INPUTS``: one buffer per input in a synchronous engine), so they are kept as copies; the inputs
M1 made are its own and are kept as they are.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from vllm_gaudi.v1.worker.flash_next_async_spec import VerifyInputAdvance, VerifyInputs  # noqa: E402
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner  # noqa: E402


def _runner(advance):
    return SimpleNamespace(_verify_advance=advance, _verify_step=5, device="cpu",
                           input_batch=SimpleNamespace(req_ids=["a", "b"]), _log_verify_advance=lambda bad: None)


def _inputs(padded=2, width=3):
    ids = torch.arange(padded * width, dtype=torch.int32).reshape(-1, 1)
    positions = torch.arange(padded * width, dtype=torch.int64).reshape(-1, 1) + 10
    context = torch.zeros((padded, 4), dtype=torch.int32)
    ple = (torch.tensor([0, width, 2 * width], dtype=torch.int32), context)
    return ids, positions, positions.clone(), torch.ones(padded, dtype=torch.int32), context, ple


def test_a_host_made_step_is_kept_as_copies(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS", "2")
    advance = VerifyInputAdvance()
    ids, positions, slots, accepted, context, ple = _inputs()
    HPUModelRunner._finish_verify_inputs(_runner(advance), None, 2, 2, 3, ids, positions, slots, accepted, ple)
    kept = advance._recorded.inputs
    assert kept.positions is not positions and kept.token_ids is not ids and kept.ngram_context is not context
    for tensor in (positions, ids, context):
        tensor.add_(100)  # the next step's upload into the same buffer
    assert int(kept.positions[0, 0]) == 10 and int(kept.token_ids[1, 0]) == 1 and int(kept.ngram_context.max()) == 0


def test_the_check_mode_keeps_copies_too(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS", "1")
    advance = VerifyInputAdvance()
    ids, positions, slots, accepted, context, ple = _inputs()
    made = VerifyInputs(ids.clone(), positions.clone(), slots.clone(), accepted.clone(), context.clone())
    HPUModelRunner._finish_verify_inputs(_runner(advance), made, 2, 2, 3, ids, positions, slots, accepted, ple)
    kept = advance._recorded.inputs
    assert kept.positions is not positions and kept.token_ids is not ids and kept.ngram_context is not context
    assert advance.stats.checked == 1 and advance.stats.mismatched_steps == 0


def test_a_device_made_step_is_kept_as_it_is(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS", "2")
    advance = VerifyInputAdvance()
    ids, positions, slots, accepted, context, ple = _inputs()
    made = VerifyInputs(ids + 1, positions + 1, slots, accepted, context + 1)
    HPUModelRunner._finish_verify_inputs(_runner(advance), made, 2, 2, 3, made.token_ids, made.positions, slots,
                                         accepted, ple)
    kept = advance._recorded.inputs
    assert kept.positions is made.positions and kept.token_ids is made.token_ids
    assert kept.ngram_context is made.ngram_context
