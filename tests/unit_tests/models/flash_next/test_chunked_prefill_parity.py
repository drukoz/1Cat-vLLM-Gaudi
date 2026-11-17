# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill and block-crossing decode parity on CPU (M1-T5).

A 700-token prompt prefilled in two chunks whose state moves from slot to slot
(``load_indices_tensor`` != ``store_indices_tensor``, the shape prefix caching gives a
resumed prompt) must give the same logits as a single-shot prefill: the first chunk
(load 2, store 3) must reproduce the reference row 511 and leave its load slot untouched,
and the second chunk (load 3, store 4, ``has_initial_states_p=[1]``) must reproduce the
reference row 699 and leave the slot it read from alone. A single-token decode that steps
across a 128-token block boundary with load 5 / store 6 must match a fresh prefill of the
same sequence, and the decode must not touch the slot it read.

Zeroing the middle slot between the chunks (the negative control) must move the logits, so
the parity checks above really do depend on the carried state. It is asserted on the first
rows of the second chunk: with 188 tokens between the boundary and row 699 the GDN decay
damps the carried state to ~2e-4 there, which is above the parity tolerance but far below
the 0.1 the boundary rows show.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.test_decode_parity import install_cached_attention  # noqa: E402

FP32_TOL = 1e-5
BF16_TOL = 3e-2
LOST_STATE_MIN_GAP = 0.1
REF_ROWS_DISTINCT_MIN = 1e-3

PROMPT_LEN = 700
CHUNK_A_LEN = 512  # 4 * 128: a whole number of GDN chunks, so the decode at 512 crosses a block boundary
NEG_CONTROL_ROWS = 4  # rows right after the boundary, where the carried state is still load-bearing
BOS_TOKEN_ID = 248044
NUM_SLOTS = 8
ATTN_SLOT = 1  # the CachedSdpaAttention K/V key: kept apart from the GDN/PLE state slots under test

LOAD_SLOT, STORE_SLOT = 2, 3
STORE_SLOT_B = 4
PREFILL_SLOT, DECODE_LOAD_SLOT, DECODE_STORE_SLOT = 5, 6, 6


def prompt_tokens() -> list[int]:
    """The deterministic 700-token prompt (BOS plus 699 ids drawn with seed 7)."""
    drawn = torch.randint(1000, 20000, (PROMPT_LEN - 1,), generator=torch.Generator().manual_seed(7))
    return [BOS_TOKEN_ID] + drawn.tolist()


def enable_slot_reuse(vllm_config) -> None:
    """Let the GDN layer keep distinct load/store state-index tensors.

    ``HPUGatedDeltaNetAttention._extract_metadata`` collapses ``store_indices_tensor`` onto
    ``load_indices_tensor`` when the cache config has no prefix caching, because that runner
    mode always builds equal tensors. Distinct load/store slots only occur under prefix
    caching, so the config has to say so for the harness metadata to reach the kernels.
    """
    vllm_config.cache_config.enable_prefix_caching = True


def state_modules(model) -> list:
    """The modules carrying hand-bound state caches: the 3 GDN layers and the PLE layer."""
    return [m for m in model.modules() if isinstance(m, MambaBase)]


def slot_tensors(modules, slot: int) -> list[torch.Tensor]:
    """``kv_cache[k][slot]`` for every state tensor of every module, flattened."""
    return [module.kv_cache[k][slot] for module in modules for k in range(len(module.kv_cache))]


def reset(model, atts) -> None:
    """Zero every GDN/PLE state cache and every cached-attention K/V cache."""
    reset_cpu_states(model)
    for att in atts:
        att.reset()
        att.slot = ATTN_SLOT


def assert_slot_untouched(modules, slot: int, label: str) -> None:
    """Every state tensor of ``slot`` is all zeros (the slot was never written)."""
    for idx, tensor in enumerate(slot_tensors(modules, slot)):
        assert not bool(tensor.any()), f"{label}: state tensor {idx} of slot {slot} was written (expected all zeros)"


def assert_slot_written(modules, slot: int, label: str) -> None:
    """Every state tensor of ``slot`` holds at least one non-zero value."""
    for idx, tensor in enumerate(slot_tensors(modules, slot)):
        assert bool(tensor.any()), f"{label}: state tensor {idx} of slot {slot} is still all zeros"


def assert_slot_equal(modules, slot: int, saved: list[torch.Tensor], label: str) -> None:
    """Every state tensor of ``slot`` equals the clone taken before the step under test."""
    for idx, (now, before) in enumerate(zip(slot_tensors(modules, slot), saved)):
        assert torch.equal(now, before), (f"{label}: state tensor {idx} of slot {slot} changed; max |delta| = "
                                          f"{(now.float() - before.float()).abs().max().item():.3e}")


def prefill_logits(model, vllm_config, atts, token_ids, start, metadata, ple) -> torch.Tensor:
    """fp32 logits ``(len(token_ids), vocab)`` for one prefill forward under ``metadata``."""
    for att in atts:
        att.slot = ATTN_SLOT
    with torch.no_grad(), set_forward_context(metadata, vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids),
                       positions=torch.arange(start, start + len(token_ids)),
                       **ple)
    return fp32_logits(model, hidden)


def decode_logits(model, vllm_config, atts, token, position, metadata, ple) -> torch.Tensor:
    """fp32 logits ``(1, vocab)`` for one decode step under ``metadata``."""
    for att in atts:
        att.slot = ATTN_SLOT
    with torch.no_grad(), set_forward_context(metadata, vllm_config):
        hidden = model(input_ids=torch.tensor([token]), positions=torch.tensor([position]), **ple)
    return fp32_logits(model, hidden)


def decode_metadata(load_slot: int, store_slot: int) -> SimpleNamespace:
    """The runner's single-token decode metadata with distinct load and store state slots."""
    return SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=torch.tensor([load_slot]),
        store_indices_tensor=torch.tensor([store_slot]),
        query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32),
        has_initial_states_p=None,
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        dflash_full_query=False,
    )


def single_shot_logits(model, vllm_config, atts, tokens, eos, slot=1) -> torch.Tensor:
    """fp32 logits of a fresh single-shot prefill of ``tokens`` (state in ``slot``, clean caches)."""
    reset(model, atts)
    return prefill_logits(model, vllm_config, atts, tokens, 0, prefill_metadata(len(tokens), slot=slot),
                          ple_kwargs(len(tokens), eos))


def test_reference_prefill_rows(tiny_checkpoint):
    """fp32 single-shot prefill of the 700-token prompt: rows 511 and 699 are finite and distinct."""
    tokens = prompt_tokens()
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        logits = single_shot_logits(model, vllm_config, atts, tokens, eos)
        assert logits.shape == (PROMPT_LEN, logits.shape[-1])
        for row in (CHUNK_A_LEN - 1, PROMPT_LEN - 1):
            assert torch.isfinite(logits[row]).all(), f"row {row} of the reference prefill is not finite"
        gap = (logits[CHUNK_A_LEN - 1] - logits[PROMPT_LEN - 1]).abs().max().item()
        assert gap >= REF_ROWS_DISTINCT_MIN, (
            f"reference rows {CHUNK_A_LEN - 1} and {PROMPT_LEN - 1} differ by only {gap:.3e} < "
            f"{REF_ROWS_DISTINCT_MIN:.0e}: the prompt does not exercise distinct positions")


def test_chunked_prefill_matches_single_shot(tiny_checkpoint):
    """fp32: chunk (load 2, store 3) then chunk (load 3, store 4, has_initial) equal the single-shot rows."""
    tokens = prompt_tokens()
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        modules = state_modules(model)
        ref = single_shot_logits(model, vllm_config, atts, tokens, eos)

        reset(model, atts)
        chunk_a = prefill_logits(model,
                                 vllm_config,
                                 atts,
                                 tokens[:CHUNK_A_LEN],
                                 0,
                                 prefill_metadata(CHUNK_A_LEN, load_slot=LOAD_SLOT, store_slot=STORE_SLOT),
                                 ple_kwargs(CHUNK_A_LEN, eos))
        gap_a = (chunk_a[CHUNK_A_LEN - 1] - ref[CHUNK_A_LEN - 1]).abs().max().item()
        assert gap_a <= FP32_TOL, (f"chunk A (tokens 0..{CHUNK_A_LEN - 1}, load {LOAD_SLOT} -> store {STORE_SLOT}): "
                                   f"row {CHUNK_A_LEN - 1} vs single-shot reference: "
                                   f"max |delta logit| = {gap_a:.3e} > {FP32_TOL:.0e}")
        assert_slot_untouched(modules, LOAD_SLOT, "after chunk A")
        assert_slot_written(modules, STORE_SLOT, "after chunk A")

        stored_state = [tensor.clone() for tensor in slot_tensors(modules, STORE_SLOT)]
        chunk_b = prefill_logits(
            model,
            vllm_config,
            atts,
            tokens[CHUNK_A_LEN:],
            CHUNK_A_LEN,
            prefill_metadata(PROMPT_LEN - CHUNK_A_LEN,
                             load_slot=STORE_SLOT,
                             store_slot=STORE_SLOT_B,
                             has_initial_state=True),
            ple_kwargs(PROMPT_LEN - CHUNK_A_LEN, eos, context=(tokens[CHUNK_A_LEN - 2], tokens[CHUNK_A_LEN - 1])))
        gap_b = (chunk_b[-1] - ref[PROMPT_LEN - 1]).abs().max().item()
        assert gap_b <= FP32_TOL, (f"chunk B (tokens {CHUNK_A_LEN}..{PROMPT_LEN - 1}, load {STORE_SLOT} -> "
                                   f"store {STORE_SLOT_B}, has_initial_state): last row vs single-shot reference: "
                                   f"max |delta logit| = {gap_b:.3e} > {FP32_TOL:.0e}")
        assert_slot_untouched(modules, LOAD_SLOT, "after chunk B")
        assert_slot_equal(modules, STORE_SLOT, stored_state, "after chunk B (chunk B read this slot)")


def test_chunked_prefill_state_is_load_bearing(tiny_checkpoint):
    """Negative control: zeroing the middle slot between the chunks moves chunk B's first rows by > 0.1.

    The assertion is on the rows right after the boundary, not on row 699: the GDN decay makes the
    state carried across the boundary contribute ~2e-4 to the last row's logits on the tiny model
    (still 20x the 1e-5 parity tolerance, asserted below), so a 0.1 gap only exists while the
    boundary is close behind the position being read.
    """
    tokens = prompt_tokens()
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        modules = state_modules(model)
        ref = single_shot_logits(model, vllm_config, atts, tokens, eos)

        reset(model, atts)
        prefill_logits(model,
                       vllm_config,
                       atts,
                       tokens[:CHUNK_A_LEN],
                       0,
                       prefill_metadata(CHUNK_A_LEN, load_slot=LOAD_SLOT, store_slot=STORE_SLOT),
                       ple_kwargs(CHUNK_A_LEN, eos))
        for tensor in slot_tensors(modules, STORE_SLOT):
            tensor.zero_()
        chunk_b = prefill_logits(
            model,
            vllm_config,
            atts,
            tokens[CHUNK_A_LEN:],
            CHUNK_A_LEN,
            prefill_metadata(PROMPT_LEN - CHUNK_A_LEN,
                             load_slot=STORE_SLOT,
                             store_slot=STORE_SLOT_B,
                             has_initial_state=True),
            ple_kwargs(PROMPT_LEN - CHUNK_A_LEN, eos, context=(tokens[CHUNK_A_LEN - 2], tokens[CHUNK_A_LEN - 1])))
        near = chunk_b[:NEG_CONTROL_ROWS]
        far = (chunk_b[-1] - ref[PROMPT_LEN - 1]).abs().max().item()
        for offset in range(NEG_CONTROL_ROWS):
            gap = (near[offset] - ref[CHUNK_A_LEN + offset]).abs().max().item()
            assert gap > LOST_STATE_MIN_GAP, (
                f"wiping the state slot between the chunks moved row {CHUNK_A_LEN + offset} by only {gap:.3e} <= "
                f"{LOST_STATE_MIN_GAP}: the parity checks would not detect a dropped state")
        assert far > FP32_TOL, (
            f"wiping the state slot between the chunks moved the last row by only {far:.3e} <= {FP32_TOL:.0e}: "
            f"row {PROMPT_LEN - 1} would not notice a dropped state either")


def test_decode_across_block_boundary_uses_store_slot(tiny_checkpoint):
    """fp32: decode across the 512-token boundary with load 5 / store 6 matches a fresh prefill, slot 5 intact."""
    tokens = prompt_tokens()
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        modules = state_modules(model)

        reset(model, atts)
        prefill_logits(model,
                       vllm_config,
                       atts,
                       tokens[:CHUNK_A_LEN],
                       0,
                       prefill_metadata(CHUNK_A_LEN, slot=PREFILL_SLOT),
                       ple_kwargs(CHUNK_A_LEN, eos))
        post_prefill = [tensor.clone() for tensor in slot_tensors(modules, PREFILL_SLOT)]

        decode_logits(model, vllm_config, atts, tokens[CHUNK_A_LEN], CHUNK_A_LEN,
                      decode_metadata(PREFILL_SLOT, DECODE_STORE_SLOT),
                      ple_kwargs(1, eos, context=(tokens[CHUNK_A_LEN - 2], tokens[CHUNK_A_LEN - 1])))
        second = decode_logits(model, vllm_config, atts, tokens[CHUNK_A_LEN + 1], CHUNK_A_LEN + 1,
                               decode_metadata(DECODE_LOAD_SLOT, DECODE_STORE_SLOT),
                               ple_kwargs(1, eos, context=(tokens[CHUNK_A_LEN - 1], tokens[CHUNK_A_LEN])))

        assert_slot_written(modules, DECODE_STORE_SLOT, "after the two decode steps")
        assert_slot_equal(modules, PREFILL_SLOT, post_prefill,
                          f"after the decode steps (they read slot {PREFILL_SLOT})")

        fresh = single_shot_logits(model, vllm_config, atts, tokens[:CHUNK_A_LEN + 2], eos)
        gap = (second[-1] - fresh[CHUNK_A_LEN + 1]).abs().max().item()
        assert gap <= FP32_TOL, (f"decode of token {CHUNK_A_LEN + 1} after crossing the block boundary vs row "
                                 f"{CHUNK_A_LEN + 1} of a fresh {CHUNK_A_LEN + 2}-token prefill: "
                                 f"max |delta logit| = {gap:.3e} > {FP32_TOL:.0e}")


def test_chunked_prefill_bf16(tiny_checkpoint):
    """bf16: the two-chunk flow of test_chunked_prefill_matches_single_shot with a 3e-2 tolerance."""
    tokens = prompt_tokens()
    with cpu_model(tiny_checkpoint, fp32=False, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        modules = state_modules(model)
        ref = single_shot_logits(model, vllm_config, atts, tokens, eos)

        reset(model, atts)
        chunk_a = prefill_logits(model,
                                 vllm_config,
                                 atts,
                                 tokens[:CHUNK_A_LEN],
                                 0,
                                 prefill_metadata(CHUNK_A_LEN, load_slot=LOAD_SLOT, store_slot=STORE_SLOT),
                                 ple_kwargs(CHUNK_A_LEN, eos))
        gap_a = (chunk_a[CHUNK_A_LEN - 1] - ref[CHUNK_A_LEN - 1]).abs().max().item()
        assert gap_a <= BF16_TOL, (f"bf16 chunk A: row {CHUNK_A_LEN - 1} vs single-shot reference: "
                                   f"max |delta logit| = {gap_a:.3e} > {BF16_TOL:.0e}")
        assert_slot_untouched(modules, LOAD_SLOT, "bf16 after chunk A")
        assert_slot_written(modules, STORE_SLOT, "bf16 after chunk A")

        stored_state = [tensor.clone() for tensor in slot_tensors(modules, STORE_SLOT)]
        chunk_b = prefill_logits(
            model,
            vllm_config,
            atts,
            tokens[CHUNK_A_LEN:],
            CHUNK_A_LEN,
            prefill_metadata(PROMPT_LEN - CHUNK_A_LEN,
                             load_slot=STORE_SLOT,
                             store_slot=STORE_SLOT_B,
                             has_initial_state=True),
            ple_kwargs(PROMPT_LEN - CHUNK_A_LEN, eos, context=(tokens[CHUNK_A_LEN - 2], tokens[CHUNK_A_LEN - 1])))
        gap_b = (chunk_b[-1] - ref[PROMPT_LEN - 1]).abs().max().item()
        assert gap_b <= BF16_TOL, (f"bf16 chunk B: last row vs single-shot reference: "
                                   f"max |delta logit| = {gap_b:.3e} > {BF16_TOL:.0e}")
        assert_slot_untouched(modules, LOAD_SLOT, "bf16 after chunk B")
        assert_slot_equal(modules, STORE_SLOT, stored_state, "bf16 after chunk B (chunk B read this slot)")


def test_null_load_slot_reads_zero_state(tiny_checkpoint):
    """A prefill with load slot 0 (the null block) and has_initial_states_p=[1] ignores whatever slot 0 holds (#1895).

    Slot 0 of every GDN and PLE state tensor is filled with NaN. The logits must be finite and equal
    to a fresh prefill of the same tokens (has_initial_states_p=[0], clean caches).
    """
    tokens = prompt_tokens()[:CHUNK_A_LEN]
    with cpu_model(tiny_checkpoint, fp32=True, num_slots=NUM_SLOTS) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        modules = state_modules(model)
        reset(model, atts)
        ref = prefill_logits(model, vllm_config, atts, tokens, 0,
                             prefill_metadata(CHUNK_A_LEN, load_slot=0, store_slot=STORE_SLOT),
                             ple_kwargs(CHUNK_A_LEN, eos))
        reset(model, atts)
        for tensor in slot_tensors(modules, 0):
            tensor.fill_(float("nan"))
        got = prefill_logits(model, vllm_config, atts, tokens, 0,
                             prefill_metadata(CHUNK_A_LEN, load_slot=0, store_slot=STORE_SLOT, has_initial_state=True),
                             ple_kwargs(CHUNK_A_LEN, eos))
        assert bool(torch.isfinite(got).all()), "a layer read the NaN in null block 0 as its initial state"
        torch.testing.assert_close(got, ref, rtol=0, atol=FP32_TOL)
        # Zero the NaN before leaving the context: freed state blocks return to torch's CPU caching
        # allocator, and a later cpu_model build's uninitialized (checkpoint-padding) weight rows
        # can reuse them and read as non-finite.
        reset(model, atts)
