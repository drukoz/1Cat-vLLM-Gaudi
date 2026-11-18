# SPDX-License-Identifier: Apache-2.0
"""TP1 sparse-QSA decode parity and side-cache state checks on CPU (M2-T4).

The 2600-token reference prompt prefilled in ``QsaMode.SPARSE``, then 16 teacher-forced
decode steps (positions 2600..2615, the reference's ``generated_ids``), must reproduce
HF's greedy picks and the logits of a fresh sparse prefill of the same 2616-token
sequence (within 1e-5 fp32, 3e-2 bf16). The decode branch of ``HpuQsaIndexer.select``
writes the raw key into the ring and, at every 4-aligned position, the compressed block
key into ``compressed_key_cache``; the side-cache test compares the rows the chain wrote
with a fresh sparse prefill of the same sequence.

The ring cannot be compared to a fresh prefill of the FULL 2616-token sequence: 2616 is
a whole number of 4-token blocks, so the prefill has no tail and ``select`` writes NO
ring rows there (the ring of a fresh 2616-prefill stays zero). The fresh reference ring
is produced the only way a prefill path can reach those rows: a 2615-token prefill
(tail 3 writes the rows for positions 2612..2614) plus the decode step at position 2615
(row for 2615).

The slot-move test walks the state from slot 2 to slot 3 with a ``load_slot=2 /
store_slot=3`` decode step at the 4-aligned position 2604 (prefix caching's slot move);
the negative control zeroes compressed rows 600..649 right before the decode step at
position 2610 and asserts the selection changes, so the parity checks above are
sensitive to the side caches.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, reset_cpu_states)
from tests.unit_tests.models.flash_next.test_chunked_prefill_parity import (  # noqa: E402
    assert_slot_equal, assert_slot_untouched, assert_slot_written, enable_slot_reuse, slot_tensors, state_modules)
from tests.unit_tests.models.flash_next.test_sparse_prefill_parity import (  # noqa: E402
    LONG_PROMPT, SPARSE_MAX_MODEL_LEN, model_sparse_attns, sparse_prefill_last_logits)

FP32_TOL = 1e-5  # the sparse decode reproduces HF's selection; the residual is fp32 reassociation
BF16_TOL = 3e-2  # bf16 decode vs fresh prefill (the M0 decode parity tolerance)
ROW_TOL = 1e-5  # side-cache rows written on the decode path vs a fresh prefill: fp32 reassociation (FP32_TOL)
FULL_LEN = LONG_PROMPT + 16  # prompt + generated_ids: 2616 tokens, positions 2600..2615 are the decode steps
DECODE_SLOT = 1  # the state slot of the plain decode chains
STATE_SLOT_A, STATE_SLOT_B = 2, 3  # the slots of the slot-move chain (num_slots=4)
MOVE_POS = 2604  # 4-aligned: the first position of block 651, moved from slot 2 to slot 3
NEG_POS = 2610  # the negative-control step (not a block completion)
ZERO_ROWS = (600, 650)  # compressed rows zeroed for the negative control (prompt blocks + rows 650 written by 2603)


def qsa_layers(model) -> list:
    return [m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]


def load_long_prompt(tiny_checkpoint) -> dict:
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    return next(p for p in reference["prompts"] if len(p["prompt_ids"]) == LONG_PROMPT)


def sparse_decode_step(model, vllm_config, atts, seq, pos, eos, att_slot=1, *, load_slot=None, store_slot=None):
    """fp32 last-position logits ``(248320,)`` for one teacher-forced decode step of token ``seq[pos]``.

    ``load_slot``/``store_slot`` pass through to ``decode_metadata`` (each defaults to the
    other); the PLE left context is the two preceding tokens of ``seq`` (as in
    ``test_decode_parity.decode``). ``att_slot`` is the sparse-attention K/V cache key.
    """
    load = load_slot if load_slot is not None else store_slot
    store = store_slot if store_slot is not None else load_slot
    for att in atts:
        att.slot = att_slot
    ctx = (seq[pos - 2], seq[pos - 1])
    with torch.no_grad(), set_forward_context(decode_metadata(store, pos, load_slot=load, store_slot=store),
                                              vllm_config):
        hidden = model(input_ids=torch.tensor([seq[pos]]),
                       positions=torch.tensor([pos]),
                       **ple_kwargs(1, eos, context=ctx))
    return fp32_logits(model, hidden)[-1]


def run_sparse_decode_chain(model, vllm_config, atts, prompt_ids, gen_ids, slot, eos):
    """Prefill ``prompt_ids``, then 16 teacher-forced decode steps at positions 2600..2615.

    Returns ``(picks, last_logits, full_seq)``: the greedy pick of the prefill and of the
    first 15 decode steps (16 picks, the convention of ``test_decode_parity.run_decode_chain``),
    the fp32 logits of the step at position 2615, and the full 2616-token sequence.
    """
    lg = sparse_prefill_last_logits(model, vllm_config, eos, prompt_ids, slot=slot)
    seq = list(prompt_ids)
    picks = [int(lg.argmax())]
    for i, tf in enumerate(gen_ids):
        seq.append(tf)
        lg = sparse_decode_step(model, vllm_config, atts, seq, len(seq) - 1, eos, att_slot=slot, load_slot=slot)
        if i + 1 < len(gen_ids):
            picks.append(int(lg.argmax()))
    return picks, lg, seq


def snapshot_state(model) -> dict:
    """Clone every tensor a forward mutates: GDN/PLE state, indexer side caches, sparse-attention K/V."""
    qsa = qsa_layers(model)
    return {
        "state": [[t.clone() for t in m.kv_cache] for m in state_modules(model)],
        "side": [(m.indexer.compressed_key_cache.clone(), m.indexer.key_ring.clone()) for m in qsa],
        "att": [{s: (k.clone(), v.clone()) for s, (k, v) in m.sparse_attn.cache.items()} for m in qsa],
    }


def restore_state(model, snap: dict) -> None:
    """Copy a :func:`snapshot_state` back (the caches are the same objects, so clones stay valid)."""
    for module, saved in zip(state_modules(model), snap["state"]):
        for tensor, was in zip(module.kv_cache, saved):
            tensor.copy_(was)
    for module, (comp, ring) in zip(qsa_layers(model), snap["side"]):
        module.indexer.compressed_key_cache.copy_(comp)
        module.indexer.key_ring.copy_(ring)
    for module, cache in zip(qsa_layers(model), snap["att"]):
        module.sparse_attn.cache = {s: (k.clone(), v.clone()) for s, (k, v) in cache.items()}


def test_sparse_decode_fp32_matches_reference(tiny_checkpoint):
    """fp32: every teacher-forced pick equals HF's token; the step at 2615 matches a fresh prefill within 1e-5."""
    prompt = load_long_prompt(tiny_checkpoint)
    gen = list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        picks, last, full = run_sparse_decode_chain(model, vllm_config, atts, prompt["prompt_ids"], gen, DECODE_SLOT,
                                                    eos)
        for i, (picked, expected) in enumerate(zip(picks, gen)):
            assert picked == expected, (f"sparse decode step {i}: teacher-forced argmax {picked} != "
                                        f"HF generated_ids[{i}] {expected}")
        fresh = sparse_prefill_last_logits(model, vllm_config, eos, full, slot=DECODE_SLOT)
        gap = (fresh - last).abs().max().item()
        assert gap <= FP32_TOL, (f"last decode step vs fresh sparse prefill of the {FULL_LEN}-token sequence: "
                                 f"max |delta logit| = {gap:.3e} > {FP32_TOL:.0e}")


def test_decode_writes_prefill_rows(tiny_checkpoint):
    """The side-cache rows the 16 decode steps write match a fresh sparse prefill of the same tokens.

    ``compressed_key_cache[600:654]`` (the rows the chain touches: 600..649 from the prefill,
    650..653 written at completions 2603/2607/2611/2615) equals a fresh 2616-token prefill's rows
    within 1e-5 (fp32 reassociation: the fused GDN decode path measured 1.07e-6).
    ``key_ring[DECODE_SLOT]`` holds the raw keys of positions 2612..2615 in rows
    ``pos % 4``; the fresh reference is a 2615-token prefill (tail 3 writes 2612..2614) plus the
    decode step at 2615, because a 2616-token prefill has no tail and writes NO ring rows.
    """
    prompt = load_long_prompt(tiny_checkpoint)
    gen = list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        _, _, full = run_sparse_decode_chain(model, vllm_config, atts, prompt["prompt_ids"], gen, DECODE_SLOT, eos)
        indexer = qsa_layers(model)[0].indexer
        chain_rows = indexer.compressed_key_cache[600:654].clone()
        chain_ring = indexer.key_ring[DECODE_SLOT].clone()
        assert bool(chain_ring.any()), "the decode chain left key_ring slot DECODE_SLOT all zeros"
        sparse_prefill_last_logits(model, vllm_config, eos, full, slot=DECODE_SLOT)
        fresh_rows = indexer.compressed_key_cache[600:654].clone()
        gap = (chain_rows - fresh_rows).abs().max().item()
        assert gap <= ROW_TOL, (f"decode rows compressed_key_cache[600:654] vs fresh sparse prefill of the "
                                f"{FULL_LEN}-token sequence: max |delta| = {gap:.3e} > {ROW_TOL:.0e}")
        sparse_prefill_last_logits(model, vllm_config, eos, full[:FULL_LEN - 1], slot=DECODE_SLOT)
        sparse_decode_step(model, vllm_config, atts, full, FULL_LEN - 1, eos, att_slot=DECODE_SLOT,
                           load_slot=DECODE_SLOT)
        fresh_ring = indexer.key_ring[DECODE_SLOT].clone()
        ring_gap = (chain_ring - fresh_ring).abs().max().item()
        assert ring_gap <= ROW_TOL, (f"decode key_ring rows (raw keys of positions {FULL_LEN - 4}..{FULL_LEN - 1}) "
                                     f"vs 2615-token prefill + decode at 2615: max |delta| = {ring_gap:.3e} > "
                                     f"{ROW_TOL:.0e}")


def test_slot_move_at_aligned_position(tiny_checkpoint):
    """fp32: the decode step at 4-aligned position 2604 moves the state from slot 2 to slot 3.

    The step runs with ``load_slot=2 / store_slot=3``; every pick still equals HF's token and
    the last step equals a fresh sparse prefill of the 2616-token sequence within 1e-5. Slot 2
    is byte-identical after the move and the steps that follow it, and slot 3 carries the state.
    """
    prompt = load_long_prompt(tiny_checkpoint)
    gen = list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        enable_slot_reuse(vllm_config)
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        modules = state_modules(model)
        lg = sparse_prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"], slot=STATE_SLOT_A)
        picks = [int(lg.argmax())]
        seq = list(prompt["prompt_ids"])
        last = lg

        def step(i: int, load: int, store: int) -> None:
            nonlocal last
            seq.append(gen[i])
            last = sparse_decode_step(model, vllm_config, atts, seq, len(seq) - 1, eos, att_slot=STATE_SLOT_A,
                                      load_slot=load, store_slot=store)
            if i + 1 < len(gen):
                picks.append(int(last.argmax()))

        for i in range(MOVE_POS - LONG_PROMPT):  # positions 2600..2603, state stays in slot 2
            step(i, STATE_SLOT_A, STATE_SLOT_A)
        saved = [t.clone() for t in slot_tensors(modules, STATE_SLOT_A)]
        step(MOVE_POS - LONG_PROMPT, STATE_SLOT_A, STATE_SLOT_B)  # the move at position 2604
        assert_slot_equal(modules, STATE_SLOT_A, saved, "after the slot-move step (it read this slot)")
        assert_slot_written(modules, STATE_SLOT_B, "after the slot-move step (it wrote this slot)")
        for i in range(MOVE_POS - LONG_PROMPT + 1, len(gen)):  # positions 2605..2615, state in slot 3
            step(i, STATE_SLOT_B, STATE_SLOT_B)
        assert_slot_equal(modules, STATE_SLOT_A, saved, "after the steps that followed the move (slot 3 only)")
        assert_slot_untouched(modules, 0, "after the slot-move chain (slot 0 is the null block)")
        for i, (picked, expected) in enumerate(zip(picks, gen)):
            assert picked == expected, (f"slot-move chain, pick {i}: teacher-forced argmax {picked} != "
                                        f"HF generated_ids[{i}] {expected}")
        fresh = sparse_prefill_last_logits(model, vllm_config, eos, seq, slot=DECODE_SLOT)
        gap = (fresh - last).abs().max().item()
        assert gap <= FP32_TOL, (f"slot-move chain last step vs fresh sparse prefill of the {FULL_LEN}-token "
                                 f"sequence: max |delta logit| = {gap:.3e} > {FP32_TOL:.0e}")


def test_zeroed_compressed_rows_change_selection(tiny_checkpoint, monkeypatch):
    """Negative control: zeroing compressed rows 600..649 before the step at 2610 changes its selection.

    The chain prefills 2600 tokens and decodes to position 2609; the step at 2610 runs twice on
    the identical snapshot state: once with the caches intact, once after ``compressed_key_cache
    [600:650] = 0``. The two ``selected`` rows must differ, so a decode parity that scored the
    side caches as all-zero would be detected.
    """
    prompt = load_long_prompt(tiny_checkpoint)
    gen = list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        layer = qsa_layers(model)[0]
        captured = []
        orig_select = layer.indexer.select

        def record_select(*args, **kwargs):
            out = orig_select(*args, **kwargs)
            captured.append(out.detach().clone())
            return out

        monkeypatch.setattr(layer.indexer, "select", record_select)
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        sparse_prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"], slot=DECODE_SLOT)
        seq = list(prompt["prompt_ids"])
        for i in range(NEG_POS - LONG_PROMPT):  # decode steps at positions 2600..2609, teacher-forced
            seq.append(gen[i])
            sparse_decode_step(model, vllm_config, atts, seq, len(seq) - 1, eos, att_slot=DECODE_SLOT,
                               load_slot=DECODE_SLOT)
        snap = snapshot_state(model)
        captured.clear()  # the prefill and the steps up to 2609 also ran the wrapper
        seq.append(gen[NEG_POS - LONG_PROMPT])
        pos = len(seq) - 1  # NEG_POS
        assert pos == NEG_POS
        sparse_decode_step(model, vllm_config, atts, seq, pos, eos, att_slot=DECODE_SLOT, load_slot=DECODE_SLOT)
        intact = captured.pop()
        assert not captured
        restore_state(model, snap)
        layer.indexer.compressed_key_cache[ZERO_ROWS[0]:ZERO_ROWS[1]].zero_()
        sparse_decode_step(model, vllm_config, atts, seq, pos, eos, att_slot=DECODE_SLOT, load_slot=DECODE_SLOT)
        zeroed = captured.pop()
        assert not captured
        assert intact.shape == zeroed.shape
        changed = int((intact != zeroed).sum())
        print(f"zeroed rows {ZERO_ROWS[0]}..{ZERO_ROWS[1] - 1}: {changed} of "
              f"{intact.numel()} selection entries changed at position {NEG_POS}")
        assert changed > 0, (f"zeroing compressed_key_cache rows {ZERO_ROWS[0]}..{ZERO_ROWS[1] - 1} left the "
                             f"selection at position {NEG_POS} unchanged; the decode parity checks are blind to "
                             "those side-cache rows")


def test_bf16_decode_is_close(tiny_checkpoint):
    """bf16: at least 14 of 16 picks equal HF's tokens (bf16 ties flip picks, M0 allows 2); last step within 3e-2."""
    prompt = load_long_prompt(tiny_checkpoint)
    gen = list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (vllm_config,
                                                                                                      model):
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        picks, last, full = run_sparse_decode_chain(model, vllm_config, atts, prompt["prompt_ids"], gen, DECODE_SLOT,
                                                    eos)
        matches = sum(1 for picked, expected in zip(picks, gen) if picked == expected)
        assert matches >= 14, f"bf16: only {matches} of {len(gen)} teacher-forced picks equal generated_ids (>= 14)"
        fresh = sparse_prefill_last_logits(model, vllm_config, eos, full, slot=DECODE_SLOT)
        gap = (fresh - last).abs().max().item()
        assert gap <= BF16_TOL, (f"bf16 last decode step vs fresh sparse prefill of the {FULL_LEN}-token sequence: "
                                 f"max |delta logit| = {gap:.3e} > {BF16_TOL:.0e}")
