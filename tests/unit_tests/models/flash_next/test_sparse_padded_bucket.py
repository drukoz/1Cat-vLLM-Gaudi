# SPDX-License-Identifier: Apache-2.0
"""Sparse QSA through the REAL HPU wrapper, in the HPU runner's padded buckets, on CPU.

Every other sparse test runs the layer with ``SparseSdpaAttention`` standing in for the HPU wrapper and with
unpadded rows. On a card the forward runs ``HpuQsaSparseAttention`` and every prompt sits in a padded bucket
(a multiple of 128 tokens), so the following never ran before a card and all of it is reachable as soon as
sparse mode is really on:

* the wrapper's first statement read ``get_forward_context().virtual_engine``, which the pinned vLLM does not
  have, and indexed ``attn.kv_cache`` with it, although the runner binds the cache tuple itself;
* the padding rows of a bucket carry position ``-1``: their K/V slot is ``-1``, so the compressed-key row of
  a block that is not complete inside the prompt was ``-1 // 4`` (an out-of-range ``index_copy_``); they
  select no position, so their bias row was all ``-inf`` and the softmax returned NaN, which the GDN layers
  after the QSA layer carry into the saved state (``x * 0`` masking keeps a NaN);
* the tail of a prompt (the last ``len % 4`` raw keys) was written to the key ring only when the ROW length
  was not a multiple of 4, which a bucket never is, so the block that straddles the end of the prompt was
  later compressed from stale ring rows.

``hpu_bucket.py`` supplies the wrapper's HPU-only surroundings and the runner's bucket layout. The oracle is
the same model on the unpadded row; ``test_wrapper_matches_the_hf_reference`` ties the wrapper itself to the HF
reference first. Both position layouts run: ``production`` (``--language-model-only``, no M-RoPE) and ``mrope``.

**Ties.** A block score is ``sum_h relu(q_h . k) / sqrt(D)``, which is exactly 0 for a block no head likes. When
fewer than 512 visible blocks score above 0 (rows 2053.. of the 2601-token prompt: 513 visible, 477 positive),
the top-512 cut falls among equal scores and ``torch.topk`` breaks the tie by the row's layout, which a bucket
changes (736 block columns instead of 650). Both selections are top-512 sets, so the padded row is compared
with the unpadded one as: the same 512 best SCORES on every row, and the same logits on every row whose
selection is the same set (most rows beyond 2050 included).
"""

import contextlib
import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

import vllm_gaudi.models.qwen4_exp.qsa as qsa_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, fp32_logits, reset_cpu_states)
from tests.unit_tests.models.flash_next.hpu_bucket import (  # noqa: E402
    bucket_decode, bucket_prefill, clamp_rotary_padding, install_hpu_sparse_wrapper, prompt_bucket, qsa_layers,
    use_mrope_rotary)
from tests.unit_tests.models.flash_next.test_chunked_prefill_parity import slot_tensors, state_modules  # noqa: E402

MAX_MODEL_LEN = 4096
BLOCK = 128
PAD_BLOCK = MAX_MODEL_LEN // BLOCK - 1  # the last kernel block: no test sequence reaches position 3968
NUM_SLOTS = 6
SLOT = 1
PAD_STATE = NUM_SLOTS - 1  # the last state row
LONG = 2600  # the sparse-sensitive reference prompt
RATIO = 4
LOGIT_TOL = 1e-5  # the HF reference tolerance of test_sparse_prefill_parity
# Padded vs unpadded rows of the same model: the GDN prefill runs over more chunks, fp32 reassociation only.
ROW_TOL = 2e-5
LAYOUTS = ["production", "mrope"]


def load_prompt(tiny_checkpoint, length: int) -> dict:
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    return next(p for p in reference["prompts"] if len(p["prompt_ids"]) == length)


@contextlib.contextmanager
def wrapper_model(tiny_checkpoint, monkeypatch, layout: str):
    """The tiny sparse model with the real HPU wrapper on its QSA layer: ``(vllm_config, model, attns, eos)``."""
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   num_slots=NUM_SLOTS,
                   max_model_len=MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        attns = install_hpu_sparse_wrapper(model, monkeypatch, cache_slots=MAX_MODEL_LEN)
        if layout == "mrope":
            use_mrope_rotary(model)
        else:
            clamp_rotary_padding(model, monkeypatch)
        yield vllm_config, model, attns, eos_token_id(vllm_config)


def reset(model, attns) -> None:
    reset_cpu_states(model)
    for attn in attns:
        attn.reset()


def forward(model, vllm_config, batch) -> torch.Tensor:
    with torch.no_grad(), set_forward_context(batch.metadata, vllm_config):
        return model(input_ids=batch.input_ids, positions=batch.positions, **batch.ple)


def prefill(model, vllm_config, attns, eos, tokens, layout: str, bucket: int | None, **kwargs) -> torch.Tensor:
    """Hidden states ``[bucket, hidden]`` of one prefill row in slot ``SLOT`` (``bucket=None``: unpadded)."""
    return forward(model, vllm_config,
                   bucket_prefill(tokens, slot=SLOT, eos=eos, bucket=bucket, mrope=layout == "mrope", **kwargs))


def decode(model, vllm_config, eos, seq, pos: int, layout: str, batch: int) -> torch.Tensor:
    """Hidden states ``[batch, hidden]`` of the decode step of ``seq[pos]`` padded to ``batch`` rows."""
    step = bucket_decode([seq[pos]], [pos], [SLOT], [(seq[pos - 2], seq[pos - 1])],
                         batch=batch,
                         eos=eos,
                         pad_state_slot=PAD_STATE,
                         pad_block=PAD_BLOCK,
                         mrope=layout == "mrope",
                         max_blocks=MAX_MODEL_LEN // BLOCK)
    return forward(model, vllm_config, step)


def snapshot(model) -> dict:
    """Clones of what a forward leaves behind for slot ``SLOT``: recurrent states and the indexer's side caches."""
    indexer = qsa_layers(model)[0].indexer
    return {
        "state": [t.clone() for t in slot_tensors(state_modules(model), SLOT)],
        "compressed": indexer.compressed_key_cache.clone(),
        "ring": indexer.key_ring[SLOT].clone(),
    }


class SelectionRecorder:
    """Records the block scores and the picked blocks of every ``select_blocks`` call of one prefill."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        original = qsa_module.select_blocks

        def select_blocks(scores, visible, block_topk):
            picked = original(scores, visible, block_topk)
            self.calls.append((scores.detach().clone(), picked.detach().clone()))
            return picked

        monkeypatch.setattr(qsa_module, "select_blocks", select_blocks)

    def take(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(best scores [rows, K] sorted, picked blocks [rows, K] sorted)`` of the first ``rows`` query rows."""
        scores = [c[0] for c in self.calls]
        picked = torch.cat([c[1] for c in self.calls])[:rows].long()
        width = min(s.shape[1] for s in scores)
        assert int(picked.max()) < width
        scores = torch.cat([s[:, :width] for s in scores])[:rows]
        best = torch.where(picked >= 0, scores.gather(1, picked.clamp_min(0)), torch.full_like(scores[:, :1], -1.0))
        self.calls.clear()
        return best.sort(dim=1, descending=True).values, picked.sort(dim=1).values


def assert_same_selection_up_to_ties(padded, oracle, label: str, first_position: int = 0) -> torch.Tensor:
    """Both selections hold the same best scores on every row; returns the rows whose block sets are equal.

    ``first_position`` is the position of row 0 (a chunk's context length).
    """
    (best, blocks), (want_best, want_blocks) = padded, oracle
    gap = (best - want_best).abs().max().item()
    assert gap <= ROW_TOL, f"{label}: the selected blocks' scores differ from the unpadded row's by {gap:.3e}"
    same = (blocks == want_blocks).all(dim=1)
    beyond = first_position + torch.arange(same.numel()) >= 2051
    assert int((same & beyond).sum()) >= 256, f"{label}: only {int((same & beyond).sum())} rows beyond 2050 compare"
    return same


def assert_all_finite(tensors, label: str) -> None:
    for idx, tensor in enumerate(tensors):
        bad = int((~torch.isfinite(tensor.float())).sum())
        assert bad == 0, f"{label}: tensor {idx} holds {bad} non-finite values"


def assert_states_close(padded: dict, oracle: dict, label: str) -> None:
    assert_all_finite(padded["state"], f"{label}: saved state")
    for idx, (got, want) in enumerate(zip(padded["state"], oracle["state"])):
        gap = (got.float() - want.float()).abs().max().item()
        assert gap <= ROW_TOL, f"{label}: saved state tensor {idx} of slot {SLOT} differs by {gap:.3e} > {ROW_TOL:.0e}"


def test_wrapper_matches_the_hf_reference(tiny_checkpoint, monkeypatch):
    """The real wrapper, unpadded: the sparse prefill of the 2600-token prompt reproduces HF's top-5 logits and
    16 teacher-forced decode steps reproduce HF's greedy tokens (the checks ``SparseSdpaAttention`` passes)."""
    prompt = load_prompt(tiny_checkpoint, LONG)
    seq = list(prompt["prompt_ids"]) + list(prompt["generated_ids"])
    with wrapper_model(tiny_checkpoint, monkeypatch, "production") as (vllm_config, model, attns, eos):
        reset(model, attns)
        hidden = prefill(model, vllm_config, attns, eos, seq[:LONG], "production", None)
        logits = fp32_logits(model, hidden)[-1]
        ref = prompt["last_prefill_top5"]
        assert int(logits.argmax()) == ref["ids"][0]
        delta = (logits[ref["ids"]] - torch.tensor(ref["logits"], dtype=torch.float32)).abs().max().item()
        assert delta <= LOGIT_TOL, f"wrapper prefill vs HF top-5 logits: {delta:.3e} > {LOGIT_TOL:.0e}"
        assert attns[0].impl.updates == 1
        picks = [int(logits.argmax())]
        for pos in range(LONG, len(seq) - 1):
            step = decode(model, vllm_config, eos, seq, pos, "production", batch=1)
            picks.append(int(fp32_logits(model, step)[0].argmax()))
        assert picks == seq[LONG:], f"teacher-forced picks {picks} != HF generated_ids {seq[LONG:]}"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_padded_prefill_matches_the_unpadded_row(tiny_checkpoint, monkeypatch, layout):
    """2601 real tokens in the 2944-token bucket equal the unpadded 2601-token row.

    Real rows: the same best block scores, and the same logits wherever the selection is the same set (see
    "Ties" above). Every row, padding included: finite (a NaN on a padding row reaches the saved state of the
    GDN layers that follow a QSA layer). Saved GDN/PLE state: equal. Compressed keys: the 650
    complete blocks equal, and nothing else written but the scratch row (block 650 holds one real token: its
    key must not land in a request's row). Key ring: the raw key of the tail token 2600 is there.
    """
    prompt = load_prompt(tiny_checkpoint, LONG)
    tokens = (list(prompt["prompt_ids"]) + list(prompt["generated_ids"]))[:LONG + 1]
    length = len(tokens)
    bucket = prompt_bucket(length)
    assert (length, bucket, length % RATIO) == (2601, 2944, 1)
    full_blocks = length // RATIO
    with wrapper_model(tiny_checkpoint, monkeypatch, layout) as (vllm_config, model, attns, eos):
        recorder = SelectionRecorder(monkeypatch)
        reset(model, attns)
        oracle_hidden = prefill(model, vllm_config, attns, eos, tokens, layout, None)
        oracle = snapshot(model)
        oracle_selection = recorder.take(length)
        assert bool(oracle["ring"][length % RATIO - 1].any()), "the oracle wrote no ring row for the tail token"

        reset(model, attns)
        hidden = prefill(model, vllm_config, attns, eos, tokens, layout, bucket)
        padded = snapshot(model)

        assert hidden.shape[0] == bucket
        assert_all_finite([hidden], f"{layout}: hidden states of the {bucket}-token bucket")
        same = assert_same_selection_up_to_ties(recorder.take(length), oracle_selection, layout)
        gap = (fp32_logits(model, hidden[:length]) - fp32_logits(model, oracle_hidden))[same].abs().max().item()
        assert gap <= ROW_TOL, f"{layout}: logits of the real rows differ by {gap:.3e} > {ROW_TOL:.0e}"
        assert_states_close(padded, oracle, f"{layout} padded prefill")

        comp_gap = (padded["compressed"][:full_blocks] - oracle["compressed"][:full_blocks]).abs().max().item()
        assert comp_gap <= ROW_TOL, f"{layout}: compressed keys of the complete blocks differ by {comp_gap:.3e}"
        assert not bool(padded["compressed"][full_blocks:-1].any()), (
            f"{layout}: a block that is not complete inside the prompt wrote a compressed row of a request")
        ring_gap = (padded["ring"] - oracle["ring"]).abs().max().item()
        assert ring_gap <= ROW_TOL, (f"{layout}: key ring of slot {SLOT} differs from the unpadded row by "
                                     f"{ring_gap:.3e}: the prompt's tail keys were not written")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_padded_prefill_then_decode_completes_the_straddling_block(tiny_checkpoint, monkeypatch, layout):
    """702 real tokens in the 768-token bucket, then the decode steps at 702 and 703 in a batch of 4 rows.

    Block 175 (tokens 700..703) straddles the end of the prompt: the step at 703 compresses it from the ring
    rows of 700 and 701 (the prompt's tail), the row of 702 and its own key. Everything is compared with the
    unpadded chain (702-token row, batch of 1): logits of every step, the compressed row of block 175, the
    ring and the saved state; and every row of every padded batch is finite.
    """
    prompt = load_prompt(tiny_checkpoint, 700)
    seq = list(prompt["prompt_ids"]) + list(prompt["generated_ids"])
    length = 702
    bucket = prompt_bucket(length)
    block = length // RATIO  # 175
    assert (bucket, length % RATIO) == (768, 2)
    with wrapper_model(tiny_checkpoint, monkeypatch, layout) as (vllm_config, model, attns, eos):

        def chain(prefill_bucket, batch):
            reset(model, attns)
            rows = [prefill(model, vllm_config, attns, eos, seq[:length], layout, prefill_bucket)]
            for pos in (length, length + 1):
                rows.append(decode(model, vllm_config, eos, seq, pos, layout, batch))
            return rows, snapshot(model)

        oracle_rows, oracle = chain(None, 1)
        assert bool(oracle["compressed"][block].any()), "the oracle chain did not complete block 175"
        rows, padded = chain(bucket, 4)

        for name, got in zip(("prefill", "decode 702", "decode 703"), rows):
            assert_all_finite([got], f"{layout} {name}: hidden states of all rows")
        for name, got, want in zip(("decode 702", "decode 703"), rows[1:], oracle_rows[1:]):
            gap = (fp32_logits(model, got[:1]) - fp32_logits(model, want)).abs().max().item()
            assert gap <= ROW_TOL, f"{layout} {name}: logits of the real row differ by {gap:.3e} > {ROW_TOL:.0e}"
        key_gap = (padded["compressed"][block] - oracle["compressed"][block]).abs().max().item()
        assert key_gap <= ROW_TOL, (
            f"{layout}: compressed key of block {block} (tokens 700..703) differs from the unpadded chain by "
            f"{key_gap:.3e}: it was compressed from stale ring rows")
        ring_gap = (padded["ring"] - oracle["ring"]).abs().max().item()
        assert ring_gap <= ROW_TOL, f"{layout}: key ring differs from the unpadded chain by {ring_gap:.3e}"
        assert_states_close(padded, oracle, f"{layout} padded chain")
        assert not bool(padded["compressed"][block + 1:-1].any()), (
            f"{layout}: a padding row or an incomplete block wrote a compressed row of a request")


def test_chunked_padded_prefill_with_context(tiny_checkpoint, monkeypatch):
    """A 2601-token prompt as two chunks, 2048 tokens and then 553 tokens in the 768-token bucket with 16
    context blocks padded to a 31-block list: the last real row equals the one-chunk unpadded prefill.

    The second chunk runs the wrapper's context path (``block_list`` gathered in front of the chunk's keys, the
    bias columns of context positions) and the indexer's context keys, with padding rows and padded blocks.
    """
    prompt = load_prompt(tiny_checkpoint, LONG)
    tokens = (list(prompt["prompt_ids"]) + list(prompt["generated_ids"]))[:LONG + 1]
    first = 2048
    rest = tokens[first:]
    with wrapper_model(tiny_checkpoint, monkeypatch, "production") as (vllm_config, model, attns, eos):
        recorder = SelectionRecorder(monkeypatch)
        reset(model, attns)
        oracle_hidden = prefill(model, vllm_config, attns, eos, tokens, "production", None)
        oracle = snapshot(model)
        best, blocks = recorder.take(len(tokens))
        oracle_selection = (best[first:], blocks[first:])

        reset(model, attns)
        prefill(model, vllm_config, attns, eos, tokens[:first], "production", first)
        recorder.calls.clear()
        chunk = bucket_prefill(rest,
                               slot=SLOT,
                               eos=eos,
                               bucket=prompt_bucket(len(rest)),
                               mrope=False,
                               context_len=first,
                               ngram_context=tokens[first - 2:first])
        blocks = chunk.metadata.block_list
        chunk.metadata.block_list = torch.cat((blocks, torch.full((31 - blocks.numel(), ), -1, dtype=torch.int32)))
        hidden = forward(model, vllm_config, chunk)
        padded = snapshot(model)

        assert_all_finite([hidden], "second chunk: hidden states of the bucket")
        same = assert_same_selection_up_to_ties(recorder.take(len(rest)), oracle_selection, "second chunk", first)
        gap = (fp32_logits(model, hidden[:len(rest)]) - fp32_logits(model, oracle_hidden[first:]))[same].abs().max()
        assert gap.item() <= ROW_TOL, f"chunked padded prefill: logits of the second chunk differ by {gap:.3e}"
        assert_states_close(padded, oracle, "chunked padded prefill")
        full_blocks = len(tokens) // RATIO
        comp_gap = (padded["compressed"][:full_blocks] - oracle["compressed"][:full_blocks]).abs().max().item()
        assert comp_gap <= ROW_TOL, f"chunked padded prefill: compressed keys differ by {comp_gap:.3e}"
        ring_gap = (padded["ring"] - oracle["ring"]).abs().max().item()
        assert ring_gap <= ROW_TOL, f"chunked padded prefill: key ring differs by {ring_gap:.3e}"


def test_a_nan_padding_row_poisons_the_gdn_state(tiny_checkpoint, monkeypatch):
    """Why every row of a bucket must be finite, padding included (the tiny model cannot show it end to end:
    its QSA layer is the last layer, in the real model GDN layers follow every QSA layer).

    A NaN is put by hand on the padding rows entering the second GDN layer of a 300-token prompt in the
    384-token bucket. The GDN prefill masks padding by multiplication (``x * 0`` keeps a NaN) and runs its
    recurrence over the whole bucket, so the state it saves for the request is not finite any more.
    """
    tokens = list(load_prompt(tiny_checkpoint, 700)["prompt_ids"])[:300]
    with wrapper_model(tiny_checkpoint, monkeypatch, "production") as (vllm_config, model, attns, eos):
        gdn = dict(model.named_modules())["language_model.model.layers.1.linear_attn"]

        def poison(module, args, kwargs):
            hidden = kwargs["hidden_states"].clone()
            hidden.view(-1, hidden.shape[-1])[len(tokens):] = float("nan")
            return args, {**kwargs, "hidden_states": hidden}

        reset(model, attns)
        prefill(model, vllm_config, attns, eos, tokens, "production", 384)
        assert_all_finite([t[SLOT] for t in gdn.kv_cache], "clean padded prefill: GDN state")

        handle = gdn.register_forward_pre_hook(poison, with_kwargs=True)
        try:
            reset(model, attns)
            prefill(model, vllm_config, attns, eos, tokens, "production", 384)
        finally:
            handle.remove()
        assert not all(bool(torch.isfinite(t[SLOT].float()).all()) for t in gdn.kv_cache), (
            "a NaN on the padding rows no longer reaches the saved GDN state: the all-rows-finite checks above "
            "can be relaxed to the real rows")
