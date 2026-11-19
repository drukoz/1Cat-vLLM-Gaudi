# SPDX-License-Identifier: Apache-2.0
"""M3-T7 sparse-QSA multi-token decode rows on CPU: write-then-score, the accept lattice, ragged rows.

The M3 verification step feeds the model ``T`` tokens per request in one decode forward, so the
QSA decode branch scores ``bs * T`` query rows: it writes the step's raw keys into the ring and
the completed blocks' compressed keys into ``compressed_key_cache`` BEFORE it scores anything
(write-then-score), and both caches are position-addressed, so a token re-processed at a position
overwrites what a rejected draft left there and a rollback needs no extra state.

``indexer_level_chain`` drives ``HpuQsaIndexer.select`` directly with the same ``T`` layer-input
rows in both orders: one ``T``-token step with the runner's padded layout (the
``[groups, bs, T]`` state-index lattice, ``query_start_loc_p`` and a per-row ``block_table`` from
``decode_metadata(num_tokens=T)``) versus ``T`` single-token steps. The selection rows, the ring
rows and the compressed row of block 16 (the block the step completes, positions 64..67) must be
the same, which is the state a full ``T``-token forward has to leave behind. The ragged test runs
two requests of 4 and 2 real tokens as one padded 8-row step and compares every ring row and the
compressed rows with the two single-request chains, so the padding rows must be parked on the
ring's scratch row and write no compressed row. ``test_write_before_score_mutation_fails_the_chain_parity``
monkeypatches ``_select_decode`` with a copy that scores before writing and expects the chain
parity of the first test to fail exactly on the row that completes a block inside the step.

The end-to-end version (the model forward of a 4-token decode step, its logits against the chain)
is here as ``test_whole_model_multi_token_step`` and is xfail: a whole-model lattice forward needs
the PLE speculative path (``ple_layer.py`` raises ``NotImplementedError("PLE speculative decode
is M3")`` on ``num_accepted_tokens``, M3-T6) and the GDN spec-state fixture (GDN asserts a
``cache_group_idx`` for a multi-dimensional state index and its conv state needs
``(kernel - 1) + T`` rows, M3-T5a). The M3 end-to-end parity is the M3-T8 MTP test.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer, HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import (  # noqa: E402
    compress_block_keys, compressed_row_slots, expand_selection, ring_block_rows, ring_write, select_blocks,
    visible_blocks)

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata)
from tests.unit_tests.models.flash_next.test_sparse_decode_parity import (  # noqa: E402
    restore_state, snapshot_state)
from tests.unit_tests.models.flash_next.test_sparse_prefill_parity import (  # noqa: E402
    SPARSE_MAX_MODEL_LEN, model_sparse_attns)

PREFILL_LEN = 64  # 4-aligned: the prefill writes compressed rows 0..15 and NO ring rows
POS0 = PREFILL_LEN  # the decode rows start here: positions 64..67 are block 16
T = 4  # tokens per request in the verification step
RATIO = 4
SLOT_A, SLOT_B = 1, 2  # the two requests of the ragged step
SCRATCH_SLOT = 3  # the ring's last row (num_slots=4): never a request's slot
NULL_SLOT = 0  # vLLM's null block: no test row may touch it
SELF_BLOCK = POS0 // RATIO  # block 16, the block the step at positions 64..67 completes
SELF_ROWS = [SELF_BLOCK * RATIO + i for i in range(RATIO)]  # its positions: 64..67
COMP_SELF = SELF_BLOCK  # compressed row of block 16 (slot_mapping == position in this harness)
# The last compressed row: the scratch row every non-completing decode row writes (also where the
# -1 pad slot_mapping lands under floor division).
COMP_TAIL = 1023
ROW_TOL = 1e-6  # side-cache rows written on the multi-token path vs the single-token chain
HIDDEN_SEED = 20261117


def qsa_layer(model):
    """The tiny checkpoint's single sparse-QSA layer (``layers[3].self_attn``)."""
    return next(m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention))


def cos_sin_at(layer):
    """The layer's own RoPE lookup, the lambda ``HpuQwen4ExpQSAAttention.forward`` passes to ``select``."""
    return lambda p: layer.rotary_emb.cos_sin_cache[p.clamp_min(0)].chunk(2, dim=-1)


def hidden_rows(layer, n: int) -> torch.Tensor:
    """Seeded ``(n, hidden_size)`` layer-input rows: the ``n`` draft tokens' hidden states."""
    gen = torch.Generator().manual_seed(HIDDEN_SEED)
    weight = next(layer.indexer.parameters())
    return torch.randn(n, layer.hidden_size, generator=gen, dtype=weight.dtype, device=weight.device)


def prefill_layer_64(model, vllm_config, eos: int) -> None:
    """Prefill ``PREFILL_LEN`` tokens into state slot ``SLOT_A`` (the state every chain starts from)."""
    ids = list(range(1000, 1000 + PREFILL_LEN))
    for att in model_sparse_attns(model):
        att.slot = SLOT_A
    with torch.no_grad(), set_forward_context(prefill_metadata(PREFILL_LEN, SLOT_A), vllm_config):
        model(input_ids=torch.tensor(ids), positions=torch.arange(PREFILL_LEN), **ple_kwargs(PREFILL_LEN, eos))


def side_caches(layer) -> tuple[torch.Tensor, torch.Tensor]:
    """``(compressed_key_cache, key_ring)`` of one layer's indexer."""
    return layer.indexer.compressed_key_cache, layer.indexer.key_ring


def select_multi(layer, hidden, md) -> torch.Tensor:
    """One multi-token decode selection over the metadata's ``T`` rows (the multi-token step)."""
    return layer.indexer.select(hidden, md.positions, cos_sin_at(layer), md)


def select_chain(layer, hidden, slot: int, n: int) -> torch.Tensor:
    """``n`` single-token decode selections over ``hidden[0:n]`` at positions ``POS0 .. POS0 + n``."""
    csa = cos_sin_at(layer)
    out = [
        layer.indexer.select(hidden[j:j + 1], torch.tensor([POS0 + j]), csa,
                             decode_metadata(slot, POS0 + j, load_slot=slot, store_slot=slot)) for j in range(n)
    ]
    return torch.cat(out)


def lattice_md(slot: int, *, query_lens=None, slots=None, num_tokens: int = T):
    """The runner's padded multi-token decode metadata for ``slot`` (the lattice state indices)."""
    return decode_metadata(slot,
                           POS0,
                           num_tokens=num_tokens,
                           query_lens=query_lens,
                           slots=slots,
                           max_blocks=SPARSE_MAX_MODEL_LEN // 128)


def test_multi_token_step_equals_single_token_chain(tiny_checkpoint):
    """One 4-token decode step == four 1-token steps: same selection, same ring, same compressed row.

    Both chains run from the same 64-token prefill on the same 4 hidden rows, the step with the
    ``[1, 1, 4]`` lattice and ``num_accepted_tokens=[1]``, the chain with the plain ``(1,)`` state
    indices. The comparison is exact for the selection (integers) and within ``1e-6`` for the
    side-cache rows, with the compressed-cache comparison EXCLUDING the scratch row (the cache's
    last row, where the step's non-completing rows collide through one duplicated ``index_put_``
    index), and the step's last row must select the block its own step wrote (block 16):
    that is what write-then-score buys.
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)  # the runner sets this
        hidden = hidden_rows(layer, T)
        prefill_layer_64(model, vllm_config, eos)

        snap = snapshot_state(model)
        sel_step = select_multi(layer, hidden, lattice_md(SLOT_A))
        comp, ring = side_caches(layer)
        step_comp, step_ring = comp.clone(), ring.clone()

        restore_state(model, snap)
        sel_chain = select_chain(layer, hidden, SLOT_A, T)
        comp, ring = side_caches(layer)
        chain_comp, chain_ring = comp.clone(), ring.clone()

        assert sel_step.shape == sel_chain.shape == (T, layer.indexer.output_width)
        assert torch.equal(sel_step, sel_chain), (
            f"the 4-token step and the 1-token chain selected differently: "
            f"{int((sel_step != sel_chain).sum())} of {sel_step.numel()} entries differ; "
            f"first rows {sel_step[0, :8].tolist()} vs {sel_chain[0, :8].tolist()}")
        ring_gap = (step_ring - chain_ring).abs().max().item()
        assert ring_gap <= ROW_TOL, (f"key_ring after the 4-token step vs the 1-token chain: "
                                     f"max |delta| = {ring_gap:.3e} > {ROW_TOL:.0e}")
        # Parity is on the request rows only: the cache's last row is the scratch row every
        # non-completing decode row writes (qsa_side_caches). In the step rows 64/65/66 hit it
        # through one duplicated ``index_put_`` index (which key survives is undefined by design);
        # the chain leaves position 66's key there, and that key compresses an incomplete block,
        # so its content is don't-care. Where those rows land is pinned by
        # ``test_decode_write_has_fixed_shape``.
        comp_gap = (step_comp[:-1] - chain_comp[:-1]).abs().max().item()
        assert comp_gap <= ROW_TOL, (f"compressed_key_cache after the 4-token step vs the 1-token chain "
                                      f"(scratch row excluded): max |delta| = {comp_gap:.3e} > {ROW_TOL:.0e}")
        assert bool(step_ring[SLOT_A].any()), "the multi-token step wrote no raw key into the request's ring slot"
        assert bool(step_comp[COMP_SELF].any()), "the step left the compressed row of block 16 at zero"
        picked = {int(p) for p in sel_step[-1] if int(p) >= 0}
        missing = [p for p in SELF_ROWS if p not in picked]
        assert not missing, (f"the last row of the step (position {POS0 + T - 1}) does not select the block its "
                             f"own step wrote (block {SELF_BLOCK}, positions {SELF_ROWS}): {missing} missing; "
                             "the caches must be written before the rows are scored")


def test_scoring_sees_the_blocks_of_its_own_step(tiny_checkpoint, monkeypatch):
    """Write-then-score, made sensitive: the step repairs a stale self row and its scores drive the top-k.

    At 64 tokens every visible block fits in the 512-block budget, so the selection is pure
    visibility and ignores the scores. Shrinking ``block_topk`` to 16 makes the 17 blocks visible
    at positions 64..67 one more than the budget, so exactly the lowest-scoring block is dropped:

    * zeroing the compressed row of block 16 (the block the step completes) *before* the step, the
      state a score-before-write step would score with, must leave the selection unchanged: the
      step rewrites that row from the ring before it scores it;
    * zeroing every visible row 0..16 must change the selection, and the last row in particular:
      the rows really are scored against the side caches.
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)
        hidden = hidden_rows(layer, T)
        prefill_layer_64(model, vllm_config, eos)
        monkeypatch.setattr(layer.indexer, "block_topk", SELF_BLOCK)  # after the prefill: it pads to output_width
        snap = snapshot_state(model)
        sel_intact = select_multi(layer, hidden, lattice_md(SLOT_A))
        restore_state(model, snap)
        comp, _ = side_caches(layer)
        comp[COMP_SELF].zero_()
        sel_stale = select_multi(layer, hidden, lattice_md(SLOT_A))
        assert torch.equal(sel_stale, sel_intact), (
            f"pre-zeroing compressed row {COMP_SELF} changed the selection: the step scored block "
            f"{SELF_BLOCK} before writing it (write-then-score broken)")
        restore_state(model, snap)
        comp[:SELF_BLOCK + 1].zero_()
        sel_zeroed = select_multi(layer, hidden, lattice_md(SLOT_A))
        changed = int((sel_intact != sel_zeroed).sum())
        print(f"zeroed compressed rows 0..{SELF_BLOCK}: {changed} of {sel_intact.numel()} entries changed")
        assert changed > 0, "the selection ignores the compressed side cache: the rows are not scored against it"
        assert not torch.equal(sel_intact[-1], sel_zeroed[-1]), "the last row (the block's own row) is not sensitive"


def score_before_write_select(self, q, raw_k, positions, cos_sin_at, md, load, store, slot_mapping, cache, rpb):
    """``_select_decode`` with the two statements swapped back: score BEFORE write (the mutation).

    A copy of the decode branch that scores the rows against the side caches as they stand at the
    step's start and writes the ring and the compressed rows AFTER the selection, the order a
    pre-M3 decoder would score with.  Only ``test_write_before_score_mutation_fails_the_chain_parity``
    monkeypatches it onto ``HpuQsaIndexer``.
    """
    D, ratio = self.index_head_dim, self.compress_ratio
    topk = self.block_topk
    T = int(positions.numel())
    qsl = getattr(md, "query_start_loc_p", None)
    if qsl is not None and int(qsl.numel()) >= 2:
        bs = int(qsl.numel()) - 1
        lens = (qsl[1:].long() - qsl[:bs].long()).to(device=positions.device)
    else:
        bs, lens = int(load.numel()), None
    T_cols = T // bs
    if lens is None:
        valid = torch.ones(T, dtype=torch.bool, device=positions.device)
    else:
        cols = torch.arange(T, device=positions.device)
        valid = (cols % T_cols) < lens.repeat_interleave(T_cols)
    n_slots = int(load.numel())
    if n_slots == T:
        load_rows, store_rows = load, store
    else:
        load_rows, store_rows = load.repeat_interleave(T_cols), store.repeat_interleave(T_cols)
    scratch = torch.full((T,), self.key_ring.shape[0] - 1, dtype=torch.long, device=positions.device)
    blk = ring_block_rows(self.key_ring, torch.where(valid, load_rows, scratch), positions, raw_k, ratio)
    complete = valid & ((positions % ratio) == ratio - 1)
    cs, sn = cos_sin_at(positions - (ratio - 1))
    keys = compress_block_keys(blk, self.k_layernorm.weight, self.eps, cs, sn)
    MB = int(md.block_table.shape[1])
    n = MB * rpb
    bt = md.block_table.long()
    blk_idx = torch.arange(n, device=positions.device)
    idx = bt.gather(1, (blk_idx // rpb)[None, :].expand(T, -1)) * rpb + (blk_idx % rpb)[None, :]
    all_rows = cache[idx.reshape(-1)].view(T, n, D)  # scored BEFORE the step writes anything
    scores = torch.relu(torch.einsum("thd,tnd->thn", q.float(), all_rows.float())).sum(1) / (D**0.5)
    sel = expand_selection(select_blocks(scores, visible_blocks(positions, ratio), topk), positions, ratio)
    ring_write(self.key_ring, torch.where(valid, store_rows, scratch), positions, raw_k)
    row_slots = compressed_row_slots(slot_mapping, ratio)
    if bool(complete.any()):
        cache.index_put_((row_slots[complete],), keys[complete])
    return sel


def test_write_before_score_mutation_fails_the_chain_parity(tiny_checkpoint, monkeypatch):
    """The Acceptance mutation: score-before-write breaks test (1)'s parity on the block-completing row.

    The chain reference is taken with the real (write-then-score) code first, then
    ``_select_decode`` is replaced by ``score_before_write_select`` and the 4-token step runs.
    With ``block_topk = 16`` the 17 blocks visible at position 67 are one over budget: the
    mutated step scores block 16's compressed row BEFORE its own step writes it (the prefill
    left it zero), drops it from the top-16, and its last row differs from the chain's, while
    the rows before the completing one (16 visible blocks, all within budget) stay equal.
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)
        hidden = hidden_rows(layer, T)
        prefill_layer_64(model, vllm_config, eos)
        monkeypatch.setattr(layer.indexer, "block_topk", SELF_BLOCK)  # after the prefill: it pads to output_width

        snap = snapshot_state(model)
        sel_chain = select_chain(layer, hidden, SLOT_A, T)  # reference: the real write-then-score code
        restore_state(model, snap)
        monkeypatch.setattr(HpuQsaIndexer, "_select_decode", score_before_write_select)
        sel_mut = select_multi(layer, hidden, lattice_md(SLOT_A))

        assert torch.equal(sel_mut[:T - 1], sel_chain[:T - 1]), (
            "the mutation changed a row that completes no block: those rows must be insensitive")
        assert not torch.equal(sel_mut[-1], sel_chain[-1]), (
            f"the mutation did not break the parity on row {T - 1} (position {POS0 + T - 1}, the row that "
            "completes block 16): test (1) is not sensitive to the write/score order")
        picked = {int(p) for p in sel_mut[-1] if int(p) >= 0}
        assert not any(p in picked for p in SELF_ROWS), (
            f"the mutated last row still selects block {SELF_BLOCK}: it scored the compressed row the step "
            "was about to write, so it did not score before writing")


def test_ragged_padding_rows_touch_no_state(tiny_checkpoint):
    """A padded 8-row step of two requests (4 and 2 real tokens) leaves every slot's state as its solo run.

    Request ``SLOT_A`` has 4 real tokens at positions 64..67, request ``SLOT_B`` 2 at 64..65, so its
    two padding rows sit on ``slot_mapping = -1`` and position 0. The padded step's ring and
    compressed rows must equal the two requests' single-token chains on the shared caches
    (A first, then B, the cache state the padded step scores B's rows with), the ring's
    scratch row ``key_ring[3]`` is the only row the padding may write, and the null slot 0 must stay at
    zero. The rows that complete no block write the compressed cache's scratch row (``COMP_TAIL``), the
    row the fixed-shape decode write parks them on; that write is what ``test_decode_write_has_fixed_shape``
    pins.
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)
        hidden = hidden_rows(layer, 2 * T)
        prefill_layer_64(model, vllm_config, eos)

        # Reference: A's 4 single-token steps, then B's 2 on the same caches (in the padded step
        # A's rows write compressed row 16 before B's rows are scored, and block 16 is visible at
        # positions 64/65, so B's reference has to see A's write too).
        snap = snapshot_state(model)
        sel_a = select_chain(layer, hidden, SLOT_A, T)
        comp, ring = side_caches(layer)
        chain_comp_a, chain_ring_a = comp.clone(), ring.clone()
        sel_b = select_chain(layer, hidden[4:], SLOT_B, 2)
        comp, ring = side_caches(layer)
        chain_comp_b, chain_ring_b = comp.clone(), ring.clone()

        # The padded ragged step: 8 rows, slots [A, B], real lengths [4, 2].
        restore_state(model, snap)
        md = lattice_md(SLOT_A, query_lens=[T, 2], slots=[SLOT_A, SLOT_B])
        sel_pad = select_multi(layer, hidden, md)
        comp, ring = side_caches(layer)
        pad_comp, pad_ring = comp.clone(), ring.clone()

        assert torch.equal(sel_pad[:T], sel_a), "the padded step's rows for request A differ from its solo chain"
        assert torch.equal(sel_pad[T:T + 2], sel_b), "the padded step's rows for request B differ from its solo chain"
        a_gap = (pad_ring[SLOT_A] - chain_ring_a[SLOT_A]).abs().max().item()
        b_gap = (pad_ring[SLOT_B] - chain_ring_b[SLOT_B]).abs().max().item()
        assert a_gap <= ROW_TOL, (f"ring slot {SLOT_A} after the padded ragged step vs its solo chain: "
                                  f"max |delta| = {a_gap:.3e} > {ROW_TOL:.0e}")
        assert b_gap <= ROW_TOL, (f"ring slot {SLOT_B} after the padded ragged step vs its solo chain: "
                                  f"max |delta| = {b_gap:.3e} > {ROW_TOL:.0e}")
        comp_gap = (pad_comp[COMP_SELF] - chain_comp_a[COMP_SELF]).abs().max().item()
        assert comp_gap <= ROW_TOL, (f"compressed row {COMP_SELF} after the padded ragged step vs A's chain: "
                                     f"max |delta| = {comp_gap:.3e} > {ROW_TOL:.0e}")
        assert bool(pad_ring[SCRATCH_SLOT].any()), (
            "the padding rows wrote nothing at all: they must be parked on the ring's scratch row "
            f"key_ring[{SCRATCH_SLOT}], not on a request's slot")
        assert not bool(pad_ring[NULL_SLOT].any()), "a padding row wrote the null slot 0 of the ring"
        solo_b = chain_ring_b[SLOT_B]
        assert bool(solo_b.any()) and bool(chain_ring_a[SLOT_A].any())


def test_decode_write_has_fixed_shape(tiny_checkpoint):
    """The decode compressed-row write is one fixed-shape ``index_put_``: only the scratch row and the
    completing row's row move, so HPU graph capture sees no data-dependent shape and no host sync.

    A step's non-completing rows (positions 64..66 complete no block) must not write their own block's
    row: they write the cache's LAST row, the scratch row no request's block table reaches. A chain of
    two such steps therefore leaves every row but the last untouched; the four-step chain 64..67
    additionally writes the row of the block position 67 completes (``COMP_SELF``) and nothing else.
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)
        prefill_layer_64(model, vllm_config, eos)

        snap = snapshot_state(model)
        before = side_caches(layer)[0].clone()
        select_chain(layer, hidden_rows(layer, 2), SLOT_A, 2)  # positions 64, 65: no block completes
        after = side_caches(layer)[0]
        assert torch.equal(after[:-1], before[:-1]), (
            "the non-completing rows of positions 64, 65 wrote a compressed row of a request: only the "
            "scratch row (the cache's last row) may move")
        assert not torch.equal(after[-1], before[-1]), (
            f"the scratch row (the cache's last row, {after.shape[0] - 1}) kept the prefill's values: the "
            "non-completing rows wrote nothing, so the write is not the fixed-shape one")

        restore_state(model, snap)
        select_chain(layer, hidden_rows(layer, 4), SLOT_A, 4)  # positions 64..67: 67 completes block 16
        after = side_caches(layer)[0]
        assert not torch.equal(after[COMP_SELF], before[COMP_SELF]), (
            f"the completing row (position {POS0 + T - 1}) left the compressed row {COMP_SELF} of block "
            f"{SELF_BLOCK} at the prefill's values")
        assert not torch.equal(after[-1], before[-1]), (
            "the non-completing rows of the 4-step chain wrote no scratch row")
        mid = torch.cat((after[:COMP_SELF], after[COMP_SELF + 1:-1]))
        mid_before = torch.cat((before[:COMP_SELF], before[COMP_SELF + 1:-1]))
        assert torch.equal(mid, mid_before), (
            "a compressed row of the prefill changed: the decode write must move the completing row's row "
            f"({COMP_SELF}) and the scratch row only")


def test_single_token_decode_metadata_is_unchanged(tiny_checkpoint):
    """``num_tokens=1`` builds exactly the single-token metadata the runner builds, and selects the same rows.

    The M3 extension of ``decode_metadata`` must be a strict superset of the single-token case:
    the 1-D ``(1,)`` state indices, no ``num_accepted_tokens``, ``query_start_loc_p = [0, 1]`` and
    the ``[1, max_blocks]`` block table, and its selection equals the chain's first row (the
    regression guard for the shared decode branch; the full chain parity is
    ``test_sparse_decode_parity.py``).
    """
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        layer = qsa_layer(model)
        layer.indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)
        plain = decode_metadata(SLOT_A, POS0, load_slot=SLOT_A)
        extended = decode_metadata(SLOT_A, POS0, num_tokens=1)
        for name in ("load_indices_tensor", "store_indices_tensor", "query_start_loc_p", "slot_mapping",
                     "block_table"):
            a, b = getattr(plain, name), getattr(extended, name)
            assert a.shape == b.shape and torch.equal(a, b), f"num_tokens=1 changed {name}: {b} != {a}"
        assert extended.num_accepted_tokens is None and plain.num_accepted_tokens is None
        assert extended.positions.tolist() == [POS0]

        hidden = hidden_rows(layer, T)
        prefill_layer_64(model, vllm_config, eos)
        snap = snapshot_state(model)
        csa = cos_sin_at(layer)
        sel_plain = layer.indexer.select(hidden[0:1], torch.tensor([POS0]), csa, plain)
        restore_state(model, snap)
        sel_extended = layer.indexer.select(hidden[0:1], torch.tensor([POS0]), csa, extended)
        assert torch.equal(sel_plain, sel_extended), "the single-token selection changed with the M3 extension"
        restore_state(model, snap)
        sel_chain = select_chain(layer, hidden, SLOT_A, T)
        assert torch.equal(sel_plain, sel_chain[0:1]), "the single-token row is not the chain's first row"


@pytest.mark.xfail(raises=(AssertionError, NotImplementedError), strict=True,
                   reason="needs the M3-T6 PLE speculative path (ple_layer.py raises NotImplementedError for "
                   "num_accepted_tokens and rejects a lattice of more than one column) and the M3-T5a GDN "
                   "spec-state fixture (GDN asserts cache_group_idx for a multi-dimensional state index and "
                   "its conv state needs (kernel - 1) + T rows, the harness binds kernel - 1); the end-to-end "
                   "MTP parity is M3-T8")
def test_whole_model_multi_token_step(tiny_checkpoint):
    """End-to-end: one 4-token decode forward == four 1-token forwards on the last row's logits (M3-T8)."""
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = model_sparse_attns(model)
        prefill_layer_64(model, vllm_config, eos)
        ids = list(range(1000, 1000 + PREFILL_LEN)) + [1234] * T

        def step_multi():
            for att in atts:
                att.slot = SLOT_A
            md = lattice_md(SLOT_A)
            with torch.no_grad(), set_forward_context(md, vllm_config):
                hidden = model(input_ids=torch.tensor(ids[POS0:POS0 + T]),
                               positions=torch.arange(POS0, POS0 + T),
                               **ple_kwargs(T, eos, context=(ids[POS0 - 2], ids[POS0 - 1])))
            return fp32_logits(model, hidden)[-1]

        def chain():
            last = None
            for j in range(T):
                for att in atts:
                    att.slot = SLOT_A
                pos = POS0 + j
                with torch.no_grad(), set_forward_context(decode_metadata(SLOT_A, pos, load_slot=SLOT_A),
                                                          vllm_config):
                    hidden = model(input_ids=torch.tensor([ids[pos]]),
                                   positions=torch.tensor([pos]),
                                   **ple_kwargs(1, eos, context=(ids[pos - 2], ids[pos - 1])))
                last = fp32_logits(model, hidden)[-1]
            return last

        snap = snapshot_state(model)
        last_chain = chain()
        restore_state(model, snap)
        last_step = step_multi()
        gap = (last_step - last_chain).abs().max().item()
        assert gap <= 1e-4, (f"4-token decode step vs four 1-token steps on the last row's logits: "
                             f"max |delta logit| = {gap:.3e}")
