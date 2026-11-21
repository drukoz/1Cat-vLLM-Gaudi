# SPDX-License-Identifier: Apache-2.0
"""Flash-Next sparse-QSA decode metadata (``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``) against the dense buffers.

With the flag on, a decode or verification step of the sparse-QSA model builds no dense paged-attention buffers
(``block_list`` / ``block_groups`` / ``block_usage``, and the ``block_mapping`` / ``attn_bias`` the metadata
processor builds from them), the block table is one row per request built with numpy, the decode bucket is
chosen by batch size only, and the MTP drafter's per-step metadata follows the same rules
(``vllm_gaudi/v1/worker/flash_next_decode_metadata.py``). These tests hold that to the dense path:

* the pure builders give the same tables and slots as the code they replace;
* the whole serving loop (``engine_harness``: vLLM's scheduler, the real ``HPUModelRunner``, the tiny model,
  sparse QSA, the recurrent state pool, prefix caching) gives the same tokens and bit-identical logits with
  the flag on and off, for plain decode and for k = 3 verification steps (oracle drafts with forced
  rejections, through the n-gram method, which needs no draft model), and the tokens are the HF reference's;
* ``test_decode_metadata_graphs.py``: one decode graph per batch size, three fewer copies per step, the
  refusal under dense QSA.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from types import MethodType, SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.v1.worker.hpu_model_runner as runner_module  # noqa: E402
from vllm_gaudi.v1.spec_decode.hpu_flash_next_mtp import HpuFlashNextMtpProposer  # noqa: E402
from vllm_gaudi.v1.worker.flash_next_decode_metadata import (  # noqa: E402
    draft_step_block_table_and_slots, fill_pad_slots, request_block_table, sparse_decode_metadata_refusal)
from vllm_gaudi.v1.worker.flash_next_inputs import build_block_table, decode_block_table_width  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = "VLLM_HPU_FLASH_NEXT_DECODE_METADATA"
PAD = 977


# ---------------------------------------------------------------------------------------------------------------
# The pure builders against the code they replace


def _random_table(rng: np.random.Generator, rows: int, width: int) -> torch.Tensor:
    return torch.from_numpy(rng.integers(0, PAD, size=(rows, width), dtype=np.int32))


@pytest.mark.parametrize("width", [40, 64, 90])
def test_request_block_table_is_build_block_table_of_the_row_prefixes(width):
    rng = np.random.default_rng(3)
    table = _random_table(rng, 9, width)
    num_blocks = [min(n, width) for n in (1, 7, 64, 3, 0, 33, 64, 2)]
    want = build_block_table([table[i, :n].tolist() for i, n in enumerate(num_blocks)], 12, 64, PAD)
    got = request_block_table(table, num_blocks, 1, 12, 64, PAD)
    assert got.dtype == torch.int32 and torch.equal(got, want)


def test_request_block_table_repeats_a_request_over_its_verification_rows():
    rng = np.random.default_rng(4)
    table = _random_table(rng, 5, 64)
    num_blocks = [5, 9, 1]
    got = request_block_table(table, num_blocks, 4, 20, 64, PAD)
    want_rows = [table[i, :n].tolist() for i, n in enumerate(num_blocks) for _ in range(4)]
    assert torch.equal(got, build_block_table(want_rows, 20, 64, PAD))


@pytest.mark.parametrize("mode", [0, 1])
def test_request_block_table_at_the_qsa_table_width(mode):
    """The runner hands VLLM_HPU_QSA_TABLE_WIDTH's width (decode_block_table_width) to both builders, the dense
    decode metadata's and the sparse one's, so the two tables stay equal at the batch's width."""
    rng = np.random.default_rng(5)
    engine_width = 1024
    table = _random_table(rng, 4, engine_width)
    num_blocks = [5, 70, 1, 129]
    width = decode_block_table_width(num_blocks, engine_width, mode)
    assert width == (256 if mode == 1 else engine_width)
    got = request_block_table(table, num_blocks, 3, 12, width, PAD)
    want_rows = [table[i, :n].tolist() for i, n in enumerate(num_blocks) for _ in range(3)]
    assert got.shape == (12, width) and torch.equal(got, build_block_table(want_rows, 12, width, PAD))


def test_request_block_table_refuses_what_build_block_table_refuses():
    table = torch.zeros((2, 70), dtype=torch.int32)
    with pytest.raises(ValueError):
        request_block_table(table, [65, 1], 1, 4, 64, PAD)
    with pytest.raises(ValueError):
        request_block_table(torch.zeros((2, 8), dtype=torch.int32), [9, 1], 1, 4, 64, PAD)


@pytest.mark.parametrize("seed", range(6))
def test_fill_pad_slots_is_the_cycle_loop(seed):
    rng = random.Random(seed)
    rows, width, block = rng.randint(1, 9), rng.randint(1, 6), 128
    lens = [rng.randint(0, width) for _ in range(rows)]
    base = PAD * block
    want = torch.arange(rows * width, dtype=torch.int64).view(rows, width) * 7
    got = want.clone()
    cycle = itertools.cycle(range(base, base + block))
    for row, length in enumerate(lens):  # hpu_model_runner._create_decode_input_data, flag 0
        if length < width:
            want[row, length:].apply_(lambda _, ds=cycle: next(ds))
    fill_pad_slots(got, lens, base, block)
    assert torch.equal(got, want)


def _dense_draft_metadata(table: torch.Tensor, positions: torch.Tensor, max_model_len: int, block_size: int):
    """``HpuFlashNextMtpProposer.prepare_attn_metadata`` with the flag off, on a runner stand-in."""
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    runner = SimpleNamespace(block_size=block_size,
                             attn_block_size=block_size,
                             _PAD_BLOCK_ID=PAD,
                             _PAD_SLOT_ID=PAD * block_size,
                             _block_table_max_blocks=64,
                             defragmenter=SimpleNamespace(enabled=False, resolve_all=lambda rows: rows,
                                                          resolve=lambda b: b),
                             use_contiguous_pa=False,
                             bucketing_manager=SimpleNamespace(find_decode_bucket=lambda bs, n: (bs, 1, n)),
                             get_dp_padding=lambda n: 0,
                             model_config=SimpleNamespace(dtype=torch.bfloat16))
    runner.get_habana_paged_attn_buffers = MethodType(HPUModelRunner.get_habana_paged_attn_buffers, runner)
    drafter = SimpleNamespace(max_model_len=max_model_len, device=torch.device("cpu"))
    return HpuFlashNextMtpProposer.prepare_attn_metadata(drafter, table, positions, runner, None)


@pytest.mark.parametrize("batch_size", [3, 5])
def test_draft_step_table_and_slots_are_prepare_attn_metadata(monkeypatch, batch_size):
    monkeypatch.delenv(FLAG, raising=False)
    rng = np.random.default_rng(5)
    table = _random_table(rng, 3, 64)
    # crossing a page (127 -> 128), a long context, the max_model_len clamp
    positions = torch.tensor([127, 128 * 40 + 5, 4096] + [0] * (batch_size - 3), dtype=torch.int32)
    dense = _dense_draft_metadata(table, positions, 4096, 128)
    got_table, got_slots = draft_step_block_table_and_slots(table, positions, batch_size, 128, 4096, 64, PAD,
                                                            PAD * 128)
    assert torch.equal(got_table, dense.block_table)
    assert got_slots.dtype == dense.slot_mapping.dtype and torch.equal(got_slots, dense.slot_mapping)


def test_refusal_reasons():
    ok = dict(flag=1, needs_block_table=True, qsa_sparse=True, framework_layout=False, sliding_window=False,
              chunked_attention=False, defragmenter_enabled=False, speculative_method=None, dense_qsa_layers=0)
    assert sparse_decode_metadata_refusal(**ok) is None
    assert sparse_decode_metadata_refusal(**(ok | {"speculative_method": "mtp"})) is None
    assert sparse_decode_metadata_refusal(**(ok | {"speculative_method": "ngram"})) is None
    for change in ({"flag": 2}, {"needs_block_table": False}, {"qsa_sparse": False}, {"dense_qsa_layers": 1},
                   {"framework_layout": True}, {"sliding_window": True}, {"defragmenter_enabled": True},
                   {"speculative_method": "eagle3"}, {"speculative_method": "dflash"}):
        assert sparse_decode_metadata_refusal(**(ok | change)) is not None, change


# ---------------------------------------------------------------------------------------------------------------
# The whole serving loop, flag off against flag on


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _signature(value):
    """What the HPU graph cache keys on (``habana_frameworks.torch.hpu.graphs.input_hash``): tensors by shape
    and dtype, everything else by value."""
    if isinstance(value, torch.Tensor):
        return ("T", tuple(value.shape), str(value.dtype))
    if isinstance(value, dict):
        return tuple(sorted((key, _signature(item)) for key, item in value.items()))
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return tuple((field, _signature(getattr(value, field))) for field in value._fields)
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    return ("V", repr(value))


class _Recorder:
    """Every decode-phase forward's graph signature and every logits tensor of one engine."""

    def __init__(self, engine) -> None:
        self.logits: list[torch.Tensor] = []
        self.decode_signatures: list[tuple] = []
        self.query_widths: list[int] = []  # the widest request of each decode step (1 + drafts)
        runner = engine.runner
        forward = runner.model.forward
        select = runner.model.select_and_compute_logits
        build = runner._create_decode_input_data

        def recording_build(num_decodes, num_scheduled_tokens, *args, **kwargs):
            self.query_widths.append(max(num_scheduled_tokens[:num_decodes]))
            return build(num_decodes, num_scheduled_tokens, *args, **kwargs)

        runner._create_decode_input_data = recording_build

        def recording_forward(*args, **kwargs):
            metadata = kwargs.get("attn_metadata")
            if metadata is not None and not metadata.is_prompt:
                self.decode_signatures.append(
                    (tuple(kwargs["input_ids"].shape), _signature({k: v for k, v in kwargs.items()})))
            return forward(*args, **kwargs)

        def recording_select(hidden_states, logits_indices):
            hidden, logits = select(hidden_states, logits_indices)
            # A verification step pads each request's logits indices with -1 (hpu_model_runner
            # _prepare_spec_decode_inputs): those rows are the batch's last token row, a padded tail row whose
            # output nothing uses. The rows that carry a real index are compared.
            real = logits_indices.as_subclass(torch.Tensor).reshape(-1) >= 0
            self.logits.append(logits.detach().as_subclass(torch.Tensor)[real].clone())
            return hidden, logits

        runner.model.forward = recording_forward
        runner.model.select_and_compute_logits = recording_select


def _oracle_drafts(engine, reference_by_id: dict, k: int, tamper_every: int = 3):
    """Drafts = the next ``k`` reference tokens; every ``tamper_every``-th draft step one position is wrong."""
    runner = engine.runner
    calls = itertools.count()

    def propose(sampled_token_ids):
        step = next(calls)
        drafts = []
        for index in range(len(sampled_token_ids)):
            req_id = runner.input_batch.req_ids[index] if index < len(runner.input_batch.req_ids) else None
            if req_id is None or req_id not in reference_by_id:
                drafts.append([-1])
                continue
            done = len(runner.requests[req_id].output_token_ids)
            draft = list(reference_by_id[req_id][done:done + k]) or [-1]
            if tamper_every and (step + index) % tamper_every == 0 and len(draft) > 1:
                wrong = (step + index) % len(draft)
                draft[wrong] = (draft[wrong] + 1) % 1000 + 1
            drafts.append(draft)
        return drafts

    runner.propose_ngram_draft_token_ids = propose


def _serve(tiny_checkpoint, reference, flag: int, *, spec_k: int = 0, max_num_seqs: int = 8):
    """All reference prompts in one batch, each stopping one token earlier than the previous (the batch shrinks
    every step, so it pads), at ``flag``. Returns tokens, logits per step and the decode graph signatures."""
    speculative = None
    extra = {}
    if spec_k:
        speculative = {"method": "ngram", "num_speculative_tokens": spec_k, "prompt_lookup_max": 4,
                       "prompt_lookup_min": 2}
        # Two defects of the program head's speculative path stop the serving loop before it gets here (both in
        # mtp-card-ready's hands): a step whose only work is a prompt chunk that completes no prompt reads an
        # unbound sampling_metadata, and the align-mode state lattice of prefix caching finds null blocks. So
        # every prompt prefills in one step and prefix caching is off; the verification rows are the same.
        extra = {"prefix_caching": False, "max_num_batched_tokens": 4096}
    options = eh.Options(max_num_seqs=max_num_seqs, speculative_config=speculative, env={FLAG: str(flag)}, **extra)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        recorder = _Recorder(engine)
        want = {}
        for index, prompt in enumerate(reference):
            request_id = f"r{index}"
            want[request_id] = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
            engine.add(request_id, prompt["prompt_ids"], max_tokens=len(want[request_id]))
        if spec_k:
            _oracle_drafts(engine, want, spec_k)
        engine.run()
        assert engine.runner.flash_next_sparse_decode_metadata() is (flag == 1)
        return dict(engine.tokens), want, recorder


def _assert_same_run(dense, sparse):
    dense_tokens, want, dense_rec = dense
    sparse_tokens, _, sparse_rec = sparse
    assert sparse_tokens == dense_tokens
    assert dense_tokens == want, "the dense path itself left the HF reference"
    assert len(sparse_rec.logits) == len(dense_rec.logits)
    for step, (a, b) in enumerate(zip(dense_rec.logits, sparse_rec.logits)):
        assert a.shape == b.shape, f"step {step}: {tuple(a.shape)} vs {tuple(b.shape)}"
        assert torch.equal(a, b), f"step {step}: max |diff| {float((a - b).abs().max())}"


def test_decode_tokens_and_logits_match_the_dense_buffers(tiny_checkpoint, reference):
    _assert_same_run(_serve(tiny_checkpoint, reference, 0), _serve(tiny_checkpoint, reference, 1))


def test_k3_verification_tokens_and_logits_match_the_dense_buffers(tiny_checkpoint, reference):
    dense = _serve(tiny_checkpoint, reference, 0, spec_k=3)
    assert dense[2].query_widths.count(4) >= 4, f"too few k = 3 verification steps: {dense[2].query_widths}"
    _assert_same_run(dense, _serve(tiny_checkpoint, reference, 1, spec_k=3))


@pytest.mark.parametrize("state_width,with_offsets", [(1, False), (1, True), (4, False)])
def test_vectorized_state_indices_are_prepare_mamba_state_idxs(state_width, with_offsets):
    """The decode builder's state slots under the flag (``prepare_mamba_state_idxs_vectorized``) against the
    per-request loop: compact GDN groups, a block-table group, padding rows, checkpoint offsets."""
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    rng = np.random.default_rng(6)
    tables = [torch.from_numpy(rng.integers(0, 900, size=(10, 40), dtype=np.int32)) for _ in range(4)]
    runner = object.__new__(HPUModelRunner)
    class _Tables:  # the input batch's MultiGroupBlockTable: .block_tables and [group]
        block_tables = tables

        def __getitem__(self, group):
            return SimpleNamespace(get_cpu_tensor=lambda: tables[group])

    runner.input_batch = SimpleNamespace(req_ids=[f"q{i}" for i in range(10)], block_table=_Tables())
    runner._compact_gdn_group_ids = {0, 2, 3}
    runner._compact_gdn_group_offset = {0: 0, 2: 1, 3: 2}
    runner._gdn_req_to_base_slot = {f"q{i}": int(s) for i, s in enumerate(rng.permutation(10))}
    runner._gdn_max_reqs, runner._gdn_state_slots_per_req, runner._MAMBA_PAD_BLOCK_ID = 12, 4, 901
    req_indices = [3, 0, 7, 5, 9]
    offsets = rng.integers(0, 40, size=len(req_indices))
    checkpoint = [0, 5, 2, -1, 3] if with_offsets else None
    want = HPUModelRunner.prepare_mamba_state_idxs(runner, req_indices, offsets, 8, state_width, checkpoint)
    got = HPUModelRunner.prepare_mamba_state_idxs_vectorized(runner, req_indices, offsets, 8, state_width, checkpoint)
    assert got.dtype == want.dtype == torch.int32 and torch.equal(got, want)
