# SPDX-License-Identifier: Apache-2.0
"""The HPU runner's side of the host PLE n-gram ids (``VLLM_HPU_PLE_HOST_NGRAM_IDS``), on CPU.

``tests/unit_tests/models/flash_next/test_ple_host_ngram_ids.py`` checks ``ple_ops.host_ngram_ids`` against the
device computation and the engine. Here the runner's glue is run with the tensors it has when it builds a batch:

* ``HPUModelRunner._ple_token_rows``: the prompt bucket's token rows as a ``[rows, bucket]`` array;
* ``HPUModelRunner._with_host_ngram_ids``: the PLE inputs of a batch plus the ids, or the PLE inputs unchanged
  when the flag is off or the batch's token ids are not all on the host (reason logged once);
* ``HPUModelRunner._ple_forward_kwargs``: the model inputs of the forward (``ple_ngram_ids`` in place of
  ``ngram_context``);
* the two batch builders themselves, ``_form_prefill_batch`` and ``_create_decode_input_data``, run for real on
  a runner object without a card (only the host-to-device copy and the metadata container are replaced): the
  ids they attach are the ids of exactly the token tensor they send to the device.

The ids are compared with a plain-Python n-gram oracle written from the model's definition (history window, EOS
cut, XOR of ``token * multiplier``, modulo the head's size plus its offset), so this file needs no model.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import inspect  # noqa: E402
import itertools  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.envs as gaudi_envs  # noqa: E402
from vllm_gaudi.extension.utils import align_and_pad  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_ops import HOST_NGRAM_IDS_KWARG  # noqa: E402
from vllm_gaudi.v1.worker import hpu_model_runner as runner_module  # noqa: E402
from vllm_gaudi.v1.worker.flash_next_inputs import build_ple_inputs, dummy_ple_inputs  # noqa: E402
from vllm_gaudi.v1.worker.hpu_model_runner import BatchContents, HPUModelRunner  # noqa: E402

FLAG = "VLLM_HPU_PLE_HOST_NGRAM_IDS"
EOS = 248044  # tests/unit_tests/models/flash_next/tiny_config.json
NGRAM = 3
PER_NGRAM = 2
MULTIPLIERS = (8052911324071, 23703573157769, 20109073645365)  # the tiny checkpoint's (above int32)
SIZES = (2053, 2063, 2069, 2081)
OFFSETS = (0, 2053, 4116, 6185)


class FakeEmbedding(torch.nn.Module):
    """The host tables ``HpuQwen4ExpNGramEmbedding`` keeps (``_set_hash_tables``), without the embedding rows."""

    def __init__(self, shift: int = 0) -> None:
        super().__init__()
        self.eos_token_id = EOS
        self.ngram_size = NGRAM
        self.heads_per_ngram = PER_NGRAM
        self.hash_multipliers = tuple(m + 2 * shift for m in MULTIPLIERS)
        self.hash_head_sizes = SIZES
        self.hash_head_offsets = tuple(o + 10000 * shift for o in OFFSETS)


class FakePleLayer(torch.nn.Module):

    def __init__(self, dense_id: int) -> None:
        super().__init__()
        self.ple_dense_layer_id = dense_id
        self.ple_embedding = FakeEmbedding(dense_id)


def fake_model(ple_layers: int = 1) -> torch.nn.Module:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)] + [FakePleLayer(i) for i in range(ple_layers)])
    return model


def make_runner(model=None, async_scheduling: bool = False) -> HPUModelRunner:
    runner = object.__new__(HPUModelRunner)
    runner.model = fake_model() if model is None else model
    runner.use_async_scheduling = async_scheduling
    return runner


def oracle_ids(row: list[int], context: list[int], shift: int = 0) -> list[list[int]]:
    """The n-gram ids of one row of token ids preceded by ``context``, from the definition (Python ints)."""
    multipliers = [m + 2 * shift for m in MULTIPLIERS]
    offsets = [o + 10000 * shift for o in OFFSETS]
    history = list(context) + [max(token, 0) for token in row]
    out = []
    for column in range(len(row)):
        here = len(context) + column
        window = [history[here]]
        for back in range(1, NGRAM):
            window.append(EOS if EOS in history[here - back:here] else history[here - back])
        mixed = window[0] * multipliers[0]
        ids = []
        for index in range(1, NGRAM):
            mixed ^= window[index] * multipliers[index]
            for head in range((index - 1) * PER_NGRAM, index * PER_NGRAM):
                ids.append(mixed % SIZES[head] + offsets[head])
        out.append(ids)
    return out


@pytest.fixture()
def warnings(monkeypatch):
    """The runner's warning lines of the test."""
    lines = []
    monkeypatch.setattr(runner_module.logger, "warning", lambda message, *args: lines.append(message % args))
    return lines


def test_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_PLE_HOST_NGRAM_IDS == 0
    monkeypatch.setenv(FLAG, "1")
    assert gaudi_envs.VLLM_HPU_PLE_HOST_NGRAM_IDS == 1


@pytest.mark.parametrize("value", [None, "0", "2"], ids=["unset", "0", "2"])
def test_flag_off_leaves_the_ple_inputs_alone(monkeypatch, value):
    """Off (unset, 0 or any value that is not 1): the same 2-tuple object, and the model is never inspected."""
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    runner = object.__new__(HPUModelRunner)  # no .model, no .use_async_scheduling: reading either would raise
    ple_inputs = dummy_ple_inputs(2, 4, EOS, NGRAM)
    assert HPUModelRunner._with_host_ngram_ids(runner, ple_inputs, np.zeros((2, 4), dtype=np.int64)) is ple_inputs
    kwargs = HPUModelRunner._ple_forward_kwargs(ple_inputs, "cpu")
    assert sorted(kwargs) == ["ngram_context", "query_start_loc"]
    assert torch.equal(kwargs["ngram_context"], ple_inputs[1]) and kwargs["ngram_context"].dtype == torch.int32
    assert torch.equal(kwargs["query_start_loc"], ple_inputs[0]) and kwargs["query_start_loc"].dtype == torch.int32


def test_prompt_bucket(monkeypatch):
    """Two requests in a (2, 8) prompt bucket: a fresh prompt and a chunk that continues at position 5.

    The rows are padded by ``align_and_pad`` in place with ``-1``, exactly what ``_form_prefill_batch`` holds
    when it builds the PLE inputs.
    """
    monkeypatch.setenv(FLAG, "1")
    token_ids_cpu = np.zeros((4, 32), dtype=np.int64)
    token_ids_cpu[0, :6] = [EOS, 11, 12, EOS, 14, 15]
    token_ids_cpu[2, :12] = [20, 21, 22, 23, EOS, 25, 26, 27, EOS, EOS, 30, 31]
    computed = np.array([0, 0, 5, 0], dtype=np.int32)
    rows = [token_ids_cpu[0, 0:6].tolist(), token_ids_cpu[2, 5:12].tolist()]
    contexts = [[EOS, EOS], [23, EOS]]
    contents_token_ids = [list(row) for row in rows]
    align_and_pad(contents_token_ids, (2, 8), itertools.repeat(-1))
    assert contents_token_ids[0] == rows[0] + [-1, -1]

    runner = make_runner()
    ple_inputs = build_ple_inputs(token_ids_cpu, computed, [0, 2], 2, 8, EOS, NGRAM)
    assert ple_inputs[1].tolist() == contexts
    tokens = HPUModelRunner._ple_token_rows(contents_token_ids, 2, 8)
    assert tokens.shape == (2, 8)
    out = runner._with_host_ngram_ids(ple_inputs, tokens)

    assert len(out) == 3 and out[0] is ple_inputs[0] and out[1] is ple_inputs[1]
    ids = out[2]
    assert ids.dtype == torch.int32 and tuple(ids.shape) == (1, 16, 4)
    for row in range(2):
        padded = rows[row] + [-1] * (8 - len(rows[row]))  # the model clamps -1 to id 0
        assert ids[0].view(2, 8, 4)[row].tolist() == oracle_ids(padded, contexts[row]), f"row {row}"


def test_unpadded_rows_and_missing_rows_are_padding(monkeypatch):
    """Rows shorter than the bucket and bucket rows without a request hash as id 0 with their context."""
    monkeypatch.setenv(FLAG, "1")
    tokens = HPUModelRunner._ple_token_rows([[5, 6, 7]], 2, 4)
    assert tokens.tolist() == [[5, 6, 7, 0], [0, 0, 0, 0]]
    ple_inputs = dummy_ple_inputs(2, 4, EOS, NGRAM)
    ids = make_runner()._with_host_ngram_ids(ple_inputs, tokens)[2]
    assert ids[0].view(2, 4, 4)[0].tolist() == oracle_ids([5, 6, 7, 0], [EOS, EOS])
    assert ids[0].view(2, 4, 4)[1].tolist() == oracle_ids([0, 0, 0, 0], [EOS, EOS])


@pytest.mark.parametrize("width", [1, 4], ids=["single_token", "verification_1_plus_3"])
def test_decode_batch(monkeypatch, width):
    """A decode batch of three requests padded to four rows, as ``_create_decode_input_data`` holds it: the CPU
    ``token_ids`` (viewed ``[rows * width, 1]`` for a multi-token step) and ``build_ple_inputs`` of the batch."""
    monkeypatch.setenv(FLAG, "1")
    token_ids_cpu = np.zeros((3, 40), dtype=np.int64)
    sequences = [[EOS, 7, 8, 9, 10, 11, 12], [3, 4, 5, 6, 7], [EOS, 2, EOS, 4, 5, 6]]
    starts = [3, 1, 2]
    for row, seq in enumerate(sequences):
        token_ids_cpu[row, :len(seq)] = seq
    ple_inputs = build_ple_inputs(token_ids_cpu, np.array(starts, dtype=np.int32), [0, 1, 2], 4, width, EOS, NGRAM)
    token_ids = torch.zeros((4, width), dtype=torch.int32)
    for row, (seq, start) in enumerate(zip(sequences, starts)):
        token_ids[row] = torch.tensor(seq[start:start + width], dtype=torch.int32)
    if width > 1:
        token_ids[1, 2] = -1  # an empty n-gram draft is scheduled as -1
        token_ids = token_ids.view(-1, 1)

    out = make_runner()._with_host_ngram_ids(ple_inputs, token_ids)
    ids = out[2].view(1, 4, width, 4)[0]
    for row, (seq, start) in enumerate(zip(sequences, starts)):
        tokens = seq[start:start + width]
        if width > 1 and row == 1:
            tokens[2] = -1
        context = [seq[start - 2] if start >= 2 else EOS, seq[start - 1]]
        assert ids[row].tolist() == oracle_ids(tokens, context), f"row {row}"
    assert ids[3].tolist() == oracle_ids([0] * width, [EOS, EOS])  # the padding row


def test_every_ple_layer_gets_its_row(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    runner = make_runner(fake_model(ple_layers=2))
    ple_inputs = dummy_ple_inputs(1, 5, EOS, NGRAM)
    ids = runner._with_host_ngram_ids(ple_inputs, torch.tensor([[9, 8, EOS, 6, 5]], dtype=torch.int32))[2]
    assert tuple(ids.shape) == (2, 5, 4)
    assert ids[0].tolist() == oracle_ids([9, 8, EOS, 6, 5], [EOS, EOS], shift=0)
    assert ids[1].tolist() == oracle_ids([9, 8, EOS, 6, 5], [EOS, EOS], shift=1)


def test_async_scheduling_keeps_the_device_path_and_says_so_once(monkeypatch, warnings):
    """Under async scheduling a decode step's token ids are written on the device from the previous step's
    samples (``_prepare_input_ids``): the host cannot hash them, so every batch keeps the device path."""
    monkeypatch.setenv(FLAG, "1")
    runner = make_runner(async_scheduling=True)
    ple_inputs = dummy_ple_inputs(2, 1, EOS, NGRAM)
    for _ in range(3):
        assert runner._with_host_ngram_ids(ple_inputs, torch.zeros((2, 1), dtype=torch.int32)) is ple_inputs
    assert len(warnings) == 1 and FLAG in warnings[0] and "async scheduling" in warnings[0]


def test_model_without_host_tables_keeps_the_device_path_and_says_so_once(monkeypatch, warnings):
    monkeypatch.setenv(FLAG, "1")
    runner = make_runner(torch.nn.Linear(2, 2))
    ple_inputs = dummy_ple_inputs(2, 1, EOS, NGRAM)
    for _ in range(3):
        assert runner._with_host_ngram_ids(ple_inputs, torch.zeros((2, 1), dtype=torch.int32)) is ple_inputs
    assert len(warnings) == 1 and FLAG in warnings[0]


def test_inconsistent_ple_tables_keep_the_device_path_and_say_so_once(monkeypatch, warnings):
    """Two PLE layers that claim the same dense id cannot be told apart in the ids tensor: no host ids, a
    warning with the reason, and the server keeps running on the device path."""
    monkeypatch.setenv(FLAG, "1")
    model = fake_model(ple_layers=2)
    model.layers[2].ple_dense_layer_id = 0
    runner = make_runner(model)
    ple_inputs = dummy_ple_inputs(2, 1, EOS, NGRAM)
    for _ in range(3):
        assert runner._with_host_ngram_ids(ple_inputs, torch.zeros((2, 1), dtype=torch.int32)) is ple_inputs
    assert len(warnings) == 1 and FLAG in warnings[0] and "ple_dense_layer_id" in warnings[0]


def test_token_rows_that_are_not_one_per_request_keep_the_device_path(monkeypatch, warnings):
    """Merged prefill packs several requests into one bucket row: the rows no longer line up with
    ``query_start_loc``, so that batch keeps the device path (per call; the next regular batch is hashed)."""
    monkeypatch.setenv(FLAG, "1")
    assert HPUModelRunner._ple_token_rows([[1, 2], [3, 4]], 1, 4) is None  # more rows than the bucket has
    assert HPUModelRunner._ple_token_rows([[1, 2, 3, 4, 5]], 1, 4) is None  # a row longer than the bucket
    runner = make_runner()
    ple_inputs = dummy_ple_inputs(1, 4, EOS, NGRAM)
    for _ in range(3):
        assert runner._with_host_ngram_ids(ple_inputs, None) is ple_inputs
    five_ids_for_four_tokens = torch.zeros(5, dtype=torch.int32)
    assert runner._with_host_ngram_ids(ple_inputs, five_ids_for_four_tokens) is ple_inputs
    assert len(warnings) == 1 and FLAG in warnings[0]
    assert len(runner._with_host_ngram_ids(ple_inputs, torch.zeros((1, 4), dtype=torch.int32))) == 3


def test_forward_kwargs_carry_the_ids_in_place_of_the_context(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    ple_inputs = make_runner()._with_host_ngram_ids(dummy_ple_inputs(2, 3, EOS, NGRAM),
                                                    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))
    kwargs = HPUModelRunner._ple_forward_kwargs(ple_inputs, "cpu")
    assert sorted(kwargs) == sorted(["query_start_loc", HOST_NGRAM_IDS_KWARG])
    assert HOST_NGRAM_IDS_KWARG == "ple_ngram_ids"
    assert kwargs[HOST_NGRAM_IDS_KWARG].dtype == torch.int32
    assert tuple(kwargs[HOST_NGRAM_IDS_KWARG].shape) == (1, 6, 4)
    assert torch.equal(kwargs[HOST_NGRAM_IDS_KWARG], ple_inputs[2])
    assert torch.equal(kwargs["query_start_loc"], ple_inputs[0])


def test_the_shapes_are_fixed_per_bucket(monkeypatch):
    """A graph is matched on its inputs' metadata: two batches of one bucket must send the same shapes and dtypes."""
    monkeypatch.setenv(FLAG, "1")
    runner = make_runner()
    shapes = set()
    for rows in ([[1, 2, 3]], [[7] * 8], [[EOS] * 5, [4]]):
        tokens = HPUModelRunner._ple_token_rows(rows, 2, 8)
        kwargs = HPUModelRunner._ple_forward_kwargs(
            runner._with_host_ngram_ids(dummy_ple_inputs(2, 8, EOS, NGRAM), tokens), "cpu")
        shapes.add(tuple((name, tuple(value.shape), value.dtype) for name, value in sorted(kwargs.items())))
    assert len(shapes) == 1, shapes


def batch_builder_runner(monkeypatch, token_ids_cpu, num_computed, **attributes) -> HPUModelRunner:
    """A runner object that can run the real batch builders on the CPU.

    Only the boundary is replaced: ``async_h2d_copy`` returns the CPU tensor it was given (so the "device" token
    tensor of the result is the tensor the model would receive) and the metadata containers are namespaces. The
    model is a Flash-Next stand-in: it wants raw input tokens and has one PLE layer with host hash tables.
    """

    def keep_on_cpu(source, dest_tensor=None, dtype=None, device=None):
        return source if isinstance(source, torch.Tensor) else torch.tensor(source, dtype=dtype)

    monkeypatch.setattr(runner_module, "async_h2d_copy", keep_on_cpu)
    for maker in ("make_prefill_metadata", "make_decode_metadata"):
        monkeypatch.setattr(runner_module.HPUAttentionMetadataV1, maker, lambda **kwargs: SimpleNamespace(**kwargs))
    model = fake_model()
    model.requires_raw_input_tokens = True
    runner = object.__new__(HPUModelRunner)
    defaults = dict(
        model=model,
        model_config=SimpleNamespace(dtype=torch.bfloat16,
                                     hf_text_config=SimpleNamespace(eos_token_id=EOS, ngram_size=NGRAM)),
        input_batch=SimpleNamespace(req_ids=["a", "b"],
                                    req_id_to_index={"a": 0, "b": 1},
                                    token_ids_cpu=token_ids_cpu,
                                    num_computed_tokens_cpu=np.asarray(num_computed, dtype=np.int32),
                                    block_table=[SimpleNamespace(block_size=128)]),
        use_framework_kv_cache_layout=False,
        use_async_scheduling=False,
        use_merged_prefill=False,
        use_contiguous_pa=False,
        uses_mrope=False,
        interleaved_sliding_window=False,
        model_has_chunked_attention=False,
        num_mamba_like_layers=0,
        attn_block_size=128,
        block_size=128,
        logits_rounding=1,
        pin_memory=False,
        device=torch.device("cpu"),
        profiler=SimpleNamespace(enabled=False),
        requests={},
        speculative_config=None,
        get_dp_padding=lambda count: 0,
        _PAD_BLOCK_ID=99,
        _resolve_block=lambda block: block,
        _resolve_all_blocks=lambda blocks: blocks,
        _get_model_type=lambda: "qwen4_exp",
    )
    for name, value in {**defaults, **attributes}.items():
        setattr(runner, name, value)
    return runner


def assert_ids_match_the_sent_tokens(ple_inputs, sent_token_ids, rows: int, width: int) -> None:
    """Every row's ids are the oracle's for the token row the model receives and the context row built for it."""
    assert len(ple_inputs) == 3, "the batch builder attached no host n-gram ids"
    query_start_loc, ngram_context, ids = ple_inputs
    assert query_start_loc.tolist() == [row * width for row in range(rows + 1)]
    assert ids.dtype == torch.int32 and tuple(ids.shape) == (1, rows * width, 4)
    sent = sent_token_ids.reshape(rows, width).tolist()
    for row in range(rows):
        assert ids[0].view(rows, width, 4)[row].tolist() == oracle_ids(sent[row], ngram_context[row].tolist()), row


SEQUENCES = ([EOS, 7, 8, 9, 10, 11, 12, 13, EOS, 15, 16, 17], [3, 4, EOS, 6, 7, 8, 9, 10])


def host_token_ids() -> np.ndarray:
    token_ids_cpu = np.zeros((2, 64), dtype=np.int32)
    for row, seq in enumerate(SEQUENCES):
        token_ids_cpu[row, :len(seq)] = seq
    return token_ids_cpu


@pytest.mark.parametrize("flag", ["0", "1"])
def test_form_prefill_batch_hashes_the_bucket_it_sends(monkeypatch, flag):
    """The real ``_form_prefill_batch``: a 7-token chunk that starts at position 5, in the (1, 8) prompt bucket."""
    monkeypatch.setenv(FLAG, flag)
    runner = batch_builder_runner(monkeypatch,
                                  host_token_ids(), [5, 0],
                                  _get_prompt_bucketing_fn=lambda: (lambda query_lens, ctx_blocks: (1, 8, 1)))
    chunk = SEQUENCES[0][5:12]
    contents = BatchContents(req_ids=["a"],
                             token_ids=[list(chunk)],
                             context_lens=[5],
                             prompt_lens=[12],
                             blocks=[[4]],
                             logits_positions=[[6]])
    data = runner._form_prefill_batch(contents)

    sent = data.token_ids[0]
    assert sent.tolist() == [chunk + [-1]]  # the bucket's padding reaches the model as -1
    ple_inputs = data.ple_inputs[0]
    assert ple_inputs[1].tolist() == [SEQUENCES[0][3:5]]
    if flag == "0":
        assert len(ple_inputs) == 2
        assert sorted(HPUModelRunner._ple_forward_kwargs(ple_inputs, "cpu")) == ["ngram_context", "query_start_loc"]
    else:
        assert_ids_match_the_sent_tokens(ple_inputs, sent, rows=1, width=8)


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("lengths", [[1, 1], [4, 2]], ids=["single_token", "ragged_verification"])
def test_create_decode_input_data_hashes_the_step_it_sends(monkeypatch, flag, lengths):
    """The real ``_create_decode_input_data``: two requests in a decode bucket of four rows, one token each or
    a speculative verification step of 4 and 2 tokens (``input_ids_cpu`` holds the drafts, as after
    ``_prepare_inputs``)."""
    monkeypatch.setenv(FLAG, flag)
    starts = [3, 1]
    scheduled = [token for seq, start, count in zip(SEQUENCES, starts, lengths) for token in seq[start:start + count]]
    positions = [position for start, count in zip(starts, lengths) for position in range(start, start + count)]
    runner = batch_builder_runner(monkeypatch,
                                  host_token_ids(),
                                  starts,
                                  bucketing_manager=SimpleNamespace(find_decode_bucket=lambda *args: (4, 0, 0)),
                                  input_ids_cpu=torch.tensor(scheduled, dtype=torch.int32),
                                  positions_cpu=torch.tensor(positions, dtype=torch.int32))
    block_table = torch.tensor([[7, 3, 11, 16], [20, 14, 18, 15]], dtype=torch.int32)
    data = runner._create_decode_input_data(2, lengths, np.asarray(starts), block_table)

    width = max(lengths)
    sent = data.token_ids.reshape(4, width)
    for row, (seq, start, count) in enumerate(zip(SEQUENCES, starts, lengths)):
        assert sent[row, :count].tolist() == seq[start:start + count]
    assert data.ple_inputs[1][:2].tolist() == [[7, 8], [EOS, 3]]
    if flag == "0":
        assert len(data.ple_inputs) == 2
    else:
        assert_ids_match_the_sent_tokens(data.ple_inputs, sent, rows=4, width=width)


def test_the_remaining_call_sites_use_the_glue():
    """The data-parallel dummy decode (``sample_tokens``) and the forward (``_execute_model_generic``) need a
    card to run: this pins that they go through the same two functions."""
    assert "_with_host_ngram_ids(" in inspect.getsource(HPUModelRunner.sample_tokens)
    assert "_ple_forward_kwargs(" in inspect.getsource(HPUModelRunner._execute_model_generic)
