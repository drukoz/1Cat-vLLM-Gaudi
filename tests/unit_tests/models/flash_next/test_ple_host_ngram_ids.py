# SPDX-License-Identifier: Apache-2.0
"""The PLE n-gram ids computed on the host (``VLLM_HPU_PLE_HOST_NGRAM_IDS=1``), on CPU.

With the flag off (the default) every forward computes the n-gram ids on the device: the token packing, a
``searchsorted``, the EOS-cut shift scan and the int32 limb hash (``HpuQwen4ExpNGramEmbedding.compute_ngram_ids``).
The runner holds every token id of every batch on the host (Flash-Next runs with async scheduling off and
speculative drafts reach the next step through the scheduler), so with the flag on it computes the same ids with
``ple_ops.host_ngram_ids`` and hands them to the model as an int32 ``[ple_layers, tokens, heads]`` tensor.

Three things are checked here:

1. **The ids.** ``host_ngram_ids`` against the device computation and against the engine's int64 reference
   (``vllm/models/qwen4_exp/nvidia/ple_layer.py``), bit for bit, in the layouts the runner builds
   (``flash_next_inputs.build_ple_inputs`` builds the context, exactly as on a card): whole prompts of 1 to 4096
   tokens, EOS at the start, in the middle, in runs and at the end, padded buckets whose rows have odd lengths,
   a chunked prefill whose n-gram window straddles the chunk boundary (EOS on either side of it), a prefix-cache
   hit, decode steps after a prefill at batch 1 and 32, and an MTP verification step of ``1 + k`` tokens per row.
2. **The layer.** The PLE layer fed the ready ids returns the same output and saves the same state as the layer
   that computes them, over two prefill chunks and a padded prompt bucket followed by decode steps in a padded
   batch, and over an MTP verification step. With ready ids ``compute_ngram_ids`` does not run.
3. **The op count.** The dispatch-level op sequence of one PLE forward with the flag off is pinned (it must not
   change), and with ready ids the id computation is gone from it.
"""

import hashlib
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402

from vllm_gaudi import envs as gaudi_envs  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpNGramEmbedding  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_ops import (  # noqa: E402
    NgramHostTables, collect_ngram_host_tables, host_ngram_ids)
from vllm_gaudi.v1.worker.flash_next_inputs import build_ple_inputs  # noqa: E402

from tests.unit_tests.models.flash_next.test_ple_layer import (  # noqa: E402,F401  (ple_layer is the fixture)
    CHANNELS, SLOTS, SPEC_STATE, TOL, _engine_embedding, _eos, _md, _ngram_inputs, _oracle, _spec_md, ple_layer)
from tests.unit_tests.models.flash_next.test_ple_padded_bucket import bf16_mask, cumulative  # noqa: E402

FLAG = "VLLM_HPU_PLE_HOST_NGRAM_IDS"
NGRAM = 3
HEADS = 16
PAD_ROW = SLOTS - 1


# --------------------------------------------------------------------------------------------------------------
# helpers


def random_sequence(gen: torch.Generator, length: int, vocab: int, eos: int, eos_rate: float = 0.05) -> torch.Tensor:
    """``length`` random ids over the whole vocabulary (0 and ``vocab - 1`` included), EOS at ``eos_rate``."""
    seq = torch.randint(0, vocab, (length, ), generator=gen, dtype=torch.int64)
    if length > 2:
        seq[int(torch.randint(0, length, (1, ), generator=gen))] = 0
        seq[int(torch.randint(0, length, (1, ), generator=gen))] = vocab - 1
    seq[torch.rand(length, generator=gen) < eos_rate] = eos
    return seq


class Batch:
    """An input batch as the runner keeps it on the host: ``token_ids_cpu`` rows and ``num_computed_tokens``."""

    def __init__(self, sequences: list[torch.Tensor], eos: int) -> None:
        self.eos = eos
        self.sequences = sequences
        width = max(int(s.numel()) for s in sequences) + 8
        self.token_ids_cpu = np.zeros((len(sequences), width), dtype=np.int64)
        for row, seq in enumerate(sequences):
            self.token_ids_cpu[row, :seq.numel()] = seq.numpy()

    def step(self, rows: list[int], computed: list[int], scheduled: list[int], bucket_rows: int, bucket_seq: int,
             pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(ids, query_start_loc, ngram_context)`` of one model call, laid out as the HPU runner lays it out.

        Bucket row ``r`` runs ``scheduled[r]`` tokens of input-batch row ``rows[r]`` from position ``computed[r]``;
        ``ids`` is ``[bucket_rows * bucket_seq]`` int32 padded with ``pad`` (``-1`` in a prompt bucket, ``0`` in a
        decode batch) and the two PLE inputs come from the runner's own ``build_ple_inputs``.
        """
        num_computed = np.zeros(len(self.sequences), dtype=np.int32)
        for row, start in zip(rows, computed):
            num_computed[row] = start
        qsl, ctx = build_ple_inputs(self.token_ids_cpu, num_computed, rows, bucket_rows, bucket_seq, self.eos, NGRAM)
        ids = torch.full((bucket_rows, bucket_seq), pad, dtype=torch.int32)
        for r, (row, start, count) in enumerate(zip(rows, computed, scheduled)):
            ids[r, :count] = torch.from_numpy(self.token_ids_cpu[row, start:start + count].astype(np.int32))
        return ids.view(-1), qsl, ctx


def whole_sequence_ids(tables: NgramHostTables, seq: torch.Tensor, eos: int) -> torch.Tensor:
    """The ids of one unpadded, unchunked sequence: ``[len, heads]`` (the oracle the chunked layouts must equal)."""
    qsl = torch.tensor([0, seq.numel()], dtype=torch.int32)
    ctx = torch.full((1, NGRAM - 1), eos, dtype=torch.int32)
    return host_ngram_ids(tables, seq, qsl, ctx)[0]


def assert_three_way(layer, engine, ids, qsl, ctx, label: str) -> torch.Tensor:
    """Host ids == device ids == engine ids for one call; returns the host ids ``[tokens, heads]`` (layer 0)."""
    tables = collect_ngram_host_tables(layer)
    host = host_ngram_ids(tables, ids, qsl, ctx)
    assert host.dtype == torch.int32, label
    assert tuple(host.shape) == (1, ids.numel(), HEADS), f"{label}: {tuple(host.shape)}"
    # The model forward clamps the runner's -1 padding to id 0 before the PLE layer sees it.
    seen = ids.clamp(min=0)
    device = layer.ple_embedding.compute_ngram_ids(seen, qsl, ctx)
    reference = engine.compute_ngram_ids(seen, qsl, ctx)
    assert torch.equal(device, reference), f"{label}: the device ids differ from the engine's (not a host problem)"
    mismatch = (host[0].long() != reference).any(dim=1).nonzero().flatten()
    assert mismatch.numel() == 0, (f"{label}: host ids differ from the engine's on {mismatch.numel()} of "
                                   f"{ids.numel()} tokens, first at flat index {int(mismatch[0])}")
    return host[0]


@pytest.fixture()
def embeddings(ple_layer):
    """``(vllm_config, layer, engine, eos, vocab)``: the HPU PLE layer and the engine's n-gram embedding."""
    vllm_config, layer = ple_layer
    engine = _engine_embedding(vllm_config)
    assert type(engine) is Qwen4ExpNGramEmbedding
    vocab = int(vllm_config.model_config.hf_text_config.vocab_size)
    return vllm_config, layer, engine, _eos(vllm_config), vocab


# --------------------------------------------------------------------------------------------------------------
# 1. the ids


def test_tables_are_the_embeddings_host_tables(embeddings):
    _, layer, engine, eos, _ = embeddings
    tables = collect_ngram_host_tables(layer)
    assert (tables.num_layers, tables.ngram_size, tables.ngram_heads, tables.heads_per_ngram) == (1, NGRAM, HEADS, 8)
    assert tables.eos_token_id == eos
    # The engine keeps the same three tables as int64 buffers on the model device.
    assert tables.multipliers[0].tolist() == engine.layer_multipliers.tolist()
    assert tables.sizes[0].tolist() == engine.ngram_heads_vocab_sizes.tolist()
    assert tables.offsets[0].tolist() == engine.ngram_heads_offsets.tolist()
    assert collect_ngram_host_tables(torch.nn.Linear(2, 2)) is None  # a model without a PLE layer


def test_tables_follow_a_checkpoint_load(embeddings):
    """The runner collects the tables after the weights are loaded: they are the checkpoint's, not the seed's."""
    _, layer, engine, _, _ = embeddings
    table = torch.tensor([8052911324071, 23703573157769, 20109073645365])  # the tiny checkpoint's multipliers
    layer.ple_embedding.load_weights([("layer_multipliers", table)])
    engine.load_weights([("layer_multipliers", table)])
    assert collect_ngram_host_tables(layer).multipliers[0].tolist() == table.tolist()
    ids, qsl, ctx = _ngram_inputs(embeddings[0])
    assert_three_way(layer, engine, ids, qsl, ctx, "after load_weights")


def test_engine_layout_with_ragged_requests(embeddings):
    """The engine's own layout: three requests of 1, 5 and 40 tokens in one flat batch, random contexts."""
    vllm_config, layer, engine, _, _ = embeddings
    ids, qsl, ctx = _ngram_inputs(vllm_config)
    assert_three_way(layer, engine, ids, qsl, ctx, "ragged requests")


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 16, 63, 64, 65, 127, 128, 129, 257, 1000, 1701, 2048, 4096])
def test_whole_prompts_of_every_length(embeddings, length):
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(1000 + length)
    seq = random_sequence(gen, length, vocab, eos)
    qsl = torch.tensor([0, length], dtype=torch.int32)
    ctx = torch.full((1, NGRAM - 1), eos, dtype=torch.int32)
    assert_three_way(layer, engine, seq.to(torch.int32), qsl, ctx, f"{length} tokens")


def eos_patterns(eos: int, vocab: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(7)
    base = torch.randint(0, vocab, (48, ), generator=gen, dtype=torch.int64)
    base[base == eos] = 1
    patterns = {"none": base.clone()}
    for name, where in (("start", [0]), ("start_run", [0, 1, 2]), ("second", [1]), ("middle", [20]),
                        ("middle_run", [20, 21, 22, 23]), ("every_other", list(range(10, 30, 2))),
                        ("gap_of_two", [20, 23]), ("end", [47]), ("end_run", [45, 46, 47]),
                        ("start_and_end", [0, 47])):
        seq = base.clone()
        seq[where] = eos
        patterns[name] = seq
    patterns["all"] = torch.full((48, ), eos, dtype=torch.int64)
    patterns["single_eos"] = torch.tensor([eos], dtype=torch.int64)
    patterns["two_eos"] = torch.tensor([eos, eos], dtype=torch.int64)
    return patterns


def test_eos_at_the_start_in_the_middle_in_runs_and_at_the_end(embeddings):
    _, layer, engine, eos, vocab = embeddings
    for name, seq in eos_patterns(eos, vocab).items():
        qsl = torch.tensor([0, seq.numel()], dtype=torch.int32)
        for ctx_name, ctx in (("eos_ctx", [eos, eos]), ("token_ctx", [11, 12]), ("eos_then_token", [eos, 12]),
                              ("token_then_eos", [11, eos])):
            assert_three_way(layer, engine, seq.to(torch.int32), qsl, torch.tensor([ctx], dtype=torch.int32),
                             f"EOS {name}, context {ctx_name}")


def test_padded_buckets_with_odd_row_lengths(embeddings):
    """Rows of 1701, 300 and 77 real tokens in one 2048-token prompt bucket (padding ``-1``), plus padding rows."""
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(21)
    lengths = [1701, 300, 77]
    batch = Batch([random_sequence(gen, n, vocab, eos) for n in lengths], eos)
    ids, qsl, ctx = batch.step([0, 1, 2], [0, 0, 0], lengths, bucket_rows=4, bucket_seq=2048, pad=-1)
    assert int((ids == -1).sum()) == 4 * 2048 - sum(lengths)
    host = assert_three_way(layer, engine, ids, qsl, ctx, "padded prompt bucket").view(4, 2048, HEADS)
    tables = collect_ngram_host_tables(layer)
    for row, seq in enumerate(batch.sequences):
        assert torch.equal(host[row, :seq.numel()], whole_sequence_ids(tables, seq, eos)), f"row {row}"


@pytest.mark.parametrize("eos_at", [None, 135, 136, 137, 138, (135, 136), (136, 137)],
                         ids=["no_eos", "two_before", "last_of_chunk", "first_of_chunk", "second_of_chunk",
                              "both_context_tokens", "across_the_boundary"])
def test_chunked_prefill_across_the_chunk_boundary(embeddings, eos_at):
    """A 300-token prompt prefilled as 137 + 163 tokens, each chunk in its own padded bucket.

    The first two tokens of the second chunk take their n-gram history from the context the runner re-feeds
    (``token_ids_cpu[start - 2 : start]``); an EOS on either side of the boundary must cut that history exactly
    as it does inside one row.
    """
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(31)
    seq = random_sequence(gen, 300, vocab, eos, eos_rate=0.0)
    if eos_at is not None:
        seq[list(eos_at) if isinstance(eos_at, tuple) else [eos_at]] = eos
    batch = Batch([seq], eos)
    whole = whole_sequence_ids(collect_ngram_host_tables(layer), seq, eos)

    first_ids, first_qsl, first_ctx = batch.step([0], [0], [137], bucket_rows=1, bucket_seq=256, pad=-1)
    second_ids, second_qsl, second_ctx = batch.step([0], [137], [163], bucket_rows=1, bucket_seq=256, pad=-1)
    assert second_ctx.tolist() == [seq[135:137].tolist()]
    first = assert_three_way(layer, engine, first_ids, first_qsl, first_ctx, "first chunk")
    second = assert_three_way(layer, engine, second_ids, second_qsl, second_ctx, "second chunk")
    assert torch.equal(torch.cat((first[:137], second[:163])), whole), "the chunked ids differ from the whole prompt's"


def test_prefix_cache_hit_prefills_only_the_tail(embeddings):
    """896 of 1000 prompt tokens are cached: the only prefill runs tokens 896.. with the context from the host."""
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(41)
    seq = random_sequence(gen, 1000, vocab, eos)
    seq[894], seq[895], seq[896] = 5, eos, 6  # an EOS right before the first prefilled token
    batch = Batch([seq], eos)
    ids, qsl, ctx = batch.step([0], [896], [104], bucket_rows=1, bucket_seq=128, pad=-1)
    assert ctx.tolist() == [[5, eos]]
    tail = assert_three_way(layer, engine, ids, qsl, ctx, "prefix-cache tail")
    whole = whole_sequence_ids(collect_ngram_host_tables(layer), seq, eos)
    assert torch.equal(tail[:104], whole[896:])


@pytest.mark.parametrize("real,padded", [(1, 1), (5, 8), (32, 32)], ids=["batch_1", "batch_5_of_8", "batch_32"])
def test_decode_steps_after_prefill(embeddings, real, padded):
    """Six single-token decode steps of ``real`` requests at different positions in a ``padded``-row batch."""
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(51 + real)
    prompts = [int(n) for n in torch.randint(1, 200, (real, ), generator=gen)]
    prompts[0] = 1  # a request whose history is shorter than the n-gram window
    batch = Batch([random_sequence(gen, n + 6, vocab, eos, eos_rate=0.15) for n in prompts], eos)
    tables = collect_ngram_host_tables(layer)
    whole = [whole_sequence_ids(tables, seq, eos) for seq in batch.sequences]
    for step in range(6):
        positions = [n + step for n in prompts]
        ids, qsl, ctx = batch.step(list(range(real)), positions, [1] * real, bucket_rows=padded, bucket_seq=1, pad=0)
        assert qsl.tolist() == list(range(padded + 1))
        host = assert_three_way(layer, engine, ids, qsl, ctx, f"decode step {step}")
        for row, position in enumerate(positions):
            assert torch.equal(host[row], whole[row][position]), f"step {step}, row {row}, position {position}"


def test_speculative_verification_step(embeddings):
    """An MTP / n-gram verification step: ``1 + k = 4`` tokens per request, ragged rows padded with 0 and the
    empty-draft placeholder ``-1`` (clamped to id 0 by the model). The drafts are on the host before the step."""
    _, layer, engine, eos, vocab = embeddings
    gen = torch.Generator().manual_seed(61)
    lengths = [40, 3, 17, 90]
    batch = Batch([random_sequence(gen, n + 4, vocab, eos, eos_rate=0.2) for n in lengths], eos)
    tables = collect_ngram_host_tables(layer)
    scheduled = [4, 4, 2, 2]  # rows 2 and 3 carry one draft; row 3's is the placeholder
    ids, qsl, ctx = batch.step([0, 1, 2, 3], lengths, scheduled, bucket_rows=8, bucket_seq=4, pad=0)
    ids = ids.clone().view(8, 4)
    ids[3, 1] = -1
    host = assert_three_way(layer, engine, ids.view(-1), qsl, ctx, "verification step").view(8, 4, HEADS)
    for row, (seq, start, count) in enumerate(zip(batch.sequences, lengths, scheduled)):
        real = count if row != 3 else 1  # the placeholder is not the sequence's token
        assert torch.equal(host[row, :real], whole_sequence_ids(tables, seq, eos)[start:start + real]), f"row {row}"


def test_two_ple_layers_get_their_own_ids(embeddings):
    """Row ``i`` of the host ids is PLE layer ``i``'s: every layer hashes with its own multipliers and sizes."""
    vllm_config, layer, _, eos, vocab = embeddings
    config = vllm_config.model_config.hf_text_config
    with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
        second = HpuQwen4ExpNGramEmbedding(config, int(config.ple_embed_dim), 1, prefix="second.ple_embedding",
                                           layer_name="second")
    first = layer.ple_embedding
    tables = NgramHostTables(
        eos_token_id=eos,
        heads_per_ngram=8,
        multipliers=np.array([first.hash_multipliers, second.hash_multipliers], dtype=np.int64),
        sizes=np.array([first.hash_head_sizes, second.hash_head_sizes], dtype=np.int64),
        offsets=np.array([first.hash_head_offsets, second.hash_head_offsets], dtype=np.int64),
    )
    gen = torch.Generator().manual_seed(71)
    seq = random_sequence(gen, 200, vocab, eos).to(torch.int32)
    qsl = torch.tensor([0, 200], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    host = host_ngram_ids(tables, seq, qsl, ctx)
    assert tuple(host.shape) == (2, 200, HEADS)
    assert torch.equal(host[0].long(), first.compute_ngram_ids(seq, qsl, ctx))
    assert torch.equal(host[1].long(), second.compute_ngram_ids(seq, qsl, ctx))
    assert not torch.equal(host[0], host[1])


def test_malformed_layouts_are_refused(embeddings):
    """A layout the engine's packing does not define gets no ids at all (never ids of the wrong request)."""
    _, layer, _, eos, _ = embeddings
    tables = collect_ngram_host_tables(layer)
    ids = torch.arange(10, 16, dtype=torch.int32)
    good_qsl = torch.tensor([0, 2, 6], dtype=torch.int32)
    good_ctx = torch.full((2, 2), eos, dtype=torch.int32)
    assert tuple(host_ngram_ids(tables, ids, good_qsl, good_ctx).shape) == (1, 6, HEADS)
    malformed = {
        "starts above 0": (torch.tensor([1, 2, 6], dtype=torch.int32), good_ctx),
        "decreasing starts": (torch.tensor([0, 4, 2], dtype=torch.int32), good_ctx),
        "no request": (torch.tensor([0], dtype=torch.int32), good_ctx),
        "a context row short": (good_qsl, good_ctx[:1]),
        "a context of one token": (good_qsl, good_ctx[:, :1]),
        "a flat context": (good_qsl, good_ctx.reshape(-1)),
    }
    for name, (qsl, ctx) in malformed.items():
        with pytest.raises(ValueError, match="host n-gram ids"):
            host_ngram_ids(tables, ids, qsl, ctx)
            pytest.fail(f"{name}: host_ngram_ids returned ids")


def test_host_cost_is_small(embeddings):
    """The host function at a batch-32 decode step and at an 8192-token prompt chunk (best of 20, printed)."""
    _, layer, _, eos, vocab = embeddings
    tables = collect_ngram_host_tables(layer)
    gen = torch.Generator().manual_seed(81)

    def best_us(ids, qsl, ctx) -> float:
        best = float("inf")
        for _ in range(20):
            start = time.perf_counter()
            host_ngram_ids(tables, ids, qsl, ctx)
            best = min(best, time.perf_counter() - start)
        return best * 1e6

    decode = best_us(torch.randint(0, vocab, (32, ), generator=gen, dtype=torch.int32),
                     torch.arange(33, dtype=torch.int32), torch.randint(0, vocab, (32, 2), generator=gen,
                                                                       dtype=torch.int32))
    chunk = best_us(torch.randint(0, vocab, (8192, ), generator=gen, dtype=torch.int32),
                    torch.tensor([0, 8192], dtype=torch.int32), torch.full((1, 2), eos, dtype=torch.int32))
    print(f"\nhost_ngram_ids: batch-32 decode {decode:.0f} us, 8192-token chunk {chunk:.0f} us")
    # Loose bounds (a decode step is tens of ms): they catch an accidental per-token Python loop, not jitter.
    assert decode < 2_000, f"batch-32 decode took {decode:.0f} us on the host"
    assert chunk < 50_000, f"an 8192-token chunk took {chunk:.0f} us on the host"


# --------------------------------------------------------------------------------------------------------------
# 2. the layer


def runner_ple_argument(layer, ids, qsl, ctx):
    """What reaches the PLE layer's ``ngram_context`` argument: the context, or with the flag on the ready ids.

    The runner computes the ids from the batch's token ids as it holds them (``-1`` padding included); the model
    forward moves them into the ``ngram_context`` slot (``_MtpTargetOutputs.forward``).
    """
    if int(gaudi_envs.VLLM_HPU_PLE_HOST_NGRAM_IDS) == 1:
        return host_ngram_ids(collect_ngram_host_tables(layer), ids, qsl, ctx)
    return ctx


class CallCounter:
    """Counts ``compute_ngram_ids`` calls of one embedding (the device id computation)."""

    def __init__(self, monkeypatch, embedding) -> None:
        self.calls = 0
        original = embedding.compute_ngram_ids

        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(embedding, "compute_ngram_ids", counted)


def padded_bucket_scenario(vllm_config, layer):
    """Two prefill chunks of one prompt, a second prompt in a padded bucket, then five padded decode steps.

    Sequence A (2001 tokens, slot 1) is prefilled as 1701 of a 2048 bucket then 300 of a 384 bucket; sequence B
    (77 tokens, slot 2) in a 128 bucket; the decode batch has 4 rows (A, B and two padding rows on the padding
    state row). Returns ``(outputs, state, seq_a, hidden_a)``: every forward's output, the final state tensor,
    and sequence A with the hidden rows of its 2001 prompt tokens (for the unpadded oracle).
    """
    eos = _eos(vllm_config)
    gen = torch.Generator().manual_seed(91)
    vocab = int(vllm_config.model_config.hf_text_config.vocab_size)
    seq_a = random_sequence(gen, 2001 + 5, vocab, eos)
    seq_b = random_sequence(gen, 77 + 5, vocab, eos)
    batch = Batch([seq_a, seq_b], eos)
    layer.kv_cache[0].zero_()
    outputs = []

    def run(ids, qsl, ctx, md):
        hidden = torch.randn(ids.numel(), CHANNELS, generator=gen)
        arg = runner_ple_argument(layer, ids, qsl, ctx)
        with torch.no_grad(), set_forward_context(md, vllm_config):
            outputs.append(layer(hidden, ids.clamp(min=0), qsl, arg))
        return hidden

    def prompt(row, start, count, bucket, slot):
        ids, qsl, ctx = batch.step([row], [start], [count], bucket_rows=1, bucket_seq=bucket, pad=-1)
        md = _md(True, [slot], has_initial=[int(start > 0)], padding=bf16_mask([count], bucket))
        md.query_start_loc_p = cumulative([count])
        return run(ids, qsl, ctx, md)

    hidden_a = [prompt(0, 0, 1701, 2048, 1)[:1701], prompt(0, 1701, 300, 384, 1)[:300]]
    prompt(1, 0, 77, 128, 2)
    for step in range(5):
        ids, qsl, ctx = batch.step([0, 1], [2001 + step, 77 + step], [1, 1], bucket_rows=4, bucket_seq=1, pad=0)
        run(ids, qsl, ctx, _md(False, [1, 2, PAD_ROW, PAD_ROW]))
    return outputs, layer.kv_cache[0].clone(), seq_a, torch.cat(hidden_a)


def test_layer_output_and_state_identical_with_the_flag_0_and_1(ple_layer, monkeypatch):
    vllm_config, layer = ple_layer
    counter = CallCounter(monkeypatch, layer.ple_embedding)

    monkeypatch.setenv(FLAG, "0")
    device_outputs, device_state, seq_a, hidden_a = padded_bucket_scenario(vllm_config, layer)
    assert counter.calls == len(device_outputs) == 8, "flag 0 must compute the ids on the device in every forward"

    counter.calls = 0
    monkeypatch.setenv(FLAG, "1")
    host_outputs, host_state, _, _ = padded_bucket_scenario(vllm_config, layer)
    assert counter.calls == 0, f"flag 1 still ran compute_ngram_ids {counter.calls} times"

    for index, (got, want) in enumerate(zip(host_outputs, device_outputs)):
        assert torch.equal(got, want), f"forward {index}: the PLE output differs between the flag 1 and 0"
    assert torch.equal(host_state, device_state), "the saved short-conv state differs between the flag 1 and 0"

    # And both are the layer's unpadded, unchunked oracle on the first 2001 tokens of sequence A.
    ref_out, _ = _oracle(layer, seq_a[:2001], hidden_a, eos=_eos(vllm_config))
    torch.testing.assert_close(torch.cat((host_outputs[0][:1701], host_outputs[1][:300])), ref_out, **TOL)


def test_verification_step_identical_with_the_flag_0_and_1(ple_layer, monkeypatch):
    """A prefill, then an all-accepted ``T = 4`` verification step and a transition on the speculative state."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    counter = CallCounter(monkeypatch, layer.ple_embedding)

    def scenario():
        layer.kv_cache = (torch.zeros(SLOTS, SPEC_STATE, CHANNELS), )
        gen = torch.Generator().manual_seed(101)
        seq = torch.randint(0, 1000, (15, ), generator=gen, dtype=torch.int64)
        seq[9], seq[12] = eos, eos
        batch = Batch([seq], eos)
        outs = []
        for start, count, md in ((0, 10, _md(True, [2], has_initial=[0])), (10, 4, _spec_md([2], [1], [0, 4])),
                                 (14, 1, _spec_md([2], [4], [0, 1]))):
            ids, qsl, ctx = batch.step([0], [start], [count], bucket_rows=1, bucket_seq=count, pad=0)
            hidden = torch.randn(count, CHANNELS, generator=gen)
            with torch.no_grad(), set_forward_context(md, vllm_config):
                outs.append(layer(hidden, ids, qsl, runner_ple_argument(layer, ids, qsl, ctx)))
        return outs, layer.kv_cache[0].clone()

    monkeypatch.setenv(FLAG, "0")
    device_outs, device_state = scenario()
    assert counter.calls == 3
    counter.calls = 0
    monkeypatch.setenv(FLAG, "1")
    host_outs, host_state = scenario()
    assert counter.calls == 0
    for got, want in zip(host_outs, device_outs):
        assert torch.equal(got, want)
    assert torch.equal(host_state, device_state)


def test_the_layer_takes_the_row_of_its_dense_id(ple_layer, monkeypatch):
    """With several PLE layers the ids tensor has one row per layer: a layer reads row ``ple_dense_layer_id``."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(111)
    ids = torch.randint(0, 1000, (12, ), dtype=torch.int32)
    hidden = torch.randn(12, CHANNELS)
    qsl = torch.tensor([0, 12], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    ready = host_ngram_ids(collect_ngram_host_tables(layer), ids, qsl, ctx)
    with torch.no_grad(), set_forward_context(None, vllm_config):
        want = layer(hidden, ids, qsl, ctx)
        monkeypatch.setattr(layer, "ple_dense_layer_id", 1)
        other_layers_row = torch.zeros_like(ready)
        got = layer(hidden, ids, qsl, torch.cat((other_layers_row, ready)))
        swapped = layer(hidden, ids, qsl, torch.cat((ready, other_layers_row)))
    assert torch.equal(got, want)
    assert not torch.equal(swapped, want)


def test_ready_ids_of_the_wrong_shape_are_refused(ple_layer):
    """The ready ids are recognised by their rank; a tensor that is not ``[ple_layers, tokens, heads]`` for this
    call is an error, never a silently wrong embedding row."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    ids = torch.randint(0, 1000, (6, ), dtype=torch.int32)
    qsl = torch.tensor([0, 6], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    ready = host_ngram_ids(collect_ngram_host_tables(layer), ids, qsl, ctx)
    hidden = torch.randn(6, CHANNELS)
    for bad in (ready[:, :5], ready[:, :, :8], ready[:0]):  # a token short, half the heads, no row for this layer
        with pytest.raises(ValueError, match="n-gram ids"), torch.no_grad(), set_forward_context(None, vllm_config):
            layer(hidden, ids, qsl, bad)


# --------------------------------------------------------------------------------------------------------------
# 3. the op count


class OpRecorder(TorchDispatchMode):
    """Every aten op one forward dispatches, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))

    @staticmethod
    def is_view(func) -> bool:
        return any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)

    def summary(self) -> tuple[int, int, str]:
        """``(ops, non-view ops, digest of the op-name sequence)``."""
        names = [str(func) for func in self.ops]
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
        return len(names), sum(not self.is_view(func) for func in self.ops), digest

    def names(self) -> set[str]:
        return {str(func) for func in self.ops}


# One PLE forward with the ids computed on the device, recorded before the host path existed
# (HpuQwen4ExpPLELayer of the tiny fixture, CPU torch): (ops, non-view ops, sha256 of the op names)[:16].
DEVICE_PATH_OPS = {
    "decode_b1": (448, 299, "42876a61b032ab3a"),
    "decode_b32": (464, 311, "00585f5ece7b0cf3"),
    "prompt_512": (491, 330, "5170dd0ff10bcbf8"),
}
SHAPES = {"decode_b1": (1, 1, False), "decode_b32": (32, 1, False), "prompt_512": (1, 512, True)}
# The ops only the id computation uses (the gate and the short conv have none of them).
ID_COMPUTATION_OPS = ("aten.searchsorted.Tensor", "aten.bitwise_and.Scalar", "aten.bitwise_right_shift.Tensor_Scalar",
                      "aten.bitwise_xor.Tensor", "aten.remainder.Tensor", "aten.masked_fill.Scalar",
                      "aten.maximum.default")


def record_forward(vllm_config, layer, shape: str, host: bool) -> OpRecorder:
    rows, seq, is_prompt = SHAPES[shape]
    eos = _eos(vllm_config)
    torch.manual_seed(1)
    tokens = rows * seq
    ids = torch.randint(0, 1000, (tokens, ), dtype=torch.int32)
    hidden = torch.randn(tokens, CHANNELS)
    qsl = torch.arange(rows + 1, dtype=torch.int32) * seq
    ctx = torch.full((rows, 2), eos, dtype=torch.int32)
    md = _md(is_prompt, [1 + (i % 3) for i in range(rows)], has_initial=[0] * rows if is_prompt else None)
    if is_prompt:
        md.query_start_loc_p = qsl.clone()
    arg = host_ngram_ids(collect_ngram_host_tables(layer), ids, qsl, ctx) if host else ctx
    with torch.no_grad(), set_forward_context(md, vllm_config), OpRecorder() as recorder:
        layer(hidden, ids, qsl, arg)
    return recorder


@pytest.mark.parametrize("shape", list(SHAPES))
def test_device_path_op_sequence_is_unchanged(ple_layer, shape):
    """Flag off: the PLE forward dispatches exactly the ops it dispatched before the host path was added."""
    vllm_config, layer = ple_layer
    got = record_forward(vllm_config, layer, shape, host=False).summary()
    assert got == DEVICE_PATH_OPS[shape], (f"the device-path op sequence of {shape} changed: (ops, non-view ops, "
                                           f"digest) is {got}, was {DEVICE_PATH_OPS[shape]}")


@pytest.mark.parametrize("shape", list(SHAPES))
def test_ready_ids_remove_the_id_computation_ops(ple_layer, shape):
    vllm_config, layer = ple_layer
    device = record_forward(vllm_config, layer, shape, host=False)
    host = record_forward(vllm_config, layer, shape, host=True)
    device_ops, device_non_view, _ = device.summary()
    host_ops, host_non_view, _ = host.summary()
    print(f"\nPLE layer {shape}: {device_ops} ops ({device_non_view} non-view) on the device path, "
          f"{host_ops} ({host_non_view} non-view) with ready ids")
    left = sorted(name for name in ID_COMPUTATION_OPS if name in host.names())
    assert not left, f"{shape}: the forward with ready ids still dispatches {left}"
    for name in ID_COMPUTATION_OPS:
        assert name in device.names(), f"{name} is not an op of the device path any more: fix ID_COMPUTATION_OPS"
    assert host_ops <= device_ops - 300, f"{shape}: {host_ops} ops with ready ids, {device_ops} without"
    assert host_non_view <= device_non_view - 200
