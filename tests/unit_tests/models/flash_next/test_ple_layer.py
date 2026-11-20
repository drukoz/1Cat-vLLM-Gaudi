# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HPU Flash-Next PLE layer (M0, T4b).

Builds ``HpuQwen4ExpPLELayer`` from the tiny fixture with a world-1 gloo group
and a vLLM config context (no HPU required) and checks that the layer matches
the T4a torch oracle (``ple_forward_torch`` / ``ple_gate_torch``) across

1. the batched short conv against the per-row reference,
2. a single-row prefill (output and slot state),
3. a padded two-row prefill (padding ignored, per-row state),
4. a chunked prefill followed by decode (state carried in the slot),
5. a decode batch whose padding row (index ``-1``) is not read and writes only the state tensor's last row
   (the padding row; one fixed-shape write, see ``test_ple_padded_bucket.py``),
6. a profile run (no metadata: gated output, cache untouched),
7. the ``Qwen4ExpPLELayer`` and ``Qwen4ExpNGramEmbedding`` seam rebinding in
   ``apply_flash_next_patches``, and the host-multiplier n-gram embedding
   (``HpuQwen4ExpNGramEmbedding``): its int32 ids equal the engine's int64 ids
   before and after a checkpoint multipliers load.

M3 adds the speculative path: verification and transition steps run from the
state lattice with ``num_accepted_tokens`` rollback and must match the
straight single-token chain (and stop matching it when the committed-history
update is mutated).
"""

import os
import shutil
import tempfile
import types
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402

import vllm_gaudi.models.qwen4_exp as flash_next  # noqa: E402
import vllm_gaudi.models.qwen4_exp.ple_layer as ple_layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_layer import (  # noqa: E402
    HpuQwen4ExpNGramEmbedding,
    HpuQwen4ExpPLELayer,
)
from vllm_gaudi.models.qwen4_exp.ple_ops import (  # noqa: E402
    ple_forward_torch, ple_gate_torch, ple_short_conv_batched_torch, ple_short_conv_spec_torch,
    ple_short_conv_torch,
)

FIXTURE = Path(__file__).resolve().parent / "tiny_config.json"
PREFIX = "model.layers.1.ple"
SLOTS = 5
STATE_LEN = 9
SPEC_STATE = 12  # (kernel_size - 1) * dilation + T - 1 for a T = 4 MTP step
CHANNELS = 512
DILATION = 3
TOL = dict(atol=1e-5, rtol=1e-5)


def _engine_config(model_dir):
    args = EngineArgs(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_seqs=4,
        enforce_eager=True,
        language_model_only=True,
        load_format="dummy",
    )
    return args.create_engine_config()


@pytest.fixture()
def ple_layer(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "config.json")
    vllm_config = _engine_config(tmp_path)
    # The n-gram replacement is built through the ple_layer module seam, so the
    # PLE layer built below must see the patched engine name.
    flash_next.apply_flash_next_patches()
    init_file = tempfile.mkstemp(dir="/tmp")[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{init_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            layer = HpuQwen4ExpPLELayer(
                vllm_config.model_config.hf_text_config,
                vllm_config=vllm_config,
                layer_idx=1,
                ple_dense_layer_id=0,
                prefix=PREFIX,
            )
        assert type(layer.ple_embedding) is HpuQwen4ExpNGramEmbedding
        layer.float()
        torch.manual_seed(0)
        for param in layer.parameters():
            if param.is_floating_point():
                param.data.normal_(0, 0.5)
        # The CPU unquantized linear packs its weights here; the refill above
        # must happen first (the dispatch captures the weight).
        layer.kv_proj.quant_method.process_weights_after_loading(layer.kv_proj)
        layer.kv_cache = (torch.zeros(SLOTS, STATE_LEN, CHANNELS),)
        yield vllm_config, layer
    destroy_model_parallel()
    destroy_distributed_environment()


def _eos(vllm_config):
    eos = vllm_config.model_config.hf_text_config.eos_token_id
    return int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos)


def _oracle(layer, ids, hidden, state=None, eos=0):
    """T4a oracle: one unpadded sequence [n] int64 with hidden [n, 512]."""
    qsl = torch.tensor([0, ids.numel()], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    emb = layer.ple_embedding.ngram_embedding(layer.ple_embedding.compute_ngram_ids(ids, qsl, ctx)).flatten(-2)
    kv, _ = layer.kv_proj(emb)
    key, value = kv.split(layer.kv_proj.output_sizes, dim=-1)
    state = torch.zeros(STATE_LEN, CHANNELS) if state is None else state
    return ple_forward_torch(key, value, hidden, layer.norm_key.weight, layer.norm_query.weight,
                             layer.norm_conv.weight, layer.norm_key.eps, layer.conv1d.weight, state, DILATION)


def _md(is_prompt, load, store=None, has_initial=None, padding=None):
    return types.SimpleNamespace(
        is_prompt=is_prompt,
        load_indices_tensor=torch.tensor(load, dtype=torch.int32),
        store_indices_tensor=None if store is None else torch.tensor(store, dtype=torch.int32),
        has_initial_states_p=None if has_initial is None else torch.tensor(has_initial, dtype=torch.int32),
        padding_mask_flat=padding,
        num_accepted_tokens=None,
    )


def test_batched_conv_matches_per_row():
    torch.manual_seed(7)
    b_size, seq, channels, kernel, dilation = 3, 12, 7, 4, 3
    state_len = (kernel - 1) * dilation
    normed = torch.randn(b_size, seq, channels)
    weight = torch.randn(channels, 1, kernel)
    state = torch.randn(b_size, state_len, channels)
    lengths = torch.tensor([12, 5, 1], dtype=torch.long)

    out, new_state = ple_short_conv_batched_torch(normed, weight, state, dilation, lengths)
    for b in range(b_size):
        n = int(lengths[b])
        ref_out, ref_state = ple_short_conv_torch(normed[b, :n], weight, state[b], dilation)
        torch.testing.assert_close(out[b, :n], ref_out, **TOL)
        torch.testing.assert_close(new_state[b], ref_state, **TOL)


def test_prefill_single(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(3)
    ids = torch.randint(0, 1000, (10,), dtype=torch.int64)
    hidden = torch.randn(10, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden, eos=eos)

    md = _md(True, [1], has_initial=[0])
    qsl = torch.tensor([0, 10], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        out = layer(hidden, ids, qsl, ctx)

    torch.testing.assert_close(out, ref_out, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][1], ref_state, **TOL)


def test_prefill_padded_batch(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(4)
    ids_a = torch.randint(0, 1000, (12,), dtype=torch.int64)
    ids_b = torch.randint(0, 1000, (5,), dtype=torch.int64)
    ids = torch.zeros(24, dtype=torch.int64)
    ids[:12] = ids_a
    ids[12:17] = ids_b
    hidden = torch.randn(24, CHANNELS)
    padding = torch.ones(24, 1)
    padding[17:, 0] = 0.0

    md = _md(True, [2, 3], has_initial=[0, 0], padding=padding)
    qsl = torch.arange(3, dtype=torch.int32) * 12
    ctx = torch.full((2, 2), eos, dtype=torch.int32)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        out = layer(hidden, ids, qsl, ctx)

    out = out.view(2, 12, CHANNELS)
    ref_a, state_a = _oracle(layer, ids_a, hidden[:12], eos=eos)
    ref_b, state_b = _oracle(layer, ids_b, hidden[12:17], eos=eos)
    torch.testing.assert_close(out[0], ref_a, **TOL)
    torch.testing.assert_close(out[1, :5], ref_b, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][2], state_a, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][3], state_b, **TOL)


def test_chunked_prefill_then_decode(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(5)
    ids = torch.randint(0, 1000, (16,), dtype=torch.int64)
    hidden = torch.randn(16, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden, eos=eos)

    outputs = []

    def run(chunk, chunk_hidden, qsl, ctx, md):
        with torch.no_grad(), set_forward_context(md, vllm_config):
            outputs.append(layer(chunk_hidden, chunk, qsl, ctx))

    run(ids[:7], hidden[:7], torch.tensor([0, 7], dtype=torch.int32),
        torch.full((1, 2), eos, dtype=torch.int32), _md(True, [4], has_initial=[0]))
    run(ids[7:13], hidden[7:13], torch.tensor([0, 6], dtype=torch.int32),
        ids[5:7].unsqueeze(0).to(torch.int32), _md(True, [4], has_initial=[1]))
    for pos in (13, 14, 15):
        run(ids[pos:pos + 1], hidden[pos:pos + 1], torch.tensor([0, 1], dtype=torch.int32),
            ids[pos - 2:pos].unsqueeze(0).to(torch.int32), _md(False, [4]))

    torch.testing.assert_close(torch.cat(outputs), ref_out, **TOL)
    torch.testing.assert_close(layer.kv_cache[0][4], ref_state, **TOL)


def test_padding_row_not_written(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(6)
    layer.kv_cache[0].copy_(torch.randn_like(layer.kv_cache[0]))
    before = layer.kv_cache[0].clone()

    ids = torch.randint(0, 1000, (2,), dtype=torch.int64)
    hidden = torch.randn(2, CHANNELS)
    md = _md(False, [1, -1])
    qsl = torch.arange(3, dtype=torch.int32)
    ctx = torch.full((2, 2), eos, dtype=torch.int32)
    with torch.no_grad(), set_forward_context(md, vllm_config):
        layer(hidden, ids, qsl, ctx)

    assert not torch.equal(layer.kv_cache[0][1], before[1])
    # The write is one fixed-shape index_copy_ (HPU graphs): the invalid row goes to the state tensor's
    # last row, the padding row of every runner layout (slot 4 here), and to no request's slot.
    for slot in (0, 2, 3):
        assert torch.equal(layer.kv_cache[0][slot], before[slot])


def test_profile_metadata_none(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(7)
    ids = torch.randint(0, 1000, (6,), dtype=torch.int64)
    hidden = torch.randn(6, CHANNELS)
    cache_before = layer.kv_cache[0].clone()

    qsl = torch.tensor([0, 6], dtype=torch.int32)
    ctx = torch.full((1, 2), eos, dtype=torch.int32)
    emb = layer.ple_embedding.ngram_embedding(layer.ple_embedding.compute_ngram_ids(ids, qsl, ctx)).flatten(-2)
    kv, _ = layer.kv_proj(emb)
    key, value = kv.split(layer.kv_proj.output_sizes, dim=-1)
    gated, _ = ple_gate_torch(key, value, hidden, layer.norm_key.weight, layer.norm_query.weight,
                              layer.norm_conv.weight, layer.norm_key.eps)

    with torch.no_grad(), set_forward_context(None, vllm_config):
        out = layer(hidden, ids, qsl, ctx)

    torch.testing.assert_close(out, gated)
    assert torch.equal(layer.kv_cache[0], cache_before)


def test_seam_rebound():
    import vllm.models.qwen4_exp.nvidia.model as upstream_model
    import vllm.models.qwen4_exp.nvidia.ple_layer as upstream_ple

    import vllm_gaudi.models.qwen4_exp as flash_next
    flash_next.apply_flash_next_patches()
    assert upstream_model.Qwen4ExpPLELayer is HpuQwen4ExpPLELayer
    assert upstream_ple.Qwen4ExpNGramEmbedding is HpuQwen4ExpNGramEmbedding


def _engine_embedding(vllm_config):
    """The engine's embedding built with the fixture's config, in the fixture's context."""
    config = vllm_config.model_config.hf_text_config
    engine_cls = HpuQwen4ExpNGramEmbedding.__bases__[0]
    with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
        return engine_cls(
            vllm_config.model_config.hf_text_config,
            int(config.ple_embed_dim),
            ple_dense_layer_id=0,
            prefix=PREFIX + ".ple_embedding",
            layer_name=PREFIX,
        )


def _ngram_inputs(vllm_config):
    """Three requests of lengths 1, 5 and 40 plus a [3, 2] n-gram context."""
    vocab_size = int(vllm_config.model_config.hf_text_config.vocab_size)
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab_size, (46,), generator=gen, dtype=torch.int32)
    ids[0], ids[1], ids[6] = 0, vocab_size - 1, _eos(vllm_config)
    qsl = torch.tensor([0, 1, 6, 46], dtype=torch.int32)
    ctx = torch.randint(0, vocab_size, (3, 2), generator=gen, dtype=torch.int32)
    return ids, qsl, ctx


def test_ngram_ids_match_engine(ple_layer):
    vllm_config, layer = ple_layer
    engine = _engine_embedding(vllm_config)
    ids, qsl, ctx = _ngram_inputs(vllm_config)
    ours = layer.ple_embedding.compute_ngram_ids(ids, qsl, ctx)
    assert torch.equal(ours, engine.compute_ngram_ids(ids, qsl, ctx))


def test_ngram_ids_after_load(ple_layer):
    vllm_config, layer = ple_layer
    engine = _engine_embedding(vllm_config)
    ids, qsl, ctx = _ngram_inputs(vllm_config)
    before = layer.ple_embedding.compute_ngram_ids(ids, qsl, ctx)

    # The tiny checkpoint's multipliers: above int32, and not the seed-1234 ones.
    table = torch.tensor([8052911324071, 23703573157769, 20109073645365])
    assert layer.ple_embedding.load_weights([("layer_multipliers", table)]) == {"layer_multipliers"}
    assert engine.load_weights([("layer_multipliers", table)]) == {"layer_multipliers"}

    ours = layer.ple_embedding.compute_ngram_ids(ids, qsl, ctx)
    assert torch.equal(ours, engine.compute_ngram_ids(ids, qsl, ctx))
    assert not torch.equal(ours, before)
    assert torch.equal(layer.ple_embedding.layer_multipliers, torch.zeros(3, dtype=torch.long))


def test_bad_multiplier_rejected(ple_layer):
    _, layer = ple_layer
    with pytest.raises(ValueError):
        layer.ple_embedding.load_weights([("layer_multipliers", torch.tensor([1, 2 ** 62, 3]))])


def _spec_md(load, accepted, qsl_p):
    """MTP verification/transition metadata: 1-D lattice slots plus acceptance."""
    return types.SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=torch.tensor(load, dtype=torch.int32),
        store_indices_tensor=None,
        has_initial_states_p=torch.ones(len(load), dtype=torch.int32),
        padding_mask_flat=None,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        query_start_loc_p=torch.tensor(qsl_p, dtype=torch.int32),
    )


def _spec_prefill_and_decode(vllm_config, layer, ids, hidden, eos, ref_state, decode_slot, spec_slot):
    """Prefill into both slots, decode tokens 10..14 into ``decode_slot``.

    Returns ``(run, decode_outs, decode_state)``.
    """
    def run(step_ids, step_hidden, qsl, ctx, md):
        with torch.no_grad(), set_forward_context(md, vllm_config):
            return layer(step_hidden, step_ids, qsl, ctx)

    for slot in (decode_slot, spec_slot):
        run(ids[:10], hidden[:10], torch.tensor([0, 10], dtype=torch.int32),
            torch.full((1, 2), eos, dtype=torch.int32), _md(True, [slot], has_initial=[0]))
        # The non-spec path with a taller state reads and writes rows [0, state_len).
        assert torch.equal(layer.kv_cache[0][slot, STATE_LEN:], torch.zeros(SPEC_STATE - STATE_LEN, CHANNELS))

    decode_outs = []
    for pos in range(10, 15):
        decode_outs.append(
            run(ids[pos:pos + 1], hidden[pos:pos + 1], torch.tensor([0, 1], dtype=torch.int32),
                ids[pos - 2:pos].unsqueeze(0).to(torch.int32), _md(False, [decode_slot])))
    decode_state = layer.kv_cache[0][decode_slot, :STATE_LEN].clone()
    torch.testing.assert_close(decode_state, ref_state, **TOL)
    return run, decode_outs, decode_state


def test_spec_verify_and_transition(ple_layer):
    """An all-accepted T=4 verification plus a T=1 transition match the single-token chain."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    layer.kv_cache = (torch.zeros(SLOTS, SPEC_STATE, CHANNELS),)
    torch.manual_seed(8)
    ids = torch.randint(0, 1000, (15,), dtype=torch.int64)
    hidden = torch.randn(15, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden, eos=eos)

    run, decode_outs, decode_state = _spec_prefill_and_decode(vllm_config, layer, ids, hidden, eos, ref_state, 1, 2)

    # Verification: the seed commit of the prefill leaves num_accepted = 1.
    spec_out = run(ids[10:14], hidden[10:14], torch.tensor([0, 4], dtype=torch.int32),
                   ids[8:10].unsqueeze(0).to(torch.int32), _spec_md([2], [1], [0, 4]))
    torch.testing.assert_close(spec_out, torch.cat(decode_outs[:4]), **TOL)
    # All four rows committed: the next step reads its window at num_accepted - 1,
    # and the window is the committed history of the 14-token straight sequence.
    ref_state_14 = _oracle(layer, ids[:14], hidden[:14], eos=eos)[1]
    torch.testing.assert_close(layer.kv_cache[0][2, 3:3 + STATE_LEN], ref_state_14, **TOL)

    # Transition: the next single token reads the committed rows of the verify step.
    trans = run(ids[14:15], hidden[14:15], torch.tensor([0, 1], dtype=torch.int32),
                ids[12:14].unsqueeze(0).to(torch.int32), _spec_md([2], [4], [0, 1]))
    torch.testing.assert_close(trans, decode_outs[4], **TOL)
    # The transition commits one token, so the next window read is at num_accepted - 1 = 0.
    torch.testing.assert_close(layer.kv_cache[0][2, 0:STATE_LEN], decode_state, **TOL)


def test_spec_rollback_layer(ple_layer):
    """Accepting 2 of 4 rows: the transitions match the chain that never saw the rejected rows."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    layer.kv_cache = (torch.zeros(SLOTS, SPEC_STATE, CHANNELS),)
    torch.manual_seed(9)
    ids = torch.randint(0, 1000, (14,), dtype=torch.int64)
    hidden = torch.randn(14, CHANNELS)
    ref_out, ref_state = _oracle(layer, ids, hidden, eos=eos)

    def run(step_ids, step_hidden, qsl, ctx, md):
        with torch.no_grad(), set_forward_context(md, vllm_config):
            return layer(step_hidden, step_ids, qsl, ctx)

    for slot in (1, 2):
        run(ids[:10], hidden[:10], torch.tensor([0, 10], dtype=torch.int32),
            torch.full((1, 2), eos, dtype=torch.int32), _md(True, [slot], has_initial=[0]))

    # Verify 4 rows, accept the first 2; the straight path continues with ids 12, 13.
    spec_out = run(ids[10:14], hidden[10:14], torch.tensor([0, 4], dtype=torch.int32),
                   ids[8:10].unsqueeze(0).to(torch.int32), _spec_md([2], [1], [0, 4]))
    chain_outs = []
    for pos in (10, 11, 12, 13):
        chain_outs.append(
            run(ids[pos:pos + 1], hidden[pos:pos + 1], torch.tensor([0, 1], dtype=torch.int32),
                ids[pos - 2:pos].unsqueeze(0).to(torch.int32), _md(False, [1])))
    torch.testing.assert_close(spec_out[:2], torch.cat(chain_outs[:2]), **TOL)

    trans_a = run(ids[12:13], hidden[12:13], torch.tensor([0, 1], dtype=torch.int32),
                  ids[10:12].unsqueeze(0).to(torch.int32), _spec_md([2], [2], [0, 1]))
    trans_b = run(ids[13:14], hidden[13:14], torch.tensor([0, 1], dtype=torch.int32),
                  ids[11:13].unsqueeze(0).to(torch.int32), _spec_md([2], [1], [0, 1]))
    torch.testing.assert_close(trans_a, chain_outs[2], **TOL)
    torch.testing.assert_close(trans_b, chain_outs[3], **TOL)
    ref_full = _oracle(layer, ids, hidden, eos=eos)[1]
    torch.testing.assert_close(layer.kv_cache[0][2, 0:STATE_LEN], ref_full, **TOL)


def _mutated_spec_kernel(normed, conv_weight, state, num_accepted, query_lens, dilation):
    """Same outputs, but keeps the full ``history`` (not ``history[1:]``) in ``new_state``."""
    out, _ = ple_short_conv_spec_torch(normed, conv_weight, state, num_accepted, query_lens, dilation)
    kernel = conv_weight.reshape(conv_weight.shape[0], -1)
    length = (kernel.shape[-1] - 1) * dilation
    offset = num_accepted.reshape(-1).long().clamp(min=1, max=state.shape[1] - length + 1) - 1
    window = offset.view(-1, 1) + torch.arange(length).view(1, length)
    history = state.gather(1, window.unsqueeze(-1).expand(-1, length, state.shape[-1]))
    mask = (torch.arange(normed.shape[1]).view(1, -1) < query_lens.reshape(-1, 1).clamp(min=0, max=normed.shape[1]))
    sequence = torch.cat([history, normed * mask.unsqueeze(-1).to(normed.dtype)], dim=1)
    pad = state.shape[1] - sequence.shape[1]
    if pad > 0:
        zeros = sequence.new_zeros(sequence.shape[0], pad, sequence.shape[2])
        sequence = torch.cat([sequence, zeros], dim=1)
    new_state = sequence[:, :state.shape[1]]
    new_state = torch.where((query_lens.reshape(-1, 1, 1) > 0), new_state, state)
    return out, new_state


def test_spec_committed_history_is_read(ple_layer, monkeypatch):
    """Keeping the full history in new_state breaks the transition: the layer reads committed rows."""
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    layer.kv_cache = (torch.zeros(SLOTS, SPEC_STATE, CHANNELS),)
    torch.manual_seed(11)
    ids = torch.randint(0, 1000, (13,), dtype=torch.int64)
    hidden = torch.randn(13, CHANNELS)

    monkeypatch.setattr(ple_layer_module, "ple_short_conv_spec_torch", _mutated_spec_kernel)

    def run(step_ids, step_hidden, qsl, ctx, md):
        with torch.no_grad(), set_forward_context(md, vllm_config):
            return layer(step_hidden, step_ids, qsl, ctx)

    run(ids[:10], hidden[:10], torch.tensor([0, 10], dtype=torch.int32),
        torch.full((1, 2), eos, dtype=torch.int32), _md(True, [1], has_initial=[0]))
    chain = []
    for pos in (10, 11, 12):
        chain.append(
            run(ids[pos:pos + 1], hidden[pos:pos + 1], torch.tensor([0, 1], dtype=torch.int32),
                ids[pos - 2:pos].unsqueeze(0).to(torch.int32), _md(False, [1])))

    run(ids[:10], hidden[:10], torch.tensor([0, 10], dtype=torch.int32),
        torch.full((1, 2), eos, dtype=torch.int32), _md(True, [2], has_initial=[0]))
    run(ids[10:14], hidden[10:14], torch.tensor([0, 4], dtype=torch.int32),
        ids[8:10].unsqueeze(0).to(torch.int32), _spec_md([2], [1], [0, 4]))

    # The mutated window is one row short of the committed history, so the
    # transition over token 12 must no longer match the straight chain.
    with pytest.raises(AssertionError):
        trans = run(ids[12:13], hidden[12:13], torch.tensor([0, 1], dtype=torch.int32),
                    ids[10:12].unsqueeze(0).to(torch.int32), _spec_md([2], [2], [0, 1]))
        torch.testing.assert_close(trans, chain[2], **TOL)
