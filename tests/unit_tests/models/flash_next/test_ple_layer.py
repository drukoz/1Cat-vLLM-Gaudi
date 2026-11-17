# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HPU Flash-Next PLE layer (M0, T4b).

Builds ``HpuQwen4ExpPLELayer`` from the tiny fixture with a world-1 gloo group
and a vLLM config context (no HPU required) and checks that the layer matches
the T4a torch oracle (``ple_forward_torch`` / ``ple_gate_torch``) across

1. the batched short conv against the per-row reference,
2. a single-row prefill (output and slot state),
3. a padded two-row prefill (padding ignored, per-row state),
4. a chunked prefill followed by decode (state carried in the slot),
5. a decode batch whose padding row (index ``-1``) is neither read nor written,
6. a profile run (no metadata: gated output, cache untouched),
7. the ``Qwen4ExpPLELayer`` seam rebinding in ``apply_flash_next_patches``.
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

from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpPLELayer  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_ops import (  # noqa: E402
    ple_forward_torch, ple_gate_torch, ple_short_conv_batched_torch, ple_short_conv_torch,
)

FIXTURE = Path(__file__).resolve().parent / "tiny_config.json"
PREFIX = "model.layers.1.ple"
SLOTS = 5
STATE_LEN = 9
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
    for slot in (0, 2, 3, 4):
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

    import vllm_gaudi.models.qwen4_exp as flash_next
    flash_next.apply_flash_next_patches()
    assert upstream_model.Qwen4ExpPLELayer is HpuQwen4ExpPLELayer
