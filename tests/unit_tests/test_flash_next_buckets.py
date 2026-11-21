# SPDX-License-Identifier: Apache-2.0
"""Prompt and decode bucket coverage for Flash-Next serving settings (M1-T7).

Three serving settings (block 128, GDN chunk 128, step budgets 2048 and 8192,
model lengths 131072 and 262144, up to 32 sequences) are built with the model
runner's initialization arguments. The tests check that long-prompt chunk and
decode lookups hit prepared buckets (a fallback bucket means a runtime graph
recompile we want to rule out) and report how many buckets warmup must compile
per setting. A lookup that falls back is an explicit xfail listing its triple;
the bucketing code is not changed here.

With speculation the manager adds, for every decode bucket ``(bs, 1, ctx)``, the buckets of
``bs * (1 + k)`` rows. ``test_spec_decode_buckets_at_every_size`` checks them for k = 3, 4 and 5
and that the number of graphs warmup must compile stays inside the bound.
"""

import pytest

from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.extension.runtime import clear_config, get_config

from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id

MAX_NUM_SEQS = 32
BLOCK_SIZE = 128
GDN_CHUNK_SIZE = 128
NUM_HPU_BLOCKS = 6000
MAX_COMPILED_BUCKETS = 600

SERVING_SETTINGS = [(2048, 131072), (8192, 131072), (8192, 262144)]


@pytest.fixture(autouse=True)
def default_config():
    clear_config()
    get_config()
    yield
    clear_config()


def make_bucketed_manager(max_num_batched_tokens, max_model_len, num_speculative_tokens=0,
                          max_num_seqs=MAX_NUM_SEQS, skip_decode_block_clamp=False):
    """Build the manager with the runner's arguments, then generate both bucket sets.

    ``skip_decode_block_clamp`` is what ``HPUModelRunner.warmup_model`` sets for a model with GDN layers:
    the decode buckets then reach the worst case of ``max_num_seqs`` full contexts.
    """
    manager = HPUBucketingManager()
    manager.initialize(max_num_seqs=max_num_seqs,
                       max_num_prefill_seqs=1,
                       block_size=BLOCK_SIZE,
                       max_num_batched_tokens=max_num_batched_tokens,
                       max_model_len=max_model_len,
                       num_speculative_tokens=num_speculative_tokens,
                       mamba_chunk_size=GDN_CHUNK_SIZE,
                       mamba_chunk_size_is_explicit=False)
    manager.num_hpu_blocks = NUM_HPU_BLOCKS
    manager.skip_decode_block_clamp = skip_decode_block_clamp
    manager.generate_prompt_buckets()
    manager.generate_decode_buckets()
    return manager


def xfail_fallbacks(fallbacks):
    triples = ", ".join(str(triple) for triple in dict.fromkeys(fallbacks))
    pytest.xfail(f"fallback bucket: {triples}")


@pytest.mark.parametrize(("max_num_batched_tokens", "max_model_len"), SERVING_SETTINGS)
def test_prompt_chunk_lookups_hit_prepared_buckets(max_num_batched_tokens, max_model_len):
    manager = make_bucketed_manager(max_num_batched_tokens, max_model_len)
    fallbacks = []
    for query in (128, 2048, max_num_batched_tokens):
        for ctx_blocks in (0, 64, 512, 1023, (max_model_len - query) // BLOCK_SIZE):
            if ctx_blocks * BLOCK_SIZE + query > max_model_len:
                # Longer than the model can hold: the scheduler never builds this chunk.
                continue
            buckets_before = len(manager.prompt_buckets)
            bucket = manager.find_prompt_bucket(1, query, ctx_blocks)
            if len(manager.prompt_buckets) != buckets_before:
                fallbacks.append((1, query, ctx_blocks))
                continue
            assert bucket[0] == 1, f"bucket {bucket} for (1, {query}, {ctx_blocks}) has bs != 1"
            if query >= 2048:
                assert bucket[1] == query, f"query padded past budget: {bucket} for (1, {query}, {ctx_blocks})"
            assert bucket[2] >= ctx_blocks, f"context under-covered: {bucket} for (1, {query}, {ctx_blocks})"
            padding_ratio = (bucket[2] + 1) / (ctx_blocks + 1)
            assert padding_ratio <= 2, (f"context padding ratio {padding_ratio:.2f} > 2: "
                                        f"{bucket} for (1, {query}, {ctx_blocks})")
    if fallbacks:
        xfail_fallbacks(fallbacks)


@pytest.mark.parametrize(("max_num_batched_tokens", "max_model_len"), SERVING_SETTINGS)
def test_decode_lookups_hit_prepared_buckets(max_num_batched_tokens, max_model_len):
    manager = make_bucketed_manager(max_num_batched_tokens, max_model_len)
    fallbacks = []
    for bs in (1, 16, 32):
        for num_blocks in (bs * 8, bs * 256, bs * 1024):
            buckets_before = len(manager.decode_buckets)
            bucket = manager.find_decode_bucket(bs, num_blocks)
            if len(manager.decode_buckets) != buckets_before:
                fallbacks.append((bs, 1, num_blocks))
                continue
            assert bucket[0] >= bs, f"decode bucket {bucket} below requested bs {bs}"
    if fallbacks:
        xfail_fallbacks(fallbacks)


@pytest.mark.parametrize(("max_num_batched_tokens", "max_model_len"), SERVING_SETTINGS)
def test_compiled_bucket_count_within_warmup_bound(max_num_batched_tokens, max_model_len):
    manager = make_bucketed_manager(max_num_batched_tokens, max_model_len)
    num_prompt = len(manager.prompt_buckets)
    num_decode = len(manager.decode_buckets)
    print(f"[flash-next buckets] max_num_batched_tokens={max_num_batched_tokens} "
          f"max_model_len={max_model_len}: prompt={num_prompt} decode={num_decode} "
          f"total={num_prompt + num_decode}")
    assert num_prompt + num_decode <= MAX_COMPILED_BUCKETS


@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size k the Flash-Next tests run at, after flash_next_spec_sizes' startup check."""
    require_startup(request.param, prefix_caching=True)
    return request.param


# Decode buckets of the 8192-token, 131072-token setting with k speculative tokens, for 32 and 16 sequences,
# as the runner builds them for a GDN model (119 and 93 buckets without speculation). Each of those gets up to
# two more with (1 + k) times the rows; how many coincide with another bucket depends on k, so k = 4 has the
# most. The prompt buckets (144) do not depend on k.
SPEC_DECODE_BUCKETS = {(32, 3): 274, (32, 4): 325, (32, 5): 308, (16, 3): 208, (16, 4): 250, (16, 5): 249}


@pytest.mark.parametrize("max_num_seqs", [32, 16])
def test_spec_decode_buckets_at_every_size(spec_k, max_num_seqs):
    """k speculative tokens: every decode bucket has its ``bs * (1 + k)``-row expansion, a verification batch
    pads to a prepared bucket of at most ``max_num_seqs`` requests, and the compiled count stays in the bound."""
    tokens = 1 + spec_k
    manager = make_bucketed_manager(8192, 131072, num_speculative_tokens=spec_k, max_num_seqs=max_num_seqs,
                                    skip_decode_block_clamp=True)
    seeds, buckets = manager.seed_decode_buckets, set(manager.decode_buckets)
    plain = make_bucketed_manager(8192, 131072, max_num_seqs=max_num_seqs, skip_decode_block_clamp=True)
    assert seeds == plain.decode_buckets
    max_ctx_per_seq = -(-131072 // BLOCK_SIZE)
    for bs, query, ctx in seeds:
        assert (bs, query, ctx) in buckets
        rows = bs * tokens
        wide = (rows, query, min(ctx * tokens, rows * max_ctx_per_seq))
        assert wide in buckets, f"no {tokens}x bucket of {(bs, ctx)}"
    assert max(bs for bs, _, _ in buckets) == max_num_seqs * tokens
    # The runner pads a verification batch by the seed buckets: never past max_num_seqs requests.
    for num_decodes in range(1, max_num_seqs + 1):
        for blocks_per_seq in (1, 8, 256, 1024):
            before = len(manager.decode_buckets)
            bucket = manager.find_decode_bucket(num_decodes, num_decodes * blocks_per_seq, True)
            assert len(manager.decode_buckets) == before, f"fallback bucket for {(num_decodes, blocks_per_seq)}"
            assert num_decodes <= bucket[0] <= max_num_seqs
    num_prompt, num_decode = len(manager.prompt_buckets), len(manager.decode_buckets)
    print(f"[flash-next buckets] k={spec_k} max_num_seqs={max_num_seqs}: prompt={num_prompt} decode={num_decode} "
          f"(seed {len(seeds)}) total={num_prompt + num_decode}")
    assert num_decode == SPEC_DECODE_BUCKETS[(max_num_seqs, spec_k)]
    assert num_prompt + num_decode <= MAX_COMPILED_BUCKETS
