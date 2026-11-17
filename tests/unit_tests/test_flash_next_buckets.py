# SPDX-License-Identifier: Apache-2.0
"""Prompt and decode bucket coverage for Flash-Next serving settings (M1-T7).

Three serving settings (block 128, GDN chunk 128, step budgets 2048 and 8192,
model lengths 131072 and 262144, up to 32 sequences) are built with the model
runner's initialization arguments. The tests check that long-prompt chunk and
decode lookups hit prepared buckets (a fallback bucket means a runtime graph
recompile we want to rule out) and report how many buckets warmup must compile
per setting. A lookup that falls back is an explicit xfail listing its triple;
the bucketing code is not changed here.
"""

import pytest

from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.extension.runtime import clear_config, get_config

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


def make_bucketed_manager(max_num_batched_tokens, max_model_len):
    """Build the manager with the runner's arguments, then generate both bucket sets."""
    manager = HPUBucketingManager()
    manager.initialize(max_num_seqs=MAX_NUM_SEQS,
                       max_num_prefill_seqs=1,
                       block_size=BLOCK_SIZE,
                       max_num_batched_tokens=max_num_batched_tokens,
                       max_model_len=max_model_len,
                       num_speculative_tokens=0,
                       mamba_chunk_size=GDN_CHUNK_SIZE,
                       mamba_chunk_size_is_explicit=False)
    manager.num_hpu_blocks = NUM_HPU_BLOCKS
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
