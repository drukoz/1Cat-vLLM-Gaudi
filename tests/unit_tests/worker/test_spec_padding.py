# SPDX-License-Identifier: Apache-2.0
"""``spec_padding.prompt_path_tokens``: the prompt path never computes the scheduler's draft placeholders."""

from vllm_gaudi.v1.worker.spec_padding import prompt_path_tokens


def test_a_padded_one_token_request_computes_its_one_token():
    # a 1-token prompt padded to 1 + k = 4 (three -1 placeholders after the prompt)
    assert prompt_path_tokens(4, num_real_tokens=1, num_computed=0) == 1
    # a 700-token prompt whose cached prefix leaves one token, padded the same way
    assert prompt_path_tokens(4, num_real_tokens=700, num_computed=699) == 1


def test_ordinary_prompt_chunks_are_unchanged():
    assert prompt_path_tokens(512, num_real_tokens=700, num_computed=0) == 512  # a chunk
    assert prompt_path_tokens(188, num_real_tokens=700, num_computed=512) == 188  # the last chunk
    assert prompt_path_tokens(1, num_real_tokens=1, num_computed=0) == 1  # an unpadded 1-token prompt
    # a resumed request replaying its prompt and output (num_tokens_no_spec counts the output)
    assert prompt_path_tokens(730, num_real_tokens=730, num_computed=0) == 730
