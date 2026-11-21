# SPDX-License-Identifier: Apache-2.0
"""Upstream's speculative padding of a one-token request, on the HPU prompt path. Pure helper, importable on CPU.

vLLM's scheduler pads a waiting request that has exactly one token left to compute to ``1 + k`` tokens with ``k``
draft placeholders (-1) when speculative decoding is on and other requests run in the step ("pad new decode
requests to uniform spec decoding size to preserve full cudagraph", ``Scheduler.schedule``): a 1-token prompt that
joins a running batch, a prompt whose cached prefix leaves one token, a resumed request with one token to replay.
The HPU runner takes such a request on its prompt path (``num_computed_tokens < num_prompt_tokens``), where the
placeholders were computed as tokens (the recurrent state ran over them) and every row was sampled as output, so
the request emitted three garbage tokens after its first. The prompt path computes only the request's real tokens:
one token is sampled and the scheduler counts the ``k`` placeholders as rejected drafts.
"""

from __future__ import annotations

__all__ = ["prompt_path_tokens"]


def prompt_path_tokens(num_scheduled: int, num_real_tokens: int, num_computed: int) -> int:
    """Tokens a request on the prompt path computes this step: the scheduled ones, never past its real tokens
    (``num_real_tokens``: prompt plus accepted output, the runner's ``num_tokens_no_spec``); at least one."""
    return max(1, min(int(num_scheduled), int(num_real_tokens) - int(num_computed)))
