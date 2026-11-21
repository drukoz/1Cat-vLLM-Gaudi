# SPDX-License-Identifier: Apache-2.0
"""The speculative-decoding counters of a running vLLM server, read from its ``/metrics`` page.

A server started with ``--speculative-config`` exports four counters (``vllm/v1/spec_decode/metrics.py``,
``SpecDecodingProm``; prometheus_client appends ``_total`` in the text format):

* ``vllm:spec_decode_num_drafts``: verification steps that carried drafts, one per request per step;
* ``vllm:spec_decode_num_draft_tokens``: draft tokens proposed;
* ``vllm:spec_decode_num_accepted_tokens``: draft tokens accepted;
* ``vllm:spec_decode_num_accepted_tokens_per_pos`` (label ``position``): accepted tokens per draft position.

A server without speculation has none of them (``parse_spec_counters`` returns None). Two readings around a
probe give its counters (``counters_delta``) and ``spec_summary`` turns them into vLLM's own figures: the
acceptance length ``1 + accepted / drafts`` (tokens per verification step per request, the bonus token
included; ``SpecDecodingLogging.log``), the draft acceptance rate ``accepted / draft_tokens`` and the rate of
every draft position. The counters are summed over engines (the ``engine`` label).

Standard library only (Python 3.10): the rental probes import it on the node.
"""

from __future__ import annotations

import re
import urllib.request

SPEC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
METRICS_TIMEOUT_S = 10.0

_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)")
_POSITION = re.compile(r'(?:^|,)\s*position="(?P<pos>\d+)"')
_SCALARS = {SPEC_DRAFTS + "_total": "drafts", SPEC_DRAFT_TOKENS + "_total": "draft_tokens",
            SPEC_ACCEPTED + "_total": "accepted"}


def parse_spec_counters(text: str) -> dict | None:
    """``{"drafts", "draft_tokens", "accepted", "accepted_per_pos": [...]}`` from a ``/metrics`` text, summed
    over engines; None when the text has no speculative counter (a server without speculation)."""
    found = False
    counters = {"drafts": 0, "draft_tokens": 0, "accepted": 0}
    per_pos: dict[int, int] = {}
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name")
        try:
            value = int(float(match.group("value")))
        except ValueError:
            continue
        if name in _SCALARS:
            counters[_SCALARS[name]] += value
            found = True
        elif name == SPEC_ACCEPTED_PER_POS + "_total":
            position = _POSITION.search(match.group("labels") or "")
            if position is None:
                continue
            pos = int(position.group("pos"))
            per_pos[pos] = per_pos.get(pos, 0) + value
            found = True
    if not found:
        return None
    width = max(per_pos) + 1 if per_pos else 0
    counters["accepted_per_pos"] = [per_pos.get(pos, 0) for pos in range(width)]
    return counters


def _get_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - the probe's own server
        return response.read().decode("utf-8", errors="replace")


def read_spec_counters(base_url: str, get_text=None) -> dict | None:
    """The counters of the server at ``base_url`` (its ``/metrics`` page). Raises OSError when it does not answer."""
    get_text = get_text or _get_text
    return parse_spec_counters(get_text(base_url.rstrip("/") + "/metrics", METRICS_TIMEOUT_S))


def counters_delta(before: dict | None, after: dict | None) -> dict | None:
    """``after - before``; None when either reading is missing. A shorter position list counts as zeros."""
    if before is None or after is None:
        return None
    width = max(len(before["accepted_per_pos"]), len(after["accepted_per_pos"]))
    pad = lambda values: list(values) + [0] * (width - len(values))  # noqa: E731
    return {"drafts": after["drafts"] - before["drafts"],
            "draft_tokens": after["draft_tokens"] - before["draft_tokens"],
            "accepted": after["accepted"] - before["accepted"],
            "accepted_per_pos": [a - b for a, b in zip(pad(after["accepted_per_pos"]), pad(before["accepted_per_pos"]))]}


def acceptance_length(drafts: int, accepted: int) -> float | None:
    """Tokens per verification step per request: ``1 + accepted / drafts`` (None without a step that drafted)."""
    if drafts <= 0:
        return None
    return 1.0 + accepted / drafts


def spec_summary(delta: dict | None) -> dict | None:
    """vLLM's figures for one probe's counters: acceptance length, draft acceptance rate, rate per position."""
    if delta is None:
        return None
    drafts, draft_tokens, accepted = delta["drafts"], delta["draft_tokens"], delta["accepted"]
    return {"drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted": accepted,
            "acceptance_length": acceptance_length(drafts, accepted),
            "draft_acceptance_rate": accepted / draft_tokens if draft_tokens > 0 else None,
            "position_rates": [count / drafts if drafts > 0 else None for count in delta["accepted_per_pos"]]}
