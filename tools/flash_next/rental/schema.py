# SPDX-License-Identifier: Apache-2.0
"""Shared record types of the Flash-Next rental harness (plan M4, RUK-2709).

Every status is an integer enum and travels as a number in JSON; text
appears only in reports. The record formats are in the plan's M4 contract.
"""

from __future__ import annotations

from enum import IntEnum

WORKLOAD_VERSION = "agent-128k-v1"
MAX_MODEL_LEN = 131072


class GateStatus(IntEnum):
    NOT_RUN = 0
    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    TIMED_OUT = 4


class RequestStatus(IntEnum):
    OK = 0
    HTTP_ERROR = 1
    TIMEOUT = 2
    SHORT_OUTPUT = 3


class OnFail(IntEnum):
    STOP = 0
    CONTINUE = 1


STARTUP_KEYS = ("server", "usable_num_blocks", "kv_cache_tokens", "max_concurrency", "max_concurrency_len",
                "weights_gib", "warmup_seconds", "gdn_layers_detected", "startup_seconds")
REQUEST_KEYS = ("agent", "turn", "server", "prompt_tokens", "cached_tokens", "output_tokens", "t_send", "t_first",
                "t_last", "itl_p50_ms", "status", "error")
METRICS_KEYS = ("t", "server", "running", "waiting", "kv_usage", "prefix_hits", "prefix_queries", "preemptions",
                "spec_accepted", "spec_drafts", "spec_draft_tokens", "generation_tokens", "prompt_tokens")

# File names inside one variant's folder, results/<run-id>/<variant-id>/
STARTUP_FILE = "startup.json"  # a list: one STARTUP_KEYS object per server
REQUESTS_FILE = "requests.jsonl"
METRICS_FILE = "metrics.jsonl"
HLSMI_FILE = "hlsmi.csv"
# {"id": str, "status": GateStatus, "reason": str | None, "seconds": float}, plus "errors" (the last error
# lines of the server log) after a failure and, for a variant, "servers"
STATUS_FILE = "status.json"
FIRST_TOKEN_FILE = "first-token.json"  # a list: one record per server, the first short completion after startup
