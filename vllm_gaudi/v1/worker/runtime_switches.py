# SPDX-License-Identifier: Apache-2.0
"""Flags a running server rereads from a file (``VLLM_HPU_RUNTIME_SWITCH_FILE``): an A/B on one server.

A real-model server takes about ten minutes to start, and two servers on two halves of a node differ by more
than a host-side fast path saves. With ``VLLM_HPU_RUNTIME_SWITCH_FILE=<path>`` every worker looks at ``<path>``
before each step (one ``os.stat``) and, when the file changed, takes the flags it lists. So one server measures
a path off and on in turn, on the same cards.

Only flags the runner reads at every step can be switched (``SWITCHABLE``); every other flag is read when the
model is built and needs a new server. The file holds one ``NAME=VALUE`` per line (``#`` starts a comment).
Replace it with a rename while no request runs: every worker then sees the change at the same step, the first
one of the next request. A file that cannot be read or names another flag is reported once and changes nothing.

This module imports nothing of the plugin (the runner cannot be imported on a machine without an HPU).
"""

import os
from typing import Optional

# VLLM_HPU_VOCAB_PARALLEL_LOGITS is read at every call of the logits region (distributed/vocab_parallel.mode()),
# which runs as lazy ops after the replayed decode graph: switching it changes the region's graph, not a captured one.
# VLLM_HPU_MTP_DRAFT_GRAPH is read at every decode draft (HpuFlashNextMtpProposer._draft_graph_enabled): 1 runs the
# k draft passes in the one-graph loop (built at its first draft), 0 the per-pass graphs; both keep their graphs.
# VLLM_HPU_MTP_DRAFT_CHAIN is decided before every verification (HPUModelRunner._mtp_chain_decision).
# VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS (async-sched's stage 2, M1) is read at every verification step by a server
# started at 1 or 2 (the device-made inputs' object exists only then); 0 turns it off while serving.
# The MoE dense range (VLLM_HPU_MOE_DENSE_MIN_TOKENS / _MAX_TOKENS, extension/ops.py) is read inside the model's
# forward, so a captured HPU graph keeps the path it was captured with: when the file names one of these
# (GRAPH_SWITCHABLE), the runner drops the captured graphs of the model and of the drafter and the next steps capture
# again. The server must start with VLLM_HPU_MOE_DENSE=1 (the stacked experts are kept only then); a minimum above
# the maximum is the vendor op at every size.
GRAPH_SWITCHABLE = ("VLLM_HPU_MOE_DENSE_MIN_TOKENS", "VLLM_HPU_MOE_DENSE_MAX_TOKENS", "VLLM_HPU_MOE_DENSE_FLAT_GATE_UP")
SWITCHABLE = ("VLLM_HPU_FLASH_NEXT_DECODE_METADATA", "VLLM_HPU_FUSED_GREEDY_LOGITS", "VLLM_HPU_CANDIDATE_SAMPLING",
              "VLLM_HPU_VOCAB_PARALLEL_LOGITS", "VLLM_HPU_MTP_DRAFT_GRAPH", "VLLM_HPU_MTP_DRAFT_CHAIN",
              "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS") + GRAPH_SWITCHABLE


def parse_switches(text: str) -> dict[str, str]:
    """``{name: value}`` of a switch file's text. Raises ``ValueError`` for a line that is not ``NAME=VALUE`` or
    names a flag outside ``SWITCHABLE``."""
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not value:
            raise ValueError(f"line {number}: expected NAME=VALUE, got {line!r}")
        if name not in SWITCHABLE:
            raise ValueError(f"line {number}: {name} cannot be switched at run time (only {', '.join(SWITCHABLE)})")
        values[name] = value
    return values


class RuntimeSwitchFile:
    """The switch file of one worker: ``poll()`` returns the flags when the file changed since the last call."""

    def __init__(self, path: str):
        self.path = path
        self._stamp: Optional[tuple[int, int, int]] = None

    def poll(self) -> Optional[dict[str, str]]:
        """The file's flags if it changed since the last call (or exists for the first time), else ``None``.

        A missing file is "no change". Raises ``OSError`` or ``ValueError`` for a file that cannot be read or
        parsed; that file is not read again until it changes."""
        try:
            status = os.stat(self.path)
        except OSError:
            return None
        stamp = (status.st_mtime_ns, status.st_size, status.st_ino)
        if stamp == self._stamp:
            return None
        self._stamp = stamp
        with open(self.path, encoding="utf-8") as handle:
            return parse_switches(handle.read())


__all__ = ["GRAPH_SWITCHABLE", "SWITCHABLE", "RuntimeSwitchFile", "parse_switches"]
