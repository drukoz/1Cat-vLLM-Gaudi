# SPDX-License-Identifier: Apache-2.0
"""The deterministic ``agent-128k-v1`` workload of the rental harness (plan M4, RUK-2709).

``build_workload`` writes the same 32-agent, 128K-context coding-agent conversation set on every run of
the same commit (plan M4 rules, version ``agent-128k-v1``): a system message shared by every agent
(the repository guide ``AGENTS.md`` and the Flash-Next design documents, repeated from the start if
still short, trimmed by a binary search on the character count to
``[shared_tokens - 64, shared_tokens]`` tokens), and per agent a first user message made of
``File: <path>`` blocks of the eligible ``vllm_gaudi`` sources (80 to 400 lines, files taken cyclically
from index ``(i * len(files)) // agents + file_offset``) whose last block is cut to whole lines so the
conversation counts land in ``[initial_tokens - 512, initial_tokens]``, plus ``turns - 1`` follow-up
prompts. ``file_offset`` shifts every agent's file window, so a warm pass with ``file_offset=1`` plays
prompts that differ from the measured ones right after the shared system block.
Token counts come from a counter function: ``tokenize_counter`` asks a running server's ``/tokenize``
endpoint, ``estimate_counter`` is the offline fallback for tests and dry runs. The JSON is written with
``json.dump(..., ensure_ascii=False, sort_keys=True)`` so its sha256 is stable.

Usage: ``python -m tools.flash_next.rental.workload --out PATH (--tokenize-url URL --model NAME |
--estimate) [--agents 32 --shared-tokens 16384 --initial-tokens 124000 --turns 4 --max-tokens 1024]
[--file-offset N]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from tools.flash_next.rental.schema import MAX_MODEL_LEN, WORKLOAD_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_HEADER = "You are a coding agent working in the repository below. Follow the repository guide exactly.\n\n"

TEMPLATES = [
    "add a one-line docstring to every function and method that lacks one.",
    "rename the local variable names that are a single letter (except loop indices i, j, k) to descriptive names.",
    "add type hints to every function signature that lacks them.",
]

SHARED_TOKEN_WINDOW = 64
INITIAL_TOKEN_WINDOW = 512
FOLLOWUP_TOKEN_ALLOWANCE = 128
BUDGET_SLACK = 512
FILE_LINE_MIN = 80
FILE_LINE_MAX = 400
SHARED_FIRST_FILE = "AGENTS.md"
SHARED_DOC_GLOB = "docs/design/flash-next/**/*.md"
AGENT_FILE_GLOB = "vllm_gaudi/**/*.py"
TOKENIZE_TIMEOUT_S = 60.0

CORPUS_REPEAT_GUARD = 64


def estimate_counter(messages: list[dict]) -> int:
    """The offline token estimate of ``--estimate``: characters over 4 plus 8 tokens per message."""
    return sum(len(m["content"]) for m in messages) // 4 + 8 * len(messages)


def tokenize_counter(base_url: str, model: str) -> Callable[[list[dict]], int]:
    """A counter that asks ``<base_url>/tokenize`` of a running server; returns its ``count``.

    The body is ``{"model", "messages", "add_generation_prompt": true, "chat_template_kwargs":
    {"enable_thinking": false}}`` (the fields of ``TokenizeChatRequest`` at the vLLM pin, protocol.py
    lines 47-50 and 93; the response field ``count`` is protocol.py line 158).
    """
    url = base_url.rstrip("/") + "/tokenize"

    def count(messages: list[dict]) -> int:
        body = json.dumps({
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(url,
                                     data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=TOKENIZE_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return int(payload["count"])

    return count


def _shared_corpus(root: Path) -> str:
    """The system message before trimming: the guide header, AGENTS.md, then the Flash-Next docs."""
    paths = [root / SHARED_FIRST_FILE]
    paths += sorted(root.glob(SHARED_DOC_GLOB), key=lambda p: p.as_posix())
    parts = [SYSTEM_HEADER]
    for path in paths:
        parts.append(f"=== {path.relative_to(root).as_posix()} ===\n")
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "".join(parts)


def _build_shared_message(count_tokens: Callable[[list[dict]], int], root: Path, shared_tokens: int) -> tuple[str, int]:
    """The system message trimmed to ``[shared_tokens - 64, shared_tokens]`` tokens."""
    corpus = _shared_corpus(root)
    text = corpus
    for _ in range(CORPUS_REPEAT_GUARD):  # repeated from the start while still short
        if count_tokens([{"role": "system", "content": text}]) >= shared_tokens:
            break
        text += corpus
    else:
        raise ValueError(f"cannot reach {shared_tokens} tokens by repeating the shared corpus ({len(text)} chars)")

    def count_chars(n: int) -> int:
        return count_tokens([{"role": "system", "content": text[:n]}])

    lo, hi = 0, len(text)  # count(0) <= shared_tokens <= count(len(text))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_chars(mid) <= shared_tokens:
            lo = mid
        else:
            hi = mid - 1
    count = count_chars(lo)
    low = shared_tokens - SHARED_TOKEN_WINDOW
    if count < low:
        raise ValueError(f"shared message counts {count} tokens, below the window [{low}, {shared_tokens}]")
    return text[:lo], count


def _agent_files(root: Path) -> list[tuple[str, str]]:
    """Every ``vllm_gaudi`` ``*.py`` with 80 to 400 lines, sorted by relative path: (path, text)."""
    files = []
    for path in sorted(root.glob(AGENT_FILE_GLOB), key=lambda p: p.as_posix()):
        text = path.read_text(encoding="utf-8", errors="replace")
        if FILE_LINE_MIN <= len(text.splitlines()) <= FILE_LINE_MAX:
            files.append((path.relative_to(root).as_posix(), text))
    return files


def _block(path: str, text: str) -> str:
    return f"File: {path}\n```python\n{text}\n```\n\n"


def _first_message(sys_msg: str, files: list[tuple[str, str]], start: int, agent: int,
                   count_tokens: Callable[[list[dict]], int],
                   initial_tokens: int) -> tuple[str, list[str], int, int]:
    """The first user message of one agent: file blocks plus the task sentence.

    Returns ``(user message, block paths, final token count, number of files used)``. The blocks are
    added in doubling chunks until the conversation ``[system, user]`` counts at least
    ``initial_tokens``; the block that crossed the target is cut to whole lines with a binary search
    on its line count so the count lands in ``[initial_tokens - 512, initial_tokens]``. All counting
    already includes the task sentence, so the final message is the one measured.
    """
    n = len(files)

    def conv(user_text: str) -> int:
        return count_tokens([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_text}])

    task = (f"Task: in File: {files[start % n][0]}, {TEMPLATES[agent % 3]}"
            " Return the complete file in one fenced code block.")
    base_count = conv(task)
    if base_count >= initial_tokens:
        raise ValueError(f"agent {agent}: initial_tokens {initial_tokens} is below the system message and task"
                         f" ({base_count} tokens)")

    idx = start
    kept_paths: list[str] = []  # paths of the blocks kept whole, in order
    kept_text = ""
    kept_count = base_count  # conv(kept_text + task), the last measured state below initial_tokens
    chunk_size = 1
    last_path = ""
    last_block = ""
    while True:
        chunk = [files[(idx + k) % n] for k in range(chunk_size)]
        idx += chunk_size
        chunk_blocks = [_block(p, t) for p, t in chunk]
        trial_count = conv(kept_text + "".join(chunk_blocks) + task)
        if trial_count < initial_tokens:
            kept_paths += [p for p, _ in chunk]
            kept_text += "".join(chunk_blocks)
            kept_count = trial_count
            chunk_size *= 2
            continue
        # the target is crossed inside this chunk: find its first crossing block by binary search
        lo, hi = 1, chunk_size  # conv(kept + chunk[:0] + task) < target <= conv(kept + chunk[:hi] + task)
        while lo < hi:
            mid = (lo + hi) // 2
            if conv(kept_text + "".join(chunk_blocks[:mid]) + task) >= initial_tokens:
                hi = mid
            else:
                lo = mid + 1
        for k in range(lo - 1):
            kept_paths.append(chunk[k][0])
            kept_text += chunk_blocks[k]
        last_path, last_block = chunk[lo - 1][0], chunk_blocks[lo - 1]
        break

    if kept_count >= initial_tokens:  # defensive: the loop above keeps this last measured state below target
        raise ValueError(f"agent {agent}: no file block left to trim (kept state {kept_count} tokens)")
    lines = last_block.splitlines(keepends=True)

    def total(keep: int) -> int:
        return conv(kept_text + "".join(lines[:keep]) + task)

    lo, hi = 0, len(lines)  # total(0) = kept_count < target; total(len) >= target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total(mid) <= initial_tokens:
            lo = mid
        else:
            hi = mid - 1
    count = total(lo)
    if count < initial_tokens - INITIAL_TOKEN_WINDOW:
        raise ValueError(f"agent {agent}: first prompt counts {count} tokens, below the window"
                         f" [{initial_tokens - INITIAL_TOKEN_WINDOW}, {initial_tokens}]")
    user = kept_text + "".join(lines[:lo]) + task
    paths = kept_paths + [last_path]
    return user, paths, count, len(paths)


def build_workload(root: Path,
                   agents: int,
                   shared_tokens: int,
                   initial_tokens: int,
                   turns: int,
                   max_tokens: int,
                   count_tokens: Callable[[list[dict]], int],
                   on_agent: Callable[[int, int, int], None] | None = None,
                   file_offset: int = 0) -> dict:
    """The ``workload.json`` dict of the ``agent-128k-v1`` rules, built over the repository at ``root``.

    ``count_tokens(messages)`` returns the token count of a message list (a server counter from
    ``tokenize_counter`` or ``estimate_counter``); ``on_agent(agent, prompt_tokens, files)`` is called
    once per agent, after its first prompt is sized; ``file_offset`` shifts each agent's first file
    index (the measured workload uses ``0``, the warm pass ``1``). Raises ``ValueError`` when the
    prompt and output budget does not fit the model length.
    """
    budget = initial_tokens + turns * max_tokens + (turns - 1) * FOLLOWUP_TOKEN_ALLOWANCE
    if budget > MAX_MODEL_LEN - BUDGET_SLACK:
        raise ValueError(f"budget {budget} tokens (initial {initial_tokens} + {turns} x {max_tokens} +"
                         f" {turns - 1} x {FOLLOWUP_TOKEN_ALLOWANCE}) does not fit"
                         f" {MAX_MODEL_LEN} - {BUDGET_SLACK} = {MAX_MODEL_LEN - BUDGET_SLACK} tokens")
    if agents < 1:
        raise ValueError(f"agents must be >= 1, got {agents}")
    files = _agent_files(root)
    if not files:
        raise ValueError(f"no *.py file under {root / 'vllm_gaudi'} has {FILE_LINE_MIN} to {FILE_LINE_MAX} lines")

    sys_msg, shared_count = _build_shared_message(count_tokens, root, shared_tokens)

    entries = []
    for i in range(agents):
        start = (i * len(files)) // agents + file_offset
        user, paths, count, n_files = _first_message(sys_msg, files, start, i, count_tokens, initial_tokens)
        followups = [
            f"Now, in File: {paths[min(t, len(paths) - 1)]}, {TEMPLATES[(i + t) % 3]}"
            " Return the complete file in one fenced code block."
            for t in range(1, turns)
        ]
        entries.append({
            "agent": i,
            "messages": [{"role": "system", "content": sys_msg}, {
                "role": "user",
                "content": user
            }],
            "followups": followups,
        })
        if on_agent is not None:
            on_agent(i, count, n_files)

    return {
        "version": WORKLOAD_VERSION,
        "shared_prefix_tokens": shared_count,
        "target_initial_tokens": initial_tokens,
        "turns": turns,
        "max_tokens": max_tokens,
        "file_offset": file_offset,
        "agents": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.workload",
                                     description="Build the deterministic agent-128k-v1 workload (plan M4).")
    parser.add_argument("--out", required=True, help="path of the workload JSON to write")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokenize-url", help="base URL of a running server whose /tokenize sizes the prompts")
    source.add_argument("--estimate", action="store_true", help="size prompts with the offline character estimate")
    parser.add_argument("--model", default=None, help="served model name for /tokenize (required with --tokenize-url)")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--shared-tokens", type=int, default=16384)
    parser.add_argument("--initial-tokens", type=int, default=124000)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--file-offset",
                        type=int,
                        default=0,
                        help="shift of each agent's first file index (the warm pass uses 1)")
    args = parser.parse_args(argv)

    if args.tokenize_url is not None and not args.model:
        parser.error("--model is required with --tokenize-url")
    if args.tokenize_url is None and args.model:
        parser.error("--model needs --tokenize-url")

    if args.tokenize_url is not None:
        counter = tokenize_counter(args.tokenize_url, args.model)

        def on_agent(agent: int, count: int, n_files: int) -> None:
            print(f"agent {agent}: {count} tokens, {n_files} files", flush=True)
    else:
        counter = estimate_counter
        on_agent = None

    try:
        workload = build_workload(REPO_ROOT,
                                  args.agents,
                                  args.shared_tokens,
                                  args.initial_tokens,
                                  args.turns,
                                  args.max_tokens,
                                  counter,
                                  on_agent=on_agent,
                                  file_offset=args.file_offset)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(workload, f, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({args.agents} agents, sha256 {sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
