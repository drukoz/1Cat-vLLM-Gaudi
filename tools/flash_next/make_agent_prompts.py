# SPDX-License-Identifier: Apache-2.0
"""Build the coding-agent prompt set for the Flash-Next speculation benchmarks.

Emits a JSONL file in the format ``vllm bench serve --dataset-name custom`` reads: one
``{"prompt": ..., "output_tokens": ...}`` object per line, consumed with
``--dataset-path prompts.jsonl --custom-output-len -1`` (with ``-1`` each row uses its own ``output_tokens``).
The random dataset of stage 3e has no repetition, so n-gram drafting accepts nothing on it; these prompts make
the answer copy the prompt: a real ``vllm_gaudi/`` file of this repository (they ship in every checkout, so the
node can rebuild the identical set) plus a small editing instruction that asks for the complete edited file
back in one fenced code block.

Selection is deterministic: ``--count`` ``.py`` files whose line count lies in ``[--min-lines, --max-lines]``,
sorted by path, taken in that order. ``output_tokens = ceil(1.2 * len(file text) / 4)`` capped at
``--max-output-tokens``: the answer is about the file's size and 4 characters per token is the estimate.

Usage: ``python tools/flash_next/make_agent_prompts.py --out prompts.jsonl [--root DIR] [--count 64]
[--min-lines 80] [--max-lines 400] [--max-output-tokens 4096]``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# File i uses TEMPLATES[i % 3]; each asks for the complete file, so the answer copies the prompt.
TEMPLATES = [
    "Add a one-line docstring to every function and method that lacks one. Return the complete file.",
    "Rename the local variable names that are a single letter (except loop indices i, j, k) to descriptive "
    "names. Return the complete file.",
    "Add type hints to every function signature that lacks them. Return the complete file.",
]


def collect_files(root: Path, min_lines: int, max_lines: int) -> list[tuple[str, str]]:
    """``.py`` files under ``vllm_gaudi/`` with a qualifying line count, sorted by path: (relative path, text)."""
    found = []
    for path in sorted((root / "vllm_gaudi").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if min_lines <= len(text.splitlines()) <= max_lines:
            found.append((path.relative_to(root).as_posix(), text))
    return found


def output_tokens_for(text: str, max_output_tokens: int) -> int:
    """ceil(1.2 * len(text) / 4) == ceil(3 * len(text) / 10), in exact integer math, capped."""
    return min((3 * len(text) + 9) // 10, max_output_tokens)


def build_prompt(template: str, relpath: str, text: str) -> str:
    """Template sentence, blank line, ``File: <relative path>``, then the file inside a ```python fence."""
    body = text if text.endswith("\n") else text + "\n"
    return f"{template}\n\nFile: {relpath}\n\n```python\n{body}```\n"


def build_rows(files: list[tuple[str, str]], count: int, max_output_tokens: int) -> list[dict]:
    """The first ``count`` files as custom-dataset rows, cycling the three templates."""
    rows = []
    for i, (relpath, text) in enumerate(files[:count]):
        rows.append({
            "prompt": build_prompt(TEMPLATES[i % 3], relpath, text),
            "output_tokens": output_tokens_for(text, max_output_tokens),
        })
    return rows


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make the coding-agent prompt set (custom-dataset JSONL) for the speculation benchmarks.",
        epilog="Feed it to: vllm bench serve --dataset-name custom --dataset-path OUT --custom-output-len -1 "
        "(with -1 each row's output_tokens is used). output_tokens = ceil(1.2 * len(file) / 4): the answer is "
        "about the file's size, estimated at 4 characters per token, capped at --max-output-tokens.")
    parser.add_argument("--out", required=True, type=Path, help="JSONL file to write")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--count", type=int, default=64, help="number of prompts (default: 64)")
    parser.add_argument("--min-lines", type=int, default=80, help="minimum lines per file (default: 80)")
    parser.add_argument("--max-lines", type=int, default=400, help="maximum lines per file (default: 400)")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="cap on each row's output_tokens (default: 4096)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = collect_files(args.root, args.min_lines, args.max_lines)
    if len(files) < args.count:
        print(f"only {len(files)} files under {args.root / 'vllm_gaudi'} have between {args.min_lines} and "
              f"{args.max_lines} lines; --count {args.count} needs more", file=sys.stderr)
        return 2
    rows = build_rows(files, args.count, args.max_output_tokens)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    chars = sum(len(row["prompt"]) for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    print(f"wrote {len(rows)} prompts, {chars} prompt chars, {tokens} output tokens total -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
