# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/make_agent_prompts.py (the custom-dataset prompt-set builder)."""

import json
from pathlib import Path

import pytest

from tools.flash_next import make_agent_prompts as m


def write_py(root: Path, rel: str, n_lines: int) -> Path:
    """A ``.py`` file of exactly ``n_lines`` lines with content unique to ``rel``."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# {rel} line {i}\n" for i in range(n_lines)), encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A fake repository: 4 qualifying files, one under --min-lines, one over --max-lines, one non-.py."""
    write_py(tmp_path, "vllm_gaudi/a.py", 100)
    write_py(tmp_path, "vllm_gaudi/b.py", 50)     # below --min-lines
    write_py(tmp_path, "vllm_gaudi/c.py", 500)    # above --max-lines
    write_py(tmp_path, "vllm_gaudi/d.py", 200)
    write_py(tmp_path, "vllm_gaudi/f.py", 150)
    write_py(tmp_path, "vllm_gaudi/sub/e.py", 80)  # exactly --min-lines
    (tmp_path / "vllm_gaudi" / "readme.txt").write_text("not python\n", encoding="utf-8")
    return tmp_path


def run(root: Path, out: Path, *extra: str) -> int:
    return m.main(["--root", str(root), "--out", str(out), *extra])


def read_rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def file_lines(rows: list[dict]) -> list[str]:
    return [row["prompt"].splitlines()[2] for row in rows]


def test_selection_order_and_line_filters(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "4") == 0
    assert file_lines(read_rows(out)) == [
        "File: vllm_gaudi/a.py",
        "File: vllm_gaudi/d.py",
        "File: vllm_gaudi/f.py",
        "File: vllm_gaudi/sub/e.py",
    ]


def test_count_caps_the_selection(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "2") == 0
    assert file_lines(read_rows(out)) == ["File: vllm_gaudi/a.py", "File: vllm_gaudi/d.py"]


def test_template_cycle(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "4") == 0
    prompts = [row["prompt"] for row in read_rows(out)]
    assert [p.splitlines()[0] for p in prompts] == [
        m.TEMPLATES[0], m.TEMPLATES[1], m.TEMPLATES[2], m.TEMPLATES[0]]
    for sentence in m.TEMPLATES:
        assert sentence.endswith("Return the complete file.")
    for prompt in prompts:
        assert "```python\n" in prompt


def test_output_tokens_formula(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "2") == 0
    for row, rel in zip(read_rows(out), ["vllm_gaudi/a.py", "vllm_gaudi/d.py"]):
        text = (tree / rel).read_text(encoding="utf-8")
        assert row["output_tokens"] == (3 * len(text) + 9) // 10  # ceil(1.2 * len / 4), uncapped here


def test_output_tokens_cap(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "4", "--max-output-tokens", "5") == 0
    assert [row["output_tokens"] for row in read_rows(out)] == [5, 5, 5, 5]


def test_deterministic_two_runs(tree: Path) -> None:
    out1, out2 = tree / "p1.jsonl", tree / "p2.jsonl"
    assert run(tree, out1, "--count", "4") == 0
    assert run(tree, out2, "--count", "4") == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_every_line_is_json_with_two_keys(tree: Path) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "4") == 0
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {"prompt", "output_tokens"}
        assert isinstance(row["prompt"], str) and isinstance(row["output_tokens"], int)


def test_exit_2_when_too_few_files_qualify(tree: Path, capsys: pytest.CaptureFixture) -> None:
    out = tree / "p.jsonl"
    assert run(tree, out, "--count", "5") == 2
    assert "only 4 files" in capsys.readouterr().err
    assert not out.exists()


def test_real_repo_count_8(tmp_path: Path) -> None:
    out = tmp_path / "p.jsonl"
    assert m.main(["--out", str(out), "--count", "8"]) == 0
    rows = read_rows(out)
    assert len(rows) == 8
    for row in rows:
        assert set(row) == {"prompt", "output_tokens"}
        assert row["prompt"].splitlines()[2].startswith("File: vllm_gaudi/")
