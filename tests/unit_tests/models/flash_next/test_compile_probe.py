# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``tools/flash_next/compile_probe.py`` (the per-component compile timer).

The CPU run drives the whole probe (build, move, capture, time, table) on the tiny checkpoint at
128 tokens: every expected component must appear in ``probe.json`` with a float ``compile_s`` and
``replay_s`` (the times are meaningless on a Mac; the flow and the records are the point). The
second test checks the lazy-mode refusal of ``--device hpu``: exit 2 with the message, before
anything HPU is imported.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import sys  # noqa: E402

from tools.flash_next import compile_probe  # noqa: E402

EXPECTED_COMPONENTS = (
    "model",
    "Qwen4ExpDecoderLayer",
    "HPUGatedDeltaNetAttention",
    "HpuQwen4ExpQSAAttention",
    "HpuQsaIndexer.select",
    "HpuQwen4ExpPLELayer",
    "HpuQwen4ExpNGramEmbedding",
    "HpuGatedResidual",
    "Qwen4ExpSparseMoeBlock",  # the MoE block: the first class whose name contains Moe/MoE
)


def test_cpu_probe_records_every_component(tiny_checkpoint, tmp_path, monkeypatch):
    """The CPU flow end to end at 128 tokens: exit 0 and one probe.json row per component."""
    monkeypatch.delenv(compile_probe.LAZY_MODE_FLAG, raising=False)
    out_dir = tmp_path / "probe-out"
    rc = compile_probe.main(["--ckpt", str(tiny_checkpoint), "--device", "cpu", "--tokens", "128", "--out",
                             str(out_dir)])
    assert rc == 0, f"compile_probe exited {rc}"
    entries = json.loads((out_dir / "probe.json").read_text(encoding="utf-8"))
    components = {entry["component"] for entry in entries}
    for component in EXPECTED_COMPONENTS:
        assert component in components, f"probe.json has no entry for {component}: {sorted(components)}"
    for entry in entries:
        assert entry["tokens"] == 128, entry
        assert isinstance(entry["compile_s"], float), entry
        assert isinstance(entry["replay_s"], float), entry
    decoder_layers = [e for e in entries if e["component"] == "Qwen4ExpDecoderLayer"]
    assert len(decoder_layers) == 4, f"expected one row per decoder layer, got {len(decoder_layers)}"


def test_hpu_probe_refuses_without_lazy_mode(tmp_path, monkeypatch, capsys):
    """--device hpu without PT_HPU_LAZY_MODE=1 exits 2 with the message, importing no HPU code."""
    monkeypatch.delenv(compile_probe.LAZY_MODE_FLAG, raising=False)
    rc = compile_probe.main([
        "--ckpt",
        str(tmp_path / "no-checkpoint-needed"),
        "--device",
        "hpu",
        "--out",
        str(tmp_path / "out"),
    ])
    assert rc == 2
    stderr = capsys.readouterr().err
    assert compile_probe.LAZY_MODE_FLAG in stderr, stderr
    assert "habana_frameworks" not in sys.modules, "the lazy-mode check must run before any HPU import"
