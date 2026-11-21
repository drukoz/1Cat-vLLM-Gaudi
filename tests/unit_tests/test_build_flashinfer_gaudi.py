# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused GDN build failure handling in tools/build_flashinfer_gaudi.py."""

import subprocess
import sys

import pytest

import tools.build_flashinfer_gaudi as bfg


def _make_stubs(monkeypatch, tmp_path, fused_stub):
    calls = {"kernel_db": [], "pytorch": [], "fused": []}

    def kernel_db_stub(source_root, build_root, output_dir):
        calls["kernel_db"].append((source_root, build_root, output_dir))

    def pytorch_stub(source_root, build_root, output_dir):
        calls["pytorch"].append((source_root, build_root, output_dir))

    def fused_wrapper(repository_root, build_root, output_dir):
        calls["fused"].append((repository_root, build_root, output_dir))
        return fused_stub(repository_root, build_root, output_dir)

    monkeypatch.setattr(bfg, "build_kernel_database", kernel_db_stub)
    monkeypatch.setattr(bfg, "build_pytorch_extension", pytorch_stub)
    monkeypatch.setattr(bfg, "build_gdn_fused", fused_wrapper)
    return calls


def test_env_build_failure_keeps_install(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"

    def fused_stub(repository_root, build_root, output_dir):
        (output_dir / "libgdn_fused_gaudi2.so").write_bytes(b"")
        (output_dir / "gdn_fused_ops.cpython-312-darwin.so").write_bytes(b"")
        raise subprocess.CalledProcessError(1, ["cmake"])

    calls = _make_stubs(monkeypatch, tmp_path, fused_stub)
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_flashinfer_gaudi.py",
            "--output-dir",
            str(out),
            "--build-root",
            str(tmp_path / "build"),
        ],
    )

    bfg.main()

    assert len(calls["kernel_db"]) == 1
    assert len(calls["pytorch"]) == 1
    assert not (out / "libgdn_fused_gaudi2.so").exists()
    assert not (out / "gdn_fused_ops.cpython-312-darwin.so").exists()
    stderr = capsys.readouterr().err
    assert "GDN fused build failed" in stderr
    assert "Continuing without libgdn_fused_gaudi2.so" in stderr


def test_gdn_fused_only_failure_exits_3(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"

    def fused_stub(repository_root, build_root, output_dir):
        (output_dir / "libgdn_fused_gaudi2.so").write_bytes(b"")
        raise FileNotFoundError("no extension")

    calls = _make_stubs(monkeypatch, tmp_path, fused_stub)
    monkeypatch.delenv("VLLM_GAUDI_BUILD_GDN_FUSED", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_flashinfer_gaudi.py",
            "--output-dir",
            str(out),
            "--build-root",
            str(tmp_path / "build"),
            "--gdn-fused-only",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        bfg.main()

    assert excinfo.value.code == 3
    assert len(calls["kernel_db"]) == 0
    assert len(calls["pytorch"]) == 0
    assert not (out / "libgdn_fused_gaudi2.so").exists()
    stderr = capsys.readouterr().err
    assert "Continuing without" not in stderr


def test_env_build_success(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"

    def fused_stub(repository_root, build_root, output_dir):
        return output_dir / "gdn_fused_ops.so"

    calls = _make_stubs(monkeypatch, tmp_path, fused_stub)
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_flashinfer_gaudi.py",
            "--output-dir",
            str(out),
            "--build-root",
            str(tmp_path / "build"),
        ],
    )

    bfg.main()

    assert len(calls["fused"]) == 1
    stderr = capsys.readouterr().err
    assert "failed" not in stderr
