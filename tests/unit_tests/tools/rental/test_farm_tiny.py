# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/farm_tiny.py: the track's tiny smoke served at the job's tensor parallel size on the
job's port, whatever TP the track's tiny_smoke.py has built in. Runs on CPU."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("farm_tiny", REPO_ROOT / "tools/flash_next/rental/farm/farm_tiny.py")
farm_tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm_tiny)

JOB = {"FARM_NCARDS": "4", "FARM_PORT_BASE": "8400", "FARM_OUT": "/results/00042-tiny",
       "FARM_RECIPE_CACHE": "/recipe-cache", "FARM_CARDS": "3,4,5,6", "FARM_MODULES": "4,5,6,7"}


def test_the_tensor_parallel_size_is_replaced_or_added():
    assert farm_tiny.with_tp(["vllm", "serve", "x", "--tensor-parallel-size", "8", "--port", "1"], 4) == \
        ["vllm", "serve", "x", "--tensor-parallel-size", "4", "--port", "1"]
    assert farm_tiny.with_tp(["vllm", "serve", "x"], 2) == ["vllm", "serve", "x", "--tensor-parallel-size", "2"]


def test_tp_is_taken_out_of_the_arguments():
    assert farm_tiny.split_tp(["--gdn", "hc", "--tp", "2"]) == (2, ["--gdn", "hc"])
    assert farm_tiny.split_tp(["--tp=1"]) == (1, [])
    assert farm_tiny.split_tp(["--gdn", "fused"]) == (None, ["--gdn", "fused"])
    with pytest.raises(SystemExit):
        farm_tiny.split_tp(["--tp", "zero"])


def test_the_job_fills_in_out_port_checkpoint_and_recipe_cache():
    args = farm_tiny.job_defaults(["--gdn", "hc"], JOB)
    assert args == ["--gdn", "hc", "--out", "/results/00042-tiny", "--port", "8410", "--ckpt", "/tmp/fn-smoke-ckpt",
                    "--recipe-cache", "/recipe-cache"]
    args = farm_tiny.job_defaults(["--port", "9000", "--out", "/x", "--no-recipe-cache"], JOB)
    assert args == ["--port", "9000", "--out", "/x", "--ckpt", "/tmp/fn-smoke-ckpt"]


def test_main_serves_the_tracks_tiny_smoke_at_the_jobs_tp(monkeypatch, tmp_path):
    from tools.flash_next.rental import tiny_smoke
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        seen["cmd"] = tiny_smoke.serve_command(tmp_path, 8410)
        return 7

    monkeypatch.setattr(tiny_smoke, "main", fake_main)
    monkeypatch.setattr(tiny_smoke, "serve_command", tiny_smoke.serve_command)  # restored after the test
    assert farm_tiny.main(["--gdn", "pytorch"], JOB) == 7
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert cmd[cmd.index("--port") + 1] == "8410"
    assert seen["argv"][:2] == ["--gdn", "pytorch"] and "--port" in seen["argv"]
    assert farm_tiny.main(["--tp", "2"], JOB) == 7
    assert seen["cmd"][seen["cmd"].index("--tensor-parallel-size") + 1] == "2"
