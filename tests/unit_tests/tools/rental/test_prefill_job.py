# SPDX-License-Identifier: Apache-2.0
"""``tools/flash_next/rental/prefill_job.sh`` end to end on a workstation.

A fake ``vllm`` on PATH (``prefill_fake_server``, which ignores every argument but ``--port``) stands in
for the real-model server, so the script's own work is checked: the server start and health wait, one
``prefill_sweep`` run per plan line with the line's own arguments (quoted JSON knobs included), the
runtime knobs file both sides agree on, the time limit that ends a job with its reports written, and the
server's stop.
"""

import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "tools" / "flash_next" / "rental" / "prefill_job.sh"
COMMON = ('--warm-tokens "" --concurrent "" --turn2-tokens 0 --shared-prefix 0 --canaries 0 --synthetic-ids '
          "--corpus-tokens 50000")

FAKE_TIMEOUT = '''#!/usr/bin/env python3
# timeout -s SIGNAL -k SECONDS SECONDS COMMAND... (GNU timeout's exit codes: 124 timed out, 137 killed)
import signal
import subprocess
import sys

args = sys.argv[1:]
sig, kill_after = signal.SIGTERM, None
while args and args[0].startswith("-"):
    if args[0] == "-s":
        sig = getattr(signal, "SIG" + args[1])
    elif args[0] == "-k":
        kill_after = float(args[1])
    else:
        sys.exit(125)
    args = args[2:]
proc = subprocess.Popen(args[1:])
try:
    sys.exit(proc.wait(timeout=float(args[0])))
except subprocess.TimeoutExpired:
    proc.send_signal(sig)
    try:
        proc.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        sys.exit(137)
    sys.exit(124)
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_is_closed(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def run_job(tmp_path: Path, plan: str, **extra_env) -> tuple[subprocess.CompletedProcess, Path, int, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm = bin_dir / "vllm"
    vllm.write_text("#!/bin/bash\nexec \"$FARM_PYTHON\" -m tests.unit_tests.tools.rental.prefill_fake_server \"$@\"\n")
    timeout = bin_dir / "timeout"
    timeout.write_text(FAKE_TIMEOUT)
    for tool in (vllm, timeout):
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    knobs = tmp_path / "knobs.json"
    plan_file = tmp_path / "plan.sweeps"
    plan_file.write_text(plan.replace("KNOBS", str(knobs)))
    out = tmp_path / "out"
    port = _free_port()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", PYTHONPATH=str(REPO_ROOT),
               FARM_OUT=str(out), FARM_PORT_BASE=str(port), FARM_NCARDS="4", FARM_PYTHON=sys.executable,
               VLLM_HPU_SCHED_KNOBS_FILE=str(knobs), PREFILL_HEALTH_TIMEOUT="120", **extra_env)
    done = subprocess.run(["bash", str(SCRIPT), str(plan_file), "--", "--long-prefill-token-threshold", "4096"],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    return done, out, port, knobs


def test_every_plan_line_runs_against_one_server_with_its_own_knobs(tmp_path):
    plan = f"""# two arms on one server: its own settings, then prompt graphs through the runtime knob
base --plan baseline --cold-tokens 2048 {COMMON} --nonce a --case-max-tokens 2

graphs --knobs-file KNOBS --knobs '{{"prompt_graph_max_tokens": 140000}}' --label graphs --plan baseline,pacing --cold-tokens 2048 --pacing 50:2048 --sweep-tokens 1024 {COMMON} --nonce b
"""
    done, out, port, knobs = run_job(tmp_path, plan)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "SERVER-HEALTHY" in done.stdout and "plan base rc 0" in done.stdout and "plan graphs rc 0" in done.stdout
    base = json.loads((out / "base" / "prefill_sweep.json").read_text())
    assert [(row["setting"], row["case"]) for row in base["cases"]] == [("server", "cold 2048")]
    assert base["cases"][0]["requests"][0]["output_tokens"] == 2
    graphs = json.loads((out / "graphs" / "prefill_sweep.json").read_text())
    assert [(row["setting"], row["case"]) for row in graphs["cases"]] == [
        ("graphs", "cold 2048"), ("graphs, share 50% slice 2048", "cold 1024")]
    assert graphs["cases"][1]["knobs"] == {"mode": 1, "share_percent": 50, "slice_tokens": 2048,
                                           "prompt_graph_max_tokens": 140000}
    assert json.loads(knobs.read_text()) == {"prompt_graph_max_tokens": 140000}  # the last arm stays set
    versions = (out / "versions.txt").read_text()
    assert f"VLLM_HPU_SCHED_KNOBS_FILE={knobs}" in versions
    assert "--tensor-parallel-size 4" in versions and "--long-prefill-token-threshold 4096" in versions
    assert "knobs file " + str(knobs) in (out / "server0.log").read_text()  # the server got the same file
    assert _port_is_closed(port)


def test_the_time_limit_ends_the_job_with_its_reports_written(tmp_path):
    # the fake server prefills 4,000 tokens per second here: the second line's 400,000-token prompt needs
    # 100 s and the job has 50 s in all (wide margins: the test also has to pass on a busy workstation)
    plan = f"""quick --plan baseline --cold-tokens 2048 {COMMON} --nonce a
slow --plan ladder --ladder 2048,400000 {COMMON} --nonce b
never --plan baseline --cold-tokens 2048 {COMMON} --nonce c
"""
    done, out, port, _ = run_job(tmp_path, plan, PREFILL_MAX_SECONDS="50", PREFILL_STOP_RESERVE="5",
                                 PREFILL_MIN_LINE_SECONDS="2", PREFILL_FAKE_RATE="4000")
    assert done.returncode == 6, done.stdout + done.stderr
    assert "plan quick rc 0" in done.stdout and "the time ran out in slow" in done.stdout
    assert json.loads((out / "quick" / "prefill_sweep.json").read_text())["cases"]
    slow = json.loads((out / "slow" / "prefill_sweep.json").read_text())  # written when the line was stopped
    assert [row["case"] for row in slow["cases"]] == ["cold 2048"]
    assert not (out / "never").exists()
    assert _port_is_closed(port)


def test_a_job_without_time_for_a_line_skips_it(tmp_path):
    plan = f"quick --plan baseline --cold-tokens 2048 {COMMON} --nonce a\n"
    done, out, port, _ = run_job(tmp_path, plan, PREFILL_MAX_SECONDS="5", PREFILL_STOP_RESERVE="4",
                                 PREFILL_MIN_LINE_SECONDS="30")
    assert done.returncode == 6, done.stdout + done.stderr
    assert "no time left for quick" in done.stdout
    assert not (out / "quick").exists()
    assert _port_is_closed(port)


def test_the_committed_plans_parse_and_never_repeat_a_cold_prompt():
    import shlex

    from tools.flash_next.rental import prefill_sweep as ps

    plans = sorted((SCRIPT.parent / "prefill_plans").glob("*.sweeps"))
    assert [p.name for p in plans] == ["arm.sweeps", "arm_then_pacing_cap.sweeps", "base_then_graphs.sweeps"]
    for plan in plans:
        names, nonces = [], []
        for line in plan.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, rest = line.split(None, 1)
            args = ps.parse_args(["--url", "u", "--model", "m", "--out", "o", *shlex.split(rest)])
            names.append(name)
            nonces.append(args.nonce)
            # a line that changes the running server says which file it writes
            changes = args.knobs is not None or "pacing" in args.plans or ("agents" in args.plans and args.agent_pacing)
            assert not changes or args.knobs_file == "/tmp/prefill_knobs.json"
        assert len(set(names)) == len(names), plan.name
        # the nonce salts a line's prompts: two lines with one nonce would send the same "cold" prompt twice
        # (the two suffix loops share theirs on purpose: the second goes on with the first one's conversations)
        once = [n for n in nonces if n != "arm-suffix"]
        assert len(set(once)) == len(once) and None not in nonces, plan.name
        assert names[:4] == ["c32k", "suffix", "streams", "c120k"], plan.name  # every arm starts with the same lines
