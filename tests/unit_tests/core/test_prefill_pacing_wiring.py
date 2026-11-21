# SPDX-License-Identifier: Apache-2.0
"""The plugin wiring of prefill pacing and the runtime knobs file, in a fresh interpreter as on a server.

vLLM loads the plugin's general plugin ``vllm_gaudi:register_utils`` in every process (setup.py
``vllm.general_plugins``); it imports ``vllm_gaudi.utils``, whose last lines install the ``Scheduler.schedule``
wrapper when ``VLLM_HPU_PREFILL_PACING`` or ``VLLM_HPU_SCHED_KNOBS_FILE`` asks for it. Each case runs in its own
interpreter, so the import order is the server's and no other test has patched the scheduler.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = """
from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (the CPU stand-ins of the Habana modules)
import vllm_gaudi
vllm_gaudi.register_utils()
from vllm.v1.core.sched.scheduler import Scheduler
print("WRAPPED" if getattr(Scheduler.schedule, "_hpu_prefill_pacing", False) else "PLAIN")
"""


def _run(extra_env: dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items()
           if k not in ("VLLM_HPU_PREFILL_PACING", "VLLM_HPU_SCHED_KNOBS_FILE")}
    env.update(extra_env)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VLLM_PLUGINS", "")
    env["TMPDIR"] = "/tmp"
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout.strip().splitlines()[-1]


def test_no_flag_leaves_the_scheduler_alone():
    assert _run({}) == "PLAIN"


@pytest.mark.parametrize("env", [{"VLLM_HPU_SCHED_KNOBS_FILE": "/tmp/fn-knobs-test.json"},
                                 {"VLLM_HPU_PREFILL_PACING": "1"}])
def test_a_flag_wraps_the_engine_scheduler(env):
    assert _run(env) == "WRAPPED"
