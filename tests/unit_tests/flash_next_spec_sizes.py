# SPDX-License-Identifier: Apache-2.0
"""The speculative sizes the Flash-Next CPU tests run at, and the startup check every sized test makes first.

``SPEC_SIZES`` are the numbers of drafted tokens per step (``k``, ``num_speculative_tokens``) the
speculative tests cover: the recipe (3) and the two sizes above it, up to the fused verify kernel's
limit (``1 + k = 6`` nodes). Every sized test first asks the platform's own startup validator
(``vllm_gaudi.platform._validate_flash_next_speculative_config``):

* a size the tests call served must start, or the test fails with the validator's message: a CPU proof
  of a size the server refuses would say nothing about the product;
* ``PROVEN_NOT_SERVED`` are sizes proven here ahead of the validator (none since the validator accepts
  ``k = 5``, 2026-11-20: at 32 sequences of 131072 tokens one 8-card server holds a little fewer full-length
  sequences at k = 5, which lowers the engine's concurrency figure, not the server's start;
  ``docs/design/flash-next/gdn-core-decode.md``, "Sizes above the recipe"). Tests of such a size run all the
  same, and fail once the validator accepts the size, until it is removed from that tuple.

The validator lives in ``vllm_gaudi.platform``, which imports ``habana_frameworks``; on a CPU-only
machine that import needs the stand-ins of ``hpu_cpu_stubs``, and those must not leak into a process
that runs the model (plugin code asks whether the Gaudi runtime imports). The validator therefore
runs once per test process in a child interpreter, for every size and configuration, and the answers
are kept.

Plain module (no pytest import): the rank functions and the harness modules may import it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

RECIPE_K = 3
SPEC_SIZES = (3, 4, 5)  # drafted tokens per step: the recipe and the sizes above it
PROVEN_NOT_SERVED = ()  # of those, the sizes the platform's validator still refuses at startup

_REPO_ROOT = Path(__file__).resolve().parents[2]

# (method, prefix caching) of the configurations the tests run; n-gram with prefix caching is refused for
# another reason and no test runs it.
_CONFIGURATIONS = (("mtp", False), ("mtp", True), ("ngram", False))

_CHILD = r"""
import json, os, sys
from types import SimpleNamespace
from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before vllm_gaudi.platform)
from vllm_gaudi.platform import _validate_flash_next_speculative_config

answers = {}
for method, prefix_caching, k in json.loads(sys.argv[1]):
    # The recurrent-state layout a server of that kind runs with: compact GDN without prefix caching,
    # the state pool (VLLM_COMPACT_GDN=0) with it.
    os.environ["VLLM_COMPACT_GDN"] = "0" if prefix_caching else "1"
    draft = ["Qwen4ExpMTP"] if method == "mtp" else []
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen4ExpForCausalLM"]),
                                     architecture=None),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=k,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=draft))))
    try:
        _validate_flash_next_speculative_config(config)
        answers[f"{method}/{int(prefix_caching)}/{k}"] = None
    except ValueError as error:
        answers[f"{method}/{int(prefix_caching)}/{k}"] = str(error)
print("ANSWERS=" + json.dumps(answers))
"""

_answers: dict[str, str | None] = {}


def _ask(cases: list[tuple[str, bool, int]]) -> None:
    env = dict(os.environ)
    inherited = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT)] + inherited)
    env.setdefault("VLLM_PLUGINS", "")
    done = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(cases)],
                          cwd=_REPO_ROOT,
                          env=env,
                          capture_output=True,
                          text=True,
                          check=False)
    lines = [line for line in done.stdout.splitlines() if line.startswith("ANSWERS=")]
    if done.returncode != 0 or not lines:
        raise RuntimeError("the startup validator could not be asked (exit "
                           f"{done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
    _answers.update(json.loads(lines[-1][len("ANSWERS="):]))


def startup_refusal(k: int, method: str = "mtp", prefix_caching: bool = False) -> str | None:
    """What the HPU platform answers at startup to a Flash-Next server that drafts ``k`` tokens with
    ``method``: ``None`` when it starts, else the validator's error message."""
    key = f"{method}/{int(prefix_caching)}/{k}"
    if key not in _answers:
        cases = [(m, pc, size) for (m, pc) in _CONFIGURATIONS for size in SPEC_SIZES]
        if (method, prefix_caching, k) not in cases:
            cases.append((method, prefix_caching, k))
        _ask(cases)
    return _answers[key]


def require_startup(k: int, method: str = "mtp", prefix_caching: bool = False) -> None:
    """Fail when the HPU platform's answer for this speculative size is not the one the tests assume:
    a served size it refuses, or a ``PROVEN_NOT_SERVED`` size it has started to accept."""
    reason = startup_refusal(k, method, prefix_caching)
    if k in PROVEN_NOT_SERVED:
        if reason is None:
            raise AssertionError(f"the HPU platform now starts Flash-Next with {method} and "
                                 f"num_speculative_tokens={k}: remove {k} from PROVEN_NOT_SERVED in "
                                 "tests/unit_tests/flash_next_spec_sizes.py")
        return
    if reason is not None:
        raise AssertionError(f"the HPU platform refuses to start Flash-Next with {method} and "
                             f"num_speculative_tokens={k} (prefix caching {'on' if prefix_caching else 'off'}): "
                             f"{reason}")


def size_id(k: int) -> str:
    return f"k{k}"
