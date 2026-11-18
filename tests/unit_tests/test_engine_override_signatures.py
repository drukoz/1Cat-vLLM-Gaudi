# SPDX-License-Identifier: Apache-2.0
"""Check that every engine function the plugin replaces still takes the engine's parameters.

On the 2026-11-18 rental all 8 workers of the tiny TP8 server died with
``TypeError: create_fused_moe_router() got an unexpected keyword argument
'bias_vl'``: the plugin replaces vLLM's ``create_fused_moe_router`` by
assignment, and vLLM at the pin had added ``bias_vl`` and
``image_sentinel_lo`` (fixed in commit ``6051eae``). This test compares
``inspect.signature`` of each engine function (read from a child process that
imports only vLLM, because the plugin override hides the original in the test
process) with the plugin function that replaces it, and fails whenever the
engine passes a parameter the replacement does not accept.

Add a pair here when the plugin replaces another engine function by assignment.
"""

import os
import sys
import json
import subprocess
import importlib
import inspect
from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

os.environ.setdefault("VLLM_PLUGINS", "")

ROOT = Path(__file__).resolve().parents[2]

# (engine "module:qualname", plugin override "module:qualname")
PAIRS = [
    ("vllm.model_executor.layers.fused_moe.router.router_factory:create_fused_moe_router",
     "vllm_gaudi.ops.hpu_fused_moe:create_fused_moe_router"),
    ("vllm.model_executor.model_loader.utils:process_weights_after_loading",
     "vllm_gaudi.ops.hpu_weights:hpu_process_weights_after_loading"),
    ("vllm.model_executor.layers.mamba.abstract:MambaBase.bind_kv_cache",
     "vllm_gaudi.patches:_hpu_mamba_bind_kv_cache"),
    ("vllm.v1.core.kv_cache_coordinator:get_kv_cache_coordinator",
     "vllm_gaudi.utils:_hpu_get_kv_cache_coordinator"),
    ("vllm.v1.core.block_pool:BlockPool.free_blocks",
     "vllm_gaudi.patches:_hpu_free_blocks"),
    ("vllm.config:ModelConfig.get_num_layers_by_block_type",
     "vllm_gaudi.patches:_hpu_get_num_layers_by_block_type"),
    ("vllm.distributed.parallel_state:cleanup_dist_env_and_memory",
     "vllm_gaudi.patches:_hpu_cleanup_dist_env_and_memory"),
    ("vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner.forward",
     "vllm_gaudi.ops.hpu_fused_moe:_patched_default_moe_runner_forward"),
    ("vllm.model_executor.layers.fused_moe.config:FusedMoEConfig.__post_init__",
     "vllm_gaudi.ops.hpu_fused_moe:_patched_moe_config_post_init"),
]

# Child process: imports only vLLM (no vllm_gaudi, no stub) and prints the
# engine signatures as JSON on the last stdout line (vLLM logs to stdout first).
CHILD_CODE = """
import sys, json, os, importlib, inspect
os.environ.setdefault("VLLM_PLUGINS", "")

def resolve(path):
    mod, attr = path.split(":")
    obj = importlib.import_module(mod)
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj

def params(fn):
    return [(p.name, int(p.kind), p.default is not inspect.Parameter.empty)
            for p in inspect.signature(fn).parameters.values()]

paths = json.loads(sys.argv[1])
print(json.dumps({p: params(resolve(p)) for p in paths}))
"""


def resolve(path):
    mod, attr = path.split(":")
    obj = importlib.import_module(mod)
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def params(fn):
    # [name, int(kind), has_default]; kinds: 0 POSITIONAL_ONLY,
    # 1 POSITIONAL_OR_KEYWORD, 2 VAR_POSITIONAL, 3 KEYWORD_ONLY, 4 VAR_KEYWORD
    return [(p.name, int(p.kind), p.default is not inspect.Parameter.empty)
            for p in inspect.signature(fn).parameters.values()]


@pytest.fixture(scope="module")
def engine_signatures():
    engine_paths = [e for e, _ in PAIRS]
    try:
        out = subprocess.run(
            [sys.executable, "-c", CHILD_CODE, json.dumps(engine_paths)],
            env=dict(os.environ, VLLM_PLUGINS="", PYTHONPATH=str(ROOT)),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"engine-signature child exited {exc.returncode}:\n{(exc.stderr or '')[-2000:]}")
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "engine,override",
    PAIRS,
    ids=[e.split(":")[1] for e, _ in PAIRS],
)
def test_override_accepts_engine_parameters(engine, override, engine_signatures):
    eng = engine_signatures[engine]
    ov = params(resolve(override))
    names = {n for n, _, _ in ov}
    varkw = any(k == 4 for _, k, _ in ov)
    # 1. every non-*args/**kwargs engine parameter has a same-named override
    # parameter, unless the override has **kwargs.
    missing = [n for n, k, _ in eng if k not in (2, 4) and n not in names and not varkw]
    # 2. the override has no parameter without a default (other than
    # *args/**kwargs) that the engine function lacks.
    extra = [n for n, k, d in ov if k not in (2, 4) and not d and n not in {x for x, _, _ in eng}]
    # 3. if the engine function has *args or **kwargs, the override has the same kind.
    var_kinds = [k for k in (2, 4)
                 if any(k == ek for _, ek, _ in eng) and not any(k == ok for _, ok, _ in ov)]

    problems = []
    if missing:
        problems.append(f"does not accept {missing} of {engine}")
    if extra:
        problems.append(f"requires {extra} that {engine} lacks")
    if var_kinds:
        problems.append(f"lacks *args/**kwargs (kind {var_kinds}) of {engine}")
    assert not problems, f"{override} {', '.join(problems)}"


def test_router_override_is_installed():
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401  (installs the override)
    import vllm.model_executor.layers.fused_moe.layer as moe_layer
    import vllm.model_executor.layers.fused_moe.router.router_factory as router_factory
    from vllm_gaudi.ops.hpu_fused_moe import create_fused_moe_router

    assert router_factory.create_fused_moe_router is create_fused_moe_router
    assert moe_layer.create_fused_moe_router is create_fused_moe_router
