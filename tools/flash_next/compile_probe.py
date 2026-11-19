# SPDX-License-Identifier: Apache-2.0
"""Per-component HPU graph compile times for the tiny Flash-Next model.

``python -m tools.flash_next.compile_probe --ckpt DIR --device hpu --tokens 128,512,2048 --out DIR``
builds the tiny Flash-Next model (sparse QSA, side caches bound, SDPA stand-ins) with the CPU
harness in ``tests/unit_tests/models/flash_next/cpu_model.py``, captures one eager prefill's
inputs per component, and times the first (compiling) and second (replaying) call of each
component with the module wrapped in an HPU graph (``htorch.hpu.wrap_in_hpu_graph``,
``disable_tensor_cache=True``). The printed markdown table (sorted by compile time, descending)
and ``<out>/probe.json`` say which component makes a prompt graph take 13-35 s to compile.

On the rented node, from the host (the container is the prepared ``fn-smoke`` one from
``tiny_smoke.sh``, and the stack is installed in it):

.. code-block:: bash

    sudo docker exec -w /bundle/plugin -e PT_HPU_LAZY_MODE=1 fn-smoke python -m tools.flash_next.compile_probe \
        --ckpt /tmp/fn-smoke-ckpt --device hpu --out /results/probe-$(date -u +%Y%m%dT%H%M%SZ)

``--device cpu`` runs the identical flow on the Mac without any graph wrapping (the times are
meaningless; the CPU path exists to keep the probe working, under
``tests/unit_tests/models/flash_next/test_compile_probe.py``). The probe refuses to start on
HPU unless ``PT_HPU_LAZY_MODE=1`` (exit 2, before importing anything HPU).

Components timed, matched by class name: the whole model (unless ``--skip-whole-model``), every
``Qwen4ExpDecoderLayer``, and the first instance of each of ``HPUGatedDeltaNetAttention``,
``HpuQwen4ExpQSAAttention``, ``HpuQwen4ExpPLELayer``, ``HpuQwen4ExpNGramEmbedding``,
``HpuGatedResidual``, the MoE block (first class whose name contains ``Moe``/``MoE``) and the
indexer's ``select`` (wrapped in a ``SelectModule``). The GDN/QSA/PLE layers read
``get_forward_context()``, so every timed call runs under ``set_forward_context``. Between
shapes the states and the attention stand-ins are reset; one component whose first call exceeds
120 s gets its replay skipped (``replay_s: null`` and a note).
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

LAZY_MODE_FLAG = "PT_HPU_LAZY_MODE"
LAZY_MODE_MESSAGE = (f"compile_probe: refusing to start: --device hpu needs {LAZY_MODE_FLAG}=1 in the environment "
                     f"(the servers' lazy mode; pass -e {LAZY_MODE_FLAG}=1 to docker exec). No HPU code was imported.")
FIRST_CALL_BUDGET_S = 120.0
MAX_MODEL_LEN = 4096  # the compressed key cache has one row per 4 tokens; 2048-token shapes need >= 512 rows

MODEL_CLASS = "model"
DECODER_LAYER_CLASS = "Qwen4ExpDecoderLayer"
SELECT_COMPONENT = "HpuQsaIndexer.select"
COMPONENT_CLASSES = (
    "HPUGatedDeltaNetAttention",
    "HpuQwen4ExpQSAAttention",
    "HpuQwen4ExpPLELayer",
    "HpuQwen4ExpNGramEmbedding",
    "HpuGatedResidual",
)
# Classes whose forward the HPU model never calls: they are captured through their entry-point
# methods (the spec's monkeypatch-and-wrap pattern, as used for HpuQsaIndexer.select).
METHOD_CLASSES = {
    "HpuQsaIndexer": ("select",),
    "HpuGatedResidual": ("mix", "combine_and_mix", "combine"),
    "HpuQwen4ExpNGramEmbedding": ("compute_ngram_ids",),  # the PLE layer calls the method, not the forward
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m tools.flash_next.compile_probe",
        description="Time the first (compiling) and second (replaying) HPU graph call of every "
        "Flash-Next component of the tiny model at each prompt shape.")
    parser.add_argument("--ckpt", required=True, help="tiny Flash-Next checkpoint directory")
    parser.add_argument("--device", choices=("cpu", "hpu"), default="cpu", help="time on the card (hpu) or the Mac")
    parser.add_argument("--tokens", default="128,512,2048", help="comma-separated prefill token counts")
    parser.add_argument("--out", required=True, help="output directory for probe.json")
    parser.add_argument("--skip-whole-model", action="store_true", help="do not time the whole model")
    return parser.parse_args(argv)


def parse_tokens(spec):
    counts = []
    for part in spec.split(","):
        if not part.strip():
            continue
        try:
            counts.append(int(part))
        except ValueError:
            raise SystemExit(f"compile_probe: --tokens needs comma-separated ints, got {spec!r}")
    if not counts or any(n < 1 for n in counts):
        raise SystemExit(f"compile_probe: --tokens needs at least one positive int, got {spec!r}")
    return counts


def detach_value(value):
    if hasattr(value, "detach") and hasattr(value, "shape"):  # torch.Tensor (and subclasses)
        return value.detach()
    return value


def detach_args(args):
    return tuple(detach_value(a) for a in args)


def detach_kwargs(kwargs):
    return {k: detach_value(v) for k, v in kwargs.items()}


def move_to_device(model, device, mamba_base, qsa_cls, qsa_sparse):
    """Move parameters/buffers with ``model.to`` and carry the hand-bound states by hand.

    The ``MambaBase`` state tuples (GDN, PLE) and the QSA indexer side caches are plain
    attributes, not buffers, so ``model.to()`` leaves them on the host. The int64 hash tables of
    the PLE n-gram embedding stay on the host by design (RUK-3101); nothing non-tensor is moved.
    """
    model.to(device)
    for module in model.modules():
        if isinstance(module, mamba_base):
            module.kv_cache = tuple(t.to(device) for t in module.kv_cache)
        if isinstance(module, qsa_cls) and module.qsa_mode is qsa_sparse:
            module.indexer.compressed_key_cache = module.indexer.compressed_key_cache.to(device)
            module.indexer.key_ring = module.indexer.key_ring.to(device)


def tensors_to_device(obj, device):
    """``.to(device)`` every tensor field of a SimpleNamespace metadata object or dict, in place."""
    items = vars(obj).items() if not isinstance(obj, dict) else obj.items()
    for key, value in list(items):
        if hasattr(value, "to") and hasattr(value, "shape"):
            if isinstance(obj, dict):
                obj[key] = value.to(device)
            else:
                setattr(obj, key, value.to(device))
    return obj


def build_targets(model, include_whole_model):
    """The probe's timed targets, in run order: model, then the tree (decoder layers all, others first-only)."""
    targets = []
    if include_whole_model:
        targets.append({
            "component": MODEL_CLASS,
            "module_name": MODEL_CLASS,
            "module": model,
            "kind": "forward"
        })
    seen = set()
    moe_class = None
    for name, module in model.named_modules():
        if not name:
            continue
        cls = type(module).__name__
        component = None
        kind = "forward"
        if cls == DECODER_LAYER_CLASS:
            component, kind = cls, "forward"
        elif cls == "HpuQsaIndexer" and SELECT_COMPONENT not in seen:
            component, kind = SELECT_COMPONENT, "select"
        elif cls in METHOD_CLASSES and cls not in seen:
            component, kind = cls, "method"
        elif cls in COMPONENT_CLASSES and cls not in seen:
            component = cls
        elif ("Moe" in cls or "MoE" in cls) and moe_class is None:
            component, moe_class = cls, cls  # the MoE block: the first class whose name contains Moe/MoE
        if component is None:
            continue
        seen.add(cls)
        seen.add(component)
        targets.append({"component": component, "module_name": name, "module": module, "kind": kind})
    return targets


def capture_inputs(model, targets, names, model_args, model_kwargs, vllm_config, md, patches):
    """Run one eager forward; record (args, kwargs) per target, keyed by (class name, module name).

    Forward targets get a forward pre-hook. ``HpuQsaIndexer.select``, ``HpuGatedResidual`` and
    ``HpuQwen4ExpNGramEmbedding`` are never called through ``forward`` (the HPU layers call their
    methods directly), so their entry-point methods are monkeypatched for the capture and restored
    afterwards; the first call of any entry method per module is the one captured.
    """
    from vllm.forward_context import set_forward_context

    captured = {}

    def record(key, args, kwargs, method=None):
        entry = captured.setdefault(key, {"args": detach_args(args), "kwargs": detach_kwargs(kwargs), "method": method})
        return entry

    handles = []
    for target in targets:
        if target["kind"] != "forward":
            continue

        def hook(module, hook_args, hook_kwargs, key=(target["component"], target["module_name"])):
            record(key, hook_args, hook_kwargs)

        handles.append(target["module"].register_forward_pre_hook(hook, with_kwargs=True))

    restored = []
    for owner, method, component in patches:
        original = getattr(owner, method)

        def wrapper(self, *args, _component=component, _method=method, _orig=original, **kwargs):
            record((_component, names[id(self)]), args, kwargs, method=_method)
            return _orig(self, *args, **kwargs)

        setattr(owner, method, wrapper)
        restored.append((owner, method, original))
    try:
        import torch
        with torch.no_grad(), set_forward_context(md, vllm_config):
            model(*model_args, **model_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        for owner, method, original in restored:
            setattr(owner, method, original)
    return captured


def make_select_module(torch):

    class SelectModule(torch.nn.Module):
        """Wraps ``indexer.select`` so the timed call goes through a module's ``forward``."""

        def __init__(self, indexer):
            super().__init__()
            self.indexer = indexer

        def forward(self, *args, **kwargs):
            return self.indexer.select(*args, **kwargs)

    return SelectModule


def make_method_module(torch):

    class MethodModule(torch.nn.Module):
        """Wraps one captured entry-point method (``HpuGatedResidual``, the n-gram embedding)."""

        def __init__(self, owner, method):
            super().__init__()
            self.owner = owner
            self.method = method

        def forward(self, *args, **kwargs):
            return getattr(self.owner, self.method)(*args, **kwargs)

    return MethodModule


def timed_call(run_callable, md, vllm_config, torch, set_forward_context, htorch):
    t0 = time.perf_counter()
    with torch.no_grad(), set_forward_context(md, vllm_config):
        run_callable()
        if htorch is not None:
            htorch.core.mark_step()
            htorch.hpu.synchronize()
    return time.perf_counter() - t0


def print_table(entries):
    print()
    print("| component | module | tokens | compile_s | replay_s | note |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for entry in sorted(entries, key=lambda e: -(e["compile_s"] or 0.0)):
        replay = "skipped" if entry["replay_s"] is None else f"{entry['replay_s']:.3f}"
        compile_text = "error" if entry["compile_s"] is None else f"{entry['compile_s']:.3f}"
        print(f"| {entry['component']} | {entry['module']} | {entry['tokens']} | {compile_text} | "
              f"{replay} | {entry['note'] or ''} |")


def run(args, token_counts):
    # Every model/HPU import happens here, after the lazy-mode check in main().
    import torch
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    from tests.unit_tests.models.flash_next.cpu_model import (
        cpu_model, eos_token_id, ple_kwargs, prefill_metadata, reset_cpu_states)
    from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual
    from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpNGramEmbedding
    from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer, HpuQwen4ExpQSAAttention, QsaMode

    htorch = None
    if args.device == "hpu":
        import habana_frameworks.torch as htorch  # noqa: F811  (the lazy-mode bridge)

    device = torch.device(args.device)
    SelectModule = make_select_module(torch)
    MethodModule = make_method_module(torch)
    patches = [(HpuQsaIndexer, method, SELECT_COMPONENT) for method in METHOD_CLASSES["HpuQsaIndexer"]]
    patches += [(HpuGatedResidual, method, "HpuGatedResidual") for method in METHOD_CLASSES["HpuGatedResidual"]]
    patches += [(HpuQwen4ExpNGramEmbedding, method, "HpuQwen4ExpNGramEmbedding")
                for method in METHOD_CLASSES["HpuQwen4ExpNGramEmbedding"]]

    entries = []
    with cpu_model(Path(args.ckpt), fp32=False, qsa_mode=QsaMode.SPARSE, max_model_len=MAX_MODEL_LEN) as (vllm_config,
                                                                                                          model):
        move_to_device(model, device, MambaBase, HpuQwen4ExpQSAAttention, QsaMode.SPARSE)
        eos = eos_token_id(vllm_config)
        names = {id(module): (name or MODEL_CLASS) for name, module in model.named_modules()}
        targets = build_targets(model, include_whole_model=not args.skip_whole_model)
        sparse_attns = [m.sparse_attn for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]

        for tokens in token_counts:
            reset_cpu_states(model)
            for att in sparse_attns:
                att.reset()
                att.slot = 1
            md = tensors_to_device(prefill_metadata(tokens, 1), device)
            pkw = tensors_to_device(ple_kwargs(tokens, eos), device)
            ids = torch.randint(5, 1000, (tokens,), device=device)
            print(f"compile_probe: capturing {tokens} tokens ...", flush=True)
            captured = capture_inputs(
                model,
                targets,
                names,
                (),
                {
                    "input_ids": ids,
                    "positions": torch.arange(tokens, device=device),
                    **pkw
                },
                vllm_config,
                md,
                patches,
            )
            for target in targets:
                key = (target["component"], target["module_name"])
                if key not in captured:
                    print(f"compile_probe: WARNING {key} captured no inputs at {tokens} tokens, skipped", flush=True)
                    continue
                cap = captured[key]
                # Always graph-wrap a fresh thin module: wrap_in_hpu_graph patches its argument's forward in
                # place, so wrapping the real module would leave the next shape's eager capture running graphs.
                component = MethodModule(target["module"], "__call__")
                if target["kind"] == "select":
                    component = SelectModule(target["module"])
                elif target["kind"] == "method":
                    component = MethodModule(target["module"], cap["method"])
                if htorch is not None:
                    component = htorch.hpu.wrap_in_hpu_graph(component, disable_tensor_cache=True)
                run_callable = functools.partial(component, *cap["args"], **cap["kwargs"])
                note = cap["method"] + "()" if target["kind"] == "method" else None
                try:
                    for att in sparse_attns:  # the stand-in appends K/V per call: keep the captured length
                        att.reset()
                    compile_s = timed_call(run_callable, md, vllm_config, torch, set_forward_context, htorch)
                except Exception as exc:  # one component failing to capture must not lose the rest of the table
                    print(f"compile_probe: ERROR {key} at {tokens} tokens: {exc!r}", flush=True)
                    entries.append({"component": target["component"], "module": target["module_name"],
                                    "tokens": tokens, "compile_s": None, "replay_s": None,
                                    "note": f"error: {exc!r}"[:300]})
                    continue
                replay_s = None
                if compile_s > FIRST_CALL_BUDGET_S:
                    note = f"first call took {compile_s:.1f} s (> {FIRST_CALL_BUDGET_S:.0f} s); replay skipped" + (
                        f" [{cap['method']}()]" if target["kind"] == "method" else "")
                else:
                    for att in sparse_attns:
                        att.reset()
                    replay_s = timed_call(run_callable, md, vllm_config, torch, set_forward_context, htorch)
                entries.append({
                    "component": target["component"],
                    "module": target["module_name"],
                    "tokens": tokens,
                    "compile_s": compile_s,
                    "replay_s": replay_s,
                    "note": note,
                })
                replay_text = "skipped" if replay_s is None else f"{replay_s:.3f} s"
                print(f"compile_probe: [{tokens:>5}] {target['component']:<28} {target['module_name']:<48} "
                      f"compile {compile_s:.3f} s  replay {replay_text}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"\ncompile_probe: device={args.device} ckpt={args.ckpt} tokens={','.join(map(str, token_counts))} "
          f"-> {out_dir / 'probe.json'}")
    print_table(entries)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.device == "hpu" and os.environ.get(LAZY_MODE_FLAG) != "1":
        print(LAZY_MODE_MESSAGE, file=sys.stderr)
        return 2
    token_counts = parse_tokens(args.tokens)
    # Match the CPU harness's env: the plugin is registered by hand, and TMPDIR must not be sandboxed.
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("TMPDIR", "/tmp")
    return run(args, token_counts)


if __name__ == "__main__":
    raise SystemExit(main())
