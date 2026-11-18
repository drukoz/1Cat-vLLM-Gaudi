# SPDX-License-Identifier: Apache-2.0
"""Rank functions for the GDN core-decode TP parity tests (T3c of track RUK-2812), on the gloo CPU
ranks of ``spikes/cpu-tp``.

Plain module (no pytest import): ``run_ranks`` spawns children that re-import this module by its
module path (``tests.unit_tests.models.flash_next.gdn_core_ranks``), so every rank function lives
here and returns numpy arrays / plain Python values only (a torch tensor over the spawn queue costs
~30 s per rank on macOS, see ``spikes/cpu-tp/README.md``). The harness directory name has a
hyphen, so it is added to ``sys.path`` here and imported as ``harness``.

Follows ``tp_ranks.prefill_rank`` (RUK-2740): register the plugin, build the engine config at
``tensor_parallel_size=tp``, load the checkpoint with ``get_model``, convert to fp32 and bind the
CPU states; only the first GDN layer runs, called directly under ``set_forward_context``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point
os.environ.setdefault("TMPDIR", "/tmp")  # the gloo rendezvous file must not sit on a sandboxed TMPDIR

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[4] / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import harness  # noqa: E402,F401  (imported so the sys.path insert above mirrors tp_ranks.py)

OP_OF_MODE = {1: "gdn_core_decode", 2: "gdn_recurrent_decode"}  # GdnCoreDecodeMode.FUSED / RECURRENCE
PREFILL_LEN = 12
DECODE_STEPS = 6  # steps 0..4 load and store slot 1; step 5 loads slot 1 and stores slot 2


def gdn_decode_rank(rank: int, ckpt_dir: str, tp: int, modes: tuple[int, ...]) -> dict:
    """Run the first GDN layer of the tiny model sharded at ``tp`` through a prefill and 6 decode steps per mode.

    For each mode (an int ``GdnCoreDecodeMode`` value): zero the layer's states, prefill rows 0..11 of the
    fixed hidden input into slot 1, decode rows 12..16 in slot 1, then decode row 17 loading slot 1 and storing
    slot 2. Returns ``{mode: {"rows": (6, 128) float32, "ssm2": (48 // tp, 16, 16) float32, "conv2": (3, C) float32,
    "slot1_unchanged": bool, "calls": {op name: int}, "ok": bool | None}}``.
    """
    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    import vllm_gaudi.models.qwen3_5 as qwen3_5
    import vllm_gaudi.models.qwen4_exp as fn
    from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode
    from tests.unit_tests.models.flash_next import cpu_model as cm
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

    fn.register()
    vc = EngineArgs(model=ckpt_dir, tokenizer=ckpt_dir, skip_tokenizer_init=True, dtype="bfloat16",
                    enable_prefix_caching=False, max_model_len=512, max_num_seqs=4, enforce_eager=True,
                    language_model_only=True, tensor_parallel_size=tp,
                    distributed_executor_backend="mp").create_engine_config()
    calls = {name: 0 for name in OP_OF_MODE.values() if hasattr(qwen3_5, name)}
    for name in calls:
        real = getattr(qwen3_5, name)

        def counting(*args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            return _real(*args, **kwargs)

        setattr(qwen3_5, name, counting)
    results = {}
    with set_current_vllm_config(vc):
        model = get_model(vllm_config=vc)
        cm.to_fp32(model)
        cm.bind_cpu_states(model, 4, True)
        vc.cache_config.enable_prefix_caching = True  # keeps load != store (see enable_slot_reuse)
        layer = [m for m in model.modules() if isinstance(m, HPUGatedDeltaNetAttention)][0]
        hidden = torch.randn(PREFILL_LEN + DECODE_STEPS, 128, generator=torch.Generator().manual_seed(0))
        for mode in modes:
            for state in layer.kv_cache:
                state.zero_()
            layer.gdn_core_decode_mode = GdnCoreDecodeMode(mode)
            layer._gdn_core_ok = None
            for name in calls:
                calls[name] = 0
            rows = []
            with torch.no_grad():
                with set_forward_context(cm.prefill_metadata(PREFILL_LEN, 1), vc):
                    layer(hidden[:PREFILL_LEN])
                for step in range(DECODE_STEPS):
                    pos = PREFILL_LEN + step
                    if step == DECODE_STEPS - 1:
                        slot1_before = [state[1].clone() for state in layer.kv_cache]
                        meta = cm.decode_metadata(1, pos, load_slot=1, store_slot=2)
                    else:
                        meta = cm.decode_metadata(1, pos)
                    with set_forward_context(meta, vc):
                        rows.append(layer(hidden[pos:pos + 1]).float().numpy())
            results[mode] = {
                "rows": np.concatenate(rows).astype(np.float32),
                "ssm2": layer.kv_cache[1][2].float().numpy(),
                "conv2": layer.kv_cache[0][2].float().numpy(),
                "slot1_unchanged": all(bool(torch.equal(state[1], before))
                                       for state, before in zip(layer.kv_cache, slot1_before)),
                "calls": dict(calls),
                "ok": layer._gdn_core_ok,
            }
    return results
