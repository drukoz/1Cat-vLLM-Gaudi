# SPDX-License-Identifier: Apache-2.0
"""Rank functions for the Flash-Next TP parity tests (M0-T7d), on the gloo CPU ranks of ``spikes/cpu-tp``.

Plain module (no pytest import): ``run_ranks`` spawns children that re-import this module by its
module path (``tests.unit_tests.models.flash_next.tp_ranks``), so every rank function lives here
and returns numpy arrays / plain Python values only (a torch tensor over the spawn queue costs
~30 s per rank on macOS, see ``spikes/cpu-tp/README.md``). The harness directory name has a
hyphen, so it is added to ``sys.path`` here and imported as ``harness``.

Derived from ``spikes/flash-next-parity/tp_probe.py`` (RUK-2740), whose ``rank_fn`` builds the
model inside each rank through vLLM's real loader at ``tensor_parallel_size=tp``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point
os.environ.setdefault("TMPDIR", "/tmp")  # the gloo rendezvous file must not sit on a sandboxed TMPDIR

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[4] / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import harness  # noqa: E402


def prefill_rank(rank: int, ckpt_dir: str, tp: int, fp32: bool) -> dict:
    """Build the tiny Flash-Next model sharded at ``tp`` on this rank and prefill the reference prompts.

    Runs on a rank of ``harness.run_ranks(tp, prefill_rank, ckpt_dir, tp, fp32)`` (gloo distributed
    state already initialized). Registers the plugin, builds the engine config
    (``tensor_parallel_size=tp``, ``distributed_executor_backend="mp"``), loads the checkpoint with
    ``get_model``, converts to fp32 when asked, binds the CPU states and the SDPA attention, then
    prefills every prompt of the checkpoint's ``reference.json`` with the collective recorder on.

    Returns ``{"hidden": np.ndarray (P, 128) float32, one row per reference prompt (the last-token
    hidden state of each prompt, full on every rank), "collectives": list of
    ``(method, input_shape, input_dtype)`` recorded on the prefills, "top5_logits": the logits at
    each prompt's reference top-5 ids as ``list[list[float]]`` when ``tp == 1``, else ``None``
    (``lm_head`` is vocab-sharded at TP>1)}``.
    """
    import json

    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    import vllm_gaudi.models.qwen4_exp as fn
    from tests.unit_tests.models.flash_next import cpu_model as cm

    fn.register()
    vc = EngineArgs(model=ckpt_dir,
                    tokenizer=ckpt_dir,
                    skip_tokenizer_init=True,
                    dtype="bfloat16",
                    enable_prefix_caching=False,
                    max_model_len=512,
                    max_num_seqs=4,
                    enforce_eager=True,
                    language_model_only=True,
                    tensor_parallel_size=tp,
                    distributed_executor_backend="mp").create_engine_config()
    reference = json.loads((Path(ckpt_dir) / "reference.json").read_text(encoding="utf-8"))

    hidden = []
    top5_logits = []
    with set_current_vllm_config(vc):
        model = get_model(vllm_config=vc)
        if fp32:
            cm.to_fp32(model)
        cm.bind_cpu_states(model, 4, fp32)
        cm.use_sdpa_attention(model)
        eos = cm.eos_token_id(vc)
        with harness.record_collectives() as recorded:
            for prompt in reference["prompts"]:
                seq = prompt["prompt_ids"]
                n = len(seq)
                cm.reset_cpu_states(model)
                with torch.no_grad(), set_forward_context(cm.prefill_metadata(n), vc):
                    h = model(input_ids=torch.tensor(seq), positions=torch.arange(n), **cm.ple_kwargs(n, eos))
                hidden.append(h[-1].float().numpy())
                if tp == 1:
                    logits = cm.fp32_logits(model, h[-1:])  # (1, 248320) fp32, full lm_head
                    ids = prompt["last_prefill_top5"]["ids"]
                    top5_logits.append([float(v) for v in logits[0][ids]])
    return {
        "hidden": np.stack(hidden).astype(np.float32),
        "collectives": list(recorded),
        "top5_logits": top5_logits if tp == 1 else None,
    }
