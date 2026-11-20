# SPDX-License-Identifier: Apache-2.0
"""Rank function for the MoE glue TP test, on the gloo CPU ranks of ``spikes/cpu-tp``.

Plain module (no pytest import): ``run_ranks`` spawns children that re-import this module by its module
path, and a rank returns numpy arrays and plain Python values only (see ``tp_ranks.py``).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("TMPDIR", "/tmp")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[4] / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import harness  # noqa: E402

PREFILL_TOKENS = (1, 33)  # a decode-sized call and a prompt
DECODE_ROWS = 32  # a full decode batch through one MoE block


def moe_rank(rank: int, ckpt_dir: str, tp: int, expert_parallel: bool, flag: str) -> dict:
    """Build the tiny model sharded at ``tp`` on this rank, on the plugin's MoE path, with VLLM_HPU_MOE_GLUE=flag.

    The vendor ops are the CPU stand-ins of ``moe_block.py`` and the fused glue ops run the torch oracle
    (``tests/unit_tests/hc_fused/moe_ops.py``).  Prefills two reference-sized sequences with the collective
    recorder on, then runs every MoE block alone on ``DECODE_ROWS`` rows.

    Returns the last-token hidden states ``(len(PREFILL_TOKENS), H)`` and the block outputs (after their
    all-reduce) as float32 arrays of the bf16 values, the collectives, the expert range of this rank, what
    the vendor op was given, the glue state of every block and every warning line of the MoE module.
    """
    os.environ["VLLM_HPU_MOE_GLUE"] = flag

    from tests.unit_tests.models.flash_next import moe_block

    moe_block.use_plugin_fast_path()

    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    import vllm_gaudi.models.qwen4_exp as fn
    import vllm_gaudi.ops.hpu_fused_moe as fused_moe
    from tests.unit_tests.hc_fused import moe_ops  # noqa: F401  the CPU stand-in: the oracle as the CPU kernels
    from tests.unit_tests.models.flash_next import cpu_model as cm

    lines: list[str] = []
    fused_moe.logger.warning = lambda message, *args: lines.append(message % args)

    fn.register()
    kwargs = dict(model=ckpt_dir, tokenizer=ckpt_dir, skip_tokenizer_init=True, dtype="bfloat16",
                  enable_prefix_caching=False, max_model_len=512, max_num_seqs=4, enforce_eager=True,
                  language_model_only=True, tensor_parallel_size=tp, distributed_executor_backend="mp")
    if expert_parallel:
        kwargs["enable_expert_parallel"] = True
    vc = EngineArgs(**kwargs).create_engine_config()
    with set_current_vllm_config(vc):
        model = get_model(vllm_config=vc)
        cm.bind_cpu_states(model, 4, False)
        cm.use_sdpa_attention(model)
        eos = cm.eos_token_id(vc)
        blocks = moe_block.moe_blocks(model)
        runner = blocks[0].experts
        routed = runner.routed_experts
        hidden_size = model.language_model.model.config.hidden_size

        hidden = []
        moe_block.VENDOR_CALLS.clear()
        with harness.record_collectives() as recorded, moe_block.OpRecorder() as recorder:
            for tokens in PREFILL_TOKENS:
                seq = [1000 + i for i in range(tokens)]
                cm.reset_cpu_states(model)
                with torch.no_grad(), set_forward_context(cm.prefill_metadata(tokens), vc):
                    out = model(input_ids=torch.tensor(seq),
                                positions=torch.arange(tokens),
                                **cm.ple_kwargs(tokens, eos))
                hidden.append(out[-1].float().numpy())
        calls = list(moe_block.VENDOR_CALLS)

        block_out = []
        g = torch.Generator().manual_seed(11)  # the same input on every rank
        x = (torch.randn(DECODE_ROWS, hidden_size, generator=g) * 2.0).to(torch.bfloat16)
        with torch.no_grad():
            for block in blocks:
                block_out.append(block(x).float().numpy())

        states = [getattr(block.experts, "_hpu_moe_glue", None) for block in blocks]
        ids = torch.cat([call["expert_routing_table"].reshape(-1) for call in calls])
    return {
        "hidden": np.stack(hidden).astype(np.float32),
        "block_out": np.stack(block_out).astype(np.float32),
        "collectives": list(recorded),
        "tp_size": int(runner.moe_config.tp_size),
        "ep_size": int(runner.moe_config.ep_size),
        "ep_rank": int(runner.moe_config.ep_rank),
        "local_experts": int(routed.local_num_experts),
        "experts_min_max": sorted({(int(call["experts_min"]), int(call["experts_max"])) for call in calls}),
        "shared_gate_up_shape": tuple(blocks[0].shared_expert.gate_up_proj.weight.shape),
        "routed_ids_min_max": (int(ids.min()), int(ids.max())),
        "ids_dtype": str(calls[0]["expert_routing_table"].dtype),
        "vendor_calls": len(calls),
        "glue_ops": sorted({name for name in recorder.names if name.startswith("flashinfer_gaudi.moe_")}),
        "glue_ok": [None if state is None else state.ok for state in states],
        "merged_gate_shape": None if states[0] is None else tuple(runner._hpu_moe_glue_gate_weight.shape),
        "lines": lines,
    }
