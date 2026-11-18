# SPDX-License-Identifier: Apache-2.0
"""CTO probe for TR (RUK-3067): the GDN verification path of one layer on 1/2/4/8 gloo CPU ranks.

Run from the repository root (``PYTHONPATH=$PWD``):

    $PY spikes/gdn-core-verify/tp_probe.py

Builds the 8-head tiny checkpoint (``tiny_tp8_config.json``), takes the first
``HPUGatedDeltaNetAttention`` of the model sharded at ``tp`` and calls it directly, as
``tests/unit_tests/models/flash_next/gdn_core_ranks.py`` does for phase 1, through:

1. a 12-row prefill into slot 1 (conv pool ``[S, 3 + K, C]`` bound by hand, K = 3),
2. verification step A: lattice ``[1, 2, 3, 4]`` (load == store), ``num_accepted = [1]``, rows 12..15,
3. verification step B: the same lattice, ``num_accepted = [3]``, rows 16..19,
4. a plain step after it: 1-D load = store = ``[3]``, ``num_accepted = [2]``, row 20,

once with ``GdnCoreVerifyMode.OFF`` (torch MTP path) and once with ``FUSED``, counting calls of
``qwen3_5.gdn_core_verify``. Prints the measured gaps.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spikes" / "cpu-tp"))

K = 3
T = 1 + K
LATTICE = [1, 2, 3, 4]
PREFILL_LEN = 12
NUM_SLOTS = 8


def verify_rank(rank: int, ckpt_dir: str, tp: int) -> dict:
    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    import vllm_gaudi.models.qwen3_5 as qwen3_5
    import vllm_gaudi.models.qwen4_exp as fn
    from flashinfer_gaudi.gdn_core_verify import GdnCoreVerifyMode
    from tests.unit_tests.models.flash_next import cpu_model as cm
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

    fn.register()
    vc = EngineArgs(model=ckpt_dir, tokenizer=ckpt_dir, skip_tokenizer_init=True, dtype="bfloat16",
                    enable_prefix_caching=False, max_model_len=512, max_num_seqs=4, enforce_eager=True,
                    language_model_only=True, tensor_parallel_size=tp,
                    distributed_executor_backend="mp").create_engine_config()
    calls = {"n": 0, "ts": []}
    real = qwen3_5.gdn_core_verify

    def counting(*args, **kwargs):
        calls["n"] += 1
        calls["ts"].append(int(args[11].shape[1]))
        return real(*args, **kwargs)

    qwen3_5.gdn_core_verify = counting
    out = {}
    with set_current_vllm_config(vc):
        model = get_model(vllm_config=vc)
        cm.to_fp32(model)
        cm.bind_cpu_states(model, NUM_SLOTS, True)
        layer = [m for m in model.modules() if isinstance(m, HPUGatedDeltaNetAttention)][0]
        layer.cache_group_idx = torch.tensor(0, dtype=torch.long)  # the runner sets it for a 3-D lattice
        conv, ssm = layer.kv_cache
        layer.kv_cache = (torch.zeros(NUM_SLOTS, 3 + K, conv.shape[-1]), ssm)
        hidden = torch.randn(PREFILL_LEN + 2 * T + 1, 128, generator=torch.Generator().manual_seed(0))
        lattice = torch.tensor([[LATTICE]], dtype=torch.int64)
        for mode in (GdnCoreVerifyMode.OFF, GdnCoreVerifyMode.FUSED):
            for state in layer.kv_cache:
                state.zero_()
            layer.gdn_core_verify_mode = mode
            layer._gdn_core_verify_ok = None
            calls["n"], calls["ts"] = 0, []
            rows = []
            with torch.no_grad():
                with set_forward_context(cm.prefill_metadata(PREFILL_LEN, 1), vc):
                    layer(hidden[:PREFILL_LEN])
                pos = PREFILL_LEN
                for acc in (1, 3):
                    md = cm.decode_metadata(LATTICE[0], pos, num_tokens=T)
                    md.load_indices_tensor = lattice
                    md.store_indices_tensor = lattice.clone()
                    md.lattice_store_differs = False
                    md.num_accepted_tokens = torch.tensor([acc], dtype=torch.int32)
                    with set_forward_context(md, vc):
                        rows.append(layer(hidden[pos:pos + T]).float().numpy())
                    pos += T
                md = cm.decode_metadata(3, pos)
                md.num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
                with set_forward_context(md, vc):
                    rows.append(layer(hidden[pos:pos + 1]).float().numpy())
            out[int(mode)] = {
                "rows": np.concatenate(rows).astype(np.float32),
                "ssm": layer.kv_cache[1][1:5].float().numpy().copy(),  # .numpy() of an fp32 pool is a view
                "conv": layer.kv_cache[0][1:5].float().numpy().copy(),
                "calls": calls["n"],
                "ts": list(calls["ts"]),
                "ok": layer._gdn_core_verify_ok,
            }
    return out


def main() -> None:
    import numpy as np

    from harness import run_ranks
    from tests.unit_tests.models.flash_next.cpu_model import build_tiny_checkpoint

    fixture = ROOT / "tests/unit_tests/models/flash_next/tiny_tp8_config.json"
    ckpt = build_tiny_checkpoint(Path(tempfile.mkdtemp(dir="/tmp")), fixture=fixture)
    res = {tp: run_ranks(tp, verify_rank, str(ckpt), tp, timeout_s=300) for tp in (1, 2, 4, 8)}
    tp1 = res[1][0][0]
    for tp, ranks in res.items():
        for r, d in enumerate(ranks):
            off, fu = d[0], d[1]
            print(f"tp{tp} r{r}: calls off={off['calls']} fused={fu['calls']} ts={fu['ts']} ok={fu['ok']} "
                  f"rows fused-off={np.abs(fu['rows'] - off['rows']).max():.2e} "
                  f"ssm={np.abs(fu['ssm'] - off['ssm']).max():.2e} "
                  f"conv={np.abs(fu['conv'] - off['conv']).max():.2e} "
                  f"rows vs r0={np.abs(fu['rows'] - ranks[0][1]['rows']).max():.2e} "
                  f"off rows nonzero={bool(np.abs(off['rows']).max() > 0)}")
        for mode in (0, 1):
            full = np.concatenate([d[mode]["ssm"] for d in ranks], axis=1)
            print(f"tp{tp} mode{mode}: r0 rows vs TP1 OFF {np.abs(ranks[0][mode]['rows'] - tp1['rows']).max():.2e}, "
                  f"concat SSM vs TP1 OFF {np.abs(full - tp1['ssm']).max():.2e} shape {full.shape}")


if __name__ == "__main__":
    main()
