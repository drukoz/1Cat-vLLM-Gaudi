"""CTO probe (RUK-2740, 2026-11-17): whole-model prefill on N gloo CPU ranks through spikes/cpu-tp/harness.py.
Saves rank 0's last-token hidden states to <ckpt parent>/h_tp<N>_<prec>.npy for comparison with N=1.
Usage: PYTHONPATH=<worktree> python tp_probe.py <ckpt_dir> <N> fp32|bf16"""
import json, os, sys
os.environ.setdefault("VLLM_PLUGINS", ""); os.environ["TMPDIR"]="/tmp"
WT = os.environ["PYTHONPATH"]
sys.path.insert(0, os.path.join(WT, "spikes", "cpu-tp"))
from pathlib import Path

def rank_fn(rank, ck, tp, fp32):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    import vllm_gaudi.models.qwen4_exp as fn
    from tests.unit_tests.models.flash_next import cpu_model as cm
    fn.register()
    vc = EngineArgs(model=ck, tokenizer=ck, skip_tokenizer_init=True, dtype="bfloat16", enable_prefix_caching=False,
                    max_model_len=512, max_num_seqs=4, enforce_eager=True, language_model_only=True,
                    tensor_parallel_size=tp, distributed_executor_backend="mp").create_engine_config()
    ref = json.loads((Path(ck)/"reference.json").read_text())
    out = []
    with set_current_vllm_config(vc):
        from vllm.model_executor.model_loader import get_model
        model = get_model(vllm_config=vc)
        if fp32: cm.to_fp32(model)
        cm.bind_cpu_states(model, 4, fp32); cm.use_sdpa_attention(model)
        eos = cm.eos_token_id(vc)
        for p in ref["prompts"]:
            seq = p["prompt_ids"]; n = len(seq); cm.reset_cpu_states(model)
            with torch.no_grad(), set_forward_context(cm.prefill_metadata(n), vc):
                h = model(input_ids=torch.tensor(seq), positions=torch.arange(n), **cm.ple_kwargs(n, eos))
            out.append(h[-1].float().numpy())
        W = model.language_model.lm_head.weight.float().numpy()
    return out, W.shape

if __name__ == "__main__":
    from harness import run_ranks
    ck, tp, prec = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    import time; t=time.time()
    res = run_ranks(tp, rank_fn, ck, tp, prec == "fp32", timeout_s=500)
    import numpy as np
    print("RESULT tp", tp, "secs", round(time.time()-t), "lm_head_shard", res[0][1])
    hs = [r[0] for r in res]
    for i in range(len(hs[0])):
        print("RESULT prompt", i, "max rank spread", max(float(np.abs(hs[k][i]-hs[0][i]).max()) for k in range(tp)))
    np.save(os.path.join(os.path.dirname(ck), f"h_tp{tp}_{prec}.npy"), np.stack(hs[0]))
