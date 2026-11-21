"""CTO probe (RUK-2740, 2026-11-17): TP1 prefill parity of the whole model with the real HPU PLE layer (T4b).
Usage: PYTHONPATH=<worktree> python prefill_probe.py <ckpt_dir> fp32|bf16  (builds the tiny checkpoint if missing)."""
import json, os, sys
os.environ.setdefault("VLLM_PLUGINS", ""); os.environ["TMPDIR"]="/tmp"
from pathlib import Path
import torch
from vllm.forward_context import set_forward_context
from tests.unit_tests.models.flash_next.cpu_model import *
ck = Path(sys.argv[1]); fp32 = sys.argv[2] == "fp32"
if not (ck/"config.json").exists(): build_tiny_checkpoint(ck)
ref = json.loads((ck/"reference.json").read_text())
with cpu_model(ck, fp32=fp32) as (vc, model):
    eos = eos_token_id(vc)
    def run(seq):
        reset_cpu_states(model); n=len(seq)
        with torch.no_grad(), set_forward_context(prefill_metadata(n), vc):
            h = model(input_ids=torch.tensor(seq), positions=torch.arange(n), **ple_kwargs(n, eos))
        return fp32_logits(model, h)
    for p in ref["prompts"]:
        lg = run(p["prompt_ids"])[-1]; r = p["last_prefill_top5"]
        d = (lg[r["ids"]] - torch.tensor(r["logits"])).abs().max().item()
        seq, gen = list(p["prompt_ids"]), []
        for _ in range(ref["max_new_tokens"]):
            t = int(run(seq)[-1].argmax()); gen.append(t); seq.append(t)
        ag = sum(a==b for a,b in zip(gen,p["generated_ids"]))
        first = next((i for i,(a,b) in enumerate(zip(gen,p["generated_ids"])) if a!=b), 16)
        print(fp32, len(p["prompt_ids"]), int(lg.argmax())==r["ids"][0], f"{d:.2e}", ag, first, "top5", lg[r["ids"]].topk(5).indices.tolist() if False else "")
