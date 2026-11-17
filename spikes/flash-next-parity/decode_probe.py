"""CTO probe (RUK-2740, 2026-11-17): TP1 decode-path parity: prefill, then 15 single-token decode steps with GDN,
PLE and attention state carried; teacher-forced on HF generated_ids. Prints greedy agreement, the last decode step's
max |dlogit| against a fresh prefill of the same sequence, and a negative control (GDN/PLE state wiped before the step).
Usage: PYTHONPATH=<worktree> python decode_probe.py <ckpt_dir> fp32|bf16"""
import json, os, sys
os.environ.setdefault("VLLM_PLUGINS", ""); os.environ["TMPDIR"]="/tmp"
from pathlib import Path
from types import SimpleNamespace
import torch, torch.nn.functional as F
from vllm.forward_context import set_forward_context
from tests.unit_tests.models.flash_next.cpu_model import *
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention
NEG = True
ck = Path(sys.argv[1]); fp32 = sys.argv[2] == "fp32"
ref = json.loads((ck/"reference.json").read_text())

class CachedSdpa(torch.nn.Module):
    def __init__(s, a):
        super().__init__(); s.h, s.kv, s.d, s.sc = a.num_heads, a.num_kv_heads, a.head_dim, a.scaling; s.reset()
    def reset(s): s.K = None; s.V = None
    def forward(s, q, k, v):
        n = q.shape[0]
        s.K = k if s.K is None else torch.cat([s.K, k]); s.V = v if s.V is None else torch.cat([s.V, v])
        L = s.K.shape[0]
        qh = q.view(n, s.h, s.d).transpose(0, 1)
        kh = s.K.view(L, s.kv, s.d).transpose(0, 1).repeat_interleave(s.h // s.kv, 0)
        vh = s.V.view(L, s.kv, s.d).transpose(0, 1).repeat_interleave(s.h // s.kv, 0)
        mask = torch.ones(n, L, dtype=torch.bool).tril(L - n)
        return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=s.sc).transpose(0, 1).reshape(n, -1)

def decode_md(slot=1):
    return SimpleNamespace(is_prompt=False, load_indices_tensor=torch.tensor([slot]), store_indices_tensor=torch.tensor([slot]),
        query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32), has_initial_states_p=None, padding_mask_flat=None,
        direct_gdn_state=False, num_accepted_tokens=None, dflash_full_query=False)

with cpu_model(ck, fp32=fp32) as (vc, model):
    eos = eos_token_id(vc)
    atts = []
    for m in model.modules():
        if isinstance(m, HpuQwen4ExpQSAAttention): m.attn = CachedSdpa(m); atts.append(m.attn)
    for p in ref["prompts"]:
        reset_cpu_states(model); [a.reset() for a in atts]
        seq = list(p["prompt_ids"]); n = len(seq)
        with torch.no_grad(), set_forward_context(prefill_metadata(n), vc):
            h = model(input_ids=torch.tensor(seq), positions=torch.arange(n), **ple_kwargs(n, eos))
        lg = fp32_logits(model, h)[-1]; gen = []; gaps = []
        for i in range(ref["max_new_tokens"]):
            t = int(lg.argmax()); gen.append(t)
            # teacher-force HF's token so later steps stay comparable
            tf = p["generated_ids"][i]; seq.append(tf)
            if i == ref["max_new_tokens"] - 1: break
            pos = len(seq) - 1
            ctx = (seq[pos-2] if pos >= 2 else eos, seq[pos-1] if pos >= 1 else eos)
            with torch.no_grad(), set_forward_context(decode_md(), vc):
                h = model(input_ids=torch.tensor([tf]), positions=torch.tensor([pos]),
                          query_start_loc=torch.tensor([0, 1], dtype=torch.int32), ngram_context=torch.tensor([ctx], dtype=torch.int32))
            lg = fp32_logits(model, h)[-1]
        ag = sum(a == b for a, b in zip(gen, p["generated_ids"]))
        # compare last decode logits against a fresh prefill of the same sequence (seq minus the last appended token)
        full = seq[:-1]; m_ = len(full)
        reset_cpu_states(model); [a.reset() for a in atts]
        with torch.no_grad(), set_forward_context(prefill_metadata(m_), vc):
            hf = model(input_ids=torch.tensor(full), positions=torch.arange(m_), **ple_kwargs(m_, eos))
        pf = fp32_logits(model, hf)[-1]
        print("RESULT", fp32, n, ag, "/16", "decode_vs_prefill", f"{(pf-lg).abs().max().item():.2e}")
        if NEG:
            # negative control: decode the last token with GDN+PLE states wiped
            reset_cpu_states(model); [a.reset() for a in atts]
            with torch.no_grad(), set_forward_context(prefill_metadata(m_-1), vc):
                model(input_ids=torch.tensor(full[:-1]), positions=torch.arange(m_-1), **ple_kwargs(m_-1, eos))
            reset_cpu_states(model)
            pos = m_-1; ctx=(full[pos-2] if pos>=2 else eos, full[pos-1] if pos>=1 else eos)
            with torch.no_grad(), set_forward_context(decode_md(), vc):
                hn = model(input_ids=torch.tensor([full[-1]]), positions=torch.tensor([pos]),
                          query_start_loc=torch.tensor([0, 1], dtype=torch.int32), ngram_context=torch.tensor([ctx], dtype=torch.int32))
            print("NEG", n, f"{(fp32_logits(model, hn)[-1]-pf).abs().max().item():.2e}")
