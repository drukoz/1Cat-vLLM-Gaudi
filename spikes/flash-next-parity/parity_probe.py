"""CTO probe (RUK-2740, 2026-11-17): whole Flash-Next HPU model class on CPU vs HF reference.json.
Usage: PYTHONPATH=<worktree> python parity_probe.py <ckpt_dir> [fp32]
The PLE forward below is a probe stand-in for T4b (single-sequence prefill, zero conv state)."""
import json, os, sys, tempfile
os.environ.setdefault("VLLM_PLUGINS", "")
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
import vllm_gaudi.models.qwen4_exp as fn
import vllm.models.qwen4_exp.nvidia.model as um
import vllm.models.qwen4_exp.nvidia.ple_layer as pl
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention
from vllm_gaudi.models.qwen4_exp.ple_ops import ple_forward_torch

um.GatedResidual = HpuGatedResidual                 # seam 1 (T7 wires it in apply_flash_next_patches)
um.Qwen4ExpQSAAttention = HpuQwen4ExpQSAAttention   # seam 2

def ple_forward(self, hidden_states, input_ids, query_start_loc, ngram_context):   # seam 3 stand-in (T4b)
    ids = self.ple_embedding.compute_ngram_ids(input_ids.reshape(-1), query_start_loc, ngram_context)
    emb = self.ple_embedding.ngram_embedding(ids).flatten(-2)
    kv, _ = self.kv_proj(emb)
    key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
    state = torch.zeros(self.conv_state_len, self.hc_hidden_size, dtype=hidden_states.dtype)
    out, _ = ple_forward_torch(key, value, hidden_states, self.norm_key.weight, self.norm_query.weight,
                               self.norm_conv.weight, self.norm_key.eps, self.conv1d.weight, state,
                               self.short_conv_dilation)
    return out
pl.Qwen4ExpPLELayer.forward = ple_forward
fn.register()

class Sdpa(torch.nn.Module):   # stands in for the HPU paged attention backend (as in test_qsa_dense.py)
    def __init__(s, a):
        super().__init__(); s.h, s.kv, s.d, s.sc = a.num_heads, a.num_kv_heads, a.head_dim, a.scaling
    def forward(s, q, k, v):
        n = q.shape[0]
        qh = q.view(n, s.h, s.d).transpose(0, 1)
        kh = k.view(n, s.kv, s.d).transpose(0, 1).repeat_interleave(s.h // s.kv, 0)
        vh = v.view(n, s.kv, s.d).transpose(0, 1).repeat_interleave(s.h // s.kv, 0)
        return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=s.sc).transpose(0, 1).reshape(n, -1)

ckpt = sys.argv[1]; fp32 = len(sys.argv) > 2
vc = EngineArgs(model=ckpt, tokenizer=ckpt, skip_tokenizer_init=True, dtype="bfloat16", enable_prefix_caching=False,
                max_model_len=512, max_num_seqs=4, enforce_eager=True, language_model_only=True).create_engine_config()
init_file = tempfile.mkstemp(dir="/tmp")[1]
with set_current_vllm_config(vc):
    init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{init_file}", local_rank=0, backend="gloo")
    initialize_model_parallel(1, 1)
    from vllm.model_executor.model_loader import get_model
    model = get_model(vllm_config=vc)
    if os.environ.get("SPECS"):
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        for name, layer in sorted(get_layers_from_vllm_config(vc, AttentionLayerBase).items()):
            spec = layer.get_kv_cache_spec(vc)
            print("SPEC", name, type(layer).__name__, type(spec).__name__, getattr(spec, "mamba_type", None),
                  getattr(spec, "shapes", None), getattr(spec, "dtypes", None), getattr(spec, "num_kv_heads", None),
                  getattr(spec, "head_size", None), getattr(spec, "block_size", None), getattr(spec, "page_size_bytes", None))
        sys.exit(0)
    if fp32:
        model.float()
        for m in model.modules():
            if hasattr(m, "_compute_cos_sin_cache") and hasattr(m, "cos_sin_cache"):
                m.cos_sin_cache = m._compute_cos_sin_cache().float()
    gdn = [m for m in model.modules() if type(m).__name__ == "HPUGatedDeltaNetAttention"]
    for m in gdn:
        m.kv_cache = tuple(torch.zeros((3, *s), dtype=(torch.float32 if fp32 else d))
                           for s, d in zip(m.get_state_shape(), m.get_state_dtype()))
    for m in model.modules():
        if type(m).__name__ == "HpuQwen4ExpQSAAttention":
            m.attn = Sdpa(m)
    eos = vc.model_config.hf_text_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    W = model.language_model.lm_head.weight.float()

    def logits(seq):
        n = len(seq)
        for m in gdn:
            for t in m.kv_cache: t.zero_()
        md = SimpleNamespace(is_prompt=True, load_indices_tensor=torch.tensor([1]), store_indices_tensor=torch.tensor([1]),
                             query_start_loc_p=torch.tensor([0, n], dtype=torch.int32),
                             has_initial_states_p=torch.tensor([0], dtype=torch.int32), padding_mask_flat=None,
                             direct_gdn_state=False, num_accepted_tokens=None, dflash_full_query=False)
        with torch.no_grad(), set_forward_context(md, vc):
            out = model(input_ids=torch.tensor(seq), positions=torch.arange(n),
                        query_start_loc=torch.tensor([0, n], dtype=torch.int32),
                        ngram_context=torch.full((1, 2), eos, dtype=torch.int32))
        return out.float() @ W.T   # fp32 logits: compute_logits returns bf16 and ties at the tiny model's flat logits

    ref = json.load(open(os.path.join(ckpt, "reference.json")))
    for p in ref["prompts"]:
        lg = logits(p["prompt_ids"])[-1]
        r = p["last_prefill_top5"]
        diff = (lg[r["ids"]] - torch.tensor(r["logits"])).abs().max().item()
        seq, gen = list(p["prompt_ids"]), []
        for _ in range(ref["max_new_tokens"]):
            t = int(logits(seq)[-1].argmax()); gen.append(t); seq.append(t)
        agree = next((i for i, (a, b) in enumerate(zip(gen, p["generated_ids"])) if a != b), len(gen))
        print(f"prompt_len={len(p['prompt_ids'])} top1_ok={int(lg.argmax()) == r['ids'][0]} "
              f"max_abs_diff_top5={diff:.2e} greedy_agree={agree}/{len(gen)}")
