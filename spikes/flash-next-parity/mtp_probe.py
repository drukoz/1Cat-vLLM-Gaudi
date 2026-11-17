# SPDX-License-Identifier: Apache-2.0
"""CTO probe (2026-11-17, M3 scoping): build Qwen4ExpMTP on CPU through vLLM with an MTP speculative config.

Usage: PYTHONPATH=<worktree> python spikes/flash-next-parity/mtp_probe.py <tiny-checkpoint-dir> <worktree>
Records: the draft builds with dummy weights, nvidia.mtp.GatedResidual must be rebound to HpuGatedResidual,
the forward returns (sample [T, H], multi [T, hc*H]) and compute_logits gives [T, vocab]. System map fact 30."""
import os, sys, tempfile, traceback
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel, init_distributed_environment,
                              initialize_model_parallel)
from vllm.engine.arg_utils import EngineArgs
import vllm_gaudi.models.qwen4_exp as fn
fn.register()
import vllm.models.qwen4_exp.nvidia.mtp as _mtp
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual
_mtp.GatedResidual = HpuGatedResidual  # the one seam the M0 patches do not reach
ckpt = sys.argv[1]
vllm_config = EngineArgs(model=ckpt, tokenizer=ckpt, skip_tokenizer_init=True, dtype="bfloat16",
                         enable_prefix_caching=False, max_model_len=512, max_num_seqs=4, enforce_eager=True,
                         language_model_only=True, load_format="dummy",
                         speculative_config={"method": "mtp", "num_speculative_tokens": 3}).create_engine_config()
sc = vllm_config.speculative_config
dhf = sc.draft_model_config.hf_config
print("SPEC method", sc.method, "num_spec", sc.num_speculative_tokens, "use_eagle", sc.use_eagle(), "use_dflash", sc.use_dflash())
print("DRAFT arch", dhf.architectures, "num_hidden_layers", getattr(dhf, "num_hidden_layers", None),
      "text num_hidden_layers", getattr(getattr(dhf, "text_config", dhf), "num_hidden_layers", None),
      "mtp_num_hidden_layers", getattr(getattr(dhf, "text_config", dhf), "mtp_num_hidden_layers", None))
print("vllm_config.num_speculative_tokens", vllm_config.num_speculative_tokens, "mamba_cache_mode", vllm_config.cache_config.mamba_cache_mode,
      "max_num_batched_tokens", vllm_config.scheduler_config.max_num_batched_tokens, "async", vllm_config.scheduler_config.async_scheduling)
fd, init_file = tempfile.mkstemp(dir="/tmp"); os.close(fd)
with set_current_vllm_config(vllm_config):
    try:
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{init_file}", local_rank=0, backend="gloo")
        initialize_model_parallel(1, 1)
        from vllm.model_executor.model_loader import get_model
        model = get_model(vllm_config=vllm_config)
        buf = model.language_model.model._mtp_hidden_buffer
        print("TARGET", type(model).__name__, "mtp buffer", None if buf is None else (tuple(buf.shape), buf.dtype),
              "has get_mtp_target_hidden_states", hasattr(model, "get_mtp_target_hidden_states"))
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        for name, m in model.named_modules():
            if isinstance(m, MambaBase):
                spec = m.get_kv_cache_spec(vllm_config)
                print("  MAMBA", name, type(m).__name__, "shapes", spec.shapes, "num_spec_blocks", spec.num_speculative_blocks, "mode", spec.mamba_cache_mode)
        import vllm.models.qwen4_exp.nvidia.mtp as mtp_mod
        print("mtp.GatedResidual =", mtp_mod.GatedResidual.__module__ + "." + mtp_mod.GatedResidual.__name__)
        from vllm.compilation.backends import set_model_tag
        from vllm.config import replace
        draft_vllm_config = replace(vllm_config, model_config=sc.draft_model_config)
        with set_model_tag("eagle_head"):
            draft = get_model(vllm_config=draft_vllm_config, model_config=sc.draft_model_config, load_config=sc.draft_load_config)
        print("DRAFT", type(draft).__name__, "params", sum(p.numel() for p in draft.parameters()))
        for n, m in draft.named_modules():
            if n.count(".") <= 3 and n:
                print("   ", n, type(m).__name__)
        names = [n for n, _ in draft.named_parameters()]
        print("DRAFT param names (first 12):", names[:12])
        print("lm_head is embed?", getattr(draft, "lm_head", None) is getattr(draft.model, "embed_tokens", None))
        # try a forward of the draft on CPU: one row, position 0, with the dense QSA attn replaced by SDPA stand-in
        sys.path.insert(0, os.path.join(sys.argv[2]))
        from tests.unit_tests.models.flash_next.cpu_model import use_sdpa_attention, to_fp32
        print("sdpa replaced in draft:", use_sdpa_attention(draft))
        from vllm.forward_context import set_forward_context
        from types import SimpleNamespace
        hc = model.language_model.model.config.hc_count; H = model.language_model.model.config.hidden_size
        T = 3
        ids = torch.tensor([5, 6, 7]); pos = torch.arange(T); hid = torch.randn(T, hc * H, dtype=torch.bfloat16)
        md = SimpleNamespace(is_prompt=True, slot_mapping=torch.arange(T), block_list=None, block_size=128, context_lens_tensor=torch.tensor([0]),
                             load_indices_tensor=None, store_indices_tensor=None, query_start_loc_p=torch.tensor([0, T], dtype=torch.int32))
        with set_forward_context(md, vllm_config), torch.no_grad():
            out = draft(input_ids=ids, positions=pos, hidden_states=hid, spec_step_idx=0)
        print("DRAFT forward returns", type(out).__name__, [tuple(t.shape) for t in out] if isinstance(out, tuple) else tuple(out.shape))
        logits = draft.compute_logits(out[0])
        print("DRAFT logits", tuple(logits.shape), logits.dtype)
    except Exception:
        traceback.print_exc()
    finally:
        destroy_model_parallel(); destroy_distributed_environment()
        try: os.unlink(init_file)
        except FileNotFoundError: pass
