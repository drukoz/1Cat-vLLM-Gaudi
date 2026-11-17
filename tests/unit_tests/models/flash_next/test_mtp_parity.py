# SPDX-License-Identifier: Apache-2.0
"""MTP draft parity of the whole Flash-Next model against the HF oracle (M3-T3).

The target runs under ``speculative_config={"method": "mtp", ...}`` and its forward is
``(hidden, [multi])``; the draft (``HpuQwen4ExpMTP``, built by ``cpu_draft_model``) takes the
teacher-forced step 0 of the engine: ``input_ids=next_ids`` (the token after each position,
the first greedy token on the last), ``positions=arange(L)``, ``hidden_states=multi``.
Parity is checked on the draft's own ``lm_head`` logits: the reference stores the per-row
argmax (``mtp.draft_argmax``) and the last row's top-5, so a row that dropped the multi
stream, mis-mapped a checkpoint tensor or ran a different attention cannot match.

Dense fp32 matches ``reference.json`` at 1e-5 (the capture is the pre-final-mixer multi
stream, M2-proven, and the draft reuses the parity-proven decoder layer); dense bf16 and
sparse fp32 (2600-token prompt, indexer selection) at 1e-2 / 1e-5. The target's MTP capture
buffer is a plain tensor attribute at engine dtype, so :func:`fp32_mtp_buffer` rebuilds it
in fp32 (and to >= 2600 rows for the sparse prompt) - without it every ``multi`` row is
bf16-rounded and no fp32 parity is possible.

``test_draft_loads_mtp_weights`` pins the weight plumbing: the 31 ``mtp.*`` checkpoint
tensors land in the draft (fused qkv included), the shared ``embed_tokens``/``lm_head``
come from the target tensors, and the target build itself ignores every ``mtp.*`` tensor.
The negative control feeds ``hidden.repeat(1, hc_count)`` instead of ``multi`` on the
33-token prompt and asserts at least one row's argmax leaves the reference.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_draft_model, cpu_model, dense_reference_prompts, eos_token_id, ple_kwargs, prefill_metadata,
    reset_cpu_states)

MTP = {"method": "mtp", "num_speculative_tokens": 3}
FP32_LOGIT_TOL = 1e-5  # the capture and the draft layer reproduce HF; the residual is fp32 reassociation
BF16_LOGIT_TOL = 1e-2  # bf16 draft numerics (engine-dtype run)
SPARSE_MAX_MODEL_LEN = 4096  # the compressed key cache has one row per 4 tokens; 2600 needs >= 650 rows
LONG_PROMPT = 2600
HC = 4  # tiny_config.json hc_count
NEGATIVE_PROMPT = 33


def fp32_mtp_buffer(model, min_rows: int) -> None:
    """Rebuild the target's MTP multi-stream capture buffer in fp32 with >= ``min_rows`` rows.

    ``_mtp_hidden_buffer`` is a plain tensor attribute built at ``model_config.dtype``
    (bf16) and sized ``max_num_batched_tokens`` (2048), so ``to_fp32`` leaves it bf16 and
    the 2600-token prefill overflows it; every captured ``multi`` row would be bf16-rounded.
    """
    found = 0
    for module in model.modules():
        buffer = getattr(module, "_mtp_hidden_buffer", None)
        if isinstance(buffer, torch.Tensor):
            rows = max(int(buffer.shape[0]), min_rows)
            module._mtp_hidden_buffer = torch.empty((rows, buffer.shape[1]), dtype=torch.float32)
            found += 1
    assert found == 1, f"expected exactly one _mtp_hidden_buffer on the MTP target, found {found}"


def sparse_attns(model) -> list:
    """The ``.sparse_attn`` stand-ins of every SPARSE QSA layer (``[]`` in dense mode)."""
    return [m.sparse_attn for m in model.modules()
            if isinstance(m, HpuQwen4ExpQSAAttention) and getattr(m, "sparse_attn", None) is not None]


def target_prefill_out(model, vllm_config, eos: int, token_ids):
    """The target's MTP forward ``(hidden, [multi])`` for one fresh prefill of ``token_ids`` in slot 1."""
    num_tokens = len(token_ids)
    reset_cpu_states(model)
    for att in sparse_attns(model):
        att.reset()
        att.slot = 1
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, 1), vllm_config):
        return model(input_ids=torch.tensor(token_ids),
                     positions=torch.arange(num_tokens),
                     **ple_kwargs(num_tokens, eos))


def draft_fp32_logits(draft, sample: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``(R, 248320)`` of the draft's OWN lm head on sample rows ``(R, H)``.

    ``sample.float() @ lm_head.weight.float().T``: ``compute_logits`` returns bf16, and the
    reference's argmax rows on the tiny model's flat logits are not bf16-stable.
    """
    return sample.float() @ draft.lm_head.weight.float().T


def draft_logits_from_multi(multi, draft, vllm_config, next_ids, rows, fp32: bool) -> torch.Tensor:
    """The teacher-forced draft step 0: ``(len(rows), vocab)`` fp32 logits from ``multi``.

    ``spec_step_idx=0`` on a fresh draft state (side caches and KV zeroed, slot 1): the same
    inputs the engine's draft runner passes after a target prefill. ``fp32=False`` rounds the
    logits to bf16 first, the dtype the engine's ``compute_logits`` produces.
    """
    num_tokens = multi.shape[0]
    reset_cpu_states(draft)
    for att in sparse_attns(draft):
        att.reset()
        att.slot = 1
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, 1), vllm_config):
        sample, _ = draft(input_ids=torch.tensor(next_ids),
                          positions=torch.arange(num_tokens),
                          hidden_states=multi,
                          spec_step_idx=0)
    logits = draft_fp32_logits(draft, sample)
    if not fp32:
        logits = logits.bfloat16().float()
    rows = torch.tensor(rows, dtype=torch.long)
    return logits[rows]


def draft_logits(target_out, draft, vllm_config, prompt_ids, next_ids, fp32: bool, rows) -> torch.Tensor:
    """``draft_logits_from_multi`` fed with the target's captured multi stream ``target_out[1][0]``."""
    _, aux = target_out
    return draft_logits_from_multi(aux[0], draft, vllm_config, next_ids, rows, fp32)


def next_token_ids(prompt: dict) -> list:
    """Teacher-forced draft inputs: the token after each position, the first greedy token on the last."""
    ids = prompt["prompt_ids"]
    return ids[1:] + [prompt["generated_ids"][0]]


def check_rows(logits: torch.Tensor, mtp: dict, tol: float, label: str) -> None:
    """Per-row argmax equals the reference, and the last row's top-5 id and logits are within ``tol``."""
    got = logits.argmax(dim=-1).tolist()
    assert got == mtp["draft_argmax"], f"{label}: draft argmax rows {got} != reference {mtp['draft_argmax']}"
    last = logits[-1]
    ref = mtp["last_row_top5"]
    top5 = last.topk(5).indices.tolist()
    assert top5 == ref["ids"], f"{label}: draft last-row top-5 {top5} != reference {ref['ids']}"
    delta = (last[ref["ids"]] - torch.tensor(ref["logits"], dtype=torch.float32)).abs().max().item()
    assert delta <= tol, (f"{label}: draft last-row max |Δlogit| at the reference top-5 ids {delta:.3e} > {tol:.0e}")


def test_dense_fp32_matches_reference(tiny_checkpoint):
    """fp32 draft parity on EVERY dense prompt: per-row argmax and last-row top-5 within 1e-5."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    with cpu_model(tiny_checkpoint, fp32=True, speculative_config=MTP) as (vllm_config, model):
        fp32_mtp_buffer(model, min_rows=1)  # dtype fix only: 2048 rows cover every dense prompt
        draft = cpu_draft_model(vllm_config, fp32=True)
        eos = eos_token_id(vllm_config)
        for prompt in dense_reference_prompts(reference):
            ids = prompt["prompt_ids"]
            out = target_prefill_out(model, vllm_config, eos, ids)
            logits = draft_logits(out, draft, vllm_config, ids, next_token_ids(prompt), True, prompt["mtp"]["rows"])
            check_rows(logits, prompt["mtp"], FP32_LOGIT_TOL, f"prompt {len(ids)}")


def test_dense_bf16_matches_reference(tiny_checkpoint):
    """bf16 draft parity on the 7/33/100-token prompts: last-row top-1 equals the reference, top-5 |Δ| <= 1e-2."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    lengths = {7, 33, 100}
    with cpu_model(tiny_checkpoint, speculative_config=MTP) as (vllm_config, model):
        draft = cpu_draft_model(vllm_config)
        eos = eos_token_id(vllm_config)
        for prompt in reference["prompts"]:
            ids = prompt["prompt_ids"]
            if len(ids) not in lengths:
                continue
            out = target_prefill_out(model, vllm_config, eos, ids)
            logits = draft_logits(out, draft, vllm_config, ids, next_token_ids(prompt), False, [len(ids) - 1])
            ref = prompt["mtp"]["last_row_top5"]
            top1 = int(logits[-1].argmax())
            assert top1 == ref["ids"][0], f"prompt {len(ids)}: bf16 draft top-1 {top1} != reference {ref['ids'][0]}"
            delta = (logits[-1][ref["ids"]] - torch.tensor(ref["logits"], dtype=torch.float32)).abs().max().item()
            assert delta <= BF16_LOGIT_TOL, (
                f"prompt {len(ids)}: bf16 draft max |Δlogit| at the reference top-5 ids {delta:.3e} "
                f"> {BF16_LOGIT_TOL:.0e}")


def test_sparse_fp32_matches_reference(tiny_checkpoint):
    """fp32 draft parity on the 2600-token prompt: sparse target capture + sparse draft indexer.

    Also asserts the draft's own ``compressed_key_cache`` has non-zero rows afterwards: the
    sparse draft path ran (a silent fall-back to dense attention would leave it empty).
    """
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == LONG_PROMPT)
    ids = prompt["prompt_ids"]
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   qsa_mode=QsaMode.SPARSE,
                   max_model_len=SPARSE_MAX_MODEL_LEN,
                   speculative_config=MTP) as (vllm_config, model):
        fp32_mtp_buffer(model, min_rows=len(ids))
        draft = cpu_draft_model(vllm_config, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN)
        eos = eos_token_id(vllm_config)
        out = target_prefill_out(model, vllm_config, eos, ids)
        logits = draft_logits(out, draft, vllm_config, ids, next_token_ids(prompt), True, prompt["mtp"]["rows"])
        check_rows(logits, prompt["mtp"], FP32_LOGIT_TOL, f"prompt {LONG_PROMPT}")
        layers = [m for m in draft.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]
        assert layers and all(m.qsa_mode is QsaMode.SPARSE for m in layers), "the draft was not built in SPARSE mode"
        for layer in layers:
            cache = layer.indexer.compressed_key_cache
            written = int(cache.abs().sum(dim=-1).gt(0).sum())
            assert written > 0, "the draft indexer wrote no compressed-key rows: the sparse draft path did not run"


def test_draft_loads_mtp_weights(tiny_checkpoint):
    """The 31 ``mtp.*`` tensors load into the draft (fused qkv included), embed/lm_head come from the
    target tensors, and the target build itself ignores every ``mtp.*`` tensor."""
    checkpoint = load_file(str(tiny_checkpoint / "model.safetensors"))
    mtp_tensors = sorted(n for n in checkpoint if n.startswith("mtp."))
    assert len(mtp_tensors) == 31, f"checkpoint carries {len(mtp_tensors)} mtp.* tensors, expected 31"
    with cpu_model(tiny_checkpoint, speculative_config=MTP) as (vllm_config, model):
        assert not any("mtp" in name for name, _ in model.named_parameters()), "a mtp.* tensor landed in the target"
        draft = cpu_draft_model(vllm_config)
        params = dict(draft.named_parameters())
        plain = [
            "model.pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden.weight",
            "model.fc_embedding.weight",
            "model.fc_hidden.weight",
            "model.layers.0.self_attn.indexer.index_qk_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
            "model.hyper_connection_mixer.input_mix_weight_up.weight",
        ]
        for name in plain:
            assert name in params, f"draft has no parameter {name}"
            assert torch.equal(params[name].detach().cpu(), checkpoint["mtp." + name[len("model."):]]), (
                f"draft {name} != checkpoint mtp.{name[len('model.'):]}")
        attn = draft.model.layers[0].self_attn
        qkv = attn.qkv_proj.weight
        q, kv = 2 * attn.q_size, attn.kv_size  # attn_output_gate: the q region holds q and gate per head
        assert torch.equal(qkv[:q].cpu(), checkpoint["mtp.layers.0.self_attn.q_proj.weight"]), (
            "draft qkv_proj q-region != checkpoint mtp.layers.0.self_attn.q_proj.weight")
        assert torch.equal(qkv[q:q + kv].cpu(), checkpoint["mtp.layers.0.self_attn.k_proj.weight"]), (
            "draft qkv_proj k-region != checkpoint mtp.layers.0.self_attn.k_proj.weight")
        assert torch.equal(draft.model.embed_tokens.weight.detach().cpu(),
                           checkpoint["model.language_model.embed_tokens.weight"]), (
            "draft embed_tokens != checkpoint target embed")
        assert torch.equal(draft.lm_head.weight.detach().cpu(), checkpoint["lm_head.weight"]), (
            "draft lm_head != checkpoint lm_head.weight")


def test_multi_stream_routing_negative_control(tiny_checkpoint):
    """Feeding ``hidden.repeat(1, hc_count)`` (no multi-stream structure) must change draft argmaxes.

    If replicating the sampled stream still reproduced ``mtp.draft_argmax`` on the 33-token
    prompt, the parity tests would not be sensitive to the multi-stream routing.
    """
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == NEGATIVE_PROMPT)
    ids = prompt["prompt_ids"]
    with cpu_model(tiny_checkpoint, fp32=True, speculative_config=MTP) as (vllm_config, model):
        fp32_mtp_buffer(model, min_rows=1)
        draft = cpu_draft_model(vllm_config, fp32=True)
        eos = eos_token_id(vllm_config)
        out = target_prefill_out(model, vllm_config, eos, ids)
        hidden, _ = out
        flat = hidden.repeat(1, HC)
        logits = draft_logits_from_multi(flat, draft, vllm_config, next_token_ids(prompt), prompt["mtp"]["rows"], True)
        got = logits.argmax(dim=-1).tolist()
        assert got != prompt["mtp"]["draft_argmax"], (
            f"prompt {NEGATIVE_PROMPT}: replicating the sampled stream reproduced the reference argmax rows; "
            "the parity tests are not sensitive to the multi-stream input")
