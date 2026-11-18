# SPDX-License-Identifier: Apache-2.0
"""Turn the tiny Flash-Next fixture into a checkpoint directory plus the Hugging Face CPU reference.

The checkpoint is a 4-layer ``Qwen4ExpForConditionalGeneration`` built from the committed fixture
``tests/unit_tests/models/flash_next/tiny_config.json`` with seeded random bfloat16 weights (~134 MB, never
committed). On top of the 132 target tensors it carries the 31 ``mtp.*`` tensors of a one-layer multi-token
predictor (MTP), built from the Hugging Face component modules, so the draft path has weights to load.
``reference.json`` is the CPU oracle every later HPU path is compared with: greedy tokens, the top-5
last-prefill logits, and the MTP teacher-forced draft reference (per-prompt argmax draft ids and the last-row
top-5) of six token-id prompts, computed by reloading the **saved** checkpoint in float32.

Usage: ``python tools/flash_next/make_tiny_checkpoint.py --out DIR [--seed 0] [--tokenizer-from DIR |
--skip-tokenizer] [--reference-only] [--fixture PATH]``.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "unit_tests" / "models" / "flash_next" / "tiny_config.json"

HF_REPO_ID = "Qwen/Qwen3.8-Flash-Next"
TOKENIZER_PATTERNS = [
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "chat_template*",
    "generation_config.json",
]

PROMPT_LENGTHS = (1, 7, 33, 100, 700, 2600)
BOS_TOKEN_ID = 248044
MAX_NEW_TOKENS = 16
TOP_K = 5
SCHEMA_VERSION = 2
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
REFERENCE_FILE = "reference.json"
MTP_PREFIX = "mtp."
# The MTP reference dumps the last row's top-k; keep it aligned with the target's TOP_K.
# The 31 multi-token-predictor tensors the checkpoint gains, in module order. vLLM's
# ``Qwen4ExpMultiTokenPredictor`` loads these (prefix ``mtp.``); the target ignores them.
MTP_TENSOR_NAMES = [
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "fc_embedding.weight",
    "fc_hidden.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.indexer.index_qk_proj.weight",
    "layers.0.self_attn.indexer.q_layernorm.weight",
    "layers.0.self_attn.indexer.k_layernorm.weight",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.experts.gate_up_proj",
    "layers.0.mlp.experts.down_proj",
    "layers.0.mlp.shared_expert.gate_proj.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.shared_expert.down_proj.weight",
    "layers.0.mlp.shared_expert_gate.weight",
    "layers.0.attn_hyper_connection.hc_norm.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "layers.0.attn_hyper_connection.block_inject_weight.weight",
    "layers.0.mlp_hyper_connection.hc_norm.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "hyper_connection_mixer.hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight",
]


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_stats(path: Path) -> tuple[int, int]:
    """Number of tensors and file size in bytes of a safetensors checkpoint."""
    with safe_open(path, framework="pt") as handle:
        n_tensors = len(handle.keys())
    return n_tensors, path.stat().st_size


def make_prompts(seed: int) -> list[list[int]]:
    """Prompts of length 1/7/33/100/700/2600: the first id is the bos token, the rest drawn from [1000, 20000)."""
    generator = torch.Generator().manual_seed(seed)
    prompts = []
    for length in PROMPT_LENGTHS:
        prompt = [BOS_TOKEN_ID]
        if length > 1:
            drawn = torch.randint(1000, 20000, (length - 1,), dtype=torch.long, generator=generator)
            prompt.extend(drawn.tolist())
        prompts.append(prompt)
    return prompts


def write_config(out_dir: Path, fixture: Path) -> bytes:
    """Write the fixture as ``config.json`` and return its bytes (the checkpoint must keep them byte-identical)."""
    fixture_bytes = fixture.read_bytes()
    (out_dir / CONFIG_FILE).write_bytes(fixture_bytes)
    return fixture_bytes


def perturb_constant_weights(model: torch.nn.Module, seed: int) -> None:
    """Add seeded N(0, 0.1) noise to every constant language-model weight, in place.

    HF's init leaves norm weights, ``dt_bias`` and the PLE ``conv1d.weight`` constant (the conv is
    all zeros), so a parity test could not see a dropped short conv, its lost state, a norm
    weight loaded into the wrong layer or a transposed one. The vision tower is never run.
    """
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith("model.visual") or param.numel() < 2 or param.float().std() != 0:
                continue
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.1
            param.add_(noise.to(param.dtype))


def make_mtp_config(text_config) -> object:
    """A one-layer draft config: the first non-``linear_attention`` target layer type, no PLE layers."""
    mtp_config = copy.deepcopy(text_config)
    mtp_config.layer_types = [t for t in text_config.layer_types if t != "linear_attention"][:1]
    mtp_config.num_hidden_layers = 1
    mtp_config.ple_layer_ids = []
    return mtp_config


def build_mtp_container(hf_model: Qwen4ExpForConditionalGeneration, seed: int) -> torch.nn.Module:
    """A one-layer multi-token predictor built from the Hugging Face component modules.

    The submodule names (and therefore the ``state_dict`` keys) match the 31 entries of ``MTP_TENSOR_NAMES``.
    Weights use the model's own ``_init_weights`` and then the constant-parameter perturbation, so the norms and
    the (zero-init) conv-free layers carry seeded signal the parity tests can see. The target's ``embed_tokens``
    is shared, so no embedding tensor lives under ``mtp.``.
    """
    mtp_config = make_mtp_config(hf_model.config.text_config)
    hidden = mtp_config.hidden_size
    hc = mtp_config.hc_count

    class MultiTokenPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(hidden, eps=mtp_config.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(hc * hidden, eps=mtp_config.rms_norm_eps)
            self.fc_embedding = torch.nn.Linear(hidden, hidden, bias=False)
            self.fc_hidden = torch.nn.Linear(hidden, hidden, bias=False)
            self.layers = torch.nn.ModuleList([Qwen4ExpTextDecoderLayer(mtp_config, 0)])
            self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(mtp_config, use_combine=False)

    container = MultiTokenPredictor()
    # The text model owns the sub-config with initializer_range; the composite config does not.
    container.apply(hf_model.model.language_model._init_weights)
    perturb_constant_weights(container, seed + 2)
    # Fail loudly if HF module names drift: the checkpoint's mtp.* keys must be exactly the 31 real names.
    expected = {MTP_PREFIX + n for n in MTP_TENSOR_NAMES}
    assert {MTP_PREFIX + n for n in container.state_dict()} == expected, sorted(container.state_dict())
    return container


def build_checkpoint(out_dir: Path, fixture: Path, seed: int) -> None:
    """Build the random bf16 model from the fixture config and save it; the config.json is restored afterwards."""
    # Not AutoConfig: vLLM's create_engine_config() replaces CONFIG_MAPPING["qwen4_exp"] with its own config
    # class (no text_config.seed), which breaks this builder when it runs in the same process after vLLM.
    fixture_bytes = write_config(out_dir, fixture)
    config = Qwen4ExpConfig.from_pretrained(out_dir)
    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(config).to(torch.bfloat16)
    perturb_constant_weights(model, seed)
    # The multi-token-predictor tensors are saved as bf16 (like the target); the reference reloads them in fp32.
    mtp = build_mtp_container(model, seed)
    # save_file (not save_pretrained) keeps the model's own tensor names: transformers 5.23 splits the fused
    # experts.gate_up_proj / experts.down_proj into per-expert gate/up/down_proj on save, and the fused names
    # are the ones the vLLM loader expects. from_pretrained reads this file back exactly.
    weights = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    weights.update({MTP_PREFIX + name: tensor.to(torch.bfloat16).contiguous()
                    for name, tensor in mtp.state_dict().items()})
    save_file(weights, out_dir / WEIGHTS_FILE)
    del model, mtp
    # save_pretrained would rewrite config.json and drop the mrope keys from rope_parameters: keep the fixture bytes.
    (out_dir / CONFIG_FILE).write_bytes(fixture_bytes)


def setup_tokenizer(out_dir: Path, tokenizer_from: str | None, skip_tokenizer: bool) -> str:
    """Fetch the real tokenizer files into ``out_dir``; print and return what happened."""
    if skip_tokenizer:
        print("tokenizer: skipped (--skip-tokenizer), no tokenizer files written")
        return "skipped"
    if tokenizer_from is not None:
        source = Path(tokenizer_from)
        copied = []
        for pattern in TOKENIZER_PATTERNS:
            for src in sorted(source.glob(pattern)):
                shutil.copy2(src, out_dir / src.name)
                copied.append(src.name)
        print(f"tokenizer: copied {len(copied)} file(s) from {source}")
        return "copied"
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=HF_REPO_ID, allow_patterns=TOKENIZER_PATTERNS, local_dir=out_dir)
        print(f"tokenizer: downloaded from {HF_REPO_ID}")
        return "downloaded"
    except Exception as exc:  # noqa: BLE001
        print(f"tokenizer: download from {HF_REPO_ID} failed ({type(exc).__name__}: {exc}); "
              f"re-run with --tokenizer-from DIR to copy local files, or --skip-tokenizer to write none")
        return "download-failed"


def load_reference_model(out_dir: Path) -> Qwen4ExpForConditionalGeneration:
    """Reload the saved bf16 checkpoint in float32, so the reference uses the exact numbers vLLM will load."""
    model = Qwen4ExpForConditionalGeneration.from_pretrained(out_dir, dtype=torch.float32)
    return model.eval()


def load_mtp_container(out_dir: Path,
                       hf_model: Qwen4ExpForConditionalGeneration) -> tuple[torch.nn.Module, object]:
    """Rebuild the MTP container and load the saved ``mtp.*`` weights in float32, so the reference uses the exact
    numbers vLLM will load. The construction seed is irrelevant: the saved state dict overwrites every tensor."""
    container = build_mtp_container(hf_model, 0)
    with safe_open(out_dir / WEIGHTS_FILE, framework="pt") as handle:
        state = {name[len(MTP_PREFIX):]: handle.get_tensor(name).float()
                 for name in handle.keys() if name.startswith(MTP_PREFIX)}
    container.load_state_dict(state)
    return container.eval(), make_mtp_config(hf_model.config.text_config)


def mtp_reference_logits(hf_model: Qwen4ExpForConditionalGeneration, container: torch.nn.Module,
                         mtp_config: object, prompt_ids: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
    """Teacher-forced MTP draft logits ``[L, vocab]`` for one prompt.

    Mirrors upstream ``Qwen4ExpMultiTokenPredictor.forward``: embed the next token, fold the target's
    pre-final-mixer multi stream (captured from a fresh target prefill) into it, run the single draft layer and
    mix to a single stream, then project with the shared LM head. The target runs with ``use_cache=False`` so the
    capture is a plain prefill at positions ``0..L-1``.
    """
    text_model = hf_model.model.language_model
    text_config = hf_model.config.text_config
    hc, hidden = text_config.hc_count, text_config.hidden_size
    seq_len = prompt_ids.shape[0]

    captured: dict[str, torch.Tensor] = {}

    def capture_multi(module, args):  # noqa: ANN001 - forward_pre_hook signature
        captured["multi"] = args[0].detach()

    handle = text_model.hyper_connection_mixer.register_forward_pre_hook(capture_multi)
    with torch.no_grad():
        text_model(input_ids=prompt_ids.unsqueeze(0), use_cache=False)
    handle.remove()
    multi = captured["multi"][0]  # [L, hc*H]

    with torch.no_grad():
        embeds = text_model.embed_tokens(next_ids)  # [L, H]
        inputs_embeds = container.fc_embedding(container.pre_fc_norm_embedding(embeds))
        branches = container.fc_hidden(container.pre_fc_norm_hidden(multi).view(seq_len, hc, hidden))
        hidden_states = (inputs_embeds.unsqueeze(-2) + branches).flatten(-2)  # [L, hc*H]

        # Mirror the text model's prefill positioning: 4D mrope positions split into text + 3 rotary planes.
        position_ids = torch.arange(seq_len).view(1, 1, -1).expand(4, 1, -1)
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
        causal_mask = create_causal_mask(
            config=mtp_config,
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
            # The indexer overlays its own selection on top of the causal mask, so never skip creating it.
            allow_is_causal_skip=False,
        )
        position_embeddings = text_model.rotary_emb(embeds.unsqueeze(0), position_ids)
        layer_out = container.layers[0](
            hidden_states=hidden_states.unsqueeze(0),
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            conv_mask=None,
            past_key_values=None,
            ple_input_ids=None,
        )
        sample = container.hyper_connection_mixer(layer_out[0])  # [L, H]
        return hf_model.lm_head(sample)  # [L, vocab]


def mtp_reference(hf_model: Qwen4ExpForConditionalGeneration, container: torch.nn.Module, mtp_config: object,
                  prompt_ids: list[int], generated_ids: list[int]) -> dict:
    """Per-prompt MTP draft reference: argmax draft ids on the stored rows and the last row's top-5.

    Row ``r`` is teacher-forced on ``next_ids[r]`` (the token after position ``r``; the last row uses the first
    greedy token). Rows are every position for short prompts, the last 8 for prompts longer than 100.
    """
    length = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    next_ids = torch.tensor(prompt_ids[1:] + [generated_ids[0]], dtype=torch.long)
    logits = mtp_reference_logits(hf_model, container, mtp_config, prompt, next_ids)  # [L, vocab]
    rows = list(range(length)) if length <= 100 else list(range(length - 8, length))
    draft_argmax = [int(logits[r].argmax()) for r in rows]
    values, indices = torch.topk(logits[length - 1], TOP_K)
    return {
        "rows": rows,
        "draft_argmax": draft_argmax,
        "last_row_top5": {"ids": [int(i) for i in indices], "logits": [round(float(v), 6) for v in values]},
    }


def last_prefill_top5(model: Qwen4ExpForConditionalGeneration, input_ids: torch.Tensor) -> dict:
    """Top-5 ids and logits of the last prompt position of a fresh (uncached) prefill."""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1]
    values, indices = torch.topk(logits, TOP_K)
    return {"ids": [int(i) for i in indices], "logits": [round(float(v), 6) for v in values]}


def manual_greedy(model: Qwen4ExpForConditionalGeneration, input_ids: torch.Tensor) -> list[int]:
    """Greedy loop without the cache: re-run the whole sequence every step and take the argmax of the last logits."""
    generated: list[int] = []
    sequence = input_ids
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False).logits
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            sequence = torch.cat([sequence, torch.tensor([[next_id]], dtype=sequence.dtype)], dim=1)
    return generated


def greedy_generate(model: Qwen4ExpForConditionalGeneration, input_ids: torch.Tensor) -> tuple[list[int], str]:
    """Greedy decoding of ``MAX_NEW_TOKENS`` new tokens; returns them and the path used ("generate" or "manual")."""
    with torch.no_grad():
        try:
            output = model.generate(input_ids=input_ids,
                                    attention_mask=torch.ones_like(input_ids),
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=False,
                                    use_cache=True)
            return output[0, input_ids.shape[1]:].tolist(), "generate"
        except Exception as exc:  # noqa: BLE001
            print(f"generation: model.generate raised {type(exc).__name__} ({exc}); "
                  f"falling back to a manual greedy loop with use_cache=False")
    return manual_greedy(model, input_ids), "manual"


def write_reference(out_dir: Path, fixture: Path, seed: int) -> dict:
    """Run the five prompts through the reloaded checkpoint and write ``reference.json``."""
    model = load_reference_model(out_dir)
    container, mtp_config = load_mtp_container(out_dir, model)
    prompts = []
    paths_used = set()
    for prompt_ids in make_prompts(seed):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        top5 = last_prefill_top5(model, input_ids)
        generated_ids, path_used = greedy_generate(model, input_ids)
        paths_used.add(path_used)
        mtp = mtp_reference(model, container, mtp_config, prompt_ids, generated_ids)
        prompts.append({"prompt_ids": prompt_ids, "generated_ids": generated_ids,
                        "last_prefill_top5": top5, "mtp": mtp})
    generation = paths_used.pop() if len(paths_used) == 1 else "mixed"
    reference = {
        "schema": SCHEMA_VERSION,
        "seed": seed,
        "config_sha256": sha256_file(fixture),
        "weights_sha256": sha256_file(out_dir / WEIGHTS_FILE),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "generation": generation,
        "max_new_tokens": MAX_NEW_TOKENS,
        "mtp_tensors": len(MTP_TENSOR_NAMES),
        "prompts": prompts,
    }
    (out_dir / REFERENCE_FILE).write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    del model, container
    return reference


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the tiny Flash-Next checkpoint and its HF CPU reference.")
    parser.add_argument("--out", required=True, type=Path, help="checkpoint directory to write")
    parser.add_argument("--seed", type=int, default=0, help="seed for the weights and the prompt ids")
    tokenizer_group = parser.add_mutually_exclusive_group()
    tokenizer_group.add_argument("--tokenizer-from", type=str, default=None, metavar="DIR",
                                 help="copy the tokenizer files from DIR instead of downloading them")
    tokenizer_group.add_argument("--skip-tokenizer", action="store_true", help="write no tokenizer files")
    parser.add_argument("--reference-only", action="store_true",
                        help="only rebuild reference.json from the checkpoint already in --out")
    parser.add_argument("--fixture", type=Path, default=None, help=f"fixture config (default: {DEFAULT_FIXTURE})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fixture = args.fixture if args.fixture is not None else DEFAULT_FIXTURE
    if not fixture.is_file():
        print(f"fixture not found: {fixture}", file=sys.stderr)
        return 1
    weights = args.out / WEIGHTS_FILE
    if args.reference_only and not weights.is_file():
        print(f"--reference-only needs a checkpoint in {args.out}: {weights} is missing", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.reference_only:
        build_checkpoint(args.out, fixture, args.seed)
        setup_tokenizer(args.out, args.tokenizer_from, args.skip_tokenizer)
    reference = write_reference(args.out, fixture, args.seed)
    n_tensors, n_bytes = safetensors_stats(weights)
    print(f"wrote {args.out}: {n_tensors} tensors, {n_bytes} bytes, "
          f"reference prompts {len(reference['prompts'])}, generation {reference['generation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
