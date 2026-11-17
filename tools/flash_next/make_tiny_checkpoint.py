# SPDX-License-Identifier: Apache-2.0
"""Turn the tiny Flash-Next fixture into a checkpoint directory plus the Hugging Face CPU reference.

The checkpoint is a 4-layer ``Qwen4ExpForConditionalGeneration`` built from the committed fixture
``tests/unit_tests/models/flash_next/tiny_config.json`` with seeded random bfloat16 weights (~134 MB, never
committed). ``reference.json`` is the CPU oracle every later HPU path is compared with: greedy tokens and the
top-5 last-prefill logits of six token-id prompts, computed by reloading the **saved** checkpoint in float32.

Usage: ``python tools/flash_next/make_tiny_checkpoint.py --out DIR [--seed 0] [--tokenizer-from DIR |
--skip-tokenizer] [--reference-only] [--fixture PATH]``.
"""

import argparse
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
SCHEMA_VERSION = 1
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
REFERENCE_FILE = "reference.json"


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


def build_checkpoint(out_dir: Path, fixture: Path, seed: int) -> None:
    """Build the random bf16 model from the fixture config and save it; the config.json is restored afterwards."""
    # Not AutoConfig: vLLM's create_engine_config() replaces CONFIG_MAPPING["qwen4_exp"] with its own config
    # class (no text_config.seed), which breaks this builder when it runs in the same process after vLLM.
    fixture_bytes = write_config(out_dir, fixture)
    config = Qwen4ExpConfig.from_pretrained(out_dir)
    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(config).to(torch.bfloat16)
    perturb_constant_weights(model, seed)
    # save_file (not save_pretrained) keeps the model's own tensor names: transformers 5.23 splits the fused
    # experts.gate_up_proj / experts.down_proj into per-expert gate/up/down_proj on save, and the fused names
    # are the ones the vLLM loader expects. from_pretrained reads this file back exactly.
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()}, out_dir / WEIGHTS_FILE)
    del model
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
    prompts = []
    paths_used = set()
    for prompt_ids in make_prompts(seed):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        top5 = last_prefill_top5(model, input_ids)
        generated_ids, path_used = greedy_generate(model, input_ids)
        paths_used.add(path_used)
        prompts.append({"prompt_ids": prompt_ids, "generated_ids": generated_ids, "last_prefill_top5": top5})
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
        "prompts": prompts,
    }
    (out_dir / REFERENCE_FILE).write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    del model
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
