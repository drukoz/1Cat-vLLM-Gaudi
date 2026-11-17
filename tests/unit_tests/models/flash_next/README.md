# Flash-Next CPU tests (program RUK-2709)

`tiny_config.json` is the shared fixture of the Flash-Next program: a 4-layer `Qwen4ExpForConditionalGeneration`
(3 Gated DeltaNet layers, 1 QSA layer, PLE at layer 2, 16 experts) with the real head counts (16/48 GDN heads,
4 query / 2 KV attention heads, 4 indexer heads) and tiny dims (hidden 128). It validates with both
`transformers.Qwen4ExpConfig` and vLLM's `Qwen4ExpConfig` at the pin and a random model built from it runs a
CPU forward in Hugging Face transformers (checked 2026-11-17). `tools/flash_next/make_tiny_checkpoint.py` (M0-T1)
turns it into a checkpoint directory plus `reference.json`, the HF CPU oracle every HPU path is compared with.

The reference carries a sixth 2600-token prompt for the sparse-mode tests (M2); dense-mode tests skip it through
`dense_reference_prompts` in `cpu_model.py`, because dense attention only equals QSA up to position 2050.

Run tests from the worktree root with the S0 environment and `PYTHONPATH="$PWD"`, so the worktree's `vllm_gaudi`
is the one imported (see `docs/design/flash-next/plan.md`, "Conventions").
