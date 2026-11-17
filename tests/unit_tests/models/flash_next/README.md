# Flash-Next CPU tests (program RUK-2709)

`tiny_config.json` is the shared fixture of the Flash-Next program: a 4-layer `Qwen4ExpForConditionalGeneration`
(3 Gated DeltaNet layers, 1 QSA layer, PLE at layer 2, 16 experts) with the real head counts (16/48 GDN heads,
4 query / 2 KV attention heads, 4 indexer heads) and tiny dims (hidden 128). It validates with both
`transformers.Qwen4ExpConfig` and vLLM's `Qwen4ExpConfig` at the pin and a random model built from it runs a
CPU forward in Hugging Face transformers (checked 2026-11-17). `tools/flash_next/make_tiny_checkpoint.py` (M0-T1)
turns it into a checkpoint directory plus `reference.json`, the HF CPU oracle every HPU path is compared with.

The reference carries a sixth 2600-token prompt for the sparse-mode tests (M2); dense-mode tests skip it through
`dense_reference_prompts` in `cpu_model.py`, because dense attention only equals QSA up to position 2050.

From M3-T1 the checkpoint also carries the 31 `mtp.*` tensors of a one-layer multi-token predictor (draft head),
built from the same HF modules as the target and saved in bfloat16. `reference.json` (schema 2) gains a top-level
`mtp_tensors` count and a per-prompt `mtp` block of teacher-forced draft ids (`rows`, `draft_argmax`,
`last_row_top5`), produced by running the MTP block on the target's prefill hidden state. The Hugging Face target
model ignores `mtp.*`, so `from_pretrained` on the checkpoint loads the target unchanged; the target-side
`generated_ids` and `last_prefill_top5` are identical to the schema-1 reference.

Run tests from the worktree root with the S0 environment and `PYTHONPATH="$PWD"`, so the worktree's `vllm_gaudi`
is the one imported (see `docs/design/flash-next/plan.md`, "Conventions").
