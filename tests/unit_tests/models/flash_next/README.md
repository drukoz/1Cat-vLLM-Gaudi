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

`test_mtp_end_to_end.py` (M3-T8) drives the full draft-K / verify / accept loop end to end: a dense 6-step greedy
stream on the 33-token prompt against the HF reference, GDN/PLE state parity against an interleaved single-token
chain, an accept-path parity test with a fixed `[3, 1, x, 2]` acceptance schedule, a forced rejection (tampered
draft) that must roll back cleanly, one step on the 2600-token prompt in `QsaMode.SPARSE` with indexer
key-ring/compressed-cache parity, and mutation probes on the drafter hidden and on `num_accepted`. The random tiny
model's own drafts are never accepted, so the accept-path, rollback, sparse and `num_accepted`-mutation tests feed
*oracle drafts*: the drafter still runs, but its drafts are replaced by the reference's own greedy continuation
(`generated_ids`) before the verification, which makes greedy verification accept exactly the oracle prefix and
drives the lattice load `idx[num_accepted - 1]` with `num_accepted > 1`. Slots (`num_slots=8`): the verification
rows write lattice slots 1–4 (state rows / sparse ring slots), the speculative chain lives at lattice slot 1 and
moves to `lattice[a]` after committing `a` drafts, the reference single-token chain runs at slot 5, the drafter at
slot 2.

Run tests from the worktree root with the S0 environment and `PYTHONPATH="$PWD"`, so the worktree's `vllm_gaudi`
is the one imported (see `docs/design/flash-next/plan.md`, "Conventions").
