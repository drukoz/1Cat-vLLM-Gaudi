# Hyper-connection kernels at verification rows (track hc-kernels)

Design note, report and handoff of the `hc-kernels` track (branch `feat/flash-next-gaudi-hc-kernels`, worktree
`/Volumes/Storage/Claude/gaudi-work/wt/hc-kernels`). The harness refuses report files for this agent, so this note is
the report. Labels: MEASURED (card), VERIFIED (simulator or CPU test named), INFERRED (arithmetic or reading).

## 0. Handoff (if this agent stops) — updated 09:40 UTC 2026-11-21

- **Goal 1:** the two hyper-connection TPC kernels (`hc_fused_gate_mix`, `hc_fused_combine_norm`) at 128
  verification rows (32 streams, MTP k = 3) from about 7 ms per step to about 2 ms. **Goal 2:** the GDN verify kernel
  `gdn_fused_core_verify` (9.45 ms per 128-row step): agreed with kernel-fixes at 09:00 UTC, reviewed by it at 09:25
  (no objection to the card job; its suggestions are in: variants 4/5, chained bench calls, device traces).
- First reading sent to the coordinator 08:25 UTC; plan approved 08:27 UTC (new GUIDs behind a default-off flag, the
  HC contract of `tests/unit_tests/hc_fused/hc_case.py` as the tolerance plus the exact-match fraction against the
  shipped kernels; the 8-card arm goes into build 6).
- **On the branch, pushed:** HC `3c2ffd72`, `6f5b9300`, `0183f133`, `6c62cc56` (interleaved kernels, ops, flag
  `VLLM_HPU_HC_FUSED_IL`, tests, bench); GDN `8ed20160` (variants 1-3 of `VLLM_HPU_GDN_VERIFY_VARIANT`). Next commit
  (09:30, after its simulator run): variants 4/5, chained benches, traces, `kernel_engines.py`: pushed as `b9ff6384`
  (simulator: variants 1-5 byte-identical, 71 passed).
- **Card (coordinator 09:33 UTC: split in two):** job A = `tools/flash_next/hc_kernels/card_hc_quick.sh` (HC op card
  test, the TP1 tiny gate with `VLLM_HPU_HC_FUSED_IL=1`, the HC bench at 4/32/96/128 rows, traces at 128 rows), hold 12
  min, replaces 00227 (its place: right after build 5, beside engine-step's 2-card job); job B =
  `tools/flash_next/hc_kernels/card_gdn_verify.sh` (GDN verify variants 0-5: op test, chained bench, bits, traces),
  placed by the coordinator about 15:00-16:00 UTC beside step-anatomy's 4-card trace. **Job A = 00231 `hc-quick`,
  job B = 00232 `gdn-variants`** (both image `b9ff6384`, queued 09:39 UTC; waiters' logs
  `reports/hc-kernels/logs/farm_hc-quick.out`, `farm_gdn-variants.out`). When a job ends: `bin/farm fetch hc-kernels <id> <id>-<name>` if its waiter
  is gone; read `pytest_il.log`, `tiny.log`, `bench.log` (VERDICT), `gdn_bench_variant*.log` (us per call and
  "bit-identical" lines); traces with `python3 tools/flash_next/hc_kernels/kernel_engines.py <trace> <kernel name>`;
  send kernel-fixes the GDN result. The TP1 tiny gate stands in for a TP8 one (coordinator, 09:33).
- **Gate:** `bin/gate reports/hc-kernels/gate-8ed20160 --tools` running (log `reports/hc-kernels/logs/
  gate_8ed20160.log`); then a PR of the branch to `feat/flash-next-gaudi` (body draft `reports/hc-kernels/pr/
  pr_body_step1.md`), merge (default-off, gate green), tell the coordinator the PR number.
- **Next:** build-6 arm for the coordinator: `VLLM_HPU_HC_FUSED_IL=1` (with `VLLM_HPU_HC_FUSED=1`), and
  `VLLM_HPU_GDN_VERIFY_VARIANT=<best of the card A/B>` (with `VLLM_HPU_GDN_KERNEL_INTERLEAVE=1`).
- Tools: `tools/flash_next/hc_kernels/sim_cost.py` (simulator instructions per index-space point, oracle check,
  bit-equality against another build), `tools/flash_next/hc_kernels/bench_hc.py` (1-card replay bench),
  `reports/hc-kernels/tools/kernel_engines.py` (per-engine time of a kernel in a profiler trace).
- Untracked scratch in the worktree: `.hc_exp/` (prototype trees), `.hc_sim_base/` (build of the shipped kernels,
  the reference for bit-equality), `.hc_b_il/`. Never `git add -A` here.

## 1. Why the shipped kernels are slow at 128 rows

MEASURED (job 00205 dense arm, TP8 + EP8, sync MTP k = 3, 32 streams = 128 rows, rank-0 device trace, per-engine
events, `reports/hc-kernels/tools/kernel_engines.py`):

| kernel | calls per step | wall per call | TPC engines | per-engine busy |
|---|---|---|---|---|
| hc_fused_gate_mix | 106 | 41.1 us | 24 of 24 | 38.7 - 41.0 us |
| hc_fused_combine_norm | 101 | 30.1 us | 22 | 9.8 - 29.6 us |
| hc_fused_silu(_split) | 106 | 2.1 us | 24 | 1.2 - 1.8 us |
| gdn_fused_core_verify | 36 | 262 us | 16 | 256.6 - 262 us |

The index spaces are row-parallel already (gate_mix `[hidden / 128, rows]`, combine_norm `[hc_count, rows]`) and
use 22 to 24 TPCs: the coordinator's INFERRED "not row-parallel" was wrong. The cost is per index-space point.
VERIFIED in the simulator (`sim_cost.py` on the shipped kernels): gate_mix 670 instruction words per point,
combine_norm 2,063, at every row count; the simulator count times ceil(points / engines) at 1.7 GHz reproduces the
trace (gate_mix 107 points x 670 = 42 us against 41.1 MEASURED; combine_norm 24 points per engine x 2,063 = 29 us
against 30.1). The 1.7 GHz is INFERRED from the GDN decode kernel (71.6K simulator instructions = 42 us per point
in trace 00113). The loops are latency-bound: tpc-clang schedules one basic block at a time, so a loop body that
is one dependent chain pays its whole latency (gate_mix: 163 words per stream for 90 VPU operations; combine_norm:
40 + 48 words per 128-column chunk for 11 + 16), and the library sigmoid costs about 45 VPU operations per 64
lanes (`v_exp_cephes_f32` 23 + `v_reciprocal_f32` 20, every special case included). Cost is linear in rows:
about 0.53 us per row per hyper-connection, 51 us per row per step.

One fused kernel per layer is not possible: in every hyper-connection the two MME GEMMs (down and up projection)
sit between combine_norm and gate_mix, and the attention or MoE block sits between gate_mix and the next
combine_norm. The GEMMs (6.5 us each at 128 rows, weight-read floor about 2.8 us) stay as they are.

## 2. The interleaved kernels

Same inputs, outputs, node parameters and scalar arguments as the shipped kernels; two new GUIDs
(`hc_fused_gate_mix_il_bf16_gaudi2`, `hc_fused_combine_norm_il_bf16_gaudi2`), listed by `GetKernelGuids` after the
MoE glue GUIDs, each optional at build time (compile-probed in `csrc/hc_fused/CMakeLists.txt`, as the MoE glue).

- **`hc_fused_gate_mix_il`** (index space unchanged, `[hidden / 128, rows]`): for `hc_count == 4` the four streams
  of a chunk are one basic block (eight independent sigmoid chains per block instead of two), and the sigmoid is
  `hc_il_sigmoid` (`kernels/hc_fused_il.h`, 25 VPU operations): `e = exp(-|x|)` by the Cephes range reduction
  (n by the 1.5 * 2^23 trick, which also leaves n << 23 in the low bits) and a degree-6 minimax polynomial in Horner
  form (1.9e-9 relative before rounding, 8.4e-8 in fp32), `d = 1 + e` in [1, 2] (no special case), `1 / d` by a
  minimax quadratic seed (1.0e-2) and two Newton steps (1e-8), `sigmoid = q` for x >= 0 and `e * q` below; from
  `n <= -126` (|x| >= 87) e is 0, so the saturated sigmoid is exactly 1 or 0 (the exact edge cases). Products and
  the stream sum stay separate, rounded instructions; the sum starts from the first product instead of +0.0 (only an
  all -0.0 column keeps its sign). Any other `hc_count`: one stream per iteration with the same sigmoid.
- **`hc_fused_combine_norm_il`** (index space `[1, rows]`: one point per token owns the whole row): for
  `hc_count == 4` the four streams of a token run side by side (four independent chains per block), the block
  output chunk is loaded and converted once for the four streams, and the per-point tail (four injection weights,
  four rrms) overlaps. The injection weights use the shipped sigmoid (`hc_injection_weight`), so `residual_out` is
  the shipped kernel's bit for bit. Two differences in the norm, each in the last fp32 bit before the bf16
  rounding: the squares of the rounded bf16 values (exact in fp32) accumulate with `v_bf16_mac_acc32_b` (two fp32
  accumulators per stream), and `y + y * w` is one fused multiply-add. Pass 2 recomputes the combine from the inputs
  (no read-back of `residual_out`; a VLM stash of four streams is 20 KiB, over the 16 KiB a kernel with lookup-table
  functions has, VERIFIED by tpc-clang). Any other `hc_count`: the streams one after the other.

### Simulator (VERIFIED, `tests/standalone/hc_fused/test_hc_il_simulation.py`, 26 passed; `sim_cost.py`)

Every `hc_case.SIM_CASES` case of both kinds (tiny and real geometry, 1 to 2,048 rows, stress, edge, rounding,
hc 2 and 3) passes the HC contract against the torch oracle with the simulator's index-space mapping test and
partition shuffle on, and against the shipped kernel on the same inputs:

| kernel | instructions per point (real geometry) | per token | vs shipped | bit-equal to the shipped kernel |
|---|---|---|---|---|
| gate_mix shipped | 670 per (chunk, token) | 13,400 | 1 | - |
| gate_mix_il | 248 | 4,960 | 2.70x | 99.990 - 100 % (100 % on stress and edge cases) |
| combine_norm shipped | 2,063 per (stream, token) | 8,252 | 1 | - |
| combine_norm_il | 2,344 per token | 2,344 | 3.52x | 100.000 % residual_out and xn on every case |

INFERRED card time at 128 rows (24 TPCs, 1.7 GHz): gate_mix_il 15.6 us (shipped 41.1 MEASURED), combine_norm_il
8.3 us (6 tokens per engine; shipped 30.1 MEASURED): about 47 us less per hyper-connection, **about 4.7 ms less per
128-row verification step** (about 100 of each per step), about 1.2 ms less per 32-row plain step, nothing at 1 row.

The intermediate prototypes (`.hc_exp/`): gate_mix with only the four streams in one block, same math: 388 per
point (bit-identical); with the Cephes-form lean sigmoid: 322; combine_norm with four chunks per iteration and one
stream per point: 1,073 per (stream, token).

## 3. Wiring

- `flashinfer_gaudi/hc_fused.py`: `IL_OP_NAMES`, `IL_SCHEMAS` (the shipped schemas under the new names), fake
  kernels, wrappers `hc_gate_mix_il` / `hc_combine_norm_il` (the same copy modes), `hc_fused_il_missing_ops(device)`.
  The new ops are not in `OP_NAMES`: an extension without them leaves the six shipped ops as they are.
- `flashinfer_gaudi/_native.py`: `hc_fused_il_missing_kernels()` asks the loaded kernel library once
  (`GetKernelGuids`) whether it serves the two GUIDs.
- `csrc/hc_fused/pytorch/hpu_hc_fused.cpp`: the two ops registered with their GUIDs (the shipped validation).
- `vllm_gaudi/models/qwen4_exp/hyperconnection.py`: with `VLLM_HPU_HC_FUSED=1` and `VLLM_HPU_HC_FUSED_IL=1` a layer
  decides once, at its first fused call, and calls the interleaved ops in `mix`, `combine_and_mix` (both use
  gate_mix; combine_and_mix uses combine_norm); a layer whose interleaved ops or kernels are not served logs one line
  (`HC fused interleaved kernels (VLLM_HPU_HC_FUSED_IL=1) is off for <layer>: ...`, matched by the rental gates'
  HC fallback pattern, so a tiny gate run with the flag fails when the interleaved kernels did not run) and keeps
  the shipped fused ops.

## 4. Tests (VERIFIED on this Mac)

- Simulator, `tests/standalone/hc_fused/test_hc_il_simulation.py`: 26 passed (08:52 UTC).
- CPU: `tests/unit_tests/models/flash_next/test_hc_fused_il_layer.py` 35 passed (the flag, the il path equals the
  torch path at tiny and real geometry, 1/32/128 rows, the final mixer, a layer chain in copy modes 0/1/2 with the
  shipped ops' copies, the fallback and its one line, the once-per-layer decision, the kernel-library question);
  `test_op_callbacks_host.py` 127 passed (the real `hpu_hc_fused.cpp` against the stub header: the il ops' output
  metadata, GUIDs and node parameters for every HPU case); `test_sources.py`, `test_moe_sources.py` updated.
- Card (not yet run): `tests/unit_tests/hc_fused/test_hpu_op_il.py` (every HPU case of both kinds against the oracle
  and the shipped op on the card, a hyper-connection chain at 4/32/96/128 rows against the shipped chain, the ops in
  an HPU graph replayed).

## 5. Card plan

1. 1-card job (`any:1`, about 10 minutes): `bash tools/flash_next/hc_kernels/card_1card.sh`.
2. Tiny TP8 token gate with `VLLM_HPU_HC_FUSED=1 VLLM_HPU_HC_FUSED_IL=1` (tokens equal the CPU reference).
3. Build-6 arm: build 5's flags plus `VLLM_HPU_HC_FUSED_IL=1`; read the 32-stream MTP step and the device trace.

## 6. GDN verify kernel (`gdn_fused_core_verify`, Goal 2)

Agreed with kernel-fixes (09:00 UTC): the work is this track's, on this branch. Conditions: a default-off flag; the
interleave flag keeps its meaning and the change goes into the INTERLEAVED build (the one served, with
`VLLM_HPU_GDN_GLUE_CACHE=1` and `VLLM_HPU_GDN_CORE_VERIFY=1`); bit-identical to today on every case of
`tests/standalone/gdn_core_decode/test_simulation.py` and `tests/unit_tests/gdn_core_decode`; the diff to kernel-fixes
before any card job; device traces (not the 00090 bench, which runs 8 in-place calls on one pool per graph) for
absolute times; no second op on one pool in one graph (rule 10); slot and load/store semantics unchanged.

**Why it is slow** (MEASURED, 00205 dense trace): 36 calls per step, 262 us each, 9.45 ms per step, on 16 of the 24
TPC engines, 4 points each. The glue's index space is `[key heads = 2, requests = 32]` at TP8; the graph compiler cut
only the request dimension (2 requests per engine). VERIFIED in the simulator: the interleaved kernel runs about 108K
instructions per (key head, request) point at T = 4 (from 134.9K at T = 5), and 4 points x 108K at 1.7 GHz = 254 us
against 262 MEASURED: the same model as the HC kernels holds. The hot loop (4 value rows x 1 node) is 226 words with
204 VPU operations (90 %): VPU-bound, and three horizontal reductions per row and node (the row's v element, the
projection, the output) are most of it.

**Variants** (`VLLM_HPU_GDN_VERIFY_VARIANT`, only with `VLLM_HPU_GDN_KERNEL_INTERLEAVE=1`; read by the glue like the
interleave variable; each an optional ELF of the library, weak symbols):
- 1, request-major (`kernels/gdn_core_verify_interleaved_rmajor_gaudi2.c`): index space `[requests, key heads]`; the
  kernel reads the request and the key head from the swapped dimensions (`GDN_VERIFY_REQUEST_DIM`,
  `GDN_VERIFY_KEY_HEAD_DIM`), every per-point mapping of the glue goes through `requestDim` / `keyHeadDim`. Same
  points, same work, same instructions: what changes is only how the graph compiler splits the points over the TPCs
  (INFERRED: it cuts the outer dimension first; at 32 requests the hope is 24 engines x at most 3 points, 262 ->
  about 197 us; at 1 request 2 engines instead of 1). Only a card shows the split.
- 2, lean (`kernels/gdn_core_verify_interleaved_lean_gaudi2.c`, `GDN_VERIFY_LEAN_BROADCAST` in
  `gdn_core_verify_recurrence_interleaved.h`): the row's v element by `v_broadcast_element_f32(vo, row) + 0.0f`
  instead of `reduce_add` of a masked vector (the sum adds +0.0 lanes, which turns -0.0 into +0.0: so does
  `+ 0.0f`), and the output sum broadcast by one shuffle of lane 0 (the reduction leaves the sum in lane 0 of every
  lane group, as the projection already relies on) instead of `v_broadcast_element_f32(out, 0)`. VERIFIED: hot loop
  226 -> 193 words (174 VPU), instructions 0.887 to 0.925 of the interleaved kernel's on the 14 verify cases.
- 3, both.
- 4, today's index space and the interleaved ELF, with the graph compiler asked to split the key-head dimension into
  partitions of one (`preferredSplitDim = 1`, every tensor dimension mapped to index dimension 0 declared
  `allRequired`, as `tpc_kernel_lib_interface.h` requires; the compiler "begins with this dimension with each
  partition size of 1 and may continue to other dimensions"): 64 points of one (key head, request) each, at most 3
  per engine on 24 engines if the compiler does what the header says. Added at kernel-fixes' review (09:25 UTC): the
  swap of variant 1 alone may still leave 2 requests x 2 key heads on an engine. A flat 1-D index space would need
  output access patterns that are not linear in the point index (a declared superset: overlapping output regions),
  so the documented split hint is the cleaner form.
- 5, as 4 with the lean ELF.

VERIFIED (simulator, `tests/standalone/gdn_core_decode/test_simulation.py -k variant`: 43 passed at 09:22 UTC for
variants 1-3, 71 passed at 09:35 UTC for variants 1-5): out, conv state and SSM state byte-identical to the interleaved
kernel on all 14 verify cases for each variant (4 and 5 keep today's index space; their split hint is for the graph
compiler only);
variant 1 swaps the index space and runs the same instruction count; variants 2 and 3 run 0.887 to 0.925 of the
instructions; the glue takes a variant only with the interleave variable at "1" and the value exactly "1", "2" or
"3". The library builds with the three optional ELFs (`test_kernel_build.py` 7 passed); CPU:
`tests/unit_tests/gdn_core_decode` every file green (`test_verify_variant_flag.py` 5 new).

Card A/B: `tools/flash_next/hc_kernels/card_gdn_verify.sh`: the verify op card test with variant 3; then
`bench_gdn_verify.py` per variant 0 to 5 at batch 1, 8, 32, T = 4: each call on its own pools and CHAINED (call i + 1's
`a` gets 1e-30 x one element of call i's output, so the compiler cannot overlap independent calls on the 8 idle
engines, kernel-fixes' point 3), the slope of two K, and the bits of variants 1 to 5 compared with variant 0's on the
card; then profiler traces (`HABANA_PROFILE=profile_api_light`) of variants 0, 1, 4 at batch 1 and 32 for the engines
per call (`tools/flash_next/hc_kernels/kernel_engines.py`).

Next candidates (not built): pack the four rows' projection (and output) reductions into one vector after their
cross-group stages (bit-identical if each lane's operand pairs stay those of `v_f32_reduce_add`; INFERRED about -20 %
on the hot loop); the value-head split of kernel-fixes' option (c) list is not needed while the variants above work.
