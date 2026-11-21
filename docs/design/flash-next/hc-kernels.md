# Hyper-connection kernels at verification rows (track hc-kernels)

Design note, report and handoff of the `hc-kernels` track (branch `feat/flash-next-gaudi-hc-kernels`, worktree
`/Volumes/Storage/Claude/gaudi-work/wt/hc-kernels`). The harness refuses report files for this agent, so this note is
the report. Labels: MEASURED (card), VERIFIED (simulator or CPU test named), INFERRED (arithmetic or reading).

## 0. Handoff (if this agent stops) — updated 12:01 UTC 2026-11-21

- **Goal 1:** the two hyper-connection TPC kernels (`hc_fused_gate_mix`, `hc_fused_combine_norm`) at 128
  verification rows (32 streams, MTP k = 3) from about 7 ms per step to about 2 ms. **Goal 2:** the GDN verify kernel
  `gdn_fused_core_verify` (9.45 ms per 128-row step): agreed with kernel-fixes at 09:00 UTC, reviewed by it at 09:25
  (variants 1-5); the fold build (variants 6/7) sent to it for review before its card run.
- First reading sent to the coordinator 08:25 UTC; plan approved 08:27 UTC (new GUIDs behind a default-off flag, the
  HC contract of `tests/unit_tests/hc_fused/hc_case.py` as the tolerance plus the exact-match fraction against the
  shipped kernels; the 8-card arm goes into build 6).
- **On the branch, pushed:** HC `3c2ffd72`, `6f5b9300`, `0183f133`, `6c62cc56` (interleaved kernels, ops, flag
  `VLLM_HPU_HC_FUSED_IL`, tests, bench); GDN `8ed20160` (variants 1-3), `b9ff6384` (variants 4/5, chained benches,
  traces, split card scripts), `aaf256ea` (merge of the program branch at `e5188341`); then the fold build
  (variants 6/7: `gdn_core_verify_recurrence_fold.h`, `GDN_VERIFY_BATCHED_GATES`), simulator 0.609-0.630 of the
  interleaved kernel's instructions, every verify case byte-identical.
- **PR step 1 (HC + GDN variants 1-5):** gate green on `aaf256ea` (113 files, 10:25 UTC); the program branch moved
  (PRs #24, #25), so `d07b2155` = `aaf256ea` + `4f070c75` was pushed as `feat/flash-next-gaudi-hc-kernels-pr1` and is
  gated from `reports/hc-kernels/gate-pr1` (log `reports/hc-kernels/logs/gate_pr1.log`); when green: `gh pr create
  --base feat/flash-next-gaudi --head feat/flash-next-gaudi-hc-kernels-pr1` (body `reports/hc-kernels/pr/
  pr_body_step1.md`), `gh pr merge --merge`, PR number to the coordinator. The fold build goes in a later PR.
- **Card:** job A = **00231** `hc-quick` (image `b9ff6384`, hold 12 min, after build 5 on card 6:
  `tools/flash_next/hc_kernels/card_hc_quick.sh`: HC op card test, the TP1 tiny gate with `VLLM_HPU_HC_FUSED_IL=1`
  (stands in for TP8, coordinator 09:33), the HC bench at 4/32/96/128 rows, traces at 128 rows). Job B =
  `gdn-variants` (`tools/flash_next/hc_kernels/card_gdn_verify.sh`, hold 35 min, placed by the coordinator about
  15:00-16:00 beside step-anatomy's 4-card trace): op card test with variant 6 first (the job stops there if it
  fails), then the chained bench with on-card bits against variant 0 for 0, 6, 1, 4, 7 (out and both pools after one
  call, and every set's pools and outputs after the whole chained recurrence of replays), the traces of 0, 6, 1, 4,
  7, then 2, 3, 5 and the op test with 3 (kernel-fixes reviewed `d528a38f` at 10:50 UTC: no objection, on these
  three conditions; if any bit differs on the card, 6/7 are not bit-identical and need the prompt-logprob accuracy
  check of `moe_rt_driver.py`'s ACC harness before anyone serves them). Its id: the latest
  `reports/hc-kernels/logs/farm_gdn-variants*.out` (00232 was the image without the fold build).
  When a job ends: `bin/farm fetch hc-kernels <id> <id>-<name>` if its waiter is gone; read `pytest_il.log`,
  `tiny.log`, `bench.log` (VERDICT), `gdn_bench_variant*.log` (us per call and "bit-identical" lines),
  `gdn_optest_variant6.log`; traces with `python3 tools/flash_next/hc_kernels/kernel_engines.py <trace> <kernel>`;
  send kernel-fixes the GDN result.
- **Job A 00231 ended 12:01 UTC, green** (section 5): op card test 41 passed, TP1 tiny gate PASSED, device time
  per call at 128 rows gate_mix 40.5 -> 15.2 us, combine_norm 29.8 -> 10.0 us (about -4.7 ms per 128-row step).
- **Next:** build-6 arm for the coordinator: `VLLM_HPU_HC_FUSED_IL=1` (with `VLLM_HPU_HC_FUSED=1`), card-proven; and
  `VLLM_HPU_GDN_VERIFY_VARIANT=6` or `7` (with `VLLM_HPU_GDN_KERNEL_INTERLEAVE=1`) once job B (00239) is green.
- Tools: `tools/flash_next/hc_kernels/sim_cost.py` (simulator instructions per index-space point, oracle check,
  bit-equality against another build), `tools/flash_next/hc_kernels/bench_hc.py` and `bench_gdn_verify.py` (1-card
  chained replay benches), `tools/flash_next/hc_kernels/kernel_engines.py` (per-engine time of a kernel in a
  profiler trace); scratch GDN driver `.hc_exp/gdn_b1_driver.py <kernels dir> <build dir>` (inside
  `tools/flash_next/tpcsim.sh`: a candidate tree's interleaved build against the repo's, every verify case, bytes and
  instruction counts).
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

## 5. Card results (job 00231 `hc-quick`, image `b9ff6384`, card 6, 11:57-12:01 UTC, exit 0, 211 s)

MEASURED, `reports/hc-kernels/card/00231-hc-quick/`:
- `tests/unit_tests/hc_fused/test_hpu_op_il.py`: **41 passed** (the interleaved ops against the oracle and the shipped
  ops on the card).
- Tiny-model token gate at TP1 with `VLLM_HPU_HC_FUSED=1 VLLM_HPU_HC_FUSED_IL=1 VLLM_HPU_HC_FUSED_COPY_MODE=2`
  (`--gdn hc`): **PASSED**, tokens match the reference, no fallback line (neither "HC fused ... is off" nor the
  interleaved one), decode 4.1 ms/token on the tiny model. Stands in for TP8 (coordinator, 09:33); build 6's own TP8
  tiny gates cover TP8.
- Parity on the card (`bench_hc.py`, against the shipped ops): gate_mix bit-equal 99.993 % (32, 96 rows) to
  99.995 % (128 rows); combine_norm 100 % / 99.9997 % to 100 %; oracle ok at 4, 32, 96, 128 rows.
- Device time per call at 128 rows (profiler traces, `kernel_engines.py`, 72 executions each):

  | kernel | shipped | interleaved | engines | ratio | simulator said |
  |---|---|---|---|---|---|
  | gate_mix | 40.5 us | 15.2 us | 24 / 24 | 2.66x | 2.70x |
  | combine_norm | 29.8 us | 10.0 us | 22 / 22 | 2.98x | 3.52x |

  The shipped numbers equal 00205's real-model trace (41.1 and 30.1 us). combine_norm_il's engines are unevenly
  loaded (per-engine busy 3.2 to 9.8 us: 128 one-token points over 22 engines), which is where its ratio falls
  short of the simulator's; an even split would give about 8.3 us (INFERRED).
- Per step at 128 rows (INFERRED from the MEASURED device times x 00205's 106 gate_mix and 101 combine_norm calls):
  7.30 ms -> 2.62 ms, **about -4.7 ms per 128-row step**.
- Wall-clock per hyper-connection (chained replay slope, gate_mix + combine_norm + the chain add, one card):
  29.6 -> 26.5 us at 4 rows, 44.6 -> 32.0 at 32, 77.4 -> 45.0 at 96, 98.7 -> 54.1 at 128. The per-op wall slopes of
  this bench are noisy at 6 iterations (one is negative); the traces above are the numbers to use.

Build-6 arm: build 5's flags plus `VLLM_HPU_HC_FUSED_IL=1` (it needs `VLLM_HPU_HC_FUSED=1`); read the 32-stream MTP
step and the device trace.

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

**How it scales with rows, and row parallelism** (INFERRED from the simulator model above, which matched the 00205
trace within 3 %): one point is one (key head, request) over its T nodes and 3 value heads (TP8); its time does not
depend on the batch. A call takes ceil(points / engines used) point times: at T = 4, 4 rows = 1 request = 2 points
(2 engines busy, one point time, about 64 us for the interleaved kernel); 32 rows = 16 points (one point time); 96
rows = 48 points (2 to 3 point times); 128 rows = 64 points on 16 engines (4 point times = the 262 us MEASURED). Two
levers, independent of each other: the point time (the fold build, x 0.62 in the simulator) and the points per
engine (the index-space variants 1, 4, 7: 4 -> 3 at 128 rows if the compiler spreads the points over 22 to 24
engines, x 0.75). The index space is row-parallel already across requests; within a request the T nodes are a
recurrence (each node's state is the previous node's), so they cannot run on separate engines without another
algorithm (the chunked delta rule: not bit-identical, and its intermediate states would still be needed for every
node's slot). The one finer split left is the value heads of a key head (kernel-fixes' option (c)): 3x the points at
a third of the work each, which matters most at few requests (1 request: 6 engines instead of 2; INFERRED about
-27 % at 32 and 128 rows too, 192 points = 8 per engine on 24), at the price of the q/k conv and norm repeated per
value head. It is not safe as the kernel stands: the q/k conv history is shifted in place (conv load slot = store
slot, the normal case), so a value-head point running after another one on its key head would read a history that
point already shifted. It needs the q/k conv-state update moved out of the points first (a separate op after the
verify, or the history read by every point before any writes, which one kernel launch cannot order).

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
- 6, fold (`kernels/gdn_core_verify_interleaved_fold_gaudi2.c`, recurrence `gdn_core_verify_recurrence_fold.h`), and
  7, the fold ELF with the split of 4. Today's index space. Row group (side, j) carries the four value rows
  `side * 64 + 16 * d + j`, d = 0..3, whose v and o elements sit at lane `16d + j` of the half's vector, the same
  place in every dual group. The two sums of each row per node are no longer four separate `v_f32_reduce_add`
  calls (12 VPU operations each plus a broadcast): `gdn_verify_fold4` folds the four product vectors into one vector
  whose dual group d holds row d's sum, by the very additions of `v_f32_reduce_add` (VERIFIED in the simulator with a
  scratch probe: `v_f32_reduce_add` is lane i + lane i ^ 16, then i ^ 32, then i ^ 8, then i ^ 1, i ^ 2, i ^ 4 in
  that order, every lane ending with the sum; the fold matches it bit for bit in every lane of 4,000 random
  quadruples with -0.0 elements). Lane bit 4 is folded for two pairs of vectors at once (x1 plus its partners in all
  lanes, then x0 plus its partners written with a vector predicate over the lanes with the bit clear), bit 5 pairs the
  two results the same way, and the four in-dual-group stages run once for the four rows: 20 operations for four sums
  instead of 52. v of the four rows is one shuffle (the shuffle reads lanes 0-7 of a dual group, so rows with j >= 8
  first move group 1 onto group 0), delta is computed for the four rows in one vector, broadcast per row by one
  dual-group move, and o lands in the half's vector by one masked move. Every row's arithmetic and its order are the
  one-row loop's. The code is laid out for tpc-clang's one-block scheduling, which pays a dependent chain's whole
  latency: node t's output and node t + 1's update depend only on the state after node t, so one block runs both;
  and the 32 row groups alternate between two register sets, so a group's last output runs in one block with the
  next group's first update (the first two versions without these overlaps were latency-bound: 0.95 of the
  interleaved kernel's instructions, 211 words for 107 operations in the node loop). Node t + 1's v and k are read
  before node t's output is stored (tpc-clang does not tell the `__local__` arrays apart); an output is merged into
  a fresh read of the v/o vector, since with one node the next group's update reads that vector before this group's
  output is stored (the stale copy broke 4 of the 14 cases in the simulator before this rule); every move writes all
  lanes, so its income is a dead register (no copy). The gates (`GDN_VERIFY_BATCHED_GATES`, defined only by the fold
  wrapper; the other builds compile to the same code as before, checked on their assembly): the inputs of all nodes
  go into lane t of a vector, the softplus/exp/sigmoid chain runs once instead of once per node; every function is
  lane-wise, so lane t holds the per-node loop's bits.
  VERIFIED (scratch driver `.hc_exp/gdn_b1_driver.py`, every verify case): bytes identical to the interleaved kernel
  on all 14 cases; instructions 0.609 to 0.630 of the interleaved kernel's at T = 1 to 6 (lean: 0.887 to 0.925); hot
  node loop 121 words for 4 rows x 1 node (89 vector operations) against 193 (lean) and 226 (interleaved); vector local
  memory 13,312 bytes of 16,384. INFERRED: at the 00205 geometry (TP8, 128 rows, 4 points per engine) 262 us per
  call x 0.62 = about 163 us, -99 us per call, about -3.6 ms per 128-row step over 36 calls, if the card follows the
  simulator as the other kernels did (job B measures it). Tried and dropped: packing the four sums after their
  cross-group stages (0.994: spills and one long serial chain); the q/k normalizations and the gated RMSNorm four
  nodes at a time (bit-identical, but 0.615 to 0.695: the clamped extra nodes cost more than the chains saved).

VERIFIED (simulator, `tests/standalone/gdn_core_decode/test_simulation.py -k variant`: 43 passed at 09:22 UTC for
variants 1-3, 71 passed at 09:35 UTC for variants 1-5, 59 passed at 10:53 UTC for variants 6/7 and the value rules):
out, conv state and SSM state byte-identical to the interleaved kernel on all 14 verify cases for each variant (4, 5
and 7 keep today's index space; their split hint is for the graph compiler only; 6 and 7 run 0.609 to 0.630 of the
instructions); the library build carries the fold ELF (`test_kernel_build.py` 7 passed, `_binary_gdn_core_verify_
interleaved_fold_gaudi2_o_start` defined);
variant 1 swaps the index space and runs the same instruction count; variants 2 and 3 run 0.887 to 0.925 of the
instructions; the glue takes a variant only with the interleave variable at "1" and the value exactly "1" to "7". The library builds with the three optional ELFs (`test_kernel_build.py` 7 passed); CPU:
`tests/unit_tests/gdn_core_decode` every file green (`test_verify_variant_flag.py` 5 new).

Card A/B: `tools/flash_next/hc_kernels/card_gdn_verify.sh`: the verify op card test with variants 6 and 3; then
`bench_gdn_verify.py` per variant 0 to 7 at batch 1, 8, 32, T = 4: each call on its own pools and CHAINED (call i + 1's
`a` gets 1e-30 x one element of call i's output, so the compiler cannot overlap independent calls on the 8 idle
engines, kernel-fixes' point 3), the slope of two K, and the bits of variants 1 to 7 compared with variant 0's on the
card; then profiler traces (`HABANA_PROFILE=profile_api_light`) of variants 0, 1, 4 at batch 1 and 32 for the engines
per call (`tools/flash_next/hc_kernels/kernel_engines.py`).

Next candidates (not built): the conv blocks and the gated RMSNorm per (node, value head), now a larger share; two
nodes' norms in one block instead of four clamped ones. The value-head split of kernel-fixes' option (c) list is not
needed while the variants above work.
