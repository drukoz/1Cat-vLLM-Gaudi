# Hyper-connection kernels at verification rows (track hc-kernels)

Design note, report and handoff of the `hc-kernels` track (branch `feat/flash-next-gaudi-hc-kernels`, worktree
`/Volumes/Storage/Claude/gaudi-work/wt/hc-kernels`). The harness refuses report files for this agent, so this note is
the report. Labels: MEASURED (card), VERIFIED (simulator or CPU test named), INFERRED (arithmetic or reading).

## 0. Handoff (if this agent stops) — updated 09:00 UTC 2026-11-21

- **Goal 1:** the two hyper-connection TPC kernels (`hc_fused_gate_mix`, `hc_fused_combine_norm`) at 128
  verification rows (32 streams, MTP k = 3) from about 7 ms per step to about 2 ms. **Goal 2** (GDN verify kernel
  `gdn_fused_core_verify`) only after agreeing with kernel-fixes (agent a3fffbeea01459cbb); not started.
- First reading sent to the coordinator 08:25 UTC; plan approved 08:27 UTC (new GUIDs behind a default-off flag,
  the HC contract of `tests/unit_tests/hc_fused/hc_case.py` as the tolerance plus the exact-match fraction against
  the shipped kernels; the 8-card arm goes into build 6).
- **Built (this branch):** kernels `hc_fused_gate_mix_il` and `hc_fused_combine_norm_il` (section 2), optional
  members of `libhc_fused_gaudi2.so`; ops `torch.ops.flashinfer_gaudi.hc_gate_mix_il` / `hc_combine_norm_il`;
  flag `VLLM_HPU_HC_FUSED_IL=1` (with `VLLM_HPU_HC_FUSED=1`, default 0); simulator, CPU and card tests; the bench.
- **Next:** commit + push; 1-card job (`tools/flash_next/hc_kernels/card_1card.sh`: il op card test, shipped op card
  test, bench at 4/32/96/128 rows) and send its id to the coordinator; then a tiny TP8 token gate with the flag;
  then the build-6 arm (`VLLM_HPU_HC_FUSED_IL=1` on top of build 5's flags). Then Goal 2 with kernel-fixes.
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
