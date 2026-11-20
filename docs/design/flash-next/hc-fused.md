# Fused hyper-connection kernels: contract and integration notes

Version 2, 2026-11-19. Six Gaudi2 TPC kernels that replace the element-wise torch code of the Flash-Next
"hyper-connection" block (`vllm_gaudi/models/qwen4_exp/hyperconnection.py`, `HpuGatedResidual`). The two GEMMs of a
block (the merged low-rank down+inject projection and the up projection) stay on the matrix engine; everything between
and around them becomes one kernel each. The kernels pass on the Gaudi2 instruction simulator against the torch oracle;
nothing here has run on a card yet ("Unproven without a card" lists what that leaves open).

The kernels are wired into the plugin behind the flag `VLLM_HPU_HC_FUSED`, **default 0**: the layer, the loader, the
build, the combined kernel database, the rental gate and the smoke test. With the flag at 0 nothing of this is loaded,
registered or logged, and the torch path is the code it was. With the flag at 1 every piece falls back to the torch
path by itself and none can stop a run. "What is wired" below describes each piece.

The same library and the same op extension also carry the four **MoE glue kernels** (flag `VLLM_HPU_MOE_GLUE`,
default 0). They are described in the second part of this page, ["MoE glue kernels"](#moe-glue-kernels-vllm_hpu_moe_glue);
the first part is about the hyper-connection kernels only, and nothing in it changed for them: their six GUIDs keep
the first six places of `GetKernelGuids`, and a MoE kernel that does not build cannot take them away.

## Why

A decode step on Gaudi2 is dominated by fixed per-node cost, not by compute. One `combine_and_mix` call runs 41 torch
ops besides its two GEMMs; a decoder layer calls it twice.

Torch ops dispatched by one decode-shaped call (1 token, hidden 2560, hc_count 4, hc_lowrank 320), counted with a
`TorchDispatchMode` on CPU (`tests/unit_tests/hc_fused/test_fused_path.py::test_op_count_before_and_after`):

| Call | Torch path: total (GEMM ops + other) | Fused path: total (GEMM ops + fused ops) |
|---|---|---|
| `mix` | 32 (4 + 28) | 7 (4 + 3) |
| `combine_and_mix` | 45 (4 + 41) | 7 (4 + 3) |
| `combine_and_mix`, final mixer (`use_combine=False`) | 44 (4 + 40) | 7 (4 + 3) |
| `combine` | 13 (0 + 13) | 1 (0 + 1) |
| **one decoder layer** (2 x `combine_and_mix`) | **90 (8 + 82)** | **14 (8 + 6)** |
| one decode step (48 layers, PLE at layer 2, final mixer), GEMM ops left out | 3963 | 292 |

The 4 GEMM ops per call are `t` + `mm`, twice (vLLM's linear on CPU); they are the same in both paths. The count is of
dispatcher ops on CPU. On the card the graph compiler may drop some views and casts of the torch path, so the saving in
graph nodes is at most these numbers; the fused numbers are exact (one node per op).

## Names

| What | Name |
|---|---|
| Ops | `torch.ops.flashinfer_gaudi.hc_norm`, `hc_silu`, `hc_silu_split`, `hc_gate_mix`, `hc_combine`, `hc_combine_norm` |
| Schemas and fake kernels (the only definition) | `flashinfer_gaudi/hc_fused.py`; imported only with the flag at 1 (or with `VLLM_HPU_MOE_GLUE=1`, whose ops the same extension implements; importing it defines the MoE glue schemas too). It registers no CPU kernel |
| CPU stand-in for the tests (the oracle as the ops' CPU kernels) | `tests/unit_tests/hc_fused/hc_ops.py` |
| Torch oracle (verbatim copies of the plugin's functions) | `tests/unit_tests/hc_fused/hc_reference.py` |
| Fused path of `HpuGatedResidual` | the layer itself (`_hc_fused_mix`, `_hc_fused_combine_and_mix`, `_hc_fused_mix_tail`); `tests/unit_tests/hc_fused/fused_path.py` calls them directly |
| TPC kernels | `csrc/hc_fused/kernels/hc_fused_{norm,silu,silu_split,gate_mix,combine,combine_norm}_gaudi2.c` + `hc_fused_common.h`, `hc_fused_silu.h` |
| GUIDs | `hc_fused_norm_bf16_gaudi2`, `hc_fused_silu_bf16_gaudi2`, `hc_fused_silu_split_bf16_gaudi2`, `hc_fused_gate_mix_bf16_gaudi2`, `hc_fused_combine_bf16_gaudi2`, `hc_fused_combine_norm_bf16_gaudi2` (`csrc/hc_fused/hc_fused_params.h`) |
| Kernel database | `libhc_fused_gaudi2.so` (CMake target `hc_fused_gaudi2`, `csrc/hc_fused/CMakeLists.txt`); its own library, nothing shared with `libgdn_fused_gaudi2.so` |
| Host glue | `csrc/hc_fused/host/hc_fused_gaudi2.cpp` (validation, index space, access patterns, ELF), `host/entry_points.cpp` |
| PyTorch extension | `hc_fused_ops*.so` (`csrc/hc_fused/pytorch/hpu_hc_fused.cpp`, `setup.py`); registers only the `HPU` kernels |
| Loader | `flashinfer_gaudi/_native.py::load_hc_fused_extension`, `hc_fused_extension_errors`, `hc_fused_extension_loaded`; overrides `HC_FUSED_GAUDI_KERNEL_LIBRARY`, `HC_FUSED_GAUDI_NATIVE_LIBRARY` (`tests/unit_tests/hc_fused/hpu_loader.py` forwards to it) |
| Flag | `VLLM_HPU_HC_FUSED`, default 0 (`vllm_gaudi/envs.py`) |
| Build switch | `VLLM_GAUDI_BUILD_HC_FUSED=1`, or `tools/build_flashinfer_gaudi.py --hc-fused-only` |
| Operator switch in the rental | `run_rental.sh --hc-fused`; smoke mode `tiny_smoke.sh --gdn hc` |

## Where each kernel sits and what it replaces

```
mix(h):                                      combine_and_mix(h, block_out, inj):
  xn = hc_norm(h, w)                [K1]       h', xn = hc_combine_norm(h, block_out, inj, w)   [K5]
  down = GEMM_down_inject(xn)       (MME)      down = GEMM_down_inject(xn)                      (MME)
  lora, inj' = hc_silu_split(down)  [K3]       lora, inj' = hc_silu_split(down)                 [K3]
  gate = GEMM_up(lora)              (MME)      gate = GEMM_up(lora)                             (MME)
  block_in = hc_gate_mix(xn, gate)  [K4]       block_in = hc_gate_mix(xn, gate)                 [K4]
  return h, block_in, inj'                     return h', block_in, inj'

combine(h, block_out, inj):  return hc_combine(h, block_out, inj)   [K6]
final mixer (use_combine=False): K5, GEMM_down, hc_silu [K2], GEMM_up, K4
```

| Kernel | Replaces | Torch ops it removes per call |
|---|---|---|
| K1 `hc_norm` | `grouped_gemma_rmsnorm_torch` | 13 |
| K2 `hc_silu` | `hc_silu_torch` (final mixer) | 5 |
| K3 `hc_silu_split` | `down.split(...)` + `hc_silu_torch` | 6 |
| K4 `hc_gate_mix` | `hc_gate_mix_torch` | 9 |
| K5 `hc_combine_norm` | `hc_combine_norm_torch` (combine, round, grouped norm) | 26 |
| K6 `hc_combine` | `hc_combine_torch` | 13 |

(`mix` = 13 + 6 + 9 = 28, `combine_and_mix` = 26 + 6 + 9 = 41, final mixer = 26 + 5 + 9 = 40.)

Three kernels per hyper-connection is the least the data flow allows with the GEMMs on the matrix engine: K5 must end
before the down GEMM (which reads `xn`), K3 sits between the two GEMMs, K4 needs the up GEMM's output, and the block
(attention or MLP) runs between K4 and the next K5.

## Schemas

```
hc_norm(Tensor x, Tensor norm_weight, float eps, int hc_count) -> Tensor
hc_silu(Tensor down, int hc_count) -> Tensor
hc_silu_split(Tensor down, int hc_count, int lora_rank) -> (Tensor, Tensor)
hc_gate_mix(Tensor xn, Tensor gate, int hc_count) -> Tensor
hc_combine(Tensor residual, Tensor block_output, Tensor injection, int hc_count) -> Tensor
hc_combine_norm(Tensor residual, Tensor block_output, Tensor injection, Tensor norm_weight,
                float eps, int hc_count) -> (Tensor, Tensor)
```

`T` = tokens (rows), `HC` = `hc_count`, `H` = `hidden_size`, `W = HC * H` (stream-major: stream `h` owns columns
`[h*H, (h+1)*H)`, the checkpoint-native layout), `R` = `hc_lowrank`. No op writes an input; every output is a new tensor.

| Arg | Torch shape, dtype (HPU) | TPC dims (fastest first) | Meaning |
|---|---|---|---|
| `x`, `xn`, `gate`, `residual` | `[T, W]`, bf16 | `{W, T}` | multi-stream tensors |
| `norm_weight` | `[W]` (`hc_per_branch_norm`) or `[H]` (shared), bf16 | `{W}` or `{H}` | Gemma affine; the kernel tells the two apart by the size |
| `down` | `[T, C]`, bf16; `hc_silu`: `C = R`; `hc_silu_split`: `C >= R + HC` (336 = 320 + 4 + 12 padding on the real model) | `{C, T}` | down GEMM output |
| `block_output` | `[T, H]`, bf16 | `{H, T}` | pending block output |
| `injection` | `[T, HC]`, bf16 | `{HC, T}` | injection logits |
| `hc_norm` -> | `[T, W]` bf16 | | grouped Gemma RMSNorm |
| `hc_silu` -> | `[T, R]` bf16 | | `xf * sigmoid(xf)`, `xf = down / HC` |
| `hc_silu_split` -> | `([T, R], [T, HC])` bf16 | | the same on `down[:, :R]`, and the exact bits of `down[:, R:R+HC]` |
| `hc_gate_mix` -> | `[T, H]` bf16 | | `sum_h sigmoid(gate_h) * xn_h / HC` |
| `hc_combine` -> | `[T, W]` bf16 | | `residual_h + block_output * 2 * sigmoid(injection_h / HC)` |
| `hc_combine_norm` -> | `([T, W], [T, W])` bf16 | | that combine, and the grouped norm of the **rounded** combine |

HPU gates, checked by the op's validation callback (`TORCH_CHECK`) and again by the glue (`GlueCodeReturn`): rank-2
contiguous bf16 activations with `T > 0`, a rank-1 bf16 norm weight of `H` or `W` elements, `W % HC == 0`,
**`H % 128 == 0`** (the kernels walk a stream in whole 128-lane bf16 vectors; 2560 and the tiny model's 128 pass),
`1 <= HC <= 64`, `eps` a positive normal fp32, `1 <= R` and `R + HC <= C`. Nothing else is fixed: `T`, `H`, `HC`, `R` and
`C` come from the tensor dims and the node parameters. `R` need not be a multiple of 128 (320 and 16 are not).
The package registers no CPU kernel. The test stand-in's CPU kernels are the oracle functions and take any float
dtype and any leading dims.

## Semantics and rounding points

The definition is the torch code (`hc_reference.py`, verbatim copies of the plugin functions, tied to the originals
by `test_reference_copy.py`). The kernels keep its structure:

- fp32 inside, one conversion to bf16 with round-half-to-even (`SW_RHNE`) where the oracle has `.to(input_dtype)`.
- Every product and every sum is its own fp32 instruction (`v_f32_mul_b`, `v_f32_add_b`, `#pragma clang fp
  contract(off)`), never a fused multiply-add, because the oracle rounds each one.
- **`hc_combine_norm` rounds the combine to bf16 before it normalizes**, as the oracle does: the mean square and the
  normalized values are computed from the rounded bf16 chunk. `residual_out` is never read back (the compiler may
  reorder a load and a store of one tensor, `docs/design/flash-next/tpcsim.md`); the second pass computes the chunk
  again from the inputs with the same instructions.
- `sigmoid(x) = 1 / (1 + exp(-x))` as `v_reciprocal_f32(1 + v_exp_cephes_f32(-x))`, **not** `v_sigmoid_f32`. The
  built-in is `0.5 * tanh(0.5 * x) + 0.5` (its source is in the compiler's built-in header), which cancels for
  negative `x`: on the simulator it failed the stress cases (`gate_mix_real_stress`: 91 % exact, largest error 8.5e6 on
  inputs of 1e15; `silu_*_stress`: 94 % exact). With the exp form the silu stress cases are bit-exact and
  `gate_mix_real_stress` differs in 5 of 7,680 elements, all of denormal size (largest difference 7.9e-39). It also
  saturates exactly: 0.5 at 0, 1 for large `x`, 0 once `exp(-x)` overflows.
- The mean of squares is `sum * (1 / H)` and the `/ HC` is `* (1 / HC)`, with both reciprocals computed on the host
  (no kernel does a floating-point division). For `HC = 4` the second is exact; the first differs from a division by
  at most one fp32 step.
- Sums over the streams (`hc_gate_mix`) run in stream order, as torch's sum over the stream dimension does.

## Kernel layout

Scalar arguments of every kernel's `main`, in this order: `float eps, float inv_group, float inv_hc, int hc_count`
(`HcFusedKernelArgs`, 16 bytes; the glue derives `inv_group = 1/H` and `inv_hc = 1/HC`). Node parameters (op -> glue):
`struct HcFusedParams { float eps; int32_t hc_count; int32_t lora_rank; }`, 12 bytes, the same for all six GUIDs
(unused fields are 0).

| Kernel | Tensors (in + out) | Index space | One point owns | Access patterns (dim 0; dim 1 is always the token, `a = 1`) |
|---|---|---|---|---|
| `hc_norm` | 2 + 1 | `{HC, T}` | one stream of one token | `x`, `xn`: `a = H, 0..H-1`; weight: `a = H` (per branch) or `a = 0` (shared), `0..H-1` |
| `hc_silu`, `hc_silu_split` | 1 + 1, 1 + 2 | `{T}` | one token | every tensor: whole row (`a = 0`) |
| `hc_gate_mix` | 2 + 1 | `{H/128, T}` | one 128-column chunk of one token | output: `a = 128, 0..127`; `xn`, `gate`: whole row (`a = 0`), because one point reads its chunk of every stream and one linear range cannot describe that |
| `hc_combine` | 3 + 1 | `{HC, T}` | one stream of one token | `residual`, output: `a = H, 0..H-1`; `block_output`: whole row; `injection`: `a = 1, 0..0` |
| `hc_combine_norm` | 4 + 2 | `{HC, T}` | one stream of one token | as `hc_combine`, weight as `hc_norm`, both outputs `a = H, 0..H-1` |

No two points write the same element. A decode step of 1 token therefore gives the graph compiler 4 points
(`hc_norm`, `hc_combine*`) or 20 points (`hc_gate_mix`) to spread over the TPC engines.

Measured on the simulator, real geometry, 1 token: `hc_combine_norm` 8,322 instructions, `hc_silu_split` 545,
`hc_gate_mix` 13,458, `hc_norm` 4,814, `hc_combine` 2,598. One `combine_and_mix` is about 22,000 TPC instructions in
three nodes.

`tpc-clang` prints `Performance warning: number of requested special function IDs exceeds LUT cache capacity for the
Gen2+ architecture` for `hc_fused_combine_norm_gaudi2.c` (it uses the reciprocal and the rsqrt tables), as it does for
the proven `gdn_core_decode_gaudi2.c`. Both special functions run once per stream here (once per 2560 elements), not
per element, so the cost of the table misses should be small; it is not measured. The other five kernels compile
without the warning.

## TPC rules this relies on, and where each was checked

| Rule | Source | Use here |
|---|---|---|
| A TPC-C program can access at most 16 tensors (15 with printf) | Intel Gaudi docs, TPC User Guide, "TPC Programming Model", General Limitations of TPC-C (docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Programming_Model.html). The same page also says "up to eight tensors" in "TPC Program Inputs/Outputs"; the 13-tensor GDN decode kernel compiles with this toolchain and runs on a card, so 16 is the working limit. `tpc_kernel_lib_interface.h` in the container: `inputsReusability :16`, `validInputTensors` is a 16-bit mask | at most 6 tensors per kernel, under both numbers (`test_sources.py` counts them) |
| Index space has 1 to 5 dimensions | same page, "Index Space"; header: `MAX_INDEX_SPACE_DIM_SIZE = 5` | rank 1 and rank 2 |
| Tensors have 1 to 5 dimensions, dim 0 is the fastest | same page, "TPC Program Inputs/Outputs", "Data Layout"; `int5` coordinates (TPC-C Language, Built-in Types) | rank 1 and rank 2, `{columns, tokens}` |
| Access pattern: per tensor dim, index point `i` touches `a*i + start_b .. a*i + end_b` | same page, "Index Space Mapping"; header: `DimIndexSpaceMapping {indexSpaceDim, a, start_b, end_b}` | the table above. Every simulator case runs with the simulator's index-space mapping test on (`e_accessPatternDefaultMode`) and its partition shuffle on, and passes. Declaring a whole row for an input that is read at several offsets is what `gdn_core_decode_gaudi2.cpp` does for `conv_state` |
| A read outside a tensor returns the pad value; a write outside is culled, on dim 0 partially | same page: "If the cartesian coordinates of a write operation fall outside the written tensor, the write operation is culled by the memory control unit. On dim0 the write operation is partially culled if some values fall inside the tensor and some fall outside of it." | `hc_silu*`: the last 128-lane store of `lora_out` keeps only the lanes inside `R`, and the injection copy keeps `HC` lanes. Simulator cases with `R` = 16, 320 and 130; `sim_host` checks guard bytes around every output |
| Output in place of an input | header: `inputsReusability`, "specifies if the output tensor can be nested on the same memory as the input"; docs (TPC-C Language): "Global memory is not coherent. Call the `aso` intrinsic when performing a read-after-write operation"; `tpcsim.md`, "Known compiler hazard: in-place tensors" | not used: the glue zeroes every access pattern (no reuse bit), no kernel reads a tensor it writes, `sim_host` checks that no input changed |
| bf16 vector types and conversions | `bfloat128` = 128 x 2-byte, `float64` = 64 x fp32 (TPC-C Language, Built-in Types). `v_convert_bf16_to_f32_all_b`, `v_convert_f32_to_bf16_all_b(..., SW_RHNE)`, `SW_RHNE = 0x0 << 16`: the built-in header inside `tpc-clang` 2.7.33 (`strings /usr/bin/tpc-clang`) and the card-proven `csrc/deepseek_v4/include/deepseek_v4_mhc_post_prepare.h` | every load and store. Round-half-to-even is pinned by the two `*_edge` cases (exact ties, bit-exact). The fp32 lane order of the conversion is not linear; every operand goes through the same conversion and the only reduction is a sum, so it does not matter |
| Vector local memory: 80 KB, 16 KB once the program uses special functions | TPC-C Language, "Local Memory Space": "The local memory size for vector types is either 16 KB or 80 KB. If the program uses special functions, the available VLM size is reduced to 16 KB"; `tpc-clang --help`: `-vlm <value>  Amount of vector local memory in KB` | no `__local__` data and no vector arrays in any kernel; compiled without `-vlm` (`test_hc_simulation.py`, `test_sources.py`) |
| Special functions | built-in header source: `v_sigmoid_f32` is `0.5*tanh(0.5x)+0.5`; `v_exp_cephes_f32` is a polynomial; `v_reciprocal_f32` and `v_rsqrt_f32` are table based and handle 0 and infinity | see "Semantics" |
| At most 32 scalar parameters, GUID under 64 bytes | header: `MAX_SCALAR_PARAMS = 32`, `MAX_NODE_NAME = 64` | 4 scalars; longest GUID 33 characters |
| `GetSupportedDataLayouts`: an all-`x` layout means layout agnostic; zero layouts or a failure fails the graph compiler's pass | header comment on `pfnGetSupportedDataLayout`; `csrc/kernel_db_adapter/adapter.cpp`; `csrc/deepseek_v4/host/entry_points.cpp` | exported and answered for the six GUIDs; checked directly and through the real adapter (`test_hc_library.py`) |
| `half` is a reserved word of tpc-clang | `tpcsim.md` | not used |

## Tests

All commands run from the repository root.

**Simulator** (tpcsim container, `docs/design/flash-next/tpcsim.md`; also runs in the rental's stage 1 container):

```bash
tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest \
    tests/standalone/hc_fused/test_hc_simulation.py tests/standalone/hc_fused/test_hc_library.py'
```

(The folder `tests/standalone/hc_fused` also holds the MoE glue kernels' simulator tests; the whole folder is 136
tests.) `test_hc_simulation.py` builds `csrc/hc_fused` with its real CMake project, links `sim_host.cpp` against
`libhc_fused_gaudi2.so`, and runs the 56 cases of `hc_case.SIM_CASES` through the library's exported entry points
(`GetShapeInference`, `InstantiateTpcKernel` with the two-call ELF handshake) and the instruction simulator. It also
compiles each kernel alone with exactly `tpc-clang -Wall -Werror -O2 -march=gaudi2` and checks that the CMake build log
has those flags and no `-vlm`. `test_hc_library.py` runs `lib_check.cpp` (`dlopen`, every entry point, every
rejection) and the same checks through the real combined kernel database with our library as a member: once
named by the members override and once as the installed sibling (no override, the adapter finds it by name). It
also runs the build script's `check_kernel_library` on the built library. 71 tests in the two files.

Cases (`tests/unit_tests/hc_fused/hc_case.py`): per kernel, tiny geometry (hidden 128, 1 and 3 rows), real geometry
(hidden 2560, rows 1, 2, 16, 32), prefill size (2048 rows), both norm-weight layouts, stress inputs (sigmoid arguments
from 0 to +-1e30; groups with mean square 0, far below eps, about eps, one spike, 1e30), bf16 rounding edges (exact
ties, bit-exact required), the rounding-point pin, and geometries with `HC` 2 and 3 and `R` = 130.

**CPU** (venv), one file per process: `python -m pytest -q -p no:cacheprovider <file>`.

| File | What it holds |
|---|---|
| `tests/unit_tests/hc_fused/test_reference_copy.py` | the copied oracle functions have the plugin's source text and return its bits |
| `.../hc_fused/test_hc_case.py` | every case's expected output passes; raw files round-trip; `compare` has teeth (2 steps, cancellation bound, propagation rule, exactness, unwritten output); the unrounded-norm result is rejected |
| `.../hc_fused/test_hc_ops.py` | schemas; the package module is the only definition, registers no CPU kernel and is not imported by `import flashinfer_gaudi`; the stand-in's CPU kernels equal the oracle; fake kernels give the shapes |
| `.../hc_fused/test_fused_path.py` | the layer's fused methods, called directly, return exactly what the torch methods return; op counts before and after |
| `.../hc_fused/test_op_callbacks_host.py` | `hpu_hc_fused.cpp` compiled against the stub `hpu_custom_op_pt2.h` under ASan + UBSan: output metadata, GUID and the 12 parameter bytes of all 81 HPU cases, 12 rejections |
| `.../hc_fused/test_hpu_loader.py` | `hpu_loader.py` is the package loader, not a second one |
| `.../hc_fused/test_build_script.py` | the build step: on request only, after the GDN step and before the combined database; a failure removes its own two files only and the install goes on; `--hc-fused-only` |
| `.../hc_fused/test_sources.py` | GUIDs, ABI and scalar order agree across the C++, TPC-C and Python sources |
| `.../hc_fused/test_hpu_op.py` | card only: see "Rental gate" |
| `tests/unit_tests/gdn_core_decode/test_hc_fused_loader.py` | the loader: ordering, `GC_KERNEL_PATH`, recipe cache, combined database (serving, old, switched off, foreign path), env overrides, failures, the status field |
| `tests/unit_tests/gdn_core_decode/test_hc_fused_startup.py` | the flag; `register_ops` calls the loader only with the flag at 1, after the GDN loader and before the status line; the one startup warning |
| `tests/unit_tests/models/flash_next/test_hc_fused_layer.py` | the layer with the flag: read once at construction; fused equals torch bit for bit (both norm-weight layouts, tiny and real geometry, 1 and 32 rows, 2048 rows, the final mixer, a chain of layers); the wiring through a stand-in that records every call and refuses what the HPU validation refuses; the rounding point (a stand-in that normalizes the unrounded combine differs); the whole fused path on meta tensors (no host read); every precondition with its one log line and the torch result; an op that raises; a process without any kernel |
| `tests/unit_tests/models/flash_next/test_hc_fused_whole_model.py` | the tiny model built with the flag at 1 is bit-identical to the flag at 0 on every reference prompt and a one-token prompt, all 9 hyper-connections run fused, none logs |

### Measured tolerances (simulator, 2026-11-19)

An element passes when (1) it is at most **one bf16 step** from the oracle, or (2) within **1e-6** absolute, or
(3) within 1e-6 of the magnitude of the operands of its last sum (cancellation: `residual + block * weight`, or the
gated sum, cancelling to a millionth of its operands, where a one-fp32-step difference in the sigmoid is many bf16
steps of the tiny result), or (4) for `xn` of `hc_combine_norm`, within the difference of its own `residual_out`
element times the norm factor plus one step (a combine element that rounded one step away moves its `xn` by the same
0.4 to 0.8 %, which is up to two steps of `xn`). The norm stage alone is held to rules 1 and 2: `xn` must be the
oracle's norm of the kernel's own `residual_out`. At least 99 % of every output must be bit-exact; the `*_edge` cases
and the injection copy must be exact. On top of all that, every element of every output must be inside the bf16
tolerance of `gdn-core-decode.md` ("Measured tolerances": atol 5e-2, rtol 1e-2), whatever rule it passed by; `compare`
checks that separately. One bf16 step is at most 0.78 % of the value.

All 56 cases pass, 0 failing elements. Over all of them: 92,238,179 output elements, 2,169 one step away, 109 further but within 1e-6, 37 passing by rule 3 or 4; the lowest bit-exact share of any output is 99.935 %. An entry above 1 under "Max bf16 steps" always belongs to an element counted under rule 2, 3 or 4 (a tiny or a cancelling value). The largest "max abs err", 0.125, is one bf16 step of a value between 16 and 32.

| Case | TPC instructions | Output | Elements | Bit-exact % | Max bf16 steps | 1 step (rule 1) | Within 1e-6 (rule 2) | Rules 3, 4 | Max abs err | Max rel err (values above 1e-3) |
|---|---|---|---|---|---|---|---|---|---|---|
| `norm_tiny_r1` | 862 | xn | 512 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_tiny_r3_shared` | 2,490 | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_real_r1` | 4,814 | xn | 10,240 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_real_r2_shared` | 9,628 | xn | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_real_r16` | 75,680 | xn | 163,840 | 99.999 | 1 | 1 | 0 | 0 | 1.221e-04 | 7.634e-03 |
| `norm_real_r32` | 151,168 | xn | 327,680 | 99.998 | 1 | 6 | 0 | 0 | 1.562e-02 | 7.576e-03 |
| `norm_prefill_r2048` | 9,662,656 | xn | 20,971,520 | 99.999 | 1 | 204 | 0 | 0 | 1.562e-02 | 7.812e-03 |
| `norm_tiny_stress` | 2,490 | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_real_stress_shared` | 14,346 | xn | 30,720 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `norm_hc2_h384_r2` | 1,388 | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_tiny_r1` | 205 | lora | 16 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 4 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_tiny_r3` | 586 | lora | 48 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 12 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_real_r1` | 545 | lora | 320 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 4 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_real_r2` | 1,090 | lora | 640 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 8 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_real_r16` | 8,314 | lora | 5,120 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 64 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_real_r32` | 16,570 | lora | 10,240 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 128 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_prefill_r2048` | 1,056,826 | lora | 655,360 | 100.000 | 1 | 3 | 0 | 0 | 3.125e-02 | 5.291e-03 |
|  |  | injection_out | 8,192 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_tiny_stress` | 586 | lora | 48 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 12 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_real_stress` | 1,606 | lora | 960 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 12 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_split_lora130_hc3_r3` | 1,096 | lora | 390 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | injection_out | 9 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_tiny_r3` | 586 | lora | 48 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_real_r32` | 16,570 | lora | 10,240 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `silu_real_stress` | 1,606 | lora | 960 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_tiny_r1` | 708 | block_input | 128 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_tiny_r3` | 2,094 | block_input | 384 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_real_r1` | 13,458 | block_input | 2,560 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_real_r2` | 26,916 | block_input | 5,120 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_real_r16` | 214,488 | block_input | 40,960 | 99.988 | 1 | 5 | 0 | 0 | 6.104e-05 | 6.993e-03 |
| `gate_mix_real_r32` | 428,856 | block_input | 81,920 | 99.988 | 1 | 10 | 0 | 0 | 1.953e-03 | 6.623e-03 |
| `gate_mix_prefill_r2048` | 27,439,224 | block_input | 5,242,880 | 99.995 | 16 | 258 | 19 | 0 | 7.812e-03 | 7.752e-03 |
| `gate_mix_tiny_stress` | 2,094 | block_input | 384 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_real_stress` | 40,314 | block_input | 7,680 | 99.935 | 86 | 1 | 4 | 0 | 7.898e-39 | 0.000e+00 |
| `gate_mix_tiny_edge` | 2,094 | block_input | 384 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `gate_mix_hc3_h256_r2` | 2,180 | block_input | 512 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_tiny_r1` | 850 | residual_out | 512 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_tiny_r3` | 2,490 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_real_r1` | 2,598 | residual_out | 10,240 | 99.990 | 1 | 1 | 0 | 0 | 1.192e-07 | 0.000e+00 |
| `combine_real_r2` | 5,196 | residual_out | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_real_r16` | 40,728 | residual_out | 163,840 | 99.998 | 1 | 4 | 0 | 0 | 1.562e-02 | 6.623e-03 |
| `combine_real_r32` | 81,336 | residual_out | 327,680 | 99.998 | 1 | 6 | 0 | 0 | 1.221e-04 | 5.952e-03 |
| `combine_prefill_r2048` | 5,197,944 | residual_out | 20,971,520 | 99.998 | 32 | 467 | 15 | 1 | 1.250e-01 | 7.812e-03 |
| `combine_tiny_stress` | 2,490 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_real_stress` | 7,734 | residual_out | 30,720 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_tiny_edge` | 2,490 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_tiny_r1` | 1,634 | residual_out | 512 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 512 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_tiny_r3_shared` | 4,826 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_real_r1` | 8,322 | residual_out | 10,240 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 10,240 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_real_r2_shared` | 16,644 | residual_out | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_real_r16` | 132,088 | residual_out | 163,840 | 99.998 | 1 | 3 | 0 | 0 | 4.883e-04 | 7.634e-03 |
|  |  | xn | 163,840 | 99.998 | 1 | 4 | 0 | 0 | 7.812e-03 | 6.369e-03 |
| `combine_norm_real_r32` | 264,024 | residual_out | 327,680 | 99.998 | 4 | 6 | 2 | 0 | 3.125e-02 | 5.525e-03 |
|  |  | xn | 327,680 | 99.996 | 5 | 10 | 2 | 0 | 7.812e-03 | 7.353e-03 |
| `combine_norm_prefill_r2048` | 16,887,960 | residual_out | 20,971,520 | 99.998 | 64 | 495 | 27 | 0 | 1.250e-01 | 7.812e-03 |
|  |  | xn | 20,971,520 | 99.996 | 79 | 685 | 40 | 36 | 1.562e-02 | 1.242e-02 |
| `combine_norm_tiny_stress` | 4,826 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_real_stress_shared` | 24,890 | residual_out | 30,720 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 30,720 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_tiny_rounding` | 4,826 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_real_rounding_shared` | 16,644 | residual_out | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 20,480 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
| `combine_norm_hc2_h384_r2` | 2,428 | residual_out | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |
|  |  | xn | 1,536 | 100.000 | 0 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 |

Rounding pin: on `combine_norm_tiny_rounding` and `combine_norm_real_rounding_shared` a third of the `xn` elements
(512 of 1,536 and 7,156 of 20,480) differ between "norm of the rounded combine" and "norm of the fp32 combine". The
kernel matches the rounded oracle on all of them. A mutant kernel that normalizes the fp32 combine (tried on the
simulator, not kept) is 65 to 67 % exact on these cases and 74 % on an ordinary one, and fails.

## What is wired

Everything is behind `VLLM_HPU_HC_FUSED` (default 0) and follows the fused GDN kernel's integration
(`VLLM_HPU_GDN_CORE_DECODE`, `load_gdn_fused_extension`, gate G1b).

### 1. Op module (`flashinfer_gaudi/hc_fused.py`)

Defines the six schemas and the fake kernels, the wrappers `hc_norm` ... `hc_combine_norm`, and
`hc_fused_missing_ops(device)` / `hc_fused_available(device)`. It is the only definition (defining a schema twice in
one process raises) and nothing imports it unless the flag is 1. It registers no kernel that computes: the HPU kernels
come from `hc_fused_ops*.so`, and a process without that extension has no kernel for any device. The CPU tests import
`tests/unit_tests/hc_fused/hc_ops.py`, which adds the oracle as the CPU kernels.

### 2. Layer (`vllm_gaudi/models/qwen4_exp/hyperconnection.py`)

`HpuGatedResidual.__init__` reads the flag once (`hc_fused_requested`; exactly 1 means fused) and keeps the layer's
name (its prefix in the model). `mix`, `combine_and_mix` and `combine` start with `self._hc_fused(...)`; when it
returns the op module they run the fused path, otherwise the torch code below them, which is unchanged.

Layer check, once per layer on its first call (`_hc_fused_layer_reason`):

- `flashinfer_gaudi.hc_fused` imports, and every op has a kernel for the input's device type. No HPU kernel means no
  extension or no kernel database for its GUIDs: the loader registers the kernels only after the database.
- `1 <= hc_count <= 64`, `hidden_size % 128 == 0`, `hc_lowrank >= 1`, `rms_norm_eps` a positive normal fp32.
- `hc_norm.weight` is a contiguous 1-D bfloat16 tensor of `hidden_size` or `hc_count * hidden_size` elements.
- The weights of the two projections are bfloat16 (so the GEMM outputs the ops take are).

Input check, on every call (`_hc_fused_call_reason`; ranks, dtypes, sizes, layout, devices: no tensor data):

- hidden states: 2-D, bfloat16, `[tokens >= 1, hc_count * hidden_size]`, contiguous, on the norm weight's device.
- `combine_and_mix` and `combine`: the block output is bfloat16 `[tokens, hidden_size]`, the injection bfloat16
  `[tokens, hc_count]`, both contiguous and on the same device.

These are the checks of the ops' validation callback and of the glue; the layer makes them first because an input
the op refuses raises inside the op. Either check failing logs, once per layer,

```
HC fused (VLLM_HPU_HC_FUSED=1) is off for <layer>: <reason>; using the torch path
```

and the call runs torch. A fused op that raises when it is called has the same effect: no op writes an input, so the
torch path takes the same inputs, the layer logs its line (`a fused op raised ...`) and stays on the torch path. The
line is a warning for every reason, the missing extension included: 97 layers per worker write it when the flag is on
and the kernels are not there, which is what the smoke mode and the fallback file look for.

`injection` returned by the fused `mix` is a new contiguous `[T, HC]` tensor with the bits of the slice the torch
path returns as a view. A fused `combine` does not take that view (it is not contiguous): a layer on the torch path
in front of a fused one makes the fused one fall back too. All layers of a model share one geometry, so in practice
all run fused or none does.

### 3. Startup (`vllm_gaudi/__init__.py::register_ops`)

After the GDN block and before the kernel-database status line: `_load_hc_fused_if_requested(flag)`. With the flag at
1 it calls `load_hc_fused_extension()` (before warmup: the loader changes `GC_KERNEL_PATH`, which SynapseAI reads when
the graph compiler starts) and, on a Gaudi machine where that returns False, warns once:
`HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel: the hc_fused_ops extension did not load (<errors>); every
hyper-connection runs the torch path. ...`. With any other value it does nothing.

### 4. Loader (`flashinfer_gaudi/_native.py`)

`load_hc_fused_extension()` runs once per process and never raises; `hc_fused_extension_errors()` says why it
returned False, and `kernel_database_status()` has `hc_fused_loaded` (None when it was never called). The order is
what makes the fallback safe:

1. Both files exist: `libhc_fused_gaudi2.so` (`HC_FUSED_GAUDI_KERNEL_LIBRARY`, else the package's `lib/`) and
   `hc_fused_ops*.so` (`HC_FUSED_GAUDI_NATIVE_LIBRARY`, else the package's `lib/`).
2. The bridge can run a custom op: in a lazy process `PT_HPU_LAZY_ACC_PAR_MODE` must be exactly 0
   (`lazy_accumulation_reason()`; [native-extensions.md](native-extensions.md), "Custom ops and the lazy
   accumulation thread"). Otherwise nothing is registered or loaded and the error names the variable.
3. The kernel database is registered under the rules of the GDN library: through the combined kernel database when
   it can be the single `GC_KERNEL_PATH` file, otherwise first in the colon list, which `_prepend_kernel_database`
   refuses next to `PT_HPU_RECIPE_CACHE_CONFIG`. One condition more: the combined file must serve the library
   (`combined_db_serves_hc_fused`: the library sits next to it and the file's bytes contain the name). An adapter
   built before the library was a member does not, and would leave the six GUIDs unserved.
4. Only then the schemas are imported and the extension, which registers the HPU kernels, is loaded.

A failure at any step leaves the ops without an HPU kernel, and the layers run torch.

### 5. Build (`tools/build_flashinfer_gaudi.py`)

`build_hc_fused` is `build_gdn_fused` with the hyper-connection names (CMake folder `hc_fused_kernel_db`, target
`hc_fused_gaudi2`, extension folder `hc_fused_pytorch`), followed by `check_kernel_library`: the built library must
load, export the three required entry points and list its Gaudi2 kernels, which is what the combined database needs
from a member. It runs with `VLLM_GAUDI_BUILD_HC_FUSED=1` or `--hc-fused-only`, in its own `try` block after the GDN
block and before the combined database. A failure of any kind prints `HC fused build failed (the other libraries are
unaffected): ...`, removes `libhc_fused_gaudi2.so` and `hc_fused_ops*.so` and nothing else, and the install goes on
(`--hc-fused-only` exits 3). `container_entry.sh` sets the switch from the same `tpc-clang` check as the other two.

### 6. Combined kernel database (`csrc/kernel_db_adapter/adapter.cpp`)

The sibling list is `libgdn_fused_gaudi2.so`, `libhc_fused_gaudi2.so`, `libflashinfer_gaudi_kernels.so`, each
optional (no GUID is shared, so the position only fixes the order of the merged list). A sibling that is not there
is not a member and not a failure; one that is there and cannot be used makes the database serve nothing, as for
every member. The library exports `GetSupportedDataLayouts` and `GetSuggestedManipulation` itself, so the adapter
forwards both. Host test: `csrc/kernel_db_adapter/test/test_adapter.cpp`; with the real library:
`tests/standalone/hc_fused/test_hc_library.py`.

### 7. Rental gate (`tools/flash_next/rental/gates.py::hc_fused`)

The orchestrator calls it right after G1b's own steps and records the result as G1b's `details["hc_fused"]`:

```bash
PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py
```

from the plugin tree with the installed libraries by path, in its own process group, 600 s, log
`fused_gdn/hc_hpu_op.log`. The step sets `PT_HPU_LAZY_ACC_PAR_MODE=0` unless the operator's environment has the
variable (`lazy_acc_par_mode` records the value); without it at 0 the loader refuses the kernels and the test
skips. `status` is `passed` (exit 0, tests passed (95 today), none skipped, failed or in error), `failed`
(anything else the test did; a skip means the kernels did not load and counts as a failure, a timeout too),
`skipped` (the library is not installed, or `--quick-gates` without `--hc-fused`) or `not_run` (the test file is
not in the bundle). `failed_tests` names the failed tests and `empty_inputs` counts the lazy bridge's failing
graphs and how many of them have an op built before the producer of its input
(`tools/flash_next/rental/lazy_graph.py`). It never changes G1b's status, the GDN mode or the GDN flags.

The 95 tests are: every op has an HPU kernel; 81 cases against the CPU oracle with `hc_case.compare` (the 56
simulator cases plus rows 4, 8, 12, 20 and 8192 at the real geometry for the five decode-loop kernels), each also
checking that no input changed; 6 cases against the same torch functions run on the card (atol 5e-2, rtol 1e-2); a
repeat call that must give the same bits; 5 cases, one per decode-loop kernel, that call the op eight times right
after queuing 96 lazy ops (`hc_stress.py`; added after the card runs below and not yet run on a card); one chain
`hc_combine_norm` -> GEMM -> `hc_silu_split` -> GEMM -> `hc_gate_mix` -> `hc_combine_norm` against the torch chain
on the card. Card runs of the first 90 on 2026-11-20 (Gaudi2, bridge 1.24.1, build `4d6bc84`): 84 and 83 passed at
`PT_HPU_LAZY_ACC_PAR_MODE=1`, a different failing set each time and every failure the same bridge error; 90 passed
at 0. What a pass means: the extension builds and
loads against the real bridge, the graph compiler accepts the six GUIDs with their glue, and the hardware gives the
simulator's numbers. If only the two "against the torch path on the card" tests fail while the 81 oracle cases pass,
the card's torch path differs from the CPU oracle; record the printed `HCHPU` lines before deciding.

### 8. Operator switch and smoke mode

`run_rental.sh --hc-fused` (through `container_entry.sh` to `main.py --hc-fused`) asks for the kernels. A server gets
`VLLM_HPU_HC_FUSED=1` only when the operator asked and the card test passed; every other server gets an explicit
`VLLM_HPU_HC_FUSED=0`. A tiny gate (G2) that fails or times out with the kernels keeps the attempt under
`tiny/attempt-hc-fused/` and runs once more without them, for the rest of the run. A server with the kernels gets
the recipe cache only through the combined kernel database. The manifest's `hc_fused` (`requested`, `status`,
`reason`, `on`, `disabled_by`, `outcome`) and the summary line `HC fused: on` / `off (not requested)` /
`off (test_hpu_op.py failed)` record the outcome; `HC fused ... is off for` lines of a server land in the variant's
`gdn_fallback.txt`.

`tiny_smoke.sh --gdn hc` serves the tiny checkpoint with the `fused` flags and `VLLM_HPU_HC_FUSED=1` and fails when
the tokens differ or the log has an `HC fused ... is off for` / `has no HPU kernel` line. It is not a default mode.
The tiny checkpoint's geometry (hidden 128, `hc_count` 4, `hc_lowrank` 16, 12 padding columns) is the simulator's
tiny geometry. This served decode against the reference is the first check of the kernels in a model; the step time
with and without the flag on the real checkpoint is not measured by anything here.

## Graph safety of the fused path

- No host read: the ops take tensors and Python scalars from the config (`eps`, `hc_count`, `lora_rank`); nothing
  calls `.item()`, `.cpu()` or branches on tensor data. The layer check reads the dispatch table and the module's
  geometry once per layer; the input check reads ranks, dtypes, sizes, contiguity and devices per call. The whole
  fused path runs on meta tensors, which have no data (`test_no_host_read_on_the_fused_path`).
- No data-dependent shape: every output shape is a function of input shapes and Python ints (`GetShapeInference`
  computes them from the input shapes and the node parameters; the simulator host checks it against every case).
- No mutation: no schema has an alias annotation, no kernel writes an input (checked by `sim_host` on every case and
  by `test_hpu_op.py`), no output aliases an input.
- The token dimension is dynamic: kernels take `T` from the index space and the widths from `get_dim_size`; one build
  serves every decode bucket and prefill size. `GetKernelGuids` leaves `supportsDynamicShapes` at 0, as the GDN library
  does, so each bucket compiles its own recipe as today.

## Unproven without a card

Since this list was written the card test ran (2026-11-20, Gaudi2, bridge 1.24.1, build `4d6bc84`): 90 of 90
passed with `PT_HPU_LAZY_ACC_PAR_MODE=0`, which by "Rental gate" above answers items 1, 2, 3 and 6 for the cases of
that test. At the bridge's default of 1 the same build failed 6 and 7 of the 90 at random; that is the bridge
behaviour of [native-extensions.md](native-extensions.md), "Custom ops and the lazy accumulation thread", not a
fault of a kernel. The items are kept as they were written.

1. `hc_fused_ops` has never been built against the real Habana bridge. Its callbacks compile and run against the
   stub header under ASan/UBSan; the real `hpu_custom_op_pt2.h`, the link with `habana_pytorch_backend` and
   `UserCustomOpDescriptor::execute` are untested. Several outputs together with node parameters through
   `registerUserCustomOp` have a card-proven precedent (`csrc/deepseek_v4/pytorch/hpu_dsv41_mhc_mme_post_collapse_pt2.cpp`),
   but that op defines its schema in C++. A schema with a tuple return that is defined in Python (`torch.library`,
   as the GDN ops define theirs) and gets its `HPU` kernel from a C++ `TORCH_LIBRARY_IMPL` returning `std::tuple`
   is new here: the GDN ops return one tensor.
2. The graph compiler has not seen the glue. The simulator's index-space mapping test passes for every case, but the
   graph compiler is a different consumer of the access patterns, of the whole-row declaration in `hc_gate_mix`, and
   of rank-1 index spaces.
3. Hardware numerics: the tables behind `v_rsqrt_f32` and `v_reciprocal_f32` on a real TPC are assumed to match the
   simulator. The dim-0 culling of out-of-bounds stores is documented and simulated, not seen on hardware.
4. Speed: no step time, no node count from a real recipe, no measurement of the LUT-cache warning's cost in
   `hc_combine_norm`, and no check of how the graph compiler spreads 4 or 20 index points over the engines.
5. Shape inference at run time (`GetShapeInference` called by the runtime with or without a GUID name) and recipes for
   every bucket.
6. The card's torch path against the CPU oracle: the oracle is the CPU run of the torch functions; the model today
   runs them on the card, whose sigmoid and rsqrt may differ. `test_hpu_op.py` prints and bounds that difference.
7. The loader and the combined database with a real stock library: tested with fakes and with the adapter's fake
   stock member.
8. The build step in the Habana image: `tools/build_flashinfer_gaudi.py` has built `csrc/hc_fused` only through
   stubs on CPU; the CMake project builds in the simulator container, the extension has never been built.
9. The served model: no server has run the fused path. The first evidence will be `tiny_smoke.sh --gdn hc` (tokens
   against the CPU reference at TP8, and no fallback line), then the tiny gate of a run with `--hc-fused`.
10. The layer's input check against the real tensors of a card: on CPU every hyper-connection of the tiny model gets
    a contiguous 2-D bfloat16 input; an HPU tensor that reports otherwise would show as `is off for` lines.

## Not done

- No served model has run the fused path, and no step time was measured.
- 3-D inputs `[batch, seq, W]` and non-bf16 dtypes are left to the torch path: a layer that gets one logs its line.
  The MTP drafter's layers are not checked with the flag.
- A fused layer does not take the injection view of a torch-path layer in front of it (it falls back instead of
  copying the view).
- The card test in G1b has no rerun without the combined kernel database: a failure through that database turns
  the kernels off for the run.
- A hidden size that is not a multiple of 128 is rejected, not handled.
- `supportsDynamicShapes` is not set and `GetLibVersion` is not exported, both as in `libgdn_fused_gaudi2.so`.

---

# MoE glue kernels (`VLLM_HPU_MOE_GLUE`)

Version 1, 2026-11-19. Four more Gaudi2 TPC kernels in the same library (`libhc_fused_gaudi2.so`) and the same op
extension (`hc_fused_ops*.so`). They replace the element-wise torch code of one Flash-Next MoE block: the routing
(softmax, top-k, renormalization, casts), the shared expert's SiLU-multiply, the per-row FP8 quantization of the routed
experts' input, and the shared-gate multiply with the final add. The GEMMs stay on the matrix engine and the routed
experts stay the vendor op `torch.ops.hpu.mixture_of_experts`. The kernels pass on the Gaudi2 instruction simulator
against the torch oracle; nothing here has run on a card ("Unproven without a card" below).

Everything is behind the flag `VLLM_HPU_MOE_GLUE`, **default 0**. With 0 no state is attached to a MoE block, no weight
is built, nothing is loaded, registered or logged, and the block dispatches the ops it dispatched before (pinned by an
op-sequence digest, `test_moe_glue_layer.py::test_flag_off_op_sequence_is_unchanged`). With 1 every piece falls back
to the torch path by itself and none can stop a run.

They live in the hyper-connection library on purpose: no new library has to be loaded, registered in
`GC_KERNEL_PATH`, built, installed, or added to the combined kernel database.

## Why, and what one block costs

Torch ops dispatched by one MoE block forward on the plugin's fast path (`patched_fused_moe_forward`), counted with a
`TorchDispatchMode` on CPU on the tiny checkpoint; a vendor op counts as one. The count does not depend on the row
count (the same at 1 and 32 rows) or on the geometry.

| Stage | Flag 0: ops (non-view) | Flag 1: ops (non-view) |
|---|---|---|
| Router logits GEMM, shared-gate GEMM | 4 (2) | 2 (1): one merged GEMM |
| Routing: cast, softmax, top-k, sum, divide, cast | 10 (6) | 6 (2): `moe_route`, the int32 to int64 cast of the ids |
| Shared expert: gate_up GEMM, SiLU, multiply, down GEMM | 8 (4) | 5 (3): `moe_silu_mul` between the two GEMMs |
| FP8 experts only: abs, max, add, divide, reciprocal, multiply, `cast_to_fp8_v2`, cast | 9 (8) | 1 (1): `moe_row_quant` |
| Routed experts: `hpu.mixture_of_experts` | 1 (1) | 1 (1) |
| Shared gate sigmoid, multiply; final add | 5 (3) | 3 (1): `moe_combine` |
| **One block, BF16 experts** | **28 (16)** | **17 (8)** |
| **One block, FP8 experts** | **37 (24)** | **18 (9)** |
| One decode step, 48 layers, FP8 | 1,776 (1,152) | 864 (432) |

On the card the vendor cast is itself a multiplication node and a cast node (`docs/features/supported_features.md`,
the `mult_fwd` note), so the row quantization replaces more nodes than the dispatcher count shows. The one cast left
in the routing may cost nothing on the card: the lazy bridge stores int64 as int32 (see "Routing ids").

## Names

| What | Name |
|---|---|
| Ops | `torch.ops.flashinfer_gaudi.moe_route`, `moe_silu_mul`, `moe_combine`, `moe_row_quant` |
| Schemas and fake kernels (the only definition) | `flashinfer_gaudi/moe_glue.py`; imported with `flashinfer_gaudi/hc_fused.py` (one extension implements both families). It registers no CPU kernel |
| CPU stand-in for the tests (the oracle as the ops' CPU kernels) | `tests/unit_tests/hc_fused/moe_ops.py` |
| Torch oracle (copies of the torch path, line for line) | `tests/unit_tests/hc_fused/moe_reference.py`, tied to the plugin by `test_moe_reference_copy.py` |
| Fused path of the block | `vllm_gaudi/ops/hpu_fused_moe.py` (`_moe_glue_forward`, `_moe_glue_layer_reason`, `_moe_glue_call_reason`, `_moe_glue_prepare`), `vllm_gaudi/ops/hpu_fp8.py` (`_moe_glue_fp8_experts`) |
| TPC kernels | `csrc/hc_fused/kernels/moe_glue_{route,silu_mul,combine,row_quant}_gaudi2.c` + `moe_glue_common.h` |
| GUIDs | `moe_glue_route_bf16_gaudi2`, `moe_glue_silu_mul_bf16_gaudi2`, `moe_glue_combine_bf16_gaudi2`, `moe_glue_row_quant_bf16_gaudi2` (`csrc/hc_fused/hc_fused_params.h`); listed by `GetKernelGuids` after the six hyper-connection GUIDs |
| Host glue | `csrc/hc_fused/host/moe_glue_gaudi2.cpp`; the entry points of `host/entry_points.cpp` serve both families |
| Loader | `flashinfer_gaudi/_native.py::load_moe_glue_extension`, `moe_glue_extension_errors`, `moe_glue_extension_loaded`, `moe_glue_missing_kernels`, `kernel_library_guids` |
| Flag | `VLLM_HPU_MOE_GLUE`, default 0 (`vllm_gaudi/envs.py`) |
| Build | the hyper-connection step (`VLLM_GAUDI_BUILD_HC_FUSED=1`, `--hc-fused-only`); each MoE kernel is optional (below) |
| Operator switch in the rental | `run_rental.sh --moe-glue`; smoke mode `tiny_smoke.sh --gdn moe` |

## Where each kernel sits

```
flag 0 (torch path)                                flag 1
  logits = GEMM_router(x)                  (MME)     gates = GEMM([router rows; shared-gate row])(x)   (MME)
  p = softmax(fp32(logits)); w, ids = topk(p, K)      ids, w = moe_route(gates)                         [M1]
  w = bf16(w / sum(w))                                ids = int64(ids)
  gu = GEMM_gate_up(x)                     (MME)     gu = GEMM_gate_up(x)                              (MME)
  act = silu(gu[:, :D]) * gu[:, D:]                   act = moe_silu_mul(gu)                            [M2]
  shared = GEMM_down(act)                  (MME)     shared = GEMM_down(act)                           (MME)
  g = sigmoid(GEMM_shared_gate(x))         (MME)
  FP8: scale, x8 = dynamic_quant(x)                   FP8: x8, scale = moe_row_quant(x)                 [M3]
  routed = hpu.mixture_of_experts(...)     vendor    routed = hpu.mixture_of_experts(...)              vendor
  out = g * shared + routed                           out = moe_combine(shared, gates, routed)          [M4]
  all-reduce(out)                                     all-reduce(out)
```

The merged gate weight `[E + 1, H]` (the router rows, then the one shared-gate row) is built once at load, in
`process_weights_after_loading` of the block's expert method, as a non-persistent buffer of the block's `MoERunner`
(`_hpu_moe_glue_gate_weight`): it is not a parameter, not in the state dict, and checkpoint loading is untouched.
Both gates are replicated at TP, so every rank routes identically; the shared expert stays TP-sharded and the routed
experts stay expert-parallel, exactly as on the torch path. The all-reduce after the block is the same call.

## Schemas

```
moe_route(Tensor gates, int experts, int top_k) -> (Tensor, Tensor)
moe_silu_mul(Tensor gate_up) -> Tensor
moe_combine(Tensor shared, Tensor gates, Tensor routed, int gate_column) -> Tensor
moe_row_quant(Tensor x, float fp8_max) -> (Tensor, Tensor)
```

`T` = tokens, `E` = experts (512; 16 on the tiny model), `K` = `top_k` (10; 2), `H` = hidden size (2560; 128), `C` =
columns of `gates` (`E + 1`; `E` is accepted too), `D` = the shared expert's width on this rank (80 at TP8; 640 at
TP1; 8 and 64 on the tiny model). No op writes an input; every output is a new tensor.

| Arg | Torch shape, dtype (HPU) | TPC dims | Meaning |
|---|---|---|---|
| `gates` | `[T, C]`, bf16 | `{C, T}` | merged gate GEMM output: router logits in columns `[0, E)`, the shared-gate logit in column `gate_column` |
| `gate_up` | `[T, 2D]`, bf16 | `{2D, T}` | the shared expert's gate_up GEMM output: gate columns, then up columns |
| `shared`, `routed`, `x` | `[T, H]`, bf16 | `{H, T}` | the shared expert's down GEMM output; the vendor op's output; the block input |
| `moe_route` -> | `([T, K] int32, [T, K] bf16)` | | expert ids, largest logit first, and their renormalized softmax weights |
| `moe_silu_mul` -> | `[T, D]` bf16 | | `bf16(bf16(silu(gate)) * up)` |
| `moe_combine` -> | `[T, H]` bf16 | | `bf16(bf16(bf16(sigmoid(logit)) * shared) + routed)` |
| `moe_row_quant` -> | `([T, H] float8_e4m3fn, [T, 1] float32)` | | the FP8 values and the per-row scale the vendor op takes as `hidden_states` and `d_scale_hidden_states` |

HPU gates, checked by the op's validation callback and again by the glue: rank-2 contiguous bf16 inputs with
`T > 0`; `1 <= K <= 64` (one int32 vector holds a row's ids), `K <= E <= 32767` (int16 lane ids), `E <= C`; an even
number of `gate_up` columns; `H % 128 == 0` for `moe_combine` and `moe_row_quant`; `0 <= gate_column < C`;
`fp8_max == 240` (Gaudi2's fp8 range; the kernel takes no other). `D` need not be a multiple of 128.

## What `mixture_of_experts` takes (the contract the routing op must meet)

From every call site in the plugin (`vllm_gaudi/extension/ops.py` `VllmMixtureOfExpertsOp`, `...OpFP8`,
`...OpFP8PerChannel`, `...OpWNA16`; `vllm_gaudi/ops/deepseek_v41_prefill_moe.py`): keyword arguments only.
`hidden_states` `[T, H]` bf16, or FP8 with `d_scale_hidden_states` `[T, 1]` fp32; `expert_routing_table` `[T, K]`
**int64**, global expert ids, in whatever order `torch.topk` returned them (no call site sorts them);
`router_weights` `[T, K]` in the activations' dtype (bf16); `w12` and `w3`, one tensor per **local** expert
(`[2I, H]`, `[H, I]`), with `d_scale_w12` / `d_scale_w3` for FP8; `permuted_weights=True`, `activation`,
`experts_min`, `experts_max` (the inclusive global id range of the rank or slice), optionally `chunk_size` and
`total_experts`. An id outside `[experts_min, experts_max]` contributes nothing: the slicing loops and the expert
parallel ranks all pass the full table and their own range. `tests/unit_tests/hpu_cpu_stubs.py` has no stand-in for
the op; the tests here use `tests/unit_tests/models/flash_next/moe_block.py`, which implements exactly this reading.
The reading is checked against an independent oracle: with that stand-in, the plugin's fast path on the tiny model
gives one MoE block the engine's CPU output to 7.5e-9 in fp32, and the whole model reproduces the Hugging Face
reference (`test_moe_glue_whole_model.py`). For that the CPU harness must keep the experts' weights in the
checkpoint layout, as a Gaudi machine does: on the CPU platform the engine prepacks `w13` and `w2` for its own
kernel at load, and the plugin's op would be handed prepacked rows (`moe_block.py`, `_keep_checkpoint_weight_layout`).

**Routing ids.** `moe_route` returns int32 and the block's existing `topk_ids.to(torch.int64)` widens them. This is
the pattern that has run on cards (`csrc/deepseek_v4/pytorch/hpu_dsv41_router_top6_pt2.cpp` returns `at::kInt`;
`deepseek_v41_prefill_moe.py` widens before the vendor op: "the ordinary BF16 fused-MoE contract uses I64 routing
tables"). An int64 output of a custom op was not chosen because how the bridge hands such a tensor to a TPC kernel is
not known here: in lazy mode it stores int64 device tensors as int32
(`tests/unit_tests/models/flash_next/test_int32_device_values.py`), and `tools/communication/tp2_native_decode_graph.h`
notes that only some bridge configurations do. The glue rejects an int64 ids output.

## Semantics and rounding points

The definition is the torch code (`moe_reference.py`). On CPU, torch computes every bf16 op in fp32 and rounds once
to bf16 (checked for silu and sigmoid over all 65,280 finite bf16 values, and for multiply, add, divide and
reciprocal on samples), so the oracle is "fp32, rounded to bf16 where the torch path stores a bf16 tensor". The
kernels keep that structure: fp32 inside, `SW_RHNE` conversions at the same points, every product and sum its own
instruction (`#pragma clang fp contract(off)`).

- **`moe_route`.** Softmax is monotonic, so the top-K probabilities belong to the top-K logits: the kernel selects
  on the bf16 logits by repeated maximum. Round r takes the largest (logit, lowest id) pair that comes after the
  winner of round r-1 in that order, so it keeps no per-expert state and reads the row again each round. Weights:
  `w_i = exp(l_i - l_max) / sum over the K winners`, rounded once to bf16. The torch path divides by the sum over
  all experts first and renormalizes after the top-K; that sum cancels, and the results differ by a few fp32 steps
  before the rounding.
- **Ties.** With bf16 logits two experts tie at the top-K cut in 11 to 14 % of the rows at 512 experts and top 10
  (random logits), 0.9 % at 16 and 2. `torch.topk` leaves the choice open: on CPU it is the lowest id in about half
  of those rows. The kernel always takes the lowest id. Every such choice is a valid routing (equal logits get
  equal weights) but it is another expert, so the block output differs. On rows without a tie at the cut the
  selected set is always `torch.topk`'s. The order of the K slots does not matter to a weighted sum, and where no
  logits are equal it is `torch.topk`'s order anyway.
- **`moe_silu_mul`** rounds the SiLU to bf16 before the multiplication, as `F.silu(x[..., :d]) * x[..., d:]` does on
  bf16 tensors. The product of two bf16 values is exact in fp32.
- **`moe_combine`** rounds the gate weight, the gated shared output and the sum, each to bf16.
- **`moe_row_quant`**: `scale = bf16(bf16(max|x| + bf16(1e-8)) / 240)`, `inverse = bf16(1 / scale)`,
  `x_fp8 = fp8(bf16(x * inverse))`, the default branch of `dynamic_quant` (`vllm_gaudi/extension/ops.py`) around
  the vendor cast, which multiplies in the input dtype before it casts. The division is a multiplication by
  `fp32(1/240)`: rounded to bf16 it is the quotient for every one of the 32,639 possible row maxima (simulator case
  `row_quant_every_amax`). The cast is round-to-nearest-even with the result clipped to 240, the conversion and
  packing sequence of the card-proven `csrc/deepseek_v4/kernels/deepseek_v41_dynamic_quant_bf16_gaudi2.c`.
- `sigmoid` is `hc_sigmoid` of `hc_fused_common.h` (`1 / (1 + exp(-x))`). The TPC `exp` returns 0 below -87.3 and
  fp32 denormals are flushed, where CPU torch keeps them: a SiLU or gate weight below 1.2e-38 is 0 in the kernels.

## Kernel layout

Scalar arguments of every MoE kernel's `main`, in this order: `int experts, int top_k, int gate_column, float
inv_fp8_max` (`MoeGlueKernelArgs`, 16 bytes; the glue derives `inv_fp8_max = 1/fp8_max`). Node parameters (op ->
glue): `struct MoeGlueParams { int32_t experts; int32_t top_k; int32_t gate_column; float fp8_max; }`, 16 bytes
(unused fields are 0). The size tells the two families apart when shape inference is called without a GUID name (12
bytes for the hyper-connection kernels); `top_k > 0` tells `moe_route` from `moe_row_quant`.

One index-space point is one token for all four kernels (index space `{T}`): rows are independent, so the graph
compiler can spread them over the TPC engines. Every output is declared as the token's whole row; every input as
the token's whole row too (the last 128-lane load of a row reads what follows the columns the kernel needs), except
`gates` of `moe_combine`, which is the one element `gate_column`.

| Kernel | Tensors (in + out) | Vector local memory (`-reg-mem-count`) | TPC instructions, real geometry: 1 row / 32 rows / 2048 rows |
|---|---|---|---|
| `moe_route` (E 512, K 10) | 1 + 2 | 0 bytes | 2,595 / 81,690 (2,553 per row) / 5,222,490 |
| `moe_silu_mul` (D 80) | 1 + 1 | 0 bytes | 215 / 6,040 / 383,032 |
| `moe_combine` (H 2560) | 3 + 1 | 0 bytes | 844 / 26,168 / 1,671,224 |
| `moe_row_quant` (H 2560) | 1 + 2 | 0 bytes | 1,102 / 34,334 / 2,193,470 |

`moe_route` on the tiny geometry (E 16, K 2): 651 instructions per row. No kernel declares a vector array or
`__local__` data; `moe_route` uses all 40 vector registers and spills nothing. One FP8 block is about 4,800 TPC
instructions per row in four nodes.

## Build isolation

Each MoE kernel is optional at build time, the way the GDN verify kernel is (`csrc/gdn_fused/CMakeLists.txt`):
`csrc/hc_fused/CMakeLists.txt` tries the compile of each one at configure time with the build's own command
(`-Wall -Werror -O2 -march=gaudi2`, no `-vlm`, `-slm` or `-spill-vlm`). A kernel that does not compile, or all four
with `-DHC_FUSED_MOE_GLUE=OFF`, is left out: `GetKernelGuids` does not list its GUID, `InstantiateTpcKernel` answers
`GLUE_NODE_NOT_FOUND` for it, and the six hyper-connection kernels are served as before
(`HC_FUSED_MOE_GLUE_BUILT` in `CMakeCache.txt` records what was built; `tools/build_flashinfer_gaudi.py` prints it).
The loader asks the installed library which MoE GUIDs it serves (`kernel_library_guids`, the library's own
`GetKernelGuids`), and a block that needs a kernel the library lacks logs its line and runs torch.

## TPC rules this relies on, and where each was checked

The rules of the table above ("TPC rules this relies on") hold here too. In addition:

| Rule | Source | Use here |
|---|---|---|
| Reductions: `v_bf16_reduce_max`, `v_i16_reduce_min`, `v_f32_reduce_add` reduce a vector and broadcast the result | Intel Gaudi docs, TPC User Guide, "Built-in Vector Reduction Intrinsics" (docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Intrinsics_Guide/Built-in_Functions.html); the names in the built-in table of `tpc-clang` 2.7.33 (`strings /usr/bin/tpc-clang`); `csrc/deepseek_v4/kernels/deepseek_v41_router_top6_gaudi2.c` (fp32 forms, on cards) | the row maximum per round, the lowest id among the lanes that hold it, the row's `max abs`. The result is broadcast again explicitly (`v_broadcast_element_*`, same doc page, "Built-in Shuffle Functions") |
| Compare and select on bf16 with int16 payload: `v_bf16_cmp_*`, `v_i16_sel2_geq_bf16_vb`, `v_i16_sel_eq_bf16_b`, predicated moves `v_bf16_mov_vb`, `v_i32_mov_vb` | the same built-in table; the fp32/int32 forms in the card-proven top-6 router kernels | repeated maximum with masking. The pair select keeps (value, id) per lane; the chunks run from the last to the first with `>=`, so a lane keeps its lowest id |
| Lane ids `V_LANE_ID_16`, `V_LANE_ID_32` | built-in header source in `tpc-clang` (`#define V_LANE_ID_16 read_lane_id_2b_b()`); the router kernels | expert id = chunk base + lane; the output slot of round r |
| `v_convert_i16_to_i32_b` of a broadcast value | the built-in table | the winner's id, the same in every lane, becomes an int32 lane value: the lane order of the conversion does not matter |
| `v_exp_cephes_f32` is a polynomial with max error 1 ULP, exactly 1 at 0 and exactly 0 at and below -87.336; `v_reciprocal_f32` 2 ULP | docs, "Built-in Special Functions" (table of max ULPs); the function's source in the `tpc-clang` built-in header | the K weights. A selected `-inf` logit gets weight 0 |
| Partial stores `v_*_st_tnsr_partial(coords, tensor, value, last lane, first lane)` | the built-in table; `deepseek_v41_router_top6_gaudi2.c` (`v_i32_st_tnsr_partial(at, output_ids, packed_ids, 5, 0)` for 6 ids) and `deepseek_v41_dynamic_quant_bf16_gaudi2.c` on cards | the K ids and weights of a row, the one scale, the last chunk of `moe_silu_mul`. The simulator host checks guard bytes around every output |
| bf16 to fp8 (1-4-3): `v_convert_bf16_to_f8_b(x, 0, SW_CLIP_FP, 0)`, then `v_f8_pack_b` twice and `v_f8_mov_dual_group_pack_b(..., SW_PACK21, ...)` | `deepseek_v41_dynamic_quant_bf16_gaudi2.c`, which runs on cards; the built-in table; header `DATA_F8_143` | `moe_row_quant`. Round-to-nearest-even and the clip are pinned by `row_quant_real_edge` (ties at every exponent, bit-exact) |
| A custom op may have an fp8 and an fp32 output, or an int32 and an fp32 output | `tools/communication/tp2_dynamic_quant.cpp` (`Float8_e4m3fn` + `kFloat`), `hpu_dsv41_router_top6_pt2.cpp` (`kInt` + `kFloat`) | the output metadata of `moe_row_quant` and `moe_route` |
| 16 KiB of vector local memory with lookup-table functions; the simulator does not check it | TPC-C Language, "Local Memory Space"; `tpcsim.md` | `tpc-clang -reg-mem-count` prints `Total VLM used: 0 bytes` for the four kernels; `test_moe_simulation.py` asserts it stays under 13,312 |

## Tests

**Simulator** (tpcsim container), all commands from the repository root:

```bash
tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest tests/standalone/hc_fused'
```

`test_moe_simulation.py` runs the 52 cases of `moe_case.SIM_CASES` through the library's entry points and the
simulator (mapping test and partition shuffle on), compiles each kernel alone with `-reg-mem-count`, and checks the
CMake build log for the flags. `test_moe_library.py` runs `moe_lib_check.cpp` (every entry point and every
rejection for the four GUIDs, directly and through the real combined kernel database) and the build guard: every
kernel of the library compiled with the one flag list and no local-memory override, a routing kernel that does not
compile costs only its own GUID, and `-DHC_FUSED_MOE_GLUE=OFF` leaves the six hyper-connection kernels. It also
runs the Python loader's `kernel_library_guids` on the real library (the ten GUIDs, and the six of a build without
the MoE kernels): the fakes of the CPU loader test cannot show that the ctypes call is right. 136 tests in the folder,
65 of them in the two MoE files.

Cases (`tests/unit_tests/hc_fused/moe_case.py`): per kernel, the tiny geometry, the real geometry at 1, 2, 16 and 32
rows and at 2048 rows; for the routing: ties at the cut by construction, all-equal logits, one to three huge logits
(1e4, 3e4, 1e30), very negative logits with `-inf` entries (rows with only 3 finite experts), a 40-sigma spread
(weights down to exp(-100)), tiny and denormal logits, and odd geometries (200 experts top 7; 130 experts top 64);
for the element-wise kernels: sigmoid stress values, exact rounding ties (bit-exact required), every finite bf16
gate value once; for the quantization: zero rows, rows of 1e-20 and 1e30, one spike, exact fp8 ties at every
exponent, and every possible row maximum.

What `moe_case.compare` requires. Routing: ids in range and distinct; their probabilities are the K largest of the
row (a valid top-K); on rows without a tie at the cut the set is `torch.topk`'s; on every row the ids are the K
largest logits in descending order with the lowest id first among equals; weights within one bf16 step of the
torch weight of the same probability. `moe_silu_mul` and `moe_combine`: one bf16 step, or the bits that a rounded
SiLU or gate weight one step away gives. `moe_row_quant`: the scale and every fp8 value exact. At most 1 % of an
output may differ at all; differences below the fp32 normal range do not count.

**CPU** (venv), one file per process:

| File | What it holds |
|---|---|
| `tests/unit_tests/hc_fused/test_moe_reference_copy.py` | every line of the copied oracle is in the plugin's (or the engine's) source, in order, and the copies return the plugin's bits |
| `.../hc_fused/test_moe_case.py` | a correct kernel passes every case; `compare` has teeth (wrong, doubled, out-of-range and misordered ids, the tie rule, one step allowed and two not, the propagation rules, exactness) |
| `.../hc_fused/test_moe_ops.py` | schemas, fake kernels, the stand-in equals the oracle, the package registers no CPU kernel and is not imported by default |
| `.../hc_fused/test_moe_sources.py` | GUIDs, ABI, scalar order, limits and build switches agree across the C++, TPC-C, CMake and Python sources |
| `.../hc_fused/test_moe_op_callbacks_host.py` | the real `hpu_hc_fused.cpp` under ASan + UBSan: output metadata (int32 ids, fp8 values), GUID and the 16 parameter bytes of all 72 HPU cases, 14 rejections |
| `.../hc_fused/test_moe_glue_loader.py` | the loader and the startup call |
| `tests/unit_tests/ops/test_hpu_moe_glue_cpu.py` | the helpers without a model: one log line per block, the input check, the merged weight on a bare runner, the wrapper that reads the flag, and that the FP8 branch calls the vendor op with the keywords of the plugin's op wrapper |
| `.../hc_fused/test_moe_hpu_op.py`, `test_moe_hpu_op_dry_run.py` | the card test, and a CPU dry run of it (the file runs, no test skips) |
| `tests/unit_tests/models/flash_next/test_moe_glue_layer.py` | the flag; the flag-off op-sequence digest; flag 1 equals flag 0 bit for bit on all four blocks at 1, 2, 32 and 300 rows, BF16 and FP8 experts; op counts; the merged weight; what every op and the vendor op are given; every fallback with its one line |
| `.../flash_next/test_moe_glue_whole_model.py` | the whole tiny model on the plugin's path: it reproduces the Hugging Face reference (fp32 with the flag at 0 within 1e-5, bf16 with the flag at 1 within 1e-2, every block fused); prefill hidden states and 16 greedy tokens of every reference prompt unchanged by the flag; the same with `VLLM_HPU_HC_FUSED=1` as well; what the kernel's tie rule does to those tokens |
| `.../flash_next/test_moe_glue_tp.py` | eight gloo ranks, TP8 with expert parallelism: flag 1 equals flag 0 bit for bit on every rank, the collectives are the same |

### Measured (simulator, 2026-11-19)

All 52 cases pass. Routing: every selection valid, the lowest-id rule on every row; weights bit-exact in every case
but one element of 20,480 at 2048 rows (one bf16 step). `moe_silu_mul`: of the 65,280 finite bf16 gate values one
gives a SiLU one bf16 step from CPU torch and 511 differ only below 1.2e-38; random inputs are 99.88 to 99.92 %
bit-exact, the rest one step (two where a one-step SiLU is multiplied). `moe_combine`: bit-exact on every random
case up to 2048 rows; the gate weight is CPU torch's for all 65,280 logits but 3 in the denormal range.
`moe_row_quant`: scale and values bit-exact in every case, all 32,639 row maxima included; the sign of an fp8 zero
can differ from CPU torch's cast.

## What is wired

1. **Startup** (`vllm_gaudi/__init__.py::register_ops`): after the hyper-connection block and before the kernel
   database status line, `_load_moe_glue_if_requested(flag)`. With the flag at 1 it calls
   `load_moe_glue_extension()`, which is `load_hc_fused_extension()` (the same library and extension, registered
   and loaded once, whichever flag asks first) plus the question to the library which MoE kernels it serves. On a
   Gaudi machine where that fails it warns once: `MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel: ...`.
   `kernel_database_status()` gets `moe_glue_loaded` only when the loader was called; such a process also reports
   `hc_fused_loaded`, because it is the same library. With `VLLM_HPU_HC_FUSED=1` alone the four MoE schemas are
   defined too (the extension implements them) and nothing else happens.
2. **Block state** (`hpu_fused_moe.py`): a wrapper of `MoERunner.__init__` reads the flag once per block; exactly
   1 attaches a state object. `process_weights_after_loading` of `HPUUnquantizedFusedMoEMethod` and
   `HPUFp8MoEMethod` then builds the merged gate weight.
3. **Layer check**, once per block on its first call (`_moe_glue_layer_reason`): no data-parallel, context-parallel
   or sequence-parallel MoE; no routed input or output transform; `routed_scaling_factor` 1; no zero expert, no
   fused shared experts; plain softmax top-k routing (not grouped, not custom, not `gpt_oss`); the runner holds the
   router gate and a shared expert with a shared expert gate, both plain bias-free linears, and `SiluAndMul`;
   `VLLM_HPU_EXPLICIT_SIGMOID_SILU` and `VLLM_HPU_TRITON_MODE` off; the expert method is exactly one of the two
   classes above; for FP8: the per-channel op with a dynamic input scale, none of the flags that send
   `dynamic_quant` down another branch, and `FP8_MAX` 240 on a Gaudi device; `H % 128 == 0`, `K` and `E` in range;
   the merged weight is there and bf16; every op the block needs (three, four for FP8) has a kernel for the device,
   and on a Gaudi device the loader ran and the library serves the op's GUID (a BF16 block does not need the
   quantization kernel).
4. **Call check**, on every call (`_moe_glue_call_reason`; ranks, dtypes, sizes, layout and devices only): 2-D
   contiguous bf16 hidden states `[T >= 1, H]` on the weight's device, not padded; the GEMM outputs are checked the
   same way before an op takes them.
5. **Fallback.** Either check failing, or a fused op that raises, logs once per block
   `MoE glue (VLLM_HPU_MOE_GLUE=1) is off for <layer>: <reason>; using the torch path` and the call runs the torch
   code on the same inputs (no op writes an input). A failed layer check or a raising op turns the block's fused
   path off for good; a call the kernels do not take costs that call only.
6. **Rental gate** (`tools/flash_next/rental/gates.py::moe_glue`): `tests/unit_tests/hc_fused/test_moe_hpu_op.py`
   on the card in lazy mode, recorded as G1b's `details["moe_glue"]`. 84 tests: every op has a kernel and the
   library serves it; 72 cases against the CPU oracle; the routing op's outputs through the vendor op against the
   torch routing through the vendor op; the vendor op with the slots reversed; the card's own tie rule, printed;
   `moe_row_quant` against the plugin's `dynamic_quant` on the card, bit for bit; five cases against the torch path
   on the card; repeat calls; one whole block chain. Lines starting with `MOEHPU` are the measurements.
7. **Operator switch and smoke mode.** `run_rental.sh --moe-glue` (through `container_entry.sh` to
   `main.py --moe-glue`): a server gets `VLLM_HPU_MOE_GLUE=1` only when the operator asked and the card test
   passed, otherwise an explicit 0; the manifest's `moe_glue` and the summary line `MoE glue:` record the outcome;
   a tiny gate that fails with the kernels runs once more without them. `tiny_smoke.sh --gdn moe` is the `fused`
   mode plus the flag and fails on a token mismatch or on a `MoE glue ... is off for` / `has no HPU kernel` line.
   The tiny checkpoint has bf16 experts, so the smoke mode does not run `moe_row_quant`: the card test does.

## Graph safety of the fused path

No host read and no data-dependent shape: the ops take tensors and Python ints and floats from the configuration;
the checks read the block's structure, the dispatch table, and ranks, dtypes, sizes, contiguity and devices. The
recorded op sequence of a fused forward has no `_local_scalar_dense`, `nonzero`, `masked_select` or `where`
(`test_op_count_before_and_after`) and is the same at 1 and 32 rows. Every output shape is a function of input
shapes and Python ints (`GetShapeInference`; the simulator host checks it on every case). No schema has an alias
annotation and no kernel writes an input (checked by the simulator host and by the card test).

## Unproven without a card: what to look for

1. **The build.** `hc_fused_ops` with the four new registrations against the real bridge (the callbacks run under
   ASan against the stub header only). Look for: `test_moe_hpu_op.py` not skipped; `MoE glue kernels in
   libhc_fused_gaudi2.so: ...` naming all four in the install log.
2. **Output dtypes through the bridge.** An int32 and a bf16 output (`moe_route`), an fp8 and an fp32 output
   (`moe_row_quant`). Both pairs have card precedents with other ops. A graph compile error naming
   `moe_glue_route_bf16_gaudi2` or `moe_glue_row_quant_bf16_gaudi2`, or `GLUE_INCOMPATIBLE_OUTPUT_SIZE`, would
   point here.
3. **The vendor op takes the routing op's outputs.** `MOEHPU vendor op fed by moe_route ...: N of 32 rows have the
   same ids and weights; max abs diff ...`: the same-row outputs must be bit-identical.
4. **Slot order.** `MOEHPU vendor op with the routing slots reversed: bit-identical ...`. `False` would mean the
   vendor op's sum depends on the slot order, which matters only on rows with equal logits.
5. **The tie rule of the card's `torch.topk`.** `MOEHPU all-equal logits: torch.topk on the card selects [...]`. If
   it is not the lowest ids, flag 1 and flag 0 route differently on rows that tie at the cut (about 1 row in 8 at
   512 experts). On CPU the tiny model's 80 reference tokens survive the kernel's rule although 34 of 7,048 routed
   rows (of 70 that tie at the cut) get another expert and three of five prefills end in other hidden states
   (`test_what_the_kernels_tie_rule_does_to_the_tokens`); a token mismatch in the smoke mode `moe` with no fallback
   line and a passing card test would point at ties first.
6. **FP8 numerics against the vendor cast.** `MOEHPU row_quant_* against dynamic_quant on the card: 0 of ... scales
   and 0 of ... fp8 values differ`. The test requires 0. A non-zero count of scales means the card rounds the bf16
   divide or reciprocal differently from CPU torch; of values, that the vendor cast rounds or clips differently
   from `v_convert_bf16_to_f8_b` with `SW_CLIP_FP`.
7. **bf16 SiLU and sigmoid of the card's torch path** against the kernels: `MOEHPU ... max abs diff against the HPU
   torch path`, bounded at atol 5e-2, rtol 1e-2 like the hyper-connection ops.
8. **Speed.** No step time and no node count of a real recipe. Whether the remaining int32 to int64 cast is a node,
   how the graph compiler spreads one index point per token at 1 row (each kernel then runs on one engine), and
   what four small nodes cost against 16 are all unmeasured.
9. **The served model.** No server has run the fused path. First `tiny_smoke.sh --gdn moe` (tokens against the CPU
   reference at TP8 with expert parallelism, no fallback line), then a run with `--moe-glue` on the FP8 checkpoint.

## Not done

- The ids stay int32 from the kernel and are widened by one cast; an int64 output was not tried.
- `moe_route`, `moe_combine` and the merged gate GEMM are only used together; a block that fails one check runs
  the whole torch path.
- The gate_up GEMM of the shared expert is not merged into the gate GEMM (it could be: both are bf16 and read `x`).
- 3-D inputs, non-bf16 dtypes, `gpt_oss`, grouped top-k, sequence-parallel and data-parallel MoE, block-FP8 experts
  (`VLLM_HPU_FORCE_CHANNEL_FP8=0`) and static input scales are left to the torch path.
- The merged gate weight is built at load only; a block whose weights are post-processed by another expert method
  has none and logs its line.
- `moe_row_quant` is Gaudi2 only (fp8 range 240).
- No case has NaN router logits. `-inf` logits are covered (they are selected last, lowest id first, with weight 0).
