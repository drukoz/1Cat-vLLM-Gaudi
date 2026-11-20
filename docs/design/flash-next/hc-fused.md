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
| Schemas and fake kernels (the only definition) | `flashinfer_gaudi/hc_fused.py`; imported only with the flag at 1. It registers no CPU kernel |
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

`test_hc_simulation.py` builds `csrc/hc_fused` with its real CMake project, links `sim_host.cpp` against
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
2. The kernel database is registered under the rules of the GDN library: through the combined kernel database when
   it can be the single `GC_KERNEL_PATH` file, otherwise first in the colon list, which `_prepend_kernel_database`
   refuses next to `PT_HPU_RECIPE_CACHE_CONFIG`. One condition more: the combined file must serve the library
   (`combined_db_serves_hc_fused`: the library sits next to it and the file's bytes contain the name). An adapter
   built before the library was a member does not, and would leave the six GUIDs unserved.
3. Only then the schemas are imported and the extension, which registers the HPU kernels, is loaded.

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
PT_HPU_LAZY_MODE=1 python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py
```

from the plugin tree with the installed libraries by path, in its own process group, 600 s, log
`fused_gdn/hc_hpu_op.log`. `status` is `passed` (exit 0, tests passed (90 today), none skipped, failed or in error), `failed` (anything else the test did;
a skip means the extension or the kernel database is missing and counts as a failure, a timeout too), `skipped` (the
library is not installed, or `--quick-gates` without `--hc-fused`) or `not_run` (the test file is not in the bundle).
It never changes G1b's status, the GDN mode or the GDN flags.

The 90 tests are: every op has an HPU kernel; 81 cases against the CPU oracle with `hc_case.compare` (the 56
simulator cases plus rows 4, 8, 12, 20 and 8192 at the real geometry for the five decode-loop kernels), each also
checking that no input changed; 6 cases against the same torch functions run on the card (atol 5e-2, rtol 1e-2); a
repeat call that must give the same bits; one chain `hc_combine_norm` -> GEMM -> `hc_silu_split` -> GEMM ->
`hc_gate_mix` -> `hc_combine_norm` against the torch chain on the card. What a pass means: the extension builds and
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
