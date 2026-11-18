# GDN core decode op: contract (track RUK-2812 of RUK-2709)

Version 1, 2026-11-17. One fused step per GDN layer per decode token (T = 1), between `in_proj` and `out_proj`:
causal conv1d update + SiLU, gates, Q/K L2 norm and scale, the delta-rule recurrence, the gated RMSNorm, and the
conv-state and SSM-state writes. Every task of the track pastes the parts of this file it needs. Checked on CPU
against the composed torch path (`spikes/gdn-core-decode/probe.py`): see "Measured tolerances".

## Names

| What | Name |
|---|---|
| Op | `torch.ops.flashinfer_gaudi.gdn_core_decode` (schema defined in Python, `flashinfer_gaudi/gdn_core_decode.py`) |
| Torch reference | `flashinfer_gaudi.gdn_core_decode.gdn_core_decode_reference` |
| Gate enum | `GdnGateActivation(IntEnum)`: `SILU = 0`, `SIGMOID = 1` (Flash-Next's checkpoint has `output_gate_type: "sigmoid"`; the Qwen3.5/3.8 models use `silu`) |
| Flag | `VLLM_HPU_GDN_CORE_DECODE`, `GdnCoreDecodeMode(IntEnum)`: `OFF = 0`, `FUSED = 1`, `RECURRENCE = 2`; default `FUSED` (1) from plan v0.4, 0 before |
| TPC kernel | `csrc/gdn_fused/kernels/gdn_core_decode_gaudi2.c` (+ two headers, below), GUID `gdn_fused_core_decode_f32state_gaudi2` |
| Kernel database | `libgdn_fused_gaudi2.so` (CMake target `gdn_fused_gaudi2`, `csrc/gdn_fused/CMakeLists.txt`) |
| PyTorch extension | `gdn_fused_ops*.so` (`csrc/gdn_fused/pytorch/`), registers only the `HPU` kernel of the op |
| Build switch | `VLLM_GAUDI_BUILD_GDN_FUSED=1` in `tools/build_flashinfer_gaudi.py` (a failed fused build removes both fused libraries and keeps the rest of the install; `--gdn-fused-only` exits 3) |

Not to be confused with the existing `flashinfer_gaudi/gdn_fused_decode.py` (`gdn_fused_decode_step`: conv + gates +
recurrence, TP1/TP2 only, no norm) or the existing flag `VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE`.

## Schema

```
gdn_core_decode(Tensor(a!) conv_state, Tensor(b!) ssm_state, Tensor mixed_qkv, Tensor z, Tensor a, Tensor b,
                Tensor A_log, Tensor dt_bias, Tensor conv_weight_t, Tensor norm_weight,
                Tensor load_indices, Tensor store_indices,
                float scale, float eps, int gate_activation) -> Tensor
```

`H` = key heads on this rank, `HV` = value heads on this rank (`HV % H == 0`, `rep = HV // H`; Flash-Next rep = 3),
`D` = head dim (key and value both; 128 on the real model, 16 in the tiny fixture), `C = 2*H*D + HV*D`,
`B` = decode rows (padded batch), `S` = state slots.

| Arg | Torch shape, dtype | TPC coords `{dim0, dim1, ...}` (innermost first) | Meaning |
|---|---|---|---|
| `conv_state` | `[S, 3, C]`, bf16 (any float on CPU) | `{channel, row, slot}` | history rows 0..2 = oldest..newest input token. Mutated |
| `ssm_state` | `[S, HV, D(v), D(k)]`, fp32 | `{k, v_row, value_head, slot}` | recurrent state. Mutated |
| `mixed_qkv` | `[B, C]`, bf16 | `{channel, request}` | `in_proj` output, order `[q (H*D) | k (H*D) | v (HV*D)]` |
| `z` | `[B, HV*D]`, bf16 | `{channel, request}` | output gate input |
| `a`, `b` | `[B, HV]`, bf16 | `{value_head, request}` | gate inputs |
| `A_log`, `dt_bias` | `[HV]`, fp32 | `{value_head}` | the layer passes fp32 copies |
| `conv_weight_t` | `[4, C]`, bf16 | `{channel, tap}` | tap-major: `conv1d.weight.view(C, 4).t().contiguous()` |
| `norm_weight` | `[D]`, bf16 | `{k}` | `self.norm.weight` |
| `load_indices`, `store_indices` | `[B]`, int32 | `{request}` | slot read / slot written per row |
| returns | `[B, HV, D]`, **fp32** | `{v_row, value_head, request}` | gated-RMSNorm output; the layer casts it to the model dtype and flattens to `[B, HV*D]` for `out_proj` |

`scale` = `D ** -0.5`. `eps` = `self.norm.eps` (1e-6 for Flash-Next). The L2-norm epsilon is the constant 1e-6.

Preconditions (the layer checks them before choosing the op; the HPU host code rejects anything else):
`conv_state.shape[1] == 3` (conv kernel width 4, no speculative reserve row), no conv bias, T = 1 per request,
non-speculative decode. HPU only: `D == 128`, all tensors contiguous.

## Semantics (the definition; the reference and the kernel both implement exactly this)

```python
for r in range(B):
    ls, ss = int(load_indices[r]), int(store_indices[r])
    if not (0 <= ls < S and 0 <= ss < S):
        out[r] = 0                       # padding row: output zeros, no state write at all
        continue
    hist = conv_state[ls, 0:3, :].float(); x = mixed_qkv[r].float(); w = conv_weight_t.float()
    conv = hist[0]*w[0] + hist[1]*w[1] + hist[2]*w[2] + x*w[3]            # fp32
    act = conv * sigmoid(conv)                                            # SiLU, fp32, no bf16 rounding
    for kh in range(H):
        q = act[kh*D:(kh+1)*D];  k = act[H*D + kh*D : H*D + (kh+1)*D]
        q = q * rsqrt(sum(q*q) + 1e-6) * scale;  k = k * rsqrt(sum(k*k) + 1e-6)
        for j in range(rep):
            vh = kh*rep + j
            v = act[2*H*D + vh*D : 2*H*D + (vh+1)*D]
            xg = float(a[r, vh]) + dt_bias[vh]
            sp = log1p(exp(xg)) if xg <= 20 else xg
            decay = exp(-exp(A_log[vh]) * sp);  beta = sigmoid(float(b[r, vh]))   # beta stays fp32
            St = ssm_state[ls, vh] * decay                       # [D(v), D(k)]
            St = St + ((v - St @ k) * beta)[:, None] * k[None, :]
            o = St @ q                                           # [D], fp32, never rounded
            y = o * rsqrt(mean(o*o) + eps) * float(norm_weight)
            zz = float(z[r, vh*D:(vh+1)*D])
            out[r, vh] = y * (sigmoid(zz) if gate_activation == SIGMOID else zz * sigmoid(zz))
            ssm_state[ss, vh] = St
    conv_state[ss, 0:3, :] = [conv_state[ls, 1], conv_state[ls, 2], mixed_qkv[r]]   # exact copies, read before write
```

Invariants the caller guarantees (the scheduler does): within one call no row's store slot is another row's load
slot, except that padded rows may all share one dummy slot. Load == store for the same row is the normal case and
must work, which is why every write of a row happens after that row's reads of the same data.

Padding: the compact pool pads with `-1` (out of range: zeros, no writes). The non-compact runner pads with
`_MAMBA_PAD_BLOCK_ID = scheduler_blocks`, a real dummy slot inside `[0, S)`; those rows compute and write garbage into
the dummy slot, exactly like the torch path. Tests check real rows and real slots only, plus "slots no row names are
untouched".

Measured on the CPU tiny model (CTO probe, 2026-11-17; one GDN layer, a 4-row decode step with rows `1 -> 1`, `2 -> 2`
and two padded rows): with `-1` padding the torch path (mode 0) writes the pool's **last** row, conv and SSM state
both, because `-1` is a valid negative index in torch; the op writes nothing for those rows and returns zeros. With
the dummy-slot padding both paths write the dummy slot the same way. The real rows match within 6e-8 (out) and 1e-9
(SSM), and conv state is bit-exact. So tests never compare the pool's last row, or the dummy slot, across modes.

## Measured tolerances (CPU probe, 2026-11-17, seed 0, B = 4 with one -1 padded row, load != store on two rows)

| Head counts (H/HV, D = 128) | fp32 inputs: out / ssm / conv vs composed path | bf16 inputs: out / ssm vs composed path (|out| max ~6) |
|---|---|---|
| 16/48 (TP1) | 9.5e-7 / 2.4e-7 / 0 | 0.0246 / 0.0118 |
| 8/24 (TP2) | 9.5e-7 / 2.4e-7 / 0 | 0.0220 / 0.0078 |
| 4/12 (TP4) | 7.2e-7 / 2.4e-7 / 0 | 0.0192 / 0.0085 |
| 2/6 (TP8) | 4.8e-7 / 1.2e-7 / 0 | 0.0132 / 0.0070 |

The composed path is `hpu_causal_conv1d_update` (with `store_cache_indices`) -> `hpu_fused_gdn_gating` ->
`hpu_fused_recurrent_gated_delta_rule` (with `ssm_store_indices`, `use_qk_l2norm_in_kernel=True`) ->
`RMSNormGated.forward_static(..., norm_before_gate=True, activation="sigmoid")`. In bf16 it rounds the conv output,
`beta` and `o` to bf16; the op does not, hence the bf16 gap. Test tolerances: fp32 `atol=1e-5, rtol=1e-5`; bf16 out
`atol=5e-2, rtol=1e-2`, ssm `atol=3e-2, rtol=1e-2`; conv state exact in both.

## TPC kernel layout (three files, so three engineers can write them at once)

Index space rank 2: `dim0 = key head` (size `H`), `dim1 = request` (size `B`). One point owns key head `kh` of
request `r`: its Q and K channels, the V channels and `z` of value heads `kh*rep .. kh*rep+rep-1`, all `D` rows of
each, and the conv-state channels of exactly those Q/K/V blocks. No two points touch the same channel or state row.

```c
// csrc/gdn_fused/kernels/gdn_core_decode_gaudi2.c   (main; conv + SiLU, Q/K norm, gates; includes both headers)
void main(tensor conv_state, tensor ssm_state, tensor mixed_qkv, tensor z, tensor a, tensor b,
          tensor A_log, tensor dt_bias, tensor conv_weight_t, tensor norm_weight,
          tensor load_indices, tensor store_indices, tensor output,
          float scale, float eps, int gate_activation);

// csrc/gdn_fused/kernels/gdn_core_decode_recurrence.h   (delta rule for one value head; returns o, linear lanes)
static inline float64_pair_t gdn_core_recurrence_head(
    tensor ssm_state, int load_slot, int store_slot, int value_head,
    float64_pair_t q, float64_pair_t k, float64_pair_t v, float64 decay, float64 beta, uint64 lanes);

// csrc/gdn_fused/kernels/gdn_core_decode_norm.h   (gated RMSNorm of one head's o; fp32 store; padding store)
static inline void gdn_core_norm_store(
    tensor z, tensor norm_weight, tensor output, int request, int value_head,
    float64_pair_t o, float eps, int gate_activation);
static inline void gdn_core_zero_store(tensor output, int request, int value_head);
```

Vector convention: a `float64_pair_t` holds one 128-wide head in **linear lane order**: `.v1` = elements 0..63,
`.v2` = 64..127. fp32 tensors load and store linearly with `v_f32_ld_tnsr_b` / `v_f32_st_tnsr` at coords 0 and 64.
bf16 128-wide loads are converted with `bf16_to_f32_linear` (copied from
`csrc/flashinfer_gaudi/kernels/gdn_packed_decode_f32_gaudi2.c:12-49`, defined once in the `.c` before the includes),
never with `v_convert_bf16_to_f32_all_b`, whose lane order differs. The output is fp32 so no f32 -> bf16 vector
conversion is needed. `lanes` = `read_lane_id_4b_b()` (as in `gdn_mtp_prepared_f32_gaudi2.c:13`).

Scalar params ABI (host -> kernel), 12 bytes: `struct GdnCoreDecodeParams { float scale; float eps; int32_t gate_activation; };`.

## Recurrence-only op (version 2, 2026-11-17)

Plan v0.3 adds a second op that runs only the recurrence, so the part vllm-gaudi #1834 measured at about 80% of the
per-sequence decode cost can be qualified on its own. From plan v0.4 it is a bisect tool, not a candidate default: if
the fused kernel is wrong on the card, this mode shows whether the recurrence or the conv/norm part is at fault. The
layer keeps the torch conv update before it and `self.norm` after it.

| What | Name |
|---|---|
| Op | `torch.ops.flashinfer_gaudi.gdn_recurrent_decode` (same Python module and `torch.library` fragment as `gdn_core_decode`) |
| Torch reference | `flashinfer_gaudi.gdn_core_decode.gdn_recurrent_decode_reference` |
| Flag value | `GdnCoreDecodeMode.RECURRENCE = 2` (`OFF = 0` and `FUSED = 1` keep their numbers) |
| TPC kernel | `csrc/gdn_fused/kernels/gdn_recurrent_decode_gaudi2.c` (includes `gdn_core_decode_recurrence.h` and `gdn_core_decode_norm.h`), GUID `gdn_fused_recurrent_decode_f32state_gaudi2` |
| Host glue | `csrc/gdn_fused/host/gdn_recurrent_decode_gaudi2.{hpp,cpp}`, a second GUID in the same `libgdn_fused_gaudi2.so` |
| Scalar params ABI | `struct GdnRecurrentDecodeParams { float scale; };`, 4 bytes, in `csrc/gdn_fused/gdn_core_decode_params.h` |

```
gdn_recurrent_decode(Tensor(a!) ssm_state, Tensor qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias,
                     Tensor load_indices, Tensor store_indices, float scale) -> Tensor
```

| Arg | Torch shape, dtype | TPC coords (innermost first) | Meaning |
|---|---|---|---|
| `ssm_state` | `[S, HV, D(v), D(k)]`, fp32 | `{k, v_row, value_head, slot}` | as in `gdn_core_decode`. Mutated |
| `qkv` | `[B, C]`, bf16 | `{channel, request}` | the conv update's output **after** SiLU (`mixed_qkv_conv` in `qwen3_5.py`), order `[q | k | v]` |
| `a`, `b`, `A_log`, `dt_bias`, `load_indices`, `store_indices` | as in `gdn_core_decode` | as there | as there |
| returns | `[B, HV, D]`, fp32 | `{v_row, value_head, request}` | `o`, the recurrence output **before** the gated RMSNorm |

Semantics: exactly the `gdn_core_decode` loop above with `act = qkv[r].float()` (no conv, no SiLU, no conv-state
read or write), and `out[r, vh] = o` (no norm, no gate). Padding rows (load or store slot outside `[0, S)`) get zeros
and no state write. Same caller invariants. Index space, vector convention and `bf16_to_f32_linear` are the same as
the full kernel's: one point per (key head, request).

Preconditions: as for `gdn_core_decode`, minus the conv-state ones. HPU only: `D == 128`, all tensors contiguous.

## Debug index check (version 2)

`VLLM_HPU_GDN_CORE_DECODE_DEBUG` is `GdnCoreDecodeDebug(IntEnum)`: `OFF = 0`, `CHECK = 1`, default 0. With `CHECK`,
the layer calls `check_gdn_core_decode_indices(load_indices, store_indices, num_slots, num_rows)` (in
`flashinfer_gaudi/gdn_core_decode.py`) before either op. `num_rows` is the number of real request rows:
`int(query_start_loc[-1])` (the decode metadata's `query_start_loc_p`), or `num_tokens` when that is `None`. It is
**not** the layer's `num_decodes`: the runner pads `query_start_loc` with zero-length rows up to the padded batch
(`hpu_model_runner.py:4715`, `num_tokens_per_req + [0] * (padded_batch_size - num_decodes)`), so `num_decodes`
counts the padding rows too. Every real decode row is one token, and the padding rows come last. The check copies
the indices to the CPU, once per layer per step. It raises `ValueError` when either
of these holds among the real rows (index `< num_rows`) whose two slots are both in `[0, S)`:
1. two rows have the same store slot;
2. one row's store slot is another row's load slot (a reordered cache).

Rows from `num_rows` on are padding and are ignored: the non-compact runner points them all at one real dummy slot
(`_MAMBA_PAD_BLOCK_ID`), which would otherwise fail check 1. With `OFF` there is no check and no host sync.

## Layer wiring (version 4, 2026-11-18)

`HPUGatedDeltaNetAttention` (`vllm_gaudi/models/qwen3_5.py`) reads `VLLM_HPU_GDN_CORE_DECODE` once, in `__init__`,
into `self.gdn_core_decode_mode`, and `VLLM_HPU_GDN_CORE_DECODE_DEBUG` into `self.gdn_core_decode_debug`.

A decode step uses an op only when all of these hold (the shared per-step gate `_gdn_core_decode_ready`):
1. the mode is `FUSED` or `RECURRENCE`;
2. the step is not speculative (`num_accepted_tokens is None`) and the layer does not defer its SSM write-back;
3. the load and store indices are 1-D, with `num_tokens` rows each;
4. the once-per-layer precondition check passed. It runs on the first step that reaches it. On a failure it logs
   `GDN core decode (VLLM_HPU_GDN_CORE_DECODE=<n>) is off for <layer>: <reason>; using the torch decode path` once,
   and the layer stays on the torch path. The line is a warning, except for the reason
   `the gdn_fused_ops extension is not loaded`, which is logged at debug level because the plugin already warned once
   at startup (below).

From plan v0.4 the default mode is `FUSED`. `register_ops()` (`vllm_gaudi/__init__.py`) loads the extension whenever
the flag is nonzero, before warmup. On a Gaudi machine where it does not load, it logs one warning per process:
`GDN core decode (VLLM_HPU_GDN_CORE_DECODE=<n>) has no HPU kernel: the gdn_fused_ops extension did not load
(<errors>); every GDN layer runs the torch decode path. Build it with tools/build_flashinfer_gaudi.py --gdn-fused-only,
or set VLLM_HPU_GDN_CORE_DECODE=0.` A machine without an HPU logs nothing: the ops' CPU kernels need no extension, so
CPU tests run `FUSED` through the Python reference kernel by default.

| Precondition | `FUSED` | `RECURRENCE` |
|---|---|---|
| output gate is `silu` or `sigmoid` | checked | not checked (`self.norm` stays) |
| no conv bias, conv width 4, conv pool `[S, 3, C]` | checked | not checked (the torch conv stays) |
| `head_k_dim == head_v_dim`, fp32 SSM pool, not Triton strict mode | checked | checked |
| on HPU: `head_v_dim == 128` and the `gdn_fused_ops` extension loaded | checked | checked |
| a kernel for the device | `gdn_core_decode_available` | `gdn_recurrent_decode_available` |

Both ops get the full pools (`self.kv_cache[0]`, `self.kv_cache[1]`) and absolute slot indices as int32, in every
runner mode. The runner turns on direct GDN state only when every row's slot already is its positional slot and the
padding is `-1` (`_can_use_direct_gdn_state`), so absolute indexing gives the same rows.

- **`FUSED`**: `_maybe_gdn_core_decode` calls `gdn_core_decode` before Part 2. Part 2c takes its output, and Part 3
  skips `self.norm`.
- **`RECURRENCE`**: Part 2b runs as in mode 0 up to and including the torch conv update (`hpu_causal_conv1d_update`),
  but skips the fork's fused conv-and-recurrence step (`maybe_run_gdn_fused_decode_step`) and the torch gates
  (`hpu_fused_gdn_gating`), because the op computes the gates itself. Then
  `gdn_recurrent_decode(self.kv_cache[1], mixed_qkv_conv, a, b, A_log, dt_bias, load, store, scale)` replaces
  `maybe_run_gdn_decode_packed` and `hpu_fused_recurrent_gated_delta_rule`, and Part 3 runs `self.norm` as in mode 0.
  A step whose SSM pool is the bound active view (`state_is_active_view`) still runs the op on the full pool, as
  `FUSED` does.
- **Debug check**: with `CHECK`, `check_gdn_core_decode_indices` runs before any state is mutated. For `FUSED` it runs
  inside `_maybe_gdn_core_decode`, before the op. For `RECURRENCE` it runs in `forward` right after the gate, before
  Part 2.

## Phase 2 contract, version 2: speculative verification steps (T = 1 + k), 2026-11-18

Track phase 2 (RUK-2943). With MTP on, every decode step is a verification step over `T = 1 + k` tokens per
request (Flash-Next's recipe: `k = 3`, `T = 4`), or the plain step that follows one, and phase 1's op never runs.
`gdn_core_verify` runs the fused GDN step over those `T` nodes in one kernel and covers both step kinds.

**Why version 2.** Version 1 (RUK-2915) gave every node its own conv checkpoint in an `[S, 3, C]` pool and a
`parent_idx` per node. M3's runner does not keep conv state that way, and the op must follow M3's state protocol
exactly, because the next step (on either path) reads what this one wrote:
- **Conv state is one expanded slot per request**, `[S, R, C]` with `R = 3 + k` rows (the GDN spec's
  `conv_kernel - 1 + num_spec`). After a T-token pass the slot holds `width - 2 = 2` history rows and then the T
  input rows; the next pass reads its 3-row history window at row `num_accepted - 1`
  (`_hpu_causal_conv1d_spec_update` in `vllm_gaudi/ops/causal_conv1d_pytorch.py`). The read slot is the lattice's
  column 0; align mode writes the store lattice's column 0.
- **SSM state is a per-node checkpoint lattice**: node 0 starts from the accepted checkpoint
  `lattice[r, num_accepted - 1]`, and node t's state is written to `store_lattice[r, t]`
  (`gated_delta_rule_mtp_rollback` in `flashinfer_gaudi/gdn_decode.py`).
- **Nodes past a row's query length** (trimmed drafts) are masked: zero output, no state change, no SSM store, and
  their conv input rows are stored as zeros.

The probe `spikes/gdn-core-verify/probe.py` checks this contract against M3's composed path (`hpu_fused_gdn_gating`
-> `hpu_causal_conv1d_update(num_accepted_tokens=...)` -> `gated_delta_rule_mtp_packed` -> `RMSNormGated`) over 4
chained steps with the accepted counts threaded forward, align and non-align lattices, trimmed rows and a padding
row, at all four TP head counts, and the plain step after a verification. Results under "Measured tolerances".

Version 1's reference stays as `gdn_core_verify_tree_reference` (renamed, same file): the executable form of a
future draft tree. A tree needs a different conv protocol in the runner (the accepted path is not a prefix of the
node order) and per-node row states in the kernel; both are out of scope until the runner verifies trees.

### Names

| What | Name |
|---|---|
| Op | `torch.ops.flashinfer_gaudi.gdn_core_verify` (schema in `flashinfer_gaudi/gdn_core_verify.py`, same `torch.library` fragment as `gdn_core_decode`) |
| Torch reference | `flashinfer_gaudi.gdn_core_verify.gdn_core_verify_reference` (version 1's function is renamed `gdn_core_verify_tree_reference`) |
| Loop oracle and shared cases | `tests/unit_tests/gdn_core_decode/verify_sim_case.py` (`verify_oracle`, `VERIFY_CASES`, `VERIFY_HPU_CASES`, `compare_verify`, raw-file I/O) |
| Flag | `VLLM_HPU_GDN_CORE_VERIFY`, `GdnCoreVerifyMode(IntEnum)`: `OFF = 0`, `FUSED = 1`; **default `OFF` (0)** until rental gate G1b has checked the op on a card |
| TPC kernel | `csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c` + `gdn_core_verify_recurrence.h`; reuses `gdn_core_decode_norm.h`. GUID `gdn_fused_core_verify_f32state_gaudi2` (`kGdnCoreVerifyGuid`) |
| Scalar params ABI | `struct GdnCoreVerifyParams { float scale; float eps; int32_t gate_activation; };`, 12 bytes, in `csrc/gdn_fused/gdn_core_decode_params.h` (with `kGdnCoreVerifyMaxNodes = 8`) |
| Host glue | `csrc/gdn_fused/host/gdn_core_verify_gaudi2.{hpp,cpp}`, a third GUID in `libgdn_fused_gaudi2.so` |
| HPU registration | `csrc/gdn_fused/pytorch/hpu_gdn_core_decode.cpp` (the `gdn_fused_ops` extension) |
| Card checks | `tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py`; `tools/flash_next/gdn_core_decode_replay.py --ops verify --case ver_tp8_b16` (and `ver_tp8_b64`) |

### Schema

```
gdn_core_verify(Tensor(a!) conv_state, Tensor(b!) ssm_state, Tensor mixed_qkv, Tensor z, Tensor a, Tensor b,
                Tensor A_log, Tensor dt_bias, Tensor conv_weight_t, Tensor norm_weight,
                Tensor row_indices, Tensor ssm_store_indices,
                float scale, float eps, int gate_activation) -> Tensor
```

Twelve inputs, as `gdn_core_decode`, with `ssm_state` at input 1 and `mixed_qkv` at input 2, so phase 1's shape
inference (`InferCoreDecodeShape`: output `{128, HV, mixed_qkv rows}`) serves both GUIDs. The per-row slots are
packed into one tensor to stay within the TPC's tensor count. Activations are **token-major**, as the layer holds
them: node `t` of row `r` is row `r * T + t` of `mixed_qkv`, `z`, `a`, `b` and the output.

| Arg | Torch shape, dtype | TPC coords (innermost first) | Meaning |
|---|---|---|---|
| `conv_state` | `[S, R, C]`, bf16 | `{channel, row, slot}` | expanded conv slots, `R >= T + 2` (M3: `R = 3 + k`). Mutated |
| `ssm_state` | `[S, HV, D(v), D(k)]`, fp32 | `{k, v_row, value_head, slot}` | as in `gdn_core_decode`. Mutated |
| `mixed_qkv` | `[B*T, C]`, bf16 | `{channel, token}` | `in_proj` output, `[q | k | v]` |
| `z` | `[B*T, HV*D]`, bf16 | `{channel, token}` | output gate input |
| `a`, `b` | `[B*T, HV]`, bf16 | `{value_head, token}` | gate inputs |
| `A_log`, `dt_bias`, `conv_weight_t`, `norm_weight` | as in `gdn_core_decode` | as there | as there |
| `row_indices` | `[B, 5]`, int32 | `{field, request}` | per row: `0` conv load slot, `1` conv offset (first history row read, `num_accepted - 1`), `2` conv store slot, `3` SSM load slot (node 0's start), `4` query length (real nodes) |
| `ssm_store_indices` | `[B, T]`, int32 | `{node, request}` | where node t's SSM state is stored; out of range = not stored |
| returns | `[B*T, HV, D]`, **fp32** | `{v_row, value_head, token}` | gated-RMSNorm output per node; the layer casts it to the model dtype |

Preconditions (the layer checks them; the HPU host code rejects anything else): `1 <= T <= 8`, `T + 2 <= R`, no conv
bias, conv width 4, fp32 SSM state; HPU only: `D == 128`, all tensors contiguous.

### Semantics (`verify_oracle` in `verify_sim_case.py` is the executable form; the reference and the kernel match it)

```python
for r in range(B):
    cl, off, cs, sl, n = row_indices[r]
    if not (0 <= cl < S and 0 <= cs < S and 0 <= sl < S and 0 <= off <= R - 3 and n >= 1):
        out[r*T:(r+1)*T] = 0              # padding row: zero output, no state write at all
        continue
    n = min(n, T)
    hist = conv_state[cl, off:off+3]      # [3, C], read before any write
    xs = mixed_qkv[r*T:(r+1)*T]; xs[n:] = 0
    seq = cat([hist, xs])                 # [3 + T, C]
    state = ssm_state[sl]                 # [HV, D, D] fp32, read before any write
    for t in range(n):                    # nodes t >= n: out = 0, state unchanged, nothing stored
        conv = seq[t]*w[0] + seq[t+1]*w[1] + seq[t+2]*w[2] + seq[t+3]*w[3]      # fp32
        # from here exactly gdn_core_decode's per-row arithmetic, with token r*T + t for z, a, b and the output,
        # and `state` carried from node to node:  act = silu(conv); q, k L2-normalized (q * scale); per value head:
        # decay, beta; St = state[vh] * decay; St += ((v - St @ k) * beta)[:, None] * k[None, :]; state[vh] = St;
        # o = St @ q; out[r*T + t, vh] = rmsnorm(o) * norm_weight * gate(z)
        if 0 <= ssm_store_indices[r, t] < S:
            ssm_state[ssm_store_indices[r, t]] = state
    conv_state[cs, 0:2] = hist[1:3]       # exact bf16 copies
    conv_state[cs, 2:2+T] = xs            # zeros for nodes t >= n
    conv_state[cs, 2+T:R] = 0
```

Caller invariants (the scheduler guarantees them): within one call no row's store slot (conv or SSM) is another
row's load slot. Load == store for the same row is the normal case (non-align lattices: `cl == cs`, and `sl` is
also `ssm_store_indices[r, num_accepted - 1]`) and must work: every write of a row happens after that row's reads
of the same data.

### Layer mapping (from M3's metadata to the op; checked in the probe)

Verification step (`num_accepted_tokens` set, 2-D `load_indices [B, T]`; `align` = `lattice_store_differs` and a
store lattice is present; `R = conv_pool.shape[1]`; `qsl` = `query_start_loc_p`):

```python
acc = num_accepted_tokens[:B].long()
row_indices = stack([load[:, 0],                                   # conv load
                     acc.clamp(1, R - 2) - 1,                      # conv offset (the conv path's clamp)
                     store[:, 0] if align else load[:, 0],         # conv store
                     load.gather(1, (acc.clamp(1, T) - 1)[:, None])[:, 0],   # SSM load (the SSM path's clamp)
                     (qsl[1:B + 1] - qsl[:B]).clamp(0, T)], dim=1).int()
ssm_store_indices = (store if align else load).int()
```

Plain step after a verification (`num_accepted_tokens` set, 1-D `load_indices [B]`, `store = store_indices`, or
`load` when that is `None`): `T = 1`, `row_indices = [store, acc.clamp(1, R - 2) - 1, store, load, qsl diff
clamped to [0, 1]]`, `ssm_store_indices = store[:, None]`. The conv path reads and writes the request's base row
(`store`), the SSM path loads the accepted checkpoint (`load`) and stores to `store`.

### Measured tolerances (CPU probe, 2026-11-18, seed 0, B = 3, k = 3, D = 128)

| Head counts (H/HV) | fp32: out / ssm / conv vs M3's composed path, worst over 4 chained steps and 4 variants | bf16: out / ssm |
|---|---|---|
| 16/48 (TP1) | 4.3e-6 / 4.8e-7 / 0 | 0.042 / 0.0100 |
| 8/24 (TP2) | 2.0e-6 / 4.8e-7 / 0 | 0.036 / 0.0100 |
| 4/12 (TP4) | 1.4e-6 / 4.8e-7 / 0 | 0.050 / 0.0125 |
| 2/6 (TP8) | 1.3e-6 / 3.6e-7 / 0 | 0.030 / 0.0074 |

The plain step after a verification (T = 1): fp32 out 9.5e-7, ssm 3.6e-7, conv exact; bf16 out 0.019. The composed
path rounds the conv output, `beta` and `o` to bf16 in bf16 runs; the op does not, hence the bf16 gap. Negative
controls: a conv offset of 0 instead of `num_accepted - 1`, or an SSM load from column 0 instead of the accepted
checkpoint, moves the outputs by 3-5.6. Test tolerances: fp32 `atol=1e-5, rtol=1e-5`; bf16 out `atol=5e-2,
rtol=1e-2`, ssm `atol=3e-2, rtol=1e-2`; conv state exact in both. Kernel vs oracle: `sim_case.py`'s `OUT_ATOL`,
`OUT_RTOL`, `SSM_ATOL`, `SSM_RTOL` (TPC exp/log/sigmoid/rsqrt approximations).

### TPC kernel layout (two files, so two engineers can write them at once)

Index space rank 2, as phase 1: `dim0 = key head` (size `H`), `dim1 = request` (size `B`). One point owns key head
`kh` of request `r` over all `T` nodes: its Q and K channels, the V channels and `z` of value heads
`kh*rep .. kh*rep+rep-1`, the conv channels of exactly those blocks in the conv slot (all `R` rows), and those
value heads' SSM rows. No two points touch the same channel or state row.

`gdn_core_verify_gaudi2.c` defines `bf16_to_f32_linear` (copied from `gdn_core_decode_gaudi2.c`) and declares,
before including the two headers:

```c
#define GDN_VERIFY_MAX_NODES 8
__local__ float64 verify_q[2 * GDN_VERIFY_MAX_NODES];   // node t: [2t] = elements 0..63, [2t+1] = 64..127
__local__ float64 verify_k[2 * GDN_VERIFY_MAX_NODES];
__local__ float64 verify_v[2 * GDN_VERIFY_MAX_NODES];   // the current value head
__local__ float64 verify_decay[GDN_VERIFY_MAX_NODES];   // the current value head, all lanes equal
__local__ float64 verify_beta[GDN_VERIFY_MAX_NODES];
__local__ float64 verify_o[2 * GDN_VERIFY_MAX_NODES];   // written by the recurrence header
__local__ int verify_store_slot[GDN_VERIFY_MAX_NODES];  // ssm_store_indices[r, t] if t < n and in [0, S), else -1
#include "gdn_core_verify_recurrence.h"
#include "gdn_core_decode_norm.h"
```

(`__local__` vector and scalar arrays with a runtime index compile with `-Wall -Werror` in tpcsim, checked with the
contract commit; the fork's precedent is `csrc/deepseek_v41_unique/kernels/deepseek_v41_softmax_parts_gaudi2.c:4`.)

```c
// csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c   (main: row fields, conv + SiLU over the nodes, Q/K norm, gates,
//                                                    conv-slot store, recurrence and norm per value head)
void main(tensor conv_state, tensor ssm_state, tensor mixed_qkv, tensor z, tensor a, tensor b,
          tensor A_log, tensor dt_bias, tensor conv_weight_t, tensor norm_weight,
          tensor row_indices, tensor ssm_store_indices, tensor output,
          float scale, float eps, int gate_activation);

// csrc/gdn_fused/kernels/gdn_core_verify_recurrence.h   (one value head over nodes 0..nodes-1; reads the arrays above)
static inline void gdn_core_verify_recurrence_head(
    tensor ssm_state, int load_slot, int value_head, int nodes, uint64 lanes);
```

The output is written per node with phase 1's helpers and the token index as the request coordinate:
`gdn_core_norm_store(z, norm_weight, output, r * T + t, vh, o_t, eps, gate_activation)` for `t < n`, and
`gdn_core_zero_store(output, r * T + t, vh)` for `t >= n` and for every node of a padding row.

**Ordering.** tpc-clang treats `ld_tnsr`/`st_tnsr` on one tensor as independent (RUK-2829). Conv: the slot's store
rows `0..R-1` overlap the history rows `off..off+2` that the same point reads when `cl == cs`, so every conv store
must data-depend on all three history loads `h0`, `h1`, `h2` (phase 1's guard, extended):

```c
const bool128 n0 = v_bf16_cmp_grt_b(h0, (bf16)(1.0f / 0.0f));   // never true
const bool128 n1 = v_bf16_cmp_grt_b(h1, (bf16)(1.0f / 0.0f));
const bool128 n2 = v_bf16_cmp_grt_b(h2, (bf16)(1.0f / 0.0f));
// for every value s to store:  s = v_bf16_mov_vb(h0, 0, s, n0, 0); s = v_bf16_mov_vb(h1, 0, s, n1, 0);
//                              s = v_bf16_mov_vb(h2, 0, s, n2, 0);   then v_bf16_st_tnsr(...)
```

SSM: each value row is loaded once from `load_slot` and stored only after it is updated, so every store of row
`i` data-depends on the load of row `i`, and stores of row `i` never alias a later load (other rows). No guard is
needed there.
