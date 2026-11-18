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
| Flag | `VLLM_HPU_GDN_CORE_DECODE`, `GdnCoreDecodeMode(IntEnum)`: `OFF = 0`, `FUSED = 1`, default 0 |
| TPC kernel | `csrc/gdn_fused/kernels/gdn_core_decode_gaudi2.c` (+ two headers, below), GUID `gdn_fused_core_decode_f32state_gaudi2` |
| Kernel database | `libgdn_fused_gaudi2.so` (CMake target `gdn_fused_gaudi2`, `csrc/gdn_fused/CMakeLists.txt`) |
| PyTorch extension | `gdn_fused_ops*.so` (`csrc/gdn_fused/pytorch/`), registers only the `HPU` kernel of the op |
| Build switch | `VLLM_GAUDI_BUILD_GDN_FUSED=1` in `tools/build_flashinfer_gaudi.py` |

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
per-sequence decode cost can be qualified on its own. If the conv or norm parts of the full kernel fail on hardware,
this mode can still be used. The layer keeps the torch conv update before it and `self.norm` after it.

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

## Layer wiring (version 3, 2026-11-17)

`HPUGatedDeltaNetAttention` (`vllm_gaudi/models/qwen3_5.py`) reads `VLLM_HPU_GDN_CORE_DECODE` once, in `__init__`,
into `self.gdn_core_decode_mode`, and `VLLM_HPU_GDN_CORE_DECODE_DEBUG` into `self.gdn_core_decode_debug`.

A decode step uses an op only when all of these hold (the shared per-step gate `_gdn_core_decode_ready`):
1. the mode is `FUSED` or `RECURRENCE`;
2. the step is not speculative (`num_accepted_tokens is None`) and the layer does not defer its SSM write-back;
3. the load and store indices are 1-D, with `num_tokens` rows each;
4. the once-per-layer precondition check passed. It runs on the first step that reaches it. On a failure it logs
   `GDN core decode (VLLM_HPU_GDN_CORE_DECODE=<n>) is off for <layer>: <reason>; using the torch decode path` once,
   and the layer stays on the torch path.

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
