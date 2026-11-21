// SPDX-License-Identifier: Apache-2.0
// moe_glue_act_quant: the activation and the per-(row, expert) FP8 quantization of the dense routed experts
// (VLLM_HPU_MOE_DENSE=1 with VLLM_HPU_MOE_DENSE_FLAT_GATE_UP=1), between the 2-D gate / up GEMM and the batched
// down GEMM of vllm_gaudi/extension/ops.py moe_dense_fp8. Replaces
//     mid = (F.silu(gate.float()) * up.float()).to(bf16)            the activation, one rounding
//     mid_fp8, scale = dynamic_quant(mid)                          per row, the default branch
// for every expert of every token, in one pass over the GEMM output.
//
//   gate_up    {experts * 2 * width, tokens} bf16   expert e's gate columns are [e * 2 * width, e * 2 * width
//                                                   + width), its up columns the next width columns (the flat
//                                                   GEMM's output: the stacked w13 is [E, 2I, H])
//   mid_out    {experts * width, tokens} fp8 (1-4-3)  fp8(bf16(act * inverse)), expert e's columns
//                                                   [e * width, (e + 1) * width)
//   scale_out  {experts, tokens} fp32               scale = bf16(bf16(max|act| + 1e-8) * inv_fp8_max) of each
//                                                   (token, expert); inverse = bf16(1 / scale)
//
// act = bf16(silu(gate) * up) with silu and the product in fp32 and one rounding to bf16, as the torch path does
// (silu of the fp32 copy of the gate, times the fp32 copy of up, then .to(bf16)). The quantization is
// moe_glue_row_quant's, rounding point for rounding point (see that kernel: the scale, the reciprocal and the product
// are each rounded to bf16; inv_fp8_max is bf16(1 / fp8_max) from the host; the cast rounds to nearest even and
// clips). sigmoid is hc_sigmoid (1 / (1 + exp(-x))), not the vendor silu: the two can differ in the last fp32 bit,
// so act can differ by one bf16 step from the card's torch path in rare elements.
//
// One index-space point is one token (dim 0 = token); it loops over the experts. width must be a multiple of the
// 128-lane vector (640 per rank on the real model at TP8). Pass 1 takes the maximum of the rounded activation,
// pass 2 computes the activation again and writes the fp8 values.
#pragma clang fp contract(off)
#include "moe_glue_common.h"

// bf16(1e-8) as fp32: torch adds the Python scalar in the tensor's dtype.
#define MOE_GLUE_QUANT_EPSILON 0x1.58p-27f

// The activation of one 128-lane chunk, rounded to bf16 once.
static inline bfloat128 moe_act_chunk(tensor gate_up, int gate_column, int up_column, int token)
{
    const float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){gate_column, token, 0, 0, 0}, gate_up));
    const float128 up = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){up_column, token, 0, 0, 0}, gate_up));
    float128 act;
    act.v1 = v_f32_mul_b(v_f32_mul_b(gate.v1, hc_sigmoid(gate.v1)), up.v1);
    act.v2 = v_f32_mul_b(v_f32_mul_b(gate.v2, hc_sigmoid(gate.v2)), up.v2);
    return v_convert_f32_to_bf16_all_b(act, SW_RHNE);
}

void main(tensor gate_up, tensor mid_out, tensor scale_out, int experts, int top_k, int gate_column, float inv_fp8_max)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int width = get_dim_size(mid_out, 0) / experts;

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        for (int expert = 0; expert < experts; ++expert) {
            const int gate_base = expert * 2 * width;
            const int up_base = gate_base + width;
            const int out_base = expert * width;

            bfloat128 peak = v_bf16_abs_b(moe_act_chunk(gate_up, gate_base, up_base, token));
            for (int feature = MOE_GLUE_LANES; feature < width; feature += MOE_GLUE_LANES) {
                peak = v_bf16_max_b(peak, v_bf16_abs_b(moe_act_chunk(gate_up, gate_base + feature, up_base + feature,
                                                                     token)));
            }
            const float64 amax = v_convert_bf16_to_f32_all_b(v_broadcast_element_bf16(v_bf16_reduce_max(peak), 0)).v1;
            const float64 shifted = moe_round_bf16(v_f32_add_b(amax, MOE_GLUE_QUANT_EPSILON));
            const float64 scale = moe_round_bf16(v_f32_mul_b(shifted, inv_fp8_max));
            const float64 inverse = moe_round_bf16(v_reciprocal_f32(scale));
            v_f32_st_tnsr_partial((int5){expert, token, 0, 0, 0}, scale_out, scale, 0, 0);

            for (int feature = 0; feature < width; feature += MOE_GLUE_LANES) {
                const float128 value = v_convert_bf16_to_f32_all_b(
                    moe_act_chunk(gate_up, gate_base + feature, up_base + feature, token));
                float128 scaled;
                scaled.v1 = v_f32_mul_b(value.v1, inverse);
                scaled.v2 = v_f32_mul_b(value.v2, inverse);
                // The vendor cast multiplies in bf16: the product is rounded to bf16 before the fp8 cast.
                const bfloat128 rounded = v_convert_f32_to_bf16_all_b(scaled, SW_RHNE);
                const minifloat256 sparse = v_convert_bf16_to_f8_b(rounded, 0, SW_CLIP_FP, (minifloat256)0);
                minifloat256 packed = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
                packed = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, packed);
                packed = v_f8_mov_dual_group_pack_b(packed, SW_PACK21, (minifloat256)0);
                v_f8_st_tnsr_partial((int5){out_base + feature, token, 0, 0, 0}, mid_out, packed, MOE_GLUE_LANES - 1, 0);
            }
        }
    }
}
