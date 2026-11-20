// SPDX-License-Identifier: Apache-2.0
// moe_glue_row_quant: per-row dynamic FP8 quantization of the routed experts' input. Replaces the
// default branch of dynamic_quant (vllm_gaudi/extension/ops.py): abs, max, add, div, reciprocal,
// multiply, cast_to_fp8_v2 and the cast of the scale to fp32.
//
//   x          {hidden, tokens} bf16
//   x_fp8_out  {hidden, tokens} fp8 (1-4-3)   fp8(bf16(x * inverse))
//   scale_out  {1, tokens} fp32               scale = bf16(bf16(max|x| + 1e-8) / fp8_max)
//                                             inverse = bf16(1 / scale)
//
// Rounding points: the torch path does all of this on bf16 tensors, so the sum, the quotient, the
// reciprocal and the product are each rounded to bf16 (round-half-to-even). The quotient is taken
// as a product with 1 / fp8_max, computed on the host; rounded to bf16 it is the quotient for
// every bf16 row maximum (simulator case row_quant_every_amax). The cast rounds to nearest even
// and clips to the largest fp8 value, the conversion the card-proven
// csrc/deepseek_v4/kernels/deepseek_v41_dynamic_quant_bf16_gaudi2.c uses.
//
// One index-space point is one token (dim 0 = token). Pass 1 takes the row maximum, pass 2 reads
// the row again and writes the fp8 values.
#pragma clang fp contract(off)
#include "moe_glue_common.h"

// bf16(1e-8) as fp32: torch adds the Python scalar in the tensor's dtype.
#define MOE_GLUE_QUANT_EPSILON 0x1.58p-27f

void main(tensor x, tensor x_fp8_out, tensor scale_out, int experts, int top_k, int gate_column, float inv_fp8_max)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(x, 0);

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        bfloat128 peak = v_bf16_abs_b(v_bf16_ld_tnsr_b((int5){0, token, 0, 0, 0}, x));
        for (int feature = MOE_GLUE_LANES; feature < hidden; feature += MOE_GLUE_LANES) {
            const int5 coords = {feature, token, 0, 0, 0};
            peak = v_bf16_max_b(peak, v_bf16_abs_b(v_bf16_ld_tnsr_b(coords, x)));
        }
        const float64 amax = v_convert_bf16_to_f32_all_b(v_broadcast_element_bf16(v_bf16_reduce_max(peak), 0)).v1;
        const float64 shifted = moe_round_bf16(v_f32_add_b(amax, MOE_GLUE_QUANT_EPSILON));
        const float64 scale = moe_round_bf16(v_f32_mul_b(shifted, inv_fp8_max));
        const float64 inverse = moe_round_bf16(v_reciprocal_f32(scale));
        const int5 scale_coords = {0, token, 0, 0, 0};
        v_f32_st_tnsr_partial(scale_coords, scale_out, scale, 0, 0);

        for (int feature = 0; feature < hidden; feature += MOE_GLUE_LANES) {
            const int5 coords = {feature, token, 0, 0, 0};
            const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, x));
            float128 scaled;
            scaled.v1 = v_f32_mul_b(value.v1, inverse);
            scaled.v2 = v_f32_mul_b(value.v2, inverse);
            // The vendor cast multiplies in bf16: the product is rounded to bf16 before the fp8 cast.
            const bfloat128 rounded = v_convert_f32_to_bf16_all_b(scaled, SW_RHNE);
            const minifloat256 sparse = v_convert_bf16_to_f8_b(rounded, 0, SW_CLIP_FP, (minifloat256)0);
            minifloat256 packed = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            packed = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, packed);
            packed = v_f8_mov_dual_group_pack_b(packed, SW_PACK21, (minifloat256)0);
            v_f8_st_tnsr_partial(coords, x_fp8_out, packed, MOE_GLUE_LANES - 1, 0);
        }
    }
}
