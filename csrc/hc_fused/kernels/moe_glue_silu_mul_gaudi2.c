// SPDX-License-Identifier: Apache-2.0
// moe_glue_silu_mul: the activation of the shared expert, between its two GEMMs. Replaces
// SiluAndMul (F.silu(x[..., :d]) * x[..., d:]).
//
//   gate_up  {first + 2 * width, tokens} bf16   columns [first, first + width) are the gate columns,
//                                               the next width columns the up columns
//   act_out  {width, tokens} bf16               bf16(bf16(silu(gate)) * up)
//
// first is the scalar gate_column: 0 when the input is the gate_up GEMM output itself, and the first
// column of the gate_up block when that GEMM is merged into the block's gate GEMM (the router logits
// and the shared-gate logit then come first). The columns before first are not read.
//
// Rounding points: torch computes silu of a bf16 tensor in fp32 and stores it as bf16, then
// multiplies two bf16 tensors. The product of two bf16 values is exact in fp32, so the second
// rounding is the only one of the multiplication.
//
// One index-space point is one token (dim 0 = token). width need not be a multiple of the 128-lane
// vector (80 per rank on the real model at TP8, 8 on the tiny one): the last chunk loads what
// follows the columns it needs (up columns after the gate columns, the tensor's pad value past the
// end) and stores only the lanes inside act_out.
#pragma clang fp contract(off)
#include "moe_glue_common.h"

void main(tensor gate_up, tensor act_out, int experts, int top_k, int gate_column, float inv_fp8_max)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int width = get_dim_size(act_out, 0);

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        for (int feature = 0; feature < width; feature += MOE_GLUE_LANES) {
            const int5 coords = {feature, token, 0, 0, 0};
            const int5 gate_coords = {gate_column + feature, token, 0, 0, 0};
            const int5 up_coords = {gate_column + width + feature, token, 0, 0, 0};
            const float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gate_coords, gate_up));
            float128 silu;
            silu.v1 = v_f32_mul_b(gate.v1, hc_sigmoid(gate.v1));
            silu.v2 = v_f32_mul_b(gate.v2, hc_sigmoid(gate.v2));
            const float128 rounded = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(silu, SW_RHNE));
            const float128 up = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(up_coords, gate_up));
            float128 act;
            act.v1 = v_f32_mul_b(rounded.v1, up.v1);
            act.v2 = v_f32_mul_b(rounded.v2, up.v2);
            const int last = width - feature < MOE_GLUE_LANES ? width - feature - 1 : MOE_GLUE_LANES - 1;
            v_bf16_st_tnsr_partial(coords, act_out, v_convert_f32_to_bf16_all_b(act, SW_RHNE), last, 0);
        }
    }
}
