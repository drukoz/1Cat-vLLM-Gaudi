// SPDX-License-Identifier: Apache-2.0
// moe_glue_combine: the gated shared-expert output plus the routed experts' output, the last step
// of the MoE block before its all-reduce. Replaces sigmoid(gate) * shared + routed.
//
//   shared  {hidden, tokens} bf16    the shared expert's down GEMM output
//   gates   {columns, tokens} bf16   the merged gate GEMM output; column gate_column is the
//                                    shared-gate logit, the other columns are not read
//   routed  {hidden, tokens} bf16    the routed experts' output
//   out     {hidden, tokens} bf16    bf16(bf16(bf16(sigmoid(logit)) * shared) + routed)
//
// Rounding points: the torch path stores the gate weight, the gated shared output and the sum as
// bf16 tensors. bf16 * bf16 is exact in fp32; the sum is rounded to fp32 and then to bf16, as the
// CPU oracle rounds it.
//
// One index-space point is one token (dim 0 = token).
#pragma clang fp contract(off)
#include "moe_glue_common.h"

void main(tensor shared, tensor gates, tensor routed, tensor out,
          int experts, int top_k, int gate_column, float inv_fp8_max)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(out, 0);

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        const int5 gate_coords = {gate_column, token, 0, 0, 0};
        const bf16 logit = s_bf16_ld_g(gen_addr(gate_coords, gates));
        const float64 weight = moe_round_bf16(hc_sigmoid((float64)(float)logit));
        for (int feature = 0; feature < hidden; feature += MOE_GLUE_LANES) {
            const int5 coords = {feature, token, 0, 0, 0};
            const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, shared));
            float128 gated;
            gated.v1 = v_f32_mul_b(value.v1, weight);
            gated.v2 = v_f32_mul_b(value.v2, weight);
            gated = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(gated, SW_RHNE));
            const float128 other = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, routed));
            float128 sum;
            sum.v1 = v_f32_add_b(gated.v1, other.v1);
            sum.v2 = v_f32_add_b(gated.v2, other.v2);
            v_bf16_st_tnsr(coords, out, v_convert_f32_to_bf16_all_b(sum, SW_RHNE));
        }
    }
}
