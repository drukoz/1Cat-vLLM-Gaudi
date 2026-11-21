// SPDX-License-Identifier: Apache-2.0
// hc_fused_gate_mix_il: hc_fused_gate_mix (same inputs, output, index space and access patterns) with the
// hc_count == 4 streams of a chunk in one basic block and the sigmoid of hc_fused_il.h.
//
//   xn               {hc_count * hidden, tokens} bf16   the grouped norm
//   gate             {hc_count * hidden, tokens} bf16   the up GEMM output (logits)
//   block_input_out  {hidden, tokens} bf16              sum over streams of sigmoid(gate) * xn, / hc_count
//
// One index-space point is one 128-column chunk of one token (dim 0 = chunk, dim 1 = token). Products and the
// sum over the streams are rounded as the oracle rounds them (separate instructions, stream order). The sum
// starts from the first product, not from +0.0 (an all -0.0 column keeps its sign: no other value changes).
#pragma clang fp contract(off)
#include "hc_fused_il.h"

static inline float128 hc_il_gated(tensor xn, tensor gate, int5 coords)
{
    const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, xn));
    const float128 logit = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, gate));
    float128 product;
    product.v1 = v_f32_mul_b(hc_il_sigmoid(logit.v1), value.v1);
    product.v2 = v_f32_mul_b(hc_il_sigmoid(logit.v2), value.v2);
    return product;
}

void main(tensor xn, tensor gate, tensor block_input_out, float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(block_input_out, 0);

    if (hc_count == 4) {
        for (int token = index_start[1]; token < index_end[1]; ++token) {
            for (int chunk = index_start[0]; chunk < index_end[0]; ++chunk) {
                const int feature = chunk * HC_FUSED_LANES;
                const int5 c0 = {feature, token, 0, 0, 0};
                const int5 c1 = {hidden + feature, token, 0, 0, 0};
                const int5 c2 = {2 * hidden + feature, token, 0, 0, 0};
                const int5 c3 = {3 * hidden + feature, token, 0, 0, 0};
                const float128 p0 = hc_il_gated(xn, gate, c0);
                const float128 p1 = hc_il_gated(xn, gate, c1);
                const float128 p2 = hc_il_gated(xn, gate, c2);
                const float128 p3 = hc_il_gated(xn, gate, c3);
                float128 acc;
                acc.v1 = v_f32_add_b(v_f32_add_b(v_f32_add_b(p0.v1, p1.v1), p2.v1), p3.v1);
                acc.v2 = v_f32_add_b(v_f32_add_b(v_f32_add_b(p0.v2, p1.v2), p2.v2), p3.v2);
                acc.v1 = v_f32_mul_b(acc.v1, inv_hc);
                acc.v2 = v_f32_mul_b(acc.v2, inv_hc);
                const int5 output_coords = {feature, token, 0, 0, 0};
                v_bf16_st_tnsr(output_coords, block_input_out, v_convert_f32_to_bf16_all_b(acc, SW_RHNE));
            }
        }
        return;
    }

    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int chunk = index_start[0]; chunk < index_end[0]; ++chunk) {
            const int feature = chunk * HC_FUSED_LANES;
            float128 acc;
            acc.v1 = 0.0f;
            acc.v2 = 0.0f;
            for (int stream = 0; stream < hc_count; ++stream) {
                const int5 coords = {stream * hidden + feature, token, 0, 0, 0};
                const float128 product = hc_il_gated(xn, gate, coords);
                acc.v1 = v_f32_add_b(acc.v1, product.v1);
                acc.v2 = v_f32_add_b(acc.v2, product.v2);
            }
            acc.v1 = v_f32_mul_b(acc.v1, inv_hc);
            acc.v2 = v_f32_mul_b(acc.v2, inv_hc);
            const int5 output_coords = {feature, token, 0, 0, 0};
            v_bf16_st_tnsr(output_coords, block_input_out, v_convert_f32_to_bf16_all_b(acc, SW_RHNE));
        }
    }
}
