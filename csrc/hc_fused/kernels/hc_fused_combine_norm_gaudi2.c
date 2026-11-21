// SPDX-License-Identifier: Apache-2.0
// hc_fused_combine_norm: inject the pending block output into the streams, then the grouped Gemma
// RMSNorm of the result (the first step of HpuGatedResidual.combine_and_mix). Replaces
// hc_combine_norm_torch.
//
//   residual      {hc_count * hidden, tokens} bf16
//   block_output  {hidden, tokens} bf16
//   injection     {hc_count, tokens} bf16   logits; the weight is 2 * sigmoid(logit / hc_count)
//   norm_weight   {hc_count * hidden} or {hidden} bf16
//   residual_out  {hc_count * hidden, tokens} bf16   residual + block_output * weight
//   xn_out        {hc_count * hidden, tokens} bf16   grouped norm of residual_out
//
// Rounding point: the oracle rounds the combine to bf16 before it normalizes, so the mean square
// and the normalized values here come from the rounded bf16 chunk, never from the fp32 sum.
//
// One index-space point is one stream of one token (dim 0 = stream, dim 1 = token). Pass 1 writes
// residual_out and sums the squares of the rounded values; pass 2 computes the same chunk again
// from the inputs (hc_combine_chunk is deterministic) and writes xn_out. residual_out is never read
// back: the compiler may reorder a load and a store of one tensor.
#pragma clang fp contract(off)
#include "hc_fused_common.h"

void main(tensor residual, tensor block_output, tensor injection, tensor norm_weight, tensor residual_out,
          tensor xn_out, float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(block_output, 0);
    const int weight_stride = get_dim_size(norm_weight, 0) == hidden ? 0 : hidden;

    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int stream = index_start[0]; stream < index_end[0]; ++stream) {
            const int base = stream * hidden;
            const int weight_base = stream * weight_stride;
            const float64 weight = hc_injection_weight(injection, stream, token, inv_hc);

            float64 sum = 0.0f;
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const int5 coords = {base + feature, token, 0, 0, 0};
                const bfloat128 rounded =
                    hc_combine_chunk(residual, block_output, base + feature, feature, token, weight);
                v_bf16_st_tnsr(coords, residual_out, rounded);
                const float128 value = v_convert_bf16_to_f32_all_b(rounded);
                sum = v_f32_add_b(sum, v_f32_mul_b(value.v1, value.v1));
                sum = v_f32_add_b(sum, v_f32_mul_b(value.v2, value.v2));
            }
            const float64 rrms = hc_rrms(sum, inv_group, eps);

            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const int5 coords = {base + feature, token, 0, 0, 0};
                const int5 weight_coords = {weight_base + feature, 0, 0, 0, 0};
                const bfloat128 rounded =
                    hc_combine_chunk(residual, block_output, base + feature, feature, token, weight);
                const float128 value = v_convert_bf16_to_f32_all_b(rounded);
                hc_norm_store(xn_out, coords, value, rrms, norm_weight, weight_coords);
            }
        }
    }
}
