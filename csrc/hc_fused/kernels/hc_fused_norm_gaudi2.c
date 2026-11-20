// SPDX-License-Identifier: Apache-2.0
// hc_fused_norm: grouped Gemma RMSNorm of the multi-stream residual (the first step of
// HpuGatedResidual.mix). Replaces grouped_gemma_rmsnorm_torch.
//
//   x            {hc_count * hidden, tokens} bf16
//   norm_weight  {hc_count * hidden} bf16 (one affine per element, hc_per_branch_norm)
//                or {hidden} bf16 (one affine shared by every stream)
//   xn_out       {hc_count * hidden, tokens} bf16
//
// One index-space point is one stream of one token (dim 0 = stream, dim 1 = token): it owns that
// stream's `hidden` columns of x and xn_out, so no two points touch the same element. Pass 1 sums
// the squares, pass 2 reads the chunks again and writes x * rrms * (1 + w).
#pragma clang fp contract(off)
#include "hc_fused_common.h"

void main(tensor x, tensor norm_weight, tensor xn_out, float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(x, 0) / hc_count;
    const int weight_stride = get_dim_size(norm_weight, 0) == hidden ? 0 : hidden;

    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int stream = index_start[0]; stream < index_end[0]; ++stream) {
            const int base = stream * hidden;
            const int weight_base = stream * weight_stride;

            float64 sum = 0.0f;
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const int5 coords = {base + feature, token, 0, 0, 0};
                const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, x));
                sum = v_f32_add_b(sum, v_f32_mul_b(value.v1, value.v1));
                sum = v_f32_add_b(sum, v_f32_mul_b(value.v2, value.v2));
            }
            const float64 rrms = hc_rrms(sum, inv_group, eps);

            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const int5 coords = {base + feature, token, 0, 0, 0};
                const int5 weight_coords = {weight_base + feature, 0, 0, 0, 0};
                const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, x));
                hc_norm_store(xn_out, coords, value, rrms, norm_weight, weight_coords);
            }
        }
    }
}
