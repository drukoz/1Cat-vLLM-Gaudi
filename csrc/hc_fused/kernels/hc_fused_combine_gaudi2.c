// SPDX-License-Identifier: Apache-2.0
// hc_fused_combine: inject a block output into the streams (HpuGatedResidual.combine: before a
// PLE or deepstack addition and at a pipeline-parallel hand-off). Replaces hc_combine_torch.
//
//   residual      {hc_count * hidden, tokens} bf16
//   block_output  {hidden, tokens} bf16
//   injection     {hc_count, tokens} bf16   logits; the weight is 2 * sigmoid(logit / hc_count)
//   residual_out  {hc_count * hidden, tokens} bf16   residual + block_output * weight
//
// One index-space point is one stream of one token (dim 0 = stream, dim 1 = token).
#pragma clang fp contract(off)
#include "hc_fused_common.h"

void main(tensor residual, tensor block_output, tensor injection, tensor residual_out,
          float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(block_output, 0);

    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int stream = index_start[0]; stream < index_end[0]; ++stream) {
            const int base = stream * hidden;
            const float64 weight = hc_injection_weight(injection, stream, token, inv_hc);
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const int5 coords = {base + feature, token, 0, 0, 0};
                v_bf16_st_tnsr(coords, residual_out,
                               hc_combine_chunk(residual, block_output, base + feature, feature, token, weight));
            }
        }
    }
}
