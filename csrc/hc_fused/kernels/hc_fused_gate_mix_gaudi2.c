// SPDX-License-Identifier: Apache-2.0
// hc_fused_gate_mix: sigmoid gate and gated mean over the streams, the block input (the last step
// of HpuGatedResidual.mix / combine_and_mix, after the up GEMM). Replaces hc_gate_mix_torch.
//
//   xn               {hc_count * hidden, tokens} bf16   the grouped norm
//   gate             {hc_count * hidden, tokens} bf16   the up GEMM output (logits)
//   block_input_out  {hidden, tokens} bf16              sum over streams of sigmoid(gate) * xn, / hc_count
//
// One index-space point is one 128-column chunk of one token (dim 0 = chunk, dim 1 = token). It
// reads that chunk of every stream and writes that chunk of the output. The products are summed
// in stream order, as torch's sum over the stream dimension does.
#pragma clang fp contract(off)
#include "hc_fused_common.h"

void main(tensor xn, tensor gate, tensor block_input_out, float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(block_input_out, 0);

    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int chunk = index_start[0]; chunk < index_end[0]; ++chunk) {
            const int feature = chunk * HC_FUSED_LANES;
            float128 acc;
            acc.v1 = 0.0f;
            acc.v2 = 0.0f;
            for (int stream = 0; stream < hc_count; ++stream) {
                const int5 coords = {stream * hidden + feature, token, 0, 0, 0};
                const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, xn));
                const float128 logit = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, gate));
                acc.v1 = v_f32_add_b(acc.v1, v_f32_mul_b(hc_sigmoid(logit.v1), value.v1));
                acc.v2 = v_f32_add_b(acc.v2, v_f32_mul_b(hc_sigmoid(logit.v2), value.v2));
            }
            acc.v1 = v_f32_mul_b(acc.v1, inv_hc);
            acc.v2 = v_f32_mul_b(acc.v2, inv_hc);
            const int5 output_coords = {feature, token, 0, 0, 0};
            v_bf16_st_tnsr(output_coords, block_input_out, v_convert_f32_to_bf16_all_b(acc, SW_RHNE));
        }
    }
}
