// SPDX-License-Identifier: Apache-2.0
// Body of hc_fused_silu and hc_fused_silu_split: silu with the 1 / hc_count prescale on the lora
// part of the down GEMM output (between the down and the up GEMM of HpuGatedResidual.mix /
// combine_and_mix). Replaces hc_silu_torch and, with HC_FUSED_SILU_SPLIT, the split of the merged
// down+inject GEMM output.
//
//   down           {columns, tokens} bf16   columns >= lora_rank (+ hc_count with the split)
//   lora_out       {lora_rank, tokens} bf16   xf * sigmoid(xf), xf = down[:lora_rank] / hc_count
//   injection_out  {hc_count, tokens} bf16    (split only) the same bits as
//                                             down[lora_rank : lora_rank + hc_count]
//
// One index-space point is one token (dim 0 = token). lora_rank is not a multiple of the 128-lane
// vector (320 on the real model, 16 on the tiny one): the last chunk loads whatever follows the
// lora columns (injection logits, padding, or the tensor's pad value past its end) and the store
// keeps only the lanes inside lora_out, because the TPC memory unit culls the part of a store that
// falls outside dim 0 of the written tensor. The injection copy relies on the same rule.
#pragma clang fp contract(off)
#include "hc_fused_common.h"

#ifdef HC_FUSED_SILU_SPLIT
void main(tensor down, tensor lora_out, tensor injection_out, float eps, float inv_group, float inv_hc, int hc_count)
#else
void main(tensor down, tensor lora_out, float eps, float inv_group, float inv_hc, int hc_count)
#endif
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int lora_rank = get_dim_size(lora_out, 0);

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        for (int feature = 0; feature < lora_rank; feature += HC_FUSED_LANES) {
            const int5 coords = {feature, token, 0, 0, 0};
            const float128 value = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, down));
            float128 act;
            act.v1 = v_f32_mul_b(value.v1, inv_hc);
            act.v2 = v_f32_mul_b(value.v2, inv_hc);
            act.v1 = v_f32_mul_b(act.v1, hc_sigmoid(act.v1));
            act.v2 = v_f32_mul_b(act.v2, hc_sigmoid(act.v2));
            v_bf16_st_tnsr(coords, lora_out, v_convert_f32_to_bf16_all_b(act, SW_RHNE));
        }
#ifdef HC_FUSED_SILU_SPLIT
        const int5 injection_coords = {lora_rank, token, 0, 0, 0};
        const int5 injection_out_coords = {0, token, 0, 0, 0};
        v_bf16_st_tnsr(injection_out_coords, injection_out, v_bf16_ld_tnsr_b(injection_coords, down));
#endif
    }
}
