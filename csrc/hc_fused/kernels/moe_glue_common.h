// SPDX-License-Identifier: Apache-2.0
#pragma once
// Helpers shared by the fused MoE glue kernels (Gaudi2 TPC-C); see hc_fused_common.h for the
// tensor layout ({columns, tokens}, dim 0 the fastest), the fp32-inside numerics and the lane
// order of the bf16 <-> fp32 conversions, which these kernels share.
//
// Every kernel's main() ends with the same four scalars (MoeGlueKernelArgs in
// ../hc_fused_params.h): int experts, int top_k, int gate_column, float inv_fp8_max. A kernel that
// has no use for one ignores it.
//
// Vector local memory: none. No kernel keeps a vector array; a row that is read twice is read
// from its input tensor twice.
#include "hc_fused_common.h"

#define MOE_GLUE_LANES 128

// One fp32 value rounded to bf16 (round-half-to-even), as fp32: the torch path computes in fp32
// and stores every intermediate result in a bf16 tensor.
static inline float64 moe_round_bf16(float64 value)
{
    float128 pair;
    pair.v1 = value;
    pair.v2 = value;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair, SW_RHNE)).v1;
}
