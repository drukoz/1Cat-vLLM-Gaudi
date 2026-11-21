// SPDX-License-Identifier: Apache-2.0
// hc_fused_combine_norm_il: hc_fused_combine_norm (same inputs and outputs) with one index-space point per
// token that holds all of the token's streams, so the per-stream chains (combine, sum of squares, norm) of the
// hc_count == 4 streams run side by side in each basic block, the block output chunk is loaded and converted
// once for the four streams, and the per-point tail (four injection weights, four rrms) overlaps.
//
//   residual      {hc_count * hidden, tokens} bf16
//   block_output  {hidden, tokens} bf16
//   injection     {hc_count, tokens} bf16   logits; the weight is 2 * sigmoid(logit / hc_count)
//   norm_weight   {hc_count * hidden} or {hidden} bf16
//   residual_out  {hc_count * hidden, tokens} bf16   residual + block_output * weight, rounded to bf16
//   xn_out        {hc_count * hidden, tokens} bf16   grouped Gemma RMSNorm of residual_out
//
// Index space {1, tokens}: point (0, token) owns the token's row of every tensor. Rounding points as in
// hc_fused_combine_norm: the product block * weight and the sum are separate instructions, the combine is
// rounded to bf16, the norm is taken of the rounded values (pass 2 computes the same chunk again from the
// inputs: residual_out is never read back). The injection weights use the shipped kernels' sigmoid
// (hc_injection_weight), so residual_out is the shipped kernel's, bit for bit. Two differences in the norm, each
// in the last fp32 bit before the bf16 rounding of xn: the squares of the rounded bf16 values (exact in fp32)
// are accumulated with v_bf16_mac_acc32_b, two fp32 accumulators per stream added before the reduction (another
// summation order), and y + y * w is one fused multiply-add (hc_il_norm_store).
#pragma clang fp contract(off)
#include "hc_fused_il.h"

// residual + block * weight for one 128-lane chunk of one stream, rounded to bf16 (block already in fp32).
static inline bfloat128 hc_il_combine(tensor residual, int column, int token, float128 block, float64 weight)
{
    const int5 coords = {column, token, 0, 0, 0};
    const float128 res = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, residual));
    float128 combined;
    combined.v1 = v_f32_add_b(res.v1, v_f32_mul_b(block.v1, weight));
    combined.v2 = v_f32_add_b(res.v2, v_f32_mul_b(block.v2, weight));
    return v_convert_f32_to_bf16_all_b(combined, SW_RHNE);
}

static inline float128 hc_il_load_block(tensor block_output, int feature, int token)
{
    const int5 coords = {feature, token, 0, 0, 0};
    return v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, block_output));
}

static inline void hc_il_store(tensor out, int column, int token, bfloat128 value)
{
    const int5 coords = {column, token, 0, 0, 0};
    v_bf16_st_tnsr(coords, out, value);
}

// Gemma RMSNorm of one 128-lane chunk and its store: y = x * rrms, then y + y * w as one fused
// multiply-add (one rounding where the oracle rounds y * w and the sum: the result differs from the
// oracle's only where that sum sits within half an fp32 step of a bf16 rounding boundary), rounded to bf16.
static inline void hc_il_norm_store(tensor output, int column, int token, float128 value, float64 rrms,
                                    tensor norm_weight, int weight_column)
{
    const int5 weight_coords = {weight_column, 0, 0, 0, 0};
    const float128 weight = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(weight_coords, norm_weight));
    float128 y;
    y.v1 = v_f32_mul_b(value.v1, rrms);
    y.v2 = v_f32_mul_b(value.v2, rrms);
    y.v1 = v_f32_mac_b(y.v1, weight.v1, y.v1, 0);
    y.v2 = v_f32_mac_b(y.v2, weight.v2, y.v2, 0);
    const int5 coords = {column, token, 0, 0, 0};
    v_bf16_st_tnsr(coords, output, v_convert_f32_to_bf16_all_b(y, SW_RHNE));
}

static inline float64 hc_il_rrms(float128 sum, float inv_group, float eps)
{
    return hc_rrms(v_f32_add_b(sum.v1, sum.v2), inv_group, eps);
}

void main(tensor residual, tensor block_output, tensor injection, tensor norm_weight, tensor residual_out,
          tensor xn_out, float eps, float inv_group, float inv_hc, int hc_count)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int hidden = get_dim_size(block_output, 0);
    const int weight_stride = get_dim_size(norm_weight, 0) == hidden ? 0 : hidden;

    if (hc_count == 4) {
        for (int token = index_start[1]; token < index_end[1]; ++token) {
            const float64 w0 = hc_injection_weight(injection, 0, token, inv_hc);
            const float64 w1 = hc_injection_weight(injection, 1, token, inv_hc);
            const float64 w2 = hc_injection_weight(injection, 2, token, inv_hc);
            const float64 w3 = hc_injection_weight(injection, 3, token, inv_hc);
            float128 s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const float128 block = hc_il_load_block(block_output, feature, token);
                const bfloat128 r0 = hc_il_combine(residual, feature, token, block, w0);
                const bfloat128 r1 = hc_il_combine(residual, hidden + feature, token, block, w1);
                const bfloat128 r2 = hc_il_combine(residual, 2 * hidden + feature, token, block, w2);
                const bfloat128 r3 = hc_il_combine(residual, 3 * hidden + feature, token, block, w3);
                hc_il_store(residual_out, feature, token, r0);
                hc_il_store(residual_out, hidden + feature, token, r1);
                hc_il_store(residual_out, 2 * hidden + feature, token, r2);
                hc_il_store(residual_out, 3 * hidden + feature, token, r3);
                s0 = v_bf16_mac_acc32_b(r0, r0, s0, 0);
                s1 = v_bf16_mac_acc32_b(r1, r1, s1, 0);
                s2 = v_bf16_mac_acc32_b(r2, r2, s2, 0);
                s3 = v_bf16_mac_acc32_b(r3, r3, s3, 0);
            }
            const float64 q0 = hc_il_rrms(s0, inv_group, eps);
            const float64 q1 = hc_il_rrms(s1, inv_group, eps);
            const float64 q2 = hc_il_rrms(s2, inv_group, eps);
            const float64 q3 = hc_il_rrms(s3, inv_group, eps);
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const float128 block = hc_il_load_block(block_output, feature, token);
                const bfloat128 r0 = hc_il_combine(residual, feature, token, block, w0);
                const bfloat128 r1 = hc_il_combine(residual, hidden + feature, token, block, w1);
                const bfloat128 r2 = hc_il_combine(residual, 2 * hidden + feature, token, block, w2);
                const bfloat128 r3 = hc_il_combine(residual, 3 * hidden + feature, token, block, w3);
                hc_il_norm_store(xn_out, feature, token, v_convert_bf16_to_f32_all_b(r0), q0, norm_weight, feature);
                hc_il_norm_store(xn_out, hidden + feature, token, v_convert_bf16_to_f32_all_b(r1), q1, norm_weight,
                                 weight_stride + feature);
                hc_il_norm_store(xn_out, 2 * hidden + feature, token, v_convert_bf16_to_f32_all_b(r2), q2,
                                 norm_weight, 2 * weight_stride + feature);
                hc_il_norm_store(xn_out, 3 * hidden + feature, token, v_convert_bf16_to_f32_all_b(r3), q3,
                                 norm_weight, 3 * weight_stride + feature);
            }
        }
        return;
    }

    // Any other hc_count: the streams of a token one after the other.
    for (int token = index_start[1]; token < index_end[1]; ++token) {
        for (int stream = 0; stream < hc_count; ++stream) {
            const int base = stream * hidden;
            const int weight_base = stream * weight_stride;
            const float64 weight = hc_injection_weight(injection, stream, token, inv_hc);
            float128 sum = {0};
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const float128 block = hc_il_load_block(block_output, feature, token);
                const bfloat128 rounded = hc_il_combine(residual, base + feature, token, block, weight);
                hc_il_store(residual_out, base + feature, token, rounded);
                sum = v_bf16_mac_acc32_b(rounded, rounded, sum, 0);
            }
            const float64 rrms = hc_il_rrms(sum, inv_group, eps);
            for (int feature = 0; feature < hidden; feature += HC_FUSED_LANES) {
                const float128 block = hc_il_load_block(block_output, feature, token);
                const bfloat128 rounded = hc_il_combine(residual, base + feature, token, block, weight);
                hc_il_norm_store(xn_out, base + feature, token, v_convert_bf16_to_f32_all_b(rounded), rrms,
                                 norm_weight, weight_base + feature);
            }
        }
    }
}
