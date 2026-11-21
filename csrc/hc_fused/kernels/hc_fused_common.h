// SPDX-License-Identifier: Apache-2.0
#pragma once
// Helpers shared by the fused hyper-connection kernels (Gaudi2 TPC-C).
//
// Tensor layout: a torch [tokens, width] bf16 tensor is the TPC tensor {width, tokens} (dim 0 is
// the fastest). The multi-stream width is hc_count * hidden, stream-major (stream h owns columns
// [h * hidden, (h + 1) * hidden)), which is the checkpoint-native layout.
//
// Numerics: fp32 inside, one rounding to bf16 (round-half-to-even) where the torch oracle rounds
// (vllm_gaudi/models/qwen4_exp/hyperconnection.py). Products and sums are separate instructions,
// never a fused multiply-add, because the oracle rounds each one.
//
// Lane order: v_convert_bf16_to_f32_all_b splits 128 bf16 lanes into two 64-lane fp32 vectors in
// the Gaudi dual-group order, not the linear one, and v_convert_f32_to_bf16_all_b undoes exactly
// that. Every operand of an element-wise expression here went through the same conversion (or is a
// broadcast scalar), and the only reduction is an order-free sum, so the order never matters.
//
// Vector local memory: none. These kernels call lookup-table special functions (rsqrt, reciprocal),
// which leaves 16 KiB of vector local memory; they keep no vector arrays and read every chunk
// again from the input tensors in the second pass instead.

#define HC_FUSED_LANES 128

// Lane 0 of `value` in every lane (v_f32_reduce_add leaves the sum in lane 0).
static inline float64 hc_broadcast_lane_zero(float64 value)
{
    const uchar256 broadcast_lane_zero = 0x80;
    return v_f32_shuffle_b(value, broadcast_lane_zero, 0, value);
}

// sigmoid(x) = 1 / (1 + exp(-x)), the form torch uses. Not the built-in sigmoid: that one is
// 0.5 * tanh(0.5 * x) + 0.5, which loses every digit of a small result to cancellation, and
// silu and the gates multiply the result by values that can be large (measured on the simulator:
// the stress cases of tests/unit_tests/hc_fused/hc_case.py fail with it and pass with this).
// Saturation is exact: 0.5 at 0, 1 for large x, 0 once exp(-x) overflows.
static inline float64 hc_sigmoid(float64 x)
{
    const float64 negated = v_f32_mul_b(x, -1.0f);
    return v_reciprocal_f32(v_f32_add_b(v_exp_cephes_f32(negated), 1.0f));
}

// 1 / sqrt(mean of squares + eps) for one stream, broadcast to every lane. `sum_of_squares` holds
// per-lane partial sums.
static inline float64 hc_rrms(float64 sum_of_squares, float inv_group, float eps)
{
    const float64 total = hc_broadcast_lane_zero(v_f32_reduce_add(sum_of_squares));
    return v_rsqrt_f32(v_f32_add_b(v_f32_mul_b(total, inv_group), eps));
}

// Gemma RMSNorm of one 128-lane chunk and its store: y = x * rrms, then y + y * w, rounded to bf16.
static inline void hc_norm_store(
    tensor output, int5 output_coords, float128 value, float64 rrms, tensor norm_weight, int5 weight_coords)
{
    const float128 weight = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(weight_coords, norm_weight));
    float128 y;
    y.v1 = v_f32_mul_b(value.v1, rrms);
    y.v2 = v_f32_mul_b(value.v2, rrms);
    y.v1 = v_f32_add_b(y.v1, v_f32_mul_b(y.v1, weight.v1));
    y.v2 = v_f32_add_b(y.v2, v_f32_mul_b(y.v2, weight.v2));
    v_bf16_st_tnsr(output_coords, output, v_convert_f32_to_bf16_all_b(y, SW_RHNE));
}

// The combine weight of one stream of one token: 2 * sigmoid(logit / hc_count), every lane.
static inline float64 hc_injection_weight(tensor injection, int stream, int token, float inv_hc)
{
    const int5 coords = {stream, token, 0, 0, 0};
    const bf16 logit = s_bf16_ld_g(gen_addr(coords, injection));
    const float64 scaled = v_f32_mul_b((float64)(float)logit, inv_hc);
    return v_f32_mul_b(hc_sigmoid(scaled), 2.0f);
}

// residual + block_output * weight for one 128-lane chunk, rounded to bf16 as the oracle rounds
// the combine. Deterministic: two calls with the same arguments return the same bits, which
// hc_fused_combine_norm relies on when it computes the chunk a second time.
static inline bfloat128 hc_combine_chunk(
    tensor residual, tensor block_output, int residual_feature, int block_feature, int token, float64 weight)
{
    const int5 residual_coords = {residual_feature, token, 0, 0, 0};
    const int5 block_coords = {block_feature, token, 0, 0, 0};
    const float128 res = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(residual_coords, residual));
    const float128 block = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(block_coords, block_output));
    float128 combined;
    combined.v1 = v_f32_add_b(res.v1, v_f32_mul_b(block.v1, weight));
    combined.v2 = v_f32_add_b(res.v2, v_f32_mul_b(block.v2, weight));
    return v_convert_f32_to_bf16_all_b(combined, SW_RHNE);
}
