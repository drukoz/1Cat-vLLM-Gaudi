// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN core decode: gated RMSNorm of one value head's 128 recurrence outputs
// stored as fp32, plus the zero store for padded rows. Uses the linear
// float64_pair_t lane convention (.v1 = 0..63, .v2 = 64..127) of the main kernel.

// Gated RMSNorm of one head's outputs o (fp32, linear lanes): mean over the
// 128 values, rsqrt with eps, norm weight, then the gate on z. gate_activation
// 1 uses sigmoid(z), 0 uses z * sigmoid(z). All arithmetic stays fp32.
static inline void gdn_core_norm_store(
    tensor z, tensor norm_weight, tensor output, int request, int value_head,
    float64_pair_t o, float eps, int gate_activation)
{
    const uchar256 broadcast_lane_zero = 0x80;

    float64 inverse_rms = v_f32_mac_b(o.v1, o.v1, o.v2 * o.v2);
    inverse_rms = v_f32_reduce_add(inverse_rms);
    inverse_rms = v_f32_shuffle_b(inverse_rms, broadcast_lane_zero, 0, inverse_rms);
    inverse_rms = v_rsqrt_f32(inverse_rms * (1.0f / 128.0f) + eps);

    const int5 z_coords = {value_head * 128, request, 0, 0, 0};
    const bfloat128 z_bf16 = v_bf16_ld_tnsr_b(z_coords, z);
    const float64_pair_t z_f32 = bf16_to_f32_linear(z_bf16);

    const int5 weight_coords = {0, 0, 0, 0, 0};
    const bfloat128 weight_bf16 = v_bf16_ld_tnsr_b(weight_coords, norm_weight);
    const float64_pair_t w = bf16_to_f32_linear(weight_bf16);

    const float64 sigmoid_lo = v_sigmoid_f32(z_f32.v1);
    const float64 sigmoid_hi = v_sigmoid_f32(z_f32.v2);
    float64_pair_t gate;
    if (gate_activation == 1) {
        gate.v1 = sigmoid_lo;
        gate.v2 = sigmoid_hi;
    } else {
        gate.v1 = z_f32.v1 * sigmoid_lo;
        gate.v2 = z_f32.v2 * sigmoid_hi;
    }

    float64_pair_t y;
    y.v1 = o.v1 * inverse_rms * w.v1 * gate.v1;
    y.v2 = o.v2 * inverse_rms * w.v2 * gate.v2;

    int5 output_coords = {0, value_head, request, 0, 0};
    v_f32_st_tnsr(output_coords, output, y.v1);
    output_coords[0] = 64;
    v_f32_st_tnsr(output_coords, output, y.v2);
}

// Zeros for one value head of a padded row (both 64-wide halves).
static inline void gdn_core_zero_store(tensor output, int request, int value_head)
{
    const float64 zero = 0.0f;
    int5 output_coords = {0, value_head, request, 0, 0};
    v_f32_st_tnsr(output_coords, output, zero);
    output_coords[0] = 64;
    v_f32_st_tnsr(output_coords, output, zero);
}
