// SPDX-License-Identifier: Apache-2.0
// Recurrence-only GDN decode for Gaudi2: one step per layer per decode token.
// Runs the Q/K L2 normalization and scale, the decay/beta gates and the delta
// rule (gdn_core_decode_recurrence.h) for each value head, and stores the
// recurrence output o as fp32. The input qkv is already convolved and
// activated, and the gated RMSNorm stays in the layer (self.norm after this).
//
// One index-space point is one key head of one request (dim0 = key head,
// dim1 = request). The point owns the Q block, the K block and the
// head_repeat V blocks of its value heads, so no two points ever touch the
// same channel, state row, or output slice. The recurrence helper reads each
// state row before it writes that row back, so load slot == store slot is the
// normal case. Padded rows write zeros through gdn_core_zero_store and touch
// no state.
float64_pair_t bf16_to_f32_linear(bfloat128 input)
{
    // The native all-lanes conversion preserves Gaudi dual-group ordering,
    // while the recurrent state rows are linear in K. Repack the two 64-wide
    // FP32 vectors into logical [0:64] and [64:128] order.
    bfloat128_pair_t unpacked;
    unpacked.v1 = v_bf16_unpack_b(
        input,
        ((e_group_0) << 8) | ((e_every_second_element) << 9) |
            ((e_lower_half_group) << 10),
        unpacked.v1);
    unpacked.v2 = v_bf16_unpack_b(
        input,
        ((e_group_1) << 8) | ((e_every_second_element) << 9) |
            ((e_lower_half_group) << 10),
        unpacked.v2);

    const bfloat128 first_groups = unpacked.v1;
    unpacked.v1 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 0, 1, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 1, 2, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 1, 3, MkWr(1, 1), unpacked.v1);

    unpacked.v2 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 2, 0, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 2, 1, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 3, 2, MkWr(1, 1), unpacked.v2);

    const float64_pair_t first = v_convert_bf16_to_f32_all_b(unpacked.v1);
    const float64_pair_t second = v_convert_bf16_to_f32_all_b(unpacked.v2);
    float64_pair_t result;
    result.v1 = first.v1;
    result.v2 = second.v1;
    return result;
}

#include "gdn_core_decode_recurrence.h"
#include "gdn_core_decode_norm.h"

// One 128-wide head of the activated qkv input at `channel`, as fp32 in linear lane order.
static inline float64_pair_t load_head(tensor qkv, int channel, int request)
{
    const int5 coords = {channel, request, 0, 0, 0};
    return bf16_to_f32_linear(v_bf16_ld_tnsr_b(coords, qkv));
}

void main(tensor ssm_state, tensor qkv, tensor a, tensor b, tensor A_log, tensor dt_bias,
          tensor load_indices, tensor store_indices, tensor output, float scale)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const uint64 lanes = read_lane_id_4b_b();
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int head_dim = 128;
    const int value_heads = get_dim_size(ssm_state, 2);
    const int state_slots = get_dim_size(ssm_state, 3);
    const int channels = get_dim_size(qkv, 0);
    const int key_heads = (channels - value_heads * head_dim) / (2 * head_dim);
    const int head_repeat = value_heads / key_heads;
    const int key_offset = key_heads * head_dim;
    const int value_offset = 2 * key_offset;

    for (int request = index_start[1]; request < index_end[1]; ++request) {
        int5 slot_coords = {request, 0, 0, 0, 0};
        const int load_slot = s_i32_ld_g(gen_addr(slot_coords, load_indices));
        const int store_slot = s_i32_ld_g(gen_addr(slot_coords, store_indices));
        const bool valid = load_slot >= 0 && load_slot < state_slots &&
                           store_slot >= 0 && store_slot < state_slots;

        for (int key_head = index_start[0]; key_head < index_end[0]; ++key_head) {
            if (!valid) {
                for (int j = 0; j < head_repeat; ++j) {
                    gdn_core_zero_store(output, request, key_head * head_repeat + j);
                }
                continue;
            }

            float64_pair_t q = load_head(qkv, key_head * head_dim, request);
            float64_pair_t k = load_head(qkv, key_offset + key_head * head_dim, request);

            float64 q_norm = v_f32_mac_b(q.v1, q.v1, q.v2 * q.v2);
            q_norm = v_f32_reduce_add(q_norm);
            q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
            q_norm = v_rsqrt_f32(q_norm + 1e-6f) * scale;
            q.v1 = q.v1 * q_norm;
            q.v2 = q.v2 * q_norm;

            float64 k_norm = v_f32_mac_b(k.v1, k.v1, k.v2 * k.v2);
            k_norm = v_f32_reduce_add(k_norm);
            k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
            k_norm = v_rsqrt_f32(k_norm + 1e-6f);
            k.v1 = k.v1 * k_norm;
            k.v2 = k.v2 * k_norm;

            for (int j = 0; j < head_repeat; ++j) {
                const int value_head = key_head * head_repeat + j;
                float64_pair_t v = load_head(qkv, value_offset + value_head * head_dim, request);

                int5 head_coords = {value_head, 0, 0, 0, 0};
                int5 gate_coords = {value_head, request, 0, 0, 0};
                const bf16 a_bf16 = s_bf16_ld_g(gen_addr(gate_coords, a));
                const float64 a_log = s_f32_ld_g(gen_addr(head_coords, A_log));
                const float64 dt = s_f32_ld_g(gen_addr(head_coords, dt_bias));
                const float64 xg = (float)a_bf16 + dt;

                // Stock softplus: log1p(exp(x)) with the small-value branch,
                // then the torch threshold that returns x for x > 20.
                const float64 exponential = v_exp_cephes_f32(xg);
                const float64 small = v_f32_mul_b(
                    exponential,
                    v_f32_mac_b(exponential, 0.5f, (float64)1.0f, SW_NEG));
                const bool64 use_small = v_f32_cmp_less_b(exponential, 5.0e-3f);
                const float64 regular = v_log_f32(1.0f + exponential);
                float64 softplus = v_f32_mov_vb(
                    small, 0, regular, use_small, 0);
                const bool64 large = v_f32_cmp_grt_b(xg, 20.0f);
                softplus = v_f32_mov_vb(xg, 0, softplus, large, 0);

                const float64 decay = v_exp_cephes_f32(-v_exp_cephes_f32(a_log) * softplus);
                const bf16 b_bf16 = s_bf16_ld_g(gen_addr(gate_coords, b));
                const float64 beta = v_sigmoid_f32((float)b_bf16);

                const float64_pair_t o = gdn_core_recurrence_head(
                    ssm_state, load_slot, store_slot, value_head, q, k, v, decay, beta, lanes);
                int5 output_coords = {0, value_head, request, 0, 0};
                v_f32_st_tnsr(output_coords, output, o.v1);
                output_coords[0] = 64;
                v_f32_st_tnsr(output_coords, output, o.v2);
            }
        }
    }
}
