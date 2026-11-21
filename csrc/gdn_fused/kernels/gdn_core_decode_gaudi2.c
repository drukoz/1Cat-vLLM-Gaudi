// SPDX-License-Identifier: Apache-2.0
// Fused GDN decode core for Gaudi2: one step per layer per decode token.
// Runs the width-4 causal conv + SiLU over the Q, K and V channel blocks,
// the Q/K L2 normalization, the decay/beta gates, and calls the recurrence
// helper (gdn_core_decode_recurrence.h) and the gated-RMSNorm helper
// (gdn_core_decode_norm.h) for each value head.
//
// One index-space point is one key head of one request (dim0 = key head,
// dim1 = request). The point owns the conv channels of its Q block, its K
// block and the head_repeat V blocks of its value heads, so no two points
// ever touch the same channel, state row, or output slice. Each block's
// history is read from the load slot before the shifted history is written
// to the store slot, because load slot == store slot is the normal case.
// Padded rows write zeros through gdn_core_zero_store and touch no state.
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

// The interleaved build of this kernel (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1) compiles this file through a
// wrapper that names another recurrence header; everything else is this file.
#ifdef GDN_CORE_DECODE_RECURRENCE_HEADER
#include GDN_CORE_DECODE_RECURRENCE_HEADER
#else
#include "gdn_core_decode_recurrence.h"
#endif
#include "gdn_core_decode_norm.h"

// Causal width-4 conv + SiLU, fp32, for the 128 channels starting at
// `channel`. Reads history rows 0..2 from load_slot and the input token,
// then writes the shifted history (h1, h2, x) to store_slot as the raw
// bf16 vectors, after every load. The stores depend on h0 through an ordering guard, so
// load_slot == store_slot works (see the guard below).
static inline float64_pair_t conv_silu_block(tensor conv_state, tensor mixed_qkv, tensor conv_weight_t,
                                             int channel, int request, int load_slot, int store_slot)
{
    int5 hist_coords = {channel, 0, load_slot, 0, 0};
    const bfloat128 h0 = v_bf16_ld_tnsr_b(hist_coords, conv_state);
    hist_coords[1] = 1;
    const bfloat128 h1 = v_bf16_ld_tnsr_b(hist_coords, conv_state);
    hist_coords[1] = 2;
    const bfloat128 h2 = v_bf16_ld_tnsr_b(hist_coords, conv_state);
    int5 input_coords = {channel, request, 0, 0, 0};
    const bfloat128 x = v_bf16_ld_tnsr_b(input_coords, mixed_qkv);

    int5 weight_coords = {channel, 0, 0, 0, 0};
    const float64_pair_t w0 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 1;
    const float64_pair_t w1 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 2;
    const float64_pair_t w2 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 3;
    const float64_pair_t w3 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));

    float64_pair_t term = bf16_to_f32_linear(h0);
    float64_pair_t conv;
    conv.v1 = term.v1 * w0.v1;
    conv.v2 = term.v2 * w0.v2;
    term = bf16_to_f32_linear(h1);
    conv.v1 = v_f32_mac_b(term.v1, w1.v1, conv.v1);
    conv.v2 = v_f32_mac_b(term.v2, w1.v2, conv.v2);
    term = bf16_to_f32_linear(h2);
    conv.v1 = v_f32_mac_b(term.v1, w2.v1, conv.v1);
    conv.v2 = v_f32_mac_b(term.v2, w2.v2, conv.v2);
    term = bf16_to_f32_linear(x);
    conv.v1 = v_f32_mac_b(term.v1, w3.v1, conv.v1);
    conv.v2 = v_f32_mac_b(term.v2, w3.v2, conv.v2);

    // Ordering guard: tpc-clang treats ld_tnsr/st_tnsr on conv_state as independent and
    // otherwise hoists these stores above the h0 load, so with load_slot == store_slot h0
    // reads the new history (RUK-2829). h0 > +inf is never true, so the moves keep h1/h2/x
    // bit for bit but make every store depend on h0.
    const bool128 never = v_bf16_cmp_grt_b(h0, (bf16)(1.0f / 0.0f));
    const bfloat128 s1 = v_bf16_mov_vb(h0, 0, h1, never, 0);
    const bfloat128 s2 = v_bf16_mov_vb(h0, 0, h2, never, 0);
    const bfloat128 s3 = v_bf16_mov_vb(h0, 0, x, never, 0);
    hist_coords[2] = store_slot;
    hist_coords[1] = 0;
    v_bf16_st_tnsr(hist_coords, conv_state, s1);
    hist_coords[1] = 1;
    v_bf16_st_tnsr(hist_coords, conv_state, s2);
    hist_coords[1] = 2;
    v_bf16_st_tnsr(hist_coords, conv_state, s3);

    float64_pair_t act;
    act.v1 = conv.v1 * v_sigmoid_f32(conv.v1);
    act.v2 = conv.v2 * v_sigmoid_f32(conv.v2);
    return act;
}

void main(tensor conv_state, tensor ssm_state, tensor mixed_qkv, tensor z, tensor a, tensor b,
          tensor A_log, tensor dt_bias, tensor conv_weight_t, tensor norm_weight,
          tensor load_indices, tensor store_indices, tensor output,
          float scale, float eps, int gate_activation)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const uint64 lanes = read_lane_id_4b_b();
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int head_dim = 128;
    const int value_heads = get_dim_size(ssm_state, 2);
    const int state_slots = get_dim_size(ssm_state, 3);
    const int conv_slots = get_dim_size(conv_state, 2);
    const int channels = get_dim_size(mixed_qkv, 0);
    const int key_heads = (channels - value_heads * head_dim) / (2 * head_dim);
    const int head_repeat = value_heads / key_heads;
    const int key_offset = key_heads * head_dim;
    const int value_offset = 2 * key_offset;

    for (int request = index_start[1]; request < index_end[1]; ++request) {
        int5 slot_coords = {request, 0, 0, 0, 0};
        const int load_slot = s_i32_ld_g(gen_addr(slot_coords, load_indices));
        const int store_slot = s_i32_ld_g(gen_addr(slot_coords, store_indices));
        const bool valid = load_slot >= 0 && load_slot < state_slots && load_slot < conv_slots &&
                           store_slot >= 0 && store_slot < state_slots && store_slot < conv_slots;

        for (int key_head = index_start[0]; key_head < index_end[0]; ++key_head) {
            if (!valid) {
                for (int j = 0; j < head_repeat; ++j) {
                    gdn_core_zero_store(output, request, key_head * head_repeat + j);
                }
                continue;
            }

            float64_pair_t q = conv_silu_block(conv_state, mixed_qkv, conv_weight_t,
                                               key_head * head_dim, request, load_slot, store_slot);
            float64_pair_t k = conv_silu_block(conv_state, mixed_qkv, conv_weight_t,
                                               key_offset + key_head * head_dim, request, load_slot, store_slot);

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
                float64_pair_t v = conv_silu_block(conv_state, mixed_qkv, conv_weight_t,
                                                   value_offset + value_head * head_dim, request,
                                                   load_slot, store_slot);

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
                gdn_core_norm_store(z, norm_weight, output, request, value_head, o, eps, gate_activation);
            }
        }
    }
}
