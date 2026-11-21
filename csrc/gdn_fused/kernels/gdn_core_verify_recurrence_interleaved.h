// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence of one value head over the `nodes` nodes of one verification row, four value rows per
// iteration (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1, kernels/gdn_core_verify_interleaved_gaudi2.c). The contract of
// gdn_core_verify_recurrence.h, the same inputs and outputs in the same __local__ arrays.
//
// Iteration `lane` (0..31) carries rows lane and lane + 32 of each half of the head. Rows of the first half read
// and write verify_vo[2t], rows of the second verify_vo[2t + 1]; a row reads only its own element of v and writes
// only its own element of o, and both rows of a half read before either writes, so the v/o sharing of the one-row
// loop holds. Every row's operations and their order are those of the one-row loop: every output and state bit is
// the same (idea I27). Simulator, TP8 geometry: 2.65x to 2.71x fewer instructions at T = 1 to 6, vector local
// memory unchanged (12544 bytes); two rows gave 1.77x.
static inline void gdn_core_verify_state_store(
    tensor ssm_state, int value_row, int value_head, int slot, float64 state_lo, float64 state_hi)
{
    int5 store_coords = {0, value_row, value_head, slot, 0};
    v_f32_st_tnsr(store_coords, ssm_state, state_lo);
    store_coords[0] = 64;
    v_f32_st_tnsr(store_coords, ssm_state, state_hi);
}

static inline void gdn_core_verify_recurrence_head(
    tensor ssm_state, int load_slot, int value_head, int nodes, float64 decay_nodes, float64 beta_nodes, uint64 lanes)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const float64 zero = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        const bool64 mask0 = v_u32_cmp_eq_b(lanes, (uint64)(lane + 0));
        const bool64 mask1 = v_u32_cmp_eq_b(lanes, (uint64)(lane + 32));
        const int row_a0 = lane + 0;
        int5 load_a0 = {0, row_a0, value_head, load_slot, 0};
        const int row_b0 = 64 + lane + 0;
        int5 load_b0 = {0, row_b0, value_head, load_slot, 0};
        const int row_a1 = lane + 32;
        int5 load_a1 = {0, row_a1, value_head, load_slot, 0};
        const int row_b1 = 64 + lane + 32;
        int5 load_b1 = {0, row_b1, value_head, load_slot, 0};
        float64 state_lo_a0 = v_f32_ld_tnsr_b(load_a0, ssm_state);
        float64 state_lo_b0 = v_f32_ld_tnsr_b(load_b0, ssm_state);
        float64 state_lo_a1 = v_f32_ld_tnsr_b(load_a1, ssm_state);
        float64 state_lo_b1 = v_f32_ld_tnsr_b(load_b1, ssm_state);
        load_a0[0] = 64;
        load_b0[0] = 64;
        load_a1[0] = 64;
        load_b1[0] = 64;
        float64 state_hi_a0 = v_f32_ld_tnsr_b(load_a0, ssm_state);
        float64 state_hi_b0 = v_f32_ld_tnsr_b(load_b0, ssm_state);
        float64 state_hi_a1 = v_f32_ld_tnsr_b(load_a1, ssm_state);
        float64 state_hi_b1 = v_f32_ld_tnsr_b(load_b1, ssm_state);
        float64 decay = v_f32_shuffle_b(decay_nodes, broadcast_lane_zero, 0, zero);
        int pending_slot = -1;
        for (int t = 0; t < nodes; ++t) {
            if (pending_slot >= 0) {
                gdn_core_verify_state_store(ssm_state, row_a0, value_head, pending_slot, state_lo_a0, state_hi_a0);
                gdn_core_verify_state_store(ssm_state, row_b0, value_head, pending_slot, state_lo_b0, state_hi_b0);
                gdn_core_verify_state_store(ssm_state, row_a1, value_head, pending_slot, state_lo_a1, state_hi_a1);
                gdn_core_verify_state_store(ssm_state, row_b1, value_head, pending_slot, state_lo_b1, state_hi_b1);
            }
            const float64 vo_0 = verify_vo[2 * t];
            const float64 vo_1 = verify_vo[2 * t + 1];
#ifdef GDN_VERIFY_LEAN_BROADCAST
            // The row's element of v in every lane by one broadcast instead of the sum of a masked vector
            // (VLLM_HPU_GDN_VERIFY_VARIANT 2 and 3). The sum adds +0.0 lanes to the element, which turns -0.0 into
            // +0.0 and keeps every other value: so does "+ 0.0f", so the bits are the sum's.
            const float64 v_value_a0 = v_broadcast_element_f32(vo_0, row_a0) + 0.0f;
            const float64 v_value_b0 = v_broadcast_element_f32(vo_1, row_a0) + 0.0f;
            const float64 v_value_a1 = v_broadcast_element_f32(vo_0, row_a1) + 0.0f;
            const float64 v_value_b1 = v_broadcast_element_f32(vo_1, row_a1) + 0.0f;
#else
            float64 v_value_a0 = v_f32_mov_vb(vo_0, 0, zero, mask0);
            float64 v_value_b0 = v_f32_mov_vb(vo_1, 0, zero, mask0);
            float64 v_value_a1 = v_f32_mov_vb(vo_0, 0, zero, mask1);
            float64 v_value_b1 = v_f32_mov_vb(vo_1, 0, zero, mask1);
            v_value_a0 = v_f32_reduce_add(v_value_a0);
            v_value_b0 = v_f32_reduce_add(v_value_b0);
            v_value_a1 = v_f32_reduce_add(v_value_a1);
            v_value_b1 = v_f32_reduce_add(v_value_b1);
            v_value_a0 = v_f32_shuffle_b(v_value_a0, broadcast_lane_zero, 0, v_value_a0);
            v_value_b0 = v_f32_shuffle_b(v_value_b0, broadcast_lane_zero, 0, v_value_b0);
            v_value_a1 = v_f32_shuffle_b(v_value_a1, broadcast_lane_zero, 0, v_value_a1);
            v_value_b1 = v_f32_shuffle_b(v_value_b1, broadcast_lane_zero, 0, v_value_b1);
#endif
            state_lo_a0 = state_lo_a0 * decay;
            state_hi_a0 = state_hi_a0 * decay;
            state_lo_b0 = state_lo_b0 * decay;
            state_hi_b0 = state_hi_b0 * decay;
            state_lo_a1 = state_lo_a1 * decay;
            state_hi_a1 = state_hi_a1 * decay;
            state_lo_b1 = state_lo_b1 * decay;
            state_hi_b1 = state_hi_b1 * decay;
            const uchar256 next_node = (unsigned char)(0x80 | (t + 1));
            decay = v_f32_shuffle_b(decay_nodes, next_node, 0, decay);
            const float64 k_lo = verify_k[2 * t];
            const float64 k_hi = verify_k[2 * t + 1];
            float64 projection_a0 = state_lo_a0 * k_lo;
            float64 projection_b0 = state_lo_b0 * k_lo;
            float64 projection_a1 = state_lo_a1 * k_lo;
            float64 projection_b1 = state_lo_b1 * k_lo;
            projection_a0 = v_f32_mac_b(state_hi_a0, k_hi, projection_a0);
            projection_b0 = v_f32_mac_b(state_hi_b0, k_hi, projection_b0);
            projection_a1 = v_f32_mac_b(state_hi_a1, k_hi, projection_a1);
            projection_b1 = v_f32_mac_b(state_hi_b1, k_hi, projection_b1);
            projection_a0 = v_f32_reduce_add(projection_a0);
            projection_b0 = v_f32_reduce_add(projection_b0);
            projection_a1 = v_f32_reduce_add(projection_a1);
            projection_b1 = v_f32_reduce_add(projection_b1);
            projection_a0 = v_f32_shuffle_b(projection_a0, broadcast_lane_zero, 0, projection_a0);
            projection_b0 = v_f32_shuffle_b(projection_b0, broadcast_lane_zero, 0, projection_b0);
            projection_a1 = v_f32_shuffle_b(projection_a1, broadcast_lane_zero, 0, projection_a1);
            projection_b1 = v_f32_shuffle_b(projection_b1, broadcast_lane_zero, 0, projection_b1);
            const uchar256 this_node = (unsigned char)(0x80 | t);
            const float64 beta = v_f32_shuffle_b(beta_nodes, this_node, 0, zero);
            const float64 delta_a0 = (v_value_a0 - projection_a0) * beta;
            const float64 delta_b0 = (v_value_b0 - projection_b0) * beta;
            const float64 delta_a1 = (v_value_a1 - projection_a1) * beta;
            const float64 delta_b1 = (v_value_b1 - projection_b1) * beta;
            state_lo_a0 = v_f32_mac_b(k_lo, delta_a0, state_lo_a0);
            state_hi_a0 = v_f32_mac_b(k_hi, delta_a0, state_hi_a0);
            state_lo_b0 = v_f32_mac_b(k_lo, delta_b0, state_lo_b0);
            state_hi_b0 = v_f32_mac_b(k_hi, delta_b0, state_hi_b0);
            state_lo_a1 = v_f32_mac_b(k_lo, delta_a1, state_lo_a1);
            state_hi_a1 = v_f32_mac_b(k_hi, delta_a1, state_hi_a1);
            state_lo_b1 = v_f32_mac_b(k_lo, delta_b1, state_lo_b1);
            state_hi_b1 = v_f32_mac_b(k_hi, delta_b1, state_hi_b1);
            const float64 q_lo = verify_q[2 * t];
            const float64 q_hi = verify_q[2 * t + 1];
            float64 out_a0 = state_lo_a0 * q_lo;
            float64 out_b0 = state_lo_b0 * q_lo;
            float64 out_a1 = state_lo_a1 * q_lo;
            float64 out_b1 = state_lo_b1 * q_lo;
            out_a0 = v_f32_mac_b(state_hi_a0, q_hi, out_a0);
            out_b0 = v_f32_mac_b(state_hi_b0, q_hi, out_b0);
            out_a1 = v_f32_mac_b(state_hi_a1, q_hi, out_a1);
            out_b1 = v_f32_mac_b(state_hi_b1, q_hi, out_b1);
            out_a0 = v_f32_reduce_add(out_a0);
            out_b0 = v_f32_reduce_add(out_b0);
            out_a1 = v_f32_reduce_add(out_a1);
            out_b1 = v_f32_reduce_add(out_b1);
#ifdef GDN_VERIFY_LEAN_BROADCAST
            // Lane 0 of every lane group holds the sum after the reduction (the projection above is broadcast the
            // same way): one shuffle instead of a broadcast of element 0, the same value in every lane.
            out_a0 = v_f32_shuffle_b(out_a0, broadcast_lane_zero, 0, out_a0);
            out_b0 = v_f32_shuffle_b(out_b0, broadcast_lane_zero, 0, out_b0);
            out_a1 = v_f32_shuffle_b(out_a1, broadcast_lane_zero, 0, out_a1);
            out_b1 = v_f32_shuffle_b(out_b1, broadcast_lane_zero, 0, out_b1);
#else
            out_a0 = v_broadcast_element_f32(out_a0, 0);
            out_b0 = v_broadcast_element_f32(out_b0, 0);
            out_a1 = v_broadcast_element_f32(out_a1, 0);
            out_b1 = v_broadcast_element_f32(out_b1, 0);
#endif
            float64 merged_0 = vo_0;
            merged_0 = v_f32_mov_vb(out_a0, 0, merged_0, mask0);
            merged_0 = v_f32_mov_vb(out_a1, 0, merged_0, mask1);
            float64 merged_1 = vo_1;
            merged_1 = v_f32_mov_vb(out_b0, 0, merged_1, mask0);
            merged_1 = v_f32_mov_vb(out_b1, 0, merged_1, mask1);
            verify_vo[2 * t] = merged_0;
            verify_vo[2 * t + 1] = merged_1;
            pending_slot = verify_store_slot[t];
        }
        if (pending_slot >= 0) {
            gdn_core_verify_state_store(ssm_state, row_a0, value_head, pending_slot, state_lo_a0, state_hi_a0);
            gdn_core_verify_state_store(ssm_state, row_b0, value_head, pending_slot, state_lo_b0, state_hi_b0);
            gdn_core_verify_state_store(ssm_state, row_a1, value_head, pending_slot, state_lo_a1, state_hi_a1);
            gdn_core_verify_state_store(ssm_state, row_b1, value_head, pending_slot, state_lo_b1, state_hi_b1);
        }
    }
}
