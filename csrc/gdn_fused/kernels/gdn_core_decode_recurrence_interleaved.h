// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence for one value head, four value rows per iteration (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1,
// kernels/gdn_core_decode_interleaved_gaudi2.c). The contract of gdn_core_decode_recurrence.h: reads the head's
// 128x128 fp32 state from load_slot, writes it to store_slot, returns the head's 128 outputs (linear-order halves).
//
// Iteration `lane` (0..31) carries rows lane and lane + 32 of each half of the head (rows lane, lane + 32, 64 + lane,
// 96 + lane). Every row's operations and their order are those of the one-row loop, and rows share only read-only
// q, k, v, decay and beta, so every output and state bit is the same. tpc-clang schedules one basic block at a
// time and fills the latency of a row's dependent chain (state -> projection -> delta -> state -> o, three
// cross-lane reductions) with nops; four independent chains in one block fill each other's (idea I27). Simulator,
// TP8 geometry (2 key heads, 6 value heads): 2.91x fewer instructions at batch 1, 16 and 32, vector local memory
// unchanged (3840 bytes); two rows gave 1.89x and eight 3.08x with 6 more spilled vectors.
//
// Ordering: each row is loaded once and stored once, the store data-depends on the load, and no two rows alias,
// so load_slot == store_slot works as in the one-row loop.
static inline float64_pair_t gdn_core_recurrence_head(
    tensor ssm_state, int load_slot, int store_slot, int value_head,
    float64_pair_t q, float64_pair_t k, float64_pair_t v, float64 decay, float64 beta, uint64 lanes)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const float64 zero = 0.0f;
    float64 out_lo = 0.0f;
    float64 out_hi = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        const bool64 mask0 = v_u32_cmp_eq_b(lanes, (uint64)(lane + 0));
        const bool64 mask1 = v_u32_cmp_eq_b(lanes, (uint64)(lane + 32));
        int5 load_a0 = {0, lane + 0, value_head, load_slot, 0};
        int5 load_b0 = {0, 64 + lane + 0, value_head, load_slot, 0};
        int5 load_a1 = {0, lane + 32, value_head, load_slot, 0};
        int5 load_b1 = {0, 64 + lane + 32, value_head, load_slot, 0};
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
        float64 v_value_a0 = v_f32_mov_vb(v.v1, 0, zero, mask0);
        float64 v_value_b0 = v_f32_mov_vb(v.v2, 0, zero, mask0);
        float64 v_value_a1 = v_f32_mov_vb(v.v1, 0, zero, mask1);
        float64 v_value_b1 = v_f32_mov_vb(v.v2, 0, zero, mask1);
        v_value_a0 = v_f32_reduce_add(v_value_a0);
        v_value_b0 = v_f32_reduce_add(v_value_b0);
        v_value_a1 = v_f32_reduce_add(v_value_a1);
        v_value_b1 = v_f32_reduce_add(v_value_b1);
        v_value_a0 = v_f32_shuffle_b(v_value_a0, broadcast_lane_zero, 0, v_value_a0);
        v_value_b0 = v_f32_shuffle_b(v_value_b0, broadcast_lane_zero, 0, v_value_b0);
        v_value_a1 = v_f32_shuffle_b(v_value_a1, broadcast_lane_zero, 0, v_value_a1);
        v_value_b1 = v_f32_shuffle_b(v_value_b1, broadcast_lane_zero, 0, v_value_b1);
        state_lo_a0 = state_lo_a0 * decay;
        state_hi_a0 = state_hi_a0 * decay;
        state_lo_b0 = state_lo_b0 * decay;
        state_hi_b0 = state_hi_b0 * decay;
        state_lo_a1 = state_lo_a1 * decay;
        state_hi_a1 = state_hi_a1 * decay;
        state_lo_b1 = state_lo_b1 * decay;
        state_hi_b1 = state_hi_b1 * decay;
        float64 projection_a0 = state_lo_a0 * k.v1;
        float64 projection_b0 = state_lo_b0 * k.v1;
        float64 projection_a1 = state_lo_a1 * k.v1;
        float64 projection_b1 = state_lo_b1 * k.v1;
        projection_a0 = v_f32_mac_b(state_hi_a0, k.v2, projection_a0);
        projection_b0 = v_f32_mac_b(state_hi_b0, k.v2, projection_b0);
        projection_a1 = v_f32_mac_b(state_hi_a1, k.v2, projection_a1);
        projection_b1 = v_f32_mac_b(state_hi_b1, k.v2, projection_b1);
        projection_a0 = v_f32_reduce_add(projection_a0);
        projection_b0 = v_f32_reduce_add(projection_b0);
        projection_a1 = v_f32_reduce_add(projection_a1);
        projection_b1 = v_f32_reduce_add(projection_b1);
        projection_a0 = v_f32_shuffle_b(projection_a0, broadcast_lane_zero, 0, projection_a0);
        projection_b0 = v_f32_shuffle_b(projection_b0, broadcast_lane_zero, 0, projection_b0);
        projection_a1 = v_f32_shuffle_b(projection_a1, broadcast_lane_zero, 0, projection_a1);
        projection_b1 = v_f32_shuffle_b(projection_b1, broadcast_lane_zero, 0, projection_b1);
        const float64 delta_a0 = (v_value_a0 - projection_a0) * beta;
        const float64 delta_b0 = (v_value_b0 - projection_b0) * beta;
        const float64 delta_a1 = (v_value_a1 - projection_a1) * beta;
        const float64 delta_b1 = (v_value_b1 - projection_b1) * beta;
        state_lo_a0 = v_f32_mac_b(k.v1, delta_a0, state_lo_a0);
        state_hi_a0 = v_f32_mac_b(k.v2, delta_a0, state_hi_a0);
        state_lo_b0 = v_f32_mac_b(k.v1, delta_b0, state_lo_b0);
        state_hi_b0 = v_f32_mac_b(k.v2, delta_b0, state_hi_b0);
        state_lo_a1 = v_f32_mac_b(k.v1, delta_a1, state_lo_a1);
        state_hi_a1 = v_f32_mac_b(k.v2, delta_a1, state_hi_a1);
        state_lo_b1 = v_f32_mac_b(k.v1, delta_b1, state_lo_b1);
        state_hi_b1 = v_f32_mac_b(k.v2, delta_b1, state_hi_b1);
        float64 out_value_a0 = state_lo_a0 * q.v1;
        float64 out_value_b0 = state_lo_b0 * q.v1;
        float64 out_value_a1 = state_lo_a1 * q.v1;
        float64 out_value_b1 = state_lo_b1 * q.v1;
        out_value_a0 = v_f32_mac_b(state_hi_a0, q.v2, out_value_a0);
        out_value_b0 = v_f32_mac_b(state_hi_b0, q.v2, out_value_b0);
        out_value_a1 = v_f32_mac_b(state_hi_a1, q.v2, out_value_a1);
        out_value_b1 = v_f32_mac_b(state_hi_b1, q.v2, out_value_b1);
        out_value_a0 = v_f32_reduce_add(out_value_a0);
        out_value_b0 = v_f32_reduce_add(out_value_b0);
        out_value_a1 = v_f32_reduce_add(out_value_a1);
        out_value_b1 = v_f32_reduce_add(out_value_b1);
        int5 store_a0 = {0, lane + 0, value_head, store_slot, 0};
        v_f32_st_tnsr(store_a0, ssm_state, state_lo_a0);
        store_a0[0] = 64;
        v_f32_st_tnsr(store_a0, ssm_state, state_hi_a0);
        int5 store_b0 = {0, 64 + lane + 0, value_head, store_slot, 0};
        v_f32_st_tnsr(store_b0, ssm_state, state_lo_b0);
        store_b0[0] = 64;
        v_f32_st_tnsr(store_b0, ssm_state, state_hi_b0);
        int5 store_a1 = {0, lane + 32, value_head, store_slot, 0};
        v_f32_st_tnsr(store_a1, ssm_state, state_lo_a1);
        store_a1[0] = 64;
        v_f32_st_tnsr(store_a1, ssm_state, state_hi_a1);
        int5 store_b1 = {0, 64 + lane + 32, value_head, store_slot, 0};
        v_f32_st_tnsr(store_b1, ssm_state, state_lo_b1);
        store_b1[0] = 64;
        v_f32_st_tnsr(store_b1, ssm_state, state_hi_b1);
        out_value_a0 = v_broadcast_element_f32(out_value_a0, 0);
        out_lo = v_f32_mov_vb(out_value_a0, 0, out_lo, mask0);
        out_value_b0 = v_broadcast_element_f32(out_value_b0, 0);
        out_hi = v_f32_mov_vb(out_value_b0, 0, out_hi, mask0);
        out_value_a1 = v_broadcast_element_f32(out_value_a1, 0);
        out_lo = v_f32_mov_vb(out_value_a1, 0, out_lo, mask1);
        out_value_b1 = v_broadcast_element_f32(out_value_b1, 0);
        out_hi = v_f32_mov_vb(out_value_b1, 0, out_hi, mask1);
    }
    float64_pair_t result;
    result.v1 = out_lo;
    result.v2 = out_hi;
    return result;
}
