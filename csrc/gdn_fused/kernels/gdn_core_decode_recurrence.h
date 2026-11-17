// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence for one value head: reads the head's 128x128 fp32 state from
// load_slot, writes it to store_slot, and returns the head's 128 outputs. v, q, k and the
// result are linear-order heads (.v1 = elements 0..63, .v2 = elements 64..127).
static inline float64_pair_t gdn_core_recurrence_head(
    tensor ssm_state, int load_slot, int store_slot, int value_head,
    float64_pair_t q, float64_pair_t k, float64_pair_t v, float64 decay, float64 beta, uint64 lanes)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const float64 zero = 0.0f;
    float64 out_lo = 0.0f;
    float64 out_hi = 0.0f;

    for (int vh = 0; vh < 2; ++vh) {
        const float64 v_half = vh == 0 ? v.v1 : v.v2;
        float64 out_half = 0.0f;
        for (int lane = 0; lane < 64; ++lane) {
            const int value_row = vh * 64 + lane;

            int5 load_coords = {0, value_row, value_head, load_slot, 0};
            float64 state_lo = v_f32_ld_tnsr_b(load_coords, ssm_state);
            load_coords[0] = 64;
            float64 state_hi = v_f32_ld_tnsr_b(load_coords, ssm_state);

            const bool64 row_lane = v_u32_cmp_eq_b(lanes, (uint64)lane);
            float64 v_value = v_f32_mov_vb(v_half, 0, zero, row_lane);
            v_value = v_f32_reduce_add(v_value);
            v_value = v_f32_shuffle_b(v_value, broadcast_lane_zero, 0, v_value);

            state_lo = state_lo * decay;
            state_hi = state_hi * decay;

            float64 projection = state_lo * k.v1;
            projection = v_f32_mac_b(state_hi, k.v2, projection);
            projection = v_f32_reduce_add(projection);
            projection = v_f32_shuffle_b(projection, broadcast_lane_zero, 0, projection);
            const float64 delta = (v_value - projection) * beta;

            state_lo = v_f32_mac_b(k.v1, delta, state_lo);
            state_hi = v_f32_mac_b(k.v2, delta, state_hi);

            float64 out_value = state_lo * q.v1;
            out_value = v_f32_mac_b(state_hi, q.v2, out_value);
            out_value = v_f32_reduce_add(out_value);

            int5 store_coords = {0, value_row, value_head, store_slot, 0};
            v_f32_st_tnsr(store_coords, ssm_state, state_lo);
            store_coords[0] = 64;
            v_f32_st_tnsr(store_coords, ssm_state, state_hi);

            out_value = v_broadcast_element_f32(out_value, 0);
            out_half = v_f32_mov_vb(out_value, 0, out_half, row_lane);
        }
        if (vh == 0) {
            out_lo = out_half;
        } else {
            out_hi = out_half;
        }
    }

    float64_pair_t result;
    result.v1 = out_lo;
    result.v2 = out_hi;
    return result;
}
