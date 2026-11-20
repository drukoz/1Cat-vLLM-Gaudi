// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence of one value head over the `nodes` real nodes of one verification row
// (docs/design/flash-next/gdn-core-decode.md, "Phase 2 contract, version 2").
//
// Inputs, in the __local__ arrays that gdn_core_verify_gaudi2.c declares before including this file
// (node t's 128-wide vectors in linear lane order: [2t] = elements 0..63, [2t + 1] = elements 64..127):
//   verify_q[2t], verify_q[2t + 1]     q of node t, L2-normalized and multiplied by scale
//   verify_k[2t], verify_k[2t + 1]     k of node t, L2-normalized
//   verify_vo[2t], verify_vo[2t + 1]   v of node t for this value head (replaced by o, see Output)
//   verify_store_slot[t]               SSM slot node t's state is stored to, or -1 for "not stored"
// and in two registers, to keep them out of the 16 KiB of vector local memory:
//   decay_nodes, beta_nodes            this value head's decay and beta of node t in lane t of every 16-lane
//                                      dual group (lanes t, 16 + t, 32 + t, 48 + t), t < 8
// Output: verify_vo[2t], verify_vo[2t + 1] = o of node t (linear lanes), for t < nodes. v and o share one array
// to save vector local memory: value row i reads only element i of v and then writes only element i of o, so
// while row i is processed the elements below i already hold o and the elements from i up still hold v.
//
// Reads the head's 128x128 fp32 state from load_slot once, row by row; each value row is carried through
// nodes 0 .. nodes - 1 in registers and stored after node t wherever verify_store_slot[t] >= 0.
//
// Gate lanes: v_f32_shuffle_b with every index byte 0x80 | t gives all 16 lanes of a dual group the value of
// that group's lane t, t < 8 (measured in the simulator: bit 7 enables the write, bits 0-2 are the lane, bit 3
// is ignored, so 0x88 reads lane 0 again), so one shuffle turns a gate register into node t's broadcast vector.
//
// Instruction count: tpc-clang schedules one basic block at a time and fills the latency of the dependent
// chain state -> projection -> delta -> state -> o with nops, so the node loop is laid out to keep everything
// else off that chain. The next node's decay is broadcast while this node runs, and the state store of node t
// sits at the top of iteration t + 1 (and after the loop for the last node), which leaves the rest of the
// iteration as one block. With the gates broadcast where they are used and the store at the end of the
// iteration this kernel executes 7.9 % more instructions at T = 4 (simulator, 2 key heads, 6 value heads).
//
// Ordering: no ld/st guard is needed (contrast the conv state in gdn_core_verify_gaudi2.c). Each value row is
// loaded from load_slot exactly once and every store of that row data-depends on the loaded row (it stores the
// row's updated values), and stores of row i never alias a load of another row. load_slot equaling the store
// slot of some node is the normal case and works.
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

    for (int vh = 0; vh < 2; ++vh) {
        for (int lane = 0; lane < 64; ++lane) {
            const int value_row = vh * 64 + lane;

            int5 load_coords = {0, value_row, value_head, load_slot, 0};
            float64 state_lo = v_f32_ld_tnsr_b(load_coords, ssm_state);
            load_coords[0] = 64;
            float64 state_hi = v_f32_ld_tnsr_b(load_coords, ssm_state);

            const bool64 row_lane = v_u32_cmp_eq_b(lanes, (uint64)lane);

            float64 decay = v_f32_shuffle_b(decay_nodes, broadcast_lane_zero, 0, zero);   // node 0
            int pending_slot = -1;   // where the state after the previous node goes; -1 = nowhere
            for (int t = 0; t < nodes; ++t) {
                if (pending_slot >= 0) {
                    gdn_core_verify_state_store(ssm_state, value_row, value_head, pending_slot, state_lo, state_hi);
                }

                const float64 vo_half = verify_vo[2 * t + vh];  // element `lane` is still v, the ones below are o
                float64 v_value = v_f32_mov_vb(vo_half, 0, zero, row_lane);
                v_value = v_f32_reduce_add(v_value);
                v_value = v_f32_shuffle_b(v_value, broadcast_lane_zero, 0, v_value);

                state_lo = state_lo * decay;
                state_hi = state_hi * decay;
                // Node t + 1's decay, broadcast now so that it is ready when the next iteration starts. After
                // the last node it reads a lane that nothing uses.
                const uchar256 next_node = (unsigned char)(0x80 | (t + 1));
                decay = v_f32_shuffle_b(decay_nodes, next_node, 0, decay);

                const float64 k_lo = verify_k[2 * t];
                const float64 k_hi = verify_k[2 * t + 1];
                float64 projection = state_lo * k_lo;
                projection = v_f32_mac_b(state_hi, k_hi, projection);
                projection = v_f32_reduce_add(projection);
                projection = v_f32_shuffle_b(projection, broadcast_lane_zero, 0, projection);
                const uchar256 this_node = (unsigned char)(0x80 | t);
                const float64 beta = v_f32_shuffle_b(beta_nodes, this_node, 0, zero);
                const float64 delta = (v_value - projection) * beta;

                state_lo = v_f32_mac_b(k_lo, delta, state_lo);
                state_hi = v_f32_mac_b(k_hi, delta, state_hi);

                float64 out_value = state_lo * verify_q[2 * t];
                out_value = v_f32_mac_b(state_hi, verify_q[2 * t + 1], out_value);
                out_value = v_f32_reduce_add(out_value);
                out_value = v_broadcast_element_f32(out_value, 0);
                verify_vo[2 * t + vh] = v_f32_mov_vb(out_value, 0, vo_half, row_lane);

                pending_slot = verify_store_slot[t];
            }
            if (pending_slot >= 0) {
                gdn_core_verify_state_store(ssm_state, value_row, value_head, pending_slot, state_lo, state_hi);
            }
        }
    }
}
