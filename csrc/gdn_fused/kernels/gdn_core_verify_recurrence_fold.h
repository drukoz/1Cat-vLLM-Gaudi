// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence of one value head over the `nodes` nodes of one verification row, four value rows per
// iteration with their row sums folded together (VLLM_HPU_GDN_VERIFY_VARIANT 6 and 7,
// kernels/gdn_core_verify_interleaved_fold_gaudi2.c; docs/design/flash-next/hc-kernels.md, "GDN verify"). The
// contract of gdn_core_verify_recurrence.h: the same inputs and outputs in the same __local__ arrays.
//
// Iteration (side, j), side 0..1 and j 0..15, carries the four value rows side * 64 + 16 * d + j, d = 0..3: row d's
// element of v and of o is lane 16 * d + j of the half's vector, the same place in every dual group. For each
// node the two sums of each row (the projection of its state on k and its output on q) are not reduced one by
// one with v_f32_reduce_add (12 instructions each, plus a broadcast): gdn_verify_fold4 folds the four rows'
// product vectors into one vector whose dual group d holds row d's sum, by the very additions of
// v_f32_reduce_add, in 20 instructions for the four. v of the four rows is one shuffle of the half's vector
// (two for j >= 8), delta is computed for the four rows at once, and o lands in the half's vector with one
// masked move. Every row's operations and their order are those of the one-row loop, so every output and state
// bit is the same.
//
// v_f32_reduce_add of a vector x (tpc-clang 2.7.33 emits it as six move-and-add stages): lane i adds lane i ^ 16
// (dual groups 0 with 1, 2 with 3), then lane i ^ 32, then lane i ^ 8 (the two groups of a dual group), then lane
// i ^ 1, i ^ 2, i ^ 4 inside its group, and every lane ends with the same sum. Measured in the simulator: that
// sequence equals v_f32_reduce_add in all 64 lanes of 4,000 random vectors, and gdn_verify_fold4 equals it in
// every lane of the row's dual group for 4,000 random quadruples (with -0.0 elements). The fold does the first
// two stages on pairs of vectors at once: for lane bit 4, the lanes with the bit clear keep x0's sums and those
// with it set keep x1's (dual groups 0 and 2 hold x0's lanes and their partners, 1 and 3 x1's); bit 5 then pairs
// the two results the same way, so dual group d holds x_d's partial sums of lane positions i, i ^ 16, i ^ 32,
// i ^ 48 added in v_f32_reduce_add's order, and the remaining four stages run once for the four rows. Addition is
// commutative bit for bit, so the order of the two operands of each addition does not matter.
static inline void gdn_core_verify_state_store(
    tensor ssm_state, int value_row, int value_head, int slot, float64 state_lo, float64 state_hi)
{
    int5 store_coords = {0, value_row, value_head, slot, 0};
    v_f32_st_tnsr(store_coords, ssm_state, state_lo);
    store_coords[0] = 64;
    v_f32_st_tnsr(store_coords, ssm_state, state_hi);
}

// Every lane of dual group d of the result holds the sum of the 64 lanes of x_d (d = 0..3), bit for bit as
// v_f32_reduce_add(x_d). bit4_clear, bit5_clear: lanes whose lane number has bit 4 / bit 5 clear. xor1, xor2, xor4:
// shuffle indices 0x80 | ((lane & 7) ^ 1, 2, 4). The shuffle reads lanes 0-7 of a dual group only
// (gdn_core_verify_recurrence.h, "Gate lanes"); after the lane-bit-3 stage lanes i and i ^ 8 hold the same value,
// so lane 8 + m reading lane m ^ s reads its partner's value.
//
// Every move below writes all lanes, so its income operand (the register it writes) is any dead value: passing one
// spares tpc-clang the copy it makes of an income that is still live. The lane-bit-4 stage of a pair: x1 plus its
// partners in every lane, then x0 plus its partners written over the lanes with bit 4 clear.
static inline float64 gdn_verify_fold4(float64 x0, float64 x1, float64 x2, float64 x3, bool64 bit4_clear,
                                       bool64 bit5_clear, uchar256 xor1, uchar256 xor2, uchar256 xor4)
{
    const float64 partner1 = v_f32_mov_dual_group_all_b(x1, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), x2);
    const float64 partner0 = v_f32_mov_dual_group_all_b(x0, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), x3);
    const float64 partner3 = v_f32_mov_dual_group_all_b(x3, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), x1);
    const float64 partner2 = v_f32_mov_dual_group_all_b(x2, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), x0);
    float64 sum01 = x1 + partner1;
    sum01 = v_f32_add_vb(x0, partner0, 0, sum01, bit4_clear, 0);
    float64 sum23 = x3 + partner3;
    sum23 = v_f32_add_vb(x2, partner2, 0, sum23, bit4_clear, 0);
    // Lane bit 5: dual groups 0 and 1 hold sum01's sums with their partners 32 lanes away, 2 and 3 sum23's.
    const float64 partner23 = v_f32_mov_dual_group_all_b(sum23, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), partner1);
    const float64 partner01 = v_f32_mov_dual_group_all_b(sum01, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), partner3);
    float64 x = sum23 + partner23;
    x = v_f32_add_vb(sum01, partner01, 0, x, bit5_clear, 0);
    // Lane bit 3 (the two groups of a dual group), then bits 0, 1 and 2 inside each group.
    float64 moved = v_f32_mov_group_b(x, 0xffffffff, 63, partner23);
    x = x + moved;
    moved = v_f32_shuffle_b(x, xor1, 0, moved);
    x = x + moved;
    moved = v_f32_shuffle_b(x, xor2, 0, moved);
    x = x + moved;
    moved = v_f32_shuffle_b(x, xor4, 0, moved);
    x = x + moved;
    return x;
}

// A row group's registers: S_ is the prefix of one of two sets (A_, B_), so that one group's last output can
// overlap the next group's first update.
#define GDN_FOLD_GROUP_DECLARE(S)                                                                                 \
    int S##side = 0;                                                                                              \
    int S##row_0 = 0;                                                                                             \
    bool S##upper = 0;                                                                                            \
    uchar256 S##pick = 0x80;                                                                                      \
    bool64 S##row_lanes = v_u32_cmp_eq_b(dual_group_lanes, (uint64)0);                                           \
    float64 S##state_lo_0 = zero, S##state_lo_1 = zero, S##state_lo_2 = zero, S##state_lo_3 = zero;             \
    float64 S##state_hi_0 = zero, S##state_hi_1 = zero, S##state_hi_2 = zero, S##state_hi_3 = zero;             \
    float64 S##decay = zero;                                                                                      \
    float64 S##vo_node = zero

// Row group g (0..31): side g / 16, j = g % 16, rows side * 64 + 16 * d + j; its state loaded from load_slot.
#define GDN_FOLD_GROUP_BEGIN(S, g)                                                                                \
    do {                                                                                                          \
        const int group_ = (g);                                                                                   \
        const int j_ = group_ & 15;                                                                               \
        S##side = group_ >> 4;                                                                                    \
        S##row_0 = S##side * 64 + j_;                                                                             \
        /* lane 16 * d + j of the half's v/o vector is row d of the group */                                      \
        S##row_lanes = v_u32_cmp_eq_b(dual_group_lanes, (uint64)j_);                                              \
        /* v of the four rows: for j >= 8 group 1 of each dual group is first moved onto group 0 (the shuffle */  \
        /* reads lanes 0-7 of a dual group), then lane j & 7 goes to the 16 lanes of the dual group */            \
        S##upper = j_ >= 8;                                                                                       \
        S##pick = (unsigned char)(0x80 | (j_ & 7));                                                               \
        int5 load_ = {0, S##row_0, value_head, load_slot, 0};                                                     \
        S##state_lo_0 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0 + 16;                                                                                 \
        S##state_lo_1 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0 + 32;                                                                                 \
        S##state_lo_2 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0 + 48;                                                                                 \
        S##state_lo_3 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[0] = 64;                                                                                            \
        S##state_hi_3 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0 + 32;                                                                                 \
        S##state_hi_2 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0 + 16;                                                                                 \
        S##state_hi_1 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        load_[1] = S##row_0;                                                                                      \
        S##state_hi_0 = v_f32_ld_tnsr_b(load_, ssm_state);                                                        \
        S##decay = v_f32_shuffle_b(decay_nodes, broadcast_lane_zero, 0, zero);                                    \
    } while (0)

// Node t's update of the group's four rows: decay, the projection on k, delta, the state update (the one-row
// loop's operations in its order). vo: verify_vo[2t + side] as it is before node t's output is merged into it;
// k_lo, k_hi: verify_k[2t], verify_k[2t + 1]. S##decay holds node t's decay on entry and node t + 1's on exit.
#define GDN_FOLD_UPDATE(S, t, vo, k_lo, k_hi)                                                                     \
    do {                                                                                                          \
        const float64 vo_ = (vo);                                                                                 \
        const float64 k_lo_ = (k_lo);                                                                             \
        const float64 k_hi_ = (k_hi);                                                                             \
        /* the one-row loop takes the row's v as the sum of a vector holding v in the row's lane and +0.0 */      \
        /* elsewhere: that is v, except that -0.0 becomes +0.0, which "+ 0.0f" reproduces */                      \
        float64 v_rows_ = v_f32_mov_group_b(vo_, 0xffffffff, 61, vo_, S##upper, 0);                               \
        v_rows_ = v_f32_shuffle_b(v_rows_, S##pick, 0, v_rows_) + 0.0f;                                           \
        S##state_lo_0 = S##state_lo_0 * S##decay;                                                                 \
        S##state_hi_0 = S##state_hi_0 * S##decay;                                                                 \
        S##state_lo_1 = S##state_lo_1 * S##decay;                                                                 \
        S##state_hi_1 = S##state_hi_1 * S##decay;                                                                 \
        S##state_lo_2 = S##state_lo_2 * S##decay;                                                                 \
        S##state_hi_2 = S##state_hi_2 * S##decay;                                                                 \
        S##state_lo_3 = S##state_lo_3 * S##decay;                                                                 \
        S##state_hi_3 = S##state_hi_3 * S##decay;                                                                 \
        S##decay = v_f32_shuffle_b(decay_nodes, (uchar256)(unsigned char)(0x80 | ((t) + 1)), 0, S##decay);        \
        float64 projection_0_ = S##state_lo_0 * k_lo_;                                                            \
        float64 projection_1_ = S##state_lo_1 * k_lo_;                                                            \
        float64 projection_2_ = S##state_lo_2 * k_lo_;                                                            \
        float64 projection_3_ = S##state_lo_3 * k_lo_;                                                            \
        projection_0_ = v_f32_mac_b(S##state_hi_0, k_hi_, projection_0_);                                         \
        projection_1_ = v_f32_mac_b(S##state_hi_1, k_hi_, projection_1_);                                         \
        projection_2_ = v_f32_mac_b(S##state_hi_2, k_hi_, projection_2_);                                         \
        projection_3_ = v_f32_mac_b(S##state_hi_3, k_hi_, projection_3_);                                         \
        const float64 projection_ = gdn_verify_fold4(projection_0_, projection_1_, projection_2_, projection_3_,  \
                                                     bit4_clear, bit5_clear, xor1, xor2, xor4);                   \
        const float64 beta_ = v_f32_shuffle_b(beta_nodes, (uchar256)(unsigned char)(0x80 | (t)), 0, zero);       \
        /* dual group d: delta of row d */                                                                        \
        const float64 delta_ = (v_rows_ - projection_) * beta_;                                                   \
        const float64 delta_0_ = v_f32_mov_dual_group_all_b(delta_, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), zero); \
        const float64 delta_1_ = v_f32_mov_dual_group_all_b(delta_, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), zero); \
        const float64 delta_2_ = v_f32_mov_dual_group_all_b(delta_, 0xffffffff, 2, 2, 2, 2, MkWrA(3, 3, 3, 3), zero); \
        const float64 delta_3_ = v_f32_mov_dual_group_all_b(delta_, 0xffffffff, 3, 3, 3, 3, MkWrA(3, 3, 3, 3), zero); \
        S##state_lo_0 = v_f32_mac_b(k_lo_, delta_0_, S##state_lo_0);                                              \
        S##state_hi_0 = v_f32_mac_b(k_hi_, delta_0_, S##state_hi_0);                                              \
        S##state_lo_1 = v_f32_mac_b(k_lo_, delta_1_, S##state_lo_1);                                              \
        S##state_hi_1 = v_f32_mac_b(k_hi_, delta_1_, S##state_hi_1);                                              \
        S##state_lo_2 = v_f32_mac_b(k_lo_, delta_2_, S##state_lo_2);                                              \
        S##state_hi_2 = v_f32_mac_b(k_hi_, delta_2_, S##state_hi_2);                                              \
        S##state_lo_3 = v_f32_mac_b(k_lo_, delta_3_, S##state_lo_3);                                              \
        S##state_hi_3 = v_f32_mac_b(k_hi_, delta_3_, S##state_hi_3);                                              \
    } while (0)

// Node t's output of the group's four rows from their state after node t, into lanes 16 * d + j of the half's
// vector. The vector is read again here, not taken from node t's update: with one node the next group's update
// reads it before this group's output is stored (GDN_FOLD_HANDOVER), and its other lanes must be current.
#define GDN_FOLD_OUTPUT(S, t)                                                                                     \
    do {                                                                                                          \
        const float64 q_lo_ = verify_q[2 * (t)];                                                                  \
        const float64 q_hi_ = verify_q[2 * (t) + 1];                                                              \
        float64 out_0_ = S##state_lo_0 * q_lo_;                                                                   \
        float64 out_1_ = S##state_lo_1 * q_lo_;                                                                   \
        float64 out_2_ = S##state_lo_2 * q_lo_;                                                                   \
        float64 out_3_ = S##state_lo_3 * q_lo_;                                                                   \
        out_0_ = v_f32_mac_b(S##state_hi_0, q_hi_, out_0_);                                                       \
        out_1_ = v_f32_mac_b(S##state_hi_1, q_hi_, out_1_);                                                       \
        out_2_ = v_f32_mac_b(S##state_hi_2, q_hi_, out_2_);                                                       \
        out_3_ = v_f32_mac_b(S##state_hi_3, q_hi_, out_3_);                                                       \
        const float64 out_ = gdn_verify_fold4(out_0_, out_1_, out_2_, out_3_, bit4_clear, bit5_clear,             \
                                              xor1, xor2, xor4);                                                  \
        verify_vo[2 * (t) + S##side] = v_f32_mov_vb(out_, 0, verify_vo[2 * (t) + S##side], S##row_lanes);         \
    } while (0)

#define GDN_FOLD_STORE(S, slot)                                                                                   \
    do {                                                                                                          \
        gdn_core_verify_state_store(ssm_state, S##row_0, value_head, (slot), S##state_lo_0, S##state_hi_0);       \
        gdn_core_verify_state_store(ssm_state, S##row_0 + 16, value_head, (slot), S##state_lo_1, S##state_hi_1);  \
        gdn_core_verify_state_store(ssm_state, S##row_0 + 32, value_head, (slot), S##state_lo_2, S##state_hi_2);  \
        gdn_core_verify_state_store(ssm_state, S##row_0 + 48, value_head, (slot), S##state_lo_3, S##state_hi_3);  \
    } while (0)

// Nodes 0 .. nodes - 2 of a group whose node 0 update is done: iteration t stores node t's state, runs node t's
// output and node t + 1's update (both depend only on the state after node t: two independent chains in one
// block); then node nodes - 1's state store. Node t + 1's v and k are read before node t's output is stored:
// tpc-clang does not tell the __local__ arrays (nor two elements of one) apart, and the store would otherwise hold
// the next update back until the output chain ends.
#define GDN_FOLD_GROUP_NODES(S)                                                                                   \
    do {                                                                                                          \
        for (int t = 0; t + 1 < nodes; ++t) {                                                                     \
            const int slot_ = verify_store_slot[t];                                                               \
            if (slot_ >= 0) {                                                                                     \
                GDN_FOLD_STORE(S, slot_);                                                                         \
            }                                                                                                     \
            const float64 vo_next_ = verify_vo[2 * (t + 1) + S##side];                                            \
            const float64 k_lo_next_ = verify_k[2 * (t + 1)];                                                     \
            const float64 k_hi_next_ = verify_k[2 * (t + 1) + 1];                                                 \
            GDN_FOLD_OUTPUT(S, t);                                                                                \
            GDN_FOLD_UPDATE(S, t + 1, vo_next_, k_lo_next_, k_hi_next_);                                          \
            S##vo_node = vo_next_;                                                                                \
        }                                                                                                         \
        const int last_slot_ = verify_store_slot[nodes - 1];                                                      \
        if (last_slot_ >= 0) {                                                                                    \
            GDN_FOLD_STORE(S, last_slot_);                                                                        \
        }                                                                                                         \
    } while (0)

// The last node's output of group P overlapped with the first node's update of group N (loaded here as group g).
// N's v and k of node 0 are read before P's output is stored (see GDN_FOLD_GROUP_NODES).
#define GDN_FOLD_HANDOVER(P, N, g)                                                                                \
    do {                                                                                                          \
        GDN_FOLD_GROUP_BEGIN(N, g);                                                                               \
        N##vo_node = verify_vo[N##side];                                                                          \
        const float64 k_lo_first_ = verify_k[0];                                                                  \
        const float64 k_hi_first_ = verify_k[1];                                                                  \
        GDN_FOLD_OUTPUT(P, nodes - 1);                                                                            \
        GDN_FOLD_UPDATE(N, 0, N##vo_node, k_lo_first_, k_hi_first_);                                              \
    } while (0)

// 32 row groups, alternately in the A_ and B_ registers. The handover after the last group (31, B_) begins group 0
// again in A_ only to keep that block free of branches: its update stays in registers and is never used or stored
// (its state loads read rows this point already wrote, and nothing depends on them).
static inline void gdn_core_verify_recurrence_head(
    tensor ssm_state, int load_slot, int value_head, int nodes, float64 decay_nodes, float64 beta_nodes, uint64 lanes)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const float64 zero = 0.0f;
    const uchar256 xor1 = as_uchar256(((lanes & 7) ^ (uint64)1) | (uint64)0x80);
    const uchar256 xor2 = as_uchar256(((lanes & 7) ^ (uint64)2) | (uint64)0x80);
    const uchar256 xor4 = as_uchar256(((lanes & 7) ^ (uint64)4) | (uint64)0x80);
    const uint64 dual_group_lanes = lanes & 15;
    const bool64 bit4_clear = v_u32_cmp_eq_b(lanes & 16, (uint64)0);
    const bool64 bit5_clear = v_u32_cmp_eq_b(lanes & 32, (uint64)0);
    GDN_FOLD_GROUP_DECLARE(A_);
    GDN_FOLD_GROUP_DECLARE(B_);
    GDN_FOLD_GROUP_BEGIN(A_, 0);
    A_vo_node = verify_vo[A_side];
    GDN_FOLD_UPDATE(A_, 0, A_vo_node, verify_k[0], verify_k[1]);
    for (int pair = 0; pair < 16; ++pair) {
        GDN_FOLD_GROUP_NODES(A_);
        GDN_FOLD_HANDOVER(A_, B_, 2 * pair + 1);
        GDN_FOLD_GROUP_NODES(B_);
        GDN_FOLD_HANDOVER(B_, A_, (2 * pair + 2) & 31);
    }
}

#undef GDN_FOLD_GROUP_DECLARE
#undef GDN_FOLD_GROUP_BEGIN
#undef GDN_FOLD_UPDATE
#undef GDN_FOLD_OUTPUT
#undef GDN_FOLD_STORE
#undef GDN_FOLD_GROUP_NODES
#undef GDN_FOLD_HANDOVER
