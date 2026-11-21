// SPDX-License-Identifier: Apache-2.0
// qsa_fused_threshold: the top-k threshold and selection bitmaps of every sparse-QSA decode row
// (VLLM_HPU_QSA_FUSED=3). With qsa_fused_select it replaces select_blocks (masked_fill, topk(512), the int32
// conversion) of the torch path. Exact: the threshold is the topk-th largest score key among the visible blocks.
//
//   scores     {N, T} f32        the block scores of each row (relu sums, request-major GEMM)
//   positions  {T} i32           the position p of each row
//   meta       {4 + 2 NW, T} i32 out: [0] visible = clamp(floordiv(p + 1, ratio), 0, N); [1] the threshold key, or
//                                -1 when visible <= topk (every visible block is selected); [2] G, the visible blocks
//                                with key > threshold; [3] 0; [4 + w] bit b: block 32 w + b has key > threshold;
//                                [4 + NW + w] bit b: key == threshold (NW = N / 32). Words of blocks >= visible: 0
//                                up to the last 128-block tile the row touches, unspecified past it.
//
// key(n) = the fp32 bits of scores[n] without the sign (qsa_score_key). The threshold is the largest key theta with
// #{n < visible : key(n) >= theta} >= topk, found bit by bit from bit 30 down (31 passes over the visible scores; the
// count is a lane sum, so theta never leaves the vector registers). One index-space point is one row.
#pragma clang fp contract(off)
#include "qsa_fused_common.h"

void main(tensor scores, tensor positions, tensor meta, int ratio_log2, int topk)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int columns = get_dim_size(scores, 0);
    const int words = columns >> 5;
    const int64 lanes = (int64)V_LANE_ID_32;

    for (int row = index_start[0]; row < index_end[0]; ++row) {
        const int p = s_i32_ld_g(gen_addr((int5){row, 0, 0, 0, 0}, positions));
        int visible = (p + 1) >> ratio_log2;
        visible = visible < 0 ? 0 : (visible > columns ? columns : visible);
        v_i32_st_tnsr_partial((int5){0, row, 0, 0, 0}, meta, (int64)visible, 0, 0);
        v_i32_st_tnsr_partial((int5){3, row, 0, 0, 0}, meta, (int64)0, 0, 0);
        if (visible <= topk) {  // the causal prefix: no threshold
            v_i32_st_tnsr_partial((int5){1, row, 0, 0, 0}, meta, (int64)-1, 0, 0);
            v_i32_st_tnsr_partial((int5){2, row, 0, 0, 0}, meta, (int64)visible, 0, 0);
            continue;
        }
        const int chunks = (visible + 63) >> 6;
        int64 theta = 0;
        for (int bit = 30; bit >= 0; --bit) {
            const int64 trial = theta | (int64)(1 << bit);
            int64 count = 0;
            for (int c = 0; c < chunks; ++c) {
                const int64 key = qsa_score_key(v_f32_ld_tnsr_b((int5){c << 6, row, 0, 0, 0}, scores));
                const bool64 take = v_i32_cmp_less_b(lanes + (c << 6), visible) & v_i32_cmp_geq_b(key, trial);
                count = v_i32_add_vb(count, 1, 0, count, take, 0);
            }
            const int64 total = qsa_sum_broadcast(count);
            theta = v_i32_sel_geq_i32_b(total, topk, trial, theta);
        }
        // The bitmaps, 128 blocks per tile (four words), and G.
        int64 greater = 0;
        const int tiles = (visible + 127) >> 7;
        for (int t = 0; t < tiles; ++t) {
            const int first = t << 7;
            const int64 key0 = qsa_score_key(v_f32_ld_tnsr_b((int5){first, row, 0, 0, 0}, scores));
            const int64 key1 = qsa_score_key(v_f32_ld_tnsr_b((int5){first + 64, row, 0, 0, 0}, scores));
            const bool64 live0 = v_i32_cmp_less_b(lanes + first, visible);
            const bool64 live1 = v_i32_cmp_less_b(lanes + (first + 64), visible);
            const bool64 gt0 = live0 & v_i32_cmp_grt_b(key0, theta);
            const bool64 gt1 = live1 & v_i32_cmp_grt_b(key1, theta);
            const bool64 eq0 = live0 & v_i32_cmp_eq_b(key0, theta);
            const bool64 eq1 = live1 & v_i32_cmp_eq_b(key1, theta);
            greater = v_i32_add_vb(greater, 1, 0, greater, gt0, 0);
            greater = v_i32_add_vb(greater, 1, 0, greater, gt1, 0);
            v_u32_st_tnsr_partial((int5){4 + (t << 2), row, 0, 0, 0}, meta,
                                  qsa_pack_words(qsa_pack_half(gt0), qsa_pack_half(gt1)), 3, 0);
            v_u32_st_tnsr_partial((int5){4 + words + (t << 2), row, 0, 0, 0}, meta,
                                  qsa_pack_words(qsa_pack_half(eq0), qsa_pack_half(eq1)), 3, 0);
        }
        v_i32_st_tnsr_partial((int5){1, row, 0, 0, 0}, meta, theta, 0, 0);
        v_i32_st_tnsr_partial((int5){2, row, 0, 0, 0}, meta, qsa_sum_broadcast(greater), 0, 0);
    }
}
