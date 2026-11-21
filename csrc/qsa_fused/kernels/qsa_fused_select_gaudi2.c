// SPDX-License-Identifier: Apache-2.0
// qsa_fused_select: the selected blocks of every decode row from qsa_fused_threshold's meta (VLLM_HPU_QSA_FUSED=3).
//
//   meta    {4 + 2 NW, T} i32   qsa_fused_threshold's output
//   blocks  {topk, T} i32       out: the selected block indices, ascending, then -1
//
// Prefix mode (meta[1] = -1): blocks 0 .. visible - 1. Otherwise every block of the greater bitmap and the first
// topk - G blocks of the equal bitmap, in ascending block order: topk blocks; ties at the cut go to the lowest
// block index (torch.topk leaves that choice open). The bitmaps are walked word by word with scalar code (the
// threshold kernel stored them; a kernel cannot read back what it stores, hence two nodes). One point is one row.
#pragma clang fp contract(off)

#define QSA_LANES32 64

void main(tensor meta, tensor blocks, int ratio_log2, int topk)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int words = (get_dim_size(meta, 0) - 4) >> 1;
    const int64 lanes = (int64)V_LANE_ID_32;

    for (int row = index_start[0]; row < index_end[0]; ++row) {
        const int visible = s_i32_ld_g(gen_addr((int5){0, row, 0, 0, 0}, meta));
        const int theta = s_i32_ld_g(gen_addr((int5){1, row, 0, 0, 0}, meta));
        if (theta < 0) {  // the causal prefix
            for (int first = 0; first < topk; first += QSA_LANES32) {
                const int64 ids = lanes + first;
                const int64 value = v_i32_sel_less_i32_b(ids, visible, ids, (int64)-1);
                const int left = topk - first;
                if (left >= QSA_LANES32) {
                    v_i32_st_tnsr((int5){first, row, 0, 0, 0}, blocks, value);
                } else {
                    v_i32_st_tnsr_partial((int5){first, row, 0, 0, 0}, blocks, value, left - 1, 0);
                }
            }
            continue;
        }
        const int greater = s_i32_ld_g(gen_addr((int5){2, row, 0, 0, 0}, meta));
        const int budget = topk - greater;  // equal blocks to take, lowest first
        int taken = 0;
        int written = 0;
        int buffered = 0;
        int64 ids = -1;
        const int last_word = (visible + 31) >> 5;
        for (int w = 0; w < last_word && written < topk; ++w) {
            unsigned keep = s_u32_ld_g(gen_addr((int5){4 + w, row, 0, 0, 0}, meta));
            unsigned equal = s_u32_ld_g(gen_addr((int5){4 + words + w, row, 0, 0, 0}, meta));
            while (equal && taken < budget) {
                keep |= equal & -equal;
                equal &= equal - 1;
                ++taken;
            }
            while (keep && written < topk) {
                const int block = (w << 5) + s_u32_find_first(keep, SW_FIND_ONE | SW_LSB);
                ids = v_i32_mov_vb((int64)block, 0, ids, v_i32_cmp_eq_b(lanes, buffered), 0);
                ++written;
                if (++buffered == QSA_LANES32) {
                    v_i32_st_tnsr((int5){written - QSA_LANES32, row, 0, 0, 0}, blocks, ids);
                    buffered = 0;
                    ids = -1;
                }
                keep &= keep - 1;
            }
        }
        if (buffered) v_i32_st_tnsr_partial((int5){written - buffered, row, 0, 0, 0}, blocks, ids, buffered - 1, 0);
        for (int first = written; first < topk; first += QSA_LANES32) {  // only when fewer than topk were set
            const int left = topk - first;
            if (left >= QSA_LANES32) {
                v_i32_st_tnsr((int5){first, row, 0, 0, 0}, blocks, (int64)-1);
            } else {
                v_i32_st_tnsr_partial((int5){first, row, 0, 0, 0}, blocks, (int64)-1, left - 1, 0);
            }
        }
    }
}
