// SPDX-License-Identifier: Apache-2.0
// qsa_fused_emit: the attended K/V cache slots of every sparse-QSA decode row (VLLM_HPU_QSA_FUSED=2).
// Replaces expand_selection (qsa_sparse_ops.py), decode_selected_slots and the valid mask with
// rows_without_selection (qsa_hpu_attention.py) of the torch path; integers only, bit-exact.
//
//   blocks       {topk, T} int32   the selected blocks of each row (select_blocks: -1 = none), any order
//   positions    {T} int32         the position p of each row (-1 on a padding row under M-RoPE)
//   block_table  {MB, T} int32     the kernel block ids of each row (md.block_table)
//   slots        {W, T} int32      out: the flat K/V cache row of every attended column
//   valid        {W, T} int32      out: 1 where the column is attended, else 0
//
// W = topk * ratio + ratio - 1. Column i * ratio + r (r < ratio) is position b * ratio + r of block
// b = blocks[t, i] when b >= 0, else position 0 with valid 0. Column topk * ratio + j (j < ratio - 1)
// is the tail position ((p + 1) floordiv ratio) * ratio + j when it is <= p, else position 0 with
// valid 0. A position q maps to slot block_table[t, q >> block_log2] * B + (q & (B - 1)); a row with
// p < 0 has no attended position at all and gets valid 1 in every column, as rows_without_selection
// gives it (the softmax of an all -inf row is NaN). A block table column outside [0, MB) is clamped:
// the kernel never reads outside its tensors, and for the indexer's own selection it never happens.
//
// One index-space point is one row. Per block one scalar load of its id and one of its page, the
// ratio slots of the block go into lanes of a 64-lane int32 vector, one full vector store per
// 64 / ratio blocks. ratio and B are powers of two (the glue checks), ratio <= 64.
#pragma clang fp contract(off)

#define QSA_LANES32 64

static inline int qsa_page(tensor block_table, int column, int columns, int row)
{
    const int clamped = column < 0 ? 0 : (column >= columns ? columns - 1 : column);
    return s_i32_ld_g(gen_addr((int5){clamped, row, 0, 0, 0}, block_table));
}

void main(tensor blocks, tensor positions, tensor block_table, tensor slots, tensor valid,
          int ratio_log2, int block_log2, int topk)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int ratio = 1 << ratio_log2;
    const int block_mask = (1 << block_log2) - 1;
    const int columns = get_dim_size(block_table, 0);
    const int per_vector = QSA_LANES32 >> ratio_log2;  // blocks in one output vector
    const int64 lanes = (int64)V_LANE_ID_32;
    const int64 lane_in_block = lanes & (ratio - 1);
    const int64 lane_block = lanes >> ratio_log2;

    for (int row = index_start[0]; row < index_end[0]; ++row) {
        const int p = s_i32_ld_g(gen_addr((int5){row, 0, 0, 0, 0}, positions));
        const int no_selection = p < 0;  // rows_without_selection: every column is valid
        const int origin = qsa_page(block_table, 0, columns, row) << block_log2;  // the slot of position 0

        for (int first = 0; first < topk; first += per_vector) {
            int64 slot_vec = origin;
            int64 valid_vec = no_selection;
            const int count = topk - first < per_vector ? topk - first : per_vector;
            for (int j = 0; j < count; ++j) {
                const int b = s_i32_ld_g(gen_addr((int5){first + j, row, 0, 0, 0}, blocks));
                const int kept = b >= 0;
                const int pos0 = kept ? (b << ratio_log2) : 0;
                const int base = (qsa_page(block_table, pos0 >> block_log2, columns, row) << block_log2) +
                                 (pos0 & block_mask);
                const bool64 group = v_i32_cmp_eq_b(lane_block, j);
                // kept: base + r in the ratio lanes of the block; dropped: the slot of position 0
                const int64 value = kept ? base + lane_in_block : (int64)origin;
                slot_vec = v_i32_mov_vb(value, 0, slot_vec, group, 0);
                valid_vec = v_i32_mov_vb((int64)(kept | no_selection), 0, valid_vec, group, 0);
            }
            const int5 out = {first << ratio_log2, row, 0, 0, 0};
            const int lanes_used = count << ratio_log2;
            if (lanes_used == QSA_LANES32) {
                v_i32_st_tnsr(out, slots, slot_vec);
                v_i32_st_tnsr(out, valid, valid_vec);
            } else {
                v_i32_st_tnsr_partial(out, slots, slot_vec, lanes_used - 1, 0);
                v_i32_st_tnsr_partial(out, valid, valid_vec, lanes_used - 1, 0);
            }
        }

        // The tail of the incomplete current block: ratio - 1 columns after the blocks.
        const int start = ((p + 1) >> ratio_log2) << ratio_log2;
        int64 tail_slots = origin;
        int64 tail_valid = no_selection;
        for (int j = 0; j < ratio - 1; ++j) {
            const int q = start + j;
            const int kept = q <= p;
            const int pos = kept ? q : 0;
            const int slot = (qsa_page(block_table, pos >> block_log2, columns, row) << block_log2) + (pos & block_mask);
            const bool64 lane = v_i32_cmp_eq_b(lanes, j);
            tail_slots = v_i32_mov_vb((int64)slot, 0, tail_slots, lane, 0);
            tail_valid = v_i32_mov_vb((int64)(kept | no_selection), 0, tail_valid, lane, 0);
        }
        if (ratio > 1) {
            const int5 out = {topk << ratio_log2, row, 0, 0, 0};
            v_i32_st_tnsr_partial(out, slots, tail_slots, ratio - 2, 0);
            v_i32_st_tnsr_partial(out, valid, tail_valid, ratio - 2, 0);
        }
    }
}
