// SPDX-License-Identifier: Apache-2.0
// qsa_fused_topk: the top-k columns of every row of the masked sparse-QSA block scores in ONE node
// (VLLM_HPU_QSA_TOPK_FORM=4). It replaces torch.topk's sort of the whole row: the order of the picks is not formed
// here (ascending columns; the caller orders the k picks by value as forms 1-3 do).
//
//   masked     {N, T} f32          the block scores of each row, -inf where a block is not visible (select_blocks)
//   ids        {k, T} i32          out: the selected columns, ascending, then -1 when a row has fewer than k columns
//                                  that are not masked
//   workspace  {4 + 2 NW, T} i32   out (scratch, read back by this kernel): [0] the threshold key theta, [3] 1 when
//                                  ties at the cut overflow (more columns equal theta than the k - G still free),
//                                  else 0. Without overflow: [4 + w] bit b: column 32 w + b has key >= theta. With it:
//                                  [1] G = #{key > theta}, [2] #{key == theta}, [4 + w] key > theta, [4 + NW + w]
//                                  key == theta (NW = N / 32). Words: the lane packing of qsa_fused_threshold.
//
// The set is qsa_sparse_ops.threshold_pick's: key(n) = the fp32 bits of masked[n] when it is >= +0 (NaN above +inf),
// 0 for -0, and "no key" for -inf (or any other negative value: never selected). theta = the largest key with
// #{key >= theta} >= k; every column with a key above theta is taken, then the lowest columns with key == theta
// until k are taken; a row with fewer than k keys takes all of them and pads with -1.
//
// One index-space point is one row; N <= 32768 and a multiple of 512 (the glue checks). Per row:
//   1. one read of the row: the top 16 bits of each key (key >> 15; 0 for a masked column) into vector local memory,
//      two per 32-bit lane (64 KiB at N = 32768; the order of the values there does not matter, only their counts);
//   2. 16 passes over local memory find theta16, the largest prefix with #{prefix >= theta16} >= k (counts are
//      16-bit lane sums of a predicated add, four independent accumulators);
//   3. a second read of the row: G16 = #{prefix > theta16} and, for the columns whose prefix is theta16 (the bucket),
//      low 15 bits + 1 (0 for every other column) into local memory over the prefixes;
//   4. 16 passes find L, the largest value with #{value >= L} >= k - G16 in the bucket: theta = theta16 << 15 | (L - 1)
//      (theta = 0 when the row has fewer than k keys: the bucket is then smaller than k - G16 and L stays 0);
//   5. theta and the overflow flag go to the workspace and, after an aso fence, the flag is read back as a scalar: a
//      third read of the row writes one bitmap, key >= theta (no ties beyond k: every such column is taken), or with
//      overflow (ties at the cut, e.g. zero scores) the bitmaps key > theta and key == theta and G;
//   6. after an aso fence, scalar code walks the bitmap words (four per load batch) and stores the ids one by one:
//      the columns of the one bitmap, or every column above theta and then the lowest k - G columns equal to it
//      (ascending within each walk); then -1 up to k.
// No lookup-table function is called (the lane sums are integer shuffles), so the 64 KiB of local memory are free.
// Every threshold and count lives in vector registers (every lane holds it); only the walk of step 6 is scalar.
#pragma clang fp contract(off)
#include "qsa_fused_common.h"

#define QSA_TOPK_MAX_COLUMNS 32768
#define QSA_TOPK_LOCAL_VECTORS (QSA_TOPK_MAX_COLUMNS / 128)

__local__ uint64 qsa_topk_values[QSA_TOPK_LOCAL_VECTORS];  // two 16-bit values per 32-bit lane

// key(n) of 64 columns: the bits of a value >= +0, 0 for -0, negative (no key) for -inf.
static inline int64 qsa_topk_key(float64 value)
{
    const int64 bits = as_int64(value);
    return v_i32_sel_eq_i32_b(bits, (int)0x80000000, (int64)0, bits);
}

// The number of the 16-bit values in local memory (2 x 64 per vector, `vectors` vectors) that are >= trial, in
// every lane. trial: the same value in every 32-bit lane, <= 0xffff.
static inline int64 qsa_topk_count_at_least(int vectors, uint64 trial)
{
    const uint64 broadcast = trial | (trial << 16);
    const ushort128 threshold = *((const ushort128*)&broadcast);
    ushort128 count0 = 0;
    ushort128 count1 = 0;
    ushort128 count2 = 0;
    ushort128 count3 = 0;
    // 16 vectors per iteration: tpc-clang schedules one basic block at a time, so the loads and the four
    // independent count chains of one iteration overlap; the loop needs vectors % 16 == 0 (N % 2048 == 0) or runs
    // the tail below
    int v = 0;
    for (; v + 16 <= vectors; v += 16) {
#pragma unroll 4
        for (int u = 0; u < 16; u += 4) {
            const uint64 w0 = qsa_topk_values[v + u];
            const uint64 w1 = qsa_topk_values[v + u + 1];
            const uint64 w2 = qsa_topk_values[v + u + 2];
            const uint64 w3 = qsa_topk_values[v + u + 3];
            count0 = v_u16_add_vb(count0, 1, 0, count0, v_u16_cmp_geq_b(*((const ushort128*)&w0), threshold), 0);
            count1 = v_u16_add_vb(count1, 1, 0, count1, v_u16_cmp_geq_b(*((const ushort128*)&w1), threshold), 0);
            count2 = v_u16_add_vb(count2, 1, 0, count2, v_u16_cmp_geq_b(*((const ushort128*)&w2), threshold), 0);
            count3 = v_u16_add_vb(count3, 1, 0, count3, v_u16_cmp_geq_b(*((const ushort128*)&w3), threshold), 0);
        }
    }
    for (; v < vectors; v += 4) {
        const uint64 w0 = qsa_topk_values[v];
        const uint64 w1 = qsa_topk_values[v + 1];
        const uint64 w2 = qsa_topk_values[v + 2];
        const uint64 w3 = qsa_topk_values[v + 3];
        count0 = v_u16_add_vb(count0, 1, 0, count0, v_u16_cmp_geq_b(*((const ushort128*)&w0), threshold), 0);
        count1 = v_u16_add_vb(count1, 1, 0, count1, v_u16_cmp_geq_b(*((const ushort128*)&w1), threshold), 0);
        count2 = v_u16_add_vb(count2, 1, 0, count2, v_u16_cmp_geq_b(*((const ushort128*)&w2), threshold), 0);
        count3 = v_u16_add_vb(count3, 1, 0, count3, v_u16_cmp_geq_b(*((const ushort128*)&w3), threshold), 0);
    }
    // at most vectors / 4 <= 64 per 16-bit lane and accumulator: the sums below stay inside 16 bits
    const ushort128 sum = (count0 + count1) + (count2 + count3);
    const uint64 pairs = *((const uint64*)&sum);
    return qsa_sum_broadcast(as_int64((pairs & 0xffff) + (pairs >> 16)));
}

// found = the largest t in 1 .. 2^16 - 1 with #{value >= t} >= need, found bit by bit from bit 15, 0 when there is
// none; found_count = #{value >= found} (0 when found is 0). need, found and found_count: the same in every lane.
#define QSA_TOPK_SEARCH(found, found_count, vectors, need)                                                        \
    do {                                                                                                          \
        found = 0;                                                                                                \
        found_count = 0;                                                                                          \
        for (int bit_ = 15; bit_ >= 0; --bit_) {                                                                  \
            const uint64 trial_ = found | (uint64)(1u << bit_);                                                   \
            const int64 count_ = qsa_topk_count_at_least((vectors), trial_);                                      \
            found = v_u32_sel_geq_i32_b(count_, (need), trial_, found);                                           \
            found_count = v_i32_sel_geq_i32_b(count_, (need), count_, found_count);                               \
        }                                                                                                         \
    } while (0)

// The bitmap words of 128 columns: a for columns 0..63, b for 64..127 of a tile; lanes 0..3 hold the words of
// columns 0..31, 32..63, 64..95, 96..127 (bit = column % 32), the words qsa_pack_words(qsa_pack_half(a),
// qsa_pack_half(b)) gives, with one OR reduction for the two predicates instead of two: lane bit 4 folds a into the
// dual groups 0 and 2 and b into 1 and 3, then lane bit 3 and bits 0..2 are OR-ed inside each dual group (after the
// bit-3 stage both groups of a dual group are equal, so the shuffle, which reads lanes 0-7 of a dual group, reads
// the right value), and three dual-group moves gather the four words. one: 1 << (lane % 32);
// xor1/2/4: shuffle indices 0x80 | ((lane & 7) ^ 1, 2, 4).
static inline uint64 qsa_topk_pack2(bool64 a, bool64 b, uint64 one, bool64 bit4_clear, uchar256 xor1,
                                    uchar256 xor2, uchar256 xor4)
{
    const uint64 bits_b = v_u32_mov_vb(one, 0, (uint64)0, b, 0);
    const uint64 bits_a = v_u32_mov_vb(one, 0, (uint64)0, a, 0);
    uint64 v = bits_b | v_u32_mov_dual_group_all_b(bits_b, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0);
    v = v_u32_or_vb(bits_a, v_u32_mov_dual_group_all_b(bits_a, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0), 0, v,
                    bit4_clear, 0);
    v |= v_u32_mov_group_b(v, 0xffffffff, 63, 0);
    v |= v_u32_shuffle_b(v, xor1, 0, 0);
    v |= v_u32_shuffle_b(v, xor2, 0, 0);
    v |= v_u32_shuffle_b(v, xor4, 0, 0);
    // dual group 0: a's word of columns 0..31, 1: b's 64..95, 2: a's 32..63, 3: b's 96..127
    uint64 words = v_u32_mov_dual_group_b(v, 0x000000f0, 2, 0, SW_WR_LOWER_GROUP, v);
    words = v_u32_mov_dual_group_b(v, 0x00000f00, 1, 0, SW_WR_LOWER_GROUP, words);
    return v_u32_mov_dual_group_b(v, 0x0000f000, 3, 0, SW_WR_LOWER_GROUP, words);
}

// Store every set column of bitmap word `bits` (word index `w`) at out, ascending; out[0] counts the stores.
#define QSA_TOPK_EMIT_ALL(bits, w)                                                                                \
    do {                                                                                                          \
        unsigned b_ = (bits);                                                                                     \
        while (b_) {                                                                                              \
            s_i32_st_g(gen_addr(out, ids), ((w) << 5) + s_u32_find_first(b_, SW_FIND_ONE | SW_LSB));             \
            out[0] += 1;                                                                                          \
            b_ &= b_ - 1;                                                                                         \
        }                                                                                                         \
    } while (0)

// Store the 64-bit lane-0 value of an int64 vector into word `word` of the row's workspace.
static inline void qsa_topk_meta(tensor workspace, int row, int word, int64 value)
{
    v_i32_st_tnsr_partial((int5){word, row, 0, 0, 0}, workspace, value, 0, 0);
}

void main(tensor masked, tensor ids, tensor workspace, int topk)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int columns = get_dim_size(masked, 0);
    const int words = columns >> 5;
    const int vectors = columns >> 7;
    const uint64 lanes = (uint64)V_LANE_ID_32;
    const uint64 one = (uint64)1 << (lanes & 31);
    const bool64 bit4_clear = v_u32_cmp_eq_b(lanes & 16, (uint64)0);
    const uint64 pattern1 = (((lanes & 7) ^ 1) | 0x80) * 0x01010101;
    const uint64 pattern2 = (((lanes & 7) ^ 2) | 0x80) * 0x01010101;
    const uint64 pattern4 = (((lanes & 7) ^ 4) | 0x80) * 0x01010101;
    const uchar256 xor1 = *((const uchar256*)&pattern1);
    const uchar256 xor2 = *((const uchar256*)&pattern2);
    const uchar256 xor4 = *((const uchar256*)&pattern4);
    for (int row = index_start[0]; row < index_end[0]; ++row) {
        // 1. the key prefixes into local memory
#pragma unroll 4
        for (int v = 0; v < vectors; ++v) {
            const int first = v << 7;
            const int64 key0 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first, row, 0, 0, 0}, masked));
            const int64 key1 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first + 64, row, 0, 0, 0}, masked));
            const uint64 prefix0 = as_uint64(v_i32_max_b(key0, (int64)0)) >> 15;
            const uint64 prefix1 = as_uint64(v_i32_max_b(key1, (int64)0)) >> 15;
            qsa_topk_values[v] = prefix0 | (prefix1 << 16);
        }
        // 2. theta16
        const int64 k = (int64)topk;
        uint64 theta16;
        int64 count16;
        QSA_TOPK_SEARCH(theta16, count16, vectors, k);
        // 3. G16 and the bucket's low bits + 1
        int64 above0 = 0;
        int64 above1 = 0;
#pragma unroll 4
        for (int v = 0; v < vectors; ++v) {
            const int first = v << 7;
            const int64 key0 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first, row, 0, 0, 0}, masked));
            const int64 key1 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first + 64, row, 0, 0, 0}, masked));
            const int64 prefix0 = key0 >> 15;  // negative for a masked column
            const int64 prefix1 = key1 >> 15;
            above0 = v_i32_add_vb(above0, 1, 0, above0, v_i32_cmp_grt_b(prefix0, as_int64(theta16)), 0);
            above1 = v_i32_add_vb(above1, 1, 0, above1, v_i32_cmp_grt_b(prefix1, as_int64(theta16)), 0);
            const bool64 bucket0 = v_i32_cmp_eq_b(prefix0, as_int64(theta16));
            const bool64 bucket1 = v_i32_cmp_eq_b(prefix1, as_int64(theta16));
            const uint64 low0 = v_u32_mov_vb(as_uint64(key0 & 0x7fff) + 1, 0, (uint64)0, bucket0, 0);
            const uint64 low1 = v_u32_mov_vb(as_uint64(key1 & 0x7fff) + 1, 0, (uint64)0, bucket1, 0);
            qsa_topk_values[v] = low0 | (low1 << 16);
        }
        const int64 need = k - qsa_sum_broadcast(above0 + above1);
        // 4. theta; ties at the cut overflow when more bucket columns reach the low threshold than are needed
        uint64 low;
        int64 low_count;
        QSA_TOPK_SEARCH(low, low_count, vectors, need);
        const uint64 theta = v_u32_sel_geq_i32_b(as_int64(low), 1, (theta16 << 15) | (low - 1), (uint64)0);
        qsa_topk_meta(workspace, row, 0, as_int64(theta));
        qsa_topk_meta(workspace, row, 3, v_i32_sel_grt_i32_b(low_count, need, (int64)1, (int64)0));
        aso(SW_INC | SW_VPU);
        aso(SW_DEC | SW_SPU);
        const int overflow = s_i32_ld_g(gen_addr((int5){3, row, 0, 0, 0}, workspace));
        // 5. the bitmaps: without overflow one bitmap of key >= theta (every such column is taken); with it, the
        //    bitmaps of key > theta and key == theta and G
        if (!overflow) {
#pragma unroll 4
            for (int v = 0; v < vectors; ++v) {
                const int first = v << 7;
                const int64 key0 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first, row, 0, 0, 0}, masked));
                const int64 key1 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first + 64, row, 0, 0, 0}, masked));
                v_u32_st_tnsr_partial((int5){4 + (v << 2), row, 0, 0, 0}, workspace,
                                      qsa_topk_pack2(v_i32_cmp_geq_b(key0, as_int64(theta)),
                                                     v_i32_cmp_geq_b(key1, as_int64(theta)), one, bit4_clear, xor1,
                                                     xor2, xor4), 3, 0);
            }
        } else {
            int64 greater = 0;
            int64 equal = 0;
#pragma unroll 2
            for (int v = 0; v < vectors; ++v) {
                const int first = v << 7;
                const int64 key0 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first, row, 0, 0, 0}, masked));
                const int64 key1 = qsa_topk_key(v_f32_ld_tnsr_b((int5){first + 64, row, 0, 0, 0}, masked));
                const bool64 gt0 = v_i32_cmp_grt_b(key0, as_int64(theta));
                const bool64 gt1 = v_i32_cmp_grt_b(key1, as_int64(theta));
                const bool64 eq0 = v_i32_cmp_eq_b(key0, as_int64(theta));
                const bool64 eq1 = v_i32_cmp_eq_b(key1, as_int64(theta));
                greater = v_i32_add_vb(greater, 1, 0, greater, gt0, 0);
                greater = v_i32_add_vb(greater, 1, 0, greater, gt1, 0);
                equal = v_i32_add_vb(equal, 1, 0, equal, eq0, 0);
                equal = v_i32_add_vb(equal, 1, 0, equal, eq1, 0);
                v_u32_st_tnsr_partial((int5){4 + (v << 2), row, 0, 0, 0}, workspace,
                                      qsa_topk_pack2(gt0, gt1, one, bit4_clear, xor1, xor2, xor4), 3, 0);
                v_u32_st_tnsr_partial((int5){4 + words + (v << 2), row, 0, 0, 0}, workspace,
                                      qsa_topk_pack2(eq0, eq1, one, bit4_clear, xor1, xor2, xor4), 3, 0);
            }
            qsa_topk_meta(workspace, row, 1, qsa_sum_broadcast(greater));
            qsa_topk_meta(workspace, row, 2, qsa_sum_broadcast(equal));
        }
        // 6. the walk (the vector stores above must be visible to the scalar loads below)
        aso(SW_INC | SW_VPU);
        aso(SW_DEC | SW_SPU);
        // without overflow: every column of the one bitmap (at most k of them); with it, every column above theta,
        // then the lowest columns equal to it up to k. The ids are stored one by one at out (its dim 0 counts them).
        int5 out = {0, row, 0, 0, 0};
        int5 at = {4, row, 0, 0, 0};
        for (int w = 0; w < words; w += 4) {
            const unsigned b0 = s_u32_ld_g(gen_addr(at, workspace));
            at[0] += 1;
            const unsigned b1 = s_u32_ld_g(gen_addr(at, workspace));
            at[0] += 1;
            const unsigned b2 = s_u32_ld_g(gen_addr(at, workspace));
            at[0] += 1;
            const unsigned b3 = s_u32_ld_g(gen_addr(at, workspace));
            at[0] += 1;
            if ((b0 | b1 | b2 | b3) == 0) {
                continue;
            }
            QSA_TOPK_EMIT_ALL(b0, w);
            QSA_TOPK_EMIT_ALL(b1, w + 1);
            QSA_TOPK_EMIT_ALL(b2, w + 2);
            QSA_TOPK_EMIT_ALL(b3, w + 3);
        }
        if (overflow) {
            const int greater = s_i32_ld_g(gen_addr((int5){1, row, 0, 0, 0}, workspace));
            int budget = topk - greater;
            for (int w = 0; w < words && budget > 0; ++w) {
                unsigned b = s_u32_ld_g(gen_addr(at, workspace));
                at[0] += 1;
                while (b && budget > 0) {
                    s_i32_st_g(gen_addr(out, ids), (w << 5) + s_u32_find_first(b, SW_FIND_ONE | SW_LSB));
                    out[0] += 1;
                    --budget;
                    b &= b - 1;
                }
            }
        }
        const int written = out[0];
        for (int first = written; first < topk; ++first) {  // fewer than k keys in the row
            s_i32_st_g(gen_addr(out, ids), -1);
            out[0] += 1;
        }
    }
}
