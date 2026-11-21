// SPDX-License-Identifier: Apache-2.0
// qsa_fused_prepare: the per-row work of the sparse-QSA indexer before the scores (VLLM_HPU_QSA_FUSED=2).
// Replaces index_queries, the ring read and compress_block_keys of HpuQsaIndexer._select_decode
// (vllm_gaudi/models/qwen4_exp/qsa.py) for one decode step. It writes only its declared outputs (card rule 10: a
// custom op's stores into its inputs are lost on the card, jobs 00063 / 00088): the layer writes the step's raw
// keys into the ring and `pooled` into the compressed key cache (index_copy_) after this kernel.
//
//   key_ring       {D, S * R} bf16      the flat raw-key ring as it stands before this step (read only)
//   qk             {(H + 1) * D, T}     index_qk_proj output: H query heads, then the raw key
//   rows           {4, T} int32         per row: position p, attention slot sm, ring slot read ls, ring slot
//                                       written ss (ls = ss = -1 on a padding row)
//   q_norm_weight  {D} bf16, k_norm_weight {D} bf16
//   cos_sin        {2 * R2, P} bf16     the attention's cos_sin_cache: R2 cosines, then R2 sines
//   q_idx          {H * D, T} bf16      out: the indexer queries after gemma_rms_norm and RoPE
//   pooled         {D, T} bf16          out: the compressed key of a row that completes a block, zeros elsewhere
//
// For every row: q_idx = rope(norm(q_h), cos_sin[max(p, 0)]) for each head h. A real row (0 <= ss < S, p >= 0)
// with p mod ratio == ratio - 1 and sm >= 0 completes a block: its ratio raw keys (ring rows
// ls * R + (p - ratio + 1 + j) mod R for j < ratio - 1 as they stand after this step's ring writes, then the row's
// own raw key) are pooled in fp32 (sequential sum, times 1 / ratio, one rounding to bf16), normed with
// k_norm_weight and rotated at max(p - ratio + 1, 0) into pooled[row]; every other row's pooled is zero.
// Write-then-read: the ring rows this step writes (ss * R + p mod R of every real row) are read from the writing
// row's qk instead (the last such row), so the result is what the torch path reads after its ring write.
// One index-space point is one row.
#pragma clang fp contract(off)
#include "qsa_fused_common.h"

static inline int qsa_mod(int n, int m, float inv_m)
{
    // n >= 0, m >= 1, n < 2^24: the float quotient is off by at most one, the two corrections fix it.
    int q = (int)((float)n * inv_m);
    int r = n - q * m;
    if (r < 0) r += m;
    if (r >= m) r -= m;
    return r;
}

// The dim values at `column` of a row, lanes >= dim zero. A partial load below 128 lanes: no lane is read
// past the values the kernel needs (the end of a weight vector, of a cos_sin row).
static inline bfloat128 qsa_load_head(tensor source, int column, int row, int dim)
{
    const int5 coords = {column, row, 0, 0, 0};
    const bfloat128 value =
        dim == QSA_BF16_LANES ? v_bf16_ld_tnsr_b(coords, source) : v_bf16_ld_tnsr_partial_b(coords, source, dim - 1, 0);
    return qsa_bf16_select_lanes_below(dim, value, (bfloat128)0);
}

static inline void qsa_store_head(tensor target, int column, int row, int dim, bfloat128 value)
{
    if (dim == QSA_BF16_LANES) {
        v_bf16_st_tnsr((int5){column, row, 0, 0, 0}, target, value);
    } else {
        v_bf16_st_tnsr_partial((int5){column, row, 0, 0, 0}, target, value, dim - 1, 0);
    }
}

void main(tensor key_ring, tensor qk, tensor rows, tensor q_norm_weight, tensor k_norm_weight, tensor cos_sin,
          tensor q_idx, tensor pooled,
          float eps, float inv_dim, float inv_ring, float inv_ratio, int heads, int ring_rows, int ratio, int r2)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int dim = get_dim_size(key_ring, 0);
    const int slots = get_dim_size(key_ring, 1) / ring_rows;
    const int positions = get_dim_size(cos_sin, 1);
    const int total_rows = get_dim_size(rows, 1);
    const int key_column = heads * dim;
    const bfloat128 q_weight = qsa_load_head(q_norm_weight, 0, 0, dim);
    const bfloat128 k_weight = qsa_load_head(k_norm_weight, 0, 0, dim);

    for (int row = index_start[0]; row < index_end[0]; ++row) {
        const int p = s_i32_ld_g(gen_addr((int5){0, row, 0, 0, 0}, rows));
        const int sm = s_i32_ld_g(gen_addr((int5){1, row, 0, 0, 0}, rows));
        const int ls = s_i32_ld_g(gen_addr((int5){2, row, 0, 0, 0}, rows));
        const int ss = s_i32_ld_g(gen_addr((int5){3, row, 0, 0, 0}, rows));
        int rope_row = p < 0 ? 0 : p;
        rope_row = rope_row >= positions ? positions - 1 : rope_row;
        const bfloat128 cs = qsa_load_head(cos_sin, 0, rope_row, 2 * r2);

        for (int h = 0; h < heads; ++h) {
            const bfloat128 x = qsa_load_head(qk, h * dim, row, dim);
            const bfloat128 y = qsa_gemma_norm(x, q_weight, inv_dim, eps);
            qsa_store_head(q_idx, h * dim, row, dim, qsa_rope(y, cs, r2));
        }

        // A padding row, or a row that completes no block: a zero pooled row.
        if (ss < 0 || ss >= slots || p < 0 || (p & (ratio - 1)) != ratio - 1 || sm < 0) {
            qsa_store_head(pooled, 0, row, dim, (bfloat128)0);
            continue;
        }
        const bfloat128 raw = qsa_load_head(qk, key_column, row, dim);

        // The block's ratio raw keys, summed in order in fp32.
        float128 sum = v_convert_bf16_to_f32_all_b((bfloat128)0);
        for (int j = 0; j < ratio - 1; ++j) {
            const int source = ls * ring_rows + qsa_mod(p - (ratio - 1) + j, ring_rows, inv_ring);
            // Write-then-read: the last real row of this call that writes ring row `source`.
            int writer = -1;
            for (int u = 0; u < total_rows; ++u) {
                const int su = s_i32_ld_g(gen_addr((int5){3, u, 0, 0, 0}, rows));
                const int pu = s_i32_ld_g(gen_addr((int5){0, u, 0, 0, 0}, rows));
                if (su >= 0 && su < slots && pu >= 0 && su * ring_rows + qsa_mod(pu, ring_rows, inv_ring) == source) {
                    writer = u;
                }
            }
            bfloat128 key;
            if (writer >= 0) {
                key = qsa_load_head(qk, key_column, writer, dim);
            } else {
                key = qsa_load_head(key_ring, 0, source, dim);
            }
            const float128 kf = v_convert_bf16_to_f32_all_b(key);
            sum.v1 = v_f32_add_b(sum.v1, kf.v1);
            sum.v2 = v_f32_add_b(sum.v2, kf.v2);
        }
        const float128 own = v_convert_bf16_to_f32_all_b(raw);
        sum.v1 = v_f32_mul_b(v_f32_add_b(sum.v1, own.v1), inv_ratio);
        sum.v2 = v_f32_mul_b(v_f32_add_b(sum.v2, own.v2), inv_ratio);
        const bfloat128 block_key = v_convert_f32_to_bf16_all_b(sum, SW_RHNE);
        const bfloat128 normed = qsa_gemma_norm(block_key, k_weight, inv_dim, eps);
        int key_row = p - (ratio - 1);
        key_row = key_row < 0 ? 0 : (key_row >= positions ? positions - 1 : key_row);
        const bfloat128 key_cs = qsa_load_head(cos_sin, 0, key_row, 2 * r2);
        qsa_store_head(pooled, 0, row, dim, qsa_rope(normed, key_cs, r2));
    }
}
