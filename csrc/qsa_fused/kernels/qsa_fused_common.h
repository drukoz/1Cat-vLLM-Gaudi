// SPDX-License-Identifier: Apache-2.0
#pragma once
// Helpers shared by the fused QSA decode kernels (Gaudi2 TPC-C). Contract: docs/design/flash-next/qsa-fused.md.
//
// Tensor layout: a torch [rows, width] tensor is the TPC tensor {width, rows} (dim 0 is the fastest).
// One indexer head (D <= 128 bf16 values) is one 128-lane bf16 vector; lanes >= D are masked to 0.
//
// Numerics: every bf16 tensor op of the torch path is "fp32, then one rounding to bf16 (round half to
// even)"; the kernels keep those rounding points: products and sums are separate instructions (no fused
// multiply-add, `fp contract(off)`), and each bf16 result goes through v_convert_f32_to_bf16_all_b with
// SW_RHNE. v_convert_bf16_to_f32_all_b splits 128 bf16 lanes into two 64-lane fp32 vectors in the
// dual-group order and the conversion back undoes it; lane shuffles and lane selects happen in the bf16
// domain only, where the lane order is linear, and every element-wise fp32 expression combines operands
// that went through the same conversion.

#define QSA_BF16_LANES 128

static inline bfloat128 qsa_bf16_select_lanes_below(int limit, bfloat128 below, bfloat128 above)
{
    const ushort128 lane = read_lane_id_2b_b();
    const ushort128 a = *((ushort128*)&below);
    const ushort128 b = *((ushort128*)&above);
    const ushort128 picked = v_u16_sel_less_u16_b(lane, (unsigned short)limit, a, b);
    return *((bfloat128*)&picked);
}

// bf16(a * b) and bf16(a + b) / bf16(a - b) of two bf16 vectors, as torch rounds them.
static inline bfloat128 qsa_bf16_mul(bfloat128 a, bfloat128 b)
{
    const float128 x = v_convert_bf16_to_f32_all_b(a);
    const float128 y = v_convert_bf16_to_f32_all_b(b);
    float128 r;
    r.v1 = v_f32_mul_b(x.v1, y.v1);
    r.v2 = v_f32_mul_b(x.v2, y.v2);
    return v_convert_f32_to_bf16_all_b(r, SW_RHNE);
}

static inline bfloat128 qsa_bf16_add(bfloat128 a, bfloat128 b)
{
    const float128 x = v_convert_bf16_to_f32_all_b(a);
    const float128 y = v_convert_bf16_to_f32_all_b(b);
    float128 r;
    r.v1 = v_f32_add_b(x.v1, y.v1);
    r.v2 = v_f32_add_b(x.v2, y.v2);
    return v_convert_f32_to_bf16_all_b(r, SW_RHNE);
}

static inline bfloat128 qsa_bf16_sub(bfloat128 a, bfloat128 b)
{
    const float128 x = v_convert_bf16_to_f32_all_b(a);
    const float128 y = v_convert_bf16_to_f32_all_b(b);
    float128 r;
    r.v1 = v_f32_sub_b(x.v1, y.v1);
    r.v2 = v_f32_sub_b(x.v2, y.v2);
    return v_convert_f32_to_bf16_all_b(r, SW_RHNE);
}

// Lane 0 of `value` in every lane (v_f32_reduce_add leaves the sum in lane 0), as hc_fused_common.h.
static inline float64 qsa_broadcast_lane_zero(float64 value)
{
    const uchar256 broadcast_lane_zero = 0x80;
    return v_f32_shuffle_b(value, broadcast_lane_zero, 0, value);
}

// gemma_rms_norm of one head (lanes >= D of x are 0): bf16(x * rsqrt(sum(x^2) * inv_dim + eps) * (1 + w)).
static inline bfloat128 qsa_gemma_norm(bfloat128 x, bfloat128 w, float inv_dim, float eps)
{
    const float128 xf = v_convert_bf16_to_f32_all_b(x);
    const float128 wf = v_convert_bf16_to_f32_all_b(w);
    const float64 squares = v_f32_add_b(v_f32_mul_b(xf.v1, xf.v1), v_f32_mul_b(xf.v2, xf.v2));
    const float64 total = qsa_broadcast_lane_zero(v_f32_reduce_add(squares));
    const float64 rrms = v_rsqrt_f32(v_f32_add_b(v_f32_mul_b(total, inv_dim), eps));
    float128 y;
    y.v1 = v_f32_mul_b(v_f32_mul_b(xf.v1, rrms), v_f32_add_b(wf.v1, 1.0f));
    y.v2 = v_f32_mul_b(v_f32_mul_b(xf.v2, rrms), v_f32_add_b(wf.v2, 1.0f));
    return v_convert_f32_to_bf16_all_b(y, SW_RHNE);
}

// rope_neox_partial of one head: lanes [0, r2) and [r2, 2 r2) rotate as the pairs (j, r2 + j), lanes >= 2 r2
// pass through. cs holds the position's cos_sin row: r2 cosines, then r2 sines (lanes >= 2 r2: don't care).
//   out[j]      = bf16(bf16(y[j] * c[j]) - bf16(y[r2 + j] * s[j]))
//   out[r2 + j] = bf16(bf16(y[r2 + j] * c[j]) + bf16(y[j] * s[j]))
// v_element_shift_xor by r2 (a power of two) swaps lane j with lane r2 + j inside every 2 r2 lanes.
static inline bfloat128 qsa_rope(bfloat128 y, bfloat128 cs, int r2)
{
    const bfloat128 y_swapped = v_element_shift_xor_bf16(y, r2);
    const bfloat128 cs_swapped = v_element_shift_xor_bf16(cs, r2);
    const bfloat128 cosines = qsa_bf16_select_lanes_below(r2, cs, cs_swapped);
    const bfloat128 sines = qsa_bf16_select_lanes_below(r2, cs_swapped, cs);
    const bfloat128 first = qsa_bf16_mul(y, cosines);
    const bfloat128 second = qsa_bf16_mul(y_swapped, sines);
    const bfloat128 low = qsa_bf16_sub(first, second);
    const bfloat128 high = qsa_bf16_add(first, second);
    return qsa_bf16_select_lanes_below(r2, low, qsa_bf16_select_lanes_below(2 * r2, high, y));
}

// ---------------------------------------------------------------------------------------------------------
// Selection bitmaps (qsa_fused_threshold / qsa_fused_select): the lane packing and the lane sum of the
// card-proven DeepSeek V4.1 index kernels (csrc/deepseek_v4/include/deepseek_v41_selection_bitmap.h and
// kernels/deepseek_v41_index_threshold_gaudi2.c), copied: word j of pack_words(pack_half(a), pack_half(b)) holds
// the predicate of lanes 32 j .. 32 j + 31 of the 128 lanes a (0..63) and b (64..127), bit (lane % 32).
// ---------------------------------------------------------------------------------------------------------
static inline uint64 qsa_pack_half(bool64 predicate)
{
    const uint64 lanes = (uint64)V_LANE_ID_32;
    uint64 bits = v_u32_mov_vb((uint64)1 << (lanes & 31), 0, (uint64)0, predicate, 0);
    bits |= v_u32_mov_dual_group_all_b(bits, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0);
    bits |= v_u32_mov_group_b(bits, 0xffffffff, 63, 0);
#pragma loop_unroll(3)
    for (int shift = 4; shift > 0; shift >>= 1) {
        const uint64 mask = (((lanes & 7) ^ shift) | 0x80) * 0x01010101;
        bits |= v_u32_shuffle_b(bits, *((const uchar256*)&mask), 0, 0);
    }
    return bits;
}

static inline uint64 qsa_pack_words(uint64 first, uint64 second)
{
    uint64 words = v_u32_mov_dual_group_b(first, 0x000000f0, 2, 0, SW_WR_LOWER_GROUP, first);
    words = v_u32_mov_dual_group_b(second, 0x00000f00, 0, 0, SW_WR_LOWER_GROUP, words);
    return v_u32_mov_dual_group_b(second, 0x0000f000, 2, 0, SW_WR_LOWER_GROUP, words);
}

// The sum over the 64 lanes of `value`, in every lane (integer shuffles: no lookup table, no local memory).
static inline int64 qsa_sum_broadcast(int64 value)
{
    value += v_i32_mov_dual_group_all_b(value, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0);
    value += v_i32_mov_dual_group_all_b(value, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0);
    value += v_i32_mov_group_b(value, 0xffffffff, 63, 0);
#pragma loop_unroll(3)
    for (int shift = 4; shift > 0; shift >>= 1) {
        const uint64 mask = ((((uint64)V_LANE_ID_32 & 7) ^ shift) | 0x80) * 0x01010101;
        value += v_i32_shuffle_b(value, *((const uchar256*)&mask), 0, 0);
    }
    return value;
}

// The order key of a block score: its fp32 bits without the sign bit. Scores are relu sums (>= +0; -0 maps to +0),
// so the integer order is the float order; NaN sorts above +inf, as torch.topk puts NaN first.
static inline int64 qsa_score_key(float64 score)
{
    return as_int64(score) & 0x7fffffff;
}
