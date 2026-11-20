// SPDX-License-Identifier: Apache-2.0
// moe_glue_route: the routing of one MoE block. Replaces softmax (fp32), top-K, the sum and the
// division of the renormalization, and the casts of the torch path.
//
//   gates        {columns, tokens} bf16   the merged gate GEMM output; columns [0, experts) are
//                                         the router logits, later columns are not used
//   ids_out      {top_k, tokens} int32    the selected experts, largest logit first
//   weights_out  {top_k, tokens} bf16     their renormalized softmax weights
//
// Selection: softmax is monotonic, so the top-K probabilities belong to the top-K logits. The
// kernel selects on the bf16 logits by repeated maximum; among equal logits the lowest expert id
// wins (torch.topk leaves that choice open). Round r takes the largest (logit, -id) that comes
// after the winner of round r - 1 in that order, so no state is kept per expert and every round
// reads the row again from the tensor.
//
// Weights: w_i = exp(l_i - l_max) / sum over the K winners. The torch path divides every
// probability by the sum over all experts first and renormalizes after the top-K; that sum cancels,
// and the two results differ by a few fp32 steps before the one rounding to bf16.
//
// One index-space point is one token (dim 0 = token). top_k <= 64 (one int32 vector holds the ids);
// experts < 32768 (int16 lane ids). Lanes of the last 128-lane chunk past the experts (the
// shared-gate logit, the pad value) are never eligible.
#pragma clang fp contract(off)
#include "moe_glue_common.h"

#define MOE_GLUE_NO_EXPERT 32767

void main(tensor gates, tensor ids_out, tensor weights_out, int experts, int top_k, int gate_column, float inv_fp8_max)
{
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const short128 lanes = (short128)V_LANE_ID_16;
    const int64 lanes32 = (int64)V_LANE_ID_32;
    const int last_base = ((experts - 1) / MOE_GLUE_LANES) * MOE_GLUE_LANES;
    const float64 minus_inf_f32 = as_float64((int64)0xff800000);
    float128 minus_inf_pair;
    minus_inf_pair.v1 = minus_inf_f32;
    minus_inf_pair.v2 = minus_inf_f32;
    const bfloat128 minus_inf = v_convert_f32_to_bf16_all_b(minus_inf_pair, SW_RHNE);
    float128 plus_inf_pair;
    plus_inf_pair.v1 = as_float64((int64)0x7f800000);
    plus_inf_pair.v2 = plus_inf_pair.v1;
    const bfloat128 plus_inf = v_convert_f32_to_bf16_all_b(plus_inf_pair, SW_RHNE);

    for (int token = index_start[0]; token < index_end[0]; ++token) {
        bfloat128 previous = plus_inf;
        short128 previous_id = (short)-1;
        bfloat128 selected = minus_inf;  // lanes [0, top_k): the winning logits; the rest stay -inf
        int64 ids = 0;

        for (int rank = 0; rank < top_k; ++rank) {
            // Per lane, the best eligible logit of the row and its expert id. The chunks go from the
            // last to the first and a later one replaces an equal earlier one, so a lane keeps its
            // lowest id; a lane with nothing eligible keeps (-inf, MOE_GLUE_NO_EXPERT).
            short128_bfloat128_pair_t best;
            best.v1 = (short)MOE_GLUE_NO_EXPERT;
            best.v2 = minus_inf;
            for (int base = last_base; base >= 0; base -= MOE_GLUE_LANES) {
                const int5 coords = {base, token, 0, 0, 0};
                const bfloat128 value = v_bf16_ld_tnsr_b(coords, gates);
                const short128 id = v_i16_add_b(lanes, (short)base);
                const bool128 after_tie = v_i16_cmp_grt_vb(id, previous_id, 0, (bool128){0},
                                                          v_bf16_cmp_eq_b(value, previous), 0);
                bool128 eligible = v_bf16_cmp_less_b(value, previous) | after_tie;
                eligible = eligible & v_i16_cmp_less_b(id, (short)experts);
                best = v_i16_sel2_geq_bf16_vb(value, best.v2, id, best.v1, 0, best, eligible, 0);
            }
            const bfloat128 top = v_broadcast_element_bf16(v_bf16_reduce_max(best.v2), 0);
            const short128 holders = v_i16_sel_eq_bf16_b(best.v2, top, best.v1, (short)MOE_GLUE_NO_EXPERT);
            const short128 winner = v_broadcast_element_i16(v_i16_reduce_min(holders), 0);

            selected = v_bf16_mov_vb(top, 0, selected, v_i16_cmp_eq_b(lanes, (short)rank), 0);
            ids = v_i32_mov_vb(v_convert_i16_to_i32_b(winner, 0), 0, ids, v_i32_cmp_eq_b(lanes32, rank), 0);
            previous = top;
            previous_id = winner;
        }

        const float128 logit = v_convert_bf16_to_f32_all_b(selected);
        const float128 peak = v_convert_bf16_to_f32_all_b(v_broadcast_element_bf16(selected, 0));
        float128 weight;
        weight.v1 = v_exp_cephes_f32(v_f32_sub_b(logit.v1, peak.v1));
        weight.v2 = v_exp_cephes_f32(v_f32_sub_b(logit.v2, peak.v2));
        const float64 sum = hc_broadcast_lane_zero(
            v_f32_add_b(v_f32_reduce_add(weight.v1), v_f32_reduce_add(weight.v2)));
        const float64 inverse = v_reciprocal_f32(sum);
        weight.v1 = v_f32_mul_b(weight.v1, inverse);
        weight.v2 = v_f32_mul_b(weight.v2, inverse);

        const int5 coords = {0, token, 0, 0, 0};
        v_i32_st_tnsr_partial(coords, ids_out, ids, top_k - 1, 0);
        v_bf16_st_tnsr_partial(coords, weights_out, v_convert_f32_to_bf16_all_b(weight, SW_RHNE), top_k - 1, 0);
    }
}
