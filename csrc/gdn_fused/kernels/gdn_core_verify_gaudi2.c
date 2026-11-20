// SPDX-License-Identifier: Apache-2.0
// Fused GDN verification core for Gaudi2: one T-node (T = 1 + k) speculative verification step
// per layer per request (docs/design/flash-next/gdn-core-decode.md, "Phase 2 contract, version 2").
// Runs the width-4 causal conv + SiLU over the T nodes of the Q, K and V channel blocks, the Q/K
// L2 normalization, the decay/beta gates, the shifted conv-slot store, and calls the recurrence
// helper (gdn_core_verify_recurrence.h) and the gated-RMSNorm helper (gdn_core_decode_norm.h)
// for each value head.
//
// One index-space point is one key head of one request (dim0 = key head, dim1 = request) over all
// T nodes. The point owns the conv channels of its Q block, its K block and the head_repeat V
// blocks of its value heads over all R conv rows, so no two points ever touch the same channel,
// state row, or output slice. Node t of request r is token r * T + t of mixed_qkv, z, a, b and the
// output; nodes at or past the row's query length are masked: zero output, zero conv input rows,
// no SSM store and no state change. Padded rows write zeros through gdn_core_zero_store and touch
// no state.
//
// Ordering: tpc-clang treats ld_tnsr/st_tnsr on one tensor as independent and reorders them
// (RUK-2829). The store slot's rows 0..R-1 overlap the history rows offset..offset+2 the same
// point reads when load slot == store slot (the normal case), so every conv store goes through
// conv_guarded_store, which makes it data-depend on all three history loads h0, h1 and h2.
//
// Vector local memory: this kernel calls lookup-table functions (v_sigmoid_f32, v_rsqrt_f32,
// v_exp_cephes_f32, v_log_f32, v_f32_reduce_add), and a kernel that does has 16 KiB (16384 bytes)
// of vector local memory for its __local__ vectors plus the compiler's vector spills. It must
// compile with the default flags, exactly as the decode kernel. Never add -vlm: it only raises the
// compiler's limit. Built with -vlm 32 (28928 bytes used) this kernel passed the simulator and
// segfaulted on a card (RUK-3105). `tpc-clang -reg-mem-count` prints the total: 12544 bytes with
// tpc-clang 2.7.33, of which the arrays below are 36 vectors x 256 bytes = 9216 bytes (6 vectors
// per node) and the rest is 13 spilled vectors; 3840 bytes stay free.
float64_pair_t bf16_to_f32_linear(bfloat128 input)
{
    // The native all-lanes conversion preserves Gaudi dual-group ordering,
    // while the recurrent state rows are linear in K. Repack the two 64-wide
    // FP32 vectors into logical [0:64] and [64:128] order.
    bfloat128_pair_t unpacked;
    unpacked.v1 = v_bf16_unpack_b(
        input,
        ((e_group_0) << 8) | ((e_every_second_element) << 9) |
            ((e_lower_half_group) << 10),
        unpacked.v1);
    unpacked.v2 = v_bf16_unpack_b(
        input,
        ((e_group_1) << 8) | ((e_every_second_element) << 9) |
            ((e_lower_half_group) << 10),
        unpacked.v2);

    const bfloat128 first_groups = unpacked.v1;
    unpacked.v1 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 0, 1, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 1, 2, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 1, 3, MkWr(1, 1), unpacked.v1);

    unpacked.v2 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 2, 0, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(
        unpacked.v2, 0xFFFFFFFF, 2, 1, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(
        first_groups, 0xFFFFFFFF, 3, 2, MkWr(1, 1), unpacked.v2);

    const float64_pair_t first = v_convert_bf16_to_f32_all_b(unpacked.v1);
    const float64_pair_t second = v_convert_bf16_to_f32_all_b(unpacked.v2);
    float64_pair_t result;
    result.v1 = first.v1;
    result.v2 = second.v1;
    return result;
}

// T <= 6 (T = 1 + k with k = 5 draft tokens). The same number is kGdnCoreVerifyMaxNodes in
// gdn_core_decode_params.h; the host glue rejects a larger T, so it never reaches this kernel.
#define GDN_VERIFY_MAX_NODES 6
#if GDN_VERIFY_MAX_NODES > 8
#error "decay and beta of node t are lane t of two registers, and the shuffle that reads them takes lanes 0 to 7"
#endif
__local__ float64 verify_q[2 * GDN_VERIFY_MAX_NODES];   // node t: [2t] = elements 0..63, [2t+1] = 64..127
__local__ float64 verify_k[2 * GDN_VERIFY_MAX_NODES];
// conv_verify_block's result for one 128-channel block: the Q block and the K block before their
// normalization, then v of the current value head. The recurrence header reads v from it and
// replaces it element by element with o, so after the recurrence it holds o of that value head.
__local__ float64 verify_vo[2 * GDN_VERIFY_MAX_NODES];
__local__ int verify_store_slot[GDN_VERIFY_MAX_NODES];  // ssm_store_indices[r, t] if t < n and in [0, S), else -1
// decay and beta of the current value head are not arrays: main packs them into two registers, node t
// in lane t, and passes them to the recurrence header (12 vectors less than one broadcast vector per
// node and gate, which at 6 nodes would leave 1280 bytes free).
#include "gdn_core_verify_recurrence.h"
#include "gdn_core_decode_norm.h"

// Every conv-state store goes through this helper. tpc-clang reorders ld_tnsr/st_tnsr on one
// tensor (RUK-2829) and, when load slot == store slot, the store slot's rows 0..R-1 overlap the
// history rows offset..offset+2 this point reads, so every store is made to data-depend on all
// three history loads: v_bf16_mov_vb(h0, 0, value, n0, 0) returns `value` bit for bit (n0 is
// never true) but ties the stored data to h0. Phase 1's guard (RUK-2832), extended to all three.
static inline void conv_guarded_store(tensor conv_state, int channel, int row, int slot, bfloat128 value,
                                      bfloat128 h0, bfloat128 h1, bfloat128 h2, bool128 n0, bool128 n1, bool128 n2)
{
    value = v_bf16_mov_vb(h0, 0, value, n0, 0);
    value = v_bf16_mov_vb(h1, 0, value, n1, 0);
    value = v_bf16_mov_vb(h2, 0, value, n2, 0);
    int5 coords = {channel, row, slot, 0, 0};
    v_bf16_st_tnsr(coords, conv_state, value);
}

// Causal width-4 conv + SiLU, fp32, for the 128 channels starting at `channel` over the `nodes`
// nodes of one request. The 3-row history is read from rows offset..offset+2 of load_slot before
// any store; store_slot receives the shifted history (h1, h2, the T node inputs, zeros to the end)
// through conv_guarded_store. The SiLU output of node t lands in verify_vo[2t] (elements 0..63)
// and verify_vo[2t + 1] (64..127); nodes at or past the row's real length `n` take a zero input
// vector, are not convolved, and store zeros.
static inline void conv_verify_block(tensor conv_state, tensor mixed_qkv, tensor conv_weight_t,
                                     int channel, int request, int nodes, int n,
                                     int load_slot, int offset, int store_slot, int conv_rows)
{
    int5 hist_coords = {channel, offset, load_slot, 0, 0};
    const bfloat128 h0 = v_bf16_ld_tnsr_b(hist_coords, conv_state);
    hist_coords[1] = offset + 1;
    const bfloat128 h1 = v_bf16_ld_tnsr_b(hist_coords, conv_state);
    hist_coords[1] = offset + 2;
    const bfloat128 h2 = v_bf16_ld_tnsr_b(hist_coords, conv_state);

    int5 weight_coords = {channel, 0, 0, 0, 0};
    const float64_pair_t w0 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 1;
    const float64_pair_t w1 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 2;
    const float64_pair_t w2 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));
    weight_coords[1] = 3;
    const float64_pair_t w3 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(weight_coords, conv_weight_t));

    // Guard predicates on all three history loads (see conv_guarded_store).
    const bool128 n0 = v_bf16_cmp_grt_b(h0, (bf16)(1.0f / 0.0f));   // never true
    const bool128 n1 = v_bf16_cmp_grt_b(h1, (bf16)(1.0f / 0.0f));
    const bool128 n2 = v_bf16_cmp_grt_b(h2, (bf16)(1.0f / 0.0f));
    const bfloat128 zero_row = (bf16)0.0f;

    bfloat128 p0 = h0;
    bfloat128 p1 = h1;
    bfloat128 p2 = h2;
    for (int t = 0; t < nodes; ++t) {
        bfloat128 x = zero_row;
        if (t < n) {
            int5 input_coords = {channel, request * nodes + t, 0, 0, 0};
            x = v_bf16_ld_tnsr_b(input_coords, mixed_qkv);

            float64_pair_t term = bf16_to_f32_linear(p0);
            float64_pair_t conv;
            conv.v1 = term.v1 * w0.v1;
            conv.v2 = term.v2 * w0.v2;
            term = bf16_to_f32_linear(p1);
            conv.v1 = v_f32_mac_b(term.v1, w1.v1, conv.v1);
            conv.v2 = v_f32_mac_b(term.v2, w1.v2, conv.v2);
            term = bf16_to_f32_linear(p2);
            conv.v1 = v_f32_mac_b(term.v1, w2.v1, conv.v1);
            conv.v2 = v_f32_mac_b(term.v2, w2.v2, conv.v2);
            term = bf16_to_f32_linear(x);
            conv.v1 = v_f32_mac_b(term.v1, w3.v1, conv.v1);
            conv.v2 = v_f32_mac_b(term.v2, w3.v2, conv.v2);

            verify_vo[2 * t] = conv.v1;   // SiLU is applied after the loop
            verify_vo[2 * t + 1] = conv.v2;
        }
        conv_guarded_store(conv_state, channel, 2 + t, store_slot, x, h0, h1, h2, n0, n1, n2);
        p0 = p1;
        p1 = p2;
        p2 = x;
    }

    conv_guarded_store(conv_state, channel, 0, store_slot, h1, h0, h1, h2, n0, n1, n2);
    conv_guarded_store(conv_state, channel, 1, store_slot, h2, h0, h1, h2, n0, n1, n2);
    for (int row = 2 + nodes; row < conv_rows; ++row) {
        conv_guarded_store(conv_state, channel, row, store_slot, zero_row, h0, h1, h2, n0, n1, n2);
    }

    // SiLU in place, x * sigmoid(x) on the same values as before. It runs here, after the weights and
    // the history rows are dead, because the lookup-table sigmoid needs many registers: inside the
    // loop above it costs 6 more spilled vectors of the 16 KiB.
    for (int i = 0; i < 2 * n; ++i) {
        const float64 pre = verify_vo[i];
        verify_vo[i] = pre * v_sigmoid_f32(pre);
    }
}

void main(tensor conv_state, tensor ssm_state, tensor mixed_qkv, tensor z, tensor a, tensor b,
          tensor A_log, tensor dt_bias, tensor conv_weight_t, tensor norm_weight,
          tensor row_indices, tensor ssm_store_indices, tensor output,
          float scale, float eps, int gate_activation)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const uint64 lanes = read_lane_id_4b_b();
    const uint64 group_lanes = lanes & 15;   // the lane number inside its 16-lane dual group
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int head_dim = 128;
    const int nodes = get_dim_size(ssm_store_indices, 0);
    const int conv_rows = get_dim_size(conv_state, 1);
    const int value_heads = get_dim_size(ssm_state, 2);
    const int state_slots = get_dim_size(ssm_state, 3);
    const int conv_slots = get_dim_size(conv_state, 2);
    const int channels = get_dim_size(mixed_qkv, 0);
    const int key_heads = (channels - value_heads * head_dim) / (2 * head_dim);
    const int head_repeat = value_heads / key_heads;
    const int key_offset = key_heads * head_dim;
    const int value_offset = 2 * key_offset;

    for (int request = index_start[1]; request < index_end[1]; ++request) {
        int5 field_coords = {0, request, 0, 0, 0};
        const int conv_load = s_i32_ld_g(gen_addr(field_coords, row_indices));
        field_coords[0] = 1;
        const int offset = s_i32_ld_g(gen_addr(field_coords, row_indices));
        field_coords[0] = 2;
        const int conv_store = s_i32_ld_g(gen_addr(field_coords, row_indices));
        field_coords[0] = 3;
        const int ssm_load = s_i32_ld_g(gen_addr(field_coords, row_indices));
        field_coords[0] = 4;
        const int query_len = s_i32_ld_g(gen_addr(field_coords, row_indices));
        // nodes > GDN_VERIFY_MAX_NODES cannot come through the host glue; if it ever did, the row is
        // treated as padding, so that no node index runs past the __local__ arrays.
        const bool real = nodes <= GDN_VERIFY_MAX_NODES &&
                          conv_load >= 0 && conv_load < conv_slots &&
                          conv_store >= 0 && conv_store < conv_slots &&
                          ssm_load >= 0 && ssm_load < state_slots &&
                          offset >= 0 && offset <= conv_rows - 3 && query_len >= 1;
        const int n = query_len < nodes ? query_len : nodes;

        for (int key_head = index_start[0]; key_head < index_end[0]; ++key_head) {
            if (!real) {
                // Padding row: zeros for every node of every owned value head, no state touched.
                for (int t = 0; t < nodes; ++t) {
                    for (int j = 0; j < head_repeat; ++j) {
                        gdn_core_zero_store(output, request * nodes + t, key_head * head_repeat + j);
                    }
                }
                continue;
            }

            for (int t = 0; t < nodes; ++t) {
                int5 slot_coords = {t, request, 0, 0, 0};
                const int s = s_i32_ld_g(gen_addr(slot_coords, ssm_store_indices));
                verify_store_slot[t] = (t < n && s >= 0 && s < state_slots) ? s : -1;
            }

            conv_verify_block(conv_state, mixed_qkv, conv_weight_t, key_head * head_dim, request,
                              nodes, n, conv_load, offset, conv_store, conv_rows);
            for (int t = 0; t < n; ++t) {
                const float64 q_lo = verify_vo[2 * t];
                const float64 q_hi = verify_vo[2 * t + 1];
                float64 q_norm = v_f32_mac_b(q_lo, q_lo, q_hi * q_hi);
                q_norm = v_f32_reduce_add(q_norm);
                q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
                q_norm = v_rsqrt_f32(q_norm + 1e-6f) * scale;
                verify_q[2 * t] = q_lo * q_norm;
                verify_q[2 * t + 1] = q_hi * q_norm;
            }

            conv_verify_block(conv_state, mixed_qkv, conv_weight_t, key_offset + key_head * head_dim, request,
                              nodes, n, conv_load, offset, conv_store, conv_rows);
            for (int t = 0; t < n; ++t) {
                const float64 k_lo = verify_vo[2 * t];
                const float64 k_hi = verify_vo[2 * t + 1];
                float64 k_norm = v_f32_mac_b(k_lo, k_lo, k_hi * k_hi);
                k_norm = v_f32_reduce_add(k_norm);
                k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
                k_norm = v_rsqrt_f32(k_norm + 1e-6f);
                verify_k[2 * t] = k_lo * k_norm;
                verify_k[2 * t + 1] = k_hi * k_norm;
            }

            for (int j = 0; j < head_repeat; ++j) {
                const int value_head = key_head * head_repeat + j;

                // v of this value head, straight into verify_vo.
                conv_verify_block(conv_state, mixed_qkv, conv_weight_t, value_offset + value_head * head_dim,
                                  request, nodes, n, conv_load, offset, conv_store, conv_rows);

                // decay and beta of this value head: node t in lane t of every 16-lane dual group.
                float64 decay_nodes = 0.0f;
                float64 beta_nodes = 0.0f;
                for (int t = 0; t < n; ++t) {
                    int5 head_coords = {value_head, 0, 0, 0, 0};
                    int5 gate_coords = {value_head, request * nodes + t, 0, 0, 0};
                    const bf16 a_bf16 = s_bf16_ld_g(gen_addr(gate_coords, a));
                    const float64 a_log = s_f32_ld_g(gen_addr(head_coords, A_log));
                    const float64 dt = s_f32_ld_g(gen_addr(head_coords, dt_bias));
                    const float64 xg = (float)a_bf16 + dt;

                    // Stock softplus: log1p(exp(x)) with the small-value branch,
                    // then the torch threshold that returns x for x > 20.
                    const float64 exponential = v_exp_cephes_f32(xg);
                    const float64 small = v_f32_mul_b(
                        exponential,
                        v_f32_mac_b(exponential, 0.5f, (float64)1.0f, SW_NEG));
                    const bool64 use_small = v_f32_cmp_less_b(exponential, 5.0e-3f);
                    const float64 regular = v_log_f32(1.0f + exponential);
                    float64 softplus = v_f32_mov_vb(
                        small, 0, regular, use_small, 0);
                    const bool64 large = v_f32_cmp_grt_b(xg, 20.0f);
                    softplus = v_f32_mov_vb(xg, 0, softplus, large, 0);

                    const float64 decay = v_exp_cephes_f32(-v_exp_cephes_f32(a_log) * softplus);
                    const bf16 b_bf16 = s_bf16_ld_g(gen_addr(gate_coords, b));
                    const float64 beta = v_sigmoid_f32((float)b_bf16);

                    const bool64 node_lane = v_u32_cmp_eq_b(group_lanes, (uint64)t);
                    decay_nodes = v_f32_mov_vb(decay, 0, decay_nodes, node_lane, 0);
                    beta_nodes = v_f32_mov_vb(beta, 0, beta_nodes, node_lane, 0);
                }

                gdn_core_verify_recurrence_head(ssm_state, ssm_load, value_head, n, decay_nodes, beta_nodes, lanes);

                for (int t = 0; t < nodes; ++t) {
                    if (t < n) {
                        float64_pair_t o;
                        o.v1 = verify_vo[2 * t];
                        o.v2 = verify_vo[2 * t + 1];
                        gdn_core_norm_store(z, norm_weight, output, request * nodes + t, value_head, o, eps,
                                            gate_activation);
                    } else {
                        gdn_core_zero_store(output, request * nodes + t, value_head);
                    }
                }
            }
        }
    }
}
