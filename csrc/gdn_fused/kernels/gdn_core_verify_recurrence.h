// SPDX-License-Identifier: Apache-2.0
#pragma once
// GDN delta-rule recurrence of one value head over the `nodes` real nodes of one verification row
// (docs/design/flash-next/gdn-core-decode.md, "Phase 2 contract, version 2").
//
// Inputs, in the __local__ arrays that gdn_core_verify_gaudi2.c declares before including this file
// (node t's 128-wide vectors in linear lane order: [2t] = elements 0..63, [2t + 1] = elements 64..127):
//   verify_q[2t], verify_q[2t + 1]   q of node t, L2-normalized and multiplied by scale
//   verify_k[2t], verify_k[2t + 1]   k of node t, L2-normalized
//   verify_v[2t], verify_v[2t + 1]   v of node t for this value head
//   verify_decay[t], verify_beta[t]  this value head's decay and beta of node t (the same value in all 64 lanes)
//   verify_store_slot[t]             SSM slot node t's state is stored to, or -1 for "not stored"
// Output: verify_o[2t], verify_o[2t + 1] = o of node t (linear lanes), for t < nodes.
//
// Reads the head's 128x128 fp32 state from load_slot once, row by row; each value row is carried through
// nodes 0 .. nodes - 1 in registers and stored after node t wherever verify_store_slot[t] >= 0.
//
// STUB (contract commit): the body below only zeroes verify_o so that gdn_core_verify_gaudi2.c compiles.
// Task KV2 replaces the body; the signature and the array contract above do not change.
static inline void gdn_core_verify_recurrence_head(
    tensor ssm_state, int load_slot, int value_head, int nodes, uint64 lanes)
{
    for (int t = 0; t < nodes; ++t) {
        verify_o[2 * t] = 0.0f;
        verify_o[2 * t + 1] = 0.0f;
    }
}
