// SPDX-License-Identifier: Apache-2.0
#pragma once
// Helpers of the interleaved hyper-connection kernels (hc_fused_gate_mix_il, hc_fused_combine_norm_il):
// docs/design/flash-next/hc-kernels.md.
//
// The shipped kernels are latency-bound: tpc-clang schedules one basic block at a time, so a loop body that is
// one dependent chain pays its whole latency, and the library sigmoid (v_exp_cephes_f32 + v_reciprocal_f32,
// every special case) costs about 45 VPU operations per 64 lanes. The interleaved kernels give each basic block
// several independent chains and use the sigmoid below (25 VPU operations).
#include "hc_fused_common.h"

// sigmoid(x) = 1 / (1 + exp(-x)), evaluated as
//   e = exp(-|x|) in [0, 1] (never overflows), d = 1 + e in [1, 2] (one binade: no special case),
//   q = 1 / d by a minimax quadratic seed (relative error 1.0e-2) and two Newton steps (1e-8 before rounding),
//   sigmoid = q for x >= 0 and e * q for x < 0.
// exp(-a), a = |x|: n = round(-a * log2(e)) by the 1.5 * 2^23 trick, which also leaves n << 23 in the low bits
// of the sum; s = a + n * ln2 in the two Cephes parts (|s| <= ln2 / 2); exp(-s) by a degree-6 minimax
// polynomial in Horner form (relative error 1.9e-9 before rounding, 8.4e-8 in fp32 Horner); 2^n by an integer
// add to the exponent field. From n <= -126 (|x| >= 87.0) the result would leave the normal range and is 0:
// sigmoid is then exactly 1 (x > 0) or exactly 0 (x < 0), the values the saturated-gate cases of
// tests/unit_tests/hc_fused/hc_case.py need bit for bit. NaN propagates.
#define HC_IL_MAGIC 12582912.0f  // 1.5 * 2^23: adding it rounds a float in (-2^22, 2^22) to an integer

static inline float64 hc_il_exp_neg(float64 a)
{
    const float64 t = v_f32_mac_b(a, 1.44269504088896341f, (float64)HC_IL_MAGIC, SW_NEG);
    const float64 n = t - HC_IL_MAGIC;
    float64 s = v_f32_mac_b(n, 0.693359375f, a, 0);
    s = v_f32_mac_b(n, -2.12194440e-4f, s, 0);
    float64 p = v_f32_mac_b(s, 0.00138367828913033f, (float64)-0.008374853990972042f, 0);
    p = v_f32_mac_b(p, s, (float64)0.041668228805065155f, 0);
    p = v_f32_mac_b(p, s, (float64)-0.16666419804096222f, 0);
    p = v_f32_mac_b(p, s, (float64)0.49999991059303284f, 0);
    p = v_f32_mac_b(p, s, (float64)-1.0f, 0);
    p = v_f32_mac_b(p, s, (float64)1.0f, 0);
    const int64 scaled = as_int64(p) + (as_int64(t) << 23);
    return v_f32_sel_less_f32_b(n, -125.5f, 0.0f, as_float64(scaled));
}

// 1 / d for d in [1, 2].
static inline float64 hc_il_reciprocal_1_2(float64 d)
{
    float64 q = v_f32_mac_b(d, 0.32322412f, (float64)-1.45450888f, 0);
    q = v_f32_mac_b(q, d, (float64)2.12117576f, 0);
    q = q * v_f32_mac_b(d, q, (float64)2.0f, SW_NEG);
    q = q * v_f32_mac_b(d, q, (float64)2.0f, SW_NEG);
    return q;
}

static inline float64 hc_il_sigmoid(float64 x)
{
    const float64 e = hc_il_exp_neg(v_f32_abs_b(x));
    const float64 q = hc_il_reciprocal_1_2(e + 1.0f);
    return v_f32_sel_less_f32_b(x, 0.0f, e * q, q);
}
