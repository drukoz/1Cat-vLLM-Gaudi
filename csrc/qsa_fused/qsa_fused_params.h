// SPDX-License-Identifier: Apache-2.0
#pragma once
// GUIDs and the node-parameter ABI of the fused QSA decode kernels. Shared by the host glue (host/), the
// PyTorch op registration (pytorch/) and the simulator host program (tests/standalone/qsa_fused/sim_host.cpp).
// Contract: docs/design/flash-next/qsa-fused.md.

#include <cstdint>

inline constexpr const char* kQsaFusedPrepareGuid = "qsa_fused_prepare_bf16_gaudi2";
inline constexpr const char* kQsaFusedEmitGuid = "qsa_fused_emit_i32_gaudi2";
inline constexpr const char* kQsaFusedThresholdGuid = "qsa_fused_threshold_f32_gaudi2";
inline constexpr const char* kQsaFusedSelectGuid = "qsa_fused_select_i32_gaudi2";
// The one-node top-k of the masked block scores (VLLM_HPU_QSA_TOPK_FORM=4; kernels/qsa_fused_topk_gaudi2.c).
inline constexpr const char* kQsaFusedTopkGuid = "qsa_fused_topk_f32_gaudi2";
inline constexpr unsigned kQsaFusedGuidCount = 5;
// The top-k kernel keeps two 16-bit values per column of a row in 64 KiB of vector local memory: at most 32,768
// columns, a multiple of 512 (the count loop reads four 128-column vectors per step).
inline constexpr int64_t kQsaFusedTopkMaxColumns = 32768;
inline constexpr int64_t kQsaFusedTopkColumnStep = 512;

// Gates: one indexer head is one 128-lane bf16 vector; at most 8 heads; ratio and block_size are powers of
// two with ratio <= 8 and ratio <= block_size; the ring keeps at least one block of rows per slot.
inline constexpr unsigned kQsaFusedMaxDim = 128;
inline constexpr int32_t kQsaFusedMaxHeads = 8;
inline constexpr int32_t kQsaFusedMaxRatio = 8;
inline constexpr int32_t kQsaFusedMaxRingRows = 64;
inline constexpr int32_t kQsaFusedMaxBlockSize = 1 << 16;

// Node parameters (PyTorch op -> glue), the same 28 bytes for every GUID. A kernel that has no use for a
// field gets 0 in it: eps, heads, ring_rows and rotary_half are read by prepare, block_size by emit, ratio by
// prepare, emit and threshold, topk by threshold and select.
struct QsaFusedParams {
    float eps;            // the indexer RMSNorm epsilon (config.rms_norm_eps)
    int32_t heads;        // indexer query heads H
    int32_t ring_rows;    // raw-key ring rows per slot R
    int32_t ratio;        // indexer_compress_ratio
    int32_t block_size;   // attention kernel block B
    int32_t rotary_half;  // R2: half the rotary dims of the attention's cos_sin_cache
    int32_t topk;         // block_topk (threshold, select)
};
static_assert(sizeof(QsaFusedParams) == 28, "the qsa_fused node-parameter ABI is one FP32 and six I32 values");

// Scalar arguments of qsa_fused_prepare's main() (glue -> kernel), in this order. The glue derives the
// reciprocals on the host, so the kernel does no division.
struct QsaFusedPrepareArgs {
    float eps;
    float inv_dim;    // 1 / D
    float inv_ring;   // 1 / R (the kernel's modulo by R)
    float inv_ratio;  // 1 / ratio
    int32_t heads;
    int32_t ring_rows;
    int32_t ratio;
    int32_t rotary_half;
};
static_assert(sizeof(QsaFusedPrepareArgs) == 32, "the prepare kernel scalar ABI is four FP32 and four I32 values");

// Scalar arguments of qsa_fused_emit's main(), in this order.
struct QsaFusedEmitArgs {
    int32_t ratio_log2;
    int32_t block_log2;
    int32_t topk;
};
static_assert(sizeof(QsaFusedEmitArgs) == 12, "the emit kernel scalar ABI is three I32 values");

// Scalar arguments of qsa_fused_threshold's main() and qsa_fused_select's main() (select reads topk only).
struct QsaFusedSelectArgs {
    int32_t ratio_log2;
    int32_t topk;
};
static_assert(sizeof(QsaFusedSelectArgs) == 8, "the selection kernels' scalar ABI is two I32 values");

// Scalar argument of qsa_fused_topk's main().
struct QsaFusedTopkArgs {
    int32_t topk;
};
static_assert(sizeof(QsaFusedTopkArgs) == 4, "the top-k kernel scalar ABI is one I32 value");
