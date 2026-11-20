// SPDX-License-Identifier: Apache-2.0
#pragma once
// GUIDs and the node-parameter ABI of the fused hyper-connection kernels. Shared by the host glue
// (host/), the PyTorch op registration (pytorch/) and the simulator host program
// (tests/standalone/hc_fused/sim_host.cpp). Contract: docs/design/flash-next/hc-fused.md.

#include <cstdint>

inline constexpr const char* kHcFusedNormGuid = "hc_fused_norm_bf16_gaudi2";
inline constexpr const char* kHcFusedSiluGuid = "hc_fused_silu_bf16_gaudi2";
inline constexpr const char* kHcFusedSiluSplitGuid = "hc_fused_silu_split_bf16_gaudi2";
inline constexpr const char* kHcFusedGateMixGuid = "hc_fused_gate_mix_bf16_gaudi2";
inline constexpr const char* kHcFusedCombineGuid = "hc_fused_combine_bf16_gaudi2";
inline constexpr const char* kHcFusedCombineNormGuid = "hc_fused_combine_norm_bf16_gaudi2";
inline constexpr unsigned kHcFusedGuidCount = 6;

// One 128-lane bf16 vector. Every multi-stream kernel walks a stream in whole vectors, so the glue
// rejects a hidden size that is not a multiple of it (2560 and the tiny model's 128 are).
inline constexpr unsigned kHcFusedVectorLanes = 128;

// Node parameters (PyTorch op -> glue), the same 12 bytes for every GUID. A kernel that has no use
// for a field gets 0 in it: eps is read by norm and combine_norm, lora_rank by silu_split.
struct HcFusedParams {
    float eps;          // the RMSNorm epsilon (config.rms_norm_eps)
    int32_t hc_count;   // number of hyper-connection streams
    int32_t lora_rank;  // hc_lowrank: the width of the lora output of silu_split
};
static_assert(sizeof(HcFusedParams) == 12, "the hc_fused node-parameter ABI is one FP32 and two I32 values");

// Scalar arguments of every kernel's main() (glue -> kernel), in this order. The glue derives the
// two reciprocals on the host, so no kernel does a floating-point division.
struct HcFusedKernelArgs {
    float eps;
    float inv_group;   // 1 / hidden_size: turns a stream's sum of squares into its mean
    float inv_hc;      // 1 / hc_count
    int32_t hc_count;
};
static_assert(sizeof(HcFusedKernelArgs) == 16, "the hc_fused kernel scalar ABI is three FP32 values and one I32");

// ---------------------------------------------------------------------------------------------
// MoE glue kernels: four more GUIDs of this library (docs/design/flash-next/hc-fused.md, "MoE glue
// kernels"). Each one is optional at build time: CMakeLists.txt tries its compile at configure time
// and leaves a kernel that does not compile out of the library, so GetKernelGuids lists the six
// hyper-connection GUIDs followed by the MoE GUIDs that were built, in this order.
// ---------------------------------------------------------------------------------------------
inline constexpr const char* kMoeGlueRouteGuid = "moe_glue_route_bf16_gaudi2";
inline constexpr const char* kMoeGlueSiluMulGuid = "moe_glue_silu_mul_bf16_gaudi2";
inline constexpr const char* kMoeGlueCombineGuid = "moe_glue_combine_bf16_gaudi2";
inline constexpr const char* kMoeGlueRowQuantGuid = "moe_glue_row_quant_bf16_gaudi2";
inline constexpr unsigned kMoeGlueGuidCount = 4;

// Gates of the MoE kernels. top_k: one 64-lane int32 vector holds the ids of a row. experts: the
// routing kernel tracks expert ids in int16 lanes. fp8_max: the largest value of Gaudi2's 1-4-3
// format, the only value the row quantization kernel accepts.
inline constexpr int32_t kMoeGlueMaxTopK = 64;
inline constexpr int32_t kMoeGlueMaxExperts = 32767;
inline constexpr float kMoeGlueFp8Max = 240.0f;

// Node parameters (PyTorch op -> glue), the same 16 bytes for the four GUIDs. A kernel that has no
// use for a field gets 0 in it: experts and top_k are read by route, gate_column by combine,
// fp8_max by row_quant.
struct MoeGlueParams {
    int32_t experts;      // router columns of the merged gate output
    int32_t top_k;        // experts per token
    int32_t gate_column;  // the column of the merged gate output that holds the shared-gate logit
    float fp8_max;        // FP8_MAX of the plugin on this device (240 on Gaudi2)
};
static_assert(sizeof(MoeGlueParams) == 16, "the moe_glue node-parameter ABI is three I32 values and one FP32");

// Scalar arguments of every MoE kernel's main() (glue -> kernel), in this order. The glue derives
// the reciprocal on the host.
struct MoeGlueKernelArgs {
    int32_t experts;
    int32_t top_k;
    int32_t gate_column;
    float inv_fp8_max;  // 1 / fp8_max, or 0 for a kernel that does not quantize
};
static_assert(sizeof(MoeGlueKernelArgs) == 16, "the moe_glue kernel scalar ABI is three I32 values and one FP32");
