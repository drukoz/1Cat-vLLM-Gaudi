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
