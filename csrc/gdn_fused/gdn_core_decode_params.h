// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

inline constexpr const char* kGdnCoreDecodeGuid = "gdn_fused_core_decode_f32state_gaudi2";
struct GdnCoreDecodeParams {
    float scale;
    float eps;
    int32_t gate_activation;
};
static_assert(sizeof(GdnCoreDecodeParams) == 12, "GDN core-decode scalar ABI must contain two FP32 values and one I32");

inline constexpr const char* kGdnRecurrentDecodeGuid = "gdn_fused_recurrent_decode_f32state_gaudi2";
struct GdnRecurrentDecodeParams {
    float scale;
};
static_assert(sizeof(GdnRecurrentDecodeParams) == 4, "GDN recurrent-decode scalar ABI must contain one FP32 value");

// Phase 2: the fused GDN step over the T = 1 + k nodes of a speculative verification step
// (docs/design/flash-next/gdn-core-decode.md, "Phase 2 contract, version 2").
inline constexpr const char* kGdnCoreVerifyGuid = "gdn_fused_core_verify_f32state_gaudi2";
inline constexpr unsigned kGdnCoreVerifyMaxNodes = 8;  // T <= 8 (GDN_VERIFY_MAX_NODES in the kernel)
struct GdnCoreVerifyParams {
    float scale;
    float eps;
    int32_t gate_activation;
};
static_assert(sizeof(GdnCoreVerifyParams) == 12, "GDN core-verify scalar ABI must contain two FP32 values and one I32");
