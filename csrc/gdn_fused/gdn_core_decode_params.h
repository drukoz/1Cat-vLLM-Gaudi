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
