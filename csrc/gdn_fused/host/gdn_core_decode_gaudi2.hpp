// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gdn_core_decode_params.h"
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class GdnCoreDecodeGaudi2 {
public:
    static constexpr const char* name = kGdnCoreDecodeGuid;
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};
