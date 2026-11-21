// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host glue of the fused QSA decode kernels (libqsa_fused_gaudi2.so). Contract: docs/design/flash-next/qsa-fused.md.

#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>

#include "qsa_fused_params.h"

namespace qsa_fused {

enum class Kind { Prepare, Emit, Threshold, Select, Topk };

struct KernelInfo {
    Kind kind;
    const char* guid;
    unsigned inputs;
    unsigned outputs;
};

const KernelInfo* Kernels();
const KernelInfo* FindKernel(const char* name);
// The kernel a shape-inference call without one of our GUID names is for (tensor counts tell them apart).
const KernelInfo* KernelForCounts(unsigned inputs, unsigned outputs);

tpc_lib_api::GlueCodeReturn Instantiate(const KernelInfo& kernel, tpc_lib_api::HabanaKernelParams* params,
                                        tpc_lib_api::HabanaKernelInstantiation* instance);
tpc_lib_api::GlueCodeReturn InferShapes(const KernelInfo& kernel, const tpc_lib_api::ShapeInferenceParams* params,
                                        tpc_lib_api::ShapeInferenceOutput* output);

}  // namespace qsa_fused
