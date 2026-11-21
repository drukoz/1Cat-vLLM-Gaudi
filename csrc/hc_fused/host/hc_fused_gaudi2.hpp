// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "hc_fused_params.h"
#include "tpc_kernel_lib_interface.h"

// Host glue of the fused hyper-connection kernels. One table row per GUID; the entry points in
// entry_points.cpp look a GUID up here and call Instantiate / InferShapes.
namespace hc_fused {

enum class Kind { Norm, Silu, SiluSplit, GateMix, Combine, CombineNorm, GateMixIl, CombineNormIl };

struct KernelInfo {
    Kind kind;
    const char* guid;
    unsigned inputs;
    unsigned outputs;
};

// The six kernels, in GetKernelGuids order.
const KernelInfo* Kernels();

// The interleaved kernels that were built into the library (CMakeLists.txt defines
// HC_FUSED_HAS_HC_FUSED_GATE_MIX_IL / HC_FUSED_HAS_HC_FUSED_COMBINE_NORM_IL), in GetKernelGuids order after the
// MoE glue GUIDs, and how many there are (0 to kHcFusedIlGuidCount).
const KernelInfo* IlKernels();
unsigned IlKernelCount();

// The row whose GUID is `name` (one of the six or a built interleaved kernel), or nullptr.
const KernelInfo* FindKernel(const char* name);

// Validates the node, fills the index space, the access patterns, the scalar arguments and the
// kernel ELF (two-call size handshake: GLUE_INSUFFICIENT_ELF_BUFFER carries the size needed).
tpc_lib_api::GlueCodeReturn Instantiate(
    const KernelInfo& kernel, tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance);

// Output shapes from the input shapes and the node parameters.
tpc_lib_api::GlueCodeReturn InferShapes(
    const KernelInfo& kernel, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output);

}  // namespace hc_fused
