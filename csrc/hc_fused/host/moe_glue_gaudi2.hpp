// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "hc_fused_params.h"
#include "tpc_kernel_lib_interface.h"

// Host glue of the fused MoE glue kernels. One table row per GUID that was built into the library
// (CMakeLists.txt defines HC_FUSED_HAS_MOE_GLUE_<KERNEL> for each); the entry points in
// entry_points.cpp look a GUID up here after the hyper-connection table.
namespace moe_glue {

enum class Kind { Route, SiluMul, Combine, RowQuant };

struct KernelInfo {
    Kind kind;
    const char* guid;
    unsigned inputs;
    unsigned outputs;
};

// The kernels in the library, in GetKernelGuids order, and how many there are (0 to 4).
const KernelInfo* Kernels();
unsigned KernelCount();

// The row whose GUID is `name`, or nullptr (also for a GUID whose kernel was not built).
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

// The kernel a shape-inference call without a GUID name is for, from the tensor counts and the
// node parameters (top_k > 0 tells route from row_quant), or nullptr.
const KernelInfo* KernelForShapes(const tpc_lib_api::ShapeInferenceParams* params);

}  // namespace moe_glue
