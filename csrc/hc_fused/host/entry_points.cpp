// SPDX-License-Identifier: Apache-2.0
// Kernel-library entry points of libhc_fused_gaudi2.so (fused hyper-connection kernels, and the
// fused MoE glue kernels that were built into the library: host/moe_glue_gaudi2.cpp).
// The same five functions csrc/gdn_fused/host/entry_points.cpp exports, plus
// GetSupportedDataLayouts, which a library reached through the combined kernel database
// (csrc/kernel_db_adapter) must answer for its own GUIDs.
#include <cstdint>
#include <cstring>

#include "hc_fused_gaudi2.hpp"
#include "moe_glue_gaudi2.hpp"

namespace tpc_lib_api {
struct _TensorManipulationSuggestion;
}

namespace {

// The MoE kernel a shape-inference call is for, or nullptr. The graph compiler names the GUID
// through pGuid (new) or guid (old); when neither holds a name of this library, the size of the
// node parameters tells the two families apart (16 bytes here, 12 for the hyper-connection
// kernels) and moe_glue::KernelForShapes picks the kernel.
const moe_glue::KernelInfo* MoeShapeInferenceKernel(const tpc_lib_api::ShapeInferenceParams* params)
{
    if (params->pGuid) {
        if (const moe_glue::KernelInfo* kernel = moe_glue::FindKernel(params->pGuid->name)) return kernel;
        if (hc_fused::FindKernel(params->pGuid->name)) return nullptr;
    }
    if (const moe_glue::KernelInfo* kernel = moe_glue::FindKernel(params->guid.name)) return kernel;
    if (hc_fused::FindKernel(params->guid.name)) return nullptr;
    return moe_glue::KernelForShapes(params);
}

// The hyper-connection kernel a shape-inference call is for. The graph compiler names the GUID
// through pGuid (new) or guid (old); when neither holds one of our names, the tensor counts and
// the rank of the second input still identify the kernel uniquely.
const hc_fused::KernelInfo* ShapeInferenceKernel(const tpc_lib_api::ShapeInferenceParams* params)
{
    if (params->pGuid) {
        if (const hc_fused::KernelInfo* kernel = hc_fused::FindKernel(params->pGuid->name)) return kernel;
    }
    if (const hc_fused::KernelInfo* kernel = hc_fused::FindKernel(params->guid.name)) return kernel;

    const hc_fused::KernelInfo* kernels = hc_fused::Kernels();
    for (unsigned index = 0; index < kHcFusedGuidCount; ++index) {
        const hc_fused::KernelInfo& kernel = kernels[index];
        if (kernel.inputs != params->inputTensorsNr || kernel.outputs != params->outputTensorsNr) continue;
        if (kernel.inputs == 2) {
            // hc_fused_norm (x, norm_weight) and hc_fused_gate_mix (xn, gate) both have two inputs
            // and one output: the norm weight is rank 1, the gate is rank 2.
            if (!params->inputTensors || !params->inputTensors[1]) return nullptr;
            const bool rankOneSecond = params->inputTensors[1]->geometry.dims == 1;
            if (rankOneSecond != (kernel.kind == hc_fused::Kind::Norm)) continue;
        }
        return &kernel;
    }
    return nullptr;
}

}  // namespace

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device, uint32_t* count, tpc_lib_api::GuidInfo* guids)
{
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    // The six hyper-connection GUIDs, then the MoE glue GUIDs that were built (0 to 4).
    const uint32_t capacity = *count;
    const uint32_t total = kHcFusedGuidCount + moe_glue::KernelCount();
    *count = device == DEVICE_ID_GAUDI2 ? total : 0;
    if (!*count || !guids || !capacity) return GLUE_SUCCESS;
    const uint32_t filled = capacity < total ? capacity : total;
    std::memset(guids, 0, filled * sizeof(*guids));
    const hc_fused::KernelInfo* kernels = hc_fused::Kernels();
    const moe_glue::KernelInfo* moeKernels = moe_glue::Kernels();
    for (uint32_t index = 0; index < filled; ++index) {
        const char* guid = index < kHcFusedGuidCount ? kernels[index].guid : moeKernels[index - kHcFusedGuidCount].guid;
        std::strncpy(guids[index].name, guid, MAX_NODE_NAME - 1);
    }
    return GLUE_SUCCESS;
}

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance)
{
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (const hc_fused::KernelInfo* kernel = hc_fused::FindKernel(params->guid.name)) {
        return hc_fused::Instantiate(*kernel, params, instance);
    }
    if (const moe_glue::KernelInfo* kernel = moe_glue::FindKernel(params->guid.name)) {
        return moe_glue::Instantiate(*kernel, params, instance);
    }
    return GLUE_NODE_NOT_FOUND;
}

extern "C" tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId device, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output)
{
    using namespace tpc_lib_api;
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (const moe_glue::KernelInfo* moeKernel = MoeShapeInferenceKernel(params)) {
        return moe_glue::InferShapes(*moeKernel, params, output);
    }
    const hc_fused::KernelInfo* kernel = ShapeInferenceKernel(params);
    if (!kernel) return GLUE_NODE_NOT_FOUND;
    return hc_fused::InferShapes(*kernel, params, output);
}

// One explicit don't-care layout for each of our GUIDs: every kernel takes its tensors exactly as
// the op passes them. Zero layouts or a failure here fails the graph compiler's supported-layouts
// pass (same answer as csrc/deepseek_v4/host/entry_points.cpp gives for its own GUIDs).
extern "C" tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count)
{
    using namespace tpc_lib_api;
    if (!params || !count) return GLUE_FAILED;
    if (!hc_fused::FindKernel(params->guid.name) && !moe_glue::FindKernel(params->guid.name)) {
        return GLUE_NODE_NOT_FOUND;
    }
    *count = 1;
    if (layouts) {
        for (uint32_t index = 0; layouts->inputs && index < layouts->inputTensorNr; ++index) {
            std::memset(layouts->inputs[index].layout, 'x', sizeof(layouts->inputs[index].layout));
        }
        for (uint32_t index = 0; layouts->outputs && index < layouts->outputTensorNr; ++index) {
            std::memset(layouts->outputs[index].layout, 'x', sizeof(layouts->outputs[index].layout));
        }
        for (uint32_t index = 0; layouts->shapeTensors && index < layouts->shapeTensorNr; ++index) {
            std::memset(layouts->shapeTensors[index].layout, 'x', sizeof(layouts->shapeTensors[index].layout));
        }
    }
    return GLUE_SUCCESS;
}

// These kernels have a fixed tensor layout and offer no optional transpose/reshape optimization.
// Older compiler adapters call this optional entry unconditionally; explicitly report that no
// manipulation is supported (as csrc/gdn_fused/host/entry_points.cpp does).
extern "C" tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::_TensorManipulationSuggestion* suggestion)
{
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
