// SPDX-License-Identifier: Apache-2.0
// Kernel-library entry points of libqsa_fused_gaudi2.so (the fused QSA decode kernels): the functions
// csrc/hc_fused/host/entry_points.cpp exports, for the GUIDs of qsa_fused_params.h.
#include <cstdint>
#include <cstring>

#include "qsa_fused_gaudi2.hpp"

namespace tpc_lib_api {
struct _TensorManipulationSuggestion;
}

namespace {

// The kernel a shape-inference call is for: by GUID name (pGuid new, guid old), else by the tensor counts
// (7 + 1 for prepare, 3 + 2 for emit).
const qsa_fused::KernelInfo* ShapeInferenceKernel(const tpc_lib_api::ShapeInferenceParams* params)
{
    if (params->pGuid) {
        if (const qsa_fused::KernelInfo* kernel = qsa_fused::FindKernel(params->pGuid->name)) return kernel;
    }
    if (const qsa_fused::KernelInfo* kernel = qsa_fused::FindKernel(params->guid.name)) return kernel;
    return qsa_fused::KernelForCounts(params->inputTensorsNr, params->outputTensorsNr);
}

}  // namespace

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device, uint32_t* count, tpc_lib_api::GuidInfo* guids)
{
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    const uint32_t capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? kQsaFusedGuidCount : 0;
    if (!*count || !guids || !capacity) return GLUE_SUCCESS;
    const uint32_t filled = capacity < kQsaFusedGuidCount ? capacity : kQsaFusedGuidCount;
    std::memset(guids, 0, filled * sizeof(*guids));
    const qsa_fused::KernelInfo* kernels = qsa_fused::Kernels();
    for (uint32_t index = 0; index < filled; ++index) {
        std::strncpy(guids[index].name, kernels[index].guid, MAX_NODE_NAME - 1);
    }
    return GLUE_SUCCESS;
}

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance)
{
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (const qsa_fused::KernelInfo* kernel = qsa_fused::FindKernel(params->guid.name)) {
        return qsa_fused::Instantiate(*kernel, params, instance);
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
    const qsa_fused::KernelInfo* kernel = ShapeInferenceKernel(params);
    if (!kernel) return GLUE_NODE_NOT_FOUND;
    return qsa_fused::InferShapes(*kernel, params, output);
}

// One explicit don't-care layout for each of our GUIDs (as csrc/hc_fused/host/entry_points.cpp).
extern "C" tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count)
{
    using namespace tpc_lib_api;
    if (!params || !count) return GLUE_FAILED;
    if (!qsa_fused::FindKernel(params->guid.name)) return GLUE_NODE_NOT_FOUND;
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

extern "C" tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::_TensorManipulationSuggestion* suggestion)
{
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
