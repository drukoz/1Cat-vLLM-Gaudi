// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include "gdn_core_decode_gaudi2.hpp"
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device, uint32_t* count, tpc_lib_api::GuidInfo* guids) {
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    const auto capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
    if (!*count || !guids || !capacity) return GLUE_SUCCESS;
    std::memset(guids, 0, sizeof(*guids));
    std::strcpy(guids[0].name, GdnCoreDecodeGaudi2::name);
    return GLUE_SUCCESS;
}

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (std::strcmp(params->guid.name, GdnCoreDecodeGaudi2::name)) return GLUE_NODE_NOT_FOUND;
    return GdnCoreDecodeGaudi2{}.GetGcDefinitions(params, instance);
}

extern "C" tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId device, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output) {
    using namespace tpc_lib_api;
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (params->inputTensorsNr != 12) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorsNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    for (unsigned input = 0; input < 12; ++input) {
        if (!params->inputTensors[input]) return GLUE_SIF_NULL_PTR;
    }
    if (!output->outputTensors[0]) return GLUE_SIF_NULL_PTR;
    const auto& ssmState = params->inputTensors[1]->geometry;
    const auto& mixedQkv = params->inputTensors[2]->geometry;
    if (ssmState.dims != 4 || ssmState.maxSizes[0] != 128 || ssmState.maxSizes[1] != 128 ||
        ssmState.maxSizes[2] == 0 || ssmState.maxSizes[3] == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (mixedQkv.dims != 2 || mixedQkv.maxSizes[0] == 0 || mixedQkv.maxSizes[1] == 0) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    output->outputTensors[0]->geometry.dims = 3;
    output->outputTensors[0]->geometry.maxSizes[0] = 128;
    output->outputTensors[0]->geometry.maxSizes[1] = ssmState.maxSizes[2];
    output->outputTensors[0]->geometry.maxSizes[2] = mixedQkv.maxSizes[1];
    return GLUE_SUCCESS;
}

// This fused decode kernel has a fixed tensor layout and offers no optional
// transpose/reshape optimization. Older compiler adapters call this optional
// entry unconditionally; explicitly report that no manipulation is supported.
extern "C" tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
