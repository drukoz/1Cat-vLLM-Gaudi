// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include "gdn_core_decode_gaudi2.hpp"
#include "gdn_core_verify_gaudi2.hpp"
#include "gdn_recurrent_decode_gaudi2.hpp"
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device, uint32_t* count, tpc_lib_api::GuidInfo* guids) {
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    const auto capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? 3 : 0;
    if (!*count || !guids || !capacity) return GLUE_SUCCESS;
    static constexpr const char* const kGuidNames[3] = {
        GdnCoreDecodeGaudi2::name, GdnRecurrentDecodeGaudi2::name, GdnCoreVerifyGaudi2::name};
    const uint32_t filled = capacity < 3 ? capacity : 3;
    std::memset(guids, 0, filled * sizeof(*guids));
    for (uint32_t i = 0; i < filled; ++i) {
        std::strcpy(guids[i].name, kGuidNames[i]);
    }
    return GLUE_SUCCESS;
}

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (!std::strcmp(params->guid.name, GdnCoreDecodeGaudi2::name)) {
        return GdnCoreDecodeGaudi2{}.GetGcDefinitions(params, instance);
    }
    if (!std::strcmp(params->guid.name, GdnRecurrentDecodeGaudi2::name)) {
        return GdnRecurrentDecodeGaudi2{}.GetGcDefinitions(params, instance);
    }
    if (!std::strcmp(params->guid.name, GdnCoreVerifyGaudi2::name)) {
        return GdnCoreVerifyGaudi2{}.GetGcDefinitions(params, instance);
    }
    return GLUE_NODE_NOT_FOUND;
}

namespace {

tpc_lib_api::GlueCodeReturn InferCoreDecodeShape(
    const tpc_lib_api::ShapeInferenceParams* params, tpc_lib_api::ShapeInferenceOutput* output) {
    using namespace tpc_lib_api;
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

tpc_lib_api::GlueCodeReturn InferRecurrentDecodeShape(
    const tpc_lib_api::ShapeInferenceParams* params, tpc_lib_api::ShapeInferenceOutput* output) {
    using namespace tpc_lib_api;
    if (params->outputTensorsNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    for (unsigned input = 0; input < 8; ++input) {
        if (!params->inputTensors[input]) return GLUE_SIF_NULL_PTR;
    }
    if (!output->outputTensors[0]) return GLUE_SIF_NULL_PTR;
    const auto& ssmState = params->inputTensors[0]->geometry;
    const auto& qkv = params->inputTensors[1]->geometry;
    if (ssmState.dims != 4 || ssmState.maxSizes[0] != 128 || ssmState.maxSizes[1] != 128 ||
        ssmState.maxSizes[2] == 0 || ssmState.maxSizes[3] == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (qkv.dims != 2 || qkv.maxSizes[0] == 0 || qkv.maxSizes[1] == 0) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    output->outputTensors[0]->geometry.dims = 3;
    output->outputTensors[0]->geometry.maxSizes[0] = 128;
    output->outputTensors[0]->geometry.maxSizes[1] = ssmState.maxSizes[2];
    output->outputTensors[0]->geometry.maxSizes[2] = qkv.maxSizes[1];
    return GLUE_SUCCESS;
}

}  // namespace

extern "C" tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId device, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output) {
    using namespace tpc_lib_api;
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    // The verify GUID has 12 inputs too, with ssm_state at input 1 and mixed_qkv
    // at input 2 as here, and its output {128, HV, mixed_qkv rows} is the same,
    // so InferCoreDecodeShape serves both GUIDs.
    if (params->inputTensorsNr == 12) return InferCoreDecodeShape(params, output);
    if (params->inputTensorsNr == 8) return InferRecurrentDecodeShape(params, output);
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
}

// This fused decode kernel has a fixed tensor layout and offers no optional
// transpose/reshape optimization. Older compiler adapters call this optional
// entry unconditionally; explicitly report that no manipulation is supported.
extern "C" tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
