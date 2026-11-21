// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>

#include "gdn_core_decode_gaudi2.hpp"
#include "gdn_kernel_variant.hpp"

extern unsigned char _binary_gdn_core_decode_gaudi2_o_start;
extern unsigned char _binary_gdn_core_decode_gaudi2_o_end;
// The interleaved build (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1): weak, absent when it did not compile.
extern unsigned char _binary_gdn_core_decode_interleaved_gaudi2_o_start __attribute__((weak));
extern unsigned char _binary_gdn_core_decode_interleaved_gaudi2_o_end __attribute__((weak));

namespace {

void MapDimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensorDim,
    unsigned indexDim,
    int a,
    int startB,
    int endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = a;
    pattern.mapping[tensorDim].start_b = startB;
    pattern.mapping[tensorDim].end_b = endB;
}

bool HasTypeAndRank(
    const tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType type,
    unsigned dims)
{
    return tensor.geometry.dataType == type && tensor.geometry.dims == dims;
}

}  // namespace

tpc_lib_api::GlueCodeReturn GdnCoreDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance)
{
    constexpr unsigned kInputCount = 12;
    constexpr unsigned kOutputCount = 1;
    if (!params || !instance) {
        return tpc_lib_api::GLUE_FAILED;
    }
    if (params->inputTensorNr != kInputCount) {
        params->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != kOutputCount) {
        params->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(GdnCoreDecodeParams)) {
        return tpc_lib_api::GLUE_FAILED;
    }
    GdnCoreDecodeParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));

    const auto& convState = params->inputTensors[0];
    const auto& ssmState = params->inputTensors[1];
    const auto& mixedQkv = params->inputTensors[2];
    const auto& z = params->inputTensors[3];
    const auto& a = params->inputTensors[4];
    const auto& b = params->inputTensors[5];
    const auto& aLog = params->inputTensors[6];
    const auto& dtBias = params->inputTensors[7];
    const auto& convWeightT = params->inputTensors[8];
    const auto& normWeight = params->inputTensors[9];
    const auto& loadIndices = params->inputTensors[10];
    const auto& storeIndices = params->inputTensors[11];
    auto& output = params->outputTensors[0];

    const bool inputsValid =
        HasTypeAndRank(convState, tpc_lib_api::DATA_BF16, 3) &&
        HasTypeAndRank(ssmState, tpc_lib_api::DATA_F32, 4) &&
        HasTypeAndRank(mixedQkv, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(z, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(a, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(b, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(aLog, tpc_lib_api::DATA_F32, 1) &&
        HasTypeAndRank(dtBias, tpc_lib_api::DATA_F32, 1) &&
        HasTypeAndRank(convWeightT, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(normWeight, tpc_lib_api::DATA_BF16, 1) &&
        HasTypeAndRank(loadIndices, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(storeIndices, tpc_lib_api::DATA_I32, 1);
    if (!inputsValid) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t valueHeads = ssmState.geometry.maxSizes[2];
    const uint64_t channels = mixedQkv.geometry.maxSizes[0];
    const uint64_t batchSize = mixedQkv.geometry.maxSizes[1];
    const uint64_t convSlots = convState.geometry.maxSizes[2];
    const uint64_t stateSlots = ssmState.geometry.maxSizes[3];
    const uint64_t keyHeads = (channels - valueHeads * 128) / 256;
    const bool shapesMatch =
        ssmState.geometry.maxSizes[0] == 128 && ssmState.geometry.maxSizes[1] == 128 &&
        valueHeads > 0 && keyHeads > 0 && valueHeads % keyHeads == 0 &&
        channels == 2 * keyHeads * 128 + valueHeads * 128 &&
        convState.geometry.maxSizes[0] == channels && convState.geometry.maxSizes[1] == 3 &&
        convSlots > 0 && stateSlots > 0 && batchSize > 0 &&
        z.geometry.maxSizes[0] == valueHeads * 128 && z.geometry.maxSizes[1] == batchSize &&
        a.geometry.maxSizes[0] == valueHeads && a.geometry.maxSizes[1] == batchSize &&
        b.geometry.maxSizes[0] == valueHeads && b.geometry.maxSizes[1] == batchSize &&
        aLog.geometry.maxSizes[0] == valueHeads &&
        dtBias.geometry.maxSizes[0] == valueHeads &&
        convWeightT.geometry.maxSizes[0] == channels && convWeightT.geometry.maxSizes[1] == 4 &&
        normWeight.geometry.maxSizes[0] == 128 &&
        loadIndices.geometry.maxSizes[0] == batchSize &&
        storeIndices.geometry.maxSizes[0] == batchSize;
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool scalarsValid =
        std::isnormal(scalar.eps) && scalar.eps > 0 &&
        std::isnormal(scalar.scale) && scalar.scale > 0 &&
        (scalar.gate_activation == 0 || scalar.gate_activation == 1);
    if (!scalarsValid) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t repetition = valueHeads / keyHeads;

    const bool outputMatches =
        HasTypeAndRank(output, tpc_lib_api::DATA_F32, 3) &&
        output.geometry.maxSizes[0] == 128 && output.geometry.maxSizes[1] == valueHeads &&
        output.geometry.maxSizes[2] == batchSize;
    if (!outputMatches) {
        output.geometry.dataType = tpc_lib_api::DATA_F32;
        output.geometry.dims = 3;
        output.geometry.maxSizes[0] = 128;
        output.geometry.maxSizes[1] = valueHeads;
        output.geometry.maxSizes[2] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = keyHeads;
    instance->indexSpaceGeometry[1] = batchSize;

    for (unsigned input = 0; input < kInputCount; ++input) {
        std::memset(
            &instance->inputTensorAccessPattern[input], 0, sizeof(tpc_lib_api::TensorAccessPattern));
    }
    auto& outputPattern = instance->outputTensorAccessPattern[0];
    std::memset(&outputPattern, 0, sizeof(outputPattern));

    // Inputs 0, 1, 6, 7, 8, 9 are read whole by every index point; the kernel
    // gathers the state slots through the load/store indices itself.
    auto& convStatePattern = instance->inputTensorAccessPattern[0];
    MapDimension(convStatePattern, 0, 0, 0, 0, static_cast<int>(channels - 1));
    MapDimension(convStatePattern, 1, 0, 0, 0, 2);
    MapDimension(convStatePattern, 2, 0, 0, 0, static_cast<int>(convSlots - 1));

    auto& ssmStatePattern = instance->inputTensorAccessPattern[1];
    MapDimension(ssmStatePattern, 0, 0, 0, 0, 127);
    MapDimension(ssmStatePattern, 1, 0, 0, 0, 127);
    MapDimension(ssmStatePattern, 2, 0, 0, 0, static_cast<int>(valueHeads - 1));
    MapDimension(ssmStatePattern, 3, 0, 0, 0, static_cast<int>(stateSlots - 1));

    auto& mixedQkvPattern = instance->inputTensorAccessPattern[2];
    MapDimension(mixedQkvPattern, 0, 0, 0, 0, static_cast<int>(channels - 1));
    MapDimension(mixedQkvPattern, 1, 1, 1, 0, 0);

    auto& zPattern = instance->inputTensorAccessPattern[3];
    MapDimension(zPattern, 0, 0, static_cast<int>(repetition * 128), 0, static_cast<int>(repetition * 128 - 1));
    MapDimension(zPattern, 1, 1, 1, 0, 0);

    auto& aPattern = instance->inputTensorAccessPattern[4];
    MapDimension(aPattern, 0, 0, static_cast<int>(repetition), 0, static_cast<int>(repetition - 1));
    MapDimension(aPattern, 1, 1, 1, 0, 0);

    auto& bPattern = instance->inputTensorAccessPattern[5];
    MapDimension(bPattern, 0, 0, static_cast<int>(repetition), 0, static_cast<int>(repetition - 1));
    MapDimension(bPattern, 1, 1, 1, 0, 0);

    auto& aLogPattern = instance->inputTensorAccessPattern[6];
    MapDimension(aLogPattern, 0, 0, 0, 0, static_cast<int>(valueHeads - 1));

    auto& dtBiasPattern = instance->inputTensorAccessPattern[7];
    MapDimension(dtBiasPattern, 0, 0, 0, 0, static_cast<int>(valueHeads - 1));

    auto& convWeightPattern = instance->inputTensorAccessPattern[8];
    MapDimension(convWeightPattern, 0, 0, 0, 0, static_cast<int>(channels - 1));
    MapDimension(convWeightPattern, 1, 0, 0, 0, 3);

    auto& normWeightPattern = instance->inputTensorAccessPattern[9];
    MapDimension(normWeightPattern, 0, 0, 0, 0, 127);

    auto& loadIndicesPattern = instance->inputTensorAccessPattern[10];
    MapDimension(loadIndicesPattern, 0, 1, 1, 0, 0);

    auto& storeIndicesPattern = instance->inputTensorAccessPattern[11];
    MapDimension(storeIndicesPattern, 0, 1, 1, 0, 0);

    MapDimension(outputPattern, 0, 0, 0, 0, 127);
    MapDimension(outputPattern, 1, 0, static_cast<int>(repetition), 0, static_cast<int>(repetition - 1));
    MapDimension(outputPattern, 2, 1, 1, 0, 0);

    instance->kernel.paramsNr = sizeof(GdnCoreDecodeParams) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));

    const GdnKernelElf elf = GdnSelectKernelElf(&_binary_gdn_core_decode_gaudi2_o_start, &_binary_gdn_core_decode_gaudi2_o_end,
                                                &_binary_gdn_core_decode_interleaved_gaudi2_o_start,
                                                &_binary_gdn_core_decode_interleaved_gaudi2_o_end);
    const unsigned isaSize = elf.size;
    const unsigned providedSize = instance->kernel.elfSize;
    instance->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    if (!instance->kernel.kernelElf) {
        return tpc_lib_api::GLUE_FAILED;
    }
    std::memcpy(instance->kernel.kernelElf, elf.start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
