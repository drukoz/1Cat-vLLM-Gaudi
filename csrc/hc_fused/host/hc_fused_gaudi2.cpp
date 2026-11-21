// SPDX-License-Identifier: Apache-2.0
// Host glue of the fused hyper-connection kernels: validation, index space, access patterns,
// scalar arguments and the embedded kernel ELF, for the six GUIDs of hc_fused_params.h.
//
// Every size comes from the tensor geometry and the node parameters. The only gates are:
// rank-2 bf16 activations {width, tokens}, a rank-1 bf16 norm weight, a multi-stream width that is
// hc_count streams of a hidden size divisible by one 128-lane vector, and a positive normal eps.
// Anything else is rejected here with a GlueCodeReturn; no kernel runs on a shape it cannot handle.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "hc_fused_gaudi2.hpp"

extern unsigned char _binary_hc_fused_norm_gaudi2_o_start;
extern unsigned char _binary_hc_fused_norm_gaudi2_o_end;
extern unsigned char _binary_hc_fused_silu_gaudi2_o_start;
extern unsigned char _binary_hc_fused_silu_gaudi2_o_end;
extern unsigned char _binary_hc_fused_silu_split_gaudi2_o_start;
extern unsigned char _binary_hc_fused_silu_split_gaudi2_o_end;
extern unsigned char _binary_hc_fused_gate_mix_gaudi2_o_start;
extern unsigned char _binary_hc_fused_gate_mix_gaudi2_o_end;
extern unsigned char _binary_hc_fused_combine_gaudi2_o_start;
extern unsigned char _binary_hc_fused_combine_gaudi2_o_end;
extern unsigned char _binary_hc_fused_combine_norm_gaudi2_o_start;
extern unsigned char _binary_hc_fused_combine_norm_gaudi2_o_end;

namespace hc_fused {
namespace {

using namespace tpc_lib_api;

constexpr unsigned kMaxInputs = 4;
constexpr unsigned kMaxOutputs = 2;
constexpr int32_t kMaxStreams = 64;

const KernelInfo kKernels[kHcFusedGuidCount] = {
    {Kind::Norm, kHcFusedNormGuid, 2, 1},
    {Kind::Silu, kHcFusedSiluGuid, 1, 1},
    {Kind::SiluSplit, kHcFusedSiluSplitGuid, 1, 2},
    {Kind::GateMix, kHcFusedGateMixGuid, 2, 1},
    {Kind::Combine, kHcFusedCombineGuid, 3, 1},
    {Kind::CombineNorm, kHcFusedCombineNormGuid, 4, 2},
};

struct Elf {
    unsigned char* start;
    unsigned char* end;
};

Elf KernelElf(Kind kind)
{
    switch (kind) {
        case Kind::Norm:
            return {&_binary_hc_fused_norm_gaudi2_o_start, &_binary_hc_fused_norm_gaudi2_o_end};
        case Kind::Silu:
            return {&_binary_hc_fused_silu_gaudi2_o_start, &_binary_hc_fused_silu_gaudi2_o_end};
        case Kind::SiluSplit:
            return {&_binary_hc_fused_silu_split_gaudi2_o_start, &_binary_hc_fused_silu_split_gaudi2_o_end};
        case Kind::GateMix:
            return {&_binary_hc_fused_gate_mix_gaudi2_o_start, &_binary_hc_fused_gate_mix_gaudi2_o_end};
        case Kind::Combine:
            return {&_binary_hc_fused_combine_gaudi2_o_start, &_binary_hc_fused_combine_gaudi2_o_end};
        case Kind::CombineNorm:
            return {&_binary_hc_fused_combine_norm_gaudi2_o_start, &_binary_hc_fused_combine_norm_gaudi2_o_end};
    }
    return {nullptr, nullptr};
}

// What the shapes of one node resolve to. Sizes that a kind does not have stay 0.
struct Resolved {
    uint64_t tokens = 0;
    uint64_t hidden = 0;   // one stream
    uint64_t width = 0;    // hc_count * hidden
    uint64_t columns = 0;  // silu kinds: columns of `down`
    uint64_t lora = 0;     // silu kinds: columns of lora_out
    bool sharedWeight = false;
    unsigned outputs = 0;
    uint64_t outputSize[kMaxOutputs][2] = {};  // every output is rank 2 {columns, tokens}
};

bool IsMatrix(const TensorGeometry& geometry)
{
    return geometry.dims == 2 && geometry.maxSizes[0] > 0 && geometry.maxSizes[1] > 0;
}

bool IsMatrixOf(const TensorGeometry& geometry, uint64_t columns, uint64_t tokens)
{
    return geometry.dims == 2 && geometry.maxSizes[0] == columns && geometry.maxSizes[1] == tokens;
}

// Splits a multi-stream width into hc_count streams of whole 128-lane vectors.
bool SplitStreams(uint64_t width, int32_t hcCount, uint64_t& hidden)
{
    const uint64_t streams = static_cast<uint64_t>(hcCount);
    if (width == 0 || width % streams != 0) return false;
    hidden = width / streams;
    return hidden % kHcFusedVectorLanes == 0;
}

// A norm weight is rank 1 and holds one affine per stream element, or one per element of the row.
bool ResolveWeight(const TensorGeometry& weight, Resolved& r)
{
    if (weight.dims != 1) return false;
    if (weight.maxSizes[0] == r.hidden) {
        r.sharedWeight = true;
        return true;
    }
    r.sharedWeight = false;
    return weight.maxSizes[0] == r.width;
}

void SetOutput(Resolved& r, unsigned index, uint64_t columns)
{
    r.outputSize[index][0] = columns;
    r.outputSize[index][1] = r.tokens;
}

// Shape validation shared by InstantiateTpcKernel and GetShapeInference. `in` has kernel.inputs
// entries. GLUE_SUCCESS, or why the node is not one of ours.
GlueCodeReturn Resolve(const KernelInfo& kernel, const TensorGeometry* const* in, const HcFusedParams& scalar,
                       Resolved& r)
{
    if (scalar.hc_count < 1 || scalar.hc_count > kMaxStreams) return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    const uint64_t streams = static_cast<uint64_t>(scalar.hc_count);
    if (!IsMatrix(*in[0])) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    r.tokens = in[0]->maxSizes[1];
    r.outputs = kernel.outputs;

    switch (kernel.kind) {
        case Kind::Norm:
            r.width = in[0]->maxSizes[0];
            if (!SplitStreams(r.width, scalar.hc_count, r.hidden) || !ResolveWeight(*in[1], r)) {
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            }
            SetOutput(r, 0, r.width);
            return GLUE_SUCCESS;
        case Kind::Silu:
            r.columns = in[0]->maxSizes[0];
            r.lora = r.columns;
            SetOutput(r, 0, r.lora);
            return GLUE_SUCCESS;
        case Kind::SiluSplit:
            if (scalar.lora_rank < 1) return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
            r.columns = in[0]->maxSizes[0];
            r.lora = static_cast<uint64_t>(scalar.lora_rank);
            if (r.columns < r.lora + streams) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            SetOutput(r, 0, r.lora);
            SetOutput(r, 1, streams);
            return GLUE_SUCCESS;
        case Kind::GateMix:
            r.width = in[0]->maxSizes[0];
            if (!SplitStreams(r.width, scalar.hc_count, r.hidden) || !IsMatrixOf(*in[1], r.width, r.tokens)) {
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            }
            SetOutput(r, 0, r.hidden);
            return GLUE_SUCCESS;
        case Kind::Combine:
        case Kind::CombineNorm:
            r.width = in[0]->maxSizes[0];
            if (!SplitStreams(r.width, scalar.hc_count, r.hidden) || !IsMatrixOf(*in[1], r.hidden, r.tokens) ||
                !IsMatrixOf(*in[2], streams, r.tokens)) {
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            }
            SetOutput(r, 0, r.width);
            if (kernel.kind == Kind::CombineNorm) {
                if (!ResolveWeight(*in[3], r)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
                SetOutput(r, 1, r.width);
            }
            return GLUE_SUCCESS;
    }
    return GLUE_NODE_NOT_FOUND;
}

bool ReadsEps(Kind kind) { return kind == Kind::Norm || kind == Kind::CombineNorm; }

void MapDimension(TensorAccessPattern& pattern, unsigned tensorDim, unsigned indexDim, uint64_t a, uint64_t startB,
                  uint64_t endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = static_cast<float>(a);
    pattern.mapping[tensorDim].start_b = static_cast<float>(startB);
    pattern.mapping[tensorDim].end_b = static_cast<float>(endB);
}

// {columns, tokens} tensor whose point (i, token) owns columns [a * i, a * i + span): a == 0 means
// every point of index dim 0 reads the same `span` columns.
void MapMatrix(TensorAccessPattern& pattern, unsigned tokenIndexDim, uint64_t a, uint64_t span)
{
    MapDimension(pattern, 0, 0, a, 0, span - 1);
    MapDimension(pattern, 1, tokenIndexDim, 1, 0, 0);
}

void MapWeight(TensorAccessPattern& pattern, const Resolved& r)
{
    MapDimension(pattern, 0, 0, r.sharedWeight ? 0 : r.hidden, 0, r.hidden - 1);
}

}  // namespace

const KernelInfo* Kernels() { return kKernels; }

const KernelInfo* FindKernel(const char* name)
{
    if (!name) return nullptr;
    for (const KernelInfo& kernel : kKernels) {
        if (!std::strncmp(name, kernel.guid, MAX_NODE_NAME)) return &kernel;
    }
    return nullptr;
}

GlueCodeReturn Instantiate(const KernelInfo& kernel, HabanaKernelParams* params, HabanaKernelInstantiation* instance)
{
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != kernel.inputs) {
        params->inputTensorNr = kernel.inputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != kernel.outputs) {
        params->outputTensorNr = kernel.outputs;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(HcFusedParams)) {
        return GLUE_FAILED;
    }
    HcFusedParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));

    const TensorGeometry* in[kMaxInputs] = {};
    for (unsigned input = 0; input < kernel.inputs; ++input) {
        if (params->inputTensors[input].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
        in[input] = &params->inputTensors[input].geometry;
    }
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;
    if (ReadsEps(kernel.kind) && !(std::isnormal(scalar.eps) && scalar.eps > 0)) {
        return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }

    bool outputsMatch = true;
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        const TensorGeometry& geometry = params->outputTensors[output].geometry;
        outputsMatch = outputsMatch && geometry.dataType == DATA_BF16 &&
                       IsMatrixOf(geometry, r.outputSize[output][0], r.outputSize[output][1]);
    }
    if (!outputsMatch) {
        for (unsigned output = 0; output < kernel.outputs; ++output) {
            TensorGeometry& geometry = params->outputTensors[output].geometry;
            geometry.dataType = DATA_BF16;
            geometry.dims = 2;
            geometry.maxSizes[0] = r.outputSize[output][0];
            geometry.maxSizes[1] = r.outputSize[output][1];
        }
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    for (unsigned input = 0; input < kernel.inputs; ++input) {
        std::memset(&instance->inputTensorAccessPattern[input], 0, sizeof(TensorAccessPattern));
    }
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        std::memset(&instance->outputTensorAccessPattern[output], 0, sizeof(TensorAccessPattern));
    }
    TensorAccessPattern* inputPattern = instance->inputTensorAccessPattern;
    TensorAccessPattern* outputPattern = instance->outputTensorAccessPattern;
    const uint64_t streams = static_cast<uint64_t>(scalar.hc_count);

    switch (kernel.kind) {
        case Kind::Norm:
            // Point (stream, token) owns that stream's columns of x and xn_out.
            instance->indexSpaceRank = 2;
            instance->indexSpaceGeometry[0] = streams;
            instance->indexSpaceGeometry[1] = r.tokens;
            MapMatrix(inputPattern[0], 1, r.hidden, r.hidden);
            MapWeight(inputPattern[1], r);
            MapMatrix(outputPattern[0], 1, r.hidden, r.hidden);
            break;
        case Kind::Silu:
        case Kind::SiluSplit:
            // Point (token) owns the token's row of every tensor.
            instance->indexSpaceRank = 1;
            instance->indexSpaceGeometry[0] = r.tokens;
            MapDimension(inputPattern[0], 0, 0, 0, 0, r.columns - 1);
            MapDimension(inputPattern[0], 1, 0, 1, 0, 0);
            for (unsigned output = 0; output < kernel.outputs; ++output) {
                MapDimension(outputPattern[output], 0, 0, 0, 0, r.outputSize[output][0] - 1);
                MapDimension(outputPattern[output], 1, 0, 1, 0, 0);
            }
            break;
        case Kind::GateMix:
            // Point (chunk, token) writes 128 output columns and reads that chunk of every stream. The
            // chunks of one point are hidden columns apart in xn and gate, which one linear range cannot
            // describe, so both are declared as the token's whole row.
            instance->indexSpaceRank = 2;
            instance->indexSpaceGeometry[0] = r.hidden / kHcFusedVectorLanes;
            instance->indexSpaceGeometry[1] = r.tokens;
            MapMatrix(inputPattern[0], 1, 0, r.width);
            MapMatrix(inputPattern[1], 1, 0, r.width);
            MapMatrix(outputPattern[0], 1, kHcFusedVectorLanes, kHcFusedVectorLanes);
            break;
        case Kind::Combine:
        case Kind::CombineNorm:
            // Point (stream, token) owns that stream's columns of residual and of the outputs, its one
            // injection logit, and reads the token's whole block output.
            instance->indexSpaceRank = 2;
            instance->indexSpaceGeometry[0] = streams;
            instance->indexSpaceGeometry[1] = r.tokens;
            MapMatrix(inputPattern[0], 1, r.hidden, r.hidden);
            MapMatrix(inputPattern[1], 1, 0, r.hidden);
            MapMatrix(inputPattern[2], 1, 1, 1);
            MapMatrix(outputPattern[0], 1, r.hidden, r.hidden);
            if (kernel.kind == Kind::CombineNorm) {
                MapWeight(inputPattern[3], r);
                MapMatrix(outputPattern[1], 1, r.hidden, r.hidden);
            }
            break;
    }

    HcFusedKernelArgs args{};
    args.eps = scalar.eps;
    args.inv_group = r.hidden ? 1.0f / static_cast<float>(r.hidden) : 0.0f;
    args.inv_hc = 1.0f / static_cast<float>(scalar.hc_count);
    args.hc_count = scalar.hc_count;
    instance->kernel.paramsNr = sizeof(HcFusedKernelArgs) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));

    const Elf elf = KernelElf(kernel.kind);
    const unsigned elfSize = static_cast<unsigned>(elf.end - elf.start);
    const unsigned providedSize = instance->kernel.elfSize;
    instance->kernel.elfSize = elfSize;
    if (providedSize < elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, elf.start, elfSize);
    return GLUE_SUCCESS;
}

GlueCodeReturn InferShapes(const KernelInfo& kernel, const ShapeInferenceParams* params, ShapeInferenceOutput* output)
{
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (params->inputTensorsNr != kernel.inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorsNr != kernel.outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorGeometry* in[kMaxInputs] = {};
    for (unsigned input = 0; input < kernel.inputs; ++input) {
        if (!params->inputTensors[input]) return GLUE_SIF_NULL_PTR;
        in[input] = &params->inputTensors[input]->geometry;
    }
    for (unsigned index = 0; index < kernel.outputs; ++index) {
        if (!output->outputTensors[index]) return GLUE_SIF_NULL_PTR;
    }
    if (!params->nodeParams.nodeParams || params->nodeParams.nodeParamsSize != sizeof(HcFusedParams)) {
        return GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    HcFusedParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;
    for (unsigned index = 0; index < kernel.outputs; ++index) {
        TensorGeometry& geometry = output->outputTensors[index]->geometry;
        geometry.dims = 2;
        geometry.maxSizes[0] = r.outputSize[index][0];
        geometry.maxSizes[1] = r.outputSize[index][1];
    }
    return GLUE_SUCCESS;
}

}  // namespace hc_fused
